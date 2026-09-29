"""Broad English candidate retrieval followed by hybrid consensus ranking.

Passage channels gather candidates. Whole-document BM25 and strongest-passage dense
scores rank documents using their strongest 60% of query scores. Representatives keep
the independently tested passage selector across the full query set.
Burmese retrieval is commented in burmese_search_disabled.py; source text is retained.
"""
import os
import hashlib
import re
import time
from collections import defaultdict

import numpy as np
from scipy import sparse

from corpus import INDEX, load_corpus, passage_table
from chunking import merge_short_passages
from embed import Embedder
from english import EN_STOP, en_tokens, stem, words
from kg import KG
from search_settings import SETTINGS

CHANNEL_DEPTH = SETTINGS.channel_depth


class BM25:
    def __init__(self, docs_tokens, k1=SETTINGS.bm25_k1, b=SETTINGS.bm25_b):
        vocab, rows, cols, vals = {}, [], [], []
        lengths = np.zeros(len(docs_tokens), dtype=np.float32)
        for d, toks in enumerate(docs_tokens):
            lengths[d] = len(toks)
            counts = defaultdict(int)
            for t in toks:
                counts[t] += 1
            for t, c in counts.items():
                rows.append(d)
                cols.append(vocab.setdefault(t, len(vocab)))
                vals.append(c)
        tf = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(docs_tokens), len(vocab)))
        df = np.bincount(tf.indices, minlength=len(vocab))
        idf = np.log(1 + (len(docs_tokens) - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg = max(float(lengths.mean()) if len(lengths) else 0, 1.0)
        tf = tf.tocoo()
        norm = k1 * (1 - b + b * lengths[tf.row] / avg)
        w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm)
        self.W = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.vocab = vocab

    def scores(self, q_tokens):
        cols = sorted({self.vocab[t] for t in q_tokens if t in self.vocab})
        if not cols:
            return None
        return np.asarray(self.W[:, cols].sum(axis=1)).ravel()


def minmax(values):
    """Finite values on [0,1]; a tied channel has equal strength for every member."""
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return a
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.ones_like(a)


def local_key(item):
    """Stable source-local hybrid order."""
    return (-item['fusion_score'], *item['key'])


def rank_strength(values, positive_only=False):
    """Normalize all allowed corpus scores, before candidate depth is applied."""
    values = np.asarray(values)
    valid = np.isfinite(values) & ((values > 0) if positive_only else True)
    ranked = np.flatnonzero(valid)
    ranked = ranked[np.argsort(-values[ranked], kind='stable')]
    result = np.zeros(len(values), dtype=np.float64)
    result[ranked] = (SETTINGS.rrf_k + 1) / (SETTINGS.rrf_k + np.arange(1, len(ranked) + 1))
    return result


def channel_strength(values, positive_only=False):
    """Equal raw/rank blend over allowed corpus units, with absent keyword evidence zero."""
    values = np.asarray(values)
    valid = np.isfinite(values) & ((values > 0) if positive_only else True)
    magnitudes = np.zeros(len(values), dtype=np.float64)
    if valid.any():
        matched = values[valid]
        lo, hi = matched.min(), matched.max()
        magnitudes[valid] = (matched - lo) / (hi - lo) if hi > lo else 1.
    return SETTINGS.rank_weight * rank_strength(values, positive_only) + (1 - SETTINGS.rank_weight) * magnitudes


def strongest_queries(values):
    """Average ceil(60% * query count) strongest scores independently for each document."""
    values = np.asarray(values)
    count = max(1, int(np.ceil(len(values) * SETTINGS.query_top_fraction)))
    return np.sort(values, axis=0)[-count:].mean(axis=0)


def source_normalized(items):
    """Merge independently normalized source scores only for a shared processing budget.

    No route counts, page hit bonuses, or cross-source raw-score fusion are involved.
    Source normalization affects only shared budgets, never the displayed source lists.
    """
    normalized = {}
    groups = defaultdict(list)
    for it in items:
        groups[it['key'][0]].append(it)
    for group in groups.values():
        vals = [it['fusion_score'] for it in group]
        normalized.update({it['key']: float(v) for it, v in zip(group, minmax(vals))})
    return sorted(items, key=lambda it: (-normalized[it['key']], *it['key']))


class Engine:
    def __init__(self):
        self.orders, self.chron = load_corpus()
        table = passage_table(self.orders, self.chron)
        self.embedder, self.kg, self.reranker = Embedder(), KG(load_graph=False), None
        self.pages = defaultdict(list)
        for i, s in enumerate(self.chron):
            self.pages[s['page_id']].append(i)
        self.page_ids = list(self.pages)
        self.page_idx = {pid: i for i, pid in enumerate(self.page_ids)}
        self.order_year = np.array([o['year'] for o in self.orders])
        self.order_part = np.array([o['part'] for o in self.orders])
        self.chron_vol = np.array([s['volume'] for s in self.chron])
        self.chron_real = np.array([bool(re.search(r'[A-Za-z0-9]', s['en'])) for s in self.chron])
        self._sid_idx = {s['id']: i for i, s in enumerate(self.chron)}
        self._oid_idx = {o['id']: i for i, o in enumerate(self.orders)}
        self.groups = {}
        self.word_forms = defaultdict(set)
        self.doc_stems = {}
        # Burmese indexes are deliberately not loaded, tokenized or scored.
        # Previous loop: for name in GROUPS (including order_my and chron_my).
        for name in ('order_en', 'chron_en'):
            rows = table[name]
            vec = np.load(INDEX / f'{name}.npy').astype(np.float32)
            assert len(vec) == len(rows), f'{name}: index/corpus mismatch; rebuild index'
            if name == 'order_en':
                rows, vec = self._coalesced_orders(rows, vec)
            tokens = [en_tokens(r[3]) for r in rows]
            doc_ids = np.array([r[1] if name == 'order_en' else self.page_idx[self.chron[r[1]]['page_id']]
                                for r in rows], np.int64)
            document_tokens = [[] for _ in (self.orders if name == 'order_en' else self.page_ids)]
            for doc, passage_tokens in zip(doc_ids, tokens):
                document_tokens[int(doc)].extend(passage_tokens)
            for r in rows:
                for w in words(r[3]):
                    if w not in EN_STOP:
                        self.word_forms[stem(w)].add(w)
            # stems present in each whole document, for narrowing results by query keyword
            self.doc_stems['o' if name == 'order_en' else 'p'] = [frozenset(t) for t in document_tokens]
            self.groups[name] = {
                'unit': np.array([r[1] for r in rows], np.int64),
                'doc': doc_ids,
                'pno': np.array([r[2] for r in rows], np.int64),
                'text': [r[3] for r in rows], 'vec': vec,
                'bm25': BM25(tokens, b=SETTINGS.candidate_bm25_b),
                'document_bm25': BM25(document_tokens),
            }
        self._sent_row = {int(u): r for r, u in enumerate(self.groups['chron_en']['unit'])}

    def _coalesced_orders(self, rows, vec):
        """Reuse stored vectors; embed only changed chunks. Original data/index files stay intact."""
        existing = {(r[1], r[3]): vec[i] for i, r in enumerate(rows)}
        merged = []
        for i, order in enumerate(self.orders):
            order['en_passages'] = merge_short_passages(order['en_passages'])
            merged += [('order', i, j, text) for j, text in enumerate(order['en_passages'])]
        texts = list(dict.fromkeys(r[3] for r in merged if (r[1], r[3]) not in existing))
        replacements = {}
        if texts:
            digest = hashlib.sha256(('\n'.join(texts) + str(self.embedder.sess.get_providers()) +
                                     os.getenv('EMBED_MODEL', 'multilingual-e5-small')).encode()).hexdigest()
            cache = INDEX / 'order_en_short_merge.npz'
            vectors = None
            if cache.exists():
                with np.load(cache) as saved:
                    if str(saved['digest']) == digest:
                        vectors = saved['vectors'].astype(np.float32)
            if vectors is None:
                vectors = self.embedder.passages(texts).astype(np.float32)
                np.savez_compressed(cache, digest=digest, vectors=vectors)
            replacements = dict(zip(texts, vectors))
        vectors = [existing[(r[1], r[3])] if (r[1], r[3]) in existing else replacements[r[3]] for r in merged]
        return merged, np.asarray(vectors, np.float32)

    def masks(self, sources=('order', 'chronicle'), year_from=None, year_to=None, parts=None, volumes=None):
        o = np.full(len(self.orders), 'order' in sources, bool)
        if year_from is not None:
            o &= self.order_year >= int(year_from)
        if year_to is not None:
            o &= self.order_year <= int(year_to)
        if parts:
            o &= np.isin(self.order_part, [int(p) for p in parts])
        c = self.chron_real & ('chronicle' in sources)
        if volumes:
            c &= np.isin(self.chron_vol, [int(v) for v in volumes])
        return o, c

    def _query_parts(self, query, my_terms=None, expand=True):
        q_en, aliases = en_tokens(query), []
        if expand and self.kg.ok:
            aliases, extra = self.kg.expand_query(query)
            q_en += en_tokens(' '.join(extra))
        # Disabled: my_tokens(query + ' ' + ' '.join(my_terms or [])).
        return self.embedder.query(query), list(dict.fromkeys(q_en)), [], aliases

    @staticmethod
    def _best(unit, scores, n_units):
        best = np.full(n_units, -np.inf, np.float32)
        np.maximum.at(best, unit, np.asarray(scores, np.float32))
        return best

    def _source_candidates(self, prefix, qparts, allowed, channels, depth, passage_scores=None,
                           scoring_tokens=None):
        g = self.groups[prefix + '_en']
        is_order = prefix == 'order'
        n = len(self.orders) if is_order else len(self.page_ids)
        valid_rows = allowed[g['unit']]
        qv, qt, _ = qparts
        raw = {}
        if 'dense' in channels:
            raw['dense'] = np.where(valid_rows, g['vec'] @ qv, -np.inf)
        if 'bm25' in channels:
            bm = g['bm25'].scores(qt if scoring_tokens is None else scoring_tokens)
            raw['bm25'] = np.where(valid_rows, bm if bm is not None else 0., -np.inf)
            # Aliases widen the candidate union but never affect final rank strengths.
            if scoring_tokens is not None and set(qt) != set(scoring_tokens):
                expanded = g['bm25'].scores(qt)
                raw['bm25_aliases'] = np.where(valid_rows, expanded if expanded is not None else 0., -np.inf)
        admitted = np.zeros(n, bool)
        linked, strengths = {}, {}
        for name, values in raw.items():
            doc_scores = self._best(g['doc'], values, n)
            keyword = name.startswith('bm25')
            eligible = np.isfinite(doc_scores) & ((doc_scores > 0) if keyword else True)
            ranked = np.flatnonzero(eligible)
            ranked = ranked[np.argsort(-doc_scores[ranked], kind='stable')]
            admitted[ranked if depth is None else ranked[:depth]] = True
            if name == 'bm25_aliases':
                continue
            if name == 'bm25':
                # Candidate admission is still passage-based. Whole-document keyword
                # scores are used only for ranking, preserving the tested recall routes.
                document_bm = g['document_bm25'].scores(qt if scoring_tokens is None else scoring_tokens)
                doc_scores = np.where(np.isfinite(doc_scores),
                                      document_bm if document_bm is not None else 0., -np.inf)
            strengths[name] = channel_strength(doc_scores, positive_only=keyword)
            best_rows = {}
            for row in np.argsort(-values, kind='stable'):
                if not np.isfinite(values[row]) or (keyword and values[row] <= 0):
                    break
                best_rows.setdefault(int(g['doc'][row]), int(row))
            linked[name] = best_rows
        weights = ({'bm25': 1 - SETTINGS.dense_weight, 'dense': SETTINGS.dense_weight}
                   if 'bm25' in channels and 'dense' in channels else {c: 1. for c in channels})
        scores = sum((weight * strengths[name] for name, weight in weights.items() if name in strengths),
                     np.zeros(n))
        if passage_scores is not None:
            representative_weights = ({'bm25': 1 - SETTINGS.representative_dense_weight,
                                       'dense': SETTINGS.representative_dense_weight}
                                      if 'bm25' in channels and 'dense' in channels else weights)
            combined = sum((weight * rank_strength(raw[name], positive_only=name == 'bm25')
                            for name, weight in representative_weights.items() if name in raw), np.zeros(len(g['text'])))
            passage_scores[prefix] = np.where(valid_rows, combined, -np.inf)
        first_rows = {}
        for row in np.flatnonzero(valid_rows):
            first_rows.setdefault(int(g['doc'][row]), int(row))
        out = []
        for doc, fallback in first_rows.items():
            row = linked.get('bm25', {}).get(doc, linked.get('dense', {}).get(doc, fallback))
            key = ('o', doc) if is_order else ('p', self.page_ids[doc])
            out.append({'key': key, 'prefix': prefix, 'row': row,
                        'unit': int(g['unit'][row]), 'passage': int(g['pno'][row]),
                        'dense_row': linked.get('dense', {}).get(doc),
                        'keyword_row': linked.get('bm25', {}).get(doc),
                        'fusion_score': float(scores[doc]), 'rerank_score': None,
                        'admitted': bool(admitted[doc])})
        return sorted(out, key=local_key)

    def retrieval(self, query, sources=('order', 'chronicle'), year_from=None, year_to=None,
                  parts=None, volumes=None, channels=('dense', 'bm25'), expand=True, depth=CHANNEL_DEPTH,
                  restrict_keys=None):
        qv, qt, qm, aliases = self._query_parts(query, None, expand)
        qp = (qv, qt, qm)
        om, cm = self.masks(sources, year_from, year_to, parts, volumes)
        if restrict_keys is not None:
            # Follow-up queries score only unread members of the original pool.
            keys = set(restrict_keys)
            om &= np.array([('o', i) in keys for i in range(len(self.orders))], dtype=bool)
            cm &= np.array([('p', s['page_id']) in keys for s in self.chron], dtype=bool)
        passage_scores = {}
        # Normal stemming/stopwords only: the original and Gemini query strings are never stripped.
        scoring_tokens = en_tokens(query)
        items = self._source_candidates('order', qp, om, channels, depth, passage_scores, scoring_tokens) if 'order' in sources else []
        if 'chronicle' in sources:
            items += self._source_candidates('chron', qp, cm, channels, depth, passage_scores, scoring_tokens)
        return {'all': {it['key']: it for it in items}, 'items': [it for it in items if it['admitted']],
                'qparts': [qp], 'aliases': aliases, 'query': query, 'passage_scores': passage_scores}

    def consensus(self, retrievals):
        """Score every allowed document across unique queries; never use discovery counts.

        Documents average their strongest 60% of unique query scores. Passage selection
        preserves the existing mean across all queries, independently of document ranking.
        """
        retrievals = list({r['query']: r for r in retrievals}.values())
        if not retrievals:
            return {}
        keys = list(retrievals[0]['all'])
        if not keys:
            return {}
        scores = strongest_queries([[r['all'].get(k, {}).get('fusion_score', 0.) for k in keys]
                                     for r in retrievals])
        representatives = {}
        for prefix in ('order', 'chron'):
            values = [r['passage_scores'][prefix] for r in retrievals if prefix in r['passage_scores']]
            if not values:
                continue
            mean = np.mean(values, axis=0)
            g = self.groups[prefix + '_en']
            rows = np.arange(len(mean))
            by_doc = {}
            for row in np.lexsort((rows, -mean)):
                if not np.isfinite(mean[row]) or mean[row] <= 0:
                    continue
                by_doc.setdefault(int(g['doc'][row]), int(row))
            representatives[prefix] = by_doc
        result = {}
        for key, score in zip(keys, scores):
            it = dict(retrievals[0]['all'][key])
            prefix = it['prefix']
            doc = key[1] if key[0] == 'o' else self.page_idx[key[1]]
            row = representatives.get(prefix, {}).get(doc, it['row'])
            g = self.groups[prefix + '_en']
            it.update(fusion_score=float(score), rerank_score=None, row=row,
                      unit=int(g['unit'][row]), passage=int(g['pno'][row]),
                      keyword_row=representatives.get(prefix, {}).get(doc), query_count=len(retrievals))
            result[key] = it
        return result

    def rank(self, query, sources=('order', 'chronicle'), year_from=None, year_to=None, parts=None,
             volumes=None, extra_queries=None, my_terms=None, rerank=False, channels=('dense', 'bm25'),
             expand=True):
        """Shared ranking; legacy rerank/my_terms arguments cannot enable retired app paths."""
        queries = list(dict.fromkeys(q.strip() for q in [query] + list(extra_queries or [])
                                    if isinstance(q, str) and q.strip()))
        retrieved = [self.retrieval(q, sources, year_from, year_to, parts, volumes, channels, expand)
                     for q in queries]
        r = retrieved[0] if retrieved else {'all': {}, 'items': [], 'aliases': [], 'qparts': [(None, [], [])]}
        ranked = self.consensus(retrieved)
        selected = {it['key'] for result in retrieved for it in result['items']}
        items = [ranked[k] for k in selected]
        r = {**r, 'aliases': list(dict.fromkeys(a for result in retrieved for a in result['aliases'])),
             'qparts': [qp for result in retrieved for qp in result['qparts']], 'queries': queries}
        items.sort(key=local_key)
        orders = [it for it in items if it['key'][0] == 'o']
        pages = [it for it in items if it['key'][0] == 'p']
        # Positive rank markers retain zero-keyword dense discoveries in benchmark page order.
        cs, oscores = np.zeros(len(self.chron)), np.zeros(len(self.orders))
        for rank, it in enumerate(pages):
            cs[it['unit']] = len(pages) - rank
        for it in orders:
            oscores[it['key'][1]] = it['fusion_score']
        return {**r, 'items': items, 'orders': [it['unit'] for it in orders],
                'sentences': [it['unit'] for it in pages], 'pages': pages, 'o_scores': oscores,
                'c_scores': cs, 'rr': {}, 'rr_ms': 0, 'burmese': False}

    def search(self, query, sources=('order', 'chronicle'), year_from=None, year_to=None,
               parts=None, volumes=None, k_orders=None, k_pages=None, extra_queries=None,
               my_terms=None, rerank=False):
        # k_* are accepted for old callers but intentionally do not truncate gathered evidence.
        r = self.rank(query, sources, year_from, year_to, parts, volumes, extra_queries, my_terms, rerank)
        orders, pages = [], []
        stems = [t for qp in r['qparts'] for t in qp[1]]
        kws = self.keywords(stems, [query, *(extra_queries or []), *r['aliases']])
        stems = [k['stem'] for k in kws]
        for it in r['items']:
            (orders if it['key'][0] == 'o' else pages).append({**self.hit(it), 'terms': self.terms_in(it['key'], stems)})
        for k in kws:
            k['n'] = sum(k['stem'] in h['terms'] for h in orders + pages)
        return {'orders': orders, 'pages': pages, 'keywords': [k for k in kws if k['n']],
                'reranked': bool(r['rr']), 'rerank_ms': int(r['rr_ms']),
                'aliases': r['aliases'], 'highlight_terms': self.highlight_terms([t for qp in r['qparts'] for t in qp[1]]),
                'settings': SETTINGS.public()}

    def keywords(self, stems, texts):
        """Query keywords in first-use order, each with a word form taken from the query or alias texts."""
        form = {}
        for t in texts:
            for w in words(t):
                if w not in EN_STOP:
                    form.setdefault(stem(w), w)
        return [{'stem': t, 'word': form.get(t, t)} for t in dict.fromkeys(stems)]

    def terms_in(self, key, stems):
        """Which of the given keyword stems occur anywhere in this Order or chronicle page."""
        have = self.doc_stems['o'][key[1]] if key[0] == 'o' else self.doc_stems['p'][self.page_idx[key[1]]]
        return [t for t in stems if t in have]

    def highlight_terms(self, tokens):
        return sorted({w for t in set(tokens) for w in self.word_forms.get(t, ())})

    def hit(self, item):
        score = item['fusion_score']
        common = {'score': score, 'fusion_score': item['fusion_score'], 'rerank': None,
                  'score_kind': 'hybrid_consensus', 'query_count': item.get('query_count', 1),
                  'representative': {'unit': item['unit'], 'passage': item['passage'],
                                     'dense_row': item['dense_row'], 'keyword_row': item['keyword_row']}}
        if item['key'][0] == 'o':
            o = self.orders[item['key'][1]]
            fields = ('id', 'part', 'date', 'date_supplied', 'date_uncertain', 'pdf', 'pdf_pages',
                      'topics', 'en_passages', 'my_passages', 'my_alignment')
            return {**{k: o[k] for k in fields}, **common,
                    'hit_en': [item['passage']], 'best_en': [item['passage']], 'hit_my': []}
        idxs = self.pages[item['key'][1]]
        first = self.chron[idxs[0]]
        return {**common, 'page_id': item['key'][1], 'volume': first['volume'], 'page': first['page'],
                'hits': [self.chron[item['unit']]['id']],
                'sentences': [{k: self.chron[i][k] for k in ('id', 'my', 'en')} for i in idxs]}

    def page_rank(self, scores, depth=None):
        by_page = {}
        for i in np.argsort(-scores, kind='stable'):
            if scores[i] <= 0:
                continue
            by_page.setdefault(self.chron[i]['page_id'], (float(scores[i]), self.chron[i]['page_id'], [int(i)]))
        return list(by_page.values())

    def _pinpoint(self, prefix, lang, unit, qparts, keep=1):
        if lang != 'en':
            return [], []
        masks = np.ones(len(self.orders) if prefix == 'order' else len(self.chron), bool)
        key = ('o', unit) if prefix == 'order' else ('p', self.chron[unit]['page_id'])
        items = self._source_candidates(prefix, qparts[0], masks, ('dense', 'bm25'), None)
        it = next((it for it in items if it['key'] == key), None)
        best = [it['passage']] if it else []
        return best, best

    def _order_hit(self, i, score, qparts):
        items = self._source_candidates('order', qparts[0], np.ones(len(self.orders), bool), ('dense', 'bm25'), None)
        return self.hit(next(it for it in items if it['key'] == ('o', i)))

    def order_text(self, oid):
        i = self._oid_idx.get(oid)
        return self.orders[i] if i is not None else None
