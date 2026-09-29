import csv
import json
import importlib.util
import math
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'scripts'))
from chunking import merge_short_passages
from build_orders import segment, english_passages, parse_date_line, preserve_order_ids
from dossier import Dossier
from english import en_tokens
from kg import KG
from search import BM25, Engine, channel_strength, local_key, rank_strength, source_normalized, strongest_queries
from search_settings import SETTINGS
from replay_search import baseline_labels, saved_plan


class Reranker:
    def __init__(self):
        self.calls = []

    def scores(self, query, texts, max_len=None):
        self.calls.append((query, list(texts), max_len))
        return np.array([-10 + int(re.search(r'\d+', t)[0]) / 100 for t in texts])


def fixture(n=350):
    e = Engine.__new__(Engine)
    e.orders = [{'id': f'o{i}', 'date': f'1800-01-{i%28+1:02}', 'year': 1800 if i%2 else 1790,
                 'part': 3 if i%2 else 4, 'date_supplied': False, 'date_uncertain': False,
                 'pdf': 'source.pdf', 'pdf_pages': [i+1], 'topics': [],
                 'en_passages': [f'order {i} dense passage', f'order {i} keyword passage'],
                 'my_passages': ['original Burmese text'], 'my_alignment': 'paired'} for i in range(n)]
    e.chron = [{'id': f's{i}_{j}', 'page_id': f'p{i}', 'volume': 1 if i%2 else 2, 'page': i+1,
                'en': f'chronicle {i} sentence {j}', 'my': 'Burmese'} for i in range(n) for j in range(3)]
    e.pages = defaultdict(list)
    for i,s in enumerate(e.chron): e.pages[s['page_id']].append(i)
    e.page_ids = list(e.pages)
    e.page_idx = {p:i for i,p in enumerate(e.page_ids)}
    e.order_year = np.array([o['year'] for o in e.orders])
    e.order_part = np.array([o['part'] for o in e.orders])
    e.chron_vol = np.array([s['volume'] for s in e.chron])
    e.chron_real = np.ones(len(e.chron),bool)
    e._sid_idx = {s['id']:i for i,s in enumerate(e.chron)}
    e._oid_idx = {o['id']:i for i,o in enumerate(e.orders)}
    e.embedder = SimpleNamespace(query=lambda q: np.array([1.,0.]))
    e.reranker = Reranker()
    e.kg = SimpleNamespace(ok=False)
    e.word_forms = {}
    e.doc_stems = {'o':[frozenset(en_tokens(' '.join(o['en_passages']))) for o in e.orders],
                   'p':[frozenset(en_tokens(' '.join(e.chron[i]['en'] for i in e.pages[p]))) for p in e.page_ids]}
    e.groups = {}
    for prefix, count in [('order',2),('chron',3)]:
        rows = [(i,j) for i in range(n) for j in range(count)]
        dense = np.array([.95-.6*i/max(n-1,1)-.01*j for i,j in rows])
        bm = np.array([1+10*i/max(n-1,1)+j for i,j in rows])
        doc_bm = bm.reshape(n,count).max(axis=1)
        e.groups[prefix+'_en'] = {'unit':np.array([i if prefix=='order' else i*3+j for i,j in rows]),
            'doc':np.array([i for i,j in rows]), 'pno':np.array([j if prefix=='order' else 0 for i,j in rows]),
            'text':[e.orders[i]['en_passages'][j] if prefix=='order' else e.chron[i*3+j]['en'] for i,j in rows],
            'vec':np.array([[v,math.sqrt(1-v*v)] for v in dense]),
            'bm25':SimpleNamespace(scores=lambda toks,bm=bm: bm if toks else None),
            'document_bm25':SimpleNamespace(scores=lambda toks,bm=doc_bm: bm if toks else None)}
    e._sent_row = {i:i for i in range(len(e.chron))}
    return e


class MockGemini:
    def __init__(self, omit=False):
        self.calls=[]
        self.requests=[]
        self.omit=omit

    def json_call(self,prompt,schema,max_out,kind):
        self.calls.append((kind,prompt))
        self.requests.append((kind,prompt,max_out))
        if kind=='plan_search': data=getattr(self,'plan',{'queries':['another query','another query']})
        elif kind=='assess':
            ids=re.findall(r'^([OC]\d+) \|',prompt,re.M)
            data={'items':[{'id':a,'v':'r' if i%3==0 else 'p' if i%3==1 else 'n','s':'Relevant evidence.'}
                           for i,a in enumerate(ids) if not(self.omit and a=='O1')]}
        elif kind in ('answer','followup','deeper_summary'): data={'answer':'Evidence [O1].'}
        else: raise AssertionError(kind)
        return data,{'in':10,'out':10,'think':0}


class RetrievalTests(unittest.TestCase):
    def test_real_stemming_and_short_words(self):
        self.assertEqual(en_tokens('Connections connected connecting'), ['connect']*3)
        self.assertEqual(en_tokens('U x 7 a the'),['u','x','7'])
        self.assertEqual(en_tokens('King king'),['king','king'])

    def test_short_records_and_fragments_preserved(self):
        self.assertEqual(merge_short_passages(['short order']),['short order'])
        chunks=['Heading:', 'long '+('word '*40), 'End.']
        merged=merge_short_passages(chunks)
        self.assertEqual(' '.join(chunks),' '.join(merged))
        self.assertEqual(len(merged),1)
        entries=segment([(1,'1 January 1800'),(1,'Keep it.'),(1,'2 January 1800'),(1,'Yes.')],(1790,1810))
        self.assertEqual(len(entries),2)
        self.assertEqual(english_passages('Yes.'),['Yes.'])

    def test_full_candidate_pool_and_no_cross_encoder_even_with_legacy_flag(self):
        e=fixture()
        result=e.search('royal events',k_orders=1,k_pages=1,rerank=True)
        self.assertEqual(len(result['orders']),350)
        self.assertEqual(len(result['pages']),350)
        self.assertEqual(e.reranker.calls,[])
        self.assertFalse(result['reranked'])
        for items in (result['orders'],result['pages']):
            self.assertTrue(all(x['rerank'] is None for x in items))
            scores=[x['score'] for x in items]
            self.assertEqual(scores,sorted(scores,reverse=True))
            self.assertTrue(all(x['score_kind']=='hybrid_consensus' for x in items))
        self.assertTrue(all(len(x['hits'])==1 for x in result['pages']))
        self.assertTrue(all(x['hit_my']==[] for x in result['orders']))

    def test_page_deduplication_before_250_channel_limit(self):
        e=fixture()
        r=e.retrieval('events',channels=('dense',))
        self.assertEqual(sum(x['key'][0]=='p' for x in r['items']),250)
        self.assertEqual(sum(x['key'][0]=='o' for x in r['items']),250)
        self.assertEqual(len({x['key'] for x in r['items']}),500)
        self.assertTrue(all(x['row'] in (x['dense_row'],x['keyword_row']) for x in r['items']))

    def test_rank_strength_has_no_raw_magnitude_component(self):
        np.testing.assert_allclose(rank_strength([2.,0.,4.,-np.inf],True),[61/62,0,1,0])
        np.testing.assert_allclose(rank_strength([20.,0.,400.,-np.inf],True),[61/62,0,1,0])
        e=fixture(3)
        r=e.retrieval('events',sources=['order'])
        for i,it in enumerate(sorted(r['items'],key=lambda x:x['key'][1],reverse=True)):
            self.assertAlmostEqual(it['fusion_score'],.6*(.5*61/(61+i)+.5*(1-i/2))
                                   +.4*(.5*61/(63-i)+.5*i/2),places=6)

    def test_default_bm25_uses_tested_saturation(self):
        docs=[['royal']*20+['event']*5,['royal','king'],['king']*10]
        default=BM25(docs).scores(['royal'])
        np.testing.assert_allclose(default,BM25(docs,k1=.6,b=.75).scores(['royal']))
        self.assertFalse(np.allclose(default,BM25(docs,k1=1.2,b=.75).scores(['royal'])))

    def test_strongest_queries_deduplication_and_independent_passage_selection(self):
        e=fixture(3)
        vectors={'origin':np.array([30,0,10,0,20,0]),
                 'plan':np.array([0,0,0,30,20,0]),
                 'third':np.array([0,0,0,30,20,0])}
        e._query_parts=lambda q,*args: (np.array([1.,0.]),en_tokens(q),[],[])
        e.groups['order_en']['bm25']=SimpleNamespace(scores=lambda toks:vectors[toks[0]])
        e.groups['order_en']['document_bm25']=SimpleNamespace(scores=lambda toks:vectors[toks[0]].reshape(3,2).max(axis=1))
        base=e.rank('original',sources=['order'])
        new=e.rank('original',sources=['order'],extra_queries=['planned','third','original','planned',' '])
        self.assertEqual(base['orders'],[0,2,1])
        self.assertEqual(new['orders'],[1,0,2])
        self.assertEqual(new['items'][0]['passage'],1)
        self.assertEqual(new['items'][0]['query_count'],3)
        self.assertAlmostEqual(new['items'][0]['fusion_score'],.6+.4*(.5*61/62+.25),places=6)
        self.assertEqual(e.reranker.calls,[])

    def test_dense_candidates_without_keyword_matches_survive(self):
        e=fixture(8)
        for g in e.groups.values():
            g['bm25']=SimpleNamespace(scores=lambda toks:None)
            g['document_bm25']=SimpleNamespace(scores=lambda toks:None)
        r=e.search('no known vocabulary')
        self.assertEqual(len(r['orders']),8)
        self.assertEqual(len(r['pages']),8)
        self.assertTrue(all(0 < x['score'] <= .4 for x in r['orders']+r['pages']))
        self.assertTrue(all(len(p['hits'])==1 for p in r['pages']))

    def test_non_english_chronicle_rows_never_supply_representative(self):
        e=fixture(3);e.chron_real[2]=False
        r=e.rank('events',sources=['chronicle'],extra_queries=['another'])
        p=next(it for it in r['items'] if it['key']==('p','p0'))
        self.assertNotEqual(p['unit'],2)

    def test_aliases_gather_candidates_without_changing_scores_or_passages(self):
        e=fixture(4)
        e.kg=SimpleNamespace(ok=True,expand_query=lambda q:(['variant'],['variant']))
        e.groups['order_en']['bm25']=SimpleNamespace(scores=lambda toks:
            np.array([0,0,0,0,0,0,50,0]) if 'variant' in toks else np.array([0,0,5,0,0,0,0,0]))
        plain=e.retrieval('royal events',sources=['order'],expand=False,depth=1)
        expanded=e.retrieval('royal events',sources=['order'],expand=True,depth=1)
        self.assertNotIn(('o',3),{it['key'] for it in plain['items']})
        self.assertIn(('o',3),{it['key'] for it in expanded['items']})
        for key in plain['all']:
            self.assertEqual(plain['all'][key]['fusion_score'],expanded['all'][key]['fusion_score'])
        np.testing.assert_array_equal(plain['passage_scores']['order'],expanded['passage_scores']['order'])

    def test_query_text_and_historical_context_are_not_stripped(self):
        e=fixture(3);embedded=[];tokenized=[]
        e.embedder=SimpleNamespace(query=lambda q: embedded.append(q) or np.array([1.,0.]))
        e.groups['order_en']['bm25']=SimpleNamespace(scores=lambda toks: tokenized.append(toks) or None)
        query='Than Tun Konbaung Burma Burmese Royal Orders'
        result=e.retrieval(query,sources=['order'])
        self.assertEqual(result['query'],query)
        self.assertEqual(embedded,[query])
        self.assertIn(en_tokens(query),tokenized)

    def test_raw_rank_normalization_handles_zeros_ties_and_filtered_rows(self):
        np.testing.assert_allclose(channel_strength([2.,0.,4.,-np.inf],True),[.5*61/62,0,1,0])
        np.testing.assert_allclose(channel_strength([7.,7.]),[1,.5+.5*61/62])
        np.testing.assert_array_equal(channel_strength([0.,-np.inf],True),[0,0])

    def test_query_fraction_uses_ceiling_and_can_keep_a_specialist_document(self):
        values=np.array([[1.,.4],[1.,.4],[0.,.4]])
        np.testing.assert_allclose(strongest_queries(values),[1.,.4])
        np.testing.assert_array_equal(strongest_queries([[.2,.8]]),[.2,.8])

    def test_whole_document_keyword_scoring_combines_separate_passages(self):
        e=fixture(2)
        e.groups['order_en']['document_bm25']=BM25([en_tokens('coronation consecration'),en_tokens('coronation other')])
        # Candidate/representative passage scores need not contain both query terms.
        e.groups['order_en']['bm25']=SimpleNamespace(scores=lambda toks:np.array([1.,1.,2.,0.]))
        r=e.retrieval('coronation consecration',sources=['order'],channels=('bm25',))
        self.assertGreater(r['all'][('o',0)]['fusion_score'],r['all'][('o',1)]['fusion_score'])
        self.assertEqual(r['all'][('o',0)]['keyword_row'],0)

    def test_filters_apply_before_fusion(self):
        e=fixture(20)
        r=e.retrieval('events',year_from=1795,parts=['3'],volumes=['1'])
        self.assertTrue(all(e.orders[it['key'][1]]['part']==3 for it in r['items'] if it['key'][0]=='o'))
        self.assertTrue(all(e.chron[it['unit']]['volume']==1 for it in r['items'] if it['key'][0]=='p'))
        self.assertEqual(e.search('query',sources=[])['orders'],[])

    def test_empty_query_set_and_filters_return_no_evidence(self):
        e=fixture(5)
        self.assertEqual(e.rank(' ')['items'],[])
        self.assertEqual(e.rank('query',sources=[])['items'],[])
        self.assertEqual(e.rank('query',year_from=2100,sources=['order'])['items'],[])

    def test_source_normalization_prevents_raw_scale_competition(self):
        items=[{'key':('o',i),'fusion_score':v,'rerank_score':None} for i,v in enumerate([100,50,1])]
        items += [{'key':('p',i),'fusion_score':v,'rerank_score':None} for i,v in enumerate([.9,.5,.1])]
        self.assertEqual({it['key'][0] for it in source_normalized(items)[:2]},{'o','p'})

    def test_aliases_any_span_any_length_ten_per_entity(self):
        kg=KG.__new__(KG);kg.ok=True
        kg.members={'e1': [('Long Name With More Than Four Words',100),('U',99),('King',98)]+[(f'alias {i}',90-i) for i in range(12)],
                    'e2':[('Second',20),('X',10)]}
        kg.mentions={'e1':100,'e2':20}
        kg._alias_index()
        for query in ['Long Name With More Than Four Words','U','King','alias 11']:
            found,_=kg.expand_query(query)
            self.assertEqual(len(found),10)
            self.assertIn('U',found)
        found,_=kg.expand_query('King Second')
        self.assertEqual(len(found),12) # no global eight-alias cap


class DossierTests(unittest.TestCase):
    def dossier(self,n=350,sp=None):
        return Dossier(fixture(n),sp or MockGemini(),{'q':'royal evidence','sources':['order','chronicle'],'response_tokens':500,'reading_rounds':0})

    def test_deduplication_and_shared_scoring(self):
        d=self.dossier(20);key=('o',1)
        d._add(key,'first');original=d.pool[key]['fusion_score']
        d._add(key,'again');d._add(key,'again')
        self.assertEqual(d.pool[key]['fusion_score'],original)
        self.assertEqual(d.pool[key]['routes'],['first','again'])
        d.score();d.score()
        self.assertEqual(d.e.reranker.calls,[])
        self.assertEqual(d.pool[key]['fusion_score'],original)

    def test_complete_pipeline_two_batches_no_entity_reranker_or_centroid(self):
        d=self.dossier();before=set(it['key'] for it in d.base['items'])
        d.run()
        self.assertEqual(d.e.reranker.calls,[])
        self.assertEqual(len(d.assessed),50)
        batches=[p for kind,p in d.sp.calls if kind=='assess']
        self.assertEqual([len(re.findall(r'^([OC]\d+) \|',p,re.M)) for p in batches],[25,25])
        self.assertEqual(before,set(d.pool))
        self.assertEqual(len(d.verdict),50)
        self.assertEqual([kind for kind,_ in d.sp.calls].count('plan_search'),1)
        self.assertTrue(all(kind in ('plan_search','assess','answer') for kind,_ in d.sp.calls))
        self.assertFalse(hasattr(d,'centroid'))
        self.assertFalse(hasattr(d,'choose_entities'))
        self.assertEqual(len(d.retrievals),2)
        self.assertEqual(d.log['scoring'][0]['units_scored'],len(d.pool))
        self.assertEqual(d.log['scoring'][0]['method'],'hybrid_consensus')
        self.assertTrue(all(it['query_count']==2 for it in d.pool.values()))

    def test_verdict_groups_use_keyword_ranking_with_no_feedback(self):
        d=self.dossier(6)
        for i in range(6):d._add(('o',i),'test')
        for i,it in enumerate(d.pool.values()):it['fusion_score']=float(i)
        d.verdict={('o',0):{'v':'r'},('o',1):{'v':'p'},('o',2):{'v':'r'},('o',5):{'v':'n'}}
        self.assertEqual([it['key'][1] for it in d.items()],[2,0,1,4,3,5])

    def test_ordinary_and_spider_share_full_query_consensus(self):
        e=fixture(30)
        d=Dossier(e,MockGemini(),{'q':'royal evidence','sources':['order','chronicle']})
        d.queries=['planned one','planned two','planned one']
        d.gather();d.score()
        ordinary=e.rank(d.q,extra_queries=d.queries)
        self.assertEqual(set(d.pool),{it['key'] for it in ordinary['items']})
        for it in ordinary['items']:
            self.assertEqual(d.pool[it['key']]['fusion_score'],it['fusion_score'])
            self.assertEqual(d.pool[it['key']]['row'],it['row'])
        self.assertEqual(e.reranker.calls,[])

    def test_missing_verdicts_are_retried_and_retained(self):
        d=self.dossier(5,MockGemini(omit=True));d.gather();d._aliases(d.items())
        for it in d.items(): it['alias']=d.alias_of[it['key']]
        d.assess(d.items(),'test')
        self.assertEqual(len(d.missing),1)
        self.assertEqual(len(d.pool),10)
        self.assertEqual(len([k for k,_ in d.sp.calls if k=='assess']),2)
        self.assertNotIn(next(iter(d.missing)),d.verdict)

    def test_excerpts_equal_budget_and_include_core(self):
        d=self.dossier(1)
        for key,it in d.base['all'].items():
            if key[0]=='o':d.e.orders[0]['en_passages']=['before '*100,'COREORDER '+('body '*150),'after '*100];it['passage']=1
            else:
                for j in range(3):d.e.chron[j]['en']=('COREPAGE ' if j==1 else '')+('content '*100)
                it['unit']=1
            text=d.excerpt(it)
            self.assertEqual(len(text),1000)
            self.assertIn('COREORDER' if key[0]=='o' else 'COREPAGE',text)

    def test_graph_never_loaded_or_planned_in_search(self):
        d=self.dossier(20)
        d.e.kg=SimpleNamespace(ok=True,expand_query=lambda q:([],[]),
            resolve=lambda *a,**k: self.fail('graph resolution called'),
            select=lambda *a,**k: self.fail('graph traversal called'))
        d.run()
        self.assertFalse(hasattr(d,'lookup'))
        self.assertFalse(hasattr(d,'_gather_graph'))
        self.assertNotIn('plan_graph',[kind for kind,_ in d.sp.calls])
        self.assertFalse(any('graph' in route for it in d.items() for route in it['routes']))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'entities.csv').write_text('entity,parent_id,mentions,parent_entity\nKing,e1,1,King\n')
            kg=KG(p,load_graph=False)
            self.assertTrue(kg.ok)
            self.assertEqual(kg.triples,[])
            self.assertFalse(hasattr(kg,'edges'))
            self.assertEqual(kg.expand_query('King')[0],['King'])

    def test_default_reading_reuses_assessment_without_more_searches(self):
        d=Dossier(fixture(100),MockGemini(),{'q':'royal evidence','sources':['order','chronicle']})
        with patch.object(d.e,'retrieval',wraps=d.e.retrieval) as retrieval, patch.object(d.e,'consensus',wraps=d.e.consensus) as consensus:
            d.run()
        self.assertEqual(len(d.assessed),100)
        self.assertEqual(len(d.log['shortlists']),2)
        actual=[kid for shortlist in d.log['shortlists'] for kid in shortlist['items']]
        expected=[d._kid(it['key']) for it in source_normalized(d.pool.values())[:100]]
        self.assertEqual(actual,expected)
        self.assertEqual(len(set(actual)),100)
        self.assertEqual(d.completed_reading_rounds,1)
        self.assertEqual(len(d.sp.calls),6)
        self.assertEqual(retrieval.call_count,1)  # planned query; original cached in constructor
        self.assertNotIn('restrict_keys',retrieval.call_args.kwargs)
        self.assertEqual(consensus.call_count,1)
        self.assertEqual(len(d.pool),200)
        self.assertEqual(d.followups,[])
        self.assertEqual(d.deeper_rounds,[])

    def test_four_reading_rounds_follow_original_order_without_duplicates(self):
        d=self.dossier(350);d.reading_rounds=4
        d.run()
        batches=[x['items'] for x in d.log['shortlists']]
        self.assertEqual([len(x) for x in batches],[50]*5)
        actual=[kid for batch in batches for kid in batch]
        expected=[d._kid(it['key']) for it in source_normalized(d.pool.values())[:250]]
        self.assertEqual(actual,expected)
        self.assertEqual(len(set(actual)),250)
        self.assertEqual(d.completed_reading_rounds,4)
        self.assertEqual(len(d.sp.calls),12)
        self.assertEqual([kind for kind,_ in d.sp.calls].count('plan_search'),1)
        self.assertEqual(len({d.alias_of[k] for k in d.pool}),len(d.pool))

    def test_reading_stops_when_pool_exhausted_or_empty(self):
        d=self.dossier(30);d.reading_rounds=4;d.run()
        self.assertEqual(len(d.assessed),60)
        self.assertEqual([len(x['items']) for x in d.log['shortlists']],[50,10])
        self.assertEqual(d.completed_reading_rounds,1)
        self.assertEqual(len(d.retrievals),2)
        self.assertEqual(len(d.sp.calls),5)
        d=Dossier(fixture(3),MockGemini(),{'q':'query','sources':[],'reading_rounds':4})
        d.run()
        self.assertEqual([k for k,_ in d.sp.calls],['plan_search'])
        self.assertEqual(d.completed_reading_rounds,0)

    def test_reading_preserves_pool_scores_representatives_and_aliases(self):
        d=self.dossier(100)
        d.gather();d.score();d._aliases(d.items())
        before={key:dict(it) for key,it in d.pool.items()}
        aliases=dict(d.alias_of)
        with patch.object(d.e,'retrieval',side_effect=AssertionError('Unexpected search')), patch.object(d.e,'consensus',side_effect=AssertionError('Unexpected ranking')):
            d._assess_next(0);d._assess_next(1)
        self.assertEqual(set(d.pool),set(before))
        for key,old in before.items():
            for field,value in old.items():
                self.assertEqual(d.pool[key][field],value)
        self.assertEqual(d.alias_of,aliases)
        expected=[d._kid(it['key']) for it in source_normalized(before.values())[:100]]
        self.assertEqual([kid for x in d.log['shortlists'] for kid in x['items']],expected)

    def test_response_budget_reaches_gemini_and_controls_prompt(self):
        for rounds,budget in ((0,500),(1,1000),(4,2500)):
            with self.subTest(budget=budget):
                d=Dossier(fixture(150),MockGemini(),{'q':'royal evidence','reading_rounds':rounds,'response_tokens':99999})
                d.run()
                kind,prompt,cap=d.sp.requests[-1]
                self.assertEqual(kind,'answer')
                self.assertEqual(cap,budget)
                self.assertIn(f'hard budget of {budget} output tokens',prompt)
                self.assertIn('Relevant evidence.',prompt)
                self.assertIn('Cite ids',prompt)
                self.assertIn('supplied summaries',prompt)
                self.assertIn('developed historical essay' if budget>=1000 else 'concise overview',prompt)
                self.assertEqual(d.log['gemini'][-1]['max_output_tokens'],budget)
                # All assessed positive summaries, including the second batch, reach synthesis.
                ids=set(re.findall(r'^([OC]\d+) \|',prompt,re.M))
                self.assertEqual(ids,{d.alias_of[k] for k,v in d.verdict.items() if v['v'] in ('r','p')})
                if rounds:
                    self.assertTrue(ids & {d.alias_of[k] for k in d.assessed if d._kid(k) in d.log['shortlists'][1]['items']})
                with tempfile.TemporaryDirectory() as tmp,patch('dossier.RUN_LOGS',Path(tmp)):
                    logged=json.loads(Path(d.save_log()).read_text(encoding='utf-8'))
                self.assertEqual(logged['response_tokens'],budget)
                self.assertEqual(logged['completed_reading_rounds'],rounds)

    def test_direct_dossier_validates_controls_and_accepts_old_round_key(self):
        self.assertEqual(Dossier(fixture(1),MockGemini(),{'q':'query','recursive_searches':0}).reading_rounds,0)
        for field,invalid in [('reading_rounds',[-1,5,11,1.5,True,'2'])]:
            for value in invalid:
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    Dossier(fixture(1),MockGemini(),{'q':'query',field:value})

    def test_no_planned_queries_still_runs_original_question(self):
        d=self.dossier(3)
        d.queries=[]
        d.gather();d.score()
        self.assertEqual(list(d.retrievals),[d.q])
        self.assertEqual(len(d.pool),6)
        self.assertTrue(all(it['query_count']==1 for it in d.pool.values()))

    def test_query_whitespace_does_not_add_an_extra_consensus_vote(self):
        d=Dossier(fixture(3),MockGemini(),{'q':'  royal evidence  ','sources':['order','chronicle']})
        d.queries=['royal evidence',' another query ','another query']
        d.gather();d.score()
        self.assertEqual(list(d.retrievals),['royal evidence','another query'])
        self.assertTrue(all(it['query_count']==2 for it in d.pool.values()))

    def test_graph_lookup_accepts_single_pair_triplet(self):
        kg=KG.__new__(KG);kg.ok=True
        kg.name={};kg.by_norm={};kg.pred_count={}
        kg.ecat_name={'E01':'King','E02':'Place'};kg.rel_name={'R01':'Visits','R02':'Rules'}
        kg.triples=[{'s':'a','o':'b','sE':'E01','oE':'E02','r':'R01','p':'VISITS'},
                    {'s':'b','o':'a','sE':'E02','oE':'E01','r':'R01','p':'VISITS'},
                    {'s':'a','o':'b','sE':'E01','oE':'E02','r':'R02','p':'RULES'}]
        self.assertEqual(kg.select(kg.resolve(subject='E01')[0]),[0,2])
        self.assertEqual(kg.select(kg.resolve(subject='E01',relation='R01')[0]),[0])
        self.assertEqual(kg.select(kg.resolve(subject='E01',relation='R01',obj='E02')[0]),[0])



class ReplayTests(unittest.TestCase):
    def test_saved_plan_excludes_entity_followup_choices_and_searches(self):
        req,queries,lookups=saved_plan({'question':'history','gemini':[
            {'kind':'plan_search','response':{'queries':['royal ceremony']}},
            {'kind':'plan_graph','response':{'lookups':[{'relation':'R01'}]}},
            {'kind':'entity_followup','response':{'entities':['e123']}}],
            'searches':[{'query':'unwanted entity','route':'entity keyword: unwanted'}]})
        self.assertEqual(req['q'],'history')
        self.assertEqual(queries,['royal ceremony'])
        self.assertEqual(lookups,[{'relation':'R01'}])

    def test_sentence_baseline_collapses_to_strongest_document_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'baseline.json'
            p.write_text(json.dumps([{'document_id':'v1p0001','grade':'partial'},
                                    {'document_id':'v1p0001','grade':'relevant'}]),encoding='utf-8')
            self.assertEqual(baseline_labels(p),{'v1p0001':2})


class AppContractTests(unittest.TestCase):
    def test_response_contract_with_reading_rounds_and_response_budget(self):
        e=fixture(80)
        e.kg=SimpleNamespace(ok=True,triples=[],ecat_name={},rel_name={},expand_query=lambda q:([],[]))
        sp=MockGemini();sp.available=True;sp.error=None
        spec=importlib.util.spec_from_file_location('test_app_contract',ROOT/'backend/app.py')
        module=importlib.util.module_from_spec(spec)
        with patch('search.Engine',return_value=e),patch('spider.Spider',return_value=sp):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp,patch('dossier.RUN_LOGS',Path(tmp)):
            result=module.run_spider(module.SpiderReq(q='royal evidence',response_tokens=2000))
        self.assertEqual(result['usage']['calls'],6)
        self.assertEqual(sorted(k for k,_ in sp.calls),['answer','assess','assess','assess','assess','plan_search'])
        self.assertEqual(sum(result['counts'].values()),160)
        self.assertEqual(sum(result['counts'][s] for s in ('relevant','partial','dropped','missing')),100)
        hits=result['orders']+result['pages']
        self.assertEqual(len({x['alias'] for x in hits}),160)
        self.assertTrue(all(x['score_kind']=='hybrid_consensus' and x['rerank'] is None for x in hits))
        self.assertNotIn('centroid_applied',result)
        self.assertTrue(all('centroid_score' not in x for x in hits))
        self.assertEqual(e.reranker.calls,[])
        self.assertEqual(result['completed_reading_rounds'],1)
        self.assertEqual(result['reading_rounds'],1)
        self.assertEqual(result['response_tokens'],1000)
        self.assertEqual(sp.requests[-1][2],1000)
        self.assertEqual(module.SpiderReq(q='query').reading_rounds,1)
        for key in ['reading_rounds','recursive_searches']:
            for value in [-1,5,11,1.5,True,'2']:
                with self.assertRaises(ValueError):module.SpiderReq(q='query',**{key:value})
            for value in [0,1,4]:
                self.assertEqual(module.SpiderReq(q='query',**{key:value}).reading_rounds,value)
        self.assertNotIn('response_tokens',module.SpiderReq(q='query',response_tokens=99999).model_dump())

        # cached run: push deeper keeps earlier verdicts, aliases, ranks and the answer; follow-up appends
        self.assertTrue(all('terms' in x for x in hits))
        before={x['alias']:(x['status'],x['rank']) for x in hits}
        d,_=module.RUNS[result['run_id']]
        verdicts=dict(d.verdict)
        with tempfile.TemporaryDirectory() as tmp,patch('dossier.RUN_LOGS',Path(tmp)):
            deep=module.spider_deeper(module.DeeperReq(run_id=result['run_id']))
            self.assertEqual(deep['deeper']['assessed'],50)
            self.assertEqual(len(deep['updated']),50)
            self.assertEqual(deep['remaining'],10)
            self.assertEqual(deep['deeper']['in'],30)  # two judging batches plus the summary, 10 each in the mock
            self.assertTrue(all(before[x['alias']][0]=='not_assessed' and x['rank']==before[x['alias']][1] and x['deeper']==1
                                for x in deep['updated']))
            self.assertTrue(all(d.verdict[k]==v for k,v in verdicts.items()))
            self.assertEqual(d.summary,result['summary'])
            self.assertEqual(sum(deep['counts'][s] for s in ('relevant','partial','dropped','missing')),150)
            fu=module.spider_followup(module.FollowupReq(run_id=result['run_id'],q='what about the queen?'))
            self.assertEqual(fu['followup']['n'],1)
            self.assertEqual(sp.requests[-1][0],'followup')
            self.assertEqual(sp.requests[-1][2],SETTINGS.followup_tokens)
            self.assertIn('what about the queen?',sp.requests[-1][1])
            self.assertIn(result['summary'][:40],sp.requests[-1][1])
            self.assertIn('reading deeper',sp.requests[-1][1])
            module.spider_followup(module.FollowupReq(run_id=result['run_id'],q='and the king?'))
            self.assertIn('Follow-up question: what about the queen?',sp.requests[-1][1])
        with self.assertRaises(module.HTTPException):
            module.spider_deeper(module.DeeperReq(run_id='missing'))

    def test_keyword_narrowing_uses_the_search_stemmer_on_whole_documents(self):
        e=fixture(3)
        kws=e.keywords(['order','keyword','order'],['Orders by keywords'])
        self.assertEqual(kws,[{'stem':'order','word':'orders'},{'stem':'keyword','word':'keywords'}])
        self.assertEqual(e.terms_in(('o',1),['order','dens','keyword','royal']),['order','dens','keyword'])
        self.assertEqual(e.terms_in(('p','p2'),['chronicl','sentenc','order']),['chronicl','sentenc'])
        r=e.search('orders passage')
        self.assertEqual([k['word'] for k in r['keywords']],['orders','passage'])
        self.assertTrue(all(h['terms']==['order','passag'] for h in r['orders']))
        self.assertTrue(all(h['terms']==[] for h in r['pages']))
        self.assertEqual(r['keywords'][0]['n'],3)

    def test_extra_keywords_gather_only_and_skip_duplicates(self):
        e=fixture(30)
        # only order 7's text contains 'seven'; one keyword duplicates a query word, one is empty
        e.orders[7]['en_passages'][1]='order 7 seven passage'
        e.groups['order_en']['text'][15]='order 7 seven passage'
        docs=[en_tokens(t) for t in e.groups['order_en']['text']]
        e.groups['order_en']['bm25']=BM25(docs); e.groups['order_en']['document_bm25']=BM25([docs[2*i]+docs[2*i+1] for i in range(30)])
        e.groups['chron_en']['bm25']=SimpleNamespace(scores=lambda toks:None)
        e.groups['chron_en']['document_bm25']=SimpleNamespace(scores=lambda toks:None)
        sp=MockGemini();sp.plan={'queries':['royal evidence'],'keywords':['seven','Evidence',' ','seven']}
        d=Dossier(e,sp,{'q':'royal evidence','reading_rounds':0})
        d.plan()
        self.assertEqual(d.keywords,['seven'])
        with patch.object(d,'search',wraps=d.search) as searched:
            d.gather()
        self.assertEqual(list(d.retrievals),['royal evidence'])  # keywords never join ranking
        self.assertIn('keywords',d.pool[('o',7)]['routes'])
        self.assertIn('extra keywords',d.rounds[-1]['text'])
        self.assertIn('Extra keywords: seven',d.rounds[0]['text'])
        self.assertIn('keywords',sp.requests[0][1])

    def test_answer_prompt_states_minimum_of_half_the_budget(self):
        d=Dossier(fixture(150),MockGemini(),{'q':'royal evidence','reading_rounds':4,'response_tokens':99999})
        d.run()
        kind,prompt,cap=d.sp.requests[-1]
        self.assertEqual(cap,2500)
        self.assertIn('minimum of 1250 output tokens',prompt)
        self.assertIn('no fewer than 937 words',prompt)



class DateParserTests(unittest.TestCase):
    def test_rebuild_preserves_unaffected_ids_and_maps_split_children(self):
        old=[dict(id='rob9_0010',pdf='a.pdf',part=9,date='1867-06-30',pdf_pages=[1,2],en='First text. 2.July 1867 Second text.'),
             dict(id='rob9_0011',pdf='a.pdf',part=9,date='1867-07-03',pdf_pages=[3],en='Later order.')]
        new=[dict(old[0],en='First text.'),dict(old[0],date='1867-07-02',en='Second text.'),dict(old[1],id='renumbered')]
        mapping=preserve_order_ids(new,old)
        self.assertEqual([r['id'] for r in new],['rob9_0010','rob9_0010_18670702','rob9_0011'])
        self.assertEqual(mapping['rob9_0010'],['rob9_0010','rob9_0010_18670702'])
        again=[dict(r,id='wrong') for r in new]
        preserve_order_ids(again,new)
        self.assertEqual([r['id'] for r in again],[r['id'] for r in new])

    def test_heading_punctuation_brackets_and_clear_exile_year(self):
        for heading,expected in [('2.July 1867','1867-07-02'),('(2 January) 1755','1755-01-02'),
                                 ('19 December (1756)','1756-12-19'),('31 .January 1872','1872-01-31'),
                                 ('25 July 1898','1898-07-25'),('25 July 1898 [8th Day of the Waxing Moon','1898-07-25')]:
            self.assertEqual(parse_date_line(heading,1886,(1853,1885))['date'],expected)
        self.assertTrue(parse_date_line('25 July 1898',1886,(1853,1885))['date_out_of_period'])
        self.assertFalse(parse_date_line('25 July 1898',1886,(1853,1885))['date_uncertain'])

    def test_ocr_ambiguity_flagged_and_prose_not_a_heading(self):
        d=parse_date_line('2k April I79U',1794,(1788,1806))
        self.assertEqual(d['date'],'1794-04-24');self.assertTrue(d['date_uncertain'])
        self.assertEqual(d['date_original'],'2k April I79U')
        for line in ['See also ROB 2 July 1867','This Order was passed on 2 July 1867.',
                     '31 February 1867','19 December 1(378']:
            self.assertIsNone(parse_date_line(line,1867,(1853,1885)))
        self.assertTrue(parse_date_line('20 April 1053',1853,(1853,1885))['date_uncertain'])

    def test_missed_date_splits_two_documents_without_splitting_citation(self):
        entries=segment([(1,'30 June 1867'),(1,'First order. See also ROB 2 July 1867.'),
                         (2,'2.July 1867'),(2,'Eight Member Brahmins conducting Ceremonies.')],(1853,1885))
        self.assertEqual([x['date'] for x in entries],['1867-06-30','1867-07-02'])
        self.assertEqual(entries[0]['lines'],['First order. See also ROB 2 July 1867.'])


if __name__=='__main__':unittest.main(verbosity=2)
