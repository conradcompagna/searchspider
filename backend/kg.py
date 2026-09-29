"""Chronicle knowledge graph: 27k triples (data/kg/triples.csv) with resolved entities (data/kg/entities.csv).

Every entity tag in a triple maps to a resolved cluster (parent_id); a cluster's members are its aliases.
Each triple carries the chronicle sentence id it was extracted from, so any edge links back to citeable text.
Royal Orders triples can later be appended to triples.csv with order ids in the sid column.
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

KG = Path(__file__).resolve().parents[1] / "data" / "kg"
from search_settings import SETTINGS


def norm(s):
    return " ".join(re.sub(r"[_\-]", " ", s).split()).lower()


def split_camel(s):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)


class KG:
    def __init__(self, kg_dir=KG, load_graph=True):
        self.dir = kg_dir
        self.vec = None
        self.ok = (kg_dir / "entities.csv").exists() and (not load_graph or (kg_dir / "triples.csv").exists())
        if not self.ok:
            return
        with open(kg_dir / "entities.csv", encoding="utf-8") as f:
            ents = list(csv.DictReader(f))
        self.cluster_of, self.members, self.mentions, self.name = {}, defaultdict(list), Counter(), {}
        for r in ents:
            pid = r["parent_id"]
            self.cluster_of[r["entity"]] = pid
            self.members[pid].append((r["entity"], int(r["mentions"] or 0)))
            self.mentions[pid] += int(r["mentions"] or 0)
            self.name[pid] = r["parent_entity"]
        self.by_norm = defaultdict(set)
        for tag, pid in self.cluster_of.items():
            self.by_norm[norm(tag)].add(pid)
        self.triples = []
        if not load_graph:
            # Runtime search uses this spelling dictionary only. No edges, schema,
            # traversal indexes or graph vectors are loaded by the application.
            self._alias_index()
            return
        self.edges = defaultdict(list)          # cluster -> triple indices
        self.by_rel = defaultdict(list)         # relation code -> triple indices
        self.rel_name, self.ecat_name = {}, {}
        self.ecat = {}                          # cluster -> most frequent entity category
        cat_votes = defaultdict(Counter)
        self.pred_count, self.pred_rel = Counter(), defaultdict(Counter)
        for t in csv.DictReader(open(kg_dir / "triples.csv", encoding="utf-8")):
            s = self._cluster(t["subj"])
            o = self._cluster(t["objs"])
            k = len(self.triples)
            self.pred_count[t["pred"]] += 1
            self.pred_rel[t["pred"]][t["rR"]] += 1
            self.triples.append({"sid": t["sid"], "vol": int(t["vol"].replace("vol", "")), "page": int(t["page"]),
                                 "s": s, "s_tag": t["subj"], "p": t["pred"], "o": o, "o_tag": t["objs"],
                                 "r": t["rR"], "sE": t["sE"], "oE": t["oE"]})
            self.rel_name[t["rR"]] = t["rL"]
            self.ecat_name[t["sE"]] = t["sL"]
            self.ecat_name[t["oE"]] = t["oL"]
            for c, e in ((s, t["sE"]), (o, t["oE"])):
                if c:
                    self.edges[c].append(k)
                    cat_votes[c][e] += 1
            self.by_rel[t["rR"]].append(k)
        self.ecat = {c: v.most_common(1)[0][0] for c, v in cat_votes.items()}
        self.rel_count = Counter(t["r"] for t in self.triples)
        self.ecat_count = Counter()
        for t in self.triples:
            self.ecat_count[t["sE"]] += 1
            self.ecat_count[t["oE"]] += 1
        self._alias_index()
        # every word in every tag of a cluster (entity-family lookup)
        self.cluster_words = {c: {w for tag, _ in m for w in norm(split_camel(tag)).split()} for c, m in self.members.items()}

    def _cluster(self, tag):
        if tag in self.cluster_of:
            return self.cluster_of[tag]
        c = self.by_norm.get(norm(tag))
        return min(c) if c else None

    # -- aliases ---------------------------------------------------------------------------------
    def aliases(self, pid, n=SETTINGS.aliases_per_entity):
        seen, out = set(), []
        for tag, _ in sorted(self.members[pid], key=lambda x: (-x[1], x[0])):
            k = norm(tag)
            if k not in seen:
                seen.add(k)
                out.append(tag.replace("_", " "))
        return out if n is None else out[:n]

    @staticmethod
    def alias_words(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def _alias_index(self):
        """Recognize every spelling, of any length. Expansion is bounded only per entity."""
        self.alias_trie = {}
        for pid, members in self.members.items():
            for tag, _ in members:
                tokens = self.alias_words(tag)
                if not tokens:
                    continue
                node = self.alias_trie
                for token in tokens:
                    node = node.setdefault(token, {})
                node.setdefault(None, set()).add(pid)

    def expand_query(self, text):
        """All matching entities, including overlapping spans; no recursive expansion."""
        if not self.ok:
            return [], []
        toks = self.alias_words(text)
        entities = set()
        for i in range(len(toks)):
            node = self.alias_trie
            for token in toks[i:]:
                if token not in node:
                    break
                node = node[token]
                entities.update(node.get(None, ()))
        found, seen = [], set()
        for pid in sorted(entities, key=lambda p: (-self.mentions[p], p)):
            for alias in self.aliases(pid):
                key = norm(alias)
                if key not in seen:
                    seen.add(key)
                    found.append(alias)
        return found, found[:]

    # -- lookups used by the agent ------------------------------------------------------------------
    def find(self, name, n=5):
        k = norm(name)
        hits = set(self.by_norm.get(k, set()))
        if len(hits) < n:
            words = set(k.split())
            if words:
                for tag_norm, pids in self.by_norm.items():
                    if words <= set(tag_norm.split()):
                        hits |= pids
        ranked = sorted(hits, key=lambda p: -self.mentions[p])[:n]
        return [{"entity": p, "name": self.name[p].replace("_", " "), "aliases": self.aliases(p, 10),
                 "type": split_camel(self.ecat_name.get(self.ecat.get(p), "")), "mentions": self.mentions[p],
                 "edges": len(self.edges[p])} for p in ranked]

    def categories(self):
        c = Counter(t["r"] for t in self.triples)
        return [(r, split_camel(self.rel_name[r]), n) for r, n in c.most_common()]

    @staticmethod
    def _letters(x):
        return re.sub(r"[^a-z0-9]", "", x.lower())

    def _match_cat(self, text, names):
        """Category codes from a code ('R52', 'R52 Punishment...'), an exact name in any casing/spacing
        ('PUNISHMENT_EXECUTION_AND...', 'PunishmentExecution...'), or words all found in the name.
        Several may be given, comma-separated ('R52, R12')."""
        if "," in text:
            out = set()
            for part in text.split(","):
                if part.strip():
                    out |= self._match_cat(part, names)
            return out
        t = text.strip()
        m = re.match(r"([RE]\d{2})\b", t.upper())
        if m and m.group(1) in names:
            return {m.group(1)}
        key = self._letters(t)
        exact = {c for c, nm in names.items() if self._letters(nm) == key}
        if exact:
            return exact
        words = [w for w in norm(split_camel(t)).split() if w not in ("and", "of", "the")]
        return {c for c, nm in names.items() if words and all(w in split_camel(nm).lower() for w in words)}

    def match_relation(self, text):
        return self._match_cat(text, self.rel_name) if text else None

    def match_ecat(self, text):
        return self._match_cat(text, self.ecat_name) if text else None

    def resolve(self, subject=None, relation=None, predicate=None, obj=None, text=None, either=None):
        """Turns the model's filter strings into codes. Returns (filters, notes); notes say how each
        filter was read, or why it matched nothing."""
        f, notes = {}, []
        for key, val in (("subject", subject), ("object", obj), ("either", either)):
            if not val:
                continue
            v = val.strip()
            if re.fullmatch(r"e\d{5}", v):
                f[key] = ("ids", {v})
                notes.append(f"{key} = entity {v} {self.name.get(v, '?')}")
                continue
            cats = self.match_ecat(v)
            if cats:
                f[key] = ("cats", cats)
                notes.append(f"{key} = category {', '.join(sorted(cats))}")
                continue
            ids = set(self.by_norm.get(norm(v), ()))
            if ids:
                f[key] = ("ids", ids)
                notes.append(f"{key} = entity {', '.join(sorted(ids))} (name match)")
                continue
            f[key] = ("ids", set())
            notes.append(f"{key} '{v}' is not an entity id, entity category or exact entity name (use kg_lookup)")
        rel = self.match_relation(relation) if relation else None
        if relation:
            notes.append(f"relation = {', '.join(sorted(rel))}" if rel else f"relation '{relation}' matches no category")
        if predicate:
            p = predicate.strip().upper().replace(" ", "_")
            preds = {p} if p in self.pred_count else {x for x in self.pred_count if p in x}
            if preds:
                f["pred"] = preds
                notes.append(f"predicate = {len(preds)} predicate(s) containing {p}" if p not in self.pred_count else f"predicate = {p}")
            else:
                as_rel = self.match_relation(predicate)
                if as_rel:
                    rel = (rel or set()) | as_rel if rel is None else rel & as_rel
                    notes.append(f"predicate '{predicate}' is a relation category: read as relation {', '.join(sorted(as_rel))}")
                else:
                    f["pred"] = set()
                    notes.append(f"predicate '{predicate}' matches no predicate (use kg_lookup kind=predicate)")
        if rel is not None:
            f["rel"] = rel
        if text:
            f["text"] = text.lower()
            notes.append(f"text contains '{text}'")
        return f, notes

    def fmt(self, t):
        s = t["s_tag"].replace("_", " ")
        o = t["o_tag"].replace("_", " ")
        return f"{s} {t['p']} {o}"

    # -- embeddings over entity tags and predicates (for "what is in the graph near this idea") ------
    def ensure_vectors(self, embedder):
        """Embeds every entity tag and predicate once (cached in data/kg/vectors_*.npz)."""
        import numpy as np
        if not self.ok or self.vec is not None:
            return
        tags = sorted(self.cluster_of)
        preds = sorted(self.pred_count)
        pt, pp = self.dir / "vectors_tags.npz", self.dir / "vectors_preds.npz"  # two files: each under 20 MB
        if pt.exists() and pp.exists():
            zt, zp = np.load(pt), np.load(pp)
            if int(zt["n"]) == len(tags) and int(zp["n"]) == len(preds):
                self.vec = {"tags": tags, "preds": preds, "T": zt["V"].astype(np.float32), "P": zp["V"].astype(np.float32)}
                return
        T = embedder.passages([split_camel(t.replace("_", " ")) for t in tags])
        P = embedder.passages([p.replace("_", " ").lower() for p in preds])
        np.savez_compressed(pt, V=T.astype(np.float16), n=len(tags))
        np.savez_compressed(pp, V=P.astype(np.float16), n=len(preds))
        self.vec = {"tags": tags, "preds": preds, "T": T, "P": P}

    def lookup(self, qvec, text, kind="any", n=8):
        """Nearest entity clusters and/or predicates to a phrase (embeddings), exact alias hits first."""
        import numpy as np
        out = {}
        if kind in ("any", "entity"):
            sims = self.vec["T"] @ qvec
            best = {}
            for i in np.argsort(-sims)[:300]:
                c = self.cluster_of[self.vec["tags"][i]]
                best.setdefault(c, float(sims[i]))
            for c in self.by_norm.get(norm(text), ()):
                best[c] = 1.0
            ranked = sorted(best, key=lambda c: -best[c])[:n]
            out["entities"] = [(c, best[c]) for c in ranked]
        if kind in ("any", "predicate"):
            sims = self.vec["P"] @ qvec
            out["predicates"] = [(self.vec["preds"][i], float(sims[i])) for i in np.argsort(-sims)[:n + 4]]
        return out

    def select(self, f):
        """Triple indices matching resolved filters (from resolve)."""
        def ent_ok(spec, cluster, cat):
            kind, vals = spec
            return (cluster in vals) if kind == "ids" else (cat in vals)
        out = []
        for k, t in enumerate(self.triples):
            if "rel" in f and t["r"] not in f["rel"]:
                continue
            if "pred" in f and t["p"] not in f["pred"]:
                continue
            if "subject" in f and not ent_ok(f["subject"], t["s"], t["sE"]):
                continue
            if "object" in f and not ent_ok(f["object"], t["o"], t["oE"]):
                continue
            if "either" in f and not (ent_ok(f["either"], t["s"], t["sE"]) or ent_ok(f["either"], t["o"], t["oE"])):
                continue
            if "text" in f and f["text"] not in (t["s_tag"] + " " + t["o_tag"] + " " + t["p"]).lower().replace("_", " "):
                continue
            out.append(k)
        return out
