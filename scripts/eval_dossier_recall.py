"""Recall of the evidence dossier against the graded thematic pool (no Gemini: the question alone is searched).

For each thematic question, relevant = items graded 2 (orders; chronicle sentences counted by their page).
Reports the share of relevant items that are
  old     in the old searchspider per-search depth (top 8 orders + top 6 chronicle pages of the question search)
  pool    anywhere in the new gathered pool (250 candidates per English channel and source)
  short   in the top-N shortlist Gemini would assess (N = 30, 50, 100), ranked by the local cross-encoder
Run:  python scripts/eval_dossier_recall.py [--split all]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from dossier import Dossier  # noqa: E402
from search import Engine, source_normalized  # noqa: E402

TDIR = ROOT / "eval" / "thematic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all")
    args = ap.parse_args()
    eng = Engine()
    grades = {}
    for f in sorted(TDIR.glob("grades_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip().startswith("{"):
                r = json.loads(l)
                grades[r["pid"]] = int(r["grade"])
    pool = {json.loads(l)["pid"]: json.loads(l) for l in open(TDIR / "pool.jsonl", encoding="utf-8")}
    sid_page = {s["id"]: s["page_id"] for s in eng.chron}
    oid = {o["id"]: i for i, o in enumerate(eng.orders)}
    rel = defaultdict(set)
    for pid, g in grades.items():
        if g == 2:
            p = pool[pid]
            rel[p["tid"]].add(("o", oid[p["unit"]]) if p["source"] == "order" else ("p", sid_page[p["unit"]]))
    qs = [json.loads(l) for l in open(TDIR / "questions.jsonl", encoding="utf-8")]
    if args.split != "all":
        qs = [q for q in qs if q["split"] == args.split]
    tot = defaultdict(float)
    rows = []
    for q in qs:
        R = rel[q["tid"]]
        if not R:
            continue
        res = eng.rank(q["question"], rerank=False)
        old = {("o", i) for i in res["orders"][:8]} | {("p", pid) for _, pid, _ in eng.page_rank(res["c_scores"])[:6]}
        d = Dossier(eng, None, {"q": q["question"], "sources": ["order", "chronicle"], "words": 0})
        d.queries, d.lookups = [], []
        d.gather()
        d.score()
        items = source_normalized(d.items())
        keys = [it["key"] for it in items]
        scored = keys  # every candidate has a keyword-consensus score
        row = {"tid": q["tid"], "n_rel": len(R), "old": len(R & old) / len(R), "pool": len(R & set(keys)) / len(R)}
        for n in (30, 50, 100):
            row[f"short{n}"] = len(R & set(scored[:n])) / len(R)
        rows.append(row)
        for k, v in row.items():
            if k not in ("tid",):
                tot[k] += v
        print(json.dumps(row), flush=True)
    n = len(rows)
    print("\nMEAN over", n, "questions:", {k: round(v / n, 3) for k, v in tot.items()})
    json.dump(rows, open(TDIR / "dossier_recall.json", "w"), indent=1)


if __name__ == "__main__":
    main()
