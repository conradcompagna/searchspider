"""Synthetic-question retrieval benchmark. No LLM involved: runs the app's own ranking (Engine.rank).

For each question in eval/questions.jsonl, finds the rank of the passage it was generated from:
  order questions      -> rank of the source order in the Royal Orders list
  chronicle questions  -> rank of the source sentence among sentences, and of its page among pages
Configurations: bm25 (keyword retrieval), dense (e5 retrieval/ranking), app (broad retrieval, keyword ranking).
Metrics: Hit@1/5/10, MRR (reciprocal rank; 0 if not ranked). Headline = test split.

Run:  python scripts/run_retrieval_eval.py [--configs bm25,dense,app] [--split test]
Out:  eval/results/ranks.csv, eval/results/summary.md, eval/results/summary.json
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from search import Engine  # noqa: E402

# Retired Burmese, multi-passage and rank-blending variants are historical results only.
CONFIGS = {
    "bm25": dict(channels=("bm25",), rerank=False),
    "dense": dict(channels=("dense",), rerank=False),
    "hybrid": dict(channels=("dense", "bm25")),  # broad retrieval, keyword ranking (same as app)
    "app": dict(),
    "noalias": dict(expand=False),
}
KS = (1, 5, 10)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def rank_of(seq, target):
    for n, x in enumerate(seq, 1):
        if x == target:
            return n
    return None  # not ranked at all (no channel scored it)


def metrics(ranks):
    n = len(ranks)
    if not n:
        return {}
    m = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    m["mrr"] = sum(1 / r for r in ranks if r) / n
    m["n"] = n
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--split", default="all", help="test | dev | all (ranks are always recorded for all)")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--lang", default="all", help="en | my | all")
    ap.add_argument("--out", default="results", help="output folder under eval/")
    ap.add_argument("--questions", default="questions.jsonl", help="question file under eval/")
    args = ap.parse_args()
    configs = args.configs.split(",")

    src = {r["eid"]: r for r in load(ROOT / "eval" / "source_passages.jsonl")}
    qs = [q for q in load(ROOT / "eval" / args.questions) if q.get("question")]
    if args.split != "all":
        qs = [q for q in qs if src[q["eid"]]["split"] == args.split]
    if args.lang != "all":
        qs = [q for q in qs if src[q["eid"]]["q_lang"] == args.lang]
    if args.limit:
        qs = qs[:args.limit]

    eng = Engine()
    order_idx = {o["id"]: i for i, o in enumerate(eng.orders)}
    sent_idx = {s["id"]: i for i, s in enumerate(eng.chron)}
    out_dir = ROOT / "eval" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in configs:
        t0, rr_ms = time.time(), 0.0
        for n, q in enumerate(qs, 1):
            s = src[q["eid"]]
            r = eng.rank(q["question"], **CONFIGS[cfg])
            rr_ms += r["rr_ms"]
            row = {"eid": q["eid"], "config": cfg, "split": s["split"], "source": s["source"], "q_lang": s["q_lang"]}
            if s["source"] == "order":
                row["rank"] = rank_of(r["orders"], order_idx[s["target_id"]])
            else:
                row["rank"] = rank_of(r["sentences"], sent_idx[s["target_id"]])
                pages = [pid for _, pid, _ in eng.page_rank(r["c_scores"])]
                row["page_rank"] = rank_of(pages, s["page_id"])
            rows.append(row)
            if n % 100 == 0 or n == len(qs):
                print(f"  {cfg}: {n}/{len(qs)}  {time.time() - t0:.0f}s", flush=True)
        print(f"{cfg}: {len(qs) / (time.time() - t0):.1f} q/s, rerank {rr_ms / len(qs):.0f} ms/q", flush=True)

    with open(out_dir / "ranks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eid", "config", "split", "source", "q_lang", "rank", "page_rank"])
        w.writeheader()
        w.writerows(rows)

    # --- summary: test split (headline), dev split, and breakdowns
    summary = defaultdict(dict)
    for split in ("test", "dev"):
        for cfg in configs:
            sel = [r for r in rows if r["config"] == cfg and r["split"] == split]
            groups = {
                "all": sel,
                "orders": [r for r in sel if r["source"] == "order"],
                "chronicle (sentence)": [r for r in sel if r["source"] == "chronicle"],
                "english questions": [r for r in sel if r["q_lang"] == "en"],
                "burmese questions": [r for r in sel if r["q_lang"] == "my"],
            }
            for g, rs in groups.items():
                summary[split][(cfg, g)] = metrics([r["rank"] for r in rs])
            summary[split][(cfg, "chronicle (page)")] = metrics(
                [r["page_rank"] for r in sel if r["source"] == "chronicle"])

    lines = ["# Retrieval benchmark (synthetic questions)", "",
             f"{len(qs)} questions with text; source passages sampled by `scripts/build_eval_set.py`.",
             "Rank = position of the passage the question was generated from. MRR counts unranked as 0.", ""]
    for split in ("test", "dev"):
        groups = ["all", "orders", "chronicle (sentence)", "chronicle (page)", "english questions", "burmese questions"]
        for g in groups:
            if not any(summary[split].get((c, g)) for c in configs):
                continue
            n = next(summary[split][(c, g)]["n"] for c in configs if summary[split].get((c, g)))
            lines += [f"## {split} · {g} (n={n})", "", "| config | Hit@1 | Hit@5 | Hit@10 | MRR |", "|---|---:|---:|---:|---:|"]
            for c in configs:
                m = summary[split].get((c, g))
                if m:
                    lines.append(f"| {c} | {m['hit@1']:.3f} | {m['hit@5']:.3f} | {m['hit@10']:.3f} | {m['mrr']:.3f} |")
            lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump({s: {f"{c}|{g}": m for (c, g), m in d.items()} for s, d in summary.items()},
              open(out_dir / "summary.json", "w"), indent=1)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
