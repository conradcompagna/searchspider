"""Write eval/FINAL_SCORES.md from the final benchmark runs (test split, app defaults).

Known-item (synthetic questions, one source passage each): eval/final_known_item/ranks.csv
Thematic (graded relevance, English questions): eval/thematic/final_thematic/per_question.csv
Run after:  python scripts/run_retrieval_eval.py --configs bm25,dense,hybrid,app --split test --out final_known_item
            python scripts/score_thematic.py --split test --configs app,bm25,dense,hybrid --out final_thematic
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"app": "**App (hybrid + reranker + Burmese routing)**", "hybrid": "Hybrid, no reranker",
         "bm25": "Keyword only (BM25)", "dense": "Meaning only (e5)"}
ORDER = ["app", "hybrid", "bm25", "dense"]


def known_item():
    rows = list(csv.DictReader(open(ROOT / "eval" / "final_known_item" / "ranks.csv", encoding="utf-8")))

    def m(ranks):
        n = len(ranks)
        r = [int(x) if x else None for x in ranks]
        return n, sum(1 for x in r if x and x <= 1) / n, sum(1 for x in r if x and x <= 5) / n, \
            sum(1 for x in r if x and x <= 10) / n, sum(1 / x for x in r if x) / n

    out = []
    for lang, label in (("en", "English questions (reranker applies)"), ("my", "Burmese questions (no reranker: keyword routing)"),
                        (None, "All questions")):
        sel = [r for r in rows if lang is None or r["q_lang"] == lang]
        for grp, gl, key in (("all", "all sources", "rank"), ("order", "Royal Orders", "rank"),
                             ("chronicle", "chronicle sentence", "rank"), ("chronicle", "chronicle page", "page_rank")):
            g = [r for r in sel if grp == "all" or r["source"] == grp]
            if not g:
                continue
            n = len({r["eid"] for r in g})
            out += [f"#### {label} · {gl} (n={n})", "", "| setup | Hit@1 | Hit@5 | Hit@10 | MRR |", "|---|---:|---:|---:|---:|"]
            for c in ORDER:
                rr = [r[key] for r in g if r["config"] == c]
                if rr:
                    _, h1, h5, h10, mrr = m(rr)
                    out.append(f"| {NAMES[c]} | {h1:.3f} | {h5:.3f} | {h10:.3f} | {mrr:.3f} |")
            out.append("")
    return out


def thematic():
    rows = list(csv.DictReader(open(ROOT / "eval" / "thematic" / "final_thematic" / "per_question.csv", encoding="utf-8")))
    cols = [("p@5", "P@5"), ("ndcg@5", "nDCG@5"), ("ndcg@10", "nDCG@10"), ("recall@10", "Recall@10"), ("mrr", "MRR"),
            ("judged@5", "judged@5")]
    out = []
    for src, label in (("order", "Royal Orders"), ("chronicle", "Chronicle sentences")):
        g = [r for r in rows if r["source"] == src]
        n = len({r["tid"] for r in g})
        out += [f"#### {label} (n={n} questions)", "", "| setup | " + " | ".join(c[1] for c in cols) + " |",
                "|---" + "|---:" * len(cols) + "|"]
        for c in ORDER:
            rr = [r for r in g if r["config"] == c]
            if rr:
                out.append(f"| {NAMES[c]} | " + " | ".join(f"{sum(float(r[k]) for r in rr) / len(rr):.3f}" for k, _ in cols) + " |")
        out.append("")
    return out


def main():
    lines = ["# Final retrieval scores", "",
             "App defaults as shipped: multilingual-e5-small, BM25, reciprocal-rank fusion, mMiniLM reranker (English only), "
             "Burmese queries routed to Burmese keyword search, name variants from the entity-resolution tables. "
             "Test split only; settings were chosen on the dev split.", "",
             "## 1. Known-item benchmark (synthetic questions)", "",
             "Each question was written from one source passage; the score is the rank of that passage. Questions are "
             "rich in names, which favours keyword search. Only the source passage counts, so relevant near-duplicates "
             "score as misses: figures are lower bounds. Burmese questions are listed separately because the reranker reads "
             "English only; they use Burmese keyword search instead.", ""] + known_item() + [
             "## 2. Thematic benchmark (graded relevance, English)", "",
             "48 topical questions. The top results of every setup tested (1,958 question-passage pairs) were graded "
             "2 relevant / 1 partial / 0 not by ChatGPT; Claude's blind grades of 200 pairs agree at weighted kappa 0.75. "
             "P@5 counts partial as half. Questions with fewer than 2 relevant items in a source are skipped for that source. "
             "judged@5 is the share of the top 5 that has a grade (unjudged count as not relevant).", ""] + thematic()
    (ROOT / "eval" / "FINAL_SCORES.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
