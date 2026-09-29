"""Score thematic questions against pooled relevance grades (ChatGPT-graded, Claude-checked).

Reads:  eval/thematic/grades_*.jsonl   ChatGPT output, lines {"pid": "Txx|unit", "grade": 0|1|2}
        eval/thematic/claude_grades.json  Claude's blind grades on 200 pairs (agreement check)
Checks: missing / unknown pids, bad grades, grade distribution per batch, agreement with Claude
        (exact, relevant-vs-not, Cohen's kappa, linear-weighted kappa).
Scores each config per source (orders; chronicle sentences) at depth 5 and 10:
        nDCG (gain = grade), P@k (grade 2 = relevant; 1 counts as 0.5), recall of pooled relevant,
        MRR of first grade-2 hit, judged@k (share of top k that has a grade; unjudged count as 0).
Questions with fewer than 2 grade-2 items in a source are skipped for that source.
Run:    python scripts/score_thematic.py [--configs app,bm25,dense,hybrid] [--split test]
Out:    eval/thematic/results/summary.md, per_question.csv
"""
import argparse
import csv
import json
import math
import os
import time
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
TDIR = ROOT / "eval" / "thematic"
KS = (5, 10)


def load_grades():
    pool = {json.loads(l)["pid"]: json.loads(l) for l in open(TDIR / "pool.jsonl", encoding="utf-8")}
    grades, bad, unknown = {}, 0, []
    files = sorted(TDIR.glob("grades_*.jsonl"))
    if not files:
        sys.exit("no eval/thematic/grades_*.jsonl files yet")
    for f in files:
        dist = Counter()
        for l in open(f, encoding="utf-8"):
            l = l.strip().strip(",")
            if not l.startswith("{"):
                continue
            try:
                r = json.loads(l)
                g = int(r["grade"])
                assert g in (0, 1, 2)
            except Exception:
                bad += 1
                continue
            if r["pid"] not in pool:
                unknown.append(r["pid"])
                continue
            grades[r["pid"]] = g
            dist[g] += 1
        print(f"{f.name}: {sum(dist.values())} grades  {dict(sorted(dist.items()))}")
    missing = [p for p in pool if p not in grades]
    print(f"pool {len(pool)} · graded {len(grades)} · missing {len(missing)} · unknown {len(unknown)} · unparseable {bad}")
    if missing:
        (TDIR / "missing_pids.txt").write_text("\n".join(missing), encoding="utf-8")
        print(f"  missing pids written to eval/thematic/missing_pids.txt")
    return pool, grades


def kappa(a, b, weighted=False):
    cats = (0, 1, 2)
    n = len(a)
    w = (lambda i, j: abs(i - j) / 2) if weighted else (lambda i, j: float(i != j))
    obs = sum(w(x, y) for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    exp = sum(pa[i] * pb[j] * w(i, j) for i in cats for j in cats) / n / n
    return 1 - obs / exp if exp else float("nan")


def agreement(grades):
    p = TDIR / "claude_grades.json"
    if not p.exists():
        return ""
    mine = json.load(open(p))
    both = [k for k in mine if k in grades]
    a, b = [mine[k] for k in both], [grades[k] for k in both]
    n = len(both)
    exact = sum(x == y for x, y in zip(a, b)) / n
    bin_ = sum((x == 2) == (y == 2) for x, y in zip(a, b)) / n
    conf = Counter(zip(a, b))
    lines = [f"Agreement with Claude's blind grades (n={n}): exact {exact:.2f}, relevant-vs-not {bin_:.2f}, "
             f"kappa {kappa(a, b):.2f}, weighted kappa {kappa(a, b, True):.2f}",
             "Confusion (rows Claude, cols ChatGPT): " +
             "; ".join(f"{i}: " + " ".join(str(conf[(i, j)]) for j in (0, 1, 2)) for i in (0, 1, 2))]
    print("\n".join(lines))
    return "\n".join(lines)


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def main():
    from run_retrieval_eval import CONFIGS
    from search import Engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="app,bm25,dense,hybrid")
    ap.add_argument("--split", default="test", help="test | dev | all")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    pool, grades = load_grades()
    agree = agreement(grades)
    qs = [json.loads(l) for l in open(TDIR / "questions.jsonl", encoding="utf-8")]
    if args.split != "all":
        qs = [q for q in qs if q["split"] == args.split]

    eng = Engine()
    unit = {"order": lambda i: eng.orders[i]["id"], "chronicle": lambda i: eng.chron[i]["id"]}
    rel = defaultdict(dict)  # (tid, source) -> {unit: grade}
    for pid, g in grades.items():
        rel[(pool[pid]["tid"], pool[pid]["source"])][pool[pid]["unit"]] = g

    rows, skipped = [], Counter()
    for cfg in args.configs.split(","):
        for q in qs:
            t0 = time.time()
            r = eng.rank(q["question"], **CONFIGS[cfg])
            ms = (time.time() - t0) * 1000
            for src, seq in (("order", r["orders"]), ("chronicle", r["sentences"])):
                judged = rel[(q["tid"], src)]
                n_rel = sum(1 for g in judged.values() if g == 2)
                if n_rel < 2:
                    skipped[src] += cfg == args.configs.split(",")[0]
                    continue
                ids = [unit[src](i) for i in seq[:max(KS)]]
                gains = [judged.get(u, 0) for u in ids]
                ideal = sorted(judged.values(), reverse=True)
                row = {"config": cfg, "tid": q["tid"], "source": src, "n_rel": n_rel, "ms": ms}
                for k in KS:
                    row[f"ndcg@{k}"] = dcg(gains[:k]) / dcg(ideal[:k])
                    row[f"p@{k}"] = sum({2: 1, 1: .5}.get(g, 0) for g in gains[:k]) / k
                    row[f"recall@{k}"] = sum(g == 2 for g in gains[:k]) / n_rel
                    row[f"judged@{k}"] = sum(u in judged for u in ids[:k]) / k
                # condensed list (unjudged items dropped): less biased against setups that surface new items
                cg = [judged[u] for u in [unit[src](i) for i in seq[:200]] if u in judged]
                row["ndcg@5c"] = dcg(cg[:5]) / dcg(ideal[:5])
                row["p@5c"] = sum({2: 1, 1: .5}.get(g, 0) for g in cg[:5]) / 5
                first = next((n for n, g in enumerate(gains, 1) if g == 2), None)
                row["mrr"] = 1 / first if first else 0
                rows.append(row)

    out = TDIR / args.out
    out.mkdir(exist_ok=True)
    with open(out / "per_question.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = ["ndcg@5", "p@5", "judged@5", "ndcg@5c", "p@5c", "ndcg@10", "recall@10", "mrr", "judged@10", "ms"]
    lines = ["# Thematic retrieval (pooled graded relevance)", "",
             f"Models: embedder {os.getenv('EMBED_MODEL', 'multilingual-e5-small')}, "
             f"reranker {os.getenv('RERANK_MODEL', 'mmarco-mMiniLMv2-L12')}. ms = search time per question.", "",
             f"Split: {args.split}. Questions per source need ≥2 fully relevant items; skipped: "
             f"orders {skipped['order']}, chronicle {skipped['chronicle']}.", "", agree, ""]
    for src in ("order", "chronicle"):
        sel = [r for r in rows if r["source"] == src]
        n = len({r["tid"] for r in sel})
        lines += [f"## {'Royal Orders' if src == 'order' else 'Chronicle sentences'} (n={n})", "",
                  "| config | " + " | ".join(cols) + " |", "|---" + "|---:" * len(cols) + "|"]
        for cfg in args.configs.split(","):
            rs = [r for r in sel if r["config"] == cfg]
            if rs:
                lines.append(f"| {cfg} | " + " | ".join(f"{sum(r[c] for r in rs) / len(rs):{'.0f' if c == 'ms' else '.3f'}}" for c in cols) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
