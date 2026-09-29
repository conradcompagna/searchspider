"""Pooled relevance set for thematic questions (TREC-style), English only.

For each question in eval/thematic/questions.jsonl, pools per source (Royal Orders; chronicle sentences):
  app top 5  +  keyword-only top 3  +  meaning-only top 3   (union)
Each pooled item gets a judging text: for an order, its passages that best match the question with one
neighbour each side (capped); for a chronicle sentence, the sentence with the sentences before and after.
Also assigns a dev/test split (20/80) by question.

Out: eval/thematic/pool.jsonl                 all pooled items (+ which setups retrieved them, at what rank)
     eval/thematic/for_chatgpt/batch_0N.jsonl what the grader sees (no setup/rank info)
Run:  python scripts/build_thematic_pool.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from search import Engine  # noqa: E402

TDIR = ROOT / "eval" / "thematic"
POOL_DEPTH = {"app": 5, "bm25": 3, "dense": 3}
SETUPS = {"app": dict(), "bm25": dict(channels=("bm25",), rerank=False), "dense": dict(channels=("dense",), rerank=False)}
MAX_CHARS = 900
N_BATCHES = 2
SEED = 20260926


def order_text(eng, i, qparts):
    o = eng.orders[i]
    ps = o["en_passages"]
    _, best = eng._pinpoint("order", "en", i, qparts)
    idx = sorted({j for b in best[:2] for j in (b - 1, b, b + 1) if 0 <= j < len(ps)}) or [0]
    text, prev = "", None
    for j in idx:
        text += (" ... " if prev is not None and j != prev + 1 else " ") + ps[j]
        prev = j
    text = text.strip()
    if len(text) > MAX_CHARS:
        text = text[:text.rfind(" ", 0, MAX_CHARS)] + " ..."
    if idx[0] > 0:
        text = "... " + text
    return f"Royal order, {o['date']}: {text}"


def chron_text(eng, i):
    s = eng.chron[i]
    page = eng.pages[s["page_id"]]
    k = page.index(i)
    before = eng.chron[page[k - 1]]["en"] if k > 0 else ""
    after = eng.chron[page[k + 1]]["en"] if k + 1 < len(page) else ""
    cut = lambda t, n: t if len(t) <= n else t[:t.rfind(" ", 0, n)] + " ..."
    parts = [f"[before] {cut(before, 250)}" if before else "", f"[PASSAGE] {cut(s['en'], 1500)}",
             f"[after] {cut(after, 250)}" if after else ""]
    return f"Konbaung chronicle, vol. {s['volume']} p. {s['page']}: " + " ".join(p for p in parts if p)


def main():
    rng = random.Random(SEED)
    qs = [json.loads(l) for l in open(TDIR / "questions.jsonl", encoding="utf-8")]
    ids = [q["tid"] for q in qs]
    rng.shuffle(ids)
    dev = set(ids[:round(len(ids) * 0.2)])
    eng = Engine()
    pool = []
    for q in qs:
        q["split"] = "dev" if q["tid"] in dev else "test"
        items = {}
        qparts = None
        for name, cfg in SETUPS.items():
            r = eng.rank(q["question"], **cfg)
            if name == "app":
                qparts = r["qparts"]
            for src, seq in (("order", r["orders"]), ("chronicle", r["sentences"])):
                for rank, i in enumerate(seq[:POOL_DEPTH[name]], 1):
                    key = (src, i)
                    items.setdefault(key, {})[name] = rank
        for (src, i), found in items.items():
            unit = eng.orders[i]["id"] if src == "order" else eng.chron[i]["id"]
            text = order_text(eng, i, qparts) if src == "order" else chron_text(eng, i)
            pool.append({"pid": f"{q['tid']}|{unit}", "tid": q["tid"], "split": q["split"], "question": q["question"],
                         "source": src, "unit": unit, "found_by": found, "text": text})
        print(f"{q['tid']}: {len(items)} pooled", flush=True)

    with open(TDIR / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(TDIR / "pool.jsonl", "w", encoding="utf-8") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # grader batches: grouped by question (consistent grading), no setup/rank info
    gdir = TDIR / "for_chatgpt"
    gdir.mkdir(exist_ok=True)
    per_batch = -(-len(qs) // N_BATCHES)
    for b in range(N_BATCHES):
        tids = {q["tid"] for q in qs[b * per_batch:(b + 1) * per_batch]}
        rows = [p for p in pool if p["tid"] in tids]
        with open(gdir / f"batch_{b + 1:02d}.jsonl", "w", encoding="utf-8") as f:
            for p in rows:
                f.write(json.dumps({"pid": p["pid"], "question": p["question"], "text": p["text"]}, ensure_ascii=False) + "\n")
        print(f"batch_{b + 1:02d}: {len(rows)} pairs")
    print(f"{len(pool)} pairs, {len(qs)} questions ({len(dev)} dev)")


if __name__ == "__main__":
    main()
