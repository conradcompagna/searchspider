"""Add the ungraded items that new setups retrieve to the thematic pool, as a new ChatGPT batch.

For every thematic question and each named config (from run_retrieval_eval.CONFIGS), takes the top
DEPTH orders and chronicle sentences; any (question, unit) pair not already in pool.jsonl is appended
to pool.jsonl and written to for_chatgpt/batch_NN.jsonl (same format and prompt as before).
Run:  python scripts/extend_thematic_pool.py --configs topic,rrtopics [--depth 5] [--batch 03]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_thematic_pool import TDIR, chron_text, order_text  # noqa: E402
from run_retrieval_eval import CONFIGS  # noqa: E402
from search import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--batch", default="03")
    args = ap.parse_args()
    pool = [json.loads(l) for l in open(TDIR / "pool.jsonl", encoding="utf-8")]
    have = {p["pid"] for p in pool}
    qs = [json.loads(l) for l in open(TDIR / "questions.jsonl", encoding="utf-8")]
    eng = Engine()
    new = {}
    for q in qs:
        qparts = eng.rank(q["question"], rerank=False, channels=("dense",))["qparts"]
        for cfg in args.configs.split(","):
            r = eng.rank(q["question"], **CONFIGS[cfg])
            for src, seq in (("order", r["orders"]), ("chronicle", r["sentences"])):
                for rank, i in enumerate(seq[:args.depth], 1):
                    unit = eng.orders[i]["id"] if src == "order" else eng.chron[i]["id"]
                    pid = f"{q['tid']}|{unit}"
                    if pid in have:
                        continue
                    if pid not in new:
                        text = order_text(eng, i, qparts) if src == "order" else chron_text(eng, i)
                        new[pid] = {"pid": pid, "tid": q["tid"], "split": q["split"], "question": q["question"],
                                    "source": src, "unit": unit, "found_by": {}, "text": text}
                    new[pid]["found_by"][cfg] = rank
        print(f"{q['tid']}: {sum(1 for p in new.values() if p['tid'] == q['tid'])} new", flush=True)
    with open(TDIR / "pool.jsonl", "a", encoding="utf-8") as f:
        for p in new.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    out = TDIR / "for_chatgpt" / f"batch_{args.batch}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in sorted(new.values(), key=lambda p: p["tid"]):
            f.write(json.dumps({"pid": p["pid"], "question": p["question"], "text": p["text"]}, ensure_ascii=False) + "\n")
    print(f"{len(new)} new pairs -> {out}")


if __name__ == "__main__":
    main()
