"""Re-split the English passages of data/processed/orders.jsonl with build_orders.english_passages, leaving
every other field unchanged (no PDF re-extraction). Run build_index.py order_en afterwards.
Run:  python scripts/rechunk_orders.py
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_orders", ROOT / "scripts" / "build_orders.py")
bo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bo)

path = ROOT / "data" / "processed" / "orders.jsonl"
orders = [json.loads(l) for l in open(path, encoding="utf-8")]
before = sum(len(o["en_passages"]) for o in orders)
for o in orders:
    o["en_passages"] = bo.english_passages(o.get("en") or " ".join(o["en_passages"]))
after = [len(p) for o in orders for p in o["en_passages"]]
with open(path, "w", encoding="utf-8") as f:
    for o in orders:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"passages {before} -> {len(after)}; under 40 chars: {sum(x < 40 for x in after)}; "
      f"mean {sum(after) // len(after)} chars; longest {max(after)}")
