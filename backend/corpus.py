"""Loads the stored texts and defines the passage table shared by the index builder and the server.

A passage is the smallest scored text: one English or Burmese clause of an order, or one chronicle
sentence in one language. Each passage belongs to a unit (an order, or a chronicle sentence).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
INDEX = ROOT / os.getenv("INDEX_DIR", "index")  # one index folder per embedding model

# passage groups -> one vector file each (kept under 20 MB)
GROUPS = ["order_en", "order_my", "chron_en", "chron_my"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_corpus():
    orders = load_jsonl(PROC / "orders.jsonl")
    chron = load_jsonl(PROC / "chronicle_sentences.jsonl")
    return orders, chron


def passage_table(orders, chron):
    """Returns {group: [(unit_kind, unit_index, passage_no, text), ...]} in a fixed order."""
    t = {g: [] for g in GROUPS}
    for i, o in enumerate(orders):
        for j, p in enumerate(o["en_passages"]):
            t["order_en"].append(("order", i, j, p))
        for j, p in enumerate(o["my_passages"]):
            t["order_my"].append(("order", i, j, p))
    for i, s in enumerate(chron):
        t["chron_en"].append(("chron", i, 0, s["en"]))
        t["chron_my"].append(("chron", i, 0, s["my"]))
    return t
