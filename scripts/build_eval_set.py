"""Sample source passages for a synthetic-question retrieval benchmark (no hand labels).

Each sampled passage gets one generated question; the benchmark then checks whether the system
retrieves the unit the question came from (Hit@1/5/10, MRR).

Sample: 500 Royal Orders windows (one per order, stratified by Part) + 500 chronicle sentences
(one per page, stratified by volume). About 10% are flagged for a Burmese-language question.
Split: 20% "dev" (may be used for tuning), 80% "test" (never used for tuning).

Run:  python scripts/build_eval_set.py      -> eval/source_passages.jsonl, eval/for_chatgpt/*.jsonl
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval"
SEED = 20260925
N_ORDERS, N_CHRON = 500, 500
MY_SHARE = 0.10          # share of items flagged for a Burmese-language question
DEV_SHARE = 0.20
BATCH = 250

BOILER = re.compile(r"^(this order (was|is) (passed|proclaimed)|proclaimed by|note\s*:|order\s*:?\s*$|see also|\(?rob\b)", re.I)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def clean_enough(text):
    """Reject OCR debris, boilerplate, bare lists: need real prose to write a question from."""
    t = text.strip()
    if BOILER.match(t):
        return False
    if re.search(r"(\b\w ){4,}", t):                       # spaced-out OCR: "W e s t P a l a c e"
        return False
    words = re.findall(r"[A-Za-z]+", t)
    if len(words) < 12:
        return False
    if sum(len(w) == 1 for w in words) / len(words) > 0.15:  # letter salad
        return False
    if len(re.findall(r"[^\x00-\x7F–—‘’“”]", t)) > 3:  # stray glyphs
        return False
    lower = [w for w in words if w.islower() and len(w) > 2]
    return len(lower) >= 8                                  # not just a list of names


def norm(t):
    return re.sub(r"[^a-z]", "", t.lower())[:80]


def order_window(o, rng):
    """1-3 adjacent clean passages, 150-600 chars. Returns (start, end, text) or None."""
    ps = o["en_passages"]
    starts = [i for i, p in enumerate(ps) if clean_enough(p)]
    rng.shuffle(starts)
    for i in starts:
        j, text = i, ps[i]
        while len(text) < 150 and j + 1 < len(ps) and clean_enough(ps[j + 1]):
            j += 1
            text += " " + ps[j]
        if 150 <= len(text) <= 600:
            return i, j, text
        if len(text) > 600 and j == i:
            continue
    return None


def stratified(groups, n, rng):
    """Proportional allocation with a floor of 20 per group; draws without replacement."""
    total = sum(len(v) for v in groups.values())
    alloc = {k: max(20, round(n * len(v) / total)) for k, v in groups.items()}
    while sum(alloc.values()) > n:  # trim the largest groups back to n
        k = max(alloc, key=lambda k: alloc[k])
        alloc[k] -= 1
    picked = []
    for k, v in sorted(groups.items()):
        rng.shuffle(v)
        picked.extend(v[:alloc[k]])
    return picked


def main():
    rng = random.Random(SEED)
    orders = load(ROOT / "data/processed/orders.jsonl")
    chron = load(ROOT / "data/processed/chronicle_sentences.jsonl")
    seen = set()

    # --- Royal Orders: one window per order, stratified by Part
    by_part = defaultdict(list)
    for o in orders:
        w = order_window(o, rng)
        if w and norm(w[2]) not in seen:
            seen.add(norm(w[2]))
            by_part[o["part"]].append((o, w))
    items = []
    for o, (i, j, text) in stratified(by_part, N_ORDERS, rng):
        items.append({
            "source": "order", "target_id": o["id"], "passages": list(range(i, j + 1)),
            "label": f"Royal Order, {o['date']} (Than Tun, ROB Part {o['part']})",
            "part": o["part"], "year": o["year"], "text_en": text,
            "text_my": " ".join(o["my_passages"]) if o["my_alignment"] == "paired" else "",
        })

    # --- Chronicle: one sentence per page, stratified by volume
    by_page = defaultdict(list)
    for s in chron:
        if 80 <= len(s["en"]) <= 700 and clean_enough(s["en"]) and norm(s["en"]) not in seen:
            by_page[s["page_id"]].append(s)
    by_vol = defaultdict(list)
    for pid, ss in by_page.items():
        s = rng.choice(ss)
        seen.add(norm(s["en"]))
        by_vol[s["volume"]].append(s)
    for s in stratified(by_vol, N_CHRON, rng):
        items.append({
            "source": "chronicle", "target_id": s["id"], "page_id": s["page_id"],
            "label": f"Konbaungset Yazawin, vol. {s['volume']}, p. {s['page']}",
            "volume": s["volume"], "text_en": s["en"], "text_my": s["my"],
        })

    # --- Burmese-question flags (only where Burmese text exists), dev/test split, ids
    rng.shuffle(items)
    for src in ("order", "chronicle"):
        pool = [it for it in items if it["source"] == src and it["text_my"]]
        for it in pool[:round(len(items) * MY_SHARE / 2)]:
            it["q_lang"] = "my"
    for it in items:
        it.setdefault("q_lang", "en")
    for src in ("order", "chronicle"):
        for lang in ("en", "my"):  # split within each source x language group
            group = [it for it in items if it["source"] == src and it["q_lang"] == lang]
            rng.shuffle(group)
            for k, it in enumerate(group):
                it["split"] = "dev" if k < round(len(group) * DEV_SHARE) else "test"
    for n, it in enumerate(items, 1):
        it["eid"] = f"E{n:04d}"

    OUT.mkdir(exist_ok=True)
    keys = ["eid", "split", "source", "target_id", "page_id", "passages", "part", "volume", "year",
            "label", "q_lang", "text_en", "text_my"]
    with open(OUT / "source_passages.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({k: it[k] for k in keys if k in it}, ensure_ascii=False) + "\n")

    # --- what ChatGPT sees: no target ids, no split (it only needs the passage)
    gdir = OUT / "for_chatgpt"
    gdir.mkdir(exist_ok=True)
    for b in range(0, len(items), BATCH):
        with open(gdir / f"batch_{b // BATCH + 1:02d}.jsonl", "w", encoding="utf-8") as f:
            for it in items[b:b + BATCH]:
                row = {"eid": it["eid"], "q_lang": it["q_lang"], "source": it["label"], "text": it["text_en"]}
                if it["q_lang"] == "my":
                    row["text_my"] = it["text_my"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    count = defaultdict(int)
    for it in items:
        count[(it["source"], it["split"], it["q_lang"])] += 1
    print(len(items), "items")
    for k in sorted(count):
        print(" ", k, count[k])
    print("orders by part:", dict(sorted(defaultdict(int, {p: sum(1 for i in items if i.get("part") == p) for p in range(3, 10)}).items())))
    print("chronicle by vol:", {v: sum(1 for i in items if i.get("volume") == v) for v in (1, 2, 3)})


if __name__ == "__main__":
    main()
