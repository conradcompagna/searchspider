"""Copy the Konbaungset Yazawin sentence corpus (Burmese + English) from dighumproject into
retrievable form.

Input : dighumproject/konbaung_sentence_translation_20260718_restoration_integrated/translations/all_volumes.jsonl
Output: data/processed/chronicle_sentences.jsonl  (id, volume, page, pages, my, en, seq)
        data/processed/chronicle_pages.jsonl      (page_id, volume, page, sentence ids)
Run:  python scripts/build_chronicle.py [path/to/all_volumes.jsonl]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "raw" / "all_volumes.jsonl"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: (r["volume"], r["id"]))
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    pages = defaultdict(list)
    with open(out / "chronicle_sentences.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            page_id = f"v{r['volume']}p{r['owner_page']:04d}"
            pages[(r["volume"], r["owner_page"])].append(r["id"])
            f.write(json.dumps({"id": r["id"], "source": "chronicle", "volume": r["volume"],
                                "page": r["owner_page"], "page_id": page_id, "pages": r["pages"],
                                "my": r["my"], "en": r["en"], "seq": i}, ensure_ascii=False) + "\n")
    with open(out / "chronicle_pages.jsonl", "w", encoding="utf-8") as f:
        for (vol, pg), ids in sorted(pages.items()):
            f.write(json.dumps({"page_id": f"v{vol}p{pg:04d}", "volume": vol, "page": pg, "sentences": ids}) + "\n")
    print(f"{len(rows)} sentences, {len(pages)} pages -> {out}")


if __name__ == "__main__":
    main()
