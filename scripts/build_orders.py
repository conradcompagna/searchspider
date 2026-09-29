"""Build retrievable Royal Orders records from Than Tun, The Royal Orders of Burma, Parts 3-9.

Output (data/processed/):
  orders.jsonl   one record per dated order: English text (all parts), Burmese (Parts 3-4),
                 clause-level passages for pinpointing, Than Tun's index topics (Part 10).
Run:  python scripts/build_orders.py
"""
import json
import hashlib
import re
import sys
from datetime import date as calendar_date
from collections import defaultdict
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "backend"))
from chunking import merge_short_passages
from search_settings import SETTINGS

PARTS = {  # part: (file, english page ranges (1-based, inclusive) or None = all, burmese ranges)
    3: ("Than_Tun-1985-Royal_Orders_of_Burma-03-bu+en-red.pdf", [(2, 122)], [(124, 323)]),
    4: ("Than_Tun-1986-Royal_Orders_of_Burma-04-bu+en-red.pdf", [(110, 316)], [(318, 793)]),
    5: ("Than_Tun-1986-Royal_Orders_of_Burma-05-en-ocr-to.pdf", None, []),
    6: ("Than_Tun-1987-Royal_Orders_of_Burma-06-en-ocr-col-tu-corr.pdf", None, []),
    7: ("Than_Tun-1988-Royal_Orders_of_Burma-07-en-ocr-to.pdf", None, []),
    8: ("Than_Tun-1988-Royal_Orders_of_Burma-08-en-ocr-tu.pdf", None, []),
    9: ("Than_Tun-1989-Royal_Orders_of_Burma-09-ocr-to-en.pdf", None, []),
}
INDEX_PDF = "Than_Tun-1990-Royal_Orders_of_Burma-10-ocr-tu-corr.pdf"

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}
OCR_DIGIT = str.maketrans({"I": "1", "l": "1", "i": "1", "O": "0", "o": "0", "S": "5", "B": "8"})

# a line that is only a date, optionally in parentheses: "23   March 17 88", "(4 March 1755)"
DATE_LINE = re.compile(r"^\s*([0-9IlkK]{1,2})[.\s]+([A-Za-z]{3,10})[.\s]+([0-9IlOoSB^Uu ]{4,8})\s*\.?$")
PAGE_NUM = re.compile(r"^[-—\s]*[0-9ivxlc]{1,4}[-—\s]*$", re.I)


def month_num(word):
    w = word.lower()
    if w in MONTHS:
        return MONTHS.index(w) + 1
    for i, m in enumerate(MONTHS):  # OCR-tolerant: same first 3 letters and similar length
        if w[:3] == m[:3] and abs(len(w) - len(m)) <= 2:
            return i + 1
    return None


PART_YEARS = {3: (1751, 1781), 4: (1782, 1787), 5: (1788, 1806), 6: (1807, 1810),
              7: (1811, 1819), 8: (1819, 1853), 9: (1853, 1885)}


def parse_date_line(line, prev_year, bounds):
    # Brackets can enclose the entire heading or just the inferred day/month.
    # Still require a standalone heading: dates cited inside prose are not boundaries.
    # Some headings append the equivalent Burmese calendar date on the same line.
    heading = re.split(r"\s+\[\d+(?:st|nd|rd|th)?\s+Day\b", line.strip(), maxsplit=1, flags=re.I)[0]
    if heading.count('(') != heading.count(')') or heading.count('[') != heading.count(']'):
        return None  # A stray OCR bracket inside a year is not an inferred date.
    supplied = any(c in heading for c in '([')
    heading = re.sub(r"[()\[\]]", " ", heading)
    m = DATE_LINE.match(heading)
    if not m:
        return None
    day_s, mon_s, yr_s = m.groups()
    mon = month_num(mon_s)
    if not mon:
        return None
    day = int(day_s.translate(OCR_DIGIT).replace('k', '4').replace('K', '4'))
    yr_raw = yr_s.replace(" ", "").translate(OCR_DIGIT)
    uncertain = not (day_s.isdigit() and yr_s.replace(' ', '').isdigit())
    year = int(yr_raw) if yr_raw.isdigit() else None
    lo, hi = bounds
    if 'U' in yr_raw or 'u' in yr_raw:
        candidates = {int(yr_raw.upper().replace('U', digit)) for digit in ('4', '8')
                      if yr_raw.upper().replace('U', digit).isdigit()}
        plausible = [y for y in candidates if lo - 1 <= y <= hi + 1]
        year = plausible[0] if len(plausible) == 1 else None
    if year and str(year).startswith('10') and lo - 1 <= int('18' + str(year)[2:]) <= hi + 1:
        year, uncertain = int('18' + str(year)[2:]), True  # known 0/8 OCR confusion; explicitly flagged
    if year is None:
        year, uncertain = prev_year, True
    # A clearly printed year remains evidence even outside this volume's usual era.
    # Never silently turn an exile-era letter into an earlier royal order.
    try:
        calendar_date(year, mon, day)
    except (ValueError, TypeError):
        return None
    return {
        "date": f"{year:04d}-{mon:02d}-{day:02d}",
        "year": year,
        "date_supplied": supplied,
        "date_uncertain": uncertain,
        "date_original": line.strip(),
        "date_out_of_period": not (lo - 1 <= year <= hi + 1),
    }


def page_ranges(doc, ranges):
    if ranges is None:
        return range(1, len(doc) + 1)
    return [p for a, b in ranges for p in range(a, b + 1)]


def visual_lines(page, tol=3.0):
    """Rebuild printed lines from word boxes (OCR text layers often split one line into several)."""
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines, cur, cur_y = [], [], None
    for w in words:
        y = (w[1] + w[3]) / 2
        if cur and abs(y - cur_y) > tol:
            lines.append(cur)
            cur = []
        if not cur:
            cur_y = y
        cur.append(w)
    if cur:
        lines.append(cur)
    return [" ".join(w[4] for w in sorted(l, key=lambda w: w[0])) for l in lines]


def english_lines(doc, pages):
    for p in pages:
        for raw in visual_lines(doc[p - 1]):
            line = raw.strip()
            if not line or PAGE_NUM.match(line):
                continue
            yield p, line


def burmese_lines(doc, pages):
    from wininnwa_convert import wininnwa_to_unicode
    for p in pages:
        spans = []
        for b in doc[p - 1].get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    if s["text"].strip():
                        t = wininnwa_to_unicode(s["text"]) if "Innwa" in s["font"] else s["text"]
                        spans.append(((s["bbox"][1] + s["bbox"][3]) / 2, s["bbox"][0], t))
        spans.sort()
        rows, cur, cur_y = [], [], None  # regroup spans by baseline, as for English
        for y, x, t in spans:
            if cur and abs(y - cur_y) > 3.0:
                rows.append(cur)
                cur = []
            if not cur:
                cur_y = y
            cur.append((x, t))
        if cur:
            rows.append(cur)
        for row in rows:
            line = re.sub(r"\s+", " ", " ".join(t for _, t in sorted(row))).strip()
            if line and not PAGE_NUM.match(line):
                yield p, line


def segment(lines, bounds):
    """Group (page, line) stream into dated entries. Text before the first dated order is dropped."""
    entries, cur, prev_year = [], None, None
    for p, line in lines:
        d = parse_date_line(line, prev_year, bounds)
        if d:
            prev_year = d["year"]
            cur = {**d, "pages": [p], "lines": []}
            entries.append(cur)
            continue
        if cur is not None:
            cur["lines"].append(line)
            if p != cur["pages"][-1]:
                cur["pages"].append(p)
    return [e for e in entries if join_lines(e["lines"])]  # retain every nonempty dated order, including short ones


def join_lines(lines):
    text = ""
    for l in lines:
        if text.endswith("-") and not text.endswith(" -"):
            text = text[:-1] + l  # rejoin hyphenated wrap
        else:
            text = f"{text} {l}" if text else l
    return re.sub(r"\s+", " ", text).strip()


# English passages: about 100 words each, always ending at a sentence end. Sentences are grouped until a chunk
# reaches TARGET_WORDS; a chunk is closed early rather than grow past MAX_WORDS; a short last chunk joins the
# one before. A single sentence longer than HARD_WORDS is divided at semicolons. The order is retrieved as a
# whole; passages only represent its meaning in pieces.
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\u2018\u201c0-9])")
ABBREV = re.compile(r"(?:\b(?:Mrs?|Dr|St|No|Nos|Vol|pp?|viz|etc|cf|ibid|i\.e|e\.g|ca)\.|\b[A-Z]\.)$")
TARGET_WORDS, MAX_WORDS, HARD_WORDS = 100, 140, 150
MIN_LAST = SETTINGS.short_chunk_words


def _sentences(text):
    parts = []
    for piece in SENT_SPLIT.split(text):
        if parts and ABBREV.search(parts[-1]):
            parts[-1] += " " + piece      # "Mrs Maha", "No. 332", "i.e. military": not a sentence end
        else:
            parts.append(piece)
    return [q.strip() for p in parts for q in _divide(p, ("; ", ", ", " ")) if q.strip()]


def _divide(sentence, seps):
    """Very long run-on sentences (lists, tables): divide at semicolons, then commas, then spaces."""
    if len(sentence.split()) <= HARD_WORDS or not seps:
        return [sentence]
    out, cur = [], ""
    for c in sentence.split(seps[0]):
        cand = f"{cur}{seps[0]}{c}" if cur else c
        if cur and len(cand.split()) > TARGET_WORDS:
            out.append(cur + seps[0].rstrip())
            cur = c
        else:
            cur = cand
    out.append(cur)
    return [q for p in out for q in _divide(p, seps[1:])]


def english_passages(text):
    chunks, cur = [], []
    for sent in _sentences(re.sub(r"\s+", " ", text).strip()):
        n = len(sent.split())
        if cur and sum(len(x.split()) for x in cur) + n > MAX_WORDS:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(sent)
        if sum(len(x.split()) for x in cur) >= TARGET_WORDS:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        last = " ".join(cur)
        if chunks and len(last.split()) < MIN_LAST:
            chunks[-1] += " " + last
        else:
            chunks.append(last)
    return merge_short_passages(chunks)


def burmese_passages(text, max_len=400):
    out = []
    for s in re.split(r"(?<=။)", text):
        s = s.strip()
        while len(s) > max_len:
            cut = s.rfind("၊", 0, max_len)
            cut = cut + 1 if cut > 80 else max_len
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if len(s) > 3:
            out.append(s)
    return out


def than_tun_topics():
    """Part 10 subject index: 'Topic (d Mon yyyy, ...)' -> {date: [topics]}."""
    path = RAW / INDEX_PDF
    if not path.exists():
        return {}
    doc = pymupdf.open(path)
    text, started = [], False
    for p in doc:
        t = p.get_text("text")
        if "(Royal Order date in parenthesis)" in t:
            started = True
        if started:
            text.append(t)
    blob = re.sub(r"\s+", " ", " ".join(text))
    topics = defaultdict(set)
    date_re = re.compile(r"(\d{1,2})\s*([A-Za-z]{3})[a-z]*\.?\s*(1\s*[78]\s*\d\s*\d)")
    for m in re.finditer(r"([A-Z][A-Za-z ,'&\-/]{2,80}?)\s*\(([^()]{6,4000})\)", blob):
        topic = m.group(1).strip(" ,")
        for d in date_re.finditer(m.group(2)):
            mon = MONTH_ABBR.get(d.group(2).lower())
            year = int(d.group(3).replace(" ", ""))
            if mon and 1751 <= year <= 1885:
                topics[f"{year:04d}-{mon:02d}-{int(d.group(1)):02d}"].add(topic)
    return {k: sorted(v) for k, v in topics.items()}


def preserve_order_ids(records, existing):
    """Keep old citations stable when newly recognized headings split an entry."""
    if not existing:
        return {}
    by_pdf = defaultdict(list)
    for old in existing:
        by_pdf[old['pdf']].append(old)
    reserved = {r['id'] for r in existing}
    used, assigned, mapping = set(), {}, defaultdict(list)
    # Reserve exact matches first; a split parent must not steal a child's ID.
    for i, rec in enumerate(records):
        exact = [old for old in by_pdf[rec['pdf']] if old['en'] == rec['en'] and old['id'] not in used]
        if exact:
            old = max(exact, key=lambda x: len(set(x['pdf_pages']) & set(rec['pdf_pages'])))
            assigned[i] = old['id']
            used.add(old['id'])
            mapping[old['id']].append(old['id'])
    for i, rec in enumerate(records):
        if i in assigned:
            rec['id'] = assigned[i]
            continue
        parents = [old for old in by_pdf[rec['pdf']] if rec['en'] in old['en']
                   and set(old['pdf_pages']) & set(rec['pdf_pages'])]
        old = min(parents, key=lambda x: len(x['en'])) if parents else None
        if old and old['id'] not in used and old['en'].startswith(rec['en']):
            oid = old['id']
        else:
            stem = old['id'] if old else f"rob{rec['part']}"
            oid = f"{stem}_{rec['date'].replace('-', '')}"
            if oid in reserved or oid in used:
                oid += '_' + hashlib.sha256(rec['en'].encode()).hexdigest()[:8]
            base, serial = oid, 2
            while oid in reserved or oid in used:
                oid = f'{base}_{serial}'
                serial += 1
        used.add(oid)
        rec['id'] = oid
        if old:
            mapping[old['id']].append(oid)
    return {old['id']: mapping[old['id']] for old in existing}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    topics = than_tun_topics()
    records, stats = [], {}
    for part, (fname, en_ranges, my_ranges) in PARTS.items():
        doc = pymupdf.open(RAW / fname)
        bounds = PART_YEARS[part]
        en = segment(english_lines(doc, page_ranges(doc, en_ranges)), bounds)
        my = segment(burmese_lines(doc, page_ranges(doc, my_ranges)), bounds) if my_ranges else []
        my_by_date = defaultdict(list)
        for e in my:
            my_by_date[e["date"]].append(e)
        en_by_date = defaultdict(list)
        for e in en:
            en_by_date[e["date"]].append(e)
        aligned = 0
        for i, e in enumerate(en, 1):
            oid = f"rob{part}_{i:04d}"
            text = join_lines(e["lines"])
            rec = {
                "id": oid, "source": "order", "part": part, "date": e["date"], "year": e["year"],
                "date_supplied": e["date_supplied"], "date_uncertain": e["date_uncertain"],
                "date_original": e["date_original"], "date_out_of_period": e["date_out_of_period"],
                "pdf": fname, "pdf_pages": e["pages"],
                "en": text, "en_passages": english_passages(text),
                "my": "", "my_passages": [], "my_alignment": None,
                "topics": topics.get(e["date"], []),
            }
            same_en, same_my = en_by_date[e["date"]], my_by_date.get(e["date"], [])
            if same_my:
                if len(same_en) == len(same_my):  # same number of orders that day: pair in sequence
                    m = same_my[same_en.index(e)]
                    rec["my_alignment"] = "paired"
                    aligned += 1
                    mtext = join_lines(m["lines"])
                    rec["my_pages"] = m["pages"]
                else:  # counts differ: attach all Burmese for that date, flagged
                    rec["my_alignment"] = "same_date"
                    mtext = "\n\n".join(join_lines(m["lines"]) for m in same_my)
                    rec["my_pages"] = sorted({p for m in same_my for p in m["pages"]})
                rec["my"] = mtext
                rec["my_passages"] = burmese_passages(mtext)
            records.append(rec)
        stats[part] = {"orders": len(en), "burmese_entries": len(my), "paired": aligned,
                       "with_burmese": sum(1 for r in records if r["part"] == part and r["my"]),
                       "date_uncertain": sum(1 for r in records if r["part"] == part and r["date_uncertain"])}
        print(f"Part {part}: {stats[part]}")
    existing_path = OUT / 'orders.jsonl'
    existing = [json.loads(line) for line in existing_path.read_text(encoding='utf-8').splitlines()] if existing_path.exists() else []
    id_mapping = preserve_order_ids(records, existing)
    if id_mapping:
        (OUT / 'orders_id_mapping.json').write_text(json.dumps(id_mapping, indent=2), encoding='utf-8')
    with open(OUT / "orders.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"parts": stats, "topics_dates": len(topics)}, open(OUT / "orders_stats.json", "w"), indent=1)
    print(f"{len(records)} orders -> {OUT / 'orders.jsonl'}; index topics for {len(topics)} dates")


if __name__ == "__main__":
    main()
