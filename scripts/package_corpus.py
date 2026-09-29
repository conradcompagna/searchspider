"""Create a complete corpus handoff for independent baseline generation.

Copies all indexed text byte-for-byte, plus available raw PDFs and graph CSVs.
Adds English reading views, manageable review batches, IDs, hashes and a prompt.
No engine, model, index, API client or existing evaluation labels are loaded.
"""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ["orders.jsonl", "chronicle_sentences.jsonl", "chronicle_pages.jsonl", "orders_stats.json"]
TARGET_WORDS = 10000


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build(out):
    if out.exists():
        raise FileExistsError(f"Archive already exists: {out}")
    processed = ROOT / "data" / "processed"
    orders = jsonl(processed / "orders.jsonl")
    sentences = jsonl(processed / "chronicle_sentences.jsonl")
    stored_pages = jsonl(processed / "chronicle_pages.jsonl")
    pages = defaultdict(list)
    for sentence in sentences:
        pages[sentence["page_id"]].append(sentence)
    assert len({o["id"] for o in orders}) == len(orders)
    assert len({s["id"] for s in sentences}) == len(sentences)
    assert {p["page_id"] for p in stored_pages} == set(pages)
    for page in stored_pages:
        assert page["sentences"] == [s["id"] for s in pages[page["page_id"]]]

    members, inputs = {}, []
    for name in CANONICAL:
        source = processed / name
        target = "canonical/" + name
        members[target] = source.read_bytes()
        inputs.append({"repository_path": source.relative_to(ROOT).as_posix(), "archive_path": target})
    for source in sorted((ROOT / "data" / "kg").glob("*.csv")):
        target = "graph/" + source.name
        members[target] = source.read_bytes()
        inputs.append({"repository_path": source.relative_to(ROOT).as_posix(), "archive_path": target})
    for source in sorted((ROOT / "data" / "raw").glob("*.pdf")):
        target = "original_pdfs/" + source.name
        members[target] = source.read_bytes()
        inputs.append({"repository_path": source.relative_to(ROOT).as_posix(), "archive_path": target})

    blocks, catalog, anchors = [], [], []
    by_file = defaultdict(list)
    for order in orders:
        doc = order["id"]
        path = f"english_by_source/royal_orders_part_{order['part']:02d}.md"
        body = "\n\n".join(order["en_passages"])
        text = (f"## {doc}\n\nPart: {order['part']} | Date: {order['date']} | "
                f"PDF: {order['pdf']} | PDF pages: {', '.join(map(str, order['pdf_pages']))}\n\n"
                f"Date supplied: {order['date_supplied']} | Date uncertain: {order['date_uncertain']}\n\n{body}\n")
        words = len(body.split())
        row = {"document_id": doc, "source": "order", "part": order["part"], "volume": "", "page": "",
               "date": order["date"], "english_words": words, "english_file": path, "review_batch": "",
               "pdf": order["pdf"], "pdf_pages": ";".join(map(str, order["pdf_pages"]))}
        catalog.append(row)
        blocks.append((row, text))
        by_file[path].append(text)
    for doc, group in sorted(pages.items(), key=lambda pair: (pair[1][0]["volume"], pair[1][0]["page"])):
        first = group[0]
        path = f"english_by_source/chronicle_volume_{first['volume']:02d}.md"
        body = "\n\n".join(f"[{s['id']}]\n{s['en']}" for s in group)
        text = f"## {doc}\n\nVolume: {first['volume']} | Page: {first['page']} | Sentences: {len(group)}\n\n{body}\n"
        words = sum(len(s["en"].split()) for s in group)
        row = {"document_id": doc, "source": "chronicle", "part": "", "volume": first["volume"],
               "page": first["page"], "date": "", "english_words": words, "english_file": path,
               "review_batch": "", "pdf": "", "pdf_pages": ""}
        catalog.append(row)
        blocks.append((row, text))
        by_file[path].append(text)
        anchors.extend({"sentence_id": s["id"], "document_id": doc, "volume": s["volume"],
                        "page": s["page"], "sequence": s["seq"]} for s in group)

    for path, texts in by_file.items():
        members[path] = ("# " + Path(path).stem.replace("_", " ").title() + "\n\n" + "\n".join(texts)).encode("utf-8")
    batch_rows, pending, word_count = [], [], 0

    def flush():
        nonlocal pending, word_count
        if not pending:
            return
        path = f"review_batches/batch_{len(batch_rows) + 1:03d}.md"
        for row, _ in pending:
            row["review_batch"] = path
        members[path] = (f"# Review batch {len(batch_rows) + 1:03d}\n\n"
                         "These are complete indexed documents; this is a duplicate reading view of the canonical corpus.\n\n"
                         + "\n".join(text for _, text in pending)).encode("utf-8")
        batch_rows.append({"file": path, "documents": len(pending), "english_words": word_count,
                           "first_document_id": pending[0][0]["document_id"],
                           "last_document_id": pending[-1][0]["document_id"]})
        pending, word_count = [], 0

    for row, text in blocks:
        # Never split an Order or Chronicle page across batches, even if one exceeds the target.
        if pending and (word_count + row["english_words"] > TARGET_WORDS or row["source"] != pending[0][0]["source"]):
            flush()
        pending.append((row, text))
        word_count += row["english_words"]
    flush()
    members["document_catalog.csv"] = csv_bytes(catalog, list(catalog[0]))
    members["sentence_catalog.csv"] = csv_bytes(anchors, list(anchors[0]))
    members["review_batches.csv"] = csv_bytes(batch_rows, list(batch_rows[0]))
    prompt = (ROOT / "docs" / "BASELINE_GENERATION_PROMPT.md").read_bytes()
    members["PROMPT_FOR_WEB_GPT.md"] = prompt

    counts = {"royal_orders": len(orders), "chronicle_pages": len(pages), "chronicle_sentences": len(sentences),
              "benchmark_documents": len(catalog), "orders_by_part": dict(Counter(o["part"] for o in orders)),
              "pages_by_volume": dict(Counter(g[0]["volume"] for g in pages.values())),
              "sentences_by_volume": dict(Counter(s["volume"] for s in sentences)),
              "english_words": sum(r["english_words"] for r in catalog), "review_batches": len(batch_rows),
              "raw_pdfs": sum(path.startswith("original_pdfs/") for path in members),
              "graph_csvs": sum(path.startswith("graph/") for path in members)}
    readme = f"""# Complete Konbaung search corpus for independent relevance baselines

Start with **PROMPT_FOR_WEB_GPT.md**. This handoff includes the complete indexed text, not a sample or a retrieved subset. No search rankings, prior baseline labels, generated answers or planned search queries are included.

## Inventory

- {len(orders):,} Royal Orders, Parts 3–9.
- {len(pages):,} Chronicle pages, volumes 1–3, with {len(sentences):,} sentences.
- {len(catalog):,} benchmark documents in total; {counts['english_words']:,} English words.
- {len(batch_rows)} review batches of approximately {TARGET_WORDS:,} English words, without splitting documents. An unusually long individual document can exceed the target.
- {counts['raw_pdfs']} available source PDFs and {counts['graph_csvs']} graph tables.

## Files and reading order

- `canonical/`: byte-for-byte copies of all four processed corpus files. `orders.jsonl` includes full English, searchable English passages, available Burmese, dates, PDF citations and metadata. `chronicle_sentences.jsonl` includes English and Burmese text with exact sentence/page IDs. `chronicle_pages.jsonl` is the sentence-to-page grouping. `orders_stats.json` is the existing source-processing summary.
- `review_batches/`: complete English reading batches; use these for systematic review and checkpoints.
- `english_by_source/`: the same English text grouped by Order Part or Chronicle volume, with document IDs and sentence anchors.
- `document_catalog.csv`: every valid benchmark document ID and its reading file/batch.
- `sentence_catalog.csv`: every Chronicle sentence ID and its owning benchmark page.
- `review_batches.csv`: expected batch contents and word counts.
- `original_pdfs/`: all available Royal Orders PDFs in this repository, including Part 10 as supplementary reference. There are no Chronicle PDFs in this repository.
- `graph/`: complete entities.csv and triples.csv. These are discovery aids, not independently verified relevance judgments.
- `MANIFEST.json`: counts, input-file provenance, byte sizes and SHA-256 checksums for all other archive members.

The reading views duplicate canonical text; count each document once. Order reading text is the complete stored `en_passages` sequence. Runtime short-passage merging changes boundaries only, so passage numbers should not be treated as stable benchmark IDs. Chronicle sentence strings are preserved exactly and kept in stored order within their owning page; pages are presented in volume/page order. Original `pages` metadata is retained in canonical JSONL for sentences spanning printed pages.

## Scope

Use exact `rob...` Order IDs and `v...p....` Chronicle page IDs from the catalog. A Chronicle page grade is the highest supported sentence grade on that page. Burmese is preserved wherever the source dataset contains it; absent Burmese has not been invented. English OCR/translation imperfections and original metadata are retained.

Benchmark the indexed English documents. Part 10 and any other unindexed PDF material can clarify context but must not create invented document IDs or positive benchmark entries. Burmese-only evidence should be reported separately if it is not expressed in the indexed English text.

Embeddings, neural-network weights, environments, application code, credentials, logs and previous evaluations are omitted because they are not source corpus evidence.

## Output compatibility

For each question, return a JSON array with one positive entry per document: `document_id`, `grade` (`relevant` or `partial`), plus rationale and source anchors. This directly fits `scripts/replay_search.py` in the originating project. Keep uncertain cases and review coverage separate. Do not silently count unreviewed documents as irrelevant.
"""
    members["README.md"] = readme.encode("utf-8")
    inventory = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in members.items()}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "counts": counts,
                "inputs": inputs, "files": inventory, "manifest_excludes_its_own_hash": True}
    members["MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    # Validate the actual deliverable, including canonical byte identity and complete ID coverage.
    with zipfile.ZipFile(out) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(members)
        for name, info in inventory.items():
            data = archive.read(name)
            assert len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["sha256"]
        for source in inputs:
            assert archive.read(source["archive_path"]) == (ROOT / source["repository_path"]).read_bytes()
        assert len(catalog) == len({r["document_id"] for r in catalog}) == len(orders) + len(pages)
        assert sum(r["documents"] for r in batch_rows) == len(catalog)
        assert all(r["document_id"] in archive.read(r["review_batch"]).decode("utf-8") for r in catalog)
        assert len(anchors) == len(sentences)
    # A sidecar prompt can be pasted without first extracting the archive.
    prompt_out = out.with_name(out.stem + "_PROMPT.md")
    if prompt_out.exists():
        raise FileExistsError(prompt_out)
    prompt_out.write_bytes(prompt)
    result = {"archive": str(out), "prompt": str(prompt_out), "bytes": out.stat().st_size,
              "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "members": len(members),
              "verified": True, "counts": counts}
    out.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_exports" / ("konbaung_complete_corpus_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip"))
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
