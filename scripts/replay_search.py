"""Replay a saved plan through the current app's local pipeline; never call Gemini.

Plan JSON: {"q": "question", "queries": [...], "lookups": [...]}.
A saved logs/runs pipeline log is also accepted. Optional baseline JSON is a list
of {"document_id": "rob... or v...", "grade": "relevant or partial"}; CSV works too.
Sentence-level baseline rows may repeat document_id: the highest grade wins.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["HF_HUB_OFFLINE"] = "1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def saved_plan(raw):
    req = raw.get("request", {})
    q = raw.get("q") or raw.get("question") or req.get("q")
    if not isinstance(q, str) or not q.strip():
        raise ValueError("The plan must contain a nonempty q or question.")
    stages = {r["kind"]: r.get("response", {}) for r in raw.get("gemini", []) if "kind" in r}
    queries = raw.get("queries", stages.get("plan_search", {}).get("queries", []))
    lookups = raw.get("lookups", stages.get("plan_graph", {}).get("lookups", []))
    filters = {k: req[k] for k in ("sources", "year_from", "year_to", "parts", "volumes") if k in req}
    filters.update({k: raw[k] for k in ("sources", "year_from", "year_to", "parts", "volumes") if k in raw})
    return {"q": q.strip(), "sources": ["order", "chronicle"], **filters}, queries, lookups


def baseline_labels(path):
    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = read(path)
    labels = {}
    for row in rows:
        doc = row.get("document_id") or row.get("id")
        grade = {"relevant": 2, "partial": 1, "2": 2, "1": 1}.get(str(row.get("grade")))
        if not doc or grade is None:
            raise ValueError("Baseline rows require document_id (or id) and grade relevant/partial or 2/1.")
        labels[doc] = max(labels.get(doc, 0), grade)
    return labels


def metrics(rows, labels):
    out = {}
    for source in ("order", "chronicle"):
        group = [r for r in rows if r["source"] == source]
        out[source] = {"candidates": len(group)}
        for depth in (10, 20, 50, 100, 250):
            head = group[:depth]
            strict = sum(labels.get(r["document_id"]) == 2 for r in head)
            broad = sum(labels.get(r["document_id"], 0) > 0 for r in head)
            out[source][str(depth)] = {"fully_relevant": strict, "relevant_or_partial": broad,
                                      "strict_precision": strict / len(head) if head else 0.,
                                      "unlabelled": len(head) - broad}
    ids = {r["document_id"] for r in rows}
    relevant = {k for k, grade in labels.items() if grade == 2}
    out["coverage"] = {"fully_relevant_found": len(ids & relevant), "fully_relevant_total": len(relevant),
                       "relevant_or_partial_found": len(ids & labels.keys()), "baseline_total": len(labels),
                       "missing_relevant": sorted(relevant - ids), "missing_baseline": sorted(labels.keys() - ids)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "replays" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    req, queries, lookups = saved_plan(read(args.plan))
    labels = baseline_labels(args.baseline)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose an empty output directory to preserve previous results.")
    args.out.mkdir(parents=True, exist_ok=True)
    from dossier import Dossier
    from search import Engine, local_key, source_normalized
    from search_settings import SETTINGS

    started = time.time()
    engine = Engine()
    print("Local engine loaded; replaying saved initial text queries only.", flush=True)
    d = Dossier(engine, None, req)  # no Gemini client exists in this replay
    d.queries = list(dict.fromkeys(q.strip() for q in queries if isinstance(q, str) and q.strip()))
    d.gather()
    d.score()
    shortlist = {it["key"] for it in source_normalized(list(d.pool.values()))[:SETTINGS.gemini_shortlist]}
    rows = []
    for kind, source in (("o", "order"), ("p", "chronicle")):
        items = sorted((it for it in d.pool.values() if it["key"][0] == kind), key=local_key)
        for rank, it in enumerate(items, 1):
            doc = d._kid(it["key"])
            rows.append({"source": source, "rank": rank, "document_id": doc, "score": it["fusion_score"],
                         "grade": labels.get(doc, 0), "shortlisted": it["key"] in shortlist,
                         "representative_unit": it["unit"], "representative_passage": it["passage"],
                         "routes": "; ".join(it["routes"])})
    columns = ["source", "rank", "document_id", "score", "grade", "shortlisted", "representative_unit",
               "representative_passage", "routes"]
    with (args.out / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    ranks = {r["document_id"]: r for r in rows}
    baseline_rows = [{"document_id": doc, "grade": grade, "source": ranks.get(doc, {}).get("source"),
                      "rank": ranks.get(doc, {}).get("rank"), "shortlisted": ranks.get(doc, {}).get("shortlisted", False)}
                     for doc, grade in labels.items()]
    payload = {"gemini_calls": 0, "seconds": round(time.time() - started, 2), "request": req,
               "queries": list(d.retrievals), "ignored_legacy_lookups": lookups, "settings": SETTINGS.public(),
               "metrics": metrics(rows, labels), "baseline_ranks": baseline_rows,
               "shortlist": [d._kid(it["key"]) for it in source_normalized(list(d.pool.values()))[:SETTINGS.gemini_shortlist]],
               "searches": d.log["searches"],
               "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT / "backend").glob("*.py")}}
    (args.out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.out), "seconds": payload["seconds"], "gemini_calls": 0,
                      "metrics": payload["metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
