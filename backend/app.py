"""Konbaung search server.  Run:  python backend/app.py   (then open http://127.0.0.1:8765)

GET  /api/search   free hybrid search (no LLM)
POST /api/spider   SearchSpider: initial search and optional additional evidence reading
POST /api/spider/followup  brief follow-up answer from a cached run (no new search)
POST /api/spider/deeper    assess the next 50 unread documents of a cached run and summarise them
GET  /api/status   corpus counts and whether SearchSpider has a key
"""
import json
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

# Local use: when a backend/*.py file changes, the server exits (code 3) and run.bat's loop starts it again with
# the new code. (uvicorn's own reloader hung on Windows, leaving the old code running.) WATCH=0 turns this off.
if __name__ == "__main__" and os.getenv("WATCH", "1") == "1":
    import threading

    def _watch():
        files = lambda: {p: p.stat().st_mtime for p in Path(__file__).parent.glob("*.py")}
        start = files()
        while True:
            time.sleep(2)
            try:
                if files() != start:
                    print("backend code changed: exiting so run.bat restarts the server with it", flush=True)
                    os._exit(3)
            except OSError:
                pass

    threading.Thread(target=_watch, daemon=True).start()

import uvicorn  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import AliasChoices, BaseModel, Field  # noqa: E402

from search import Engine  # noqa: E402
from search_settings import SETTINGS  # noqa: E402
from dossier import Dossier  # noqa: E402
from english import en_tokens  # noqa: E402
from spider import MODEL, Spider  # noqa: E402
import access  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "frontend" / "dist" / "index.html"
VERSION = time.strftime("%Y-%m-%d %H:%M", time.localtime(max(p.stat().st_mtime for p in Path(__file__).parent.glob("*.py"))))

t0 = time.time()
engine = Engine()
spider = Spider()
print(f"server code {VERSION}; engine ready in {time.time() - t0:.1f}s: {len(engine.orders)} orders, {len(engine.chron)} chronicle sentences; "
      f"SearchSpider {'on (' + MODEL + ')' if spider.available else 'unavailable: ' + str(spider.error)}")

app = FastAPI(title="SearchSpider", docs_url=None if access.PUBLIC else "/docs", redoc_url=None if access.PUBLIC else "/redoc")
# allow the page to be opened directly from disk (origin "null") or from the Vite dev server
if not access.PUBLIC:
    app.add_middleware(CORSMiddleware, allow_origins=["null", "http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
access.install_routes(app)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


SEARCH_SLOTS = threading.BoundedSemaphore(2)


def search_slot():
    if not SEARCH_SLOTS.acquire(blocking=False):
        raise HTTPException(429, "Search is busy. Please try again shortly.")
    try:
        yield
    finally:
        SEARCH_SLOTS.release()


def model_for(who):
    if who["mode"] == "local":
        return spider
    client = Spider()
    if not client.available:
        raise HTTPException(503, "Hosted AI is temporarily unavailable. Basic search is still available.")
    return access.MeteredSpider(client, who)


def _split(v):
    return [x for x in (v or "").split(",") if x]


@app.get("/")
def page():
    return FileResponse(PAGE)


@app.get("/api/status")
def status():
    return {"orders": len(engine.orders), "sentences": len(engine.chron), "pages": len(engine.pages),
            "reranker": engine.reranker is not None, "kg_triples": 0,
            "settings": SETTINGS.public(), "spider": spider.available or access.PUBLIC, "hosted_spider": spider.available,
            "spider_model": MODEL, "spider_error": None if access.PUBLIC else spider.error, "version": VERSION}


@app.get("/api/search", dependencies=[Depends(search_slot)])
def search(q: str = Query(..., min_length=1, max_length=2000), sources: str = "order,chronicle",
           year_from: Optional[int] = None, year_to: Optional[int] = None,
           parts: str = "", volumes: str = "", k: int = 20, rerank: bool = False):
    if not q.strip():
        raise HTTPException(400, "Enter an English search query.")
    t = time.time()
    res = engine.search(q, sources=_split(sources), year_from=year_from, year_to=year_to,
                        parts=_split(parts), volumes=_split(volumes),
                        rerank=rerank)
    res["ms"] = int((time.time() - t) * 1000)
    return res


class SpiderReq(BaseModel):
    q: str = Field(min_length=1, max_length=2000)
    sources: List[str] = ["order", "chronicle"]
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    parts: List[str] = []
    volumes: List[str] = []
    reading_rounds: int = Field(default=SETTINGS.reading_rounds, ge=0,
                                le=SETTINGS.max_reading_rounds, strict=True,
                                validation_alias=AliasChoices("reading_rounds", "recursive_searches"))



STATUS = {"r": "relevant", "p": "partial", "n": "dropped"}

# Finished runs, kept in memory so follow-ups and deeper reading reuse the frozen search.
# Lost when the server restarts (including the automatic restart after a backend code change).
RUNS: "OrderedDict[str, tuple]" = OrderedDict()
RUNS_LOCK = threading.Lock()


def _cached(run_id, who):
    with RUNS_LOCK:
        if run_id not in RUNS:
            raise HTTPException(410, "This search is no longer cached on the server (it restarted or the run is old). "
                                     "Run the search again.")
        d, lock = RUNS[run_id]
        if d.access_subject != who["subject"] and not (who.get("browser") and d.access_browser == who["browser"]):
            raise HTTPException(403, "This research session belongs to a different account or browser.")
        RUNS.move_to_end(run_id)
        return d, lock


def _status(d, key):
    v = d.verdict.get(key)
    return v, STATUS[v["v"]] if v else ("missing" if key in d.missing else "not_assessed")


def _hit(d, it):
    key = it["key"]
    v, status = _status(d, key)
    extra = {"alias": d.alias_of[key], "spider": v, "status": status, "judged": v is not None,
             "found": it["routes"], "n_routes": len(it["routes"]), "rank": d.first_rank[key],
             "terms": engine.terms_in(key, d.keyword_stems)}
    return {**engine.hit(it), **extra}


def _counts(d):
    counts = {"relevant": 0, "partial": 0, "dropped": 0, "missing": 0, "not_assessed": 0}
    for key in d.pool:
        counts[_status(d, key)[1]] += 1
    return counts


def _usage(d):
    u = d.usage
    return {"in": sum(x["in"] for x in u), "out": sum(x["out"] for x in u),
            "think": sum(x.get("think", 0) for x in u), "calls": len(u)}


@app.post("/api/spider", dependencies=[Depends(search_slot)])
def run_spider(req: SpiderReq, request: Request = None):
    r = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    if not req.q.strip():
        raise HTTPException(400, "Enter an English search query.")
    who = access.authorize(request)
    if who["mode"] == "free":
        r["reading_rounds"] = 0
    client = model_for(who)
    d = Dossier(engine, client, r)
    started = time.time()
    stopped = None
    try:
        answer, secs = d.run()
    except access.BudgetExhausted as ex:
        stopped = ex.detail
        if not d.pool:
            d.gather(include_planned=False)
        d._aliases(d.items())
        answer = d.summary = "AI budget reached. " + stopped + " The retrieved documents and any completed assessments are available below."
        secs = time.time() - started
    except HTTPException:
        raise
    except Exception as e:  # network, quota, malformed output
        d.save_log({"error": type(e).__name__})
        raise HTTPException(502, "SearchSpider could not finish. Check your token budget and try again. Budget reserved for failed calls remains used.") from None
    finally:
        if access.PUBLIC:
            client.close()
            d.sp = None  # Keys and subscription tokens never remain in cached dossiers.
    d.save_log({"answer": answer, "secs": round(secs, 1)})
    orders, pages, refs = [], [], {}
    items = d.items()
    d.first_rank = {it["key"]: rank for rank, it in enumerate(items, 1)}
    aliases = list(dict.fromkeys(a for r in d.retrievals.values() for a in r["aliases"]))
    extra = [t for k in d.keywords for t in en_tokens(k)]
    keywords = engine.keywords([t for r in d.retrievals.values() for t in r["qparts"][0][1]] + extra,
                               [*d.retrievals, *aliases, *d.keywords])
    d.keyword_stems = [k["stem"] for k in keywords]
    for it in items:
        key = it["key"]
        (orders if key[0] == "o" else pages).append(_hit(d, it))
        refs[d.alias_of[key]] = {"kind": "order" if key[0] == "o" else "page", "id": d._kid(key), "label": d.label(it)}
    for k in keywords:
        k["n"] = sum(k["stem"] in h["terms"] for h in orders + pages)
    run_id = uuid.uuid4().hex
    d.access_subject, d.access_mode = who["subject"], who["mode"]
    d.access_browser = who.get("browser", "")
    with RUNS_LOCK:
        RUNS[run_id] = (d, threading.Lock())
        while len(RUNS) > SETTINGS.cached_runs:
            RUNS.popitem(last=False)
    return {"run_id": run_id, "can_continue": True, "access_mode": who["mode"],
            "orders": orders, "pages": pages, "summary": answer, "refs": refs, "trail": d.trail, "rounds": d.rounds,
            "aliases": aliases, "keywords": [k for k in keywords if k["n"]],
            "highlight_terms": engine.highlight_terms([t for r in d.retrievals.values() for t in r["qparts"][0][1]] + extra),
            "extra_keywords": d.keywords,
            "reading_rounds": d.reading_rounds, "completed_reading_rounds": d.completed_reading_rounds,
            "response_tokens": d.response_tokens,
            "counts": _counts(d), "settings": SETTINGS.public(), "stop": "budget" if stopped else "done", "ms": int(secs * 1000), "log": None if access.PUBLIC else d.log_path,
            "usage": _usage(d)}


class FollowupReq(BaseModel):
    run_id: str = Field(max_length=64)
    q: str = Field(min_length=1, max_length=2000)


@app.post("/api/spider/followup", dependencies=[Depends(search_slot)])
def spider_followup(req: FollowupReq, request: Request = None):
    if not req.q.strip():
        raise HTTPException(400, "Enter a follow-up question.")
    who = access.authorize(request)
    d, lock = _cached(req.run_id, who)
    with lock:
        d.sp = model_for(who)
        try:
            entry = d.followup(req.q)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, "Follow-up could not finish. Check your token budget.") from None
        finally:
            if access.PUBLIC:
                d.sp.close()
                d.sp = None
        d.save_log()
        return {"followup": entry, "trail": d.trail, "rounds": d.rounds, "usage": _usage(d)}


class DeeperReq(BaseModel):
    run_id: str = Field(max_length=64)


@app.post("/api/spider/deeper", dependencies=[Depends(search_slot)])
def spider_deeper(req: DeeperReq, request: Request = None):
    who = access.authorize(request)
    d, lock = _cached(req.run_id, who)
    with lock:
        d.sp = model_for(who)
        before, usage_start = set(d.assessed), len(d.usage)
        try:
            entry = d.deeper()
        except access.BudgetExhausted as ex:
            keys = d.assessed - before
            spent = d.usage[usage_start:]
            entry = {"n": len(d.deeper_rounds) + 1, "keys": [list(k) for k in keys], "assessed": len(keys),
                     "tally": "budget reached", "text": ex.detail + " Completed document assessments have been kept.",
                     "in": sum(x["in"] for x in spent), "out": sum(x["out"] for x in spent)}
            d.deeper_rounds.append(entry)
            d.history.append(("deeper", entry))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, "Deeper reading could not finish. Check your token budget.") from None
        finally:
            if access.PUBLIC:
                d.sp.close()
                d.sp = None
        d.save_log()
        keys = {tuple(k) for k in entry["keys"]}
        updated = [{**_hit(d, d.pool[k]), "deeper": entry["n"]} for k in keys]
        return {"deeper": {k: v for k, v in entry.items() if k != "keys"}, "updated": updated,
                "counts": _counts(d), "remaining": len(d.unread()),
                "trail": d.trail, "rounds": d.rounds, "usage": _usage(d)}


class PlanReq(BaseModel):
    q: str
    tid: Optional[str] = None


@app.post("/api/plan")
def plan_only(req: PlanReq):
    """Evaluation helper: Gemini's text search plan, without gathering or assessing.
    With tid, the plan is also appended to eval/thematic/plans.jsonl."""
    if access.PUBLIC:
        raise HTTPException(404)
    if not spider.available:
        raise HTTPException(503, spider.error or "SearchSpider unavailable")
    d = Dossier(engine, spider, {"q": req.q, "sources": ["order", "chronicle"]})
    d.plan()
    out = {"tid": req.tid, "q": req.q, "queries": d.queries, "keywords": d.keywords,
           "usage": d.usage}
    if req.tid:
        f = Path(__file__).resolve().parents[1] / "eval" / "thematic" / "plans.jsonl"
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


@app.post("/api/eval_shortlist")
def eval_shortlist(tid: str, k: int = SETTINGS.gemini_shortlist):
    """Evaluation helper: Gemini assesses the union of the shortlists each ranking variant would send."""
    if access.PUBLIC:
        raise HTTPException(404)
    import evaltools
    res = evaltools.run(engine, spider, tid, k)
    f = Path(__file__).resolve().parents[1] / "eval" / "thematic" / "shortlist_gemini.jsonl"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res, ensure_ascii=False) + "\n")
    return {x: res[x] for x in ("tid", "union", "usage_in", "usage_out", "variants")}


@app.on_event("startup")
def _open_browser():
    if os.getenv("OPEN_BROWSER") == "1":
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{os.getenv('PORT', '8765')}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8765")))
