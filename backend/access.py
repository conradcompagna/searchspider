"""Production access control. Private keys and billing never enter search requests/logs."""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request as URLRequest, urlopen

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

PUBLIC = os.getenv("SEARCHSPIDER_PUBLIC", "0") == "1"
ORIGIN = os.getenv("SEARCHSPIDER_ORIGIN", "https://burmeseneuralreader.com").rstrip("/")
PREFIX = os.getenv("SEARCHSPIDER_PREFIX", "/searchspider" if PUBLIC else "")
BRIDGE = os.getenv("SEARCHSPIDER_BRIDGE", "https://language-engine.ai/searchspider")
SECRET = os.getenv("SEARCHSPIDER_BRIDGE_SECRET", "")
STATE_DB = Path(os.getenv("SEARCHSPIDER_STATE_DB", str(Path(__file__).resolve().parents[1] / "state" / "access.sqlite3")))
COOKIE = "__Secure-searchspider"
VISITOR_COOKIE = "__Secure-searchspider-visitor"
STATE_COOKIE = "__Secure-searchspider-state"
# Account connections have no routine server expiry. Browsers cap persistent
# cookies, so renew the 400-day cookie whenever SearchSpider checks access.
SESSION_EXPIRES = 253402300799  # 9999-12-31, compatible with the existing schema.
SESSION_COOKIE_AGE = 86400 * 400
# Integer nanodollars avoid rounding drift. USD $0.03/day = at most $0.93/31 days.
FREE_BUDGET_NANODOLLARS = 30_000_000
PRICING_NANODOLLARS = {"gemini-3.1-flash-lite": (250, 1500)}
if PUBLIC and len(SECRET) < 32:
    raise RuntimeError("Production requires SEARCHSPIDER_BRIDGE_SECRET (at least 32 characters).")


class BudgetExhausted(HTTPException):
    pass


def digest(value):
    return hmac.new(SECRET.encode(), value.encode(), hashlib.sha256).hexdigest()


@contextmanager
def database():
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(STATE_DB, timeout=15)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, token TEXT NOT NULL, expires INTEGER NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS free_spend (id TEXT PRIMARY KEY, day TEXT NOT NULL, amount INTEGER NOT NULL, settled INTEGER NOT NULL DEFAULT 0)")
        yield con
        con.commit()
    finally:
        con.close()


def bridge(action, payload):
    req = URLRequest(BRIDGE + "/api/" + action, data=json.dumps(payload).encode(),
                     headers={"Authorization": "Bearer " + SECRET, "Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=20) as response:
            return json.load(response)
    except HTTPError as ex:
        status = ex.code if ex.code in (401, 402, 403, 429) else 503
        messages = {401: "Sign in to Language Engine again.", 402: "Your Language Engine token budget is used up. Manage your subscription or wait for its next reset.",
                    403: "A verified Language Engine account is required.", 429: "Too many requests. Please try again shortly."}
        error = BudgetExhausted if status == 402 else HTTPException
        raise error(status, messages.get(status, "Language Engine access verification is temporarily unavailable.")) from None
    except (URLError, TimeoutError, ValueError):
        raise HTTPException(503, "Language Engine access verification is temporarily unavailable.") from None


def session_token(request):
    if request is None:
        return ""
    sid = request.cookies.get(COOKIE, "")
    if not sid:
        return ""
    with database() as con:
        con.execute("DELETE FROM sessions WHERE expires < ?", (int(time.time()),))
        row = con.execute("SELECT token FROM sessions WHERE id=?", (digest(sid),)).fetchone()
    return row[0] if row else ""


def identity(request):
    token = session_token(request)
    if not token:
        return {"mode": "free", "subject": visitor_subject(request), "token": ""}
    who = bridge("identity", {"token": token})
    # Upgrade an existing, still-valid seven-day connection on its next visit.
    # UPDATE cannot recreate a session that was concurrently disconnected.
    with database() as con:
        con.execute("UPDATE sessions SET expires=? WHERE id=? AND token=?",
                    (SESSION_EXPIRES, digest(request.cookies.get(COOKIE, "")), token))
    return {"mode": "paid" if who["paid"] else "free", "subject": who["subject"], "token": token}


def visitor_subject(request):
    value = request.cookies.get(VISITOR_COOKIE, "") if request else ""
    raw, _, signature = value.partition(".")
    return digest("visitor:" + raw) if raw and hmac.compare_digest(digest(raw), signature) else ""


def day():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def free_remaining():
    with database() as con:
        spent = con.execute("SELECT COALESCE(SUM(amount),0) FROM free_spend WHERE day=?", (day(),)).fetchone()[0]
    return max(0, FREE_BUDGET_NANODOLLARS - spent)


def reserve_free(model, input_tokens, output_tokens):
    if model not in PRICING_NANODOLLARS:
        raise HTTPException(503, "Sponsored search is disabled until this model's pricing is configured.")
    ir, outr = PRICING_NANODOLLARS[model]
    amount = input_tokens * ir + output_tokens * outr
    reservation = secrets.token_hex(24)
    with database() as con:
        con.execute("BEGIN IMMEDIATE")
        spent = con.execute("SELECT COALESCE(SUM(amount),0) FROM free_spend WHERE day=?", (day(),)).fetchone()[0]
        if spent + amount > FREE_BUDGET_NANODOLLARS:
            raise BudgetExhausted(429, "Today's shared free-search budget has no room for another AI call. It resets at 00:00 UTC. Use basic search or connect Language Engine Pro.")
        con.execute("INSERT INTO free_spend (id,day,amount) VALUES (?,?,?)", (reservation, day(), amount))
    return reservation


def settle_free(reservation, model, usage):
    ir, outr = PRICING_NANODOLLARS[model]
    actual = usage["in"] * ir + (usage["out"] + usage.get("think", 0)) * outr
    with database() as con:
        con.execute("UPDATE free_spend SET amount=?, settled=1 WHERE id=? AND settled=0", (actual, reservation))


def check_origin(request):
    if not PUBLIC:
        return
    if request is None:
        raise HTTPException(403, "A public request is required.")
    # No cookie-authenticated cross-origin writes. Non-browser API calls may omit Origin.
    if request.headers.get("origin") not in (None, ORIGIN):
        raise HTTPException(403, "Cross-origin request denied.")
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "Cross-origin request denied.")


def authorize(request):
    if not PUBLIC:
        return {"mode": "local", "subject": "local", "browser": "local", "token": ""}
    check_origin(request)
    if "x-gemini-api-key" in request.headers:
        raise HTTPException(403, "Personal API keys are not supported. Connect your Language Engine account.")
    who = identity(request)
    if not who["subject"]:
        raise HTTPException(401, "Refresh SearchSpider to start a browser session.")
    return {**who, "browser": visitor_subject(request)}


def public_status(request):
    if not PUBLIC:
        return {"mode": "local", "connected": False, "trial_available": True, "login_url": "", "billing_url": "https://language-engine.ai/account"}
    try:
        who = identity(request)
    except HTTPException as ex:
        if ex.status_code not in (401, 403):
            raise
        who = {"mode": "free", "subject": visitor_subject(request)}
    return {"mode": who["mode"], "connected": bool(who.get("token")), "trial_available": free_remaining() > 0,
            "login_url": PREFIX + "/auth/login", "billing_url": "https://language-engine.ai/account",
            "free_reading_rounds": 0, "free_response_tokens": 500, "reset_timezone": "UTC",
            "free_daily_budget_usd": 0.03, "free_budget_remaining_usd": free_remaining() / 1_000_000_000}


def install_routes(app):
    @app.get("/api/access")
    def access_status(request: Request):
        from fastapi.responses import JSONResponse
        status = public_status(request)
        response = JSONResponse(status)
        response.headers["Cache-Control"] = "no-store"
        sid = request.cookies.get(COOKIE, "")
        if PUBLIC and status["connected"]:
            response.set_cookie(COOKIE, sid, max_age=SESSION_COOKIE_AGE, secure=True,
                                httponly=True, samesite="lax", path=PREFIX)
        elif PUBLIC and sid:
            with database() as con:
                con.execute("DELETE FROM sessions WHERE id=?", (digest(sid),))
            response.delete_cookie(COOKIE, path=PREFIX, secure=True, httponly=True, samesite="lax")
        if PUBLIC and not visitor_subject(request):
            raw = secrets.token_urlsafe(32)
            response.set_cookie(VISITOR_COOKIE, raw + "." + digest(raw), max_age=86400 * 30, secure=True,
                                httponly=True, samesite="lax", path=PREFIX)
        return response

    @app.get("/auth/login")
    def login():
        if not PUBLIC:
            raise HTTPException(404)
        state = secrets.token_urlsafe(32)
        response = RedirectResponse(BRIDGE + "/connect?" + urlencode({"state": state}), status_code=303)
        response.set_cookie(STATE_COOKIE, state, max_age=900, secure=True, httponly=True, samesite="lax", path=PREFIX + "/auth")
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/auth/callback")
    def callback(request: Request, code: str = "", state: str = ""):
        expected = request.cookies.get(STATE_COOKIE, "")
        if not PUBLIC or not expected or not hmac.compare_digest(expected, state) or len(code) > 200:
            raise HTTPException(400, "Sign-in expired or was invalid. Please start again from SearchSpider.")
        result = bridge("exchange", {"code": code, "state": state})
        sid = secrets.token_urlsafe(32)
        with database() as con:
            con.execute("INSERT INTO sessions VALUES (?,?,?)", (digest(sid), result["token"], SESSION_EXPIRES))
        response = RedirectResponse(PREFIX + "/", status_code=303)
        response.set_cookie(COOKIE, sid, max_age=SESSION_COOKIE_AGE, secure=True, httponly=True, samesite="lax", path=PREFIX)
        response.delete_cookie(STATE_COOKIE, path=PREFIX + "/auth", secure=True, httponly=True, samesite="lax")
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.post("/api/logout")
    def logout(request: Request):
        check_origin(request)
        sid = request.cookies.get(COOKIE, "")
        token = session_token(request)
        if token:
            # Do not report success while a server-side connection remains valid.
            bridge("revoke", {"token": token})
        with database() as con:
            con.execute("DELETE FROM sessions WHERE id=?", (digest(sid),))
        from fastapi.responses import JSONResponse
        response = JSONResponse({"ok": True})
        response.headers["Cache-Control"] = "no-store"
        response.delete_cookie(COOKIE, path=PREFIX, secure=True, httponly=True, samesite="lax")
        return response


class MeteredSpider:
    """Reserve the maximum token allowance before each paid model call; reconcile success.

    Failed/uncertain calls keep the reservation, so retries never create free spending.
    Free calls share a durable $0.03/day budget.
    """
    def __init__(self, client, who):
        self.client, self.who = client, who

    def close(self):
        self.client.client.close()

    def json_call(self, prompt, schema, max_out, kind):
        if len(prompt.encode("utf-8")) > 300000:
            raise HTTPException(400, "This conversation is too long. Start a new search.")
        reservation = None
        free_reservation = None
        # UTF-8 bytes conservatively bound text token count; include the JSON schema and envelope.
        input_bound = len(prompt.encode("utf-8")) + len(json.dumps(schema).encode("utf-8")) + 4096
        if self.who["mode"] == "free":
            free_reservation = reserve_free(self.client.model, input_bound, max_out)
        if self.who["mode"] == "paid":
            reservation = bridge("reserve", {"token": self.who["token"], "input_tokens": input_bound,
                                               "output_tokens": max_out, "model": self.client.model})
        data, usage = self.client.json_call(prompt, schema, max_out, kind)
        if free_reservation and usage.get("complete") is True:
            settle_free(free_reservation, self.client.model, usage)
        if reservation and usage.get("complete") is True:
            bridge("settle", {"token": self.who["token"], "reservation": reservation["id"],
                               "input_tokens": usage["in"], "output_tokens": usage["out"] + usage.get("think", 0)})
        return data, usage
