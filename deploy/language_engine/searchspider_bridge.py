"""Language Engine sign-in and subscription bridge for searchspider.

Register this blueprint before db.create_all(). Uses the existing account and
API allowance database. The shared bridge secret is never sent to a browser.
"""
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
from functools import wraps
from urllib.parse import urlencode

from flask import Blueprint, abort, jsonify, redirect, render_template, request
from flask_login import current_user
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from db import AccountActionToken, ApiUsage, User, db
from config import GEMINI_MODEL_PRICING_USD_PER_1M, TIER_CAPS

searchspider_bp = Blueprint("searchspider", __name__, url_prefix="/searchspider")
CALLBACK = "https://burmeseneuralreader.com/searchspider/auth/callback"
SECRET = os.environ.get("SEARCHSPIDER_BRIDGE_SECRET", "")
# Connection tokens remain revocable, while short-lived sign-in codes still expire.
SESSION_EXPIRES = dt.datetime.max


class PaymentRequired(HTTPException):
    code = 402
    description = "An active subscription with sufficient token budget is required."


class SearchSpiderReservation(db.Model):
    __tablename__ = "searchspider_reservations"
    id = db.Column(db.String(64), primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    input_tokens = db.Column(db.Integer, nullable=False)
    output_tokens = db.Column(db.Integer, nullable=False)
    period_start = db.Column(db.Date, nullable=False)
    settled = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)


def token_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def subject(user):
    # Email identity survives account deletion/recreation, without sharing email with the reader.
    return hmac.new(SECRET.encode(), ("account:" + user.email.strip().lower()).encode(), hashlib.sha256).hexdigest()


def paid(user):
    if user.email.lower() in User.PREMIUM_EMAILS:
        return True
    sub = user.subscription
    return bool(user.is_subscribed and sub and sub.current_period_end and sub.current_period_end > dt.datetime.utcnow())


def internal(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        supplied = request.headers.get("Authorization", "")
        if len(SECRET) < 32 or not hmac.compare_digest(supplied, "Bearer " + SECRET):
            abort(403)
        if request.content_length and request.content_length > 8192:
            abort(413)
        try:
            return function(*args, **kwargs)
        except Exception:
            db.session.rollback()
            raise
    return wrapped


def data():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400)
    return payload


def load_token(raw, purpose):
    if not isinstance(raw, str) or not 20 <= len(raw) <= 200:
        abort(401)
    row = AccountActionToken.query.filter_by(token_hash=token_hash(raw), purpose=purpose).first()
    if not row or row.used_at or row.expires_at <= dt.datetime.utcnow():
        abort(401)
    user = db.session.get(User, row.user_id)
    if not user or not user.email_verified_at:
        abort(403)
    return row, user


@searchspider_bp.after_request
def private(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@searchspider_bp.get("/connect")
def connect():
    state = request.args.get("state", "")
    if len(SECRET) < 32 or not re.fullmatch(r"[A-Za-z0-9_-]{40,100}", state):
        abort(400)
    if not current_user.is_authenticated:
        return render_template("searchspider_connect.html", google_url="/auth/google/login?" + urlencode({"next": request.full_path}))
    if not current_user.email_verified_at:
        return "Verify your Language Engine email address, then sign in again.", 403
    code = secrets.token_urlsafe(32)
    db.session.add(AccountActionToken(user_id=current_user.id, purpose="searchspider_code", token_hash=token_hash(code),
                                     payload_json=json.dumps({"state": state}), expires_at=dt.datetime.utcnow() + dt.timedelta(minutes=2)))
    db.session.commit()
    return redirect(CALLBACK + "?" + urlencode({"code": code, "state": state}), code=303)


@searchspider_bp.post("/api/exchange")
@internal
def exchange():
    payload = data()
    db.session.execute(text("BEGIN IMMEDIATE"))
    row, user = load_token(payload.get("code"), "searchspider_code")
    if not hmac.compare_digest(str(json.loads(row.payload_json)["state"]), str(payload.get("state", ""))):
        abort(401)
    row.used_at = dt.datetime.utcnow()
    raw = secrets.token_urlsafe(40)
    db.session.add(AccountActionToken(user_id=user.id, purpose="searchspider_session", token_hash=token_hash(raw),
                                     expires_at=SESSION_EXPIRES))
    db.session.commit()
    return jsonify(token=raw)


@searchspider_bp.post("/api/identity")
@internal
def identity():
    row, user = load_token(data().get("token"), "searchspider_session")
    if row.expires_at != SESSION_EXPIRES:
        # Only upgrade a connection after validating its current expiry and user.
        row.expires_at = SESSION_EXPIRES
        db.session.commit()
    return jsonify(subject=subject(user), paid=paid(user))


@searchspider_bp.post("/api/revoke")
@internal
def revoke():
    raw = data().get("token")
    if not isinstance(raw, str) or not 20 <= len(raw) <= 200:
        abort(400)
    db.session.execute(text("BEGIN IMMEDIATE"))
    AccountActionToken.query.filter_by(token_hash=token_hash(raw), purpose="searchspider_session", used_at=None).update(
        {"used_at": dt.datetime.utcnow()}, synchronize_session=False)
    db.session.commit()
    return jsonify(ok=True)


def token_count(payload, field, maximum):
    value = payload.get(field)
    if type(value) is not int or not 0 <= value <= maximum:
        abort(400)
    return value


@searchspider_bp.post("/api/reserve")
@internal
def reserve():
    payload = data()
    inp = token_count(payload, "input_tokens", 310000)
    out = token_count(payload, "output_tokens", 20000)
    db.session.execute(text("BEGIN IMMEDIATE"))
    _, user = load_token(payload.get("token"), "searchspider_session")
    if not paid(user):
        raise PaymentRequired()
    model = TIER_CAPS.get(user.tier, {}).get("gemini_model")
    if payload.get("model") != model:
        abort(400)
    pricing = GEMINI_MODEL_PRICING_USD_PER_1M.get(model)
    if not pricing:
        abort(503)
    usage = ApiUsage.query.filter_by(user_id=user.id).first()
    if not usage:
        usage = ApiUsage(user_id=user.id)
        db.session.add(usage)
    usage._maybe_reset(user)
    cost = (inp * pricing["input_tokens"] + out * pricing["output_tokens"]) / 1_000_000
    if usage.llm_cost_usd(user) + cost > usage.llm_budget_usd(user):
        raise PaymentRequired()
    usage.record_llm(inp, out)
    reservation_id = secrets.token_hex(24)
    db.session.add(SearchSpiderReservation(id=reservation_id, user_id=user.id, input_tokens=inp, output_tokens=out,
                                           period_start=usage.period_start))
    db.session.commit()
    return jsonify(id=reservation_id)


@searchspider_bp.post("/api/settle")
@internal
def settle():
    payload = data()
    inp = token_count(payload, "input_tokens", 1_000_000)
    out = token_count(payload, "output_tokens", 100000)
    db.session.execute(text("BEGIN IMMEDIATE"))
    _, user = load_token(payload.get("token"), "searchspider_session")
    reservation = db.session.get(SearchSpiderReservation, str(payload.get("reservation", "")))
    if not reservation or reservation.user_id != user.id:
        abort(403)
    if not reservation.settled:
        usage = ApiUsage.query.filter_by(user_id=user.id).first()
        if not usage:
            abort(409)
        # Repeated settlement is harmless; never subtract from a different billing period.
        if usage.period_start == reservation.period_start:
            usage.llm_prompt_tokens_used = max(0, int(usage.llm_prompt_tokens_used or 0) - reservation.input_tokens) + inp
            usage.llm_output_tokens_used = max(0, int(usage.llm_output_tokens_used or 0) - reservation.output_tokens) + out
            usage.llm_tokens_used = usage.llm_prompt_tokens_used + usage.llm_output_tokens_used
        reservation.settled = True
        db.session.commit()
    return jsonify(ok=True)
