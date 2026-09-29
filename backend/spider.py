"""Gemini client for SearchSpider. Retrieval and orchestration live in search.py / dossier.py.

The retired Session/Spider.run pipeline has been removed so there is only one app search path.
"""
import json
import os
import time
from pathlib import Path

MODEL = os.getenv("SPIDER_MODEL", "gemini-3.1-flash-lite")
ROOT = Path(__file__).resolve().parents[1]
KEY_FILES = [ROOT / ".env"]
LOG = ROOT / "logs" / "spider.jsonl"


def api_key():
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k:
        return k
    for f in KEY_FILES:
        if f.exists():
            for line in f.read_text(encoding="utf-8-sig").splitlines():
                if line.strip().startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return ""


class Spider:
    def __init__(self, client=None, key=None):
        self.client = client
        self.model = MODEL
        self.error = None
        if client is None:
            key = api_key() if key is None else key
            if not key:
                self.error = "No GEMINI_API_KEY configured."
                return
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=180000, retry_options=types.HttpRetryOptions(attempts=1)))

    @property
    def available(self):
        return self.client is not None

    def _log(self, kind, request, r):
        try:
            LOG.parent.mkdir(exist_ok=True)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "model": MODEL, "kind": kind,
                                    "request": request, "response": str(r.candidates[0].content)[:20000]
                                    if r.candidates else None}, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    @staticmethod
    def _usage(r):
        u = getattr(r, "usage_metadata", None)
        counts = [getattr(u, field, None) for field in ("prompt_token_count", "candidates_token_count")]
        return {"in": getattr(u, "prompt_token_count", 0) or 0, "out": getattr(u, "candidates_token_count", 0) or 0,
                "think": getattr(u, "thoughts_token_count", 0) or 0,
                "complete": all(type(value) is int and value >= 0 for value in counts) and counts[0] > 0}

    def json_call(self, prompt, schema, max_out, kind):
        return self._json_call(prompt, schema, max_out, kind)

    def _json_call(self, prompt, schema, max_out, kind):
        from google.genai import types
        cfg = types.GenerateContentConfig(
            temperature=0, max_output_tokens=max_out, response_mime_type="application/json", response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL))
        r = self.client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
        self._log(kind, {"prompt": prompt}, r)
        return json.loads(r.text), self._usage(r)
