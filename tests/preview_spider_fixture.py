"""Replay a saved API result to test pagination/citations without making more Gemini calls.
Run: .venv/Scripts/python tests/preview_spider_fixture.py
The server binds only loopback. Stop it with Ctrl+C.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'logs/search_changes_live_spider.json'

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/status'):
            data=json.loads(RESULT.read_text(encoding='utf8'))
            body=json.dumps({'orders':2722,'sentences':11282,'pages':1207,'reranker':True,'spider':True,
                             'spider_model':'saved live Gemini result (UI verification)','spider_error':None,
                             'kg_triples':27129,'version':'test fixture','settings':data['settings']}).encode()
            content='application/json'
        elif self.path.startswith('/api/search'):
            body=RESULT.read_bytes();content='application/json'
        else:
            body=(ROOT/'frontend/dist/index.html').read_bytes();content='text/html; charset=utf-8'
        self.send_response(200);self.send_header('Content-Type',content);self.end_headers();self.wfile.write(body)
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length',0)))
        body=RESULT.read_bytes()
        self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.end_headers();self.wfile.write(body)

if __name__=='__main__':
    print('Fixture preview: http://127.0.0.1:8766',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
