"""Public access and budget checks use fake clients; no external AI calls."""
import importlib.util
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
import access
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from test_search_pipeline import fixture, MockGemini


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = patch.object(access, 'STATE_DB', Path(self.tmp.name) / 'access.sqlite3')
        self.p.start()
        with access.database():
            pass

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_concurrent_callers_cannot_exceed_daily_cap(self):
        def reserve(_):
            try:
                return access.reserve_free('gemini-3.1-flash-lite', 4000, 0)
            except access.BudgetExhausted:
                return None
        with ThreadPoolExecutor(12) as ex:
            reservations = list(ex.map(reserve, range(50)))
        self.assertEqual(sum(bool(r) for r in reservations), 30)
        self.assertEqual(access.free_remaining(), 0)
        with self.assertRaises(access.BudgetExhausted):
            access.reserve_free('gemini-3.1-flash-lite', 1, 0)

    def test_actual_tokens_refund_once_and_include_thinking(self):
        r = access.reserve_free('gemini-3.1-flash-lite', 10000, 1000)
        usage = {'in': 2000, 'out': 100, 'think': 100}
        access.settle_free(r, 'gemini-3.1-flash-lite', usage)
        self.assertEqual(access.free_remaining(), 30_000_000 - 800000)
        access.settle_free(r, 'gemini-3.1-flash-lite', {'in': 0, 'out': 0})
        self.assertEqual(access.free_remaining(), 30_000_000 - 800000)

    def test_restart_retains_spend_and_utc_day_resets_allowance(self):
        with patch.object(access, 'day', return_value='2026-09-28'):
            access.reserve_free('gemini-3.1-flash-lite', 120000, 0)
            self.assertEqual(access.free_remaining(), 0)
        with patch.object(access, 'day', return_value='2026-09-29'):
            self.assertEqual(access.free_remaining(), 30_000_000)

    def test_failed_model_call_keeps_reservation_and_unknown_price_blocks(self):
        client = SimpleNamespace(model='gemini-3.1-flash-lite', json_call=lambda *a: (_ for _ in ()).throw(RuntimeError('failed')))
        with self.assertRaises(RuntimeError):
            access.MeteredSpider(client, {'mode': 'free'}).json_call('prompt', {}, 500, 'answer')
        self.assertLess(access.free_remaining(), 30_000_000)
        with self.assertRaises(HTTPException):
            access.reserve_free('unknown-model', 1000, 500)

    def test_paid_calls_use_subscription_budget_and_reconcile(self):
        client = SimpleNamespace(model='gemini-3.1-flash-lite', json_call=lambda *a: ({}, {'in': 100, 'out': 20, 'complete': True}))
        with patch.object(access, 'bridge', return_value={'id': 'reservation'}) as bridge:
            access.MeteredSpider(client, {'mode': 'paid', 'token': 'session'}).json_call('prompt', {}, 500, 'answer')
        self.assertEqual([c.args[0] for c in bridge.call_args_list], ['reserve', 'settle'])
        self.assertEqual(bridge.call_args.args[1]['input_tokens'], 100)
        self.assertEqual(access.free_remaining(), 30_000_000)

    def test_missing_usage_keeps_full_reservation(self):
        from spider import Spider
        missing = Spider._usage(SimpleNamespace(usage_metadata=None))
        self.assertFalse(missing['complete'])
        client = SimpleNamespace(model='gemini-3.1-flash-lite',json_call=lambda *a: ({'answer':'ok'},missing))
        access.MeteredSpider(client, {'mode':'free'}).json_call('prompt',{},500,'answer')
        self.assertLess(access.free_remaining(),30_000_000)
        with access.database() as con:
            self.assertEqual(con.execute('SELECT settled FROM free_spend').fetchone()[0],0)


class PublicApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(access, 'STATE_DB', Path(self.tmp.name)/'state.sqlite3'),
                        patch.object(access, 'PUBLIC', True), patch.object(access, 'SECRET', 'test-secret-' * 6),
                        patch.object(access, 'PREFIX', '')]
        for p in self.patches: p.start()
        self.sp = MockGemini(); self.sp.available = True; self.sp.error = None
        e = fixture(150)
        e.kg = SimpleNamespace(ok=True, triples=[], ecat_name={}, rel_name={}, expand_query=lambda q:([],[]))
        spec = importlib.util.spec_from_file_location('access_test_app', ROOT/'backend/app.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch('search.Engine', return_value=e), patch('spider.Spider', return_value=self.sp):
            spec.loader.exec_module(self.module)
        self.client = TestClient(self.module.app, base_url='https://burmeseneuralreader.com')
        self.client.get('/api/access')
        self.sp.close = lambda: None
        self.patches.extend([patch.object(self.module, 'model_for', return_value=self.sp), patch('dossier.RUN_LOGS', Path(self.tmp.name)/'logs')])
        for p in self.patches[-2:]: p.start()

    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def test_free_forces_zero_rounds_and_500_tokens_despite_forged_body(self):
        result = self.client.post('/api/spider', json={'q':'royal evidence','reading_rounds':4,'response_tokens':10000})
        self.assertEqual(result.status_code, 200, result.text)
        body = result.json()
        self.assertEqual((body['reading_rounds'],body['response_tokens']), (0,500))
        self.assertEqual(sum(body['counts'][s] for s in ('relevant','partial','dropped','missing')), 50)
        d,_ = self.module.RUNS[body['run_id']]
        self.assertIsNone(d.sp)
        # Free follow-ups are permitted, but cached sessions cannot be taken by another browser.
        follow = self.client.post('/api/spider/followup', json={'run_id':body['run_id'],'q':'why?'})
        self.assertEqual(follow.status_code,200)
        other = TestClient(self.module.app,base_url='https://burmeseneuralreader.com')
        other.get('/api/access')
        self.assertEqual(other.post('/api/spider/followup',json={'run_id':body['run_id'],'q':'why?'}).status_code,403)
        other.close()

    def test_paid_maximum_is_server_controlled_and_personal_keys_are_rejected(self):
        r = self.client.post('/api/spider',headers={'X-Gemini-API-Key':'test-key-with-at-least-twenty-characters'},json={'q':'royal evidence','reading_rounds':4})
        self.assertEqual(r.status_code,403)
        with patch.object(access, 'identity', return_value={'mode':'paid','subject':'paid-account','token':'session'}):
            r = self.client.post('/api/spider',json={'q':'royal evidence','reading_rounds':4,'response_tokens':99999})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['response_tokens'],2500)
        self.assertEqual(sum(r.json()['counts'][s] for s in ('relevant','partial','dropped','missing')),250)
        self.assertEqual(self.client.post('/api/spider',json={'q':'q','reading_rounds':5}).status_code,422)

    def test_eval_and_cross_origin_spending_are_blocked(self):
        self.assertEqual(self.client.post('/api/plan',json={'q':'q'}).status_code,404)
        self.assertEqual(self.client.post('/api/eval_shortlist?tid=x').status_code,404)
        self.assertEqual(self.client.post('/api/spider',headers={'Origin':'https://untrusted.example'},json={'q':'q'}).status_code,403)
        self.assertEqual(self.client.get('/auth/callback?code=wrong&state=wrong').status_code,400)

    def test_budget_stop_keeps_basic_results(self):
        with patch.object(self.sp,'json_call',side_effect=access.BudgetExhausted(429,'Daily budget reached')):
            r=self.client.post('/api/spider',json={'q':'royal evidence'})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['stop'],'budget')
        self.assertGreater(len(r.json()['orders']),0)


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(access, 'STATE_DB', Path(self.tmp.name)/'state.sqlite3'),
                        patch.object(access, 'PUBLIC', True), patch.object(access, 'SECRET', 'session-test-' * 6),
                        patch.object(access, 'PREFIX', '/searchspider')]
        for p in self.patches:
            p.start()
        self.app = self.make_app()
        self.client = TestClient(self.app, base_url='https://burmeseneuralreader.com')

    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def make_app(self):
        root = FastAPI()
        inner = FastAPI()
        access.install_routes(inner)
        root.mount('/searchspider', inner)
        return root

    def connect(self):
        self.client.get('/searchspider/auth/login', follow_redirects=False)
        state = self.client.cookies.get(access.STATE_COOKIE)
        with patch.object(access, 'bridge', return_value={'token':'persistent-bridge-token'}):
            response = self.client.get('/searchspider/auth/callback', params={'code':'valid-code','state':state}, follow_redirects=False)
        self.assertEqual(response.status_code,303,response.text)
        return response, self.client.cookies.get(access.COOKIE)

    def test_connection_survives_time_and_restart_and_cookie_is_renewed(self):
        response, sid = self.connect()
        self.assertEqual(response.status_code,303,response.text)
        cookie=response.headers['set-cookie']
        for flag in ['HttpOnly','Secure','SameSite=lax','Max-Age=34560000']:
            self.assertIn(flag,cookie)
        with access.database() as con:
            self.assertEqual(con.execute('SELECT expires FROM sessions').fetchone()[0],access.SESSION_EXPIRES)
        self.client.close()
        self.client=TestClient(self.make_app(),base_url='https://burmeseneuralreader.com')
        with patch.object(access.time,'time',return_value=access.time.time()+86400*300), \
             patch.object(access,'bridge',return_value={'paid':True,'subject':'account'}):
            response=self.client.get('/searchspider/api/access',headers={'Cookie':access.COOKIE+'='+sid})
        self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(response.json()['connected'])
        self.assertEqual(response.json()['mode'],'paid')
        self.assertIn('Max-Age=34560000',response.headers['set-cookie'])
        self.assertEqual(response.headers['cache-control'],'no-store')

    def seed_connection(self, expires=None):
        sid='opaque-browser-session'
        with access.database() as con:
            con.execute('INSERT INTO sessions VALUES (?,?,?)',(access.digest(sid),'persistent-bridge-token',expires or access.SESSION_EXPIRES))
        return {'Cookie':access.COOKIE+'='+sid}

    def test_legacy_connection_is_upgraded_and_free_account_remains_connected(self):
        headers=self.seed_connection(int(access.time.time())+86400)
        with patch.object(access,'bridge',return_value={'paid':False,'subject':'account'}):
            response=self.client.get('/searchspider/api/access',headers=headers)
        self.assertTrue(response.json()['connected'])
        self.assertEqual(response.json()['mode'],'free')
        with access.database() as con:
            self.assertEqual(con.execute('SELECT expires FROM sessions').fetchone()[0],access.SESSION_EXPIRES)

    def test_disconnect_revokes_bridge_and_replayed_cookie_cannot_reconnect(self):
        headers=self.seed_connection()
        with patch.object(access,'bridge',return_value={'ok':True}) as bridge:
            response=self.client.post('/searchspider/api/logout',headers=headers)
            bridge.assert_called_once_with('revoke',{'token':'persistent-bridge-token'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertIn('Max-Age=0',response.headers['set-cookie'])
        with patch.object(access,'bridge',side_effect=AssertionError('Disconnected token must not be used')):
            self.assertFalse(self.client.get('/searchspider/api/access',headers=headers).json()['connected'])
            self.assertEqual(self.client.post('/searchspider/api/logout',headers=headers).status_code,200)

    def test_outage_does_not_silently_disconnect_but_revoked_session_is_cleared(self):
        headers=self.seed_connection()
        with patch.object(access,'bridge',side_effect=HTTPException(503,'temporary outage')):
            self.assertEqual(self.client.get('/searchspider/api/access',headers=headers).status_code,503)
            self.assertEqual(self.client.post('/searchspider/api/logout',headers=headers).status_code,503)
        with access.database() as con:
            self.assertEqual(con.execute('SELECT COUNT(*) FROM sessions').fetchone()[0],1)
        with patch.object(access,'bridge',side_effect=HTTPException(401,'revoked')):
            response=self.client.get('/searchspider/api/access',headers=headers)
        self.assertFalse(response.json()['connected'])
        self.assertIn('Max-Age=0',response.headers['set-cookie'])
        with access.database() as con:
            self.assertEqual(con.execute('SELECT COUNT(*) FROM sessions').fetchone()[0],0)

    def test_cross_origin_disconnect_and_expired_legacy_session_are_rejected(self):
        headers=self.seed_connection(int(access.time.time())-60)
        with patch.object(access,'bridge',side_effect=AssertionError('Must not call bridge')):
            self.assertEqual(self.client.post('/searchspider/api/logout',headers={**headers,'Origin':'https://untrusted.example'}).status_code,403)
            self.assertFalse(self.client.get('/searchspider/api/access',headers=headers).json()['connected'])


if __name__ == '__main__':
    unittest.main()
