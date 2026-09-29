"""Run beside searchspider_bridge.py with Language Engine's dependencies available.
Creates an isolated temporary SQLite database, never touches production accounts.
"""
import datetime as dt
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ['SEARCHSPIDER_BRIDGE_SECRET'] = 'isolated-bridge-test-' * 3
from flask import Flask
from flask_login import LoginManager
from db import db, User, Subscription, ApiUsage, AccountActionToken
import searchspider_bridge as bridge


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = Flask(__name__)
        self.app.config.update(TESTING=True, SECRET_KEY='test', SQLALCHEMY_DATABASE_URI='sqlite:///' + self.tmp.name + '/test.db')
        db.init_app(self.app)
        self.app.register_blueprint(bridge.searchspider_bp)
        lm = LoginManager(self.app)
        lm.user_loader(lambda uid: db.session.get(User, int(uid)))
        self.context = self.app.app_context(); self.context.push()
        db.create_all()
        user = User(email='test@example.invalid', username='test@example.invalid', email_verified_at=dt.datetime.utcnow())
        db.session.add(user); db.session.flush(); self.uid = user.id
        db.session.add(Subscription(user_id=user.id,status='active',current_period_end=dt.datetime.utcnow()+dt.timedelta(days=10)))
        db.session.add(AccountActionToken(user_id=user.id,purpose='searchspider_session',token_hash=bridge.token_hash('test-session-token-long-enough'),expires_at=dt.datetime.utcnow()+dt.timedelta(days=7)))
        db.session.commit()
        self.client = self.app.test_client()
        self.headers={'Authorization':'Bearer '+bridge.SECRET}

    def tearDown(self):
        db.session.remove(); db.engine.dispose(); self.context.pop(); self.tmp.cleanup()

    def post(self, action, **payload):
        return self.client.post('/searchspider/api/'+action, headers=self.headers,json={'token':'test-session-token-long-enough',**payload})

    def test_paid_identity_reservation_and_idempotent_settlement(self):
        r=self.post('identity'); self.assertTrue(r.json['paid'])
        r=self.post('reserve',input_tokens=10000,output_tokens=1000,model='gemini-3.1-flash-lite')
        self.assertEqual(r.status_code,200,r.data); reservation=r.json['id']
        db.session.expire_all(); usage=ApiUsage.query.filter_by(user_id=self.uid).one()
        self.assertEqual(usage.llm_prompt_tokens_used,10000)
        r=self.post('settle',reservation=reservation,input_tokens=2000,output_tokens=100)
        self.assertEqual(r.status_code,200,r.data)
        self.post('settle',reservation=reservation,input_tokens=0,output_tokens=0)
        db.session.expire_all(); usage=ApiUsage.query.filter_by(user_id=self.uid).one()
        self.assertEqual((usage.llm_prompt_tokens_used,usage.llm_output_tokens_used),(2000,100))

    def test_free_expired_and_exhausted_accounts_fail_closed(self):
        sub=Subscription.query.filter_by(user_id=self.uid).one();sub.status='canceled';db.session.commit()
        self.assertFalse(self.post('identity').json['paid'])
        self.assertEqual(self.post('reserve',input_tokens=1000,output_tokens=500,model='gemini-3.1-flash-lite').status_code,402)
        db.session.rollback();sub=Subscription.query.filter_by(user_id=self.uid).one();sub.status='active';sub.current_period_end=dt.datetime.utcnow()-dt.timedelta(days=1);db.session.commit()
        self.assertFalse(self.post('identity').json['paid'])
        sub.current_period_end=dt.datetime.utcnow()+dt.timedelta(days=10);db.session.add(ApiUsage(user_id=self.uid,llm_prompt_tokens_used=8000000));db.session.commit()
        self.assertEqual(self.post('reserve',input_tokens=1000,output_tokens=500,model='gemini-3.1-flash-lite').status_code,402)

    def test_code_exchange_is_one_use_and_state_bound(self):
        code='single-use-code-long-enough'
        db.session.add(AccountActionToken(user_id=self.uid,purpose='searchspider_code',token_hash=bridge.token_hash(code),payload_json=json.dumps({'state':'expected'}),expires_at=dt.datetime.utcnow()+dt.timedelta(minutes=2)));db.session.commit()
        self.assertEqual(self.post('exchange',code=code,state='wrong').status_code,401)
        db.session.rollback()
        self.assertEqual(self.post('exchange',code=code,state='expected').status_code,200)
        self.assertEqual(self.post('exchange',code=code,state='expected').status_code,401)

    def test_internal_key_required(self):
        r=self.client.post('/searchspider/api/identity',json={'token':'test-session-token-long-enough'})
        self.assertEqual(r.status_code,403)

    def test_connected_session_survives_old_deadline_but_subscription_is_rechecked(self):
        self.assertTrue(self.post('identity').json['paid'])
        db.session.expire_all()
        row=AccountActionToken.query.filter_by(purpose='searchspider_session').one()
        self.assertEqual(row.expires_at,bridge.SESSION_EXPIRES)
        db.session.remove()
        future=dt.datetime.utcnow()+dt.timedelta(days=300)
        clock=SimpleNamespace(datetime=SimpleNamespace(utcnow=lambda:future))
        with patch.object(bridge,'dt',clock):
            response=self.post('identity')
            self.assertEqual(response.status_code,200,response.data)
            self.assertFalse(response.json['paid'])
            self.assertEqual(self.post('reserve',input_tokens=1000,output_tokens=500,model='gemini-3.1-flash-lite').status_code,402)

    def test_disconnect_is_idempotent_and_revokes_identity_and_spending(self):
        self.post('identity')
        self.assertEqual(self.post('revoke').status_code,200)
        self.assertEqual(self.post('revoke').status_code,200)
        self.assertEqual(self.post('identity').status_code,401)
        self.assertEqual(self.post('reserve',input_tokens=1000,output_tokens=500,model='gemini-3.1-flash-lite').status_code,401)

    def test_expired_and_revoked_legacy_sessions_are_never_upgraded(self):
        row=AccountActionToken.query.filter_by(purpose='searchspider_session').one()
        row.expires_at=dt.datetime.utcnow()-dt.timedelta(seconds=1)
        db.session.commit()
        self.assertEqual(self.post('identity').status_code,401)
        db.session.rollback()
        row=AccountActionToken.query.filter_by(purpose='searchspider_session').one()
        row.expires_at=bridge.SESSION_EXPIRES;row.used_at=dt.datetime.utcnow();db.session.commit()
        self.assertEqual(self.post('identity').status_code,401)

    def test_new_connection_is_persistent_but_codes_still_expire(self):
        code='new-session-connection-code'
        db.session.add(AccountActionToken(user_id=self.uid,purpose='searchspider_code',token_hash=bridge.token_hash(code),payload_json=json.dumps({'state':'expected'}),expires_at=dt.datetime.utcnow()+dt.timedelta(minutes=2)))
        expired='expired-connection-code'
        db.session.add(AccountActionToken(user_id=self.uid,purpose='searchspider_code',token_hash=bridge.token_hash(expired),payload_json=json.dumps({'state':'expected'}),expires_at=dt.datetime.utcnow()-dt.timedelta(minutes=1)))
        db.session.commit()
        self.assertEqual(self.post('exchange',code=expired,state='expected').status_code,401)
        response=self.post('exchange',code=code,state='expected')
        self.assertEqual(response.status_code,200,response.data)
        row=AccountActionToken.query.filter_by(token_hash=bridge.token_hash(response.json['token'])).one()
        self.assertEqual(row.expires_at,bridge.SESSION_EXPIRES)

    def test_revocation_cannot_revoke_other_purposes_or_unauthenticated_requests(self):
        raw='not-a-searchspider-session'
        db.session.add(AccountActionToken(user_id=self.uid,purpose='password_reset',token_hash=bridge.token_hash(raw),expires_at=dt.datetime.utcnow()+dt.timedelta(hours=1)))
        db.session.commit()
        self.assertEqual(self.post('revoke',token=raw).status_code,200)
        self.assertIsNone(AccountActionToken.query.filter_by(token_hash=bridge.token_hash(raw)).one().used_at)
        self.assertEqual(self.client.post('/searchspider/api/revoke',json={'token':'test-session-token-long-enough'}).status_code,403)


if __name__=='__main__': unittest.main()
