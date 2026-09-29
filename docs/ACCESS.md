# Hosted access and AI budgets

This describes the access-control implementation in the public source snapshot. The historical retrieval scores do not validate the payment integration or production deployment.

## Access modes

| Mode | Token budget | Initial search settings |
|---|---|---|
| Basic search | No model call; freely available | Local hybrid retrieval |
| Anonymous sponsored use | One **shared USD $0.03/day** pool, first come, first served | At most 50 assessed documents; 500-token answer |
| Language Engine Pro | Existing account's shared monthly token budget | 0–4 additional 50-document rounds; up to 250 initially assessed documents |

The sponsored pool resets at **00:00 UTC**. It covers sponsored SearchSpider model calls, including follow-up questions and deeper reading. It is not a per-person daily quota and does not cap unrelated calls elsewhere on the same Google account or key. At that configured allowance, 31 daily pools total $0.93.

The answer budget is derived on the server: **500 × (1 + additional reading rounds)**, up to 2,500 tokens. Sponsored initial searches are forced to zero additional rounds regardless of the request body. Subsequent "Push deeper" actions each assess up to 50 more documents and use the caller's current allowance. No separate response-length setting is accepted.

## How the sponsored cap is enforced

Before each provider call, `MeteredSpider` obtains a durable reservation in SQLite. `BEGIN IMMEDIATE` serializes reservations, including competing workers sharing the same database. The reservation includes a conservative UTF-8 byte bound for the input prompt/schema plus an envelope allowance and the requested output-token cap. Integer nanodollars avoid monetary rounding drift.

The checked standard text prices for `gemini-3.1-flash-lite` are **$0.25 per million input tokens and $1.50 per million output tokens, including thinking**. These prices are encoded in `backend/access.py`; a model without a configured price is rejected for sponsored use. Operators must update the table if the provider changes its prices. [Google pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite), checked 28 September 2026.

Successful calls reconcile the reservation with complete, valid reported input, output and thinking usage. Failed calls and responses with missing or invalid usage retain their reservation. SDK automatic retries are disabled; explicit retries require another reservation. A positive displayed balance is not a promise that the next whole call fits its maximum reservation. When the allowance cannot cover another call, the application stops model work and preserves retrieved documents and completed assessments where available.

The budget state must remain on persistent storage across service restarts and deployments. Deleting or rolling back the database resets accounting. This is an application-level budget for the configured request paths and prices, not a Google billing-account spending limit.

## Hosted connections and cached runs

The hosted application accepts sponsored use or a connected Language Engine account. Personal API-key requests are rejected server-side.

Run ownership uses an account subject or a signed HttpOnly browser cookie. Follow-up and deeper-reading requests are authorized again; provider clients are closed after each request and removed from cached dossiers.

Research prompts and model responses can be recorded in server-side research logs. Those logs are excluded from this repository. Operators should protect them as user research content; "key not logged" does not mean queries are never logged.

## Language Engine bridge

The [Flask blueprint](../deploy/language_engine/searchspider_bridge.py) and [connection template](../deploy/language_engine/searchspider_connect.html) integrate with the existing Language Engine account application. They are an adapter, not a standalone identity or payment service.

The flow uses a browser state nonce and a short-lived, single-use connection code. The backend exchanges that code for a session token through an authenticated server-to-server request. The shared bridge secret remains on the two servers; the browser receives an opaque secure session cookie.

Connections have no routine server-side expiry. A Secure, HttpOnly, SameSite=Lax cookie is renewed for 400 days each time access is checked; browser cookie deletion or browser-imposed retention limits can still require reconnection. Existing valid seven-day connections are upgraded on their next visit. Expired or revoked connections are never revived.

Disconnect revokes the connection token on Language Engine, deletes the local session, and clears the browser cookie. A temporary bridge outage preserves the connection and reports failed disconnect attempts. Connection state is separate from subscription status: free accounts remain connected, while paid requests always recheck entitlement and budget.

The bridge checks a verified account and active subscription. Before each paid model call it reserves against the existing `ApiUsage` allowance, then reconciles actual usage once. It validates the configured model against the account tier. Settlement is idempotent and does not subtract a reservation from a different billing period.

The bridge integrates with Language Engine's `User`, `ApiUsage`, `AccountActionToken`, `TIER_CAPS` and pricing definitions. Its exchange, identity, revocation, reservation and settlement endpoints authenticate server-to-server requests with a shared secret.

## Deployment configuration

The deployment uses public access controls, HTTPS, a shared bridge secret and persistent SQLite budget accounting. The backend runs on a private listener behind the reverse proxy.

Public mode disables the model-calling evaluation endpoints. Same-origin checks guard cookie-authenticated writes, responses are marked `no-store`, and a process-level semaphore bounds concurrent searches. These controls are separate from the dollar ledger; the ledger is what prevents additional sponsored reservations after the daily allowance is allocated.

The regression suite covers concurrent budget claims, retained reservations after failures, idempotent settlement, UTC rollover, sponsored settings despite forged request bodies, run ownership, cross-origin rejection, persistent connections across restarts and time advances, revocation, rejected personal keys, and disabled evaluation endpoints. Provider responses, reverse-proxy behavior and the live account service also require deployment verification.
