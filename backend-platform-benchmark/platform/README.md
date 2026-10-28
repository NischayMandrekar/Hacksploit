# Local Backend Platform

A compact, domain-agnostic backend foundation: Nginx reverse proxy + FastAPI modular monolith + independent worker + PostgreSQL. It accepts jobs durably, executes only explicitly registered bounded handlers, and recovers work after worker failure. Optional RabbitMQ wake-ups never replace PostgreSQL polling. Kafka lifecycle events are best-effort (there is intentionally no transactional outbox or exactly-once claim).

## What is implemented

- Email/password registration and login with Argon2id password hashes, short-lived JWT access tokens, rotating opaque refresh tokens, refresh-reuse detection, logout and logout-all.
- User profiles, user/admin roles, a master-key bootstrap route to promote the first administrator, and admin user/role endpoints.
- User-managed API keys plus bootstrap-issued service API keys; API key hashes only are stored, raw key is shown once.
- Nginx is the only published API entry point by default; FastAPI is internal to the Compose network. Nginx applies request-size limits, route-level rate limits, proxy timeouts and safe forwarded headers.
- Scopes and per-key job ownership checks.
- Durable job submission (`202`), status, queued/running cancellation, bounded inputs/outputs and timeouts.
- Idempotency keys scoped to API key; same key/same normalized request returns the same job; same key/different request returns `409`.
- Separate worker process, PostgreSQL `FOR UPDATE SKIP LOCKED` claims, random per-claim lease token, bounded concurrency, lease recovery, retry backoff and stale-worker result rejection.
- Built-in safe handler allow-list: `demo.echo` and `demo.compute`. No `eval`, shell execution, dynamic imports, or user-provided code execution.
- JSON logs, request IDs, `/health`, `/ready`, `/metrics`, Prometheus and Grafana provisioning.
- Optional RabbitMQ wake-up queue, Redis-backed atomic fixed-window rate limiter, and best-effort Kafka job lifecycle events.
- Alembic migration and PostgreSQL integration tests.

## Milestones

1. **Foundation:** package, configuration, Compose, PostgreSQL schema/migrations, health endpoints and API/worker processes.
2. **Secure API + identity:** API keys, Argon2id password hashing, JWT access tokens, rotating refresh sessions, user/admin roles, scopes, ownership and idempotency.
3. **Worker reliability:** bounded handlers, row locking, leases, retries, cancellation and recovery.
4. **Operations + ingress:** Nginx entry point, metrics/logging, optional adapters and isolated integration-test stack.

The repository contains the first end-to-end implementation of these milestones. Run the checks below before relying on it for a real workload.

## Quick start

Requires Docker Engine and Docker Compose v2. From this directory:

```bash
cp .env.example .env
# Replace both APP_MASTER_KEY and AUTH_JWT_SECRET with independent outputs from `openssl rand -hex 32`. Do not keep the placeholders.
docker compose up --build -d
```

The default profile starts PostgreSQL, runs the one-shot migration service, then starts API, Nginx and worker. The database is persisted in a named volume. Only Nginx is published, bound to `127.0.0.1:8080`; FastAPI and PostgreSQL have no published host ports. Before starting, set independent random values for `APP_MASTER_KEY` and `AUTH_JWT_SECRET` in `.env` (for example, generate each with `openssl rand -hex 32`).

Check health:

```bash
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8080/ready
```

`/health` is liveness. `/ready` checks PostgreSQL readiness.

## User authentication demo

The default Compose entry point is `http://localhost:8080`. Set independent secrets in `.env` before startup:

```bash
openssl rand -hex 32  # APP_MASTER_KEY
openssl rand -hex 32  # AUTH_JWT_SECRET; use a different value
```

Register and log in:

```bash
curl -sS -X POST http://localhost:8080/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"demo@example.com","password":"a-long-password-for-local-demo"}'

curl -sS -X POST http://localhost:8080/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"demo@example.com","password":"a-long-password-for-local-demo"}'
```

The login response contains a short-lived bearer access token and a rotating refresh token. Use the access token with `Authorization: Bearer ...`; store refresh tokens securely and never log them. Refresh-token reuse revokes the token family. Access tokens are short-lived (default 10 minutes), so logout revokes refresh sessions but an already-issued access token may remain valid until it expires.

The first administrator is provisioned explicitly using the bootstrap master key. Register a user, copy its ID, then:

```bash
curl -sS -X POST "http://localhost:8080/api/v1/admin/bootstrap/USER_ID/role" \
  -H "X-Master-Key: $MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"role":"admin"}'
```

Do not expose the bootstrap master key to ordinary clients. After bootstrap, use admin-authenticated endpoints for routine role management.

## End-to-end demo

Create an API key using the bootstrap master key. **The raw API key is returned only once.**

```bash
export MASTER_KEY='the-value-you-set-in-.env'
curl -sS -X POST http://localhost:8080/api/v1/api-keys \
  -H "X-Master-Key: $MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"local-demo","scopes":["jobs:create","jobs:read","jobs:cancel"]}'
```

Copy `raw_key` from the response, then:

```bash
export API_KEY='paste-raw-key-here'
curl -sS -X POST http://localhost:8080/api/v1/jobs \
  -H "X-API-Key: $API_KEY" -H 'Idempotency-Key: demo-001' -H 'Content-Type: application/json' \
  -d '{"job_type":"demo.compute","input":{"operation":"sum","values":[10,20,30]}}'
```

Copy the returned `id`, then poll it:

```bash
curl -sS "http://localhost:8080/api/v1/jobs/JOB_ID" -H "X-API-Key: $API_KEY"
```

Expected result after worker execution: `status: succeeded`, result `{"operation":"sum","count":3,"result":60}`. Submit the same request with the same idempotency key to receive the same job. Reusing the key for a different payload returns `409`.

Other useful commands:

```bash
docker compose logs -f api worker
docker compose ps
docker compose down                 # preserve local data volume
docker compose down -v              # destructive: remove database and other volumes
```

## Optional profiles

Profiles are additive. `full` adds RabbitMQ, Redis, Prometheus and Grafana; `streaming` adds Kafka. To make adapters active, set the corresponding **container-network** URLs in `.env` before starting the profiles:

```dotenv
RABBITMQ_URL=amqp://platform:platform_dev_only@rabbitmq:5672/%2F
REDIS_URL=redis://redis:6379/0
KAFKA_BOOTSTRAP=kafka:9092
```

Then start the chosen services:

```bash
docker compose --profile full up --build -d
docker compose --profile streaming up --build -d
# Both profiles can be combined:
docker compose --profile full --profile streaming up --build -d
```

The `full` profile exposes Prometheus on `127.0.0.1:9090` and Grafana on `127.0.0.1:3000` (local-only demo credentials: `admin` / `local_only_change_me`; change them before any shared environment). There are no authored Grafana dashboards. RabbitMQ messages are only wake-up hints; the worker always polls PostgreSQL. Kafka events may be lost or duplicated because no outbox is implemented. Do not treat Kafka events as an audit trail.

## API overview

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/auth/register` | Public + rate limited | Create a user account |
| `POST` | `/api/v1/auth/login` | Public + rate limited | Issue access and refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh token + rate limited | Rotate refresh token and issue access token |
| `POST` | `/api/v1/auth/logout` | Refresh token | Revoke the presented refresh token |
| `POST` | `/api/v1/auth/logout-all` | Bearer JWT | Revoke all refresh sessions for the current user |
| `POST` | `/api/v1/auth/change-password` | Bearer JWT + current password | Change password and revoke refresh sessions |
| `GET`, `PATCH` | `/api/v1/users/me` | Bearer JWT | Read/update own profile email |
| `GET` | `/api/v1/users/me/sessions` | Bearer JWT | List paginated session-family metadata |
| `DELETE` | `/api/v1/users/me/sessions/{family_id}` | Bearer JWT | Revoke one owned session family |
| `POST`, `GET` | `/api/v1/users/me/api-keys` | Bearer JWT | Issue/list own API keys; raw key shown once on issue |
| `DELETE` | `/api/v1/users/me/api-keys/{id}` | Bearer JWT | Revoke an owned API key |
| `POST` | `/api/v1/admin/bootstrap/{user_id}/role` | `X-Master-Key` | Explicit first-admin bootstrap / emergency role recovery |
| `GET` | `/api/v1/admin/users` | Admin JWT | List users (maximum 200) |
| `PATCH` | `/api/v1/admin/users/{id}/role` | Admin JWT | Change a user's role |
| `POST` | `/api/v1/api-keys` | `X-Master-Key` | Issue a service key; raw key shown once |
| `DELETE` | `/api/v1/api-keys/{id}` | `X-Master-Key` | Revoke a service key |
| `POST` | `/api/v1/jobs` | Bearer JWT or API key with `jobs:create` | Submit a durable job |
| `GET` | `/api/v1/jobs` | Bearer JWT or API key with `jobs:read` | List owned jobs with bounded pagination |
| `GET` | `/api/v1/jobs/{id}` | Bearer JWT or API key with `jobs:read` | Read an owned job |
| `POST` | `/api/v1/jobs/{id}/cancel` | Bearer JWT or API key with `jobs:cancel` | Cancel owned queued work or request cooperative cancellation |
| `GET` | `/health` | None | Liveness through Nginx |
| `GET` | `/ready` | None | Database readiness through Nginx |
| `GET` | `/metrics` | None | Prometheus metrics through the local proxy |

Errors use `{ "error": { "code": "...", "message": "...", "request_id": "..." } }`. Job lookups that are unknown or owned by another identity both return `404` to avoid leaking existence. API keys remain scoped to their own submitted jobs; a user can see jobs created through that user's account-owned API keys.

## Testing

Tests require a **dedicated disposable PostgreSQL database**; SQLite is deliberately not used as a substitute for PostgreSQL locking and partial-index behavior.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[all]'
export TEST_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@localhost:5432/platform_test'
pytest -q
```

The suite includes unit checks for Argon2id password hashing, email normalization, refresh-token hashing, safe compute overflow handling and integrity-constraint classification. The PostgreSQL-only integration suite covers registration/login, case-insensitive email uniqueness, refresh-token rotation/reuse detection, JWT profile access, user-owned jobs and API keys, API-key master authentication, scopes, revocation/expiry, ownership, sequential and concurrent HTTP idempotency, rate-limit rejection, queued cancellation, concurrent worker claims, lease recovery/fencing, deadline handling, and partial idempotency indexes. These tests must run against a dedicated disposable PostgreSQL database; do not point `TEST_DATABASE_URL` at a database containing data you need.

### Run the complete suite in an isolated Docker test database

A separate test Compose file starts a disposable PostgreSQL 17 container, runs `alembic upgrade head` against it, then runs the suite. It does not use the development database volume:

```bash
docker compose -f compose.test.yaml up --build --abort-on-container-exit --exit-code-from tests
docker compose -f compose.test.yaml down --remove-orphans
```

The test database is disposable. The command's exit status comes from pytest, so a failing test makes the command fail. The test stack has no published ports and no persistent database volume.

## Reliability boundaries

- The database commit is the acceptance boundary. A client can safely retry a submission if it supplies an idempotency key.
- Execution is **at least once**, not exactly once. A process can perform a side effect and crash before recording success. Future handlers with external side effects must use downstream idempotency keys.
- A lease prevents an old worker from committing a result after reclaim, but cannot undo side effects already performed by that worker.
- Cancellation is cooperative. Python threads cannot be safely force-killed; the included handlers check cancellation at bounded points.
- Redis rate limiting is shared/atomic when Redis is configured and reachable; without Redis, the fallback is process-local and is not a global quota. Redis outage fails open for rate limiting but never bypasses authentication.
- Master key is bootstrap administration, not a user credential. Do not use a shared default key in a network-accessible deployment.
- PostgreSQL development credentials and Grafana credentials are local-only examples. Change them before any shared deployment.

## Architecture guardrails

PostgreSQL owns user identity, refresh-session history and job state. Nginx is the only host-published application entry point in the default profile. Passwords use Argon2id; JWT access tokens are signed with HS256 and validated for issuer, audience, expiry and token type; refresh tokens are opaque, stored as hashes, rotated on use, and family-revoked on reuse. Access tokens remain valid until expiry after logout, so the default short TTL is an explicit trade-off. Email verification, password-reset delivery and MFA are not implemented yet. Password change requires the current password and revokes refresh sessions; access tokens already issued remain valid until their short expiry. Session management exposes family metadata only and never returns refresh-token values. Keep transactions short and never hold a database transaction open during handler execution. The API and worker share a package/schema but are separate processes (a modular monolith, not microservices). Optional adapters are chosen at process bootstrap; business handlers do not branch on Compose profiles. No arbitrary code execution, Kubernetes, distributed tracing stack, S3, workflow engine or transactional outbox is included.


### Input validation hardening

Job inputs reject non-finite numbers (`NaN`/`Infinity`) and nesting deeper than 32 levels. Timeout and attempt-count fields reject booleans instead of accepting them as integers. Canonical request hashes use strict JSON number encoding.


### Administrator safety

The API prevents role changes from removing the final administrator. Role changes lock current administrator rows in stable order to serialize concurrent changes. `APP_MASTER_KEY` and `AUTH_JWT_SECRET` must be separate random secrets. Unit and PostgreSQL integration coverage for these rules is part of the test suite; the PostgreSQL tests must be run using the isolated test Compose stack.

## Repeatable API + worker baseline benchmark

After starting the full default stack and creating an API key with both `jobs:create` and `jobs:read` scopes, run the bundled standard-library benchmark through Nginx:

```bash
export PLATFORM_API_KEY='paste-the-raw-key-here'
python scripts/benchmark_jobs.py \
  --base-url http://127.0.0.1:8080 \
  --jobs 100 \
  --concurrency 8 \
  --deadline 120 \
  --report benchmark-baseline.json
```

It reports accepted submissions/second, submission latency p50/p95/p99, terminal job counts, approximate completion throughput, and bounded error samples. The script exits non-zero if any job fails submission, polling encounters errors, the deadline is reached, or not all requested jobs succeed. Keep the report alongside the machine specs, Docker image versions, worker concurrency, and commit ID when comparing runs. Start small (e.g. 20 jobs and concurrency 2), then increase gradually. Do not run load tests against a service you do not own or have permission to test. This is a baseline harness, not a formal capacity or tail-latency certification.
