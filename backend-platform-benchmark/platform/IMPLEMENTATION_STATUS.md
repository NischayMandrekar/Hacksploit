# Implementation status

## Milestones present in this repository

- **M1 — Foundation:** Python package, environment settings, PostgreSQL schema, Alembic initial migration, Dockerfile, Compose services, API/worker entrypoints, liveness/readiness endpoints.
- **M2 — Secure API:** bootstrap master-key administration, hashed API keys, scope checks, ownership-constrained reads/cancellation, durable job submission, idempotency conflict/replay behavior, bounded input, status and cancellation routes.
- **M3 — Worker reliability:** separate process, bounded thread concurrency, `FOR UPDATE SKIP LOCKED`, random lease token, lease recovery, deadlines, retries with backoff, cooperative cancellation, stale-token result rejection.
- **M4 — Operations:** JSON logs, request IDs, API and worker Prometheus metrics, Prometheus/Grafana provisioning, optional RabbitMQ wake-up and Redis rate limiting, best-effort Kafka lifecycle events, unit and PostgreSQL integration tests.
- **Hardening pass:** added HTTP-level PostgreSQL tests for master-key auth, scopes, idempotency replay/conflict, ownership, revocation, expiry, and queued cancellation. Expired-lease recovery now finalizes a pending cancellation instead of requeueing it. Settings reject a maximum job timeout that exceeds the worker's configured execution cap. Best-effort Kafka lifecycle events now cover worker claim, lease-recovery, deadline-failure and cancellation transitions as well as API submission/completion.
- **Latest pass:** added a dedicated `compose.test.yaml` and `Dockerfile.test` so the complete PostgreSQL suite can run against an isolated, disposable PostgreSQL 17 container. HTTP tests now use a fresh database session per request, and tests cover concurrent HTTP idempotency and rate-limit rejection. Repeated cancellation of an already-cancelled queued job no longer emits another lifecycle event or increments the terminal-outcome metric. Idempotency-race handling translates only the named `uq_jobs_idempotency` PostgreSQL constraint. The compute handler rejects arithmetic overflow. API job inputs now reject NaN/Infinity and nesting deeper than 32 levels; timeout and attempt counts reject booleans despite Python treating `bool` as a subclass of `int`; canonical request hashing disables non-standard NaN JSON encoding.

## Validation performed in the generation environment

- Python compile check: passed.
- Unit tests: 10 passed (including overflow rejection, constraint classification, invalid JSON numeric values, nesting bounds and integer-field validation).
- PostgreSQL integration tests: 17 collected but skipped because no dedicated `TEST_DATABASE_URL`/PostgreSQL service was available (7 HTTP-level tests and 10 worker/database tests). Total current result: 10 passed, 17 skipped.
- YAML syntax parse for Compose, the isolated test Compose file, Prometheus and Grafana datasource: passed.
- FastAPI import and HTTP smoke checks for `/health`, `/metrics` (disabled-mode `404`), and `/openapi.json` passed with a temporary SQLite URL for non-database endpoints only (in the earlier validation pass). SQLite was not used to validate job locking or persistence behavior.
- Docker/Podman are unavailable here, so the new isolated test stack was not launched and PostgreSQL behavior remains unverified.

## Not verified here

Docker Engine/Compose and a PostgreSQL server were unavailable in the generation environment, so the Compose stack was not built or launched, migrations were not executed against PostgreSQL, and the 15 PostgreSQL integration tests have not been run. Do those checks on the target machine before treating the implementation as validated. No claim of production readiness is made.

## Remaining hardening before production

- Run the complete PostgreSQL integration suite repeatedly, including concurrent claims and lease-expiry fencing.
- Add a transactional outbox only if Kafka lifecycle events must become durable/auditable.
- Use process isolation for any future untrusted or unbounded CPU handler; current registered demo handlers are intentionally bounded and cooperative.
- Replace local-development database and Grafana credentials and configure real secrets before any shared deployment.


## Identity + reverse-proxy milestone (2026-10-27)

- Added user accounts with case-insensitive email uniqueness, Argon2id password hashing, generic invalid-credential responses, short-lived HS256 JWT access tokens (issuer/audience/expiry/type validation), and active-account lookup on each authenticated request.
- Added opaque refresh tokens stored only as SHA-256 hashes, one-row-per-token rotation history, expiry, logout/logout-all, and token-family revocation when a rotated refresh token is replayed. Access tokens can remain valid until their short expiry after logout; this is documented.
- Added user/admin roles, explicit job scopes for API keys, first-admin bootstrap through `X-Master-Key`, admin user listing/role changes, user profiles, user-managed API-key issue/list/revoke, and user-owned jobs. Service API keys remain supported.
- Jobs can be owned by a user, an API key, or both (for a user-owned API key). API-key principals remain limited to jobs created with that key; user JWTs can see jobs belonging to their account, including jobs submitted with their own API keys. User/API-key idempotency indexes are separated.
- Added Nginx as the only published default-stack entry point on `127.0.0.1:8080`. FastAPI and PostgreSQL have no published host ports. Nginx sets request IDs/forwarding headers, applies request-size/time limits and route-level rate limits, and returns JSON for Nginx-level 429 responses.
- Added migration `0002_identity`, user-authentication unit/integration tests, and a `migrate-test` service in the isolated test Compose stack so Alembic migrations execute before pytest in that stack.
- Added `PyJWT` and `argon2-cffi` runtime dependencies. `.env.example` now includes independent JWT and master-key secret placeholders; configuration rejects the placeholders so they must be replaced before startup.

### Validation in the generation environment

- `python -m compileall -q src tests migrations`: passed.
- `DATABASE_URL=sqlite:// ... pytest -q`: **12 passed, 20 skipped**. The skipped tests require PostgreSQL; SQLite was used only to allow imports and execute unit tests, not to test database behavior.
- FastAPI `TestClient` smoke checks for `/health`, `/metrics`, `/openapi.json` and presence of auth/profile/job/admin routes: passed. These checks do not validate PostgreSQL-backed endpoints.
- YAML parse for Compose/test Compose/Prometheus/Grafana datasource: passed.
- `nginx -t` with the platform Nginx config and a local stub upstream: passed. This validates syntax only, not live proxy routing.
- Docker Engine/Compose and a PostgreSQL server are unavailable here. The migration, PostgreSQL integration tests, Nginx-in-Compose startup, and complete end-to-end flows have **not** been executed. Do not call this production-ready until those checks pass.

### Remaining security/product work

- Run the new migration and PostgreSQL integration tests in the isolated test Compose stack.
- Verify Nginx + API + worker end-to-end on the target machine.
- Email verification, password-reset delivery, MFA, and immediate access-token revocation are not implemented. Access tokens are intentionally short-lived (default 10 minutes); refresh-session revocation does not invalidate an already-issued access token immediately.
- Review refresh-token concurrency/reuse behaviour against a real PostgreSQL database and add monitoring/alerts before shared deployment.


### Identity follow-up hardening (2026-10-27)

- Added authenticated password change with current-password verification, new-password minimum length, unchanged-password rejection, and refresh-session revocation. Existing access tokens remain valid until their short expiry by design.
- Added refresh-session family listing and per-family revocation. Responses expose session metadata only, never refresh tokens; revocation is constrained to the authenticated user's own session families.
- Added two PostgreSQL HTTP integration tests for password change and session-family ownership/revocation.
- Validation after this pass: Python compilation passed; `pytest -q` with SQLite used only for unit-test imports reported 12 passed and 22 PostgreSQL-only tests skipped. No PostgreSQL integration test was executed because no PostgreSQL service is available here. Docker/Compose and live Nginx end-to-end behaviour remain unverified.

- Follow-up refinements: password-change and per-session-family revocation endpoints; session listing uses bounded PostgreSQL aggregation/pagination instead of loading all refresh-token history into application memory. The Nginx authentication rate-limit location now covers password change and logout endpoints. Application auth rate limiting keys off the `X-Real-IP` value overwritten by the trusted Nginx ingress rather than treating all proxied clients as one Nginx peer.
- Latest local validation: `python -m compileall -q src tests migrations` passed; `pytest -q` reported **12 passed, 22 skipped** (PostgreSQL-only); all Compose/monitoring YAML parsed; FastAPI health/metrics/OpenAPI and new route-discovery smoke checks passed; `nginx -t` passed using a temporary wrapper and local stub upstream. No live Docker/Nginx/PostgreSQL stack was started.


### Identity hardening follow-up

- Added protection against demoting the final administrator, including the master-key bootstrap role-assignment path. Role changes lock current admin rows in stable order so concurrent role changes serialize rather than independently demoting the last admins.
- Configuration now rejects reuse of the same secret for `APP_MASTER_KEY` and `AUTH_JWT_SECRET`.
- Added configuration unit tests and PostgreSQL HTTP integration tests for final-admin protection and ordinary admin demotion when another admin remains.
- These PostgreSQL tests are added but not run in this environment; verify them with the isolated Compose test stack.

- Latest post-hardening validation: `DATABASE_URL=sqlite:// pytest -q` reports **17 passed, 24 skipped** after adding pure unit coverage for final-admin protection. The skipped tests are PostgreSQL-only and have not been executed here. Python compilation and Compose/monitoring YAML parsing passed. No Docker or PostgreSQL end-to-end validation is claimed.

### Benchmarking milestone

- Added `scripts/benchmark_jobs.py`, a dependency-free load/baseline harness that submits bounded `demo.compute` jobs concurrently through Nginx, polls job completion, and emits JSON with submission p50/p95/p99 latency, accepted submission rate, terminal status counts, approximate successful completion rate, and bounded error samples.
- Added unit tests for the percentile calculation and documented repeatable invocation and interpretation in README.
- Validation in this environment: `python -m compileall -q src tests migrations scripts` passed; `DATABASE_URL=sqlite:// ... pytest -q` reports 20 passed and 24 PostgreSQL-only tests skipped. The benchmark was not run against a live platform because Docker/PostgreSQL are unavailable here; no throughput or latency claims are made.
