# SEO Health Agent

Multi-agent SEO and site-health auditor on Groq's free-tier API. A CLI
(`main.py`), a FastAPI server (`api.py`) and a one-file web page
(`web/index.html`) all sit on one function: `agent.run_full_audit()`.

## Commands

Everything runs in Docker. Do not create a venv or `pip install` on the host.

```bash
docker compose up --build                     # app on http://localhost:3003
docker compose run --rm --build test          # full test suite (~2.5 min)
docker compose run --rm --build test pytest tests/test_tools.py -k guard   # one file / one test
docker compose run --rm app python main.py audit https://example.com --mode quick
docker compose run --rm app python main.py history|eval|analyze|similar|predict-approval ...
```

`--build` matters on `run`: without it Compose reuses the last image and you
test stale code. A real audit needs `GROQ_API_KEY` or `GROQ_API_KEYS` in
`.env` (see `.env.example`); the tests need neither a key nor a network.

## How an audit flows

```
planner (a rule) -> specialists (concurrent, tool-calling) -> postprocess.py (deterministic fact-check)
        -> evidence gate -> synthesizer <-> critic (reflection loop) -> orchestrator.py reconciliation -> SQLite
```

- `agent/orchestrator.py` — the pipeline and the final deterministic fixes
  (fixed category weights, score recomputed from them, Lighthouse-measured
  scores pinned, empty categories recovered or dropped and listed as skipped).
  Raises `AuditFailed` (`agent/errors.py`) instead of grading a blocked or
  mostly-failed scan.
- `agent/base_agent.py` — the agent loop: rate-limit handling, model fallback,
  API-key rotation, payload shrinking, JSON repair.
- `agent/tools.py` — everything that touches the network (fetch, SSL, links,
  PageSpeed/Lighthouse).
- `agent/postprocess.py` — overwrites model claims that contradict tool results.
- `agent/netguard.py` — blocks fetches to private/loopback addresses.
- `api.py` — job-based HTTP API plus the static page.

## Rules that are not obvious from the code

- **Fix model mistakes in code, not in prompts.** Anything verifiable or
  computable (arithmetic, cert validity, Lighthouse numbers, trend direction)
  is enforced deterministically in `postprocess.py` / `orchestrator.py`.
  Each such fix ships with a regression test reproducing the real failure.
- **Tests never touch the network or a real API key.** `tests/conftest.py`
  fakes the Groq client; tool tests patch `requests`/`socket`. The `test`
  container runs with networking disabled, so a test that reaches out fails.
- **`agent/config.py` reads env vars at import time.** `load_dotenv()` must
  run before the first `import agent` (see the top of `main.py` and `api.py`).
- **Exactly one uvicorn worker.** `api.py` keeps jobs and rate-limit counters
  in process memory. More workers means lost jobs and a multiplied rate limit.
- **Tool caches are per-audit.** `tools._page_cache`, `_fetch_result_cache`,
  `_lighthouse_cache` and `_lighthouse_errors` share data (and a failed
  Lighthouse call) between one audit's specialists. `api.py` empties them
  with `tools.clear_caches()` when no audit is running; any other long-lived
  caller must do the same or re-audits get scored on stale data. Tests that
  touch the tools reset with `tools.clear_caches()`, not one cache.
- **A job id is also the stored report's public id.** `api.py` passes it to
  `run_full_audit(audit_id=...)`; `GET /api/audit/{id}` falls back to SQLite
  when the job is no longer in memory. Row ids are never exposed.
- **Visitors only see `AuditFailed.public_message` or a fixed text.** Raw
  exceptions and unredacted progress lines go to the server log.
- **Report text is untrusted.** It is written by a model reading someone
  else's website. The page inserts it as text nodes only — never `innerHTML`.
- **`SEO_AGENT_BLOCK_PRIVATE_HOSTS=1` must stay on for anything public.** The
  Docker image sets it. It is off by default only so the CLI can audit a
  localhost dev site. New network-touching tools must call the guard, because
  the model (steered by page content) chooses their arguments.
- **`data/audit_history.db` in git is seed data.** The live database is the
  `seo_data` Docker volume, filled from the seed on first start only. Local
  audits through Docker write to the volume, not to the tracked file.

## Branches and deploy

- `main` — integration. Merge and test here. CI runs the tests, never deploys.
- `prod` — every push runs the tests, then deploys to the VPS.
- Release = merge `main` into `prod` (PR, or `git push origin main:prod`).
- **Do not push to `prod` unless asked.** It changes the live site.
- **Do not commit.** Leave changes in the working tree and propose the commit
  message; the maintainer commits. No AI co-author lines in messages.

Deploy (`.github/workflows/ci.yml`, prod only): build the runtime image and
push it to `ghcr.io/neonara/seo-checker-with-agentic-ai` (tags: commit SHA
and `prod`), copy `deploy/docker-compose.yml` to the VPS, then pipe
`deploy/deploy.sh` over SSH. That script pulls the SHA, switches the
container, waits for the health check, and rolls back to the previous image
if it fails.

- Two compose files: the root `docker-compose.yml` is for development (builds
  from source); `deploy/docker-compose.yml` is what the VPS runs (pulls from
  GHCR). A change to ports, volumes or env handling usually belongs in both.
- The VPS holds only that compose file and a hand-made `.env` (Groq keys,
  `APP_PORT`) in `$VPS_PATH`; deploys never touch `.env`.
- Served over plain HTTP on port 3003; 3000 and 3001 on that VPS belong to
  other projects.

If a reverse proxy is ever put in front, set uvicorn's
`--forwarded-allow-ips` to it, or every visitor shares one rate-limit bucket.
