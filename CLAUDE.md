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
planner -> specialists (concurrent, tool-calling) -> postprocess.py (deterministic fact-check)
        -> synthesizer <-> critic (reflection loop) -> orchestrator.py reconciliation -> SQLite
```

- `agent/orchestrator.py` — the pipeline and the final deterministic fixes
  (score recomputed from category weights, empty categories recovered or dropped).
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
- **Tool caches are per-audit.** `tools._page_cache` / `_lighthouse_cache`
  share data between one audit's specialists. `api.py` clears them when no
  audit is running; any other long-lived caller must do the same or
  re-audits get scored on stale data.
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

Deploy (`.github/workflows/ci.yml`): rsync the repo to the VPS over SSH,
`docker compose up -d --build`, wait for the container health check. The
server's `.env` (Groq keys, `APP_PORT`) lives only on the VPS at
`$VPS_PATH/.env` and is never overwritten. Served over plain HTTP on port
3003; 3000 and 3001 on that VPS belong to other projects.

If a reverse proxy is ever put in front, set uvicorn's
`--forwarded-allow-ips` to it, or every visitor shares one rate-limit bucket.
