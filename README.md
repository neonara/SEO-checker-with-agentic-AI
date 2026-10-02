# SEO Health Agent — Multi-Agent Edition (Groq-powered)

An agentic SEO & website-health auditing system. A planner decides what to
investigate, specialist agents each independently research one category
using real tool calls, a synthesizer merges their findings into one weighted
report, a critic agent reflects on that report and can send it back for
revision, and everything is persisted so later runs can reason about trends
over time.

Runs entirely on **Groq's free-tier API** (OpenAI-compatible, very fast,
open-weight models) — no paid model provider required. Hardened through
extensive real-world testing against live sites (Wikipedia, YouTube, Apple,
Samsung, Internet Archive's blog, and others) to survive both Groq's
free-tier limits and the kinds of mistakes LLMs make when asked to
self-report facts and do arithmetic. Backed by a 435-test automated
`pytest` suite and a self-grading eval harness (see Testing / Eval harness
below).

Three ways in, all on the same `run_full_audit()` function: a CLI, an
importable library, and a small web app (a FastAPI backend plus a one-page
frontend, see "Web app" below) that deploys to a VPS with Docker.

## Architecture

```
                         ┌─────────────┐
                         │   Planner   │  decides which specialists to run
                         └──────┬──────┘
                                │
        ┌───────────────────────┴────────────────────────┐
        │       Up to 8 specialists run concurrently      │
        │  Technical SEO · Content · Performance+LH ·     │
        │  Security · Links · Accessibility ·             │
        │  Best Practices · Competitive*                  │
        │               (*Groq Compound, built-in search) │
        └───────────────────────┬────────────────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │  Deterministic cleanup  │  SSL / CWV / accessibility /
                     │     (postprocess.py)    │  best-practices fact-check,
                     │                         │  on-page overlap removal
                     └───────────┬─────────────┘
                                 ▼
                         ┌───────────────┐
                         │  Synthesizer  │  merges + weights + scores
                         └───────┬───────┘
                                 ▼
                         ┌───────────────┐
               ┌────────►│    Critic     │  reflection / self-critique
               │         └───────┬───────┘
               │ revise if       ▼
               │ not approved  approved?
               └────────────── no ── yes ──► Draft Report
                                                   │
                                                   ▼
                          Category recovery/cleanup, deterministic
                       score/weight/trend reconciliation, stale-issue
                                        filtering
                                                   │
                                                   ▼
                                             Final Report
                                                   │
                                                   ▼
                                        SQLite memory (trend tracking)
```

## How this project evolved

This started as a single-model script and became a hardened multi-agent
pipeline through real, repeated testing on Groq's free tier. Nearly every
part of the current design exists because of a specific bug caught this
way — LLMs are treated as fallible reasoning engines that propose findings,
while anything verifiable or computable outright is enforced
deterministically in code around them instead of just prompted more firmly.

- **Bad arithmetic.** The model's own overall-score math rarely matched its
  category scores/weights. Fixed: recompute deterministically
  (`_reconcile_overall_score`), near-zero tolerance.
- **Restating already-known facts wrong.** A smaller model would sometimes
  hallucinate "certificate expired" from a perfectly valid cert. Fixed: the
  tool computes `is_expired` with certainty; any contradicting finding gets
  deterministically overwritten (`reconcile_ssl_findings`).
- **Same pattern for real performance/accessibility/best-practices data.**
  Findings sometimes stayed vague instead of citing the real Lighthouse
  numbers. Fixed: canonical findings get injected from the real data when
  missing (`reconcile_core_web_vitals`, and a shared matcher —
  `keyword_topic_covered` — for accessibility/best-practices). That matcher
  itself needed two rounds of fixing: exact-substring matching produced
  false "not covered" results on ordinary rewording (plurals, word order),
  and a coincidental digit match (score "50" inside an unrelated "30-50%"
  phrase) once wrongly suppressed a real injection — both fixed with
  stemmed, boundary-aware matching.
- **Specialists don't share ground truth.** The competitive specialist
  repeatedly re-judged title/meta/canonical facts another specialist had
  already measured, sometimes contradicting them. Fixed: stripped
  deterministically (`strip_competitive_onpage_overlap`).
- **Prose contradicting its own structured data.** Wrong trend direction,
  wrong magnitude, or a fabricated comparison to a "previous audit" that
  never existed. Fixed: `fix_summary_trend_mismatch` and
  `fix_fabricated_trend_claim`.
- **Categories silently losing real data.** A specialist could succeed and
  still have the synthesizer drop its findings during merging. Fixed:
  `_recover_or_drop_empty_categories` recovers the original findings first
  (and now the score too, independently — see below), before dropping.
- **Anti-bot blocking corrupting an audit.** A block page's "no headings,
  no images" got scored as real content. Fixed: `fetch_page` flags
  likely-blocked responses; a standard browser User-Agent and a
  high-failure-rate caution flag in link checking help too.
- **A rejected draft shipping anyway, via a wasted extra call.** The critic
  could reject on its final round and the pipeline would still run one
  more, never-reviewed pass. Fixed: stop at the last-reviewed draft;
  `review_status` surfaces exactly what's unresolved.
- **Free-tier quota exhaustion.** Now handled with real strategy: parses
  Groq's wait-time formats, prefers instant model-fallback over waiting,
  round-robins every configured API key continuously across the whole run
  (not per-stage), shrinks oversized payloads reactively on 413s *and*
  proactively before sending (`agent/compaction.py`), and only fails fast
  when a wait is genuinely too long.
- **A stale cert trust store.** `requests` and the raw `ssl` module used
  different root CA lists, occasionally producing a false "expired."
  Fixed: both now use the same `certifi` bundle.
- **A resolved complaint still shown as "unresolved."** The critic reviews
  *before* weight normalization runs, so "weights don't sum to 1.0" could
  survive into the final report as an active complaint about something
  already fixed. Fixed: `_drop_issues_resolved_by_reconciliation` strips
  only that mechanically-resolved class, leaving genuine judgment calls alone.
- **A finding saying "OK" and "Failing checks include: X" at once.**
  Lighthouse can have a 100/100 score with one zero-weight informational
  audit still "failing" — both facts true, but self-contradictory to read.
  Fixed: never label a finding "good" while it's also naming a failing check.
- **Every audit starting key rotation from the same key.** Running several
  audits back-to-back (as the eval harness does) never actually spread
  them across multiple configured keys. Fixed: `starting_key_index` on
  `run_full_audit()`, rotated per case.
- **A failed specialist's raw JSON leaking into the synthesizer/critic as
  trusted data.** The error message embedding a truncated JSON attempt got
  copied verbatim into `raw_evidence_notes` — both downstream agents then
  cited "findings" that were never actually parsed. Fixed: a short,
  explicitly-non-data placeholder message instead.
- **A category shipping with real-looking findings but a `null` score.**
  Downstream of the bug above. Fixed: `_recover_or_drop_empty_categories`
  now validates `score` independently of findings, dropping the whole
  category if the specialist itself never had a valid one either.
- **A benchmark case's own premise being wrong.** The eval harness assumed
  a site "never redirecting to https" meant "no valid TLS" — a real run
  showed a genuinely valid handshake, falsifying the premise. Removed
  rather than left as a permanent false-negative generator.
- **Correlation ranking rewarding small-sample noise.** In Score Analytics,
  sorting by raw `|r|` let an 8-sample correlation (nonsensical sign) rank
  above a 51-sample one (sensible sign). Fixed: reliable results (n≥15)
  always rank first, unreliable ones are flagged, not hidden.

All of the above were found by actually running the pipeline against real
sites and real accumulated data — not by adding a feature and assuming a
clean first run meant it worked.

## Agentic AI features

1. **Multi-agent orchestration** — a planner and up to 7 specialists
   (technical SEO, content, performance, security, links, accessibility,
   best practices, plus competitive when relevant) run **concurrently**,
   reconciled by a synthesizer.
2. **Reflection / self-critique loop** — a critic reviews the draft against
   raw evidence and can send it back for revision (max 2 rounds by
   default), stopping at the last-reviewed draft rather than shipping an
   unreviewed final pass. `review_status`/`unresolved_review_issues`
   surface what's still unresolved, filtered to exclude complaints already
   mechanically fixed.
3. **Tool use / function calling** — each specialist chooses its own tool
   calls (fetch, parse, SSL check, headers, links, real Lighthouse audit).
4. **Real performance/accessibility/best-practices data** via Google
   PageSpeed Insights — genuine Lighthouse data, not a proxy signal, shared
   across one cached API call per `(url, strategy)` so the extra categories
   are near-free. Best Practices deterministically excludes HTTPS/SSL
   audits (Security's domain) rather than relying on a prompt to avoid the
   overlap.
5. **Live web-search-augmented research** — the competitive specialist runs
   on Groq's **Compound** system (`groq/compound-mini`), web search
   server-side.
6. **Persistent long-term memory** — every audit written to SQLite
   (`agent/memory.py`), enabling trend tracking and the Score Analytics
   dataset below.
7. **Schema-validated structured output** — every final report validated
   against a Pydantic schema.
8. **A deterministic fact-checking layer** (`postprocess.py`) — see "How
   this project evolved" for the full list of what it catches and why.
9. **Multi-key, multi-model resilience** — automatic model fallback,
   proactive round-robin across all configured keys for the entire run,
   proactive *and* reactive payload shrinking, JSON self-repair retries.
10. **435-test automated regression suite** — see Testing below.
11. **Self-grading eval harness** — runs the real pipeline against a
    randomly-sampled pool of benchmark sites with known, verifiable
    issues, graded in pure Python (no LLM call spent on grading). See Eval
    harness below.
12. **Score Analytics** — classical ML / statistics on top of the SQLite
    audit history: feature engineering, Pearson correlation, and
    cross-validated comparison of 4 scikit-learn regressors predicting
    `overall_score`. See Score Analytics below.
13. **Findings Similarity Search** — vector search (TF-IDF by default;
    optional real sentence embeddings) over a seed knowledge base plus real
    audit history, so a past finding/fix can be looked up by meaning rather
    than exact wording. See Findings Similarity Search below.
14. **Critic-Approval Predictor** — a small classifier trained from scratch
    on real audit history, predicting whether the critic will approve a
    draft. Honestly scoped: this is genuine classical-ML training on this
    project's own data, not a rebranded LLM fine-tune Groq's API can't do
    anyway. See Critic-Approval Predictor below.

## Setup

Everything runs in Docker (Compose 2.24 or newer). There is no virtualenv to
create and nothing to install on the host.

```bash
cp .env.example .env   # then edit .env and add your GROQ_API_KEY
docker compose up --build
```

That builds the image and serves the web app on http://localhost:3003.

Get a **free** API key at https://console.groq.com/keys. Optionally get a
free Google PageSpeed Insights key at
https://developers.google.com/speed/docs/insights/v5/get-started for a
higher rate limit on real Lighthouse audits (works without one too, at a
lower rate limit).

## Web app

Open http://localhost:3003, paste an address, and press **Scan site**. The
page shows the four pipeline stages and the live agent log while the audit
runs, then the graded report: findings by category (worst first), quick
wins, and details including what the reviewer agent still objected to and
the site's score history.

`api.py` is the backend behind it:

| Route | Purpose |
|---|---|
| `POST /api/audit` | Start an audit: `{"url", "mode", "competitor_url"}` → `{"job_id"}` |
| `GET /api/audit/{job_id}` | Poll: `status` (`running` / `done` / `error`), live `logs`, final `report` |
| `GET /api/history/{domain}` | Past scores for a domain |
| `GET /api/health` | Liveness, used by the container health check |

Because it is meant to be reachable from the internet, starting an audit is
limited (all adjustable in `.env`, see Configuration):

- **Public addresses only.** Loopback, private-network and cloud-metadata
  addresses, and non-web ports, are refused — both for the submitted URL and
  for every URL the agents' tools fetch afterwards, redirects included
  (`agent/netguard.py`).
- **5 audits per hour per visitor IP**, and **2 audits at a time** overall.
  Every audit spends shared Groq free-tier quota.
- Asking for an audit that is already running returns the running job.

Jobs live in the server's memory: results stay retrievable for an hour and
are lost on restart (the audit itself is still saved to the history
database). This is also why the server runs exactly one worker.

## Usage (CLI)

Each command below runs inside the container, so prefix it with
`docker compose run --rm app` — for example
`docker compose run --rm app python main.py audit https://example.com`.
Files written with `--out` / `--pdf` land inside the container; point them at
`/app/data/` to keep them in the data volume.

```bash
# Full multi-agent audit (auto mode: strong model, falls back if needed)
python main.py audit https://example.com

# Fast, always-available mode -- skips the large model entirely
python main.py audit https://example.com --mode quick

# Best-quality mode -- only the strong model, fails clearly rather than
# silently downgrading if its quota is exhausted
python main.py audit https://example.com --mode deep

# Also benchmark against a competitor
python main.py audit https://example.com --competitor https://competitor.com

# Save JSON + a polished PDF report
python main.py audit https://example.com --out report.json --pdf report.pdf

# Score history for a domain
python main.py history https://example.com

# Self-grading eval harness against known-issue benchmark sites
python main.py eval

# Classical ML / statistical analysis of your audit history
python main.py analyze

# Search past findings for ones similar to a new issue
python main.py similar "page missing meta description"

# Train a classifier predicting whether the critic will approve a draft
python main.py predict-approval
```

Or use it as a library:

```python
from agent import run_full_audit

report = run_full_audit("https://example.com", competitor_url=None, mode="auto")
print(report["overall_score"], report["grade"], report["review_status"])
```

## Project layout

```
.
├── main.py                  CLI entry point (audit / history / eval / analyze, --mode flag)
├── api.py                   FastAPI backend: job-based audit API + serves web/
├── web/index.html           the one-page frontend (no build step)
├── Dockerfile               base / test / runtime stages
├── docker-compose.yml       development: `app` (builds from source) and `test` (the suite)
├── deploy/                  production: the compose file and script the VPS runs
├── .github/workflows/ci.yml tests on main; tests, image to GHCR and deploy on prod
├── pytest.ini
├── requirements.txt         core dependencies
├── requirements-api.txt     web server dependencies
├── requirements-dev.txt     test dependencies
├── .env.example
├── CLAUDE.md                working notes for Claude Code
├── README.md
├── agent/
│   ├── __init__.py           exposes run_full_audit
│   ├── config.py             model names, API keys, & tunables, all overridable via env vars
│   ├── tools.py              fetch/parse/ssl/headers/links + real Lighthouse data via
│   │                         PageSpeed Insights (CWV, accessibility, best-practices share
│   │                         one cached API call per (url, strategy))
│   ├── tool_schemas.py        Groq/OpenAI-format tool-use schemas, grouped per specialist
│   ├── base_agent.py          the agentic loop runtime: rate-limit/quota handling, model
│   │                          fallback, API-key rotation, payload shrinking, JSON self-repair
│   ├── specialists.py         specialist system prompts + tool assignments
│   ├── planner.py             planning agent
│   ├── synthesizer.py         synthesizer agent
│   ├── critic.py              critic agent + reflection loop controller
│   ├── postprocess.py         the deterministic fact-checking layer -- see "How this project
│   │                          evolved" above
│   ├── orchestrator.py        top-level pipeline: mode configs, category recovery, score/
│   │                          weight reconciliation, stale-complaint filtering
│   ├── memory.py              SQLite persistence, trend lookups, full-history dataset access
│   ├── schemas.py             Pydantic validation of the final report contract
│   ├── report_pdf.py          reportlab-based PDF export
│   ├── eval_harness.py        self-grading eval harness -- see "Eval harness" below
│   ├── analytics.py           Score Analytics -- see "Score Analytics" below
│   ├── similarity_search.py   Findings Similarity Search -- see below
│   ├── critic_predictor.py    Critic-Approval Predictor -- see below
│   ├── compaction.py          proactive payload compaction (see "How this project evolved")
│   └── netguard.py            refuses fetches to private/loopback addresses (web deployment)
└── tests/                     435 pytest tests -- see "Testing" below
    ├── conftest.py             shared fixtures: fake Groq client/errors, sample report data
    ├── test_analytics.py       feature engineering, correlations, model training/comparison
    ├── test_api.py             the HTTP API: job lifecycle, URL validation, rate/concurrency limits
    ├── test_base_agent.py      retry/backoff/rate-limit engine, the tool-call loop
    ├── test_compaction.py      proactive payload trimming, incl. actual wiring
    ├── test_critic.py          reflection loop, incl. a dedicated regression test
    ├── test_critic_predictor.py  classification feature engineering, class balance, classifier comparison
    ├── test_eval_harness.py    eval harness grading logic (run_full_audit fully mocked)
    ├── test_memory.py          SQLite persistence
    ├── test_netguard.py        which addresses, ports and schemes are refused
    ├── test_orchestrator.py    score/weight reconciliation, category recovery, pipeline wiring
    ├── test_postprocess.py     every deterministic reconciliation function
    ├── test_schemas.py         the Pydantic report contract
    ├── test_similarity_search.py  corpus building, TF-IDF search, embedding fallback
    └── test_tools.py           every tool implementation, network fully mocked
```

## Report shape

```json
{
  "url": "...",
  "overall_score": 78,
  "grade": "B",
  "summary": "...",
  "review_status": "approved",
  "unresolved_review_issues": [],
  "categories": [
    {
      "name": "Technical SEO",
      "score": 85,
      "weight": 0.25,
      "findings": [
        {"severity": "warning", "issue": "...", "recommendation": "..."}
      ]
    }
  ],
  "quick_wins": ["..."],
  "data_limitations": "...",
  "trend": {"previous_score": 70, "previous_timestamp": "...", "score_delta": 8}
}
```

`review_status` is `"not_approved"` (with `unresolved_review_issues` populated)
if the critic never signed off after the max reflection rounds — treat those
reports with extra scrutiny.

## Configuration

All overridable via environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — (required unless `GROQ_API_KEYS` set) | API auth |
| `GROQ_API_KEYS` | — (optional) | Comma-separated list of multiple keys; specialists and synthesizer/critic calls round-robin across them for the whole run |
| `GOOGLE_PAGESPEED_API_KEY` | — (optional) | Raises the rate limit on real Lighthouse audits; works without one at a lower limit |
| `SEO_AGENT_MODEL` | `openai/gpt-oss-120b` | Primary model used by specialists, synthesizer |
| `SEO_AGENT_PLANNER_MODEL` | same as above | Model for the planner agent |
| `SEO_AGENT_CRITIC_MODEL` | same as above | Model for the critic agent |
| `SEO_AGENT_FALLBACK_MODEL` | `openai/gpt-oss-20b` | Used automatically on a rate/quota limit (separate quota pool); empty disables fallback |
| `SEO_AGENT_COMPETITIVE_MODEL` | `groq/compound-mini` | Competitive specialist only. Groq shut Compound down on 2026-09-21, so this section is currently skipped in auto/deep mode until `browser_search` is wired in |
| `SEO_AGENT_MAX_ITER` | 10 | Max tool-call iterations per agent |
| `SEO_AGENT_MAX_REFLECTION_ROUNDS` | 2 | Max critic revision rounds |
| `SEO_AGENT_MAX_WORKERS` | 2 | Max concurrent specialist agents |
| `SEO_AGENT_DISPATCH_STAGGER` | 2.0 | Seconds between dispatching each specialist |
| `SEO_AGENT_RATE_LIMIT_RETRIES` | 4 | Max retry attempts on a rate-limited call |
| `SEO_AGENT_DB_PATH` | `./data/audit_history.db` | SQLite history location |
| `SEO_AGENT_COMPACTION_TOKEN_THRESHOLD` | 4000 | Estimated-token threshold above which the synthesizer/critic payload is proactively compacted |
| `SEO_AGENT_COMPACTION_MAX_FINDINGS` | 12 | Max findings kept per category when compacting |
| `SEO_AGENT_COMPACTION_MAX_EVIDENCE_CHARS` | 600 | Max length for evidence notes/critic instructions when compacting |
| `SEO_AGENT_COMPACTION_MAX_FINDING_CHARS` | 400 | Max length for a finding's issue/recommendation text when compacting |
| `SEO_AGENT_BLOCK_PRIVATE_HOSTS` | off (`1` in the Docker image) | Refuse to fetch loopback/private/link-local addresses and non-web ports. Keep on for anything public |
| `APP_PORT` | 3003 | Host port the web app is published on (docker compose) |
| `SEO_API_RATE_LIMIT_PER_HOUR` | 5 | Audits one visitor IP may start per hour; 0 disables |
| `SEO_API_MAX_CONCURRENT` | 2 | Audits allowed to run at once |
| `SEO_API_JOB_TTL_SECONDS` | 3600 | How long a finished job's result stays retrievable |
| `SEO_API_CORS_ORIGINS` | — | Comma-separated origins allowed to call the API from another site |

CLI-only: `--mode {quick,deep,auto}` on `audit` and `eval` (see Usage above).

## Testing

```bash
docker compose run --rm --build test                                    # all 435 tests
docker compose run --rm --build test pytest tests/test_tools.py -k ssl   # a subset
```

Every test mocks the Groq client and any network calls — the suite runs
with no API key, and the test container has networking switched off, so a
test that tried to reach the internet would fail. A few worth calling out:

- `test_critic.py` reintroduces the exact reflection-loop regression noted
  above to confirm it's genuinely caught, verified by temporarily
  reverting the fix and watching the test fail before restoring it.
- `test_postprocess.py` reproduces the exact real-world text that triggered
  the accessibility/best-practices dedup and stale-weight-complaint bugs.
- `test_eval_harness.py` feeds the grader a synthetic hallucinated
  "certificate is valid" report against the real
  `expired-ssl-certificate` case, confirming it would catch that
  regression if `reconcile_ssl_findings` ever broke.
- `test_analytics.py` reproduces the exact small-sample-correlation
  scenario above to confirm reliable results always outrank unreliable ones.

## Eval harness

```bash
python main.py eval                                        # random 4-of-7 sample (default)
python main.py eval --sample-size 7                         # run the whole pool
python main.py eval --sample-size 3 --seed 123               # reproduce an exact past sample
python main.py eval --mode auto --out results.json           # more thorough, more expensive
```

Runs the **real** pipeline against a random sample of a curated pool of
benchmark sites, each chosen for a known, stable, verifiably-true issue:

| Case | Site | Known-true issue |
|---|---|---|
| `expired-ssl-certificate` | `expired.badssl.com` | permanently expired cert |
| `self-signed-ssl-certificate` | `self-signed.badssl.com` | fails cert verification |
| `wrong-hostname-ssl-certificate` | `wrong.host.badssl.com` | cert issued for a different domain |
| `untrusted-root-ssl-certificate` | `untrusted-root.badssl.com` | signed by an untrusted CA |
| `no-encryption-null-cipher` | `null.badssl.com` | refuses real encryption |
| `minimal-page-missing-meta-description` | `example.com` | no meta description tag |
| `minimal-historical-page` | `info.cern.ch` | no meta description tag |

By default, each run randomly samples 4 of these 7 rather than always
running the full pool, logged with a seed (`--seed 123` reproduces it
exactly). Each sampled case starts its audit on a different configured
`GROQ_API_KEYS` entry (round-robin), so more configured keys directly buys
headroom to raise `--sample-size`.

Grading is **100% deterministic Python** — `keyword_topic_covered` (the
same matcher used internally for accessibility/best-practices dedup)
checks whether each site's known issue shows up in the right category at
the right severity. No LLM call is spent grading; only the audits
themselves consume API quota. Every SSL-failure case also asserts the
report must *not* say "is valid" — a direct regression guard.

**Honest framing:** this measures *recall of known-true issues*, not full
precision/recall — there's no ground truth for "every issue a page does or
doesn't have." What it verifies: did the pipeline still catch the specific
things we know for certain are true.

**Cost:** each sampled case is one full `run_full_audit()` call. Defaults
to `--mode quick` to keep repeated/CI runs affordable — that model has a
separate daily quota pool from `auto`/`deep` mode's primary model, so
running `eval` regularly doesn't eat into everyday audit quota unless
everyday usage is *also* run in `--mode quick`.

## Score Analytics

```bash
python main.py analyze                          # real data if >=20 audits, else synthetic
python main.py analyze --source synthetic --n-synthetic 200
python main.py analyze --source real             # force real data even if sparse
python main.py analyze --out results.json
```

Classical ML / statistics on top of your SQLite audit history:

- **Feature engineering** — category scores/weights, finding counts by
  severity, category presence, review outcome, extracted from every
  stored report (`agent/analytics.py`).
- **Statistical modeling** — Pearson correlation of every feature against
  `overall_score`, with p-values. Results are tagged `reliable` (n≥15) and
  sorted so well-supported results always outrank raw-magnitude noise —
  see "How this project evolved" for the real example that motivated this.
- **Classical ML + quantitative comparison** — `LinearRegression`, `Ridge`,
  `RandomForestRegressor`, `GradientBoostingRegressor`, compared via
  k-fold cross-validated R²/MAE/RMSE, never a single train/test split.

**Honest limitation:** real audit history starts small. `--source auto`
(default) only uses real data once you have 20+ audits; below that, it
falls back to a clearly-labeled **synthetic** dataset (built from a known,
verifiable weighted-sum generative formula) so the module is genuinely
runnable and testable before real data accumulates. Every output states
explicitly whether it used real or synthetic data — never silently.

## Findings Similarity Search

```bash
python main.py similar "page missing meta description"
python main.py similar "weak TLS cipher" --category "Web Security" --top-k 3
python main.py similar "slow page load" --backend embedding   # real semantic search, if available
```

Vector search over collected findings, so a similar past issue+fix can be
looked up by meaning instead of exact wording. Standalone (`agent/
similarity_search.py`) — **not currently wired into the live audit
pipeline**; see Possible next steps.

Two backends, both in `build_index()`:
- **TF-IDF** (default) — classical lexical vector search via scikit-learn,
  no downloads, always available.
- **Sentence embeddings** (`--backend embedding`) — genuine semantic
  similarity via `sentence-transformers`, if installed and a model can be
  downloaded. Falls back to TF-IDF automatically, with a clear log
  message, if either isn't available — never raises just because the
  optional path is missing.

The searchable corpus mixes two clearly-labeled kinds of entries: a small,
hand-authored **seed knowledge base** (~26 common findings/fixes across all
7 categories, so this is useful before much real history exists) and real
findings pulled from your stored audit history. Every search result states
which one it came from.

**Honest limitation:** TF-IDF matches shared vocabulary, not meaning — a
query like "takes forever to load" won't match a finding about "Largest
Contentful Paint" the way a real embedding model would. Install
`sentence-transformers` for genuine semantic matching.

## Critic-Approval Predictor

```bash
python main.py predict-approval                 # real data if >=20 audits, else synthetic
python main.py predict-approval --source real     # force real data even if sparse
```

A small classifier (`agent/critic_predictor.py`) trained from scratch to
predict whether the critic will approve a draft, using the real
`review_status` every stored audit already has as the training label.

**Honest scoping, precisely stated:** "fine-tuning llama-3.3-70b" isn't a
real option — Groq is an inference API, not a fine-tuning platform. This is
a genuinely small, genuinely local model instead (`LogisticRegression` /
`RandomForestClassifier` / `GradientBoostingClassifier`), and more
precisely, it's **trained from scratch**, not fine-tuned in the strict
sense of adjusting pretrained weights. Described accurately rather than
dressed up — the practical skill demonstrated (adapting a model to a
specific task and dataset) is the same either way.

Reuses `agent/analytics.py`'s feature engineering and real/synthetic
fallback pattern; the target here is `review_approved` (binary), which is
therefore excluded from the feature set to avoid target leakage.

**Always compares against a majority-class baseline**, not just against
itself — if no model beats blindly guessing the majority outcome, the
summary says so explicitly rather than reporting a misleadingly "good"
accuracy number on an imbalanced dataset. Verified in both directions: a
synthetic dataset with a genuinely random target stays at baseline; one
with a clean, real signal reaches ~100% cross-validated accuracy.

**Honest limitation:** real approval outcomes so far are heavily skewed —
most real runs across this project's own testing ended up `not_approved`.
Training needs a reasonable count of *both* outcomes (≥5 each); with too
few examples of one class, it refuses to report misleading metrics rather
than training on an unusable split.

## Deployment

Hosted with Docker on a VPS. GitHub Actions (`.github/workflows/ci.yml`)
builds the image, pushes it to GitHub's container registry (GHCR), and has
the VPS pull and run it. The VPS never holds the source code.

**Branches.** `main` is for merging and testing: every push and pull request
runs the test suite and stops there. `prod` is the only branch that deploys.
To release, merge `main` into `prod` (open a pull request, or fast-forward
with `git push origin main:prod`).

**What a push to `prod` does**

1. Runs the tests.
2. Builds the runtime image and pushes it to
   `ghcr.io/neonara/seo-checker-with-agentic-ai`, tagged with the commit SHA
   and with `prod` (always the latest release).
3. Copies `deploy/docker-compose.yml` to the VPS and runs `deploy/deploy.sh`
   there: pull that exact SHA, switch the container to it, wait for the
   health check. If the new container does not come up healthy, the previous
   image is put back and the job fails.

**One-time setup on the VPS**

1. Docker with the Compose plugin, and a user allowed to run it (in the
   `docker` group) that accepts the deploy SSH key.
2. Create the directory (default `/opt/seo-checker`), owned by that user.
3. Put a `.env` in it, based on `.env.example`, with the Groq key(s) and
   `APP_PORT`. Deploys never touch this file, and refuse to run without it.
4. Open the port in the firewall (default 3003).

After a deploy that directory holds exactly two files: `docker-compose.yml`
(overwritten on each deploy) and your `.env`.

**One-time setup on GitHub** (Settings → Secrets and variables → Actions)

| Name | Kind | Value |
|---|---|---|
| `VPS_HOST` | secret | Server address |
| `VPS_USER` | secret | SSH user |
| `VPS_SSH_KEY` | secret | Private key for that user (the whole file) |
| `GHCR_PAT` | secret | Personal access token (classic) with `write:packages`, used to push the image and to let the VPS pull it |
| `GHCR_USER` | variable, optional | Username for the GHCR login (default `neonara`) |
| `VPS_KNOWN_HOSTS` | secret, optional | Output of `ssh-keyscan <host>`, to pin the server's host key |
| `VPS_SSH_PORT` | variable, optional | SSH port (default 22) |
| `VPS_PATH` | variable, optional | Directory on the VPS (default `/opt/seo-checker`) |

The workflow logs in to GHCR with `GHCR_PAT` to push, and hands the VPS that
same token for the pull, through a temporary Docker config that is deleted
afterwards, so the token is not stored on the VPS and any GHCR login other
projects use on that machine is left alone. (Without `GHCR_PAT` it falls
back to the run's own token, which only works if the organisation allows
Actions to write packages.) A branch protection rule on `prod` is worth adding, so only reviewed
merges can trigger a deploy.

**Running or rolling back by hand on the VPS**

```bash
cd /opt/seo-checker
docker compose up -d                       # latest release (the `prod` tag)
IMAGE_TAG=<commit sha> docker compose up -d   # a specific release
docker compose logs -f app
```

The VPS keeps the current and the previous image locally, so going back one
release needs no download. Pulling by hand needs the package to be readable:
either make it public once (the package's settings page on GitHub), or run
`docker login ghcr.io` on the VPS with a token that has `read:packages`.

**Data.** Audit history lives in the Docker volume `seo-checker_seo_data`,
not in the deploy directory. It is filled from the image's seed database the
first time the container starts and never overwritten after that. Back it up
with `docker compose cp app:/app/data/audit_history.db ./backup.db`.

**After the first deploy**, check `docker compose logs app`: each request line
starts with the visitor's IP. If every line shows the same Docker-internal
address, the per-visitor rate limit is being shared by everyone, which
happens on hosts where Docker does not preserve source addresses.

## Honest limitations

- The models changed under this project. It was built, tuned and tested on
  `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`, which Groq shut down
  for free-tier keys on 2026-08-16. The defaults are now Groq's named
  replacements (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`). The offline
  test suite does not exercise a real model, so the behaviour described in
  "How this project evolved" was observed on the old models, not these.

- No JavaScript rendering for HTML parsing (though real Core Web Vitals do
  come from a genuine browser-based Lighthouse audit, which renders JS).
- Link checking samples a handful of links, not a full crawl; a high
  failure rate is flagged as likely bot-blocking rather than reported as a
  confident broken-links crisis, but still needs manual spot-checking.
- Automated accessibility/best-practices audits only catch roughly 30-50%
  of real-world WCAG issues — a clean result is a floor, not proof of
  compliance.
- The eval harness measures recall of known-true issues, not full
  precision. Score Analytics results are only as trustworthy as the
  sample size behind them (see the `reliable` flag). The Critic-Approval
  Predictor refuses to train when either outcome class is too small,
  rather than reporting a misleading accuracy number.
- The web app keeps running jobs and rate-limit counters in memory: one
  server process, results gone after a restart, and no HTTPS until a domain
  and reverse proxy are put in front of it.
- The private-address guard resolves a hostname and then lets the HTTP client
  resolve it again, so a DNS record that changes between the two lookups (DNS
  rebinding) is not caught.
- Groq's free tier has real daily quota limits per model *and*
  organization — multiple API keys only help if they're genuinely separate
  accounts, not just multiple keys on one.
- Google's PageSpeed Insights API has occasional transient outages;
  retried automatically, but can still fail on a bad day for Google.
- The smaller fallback model can still produce lower-quality prose in
  findings not covered by an existing deterministic rule. `--mode deep`
  avoids it entirely for the highest-confidence results.
- The system reports its known limitations in `data_limitations` on every run.

## Possible next steps

- Put the web app behind a domain with HTTPS, and move its job store out of
  process memory (Redis or the existing SQLite database) so results survive
  a restart and get a permanent link.
- Add a `crawl` mode that audits multiple pages of a site for a site-wide score.
- Extend `postprocess.py`'s deterministic-reconciliation pattern to other
  recurring hallucination classes as they're discovered.
- Add an "auto-fix" agent that drafts corrected meta tags / alt text.
- Keep growing the eval harness's benchmark pool (now 7) and Score
  Analytics' real dataset (currently well under the 20-row threshold for
  trustworthy real-data analysis) as more audits accumulate.
- Wire Findings Similarity Search into the live pipeline — e.g. giving the
  synthesizer retrieved similar past fixes as grounding context instead of
  writing recommendations from scratch each time.
- Wire the Critic-Approval Predictor into the live pipeline once its real
  dataset is large and balanced enough — e.g. flagging a low predicted
  approval probability before spending an actual critic call.
- Hook `python main.py eval` (and `analyze`) into CI as a scheduled job — the
  unit tests already run on every push, but `eval` exercises the real pipeline
  against live sites and needs API quota.