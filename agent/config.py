import os

# Groq's OpenAI-compatible chat model used by the planner, specialists,
# synthesizer, and critic. openai/gpt-oss-120b is Groq's named replacement
# for llama-3.3-70b-versatile, which this project was built and tuned on and
# which Groq shut down for free/developer-tier keys on 2026-08-16.
DEFAULT_MODEL = os.environ.get("SEO_AGENT_MODEL", "openai/gpt-oss-120b")
PLANNER_MODEL = os.environ.get("SEO_AGENT_PLANNER_MODEL", DEFAULT_MODEL)
CRITIC_MODEL = os.environ.get("SEO_AGENT_CRITIC_MODEL", DEFAULT_MODEL)

# Groq's per-model daily token quota (TPD) is tracked separately per model.
# If the primary model's quota is exhausted, agents automatically fall back
# to this smaller/faster model instead of failing outright, since it draws
# from a completely separate quota pool. Set to "" to disable fallback.
# (openai/gpt-oss-20b replaces llama-3.1-8b-instant, shut down the same day.)
FALLBACK_MODEL = os.environ.get("SEO_AGENT_FALLBACK_MODEL", "openai/gpt-oss-20b")

# Groq's built-in agentic "Compound" system, used only by the competitive/
# benchmarking specialist. It performs live web search server-side, so no
# custom tool schema is needed (and Groq does not allow mixing custom tools
# with Compound systems at this time).
# NOTE: Groq shut the Compound systems down on 2026-09-21 with no drop-in
# replacement, so this specialist currently fails (and its category is
# dropped) in auto/deep mode. Do not just point this at a plain chat model:
# the specialist's prompt assumes built-in search and it would invent its
# "research". It needs Groq's browser_search tool wired in first.
COMPETITIVE_MODEL = os.environ.get("SEO_AGENT_COMPETITIVE_MODEL", "groq/compound-mini")

# One or more Groq API keys, tried in order. Each key has its own separate
# daily quota, so a second key (e.g. from a second free Groq account) gives
# a fresh quota pool once the first is exhausted on both models. Comma-
# separate multiple keys: GROQ_API_KEYS=key1,key2. Falls back to the
# standard single GROQ_API_KEY if GROQ_API_KEYS isn't set.
_keys_env = os.environ.get("GROQ_API_KEYS")
if _keys_env:
    GROQ_API_KEYS = [k.strip() for k in _keys_env.split(",") if k.strip()]
else:
    _single_key = os.environ.get("GROQ_API_KEY")
    GROQ_API_KEYS = [_single_key] if _single_key else []

MAX_TOOL_ITERATIONS = int(os.environ.get("SEO_AGENT_MAX_ITER", 10))
MAX_REFLECTION_ROUNDS = int(os.environ.get("SEO_AGENT_MAX_REFLECTION_ROUNDS", 2))

# Groq's free tier shares one tokens-per-minute budget across all concurrent
# requests, so running many specialist agents at once easily triggers 429s.
# Keep this low by default; raise it if you're on a paid tier with more TPM.
MAX_PARALLEL_SPECIALISTS = int(os.environ.get("SEO_AGENT_MAX_WORKERS", 2))
SPECIALIST_DISPATCH_STAGGER_SECONDS = float(os.environ.get("SEO_AGENT_DISPATCH_STAGGER", 2.0))
RATE_LIMIT_MAX_RETRIES = int(os.environ.get("SEO_AGENT_RATE_LIMIT_RETRIES", 4))

DB_PATH = os.environ.get("SEO_AGENT_DB_PATH", os.path.join(os.getcwd(), "data", "audit_history.db"))

# Proactive payload compaction (agent/compaction.py) -- estimates the
# synthesizer/critic request size BEFORE sending and, only above this
# threshold, trims lower-priority findings and overlong free-text fields so
# the first request has a real chance of succeeding instead of relying on
# base_agent.py's reactive 413-triggered shrinking. Deliberately a rough
# characters-per-token estimate, not an exact count -- see compaction.py.
COMPACTION_TOKEN_THRESHOLD = int(os.environ.get("SEO_AGENT_COMPACTION_TOKEN_THRESHOLD", 4000))
COMPACTION_MAX_FINDINGS_PER_CATEGORY = int(os.environ.get("SEO_AGENT_COMPACTION_MAX_FINDINGS", 12))
COMPACTION_MAX_EVIDENCE_NOTES_CHARS = int(os.environ.get("SEO_AGENT_COMPACTION_MAX_EVIDENCE_CHARS", 600))
COMPACTION_MAX_FINDING_TEXT_CHARS = int(os.environ.get("SEO_AGENT_COMPACTION_MAX_FINDING_CHARS", 400))