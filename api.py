"""
FastAPI backend + single-page web UI for the SEO Health Agent.

    docker compose up --build        # http://localhost:3003

Audits take a while (multiple LLM calls, sometimes a real Lighthouse call),
so this exposes an async job pattern instead of one long blocking request:

    POST /api/audit          -> starts a job, returns {"job_id": "..."}
    GET  /api/audit/{job_id} -> {"status": "running"|"done"|"error", ...}
    GET  /api/history/{url}  -> past audit scores for a domain

The page in web/ polls GET /api/audit/{job_id} every couple seconds until
status is "done" or "error".

This is meant to be reachable from the public internet, so starting an audit
is limited three ways: the target must be a public address (agent/netguard.py),
each client IP gets a fixed number of audits per hour, and only a few audits
run at once -- every audit spends shared Groq free-tier quota.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # MUST run before importing agent.* -- agent/config.py reads
                # env vars (like GROQ_API_KEYS) at import time, not lazily.

import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import run_full_audit
from agent import config
from agent import memory
from agent import netguard
from agent import tools

# Audits one client IP may start per hour. 0 disables the limit.
RATE_LIMIT_PER_HOUR = int(os.environ.get("SEO_API_RATE_LIMIT_PER_HOUR", 5))
RATE_LIMIT_WINDOW_SECONDS = 3600
# Audits allowed to run at the same time, across all clients.
MAX_CONCURRENT_AUDITS = int(os.environ.get("SEO_API_MAX_CONCURRENT", 2))
# How long a finished job's result stays retrievable.
JOB_TTL_SECONDS = int(os.environ.get("SEO_API_JOB_TTL_SECONDS", 3600))
# The page is served from this same origin, so no cross-origin access is
# needed by default. Comma-separate origins here to allow another frontend.
CORS_ORIGINS = [o.strip() for o in os.environ.get("SEO_API_CORS_ORIGINS", "").split(",") if o.strip()]

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="SEO Health Agent API")

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

# In-memory job store, guarded by one lock together with the rate-limit
# table. Fine for a single process; this is why the server must run with
# exactly one worker. Swap for Redis or a database before adding more.
_jobs: dict[str, dict] = {}
_recent_starts: dict[str, deque[float]] = {}
_jobs_lock = threading.Lock()


class AuditRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    mode: Literal["quick", "deep", "auto"] = "auto"
    competitor_url: str | None = Field(default=None, max_length=2048)


class AuditJobResponse(BaseModel):
    job_id: str


def _clean_url(raw: str, prefix: str = "") -> str:
    """Normalize a submitted URL and refuse anything that isn't an auditable
    public web address. Raises a 400 the page can show as-is; `prefix` says
    which field it was when it isn't the main one."""
    url = tools.normalize_url(raw.strip())
    try:
        hostname = urlparse(url).hostname or ""
    except ValueError:
        hostname = ""
    # A dotted name, an IPv6 literal, or localhost -- not a bare word.
    if not ("." in hostname or ":" in hostname or hostname == "localhost"):
        raise HTTPException(status_code=400, detail=f"{prefix}Enter a full web address, like example.com.")
    if netguard.enabled():
        reason = netguard.check_url(url)
        if reason:
            raise HTTPException(status_code=400, detail=f"{prefix}{reason}")
    return url


def _public_report(report: dict) -> dict:
    """The stored report carries internal working data under underscore
    keys (raw specialist output, the critic's reflection log). Nothing in
    the page uses it, so it isn't sent."""
    return {k: v for k, v in report.items() if not k.startswith("_")}


def _evict_expired_jobs(now: float) -> None:
    """Caller holds _jobs_lock."""
    expired = [
        job_id for job_id, job in _jobs.items()
        if job["status"] != "running" and now - (job["finished_at"] or now) > JOB_TTL_SECONDS
    ]
    for job_id in expired:
        del _jobs[job_id]
    for ip in [ip for ip, starts in _recent_starts.items()
               if not starts or now - starts[-1] > RATE_LIMIT_WINDOW_SECONDS]:
        del _recent_starts[ip]


def _finish_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields, finished_at=time.time())
        # The tool caches only make sense within one audit. Empty them once
        # nothing is running so the next audit of the same site is fresh.
        if not any(j["status"] == "running" for j in _jobs.values()):
            tools.clear_caches()


def _run_job(job_id: str, url: str, mode: str, competitor_url: str | None) -> None:
    def log_fn(message: str) -> None:
        with _jobs_lock:
            _jobs[job_id]["logs"].append(message)

    try:
        report = run_full_audit(url, competitor_url=competitor_url, mode=mode, log_fn=log_fn)
    except Exception as e:
        _finish_job(job_id, status="error", error=str(e))
    else:
        _finish_job(job_id, status="done", report=_public_report(report))


@app.post("/api/audit", response_model=AuditJobResponse)
def start_audit(req: AuditRequest, request: Request) -> AuditJobResponse:
    if not config.GROQ_API_KEYS:
        # Say so up front rather than letting every audit die on its first
        # model call with an SDK error the visitor can do nothing about.
        raise HTTPException(status_code=503, detail="The scanner isn't set up yet: no model API key is configured on the server.")
    url = _clean_url(req.url)
    competitor_url = _clean_url(req.competitor_url, "Competitor: ") if (req.competitor_url or "").strip() else None
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    with _jobs_lock:
        _evict_expired_jobs(now)

        # Someone asking for an audit that is already running gets that job
        # rather than a second copy of the same work.
        for job_id, job in _jobs.items():
            if (job["status"] == "running" and job["url"] == url and job["mode"] == req.mode
                    and job["competitor_url"] == competitor_url):
                return AuditJobResponse(job_id=job_id)

        running = sum(1 for j in _jobs.values() if j["status"] == "running")
        if running >= MAX_CONCURRENT_AUDITS:
            raise HTTPException(
                status_code=429,
                detail="The scanner is busy with other audits right now. Try again in a few minutes.",
            )

        if RATE_LIMIT_PER_HOUR > 0:
            starts = _recent_starts.setdefault(client_ip, deque())
            while starts and now - starts[0] > RATE_LIMIT_WINDOW_SECONDS:
                starts.popleft()
            if len(starts) >= RATE_LIMIT_PER_HOUR:
                wait_minutes = max(1, round((RATE_LIMIT_WINDOW_SECONDS - (now - starts[0])) / 60))
                raise HTTPException(
                    status_code=429,
                    detail=f"You've reached the limit of {RATE_LIMIT_PER_HOUR} audits per hour. "
                           f"Try again in about {wait_minutes} minute(s).",
                    headers={"Retry-After": str(wait_minutes * 60)},
                )
            starts.append(now)

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {
            "status": "running",
            "url": url,
            "mode": req.mode,
            "competitor_url": competitor_url,
            "started_at": now,
            "finished_at": None,
            "logs": [],
            "report": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_job, args=(job_id, url, req.mode, competitor_url), daemon=True
    )
    thread.start()

    return AuditJobResponse(job_id=job_id)


@app.get("/api/audit/{job_id}")
def get_audit(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return {**job, "logs": list(job["logs"])}


@app.get("/api/history/{domain}")
def get_history(domain: str, limit: int = Query(default=10, ge=1, le=50)) -> dict:
    rows = memory.get_history(domain, limit=limit)
    return {"domain": memory.domain_of(domain), "history": rows}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


# Last, so it never shadows the /api routes above.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
