"""
FastAPI backend + single-page web UI for the SEO Health Agent.

    docker compose up --build        # http://localhost:3003

Audits take a while (multiple LLM calls, sometimes a real Lighthouse call),
so this exposes an async job pattern instead of one long blocking request:

    POST /api/audit              -> starts a job, returns {"job_id": "..."}
    GET  /api/audit/{job_id}     -> {"status": "running"|"done"|"error", ...}
    GET  /api/audit/{job_id}/pdf -> the finished report as a PDF
    GET  /api/history/{url}      -> past audit scores for a domain

Running jobs live in memory. A finished report is also stored in SQLite under
its job id, so GET /api/audit/{job_id} keeps answering after the in-memory
job has expired or the server has restarted.

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

import io
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import run_full_audit
from agent import config
from agent import memory
from agent import netguard
from agent import tools
from agent.errors import AuditFailed, is_quota_error
from agent.report_pdf import export_report_pdf

# uvicorn configures this logger; anything logged here lands in the
# container's output next to the access log.
log = logging.getLogger("uvicorn.error")

# Audits one client IP may start per hour. 0 disables the limit.
RATE_LIMIT_PER_HOUR = int(os.environ.get("SEO_API_RATE_LIMIT_PER_HOUR", 5))
RATE_LIMIT_WINDOW_SECONDS = 3600
# Audits allowed to run at the same time, across all clients.
MAX_CONCURRENT_AUDITS = int(os.environ.get("SEO_API_MAX_CONCURRENT", 2))
# How long a finished job's result stays retrievable.
JOB_TTL_SECONDS = int(os.environ.get("SEO_API_JOB_TTL_SECONDS", 3600))
# Time budget for one audit. Past it the audit stops instead of sleeping
# through rate-limit waits while holding one of the few scan slots. 0 = none.
AUDIT_TIMEOUT_SECONDS = int(os.environ.get("SEO_API_AUDIT_TIMEOUT_SECONDS", 600))
# Progress lines kept per job, and how long one line may be.
MAX_JOB_LOG_LINES = 500
MAX_LOG_LINE_CHARS = 300
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


# What a visitor is told when an audit fails for a reason that is ours, not
# theirs. The real exception goes to the server log only: provider errors
# name the account's organization id and point at its billing page.
_PUBLIC_ERRORS = {
    "quota": "The models are out of free quota for now.",
    "internal": "The scan stopped because of an internal error.",
}
# Failures that are the server's doing give the visitor their hourly slot back.
_REFUNDED_ERROR_CODES = {"quota", "timeout", "internal"}

_PROVIDER_TAIL_RE = re.compile(r"\s*Groq's message:.*", re.DOTALL)
_ORG_ID_RE = re.compile(r"\borg_[A-Za-z0-9]+")
_JOB_ID_RE = re.compile(r"[0-9a-f]{32}")


def _public_log_line(message: str) -> str:
    """A progress line as the page may show it: no quoted provider error, no
    organization id, bounded length."""
    line = _PROVIDER_TAIL_RE.sub("", message)
    line = _ORG_ID_RE.sub("org_***", line)
    if len(line) > MAX_LOG_LINE_CHARS:
        line = line[:MAX_LOG_LINE_CHARS] + "..."
    return line


def _classify_failure(exc: Exception) -> tuple[str, str]:
    """(error_code, visitor-safe message) for an audit that raised."""
    if isinstance(exc, AuditFailed):
        return exc.code, exc.public_message
    code = "quota" if is_quota_error(exc) else "internal"
    return code, _PUBLIC_ERRORS[code]


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
        job = _jobs[job_id]
        job.update(fields, finished_at=time.time())
        if job["error_code"] in _REFUNDED_ERROR_CODES and job["counted_at"] is not None:
            try:
                _recent_starts.get(job["client_ip"], deque()).remove(job["counted_at"])
            except ValueError:
                pass  # already aged out of the window
        # The tool caches only make sense within one audit. Empty them once
        # nothing is running so the next audit of the same site is fresh.
        if not any(j["status"] == "running" for j in _jobs.values()):
            tools.clear_caches()


def _run_job(job_id: str, url: str, mode: str, competitor_url: str | None) -> None:
    def log_fn(message: str) -> None:
        log.info("[job %s] %s", job_id[:8], message)
        with _jobs_lock:
            logs = _jobs[job_id]["logs"]
            if len(logs) < MAX_JOB_LOG_LINES:
                logs.append(_public_log_line(message))

    try:
        report = run_full_audit(url, competitor_url=competitor_url, mode=mode, log_fn=log_fn,
                                audit_id=job_id, deadline_seconds=AUDIT_TIMEOUT_SECONDS or None)
    except Exception as e:
        code, message = _classify_failure(e)
        log.warning("[job %s] audit of %s failed (%s): %s", job_id[:8], url, code, e,
                    exc_info=not isinstance(e, AuditFailed))
        _finish_job(job_id, status="error", error=message, error_code=code)
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

        counted_at = None
        if RATE_LIMIT_PER_HOUR > 0:
            counted_at = now
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
            "error_code": None,
            # Private: used to refund the rate-limit slot, never returned.
            "client_ip": client_ip,
            "counted_at": counted_at,
        }

    thread = threading.Thread(
        target=_run_job, args=(job_id, url, req.mode, competitor_url), daemon=True
    )
    thread.start()

    return AuditJobResponse(job_id=job_id)


_PUBLIC_JOB_FIELDS = ("status", "url", "mode", "competitor_url", "started_at", "finished_at",
                      "report", "error", "error_code")


def _stored_job(job_id: str) -> dict | None:
    """A finished audit rebuilt from SQLite in the shape of a job, for ids no
    longer (or never) in this process's memory."""
    if not _JOB_ID_RE.fullmatch(job_id):
        return None
    report = memory.get_audit_by_public_id(job_id)
    if report is None:
        return None
    try:
        finished_at = datetime.fromisoformat(report["_timestamp"]).timestamp()
    except (TypeError, ValueError):
        finished_at = None
    duration = report.get("duration_seconds")
    has_times = finished_at is not None and isinstance(duration, (int, float))
    return {
        "status": "done",
        "url": report.get("url") or report["_stored_url"],
        "mode": report.get("mode"),
        "competitor_url": report.get("competitor_url"),
        "started_at": finished_at - duration if has_times else None,
        "finished_at": finished_at,
        "logs": [],
        "report": _public_report(report),
        "error": None,
        "error_code": None,
    }


def _find_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            return {**{k: job[k] for k in _PUBLIC_JOB_FIELDS}, "logs": list(job["logs"])}
    stored = _stored_job(job_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return stored


@app.get("/api/audit/{job_id}")
def get_audit(job_id: str) -> dict:
    return _find_job(job_id)


@app.get("/api/audit/{job_id}/pdf")
def get_audit_pdf(job_id: str) -> Response:
    job = _find_job(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=404, detail="This audit has no finished report.")
    buffer = io.BytesIO()
    export_report_pdf(job["report"], buffer)
    host = re.sub(r"[^A-Za-z0-9.-]", "_", urlparse(job["url"] or "").hostname or "site")
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="seo-report-{host}.pdf"'},
    )


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
