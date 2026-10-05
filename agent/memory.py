"""
Persistent memory layer. Every completed audit is stored so future runs on
the same domain can reason about trends ("score dropped from 82 to 71 since
last week") instead of treating each audit as a stateless one-off. This is
what gives the agent long-term memory across separate invocations.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from .config import DB_PATH


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                url TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                overall_score REAL NOT NULL,
                grade TEXT,
                report_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audits_domain ON audits(domain)")
        # public_id is the unguessable handle a stored report is fetched by
        # from outside (row ids are sequential, so they would let anyone walk
        # through every report). Added after the table already existed in the
        # wild, hence the in-place migration and the backfill.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(audits)")}
        if "public_id" not in columns:
            try:
                conn.execute("ALTER TABLE audits ADD COLUMN public_id TEXT")
            except sqlite3.OperationalError:
                pass  # another thread added it between the check and here
        conn.execute("UPDATE audits SET public_id = lower(hex(randomblob(16))) WHERE public_id IS NULL")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_audits_public_id ON audits(public_id)")


def domain_of(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return urlparse(url).netloc


def _page_key(url: str) -> tuple[str, str, str]:
    """Identity of a page for trend purposes: the scheme and a trailing
    slash don't make it a different page. (www and non-www stay distinct,
    as they do everywhere else in this module.)"""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    return parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query


def save_audit(url: str, report: dict, public_id: str | None = None) -> int:
    init_db()
    domain = domain_of(url)
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO audits (domain, url, timestamp, overall_score, grade, report_json, public_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (domain, url, timestamp, report.get("overall_score", 0), report.get("grade", ""),
             json.dumps(report), public_id or uuid.uuid4().hex),
        )
        return cur.lastrowid


# How far back get_last_audit looks for a comparable earlier audit.
_BASELINE_LOOKBACK = 25


def get_last_audit(url: str) -> dict | None:
    """The most recent earlier audit this one can fairly be compared with:
    the same page (not merely the same domain -- /blog against the homepage
    is not a trend), and a complete one. An audit that lost categories was
    scored on fewer checks, so a delta against it says nothing about the
    site."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT url, timestamp, report_json FROM audits WHERE domain = ? "
            "ORDER BY id DESC LIMIT ?", (domain_of(url), _BASELINE_LOOKBACK),
        ).fetchall()
    wanted = _page_key(url)
    for row in rows:
        if _page_key(row["url"]) != wanted:
            continue
        try:
            report = json.loads(row["report_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if report.get("skipped_categories"):
            continue
        report["_timestamp"] = row["timestamp"]
        return report
    return None


def get_audit_by_public_id(public_id: str) -> dict | None:
    """A stored report by its public handle, with '_timestamp' and
    '_stored_url' merged in from the row. None if there is no such audit."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT url, timestamp, report_json FROM audits WHERE public_id = ?", (public_id,)
        ).fetchone()
    if not row:
        return None
    try:
        report = json.loads(row["report_json"])
    except (json.JSONDecodeError, TypeError):
        return None
    report["_timestamp"] = row["timestamp"]
    report["_stored_url"] = row["url"]
    return report


def get_history(url: str, limit: int = 10) -> list[dict]:
    init_db()
    domain = domain_of(url)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, public_id, url, timestamp, overall_score, grade FROM audits "
            "WHERE domain = ? ORDER BY id DESC LIMIT ?",
            (domain, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_full_audits(limit: int | None = None) -> list[dict]:
    """Return every stored audit's full parsed report (not just the summary
    columns get_history returns), across all domains -- this is the raw
    dataset agent/analytics.py trains on. Each report dict gets '_id',
    '_domain', and '_timestamp' merged in from the row, alongside whatever
    the report itself already contains (overall_score, grade, categories,
    etc.)."""
    init_db()
    with _connect() as conn:
        query = "SELECT id, domain, url, timestamp, report_json FROM audits ORDER BY id ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()

    reports = []
    for row in rows:
        try:
            report = json.loads(row["report_json"])
        except (json.JSONDecodeError, TypeError):
            continue  # skip a corrupted row rather than crash the whole dataset load
        report["_id"] = row["id"]
        report["_domain"] = row["domain"]
        report["_timestamp"] = row["timestamp"]
        reports.append(report)
    return reports