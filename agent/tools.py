"""
Concrete implementations of every tool an agent can call. Pure, testable
functions -- each returns a JSON-serializable dict so it can be dropped
straight into a tool-result message for the agent loop.
"""
from __future__ import annotations

import os
import re
import ssl
import socket
import threading
import certifi
import time
import json
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

from . import netguard

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 10

# Most a single response body may take in memory. The urls come from visitors
# and from the model, so without a cap one link to a huge file is enough to
# exhaust the server. Far above any real HTML page, robots.txt or sitemap.
MAX_PAGE_BYTES = int(os.environ.get("SEO_AGENT_MAX_PAGE_BYTES", 5_000_000))

# Server-side cache: url -> raw HTML. This exists so fetched HTML never has to
# be sent through the model's context (which was previously blowing through
# the free-tier token budget in one or two tool calls). Tools that need the
# HTML look it up here instead of receiving it as a function argument.
_page_cache: dict[str, str] = {}

# url -> the result fetch_page already returned for it during this audit.
# Every specialist starts with fetch_page on the same url; without this the
# target was requested once per specialist (seven times an audit), which is
# slower, more likely to trip bot protection halfway through, and could leave
# specialists disagreeing about whether the site was blocked.
_fetch_result_cache: dict[str, dict] = {}

# One lock per cache key, so concurrent specialists asking for the same thing
# wait for the first fetch instead of each starting their own.
_key_locks: dict[tuple, threading.Lock] = {}
_key_locks_guard = threading.Lock()


def _lock_for(*key) -> threading.Lock:
    with _key_locks_guard:
        return _key_locks.setdefault(key, threading.Lock())

# Status codes commonly returned by bot/WAF protection (Cloudflare "Are You
# Human" challenges, rate limiters, etc.) rather than reflecting a genuine
# site problem. When we see one, the fetched "content" is very likely a
# block/challenge page, not the real site -- callers should not treat it as
# representative of the site's actual SEO quality.
_LIKELY_BLOCK_STATUS_CODES = {401, 403, 429, 503}


def normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _blocked_reason(url: str) -> str | None:
    """Why this url must not be fetched, or None. Always None unless the
    outbound guard is switched on (see netguard.py) -- the model picks the
    urls these tools receive, so the check has to live here and not only
    where the visitor's url first comes in."""
    return netguard.check_url(url) if netguard.enabled() else None


def _refuse_private_redirect(resp, *args, **kwargs):
    # requests runs this on every hop, before following it.
    if resp.is_redirect:
        target = urljoin(resp.url, resp.headers.get("location", ""))
        reason = netguard.check_url(target)
        if reason:
            raise requests.exceptions.InvalidURL(f"Redirect refused: {reason}")


def _guard_kwargs() -> dict:
    """Extra requests kwargs that re-check each redirect hop when the guard
    is on; nothing at all when it is off."""
    return {"hooks": {"response": _refuse_private_redirect}} if netguard.enabled() else {}


def clear_caches() -> None:
    """Drop the server-side HTML and Lighthouse caches. They exist to share
    data between the specialists of ONE audit; a long-running server has to
    empty them between audits or a re-audit hours later would silently be
    scored on the first audit's stale data (and memory would only grow)."""
    _page_cache.clear()
    _fetch_result_cache.clear()
    _lighthouse_cache.clear()
    _lighthouse_errors.clear()
    with _key_locks_guard:
        _key_locks.clear()


def _get_capped(url: str, timeout: float = DEFAULT_TIMEOUT, **kwargs):
    """GET `url`, reading at most MAX_PAGE_BYTES of the body. Returns
    (response, text, body_bytes_read, truncated)."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=True, **kwargs)
    chunks, size, truncated = [], 0, False
    # `timeout` only bounds each read, so a server trickling bytes could hold
    # this open indefinitely; give the whole body a wall-clock budget too.
    give_up_at = time.time() + timeout * 3
    try:
        for chunk in resp.iter_content(chunk_size=65536):
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_PAGE_BYTES or time.time() > give_up_at:
                truncated = True
                break
    finally:
        resp.close()
    body = b"".join(chunks)[:MAX_PAGE_BYTES]
    # Trust a charset the server declared; otherwise assume UTF-8 (requests'
    # own default for undeclared text/* is ISO-8859-1, which garbles most
    # modern pages).
    declared = "charset=" in (resp.headers.get("Content-Type") or "").lower()
    encoding = resp.encoding if declared and isinstance(resp.encoding, str) else "utf-8"
    try:
        text = body.decode(encoding, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    return resp, text, len(body), truncated


def fetch_page(url: str) -> dict:
    """Fetch a page, cache its HTML server-side, and return only lightweight
    metadata to the model (status, timing, headers) -- NOT the HTML itself.
    Call parse_seo_elements with the same url to extract SEO signals."""
    url = normalize_url(url)
    blocked = _blocked_reason(url)
    if blocked:
        return {"ok": False, "requested_url": url, "error": blocked}
    with _lock_for("page", url):
        cached = _fetch_result_cache.get(url)
        if cached is not None and url in _page_cache:
            return cached
        result = _fetch_page_uncached(url)
        if result.get("ok"):
            _fetch_result_cache[url] = result
        return result


def _fetch_page_uncached(url: str) -> dict:
    try:
        start = time.time()
        resp, text, body_bytes, truncated = _get_capped(url, allow_redirects=True, **_guard_kwargs())
        elapsed_ms = round((time.time() - start) * 1000, 1)
        _page_cache[url] = text
        _page_cache[resp.url] = text

        result = {
            "ok": True,
            "requested_url": url,
            "final_url": resp.url,
            "redirected": resp.url != url,
            "redirect_chain_length": len(resp.history),
            "status_code": resp.status_code,
            "response_time_ms": elapsed_ms,
            "content_length_bytes": body_bytes,
            "headers": dict(resp.headers),
            "note": "HTML fetched and cached server-side. Call parse_seo_elements(url=...) "
                    "with this same url to extract SEO signals -- do not request raw HTML.",
        }

        if resp.status_code in _LIKELY_BLOCK_STATUS_CODES:
            result["likely_blocked"] = True
            result["block_warning"] = (
                f"Status code {resp.status_code} commonly indicates the site is blocking "
                f"automated requests (bot/WAF/anti-bot protection) rather than reflecting a "
                f"genuine site problem. The fetched content, if any, is very likely a "
                f"block/challenge page, NOT the real page. Do NOT score or report on it as if "
                f"it were the site's actual content -- report a single critical finding noting "
                f"suspected blocking instead of a normal quality assessment."
            )
        if truncated:
            result["truncated"] = True
            result["truncated_note"] = (
                f"The response was larger than {MAX_PAGE_BYTES} bytes; only the first part "
                f"was read, so content_length_bytes is a lower bound."
            )

        return result
    except requests.exceptions.RequestException as e:
        return {"ok": False, "requested_url": url, "error": str(e)}


def _get_cached_html(url: str) -> str | None:
    """Look up previously-fetched HTML, fetching fresh if not cached."""
    url_n = normalize_url(url)
    if url_n in _page_cache:
        return _page_cache[url_n]
    result = fetch_page(url_n)
    if result.get("ok"):
        return _page_cache.get(result["final_url"]) or _page_cache.get(url_n)
    return None


def parse_seo_elements(url: str) -> dict:
    """Extract on-page SEO signals for a previously-fetched url. Looks up the
    server-side HTML cache populated by fetch_page (fetches fresh if needed)."""
    html = _get_cached_html(url)
    if html is None:
        return {"ok": False, "error": f"Could not retrieve HTML for {url}. Try fetch_page first."}
    base_url = normalize_url(url)

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    meta_robots = soup.find("meta", attrs={"name": "robots"})
    canonical = soup.find("link", attrs={"rel": "canonical"})
    viewport = soup.find("meta", attrs={"name": "viewport"})
    charset = soup.find("meta", attrs={"charset": True})

    headings = {f"h{i}": len(soup.find_all(f"h{i}")) for i in range(1, 7)}
    h1_texts = [h.get_text(strip=True) for h in soup.find_all("h1")][:5]

    images = soup.find_all("img")
    images_missing_alt = [img.get("src", "")[:100] for img in images if not img.get("alt")]

    links = soup.find_all("a", href=True)
    domain = urlparse(base_url).netloc
    internal_links, external_links = [], []
    for a in links:
        href = a["href"]
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full = urljoin(base_url, href)
        (internal_links if urlparse(full).netloc == domain else external_links).append(full)

    json_ld_blocks = soup.find_all("script", attrs={"type": "application/ld+json"})
    structured_data_types = []
    for block in json_ld_blocks:
        try:
            data = json.loads(block.string or "{}")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and "@type" in item:
                    structured_data_types.append(item["@type"])
        except Exception:
            continue

    text = soup.get_text(separator=" ", strip=True)
    word_count = len(text.split())

    og_tags = {
        m.get("property"): m.get("content")
        for m in soup.find_all("meta", property=lambda p: p and p.startswith("og:"))
    }

    return {
        "title": title_tag.get_text(strip=True) if title_tag else None,
        "title_length": len(title_tag.get_text(strip=True)) if title_tag else 0,
        "meta_description": meta_desc.get("content") if meta_desc else None,
        "meta_description_length": len(meta_desc.get("content")) if meta_desc and meta_desc.get("content") else 0,
        "meta_robots": meta_robots.get("content") if meta_robots else None,
        "canonical_url": canonical.get("href") if canonical else None,
        "has_viewport_meta": viewport is not None,
        "viewport_content": viewport.get("content") if viewport else None,
        "has_charset_meta": charset is not None,
        "heading_counts": headings,
        "h1_texts": h1_texts,
        "total_images": len(images),
        "images_missing_alt_count": len(images_missing_alt),
        "images_missing_alt_sample": images_missing_alt[:10],
        "internal_link_count": len(internal_links),
        "external_link_count": len(external_links),
        "internal_links_sample": internal_links[:15],
        "external_links_sample": external_links[:10],
        "structured_data_types_found": structured_data_types,
        "open_graph_tags": og_tags,
        "word_count": word_count,
        "script_tag_count": len(soup.find_all("script")),
        "stylesheet_count": len(soup.find_all("link", rel="stylesheet")),
    }


def fetch_robots_txt(domain_or_url: str) -> dict:
    parsed = urlparse(normalize_url(domain_or_url))
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    blocked = _blocked_reason(robots_url)
    if blocked:
        return {"ok": False, "url": robots_url, "error": blocked}
    try:
        resp, text, _, _ = _get_capped(robots_url, **_guard_kwargs())
        return {
            "ok": True,
            "url": robots_url,
            "status_code": resp.status_code,
            "exists": resp.status_code == 200,
            "content": text[:5000] if resp.status_code == 200 else None,
            "mentions_sitemap": "sitemap:" in text.lower() if resp.status_code == 200 else False,
        }
    except requests.exceptions.RequestException as e:
        return {"ok": False, "url": robots_url, "error": str(e)}


def fetch_sitemap(domain_or_url: str) -> dict:
    parsed = urlparse(normalize_url(domain_or_url))
    candidates = [
        f"{parsed.scheme}://{parsed.netloc}/sitemap.xml",
        f"{parsed.scheme}://{parsed.netloc}/sitemap_index.xml",
    ]
    blocked = _blocked_reason(candidates[0])
    if blocked:
        return {"ok": False, "exists": False, "error": blocked}
    for sitemap_url in candidates:
        try:
            resp, text, _, _ = _get_capped(sitemap_url, **_guard_kwargs())
            if resp.status_code == 200 and "xml" in resp.headers.get("Content-Type", "").lower():
                url_count = text.count("<url>") + text.count("<sitemap>")
                return {
                    "ok": True,
                    "url": sitemap_url,
                    "exists": True,
                    "status_code": resp.status_code,
                    "entry_count_estimate": url_count,
                    "sample_content": text[:2000],
                }
        except requests.exceptions.RequestException:
            continue
    return {"ok": True, "exists": False, "checked_urls": candidates}


def check_ssl_certificate(domain_or_url: str) -> dict:
    parsed = urlparse(normalize_url(domain_or_url))
    hostname = parsed.netloc.split(":")[0]
    if netguard.enabled():
        blocked = netguard.check_host(hostname)
        if blocked:
            return {"ok": False, "error": blocked}
    try:
        # Use certifi's actively-maintained CA bundle (same one `requests` uses
        # internally) instead of relying solely on the OS trust store, which can
        # lag behind on some systems and cause false "certificate expired"
        # verification failures for perfectly valid certificates.
        ctx = ssl.create_default_context(cafile=certifi.where())
        with socket.create_connection((hostname, 443), timeout=DEFAULT_TIMEOUT) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        expires_raw = cert.get("notAfter")
        is_expired = None
        days_until_expiry = None
        summary = "SSL status could not be determined."
        if expires_raw:
            expires_dt = datetime.strptime(expires_raw, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            is_expired = expires_dt < now
            days_until_expiry = (expires_dt - now).days
            summary = (
                f"CERTIFICATE IS EXPIRED (expired {-days_until_expiry} day(s) ago)."
                if is_expired else
                f"Certificate is VALID and NOT expired ({days_until_expiry} day(s) remaining "
                f"until it expires on {expires_raw})."
            )

        return {
            "ok": True,
            "has_valid_ssl": True,
            "issuer": dict(x[0] for x in cert.get("issuer", [])),
            "expires": expires_raw,
            "is_expired": is_expired,
            "days_until_expiry": days_until_expiry,
            "ssl_status_summary": summary,
            "subject": dict(x[0] for x in cert.get("subject", [])),
        }
    except Exception as e:
        return {"ok": True, "has_valid_ssl": False, "error": str(e)}


def analyze_security_headers(headers: dict) -> dict:
    """Evaluate HTTP response headers (from fetch_page) for security best practices."""
    looks_malformed = (
        not isinstance(headers, dict)
        or not headers
        or {"function", "args"}.issubset(headers.keys())
        or {"function", "arguments"}.issubset(headers.keys())
        or "url" in headers
        or any(not isinstance(v, str) for v in headers.values())
    )
    if looks_malformed:
        return {
            "ok": False,
            "error": (
                "Invalid 'headers' argument. Pass the exact 'headers' object returned "
                "by fetch_page (a flat dict of header-name -> header-value strings), "
                "e.g. {\"Content-Type\": \"text/html\", \"Strict-Transport-Security\": \"...\"}. "
                "Call fetch_page first if you haven't already, then copy its result.headers verbatim."
            ),
        }

    lower_headers = {k.lower(): v for k, v in headers.items()}

    checks = {
        "strict-transport-security": "HSTS forces browsers to use HTTPS.",
        "content-security-policy": "CSP mitigates XSS and data-injection attacks.",
        "x-content-type-options": "Prevents MIME-sniffing (should be 'nosniff').",
        "x-frame-options": "Mitigates clickjacking (should be 'DENY' or 'SAMEORIGIN').",
        "referrer-policy": "Controls how much referrer info is leaked cross-origin.",
        "permissions-policy": "Restricts access to browser features/APIs.",
    }

    present, missing = {}, []
    for header, explanation in checks.items():
        if header in lower_headers:
            present[header] = lower_headers[header]
        else:
            missing.append({"header": header, "why_it_matters": explanation})

    return {
        "present_headers": present,
        "missing_headers": missing,
        "server_banner": lower_headers.get("server"),
        "powered_by_banner": lower_headers.get("x-powered-by"),
    }


PAGESPEED_API_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

# Server-side cache for full PageSpeed Insights/Lighthouse responses, keyed by
# (normalized_url, strategy). Google's API accepts multiple &category=
# params in a single request, so check_core_web_vitals and
# check_accessibility_and_best_practices both request all four categories
# (performance, accessibility, best-practices, seo) and share one cached
# response instead of each specialist paying for its own PSI call --
# whichever tool runs first for a given url+strategy fetches it, the other
# reads the cache.
_lighthouse_cache: dict[tuple[str, str], dict] = {}

# Why the PageSpeed call failed for a url+strategy during this audit. Three
# specialists ask for the same Lighthouse run; once it has failed after its
# retries, the others get this answer at once instead of each spending
# another round of retries (minutes, on a timeout) on the same failure.
_lighthouse_errors: dict[tuple[str, str], str] = {}

_API_KEY_PARAM_RE = re.compile(r"key=[^&\s'\")]+")

_ALL_LIGHTHOUSE_CATEGORIES = ["performance", "accessibility", "best-practices", "seo"]


def _fetch_lighthouse(url: str, strategy: str, categories: list[str]) -> dict:
    """Call the PageSpeed Insights API for the given Lighthouse categories,
    retrying transient 5xx errors. Returns {"ok": True, "data": <raw PSI
    JSON>} on success, or {"ok": False, "error": ...} on failure. Callers
    are responsible for caching.

    Google's own docs note transient 500 "Unable to process request, please
    wait a while and try again" errors are common for this API, especially
    on heavier/complex pages -- retried automatically here rather than
    relying on the model to notice and retry on its own."""
    api_key = os.environ.get("GOOGLE_PAGESPEED_API_KEY")
    params = [("url", normalize_url(url)), ("strategy", strategy)]
    for category in categories:
        params.append(("category", category))
    if api_key:
        params.append(("key", api_key))

    max_attempts = 3
    retry_delay_seconds = 6
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(PAGESPEED_API_URL, params=params, timeout=45)
            if resp.status_code == 200:
                return {"ok": True, "data": resp.json()}

            last_error = f"PageSpeed Insights API returned {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            return {"ok": False, "error": last_error}
        except (requests.exceptions.RequestException, ValueError) as e:
            # A connection error's text quotes the request url, API key included.
            last_error = _API_KEY_PARAM_RE.sub("key=***", str(e))
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            return {"ok": False, "error": last_error}

    return {"ok": False, "error": last_error}


def _get_lighthouse_data(url: str, strategy: str) -> dict | None:
    """Return the cached full-Lighthouse-run PSI response for this
    url+strategy, fetching (and caching on success only) if not already
    present. Returns None on failure -- callers build their own
    tool-specific error response in that case (see _lighthouse_error).
    A failure is remembered until clear_caches(), i.e. for the rest of the
    audit."""
    cache_key = (normalize_url(url), strategy)
    with _lock_for("lighthouse", *cache_key):
        cached = _lighthouse_cache.get(cache_key)
        if cached is not None:
            return cached
        if cache_key in _lighthouse_errors:
            return None

        fetched = _fetch_lighthouse(url, strategy, _ALL_LIGHTHOUSE_CATEGORIES)
        if not fetched.get("ok"):
            _lighthouse_errors[cache_key] = fetched.get("error") or "unknown error"
            return None

        _lighthouse_cache[cache_key] = fetched["data"]
        return fetched["data"]


def _lighthouse_error(url: str, strategy: str) -> str:
    reason = _lighthouse_errors.get((normalize_url(url), strategy), "unknown error")
    return f"PageSpeed Insights API call failed: {reason}"


def prefetch_lighthouse(url: str, strategy: str = "mobile") -> None:
    """Warm the Lighthouse cache. The orchestrator runs this in the
    background as an audit starts: the PageSpeed call takes 10-30+ seconds,
    and done here it overlaps with the first specialists instead of holding
    up the one that asks for it."""
    try:
        _get_lighthouse_data(url, strategy)
    except Exception:
        pass  # the specialist's own call will report whatever is wrong


def check_core_web_vitals(url: str, strategy: str = "mobile") -> dict:
    """Run a real Lighthouse audit via Google's PageSpeed Insights API and
    return both lab data (LCP, CLS, INP proxy via TBT, Speed Index,
    Performance score) and, when available, real-world field data (CrUX --
    actual Chrome user metrics for this site). This is a genuine Lighthouse
    audit, not a proxy signal -- can take 10-30+ seconds since Google runs it
    server-side. Requires network access to googleapis.com."""
    data = _get_lighthouse_data(url, strategy)
    if data is None:
        return {
            "ok": False,
            "error": _lighthouse_error(url, strategy),
            "note": "Real Core Web Vitals data unavailable this run -- fall back to the "
                    "proxy signals from fetch_page/parse_seo_elements instead.",
        }

    lighthouse = data.get("lighthouseResult", {})
    audits = lighthouse.get("audits", {})
    perf_category = lighthouse.get("categories", {}).get("performance", {})

    def _audit_value(key, field="numericValue"):
        audit = audits.get(key)
        return audit.get(field) if audit else None

    lab_data = {
        "performance_score_0_100": round(perf_category["score"] * 100) if perf_category.get("score") is not None else None,
        "lcp_ms": _audit_value("largest-contentful-paint"),
        "cls": _audit_value("cumulative-layout-shift"),
        "total_blocking_time_ms": _audit_value("total-blocking-time"),  # proxy for INP in lab data
        "speed_index_ms": _audit_value("speed-index"),
        "first_contentful_paint_ms": _audit_value("first-contentful-paint"),
        "time_to_interactive_ms": _audit_value("interactive"),
    }

    field_data = None
    loading_experience = data.get("loadingExperience", {})
    metrics = loading_experience.get("metrics")
    if metrics:
        field_data = {
            "source": "Chrome UX Report (real user data for this site, last 28 days)",
            "overall_category": loading_experience.get("overall_category"),
            "lcp_ms_p75": metrics.get("LARGEST_CONTENTFUL_PAINT_MS", {}).get("percentile"),
            "cls_p75": metrics.get("CUMULATIVE_LAYOUT_SHIFT_SCORE", {}).get("percentile"),
            "inp_ms_p75": metrics.get("INTERACTION_TO_NEXT_PAINT", {}).get("percentile"),
            "fcp_ms_p75": metrics.get("FIRST_CONTENTFUL_PAINT_MS", {}).get("percentile"),
        }

    # Current (2026) Google "Good" thresholds, for the model to compare against
    # without having to search for or guess at them.
    thresholds = {"lcp_ms": 2500, "cls": 0.1, "inp_ms": 200}

    return {
        "ok": True,
        "strategy": strategy,
        "lab_data_note": "From a real Lighthouse run (server-side, this exact moment).",
        "lab_data": lab_data,
        "field_data": field_data,
        "field_data_note": (
            "Real Chrome User Experience Report data -- only available for sites with enough "
            "Chrome traffic. None available for this site/URL." if field_data is None else
            "Real-world data from actual Chrome users visiting this site."
        ),
        "good_thresholds_2026": thresholds,
    }


def _extract_category_score_and_failures(data: dict, category_key: str, exclude_audit_ids: set[str] | None = None) -> tuple[int | None, list[dict]]:
    """Shared extraction logic for any Lighthouse category: returns
    (0-100 score, [failing audits]). A failing audit is one Lighthouse
    scored 0 (pass/fail check that failed) -- audits scored None are
    informative/non-scored and excluded, same as audits scored 1 (passed).
    exclude_audit_ids lets a caller deterministically drop specific audits
    that belong to another specialist's domain (e.g. "is-on-https" belongs
    to the Security specialist, not Best Practices) rather than relying on
    a prompt instruction to ignore them."""
    exclude_audit_ids = exclude_audit_ids or set()
    lighthouse = data.get("lighthouseResult", {})
    categories = lighthouse.get("categories", {})
    audits = lighthouse.get("audits", {})

    category = categories.get(category_key, {})
    score = round(category["score"] * 100) if category.get("score") is not None else None

    failing_audits = []
    for ref in category.get("auditRefs", []):
        audit_id = ref.get("id")
        if audit_id in exclude_audit_ids:
            continue
        audit = audits.get(audit_id)
        if audit and audit.get("score") == 0:
            failing_audits.append({
                "id": audit_id,
                "title": audit.get("title"),
                "description": (audit.get("description") or "")[:300],
            })

    return score, failing_audits


def check_accessibility_and_best_practices(url: str, strategy: str = "mobile") -> dict:
    """Run a real Lighthouse audit (same PageSpeed Insights call as
    check_core_web_vitals -- shares its cache, so calling both for the same
    url/strategy costs at most one extra API request) and return the
    Accessibility category: a 0-100 score plus the specific WCAG-adjacent
    checks that failed (color contrast, missing alt text, unlabeled form
    fields, missing ARIA attributes, etc.), each with Lighthouse's own audit
    id/title/description. This is real automated-audit data, not a guess --
    but automated tools only catch roughly 30-50% of real WCAG issues, so
    treat this as a floor, not a substitute for manual/assistive-tech review."""
    data = _get_lighthouse_data(url, strategy)
    if data is None:
        return {
            "ok": False,
            "error": _lighthouse_error(url, strategy),
            "note": "Real accessibility audit data unavailable this run.",
        }

    accessibility_score, failing_audits = _extract_category_score_and_failures(data, "accessibility")

    return {
        "ok": True,
        "strategy": strategy,
        "accessibility_score_0_100": accessibility_score,
        "failing_accessibility_audits": failing_audits[:12],
        "note": (
            "Real Lighthouse accessibility audit (WCAG-adjacent automated checks), not a "
            "manual accessibility audit -- automated tools catch roughly 30-50% of real WCAG "
            "issues, so a high score here is NOT proof of full accessibility compliance. "
            "Report it as a floor / starting point, and recommend manual or assistive-tech "
            "testing for anything beyond these specific failures."
        ),
    }


def check_best_practices(url: str, strategy: str = "mobile") -> dict:
    """Run a real Lighthouse audit (same PageSpeed Insights call as
    check_core_web_vitals/check_accessibility_and_best_practices -- shares
    its cache, so calling this alongside either costs at most one extra API
    request) and return the Best Practices category: a 0-100 score plus the
    specific checks that failed (known-vulnerable JS libraries, deprecated
    browser APIs, missing charset/doctype, unsafe third-party patterns,
    console errors, etc.), each with Lighthouse's own audit id/title/
    description.

    Deliberately excludes HTTPS/SSL-related audits ("is-on-https",
    "redirects-http") even though Lighthouse's Best Practices category
    includes them -- that's the Security specialist's domain, which already
    runs a dedicated, more authoritative SSL check
    (check_ssl_certificate). Excluding it here in code, rather than only
    telling the model not to mention it, avoids the exact
    same-fact-from-two-sources contradiction bug this project has already
    had to fix once for the Competitive specialist's on-page overlap."""
    data = _get_lighthouse_data(url, strategy)
    if data is None:
        return {
            "ok": False,
            "error": _lighthouse_error(url, strategy),
            "note": "Real best-practices audit data unavailable this run.",
        }

    score, failing_audits = _extract_category_score_and_failures(
        data, "best-practices", exclude_audit_ids={"is-on-https", "redirects-http"}
    )

    return {
        "ok": True,
        "strategy": strategy,
        "best_practices_score_0_100": score,
        "failing_best_practices_audits": failing_audits[:12],
        "note": (
            "Real Lighthouse Best Practices audit (browser/platform-level checks: vulnerable "
            "JS libraries, deprecated APIs, unsafe patterns, console errors, etc.). HTTPS/SSL "
            "checks are deliberately excluded from this data -- that's covered by the "
            "Security specialist's dedicated SSL check instead."
        ),
    }


def check_links_status(urls: list[str]) -> dict:
    """HEAD (fallback GET) a small sample of links to find broken ones."""
    results = []
    for url in urls[:10]:
        # Only http(s) links are guarded; anything else falls through to
        # requests and fails there exactly as it did before.
        blocked = _blocked_reason(url) if url.startswith(("http://", "https://")) else None
        if blocked:
            # Not checked is not the same as broken -- don't report it as one.
            results.append({"url": url, "status_code": None, "broken": False, "skipped": blocked})
            continue
        try:
            resp = requests.head(url, headers={"User-Agent": USER_AGENT}, timeout=6, allow_redirects=True,
                                 **_guard_kwargs())
            if resp.status_code >= 400:
                # Only the status matters: stream and close, never download the body.
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=6, allow_redirects=True,
                                    stream=True, **_guard_kwargs())
                resp.close()
            results.append({"url": url, "status_code": resp.status_code, "broken": resp.status_code >= 400})
        except requests.exceptions.RequestException as e:
            results.append({"url": url, "status_code": None, "broken": True, "error": str(e)})

    broken_count = sum(1 for r in results if r["broken"])
    broken_fraction = broken_count / len(results) if results else 0
    warning = None
    if broken_fraction >= 0.5 and len(results) >= 3:
        warning = (
            f"CAUTION: {broken_count}/{len(results)} sampled links failed at once. A failure rate this "
            f"high on a small sample often indicates the target site is blocking automated requests "
            f"(bot/WAF protection) rather than the links being genuinely broken. Report this as a "
            f"possible false positive needing manual verification, not a confirmed critical issue, "
            f"unless the errors clearly indicate real 404s (e.g. consistent 'page not found' content)."
        )

    return {
        "checked": len(results),
        "broken_count": broken_count,
        "results": results,
        "high_failure_rate_warning": warning,
    }