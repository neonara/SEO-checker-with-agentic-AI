"""
Outbound-request guard for when this runs as a public web service.

Every audit makes the server fetch URLs chosen by someone else: the URL a
visitor submits, and then whatever URLs the model decides to pass to its
tools (which page content can steer). Without a check, that lets anyone use
the server to reach things only the server can see -- loopback services,
the private network, the cloud metadata endpoint -- and read the result back
out of the audit report.

Off by default so the CLI can still audit a localhost dev site and the test
suite stays offline. Turned on with SEO_AGENT_BLOCK_PRIVATE_HOSTS=1, which
the Docker image sets.

Known gap: the hostname is resolved here and again by the HTTP client, so a
DNS record that changes between the two lookups (DNS rebinding) is not caught.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import time
from urllib.parse import urlparse

ALLOWED_SCHEMES = ("http", "https")
# Default web ports plus the common alternates. Anything else is far more
# likely to be a probe of a non-web service than a site someone wants audited.
ALLOWED_PORTS = {80, 443, 8080, 8443}
# A lookup that times out (EAI_AGAIN) says nothing about whether the site
# exists: slow nameservers do it routinely. Tried this many times in all.
DNS_ATTEMPTS = 3
DNS_RETRY_DELAY = 0.5


def enabled() -> bool:
    return os.environ.get("SEO_AGENT_BLOCK_PRIVATE_HOSTS", "").strip().lower() in ("1", "true", "yes", "on")


def _is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_host(hostname: str) -> str | None:
    """Return why `hostname` must not be contacted, or None if it is fine.
    Every address the name resolves to has to be public -- one private
    record among several is enough to refuse."""
    hostname = (hostname or "").strip().strip("[]").rstrip(".")
    if not hostname:
        return "URL has no hostname."
    for attempt in range(1, DNS_ATTEMPTS + 1):
        try:
            infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            break
        except socket.gaierror as e:
            if e.errno != socket.EAI_AGAIN:
                return f"No website found at '{hostname}'. Check the spelling."
            if attempt == DNS_ATTEMPTS:
                return f"Could not look up '{hostname}' right now. Try again in a moment."
            time.sleep(DNS_RETRY_DELAY)
        except UnicodeError:
            return f"No website found at '{hostname}'. Check the spelling."
    addresses = {info[4][0].split("%")[0] for info in infos}
    if not addresses:
        return f"No website found at '{hostname}'. Check the spelling."
    if not all(_is_public_ip(a) for a in addresses):
        return f"Host '{hostname}' is not a public internet address."
    return None


def check_url(url: str) -> str | None:
    """Return why `url` must not be fetched, or None if it is fine."""
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return "URL is not valid."
    if parsed.scheme not in ALLOWED_SCHEMES:
        return "Only http:// and https:// URLs can be audited."
    if port is not None and port not in ALLOWED_PORTS:
        return f"Port {port} is not allowed."
    return check_host(parsed.hostname or "")
