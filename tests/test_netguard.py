"""Tests for agent/netguard.py. Name resolution is faked; IP literals go
through the real resolver, which needs no network for them."""
import socket

import pytest

from agent import netguard


def resolves_to(*ips):
    """A fake getaddrinfo answering every lookup with these addresses."""
    def fake(host, *args, **kwargs):
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
                for ip in ips]
    return fake


class TestEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("SEO_AGENT_BLOCK_PRIVATE_HOSTS", raising=False)
        assert netguard.enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_truthy_values_turn_it_on(self, monkeypatch, value):
        monkeypatch.setenv("SEO_AGENT_BLOCK_PRIVATE_HOSTS", value)
        assert netguard.enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_falsy_values_leave_it_off(self, monkeypatch, value):
        monkeypatch.setenv("SEO_AGENT_BLOCK_PRIVATE_HOSTS", value)
        assert netguard.enabled() is False


class TestCheckHost:
    @pytest.mark.parametrize("address", [
        "127.0.0.1",         # loopback
        "10.0.0.5",          # private
        "172.16.3.4",        # private
        "192.168.1.1",       # private
        "169.254.169.254",   # link-local: cloud metadata endpoint
        "100.64.0.1",        # carrier-grade NAT shared space
        "0.0.0.0",           # unspecified
        "224.0.0.1",         # multicast
        "::1",               # IPv6 loopback
        "fe80::1",           # IPv6 link-local
        "fd00::1",           # IPv6 unique local
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
    ])
    def test_non_public_literals_are_refused(self, address):
        assert "not a public internet address" in netguard.check_host(address)

    @pytest.mark.parametrize("address", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"])
    def test_public_literals_are_allowed(self, address):
        assert netguard.check_host(address) is None

    def test_name_resolving_to_public_address_is_allowed(self, monkeypatch):
        monkeypatch.setattr(netguard.socket, "getaddrinfo", resolves_to("93.184.216.34"))
        assert netguard.check_host("example.com") is None

    def test_name_resolving_to_private_address_is_refused(self, monkeypatch):
        monkeypatch.setattr(netguard.socket, "getaddrinfo", resolves_to("10.1.2.3"))
        assert "not a public internet address" in netguard.check_host("intranet.example.com")

    def test_one_private_record_among_public_ones_is_enough_to_refuse(self, monkeypatch):
        monkeypatch.setattr(netguard.socket, "getaddrinfo", resolves_to("93.184.216.34", "127.0.0.1"))
        assert netguard.check_host("mixed.example.com") is not None

    def test_integer_form_of_loopback_is_refused(self):
        # http://2130706433/ is 127.0.0.1 written as one number.
        assert netguard.check_host("2130706433") is not None

    def test_unresolvable_name_is_refused(self, monkeypatch):
        def boom(*args, **kwargs):
            raise socket.gaierror("no such host")
        monkeypatch.setattr(netguard.socket, "getaddrinfo", boom)
        assert "No website found" in netguard.check_host("nope.invalid")

    def test_dns_timeout_is_retried_not_reported_as_missing_site(self, monkeypatch):
        # cjdevent.tn: a live site whose nameserver answered slowly, so the
        # first lookup timed out and the visitor was told to check the spelling.
        calls = []

        def flaky(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
            return resolves_to("51.254.132.203")(*args, **kwargs)
        monkeypatch.setattr(netguard.socket, "getaddrinfo", flaky)
        monkeypatch.setattr(netguard.time, "sleep", lambda s: None)
        assert netguard.check_host("cjdevent.tn") is None
        assert len(calls) == 2

    def test_dns_timeout_on_every_attempt_is_refused_as_temporary(self, monkeypatch):
        def down(*args, **kwargs):
            raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
        monkeypatch.setattr(netguard.socket, "getaddrinfo", down)
        monkeypatch.setattr(netguard.time, "sleep", lambda s: None)
        reason = netguard.check_host("cjdevent.tn")
        assert "Try again" in reason and "No website found" not in reason

    def test_empty_hostname_is_refused(self):
        assert netguard.check_host("") == "URL has no hostname."

    def test_bracketed_ipv6_literal(self):
        assert netguard.check_host("[::1]") is not None


class TestCheckUrl:
    @pytest.fixture(autouse=True)
    def public_dns(self, monkeypatch):
        monkeypatch.setattr(netguard.socket, "getaddrinfo", resolves_to("93.184.216.34"))

    def test_plain_https_url_is_allowed(self):
        assert netguard.check_url("https://example.com/path?q=1") is None

    @pytest.mark.parametrize("url", ["ftp://example.com", "file:///etc/passwd", "gopher://example.com", "example.com"])
    def test_other_schemes_are_refused(self, url):
        assert "Only http:// and https://" in netguard.check_url(url)

    @pytest.mark.parametrize("port", [22, 25, 3306, 5432, 6379])
    def test_non_web_ports_are_refused(self, port):
        assert netguard.check_url(f"http://example.com:{port}/") == f"Port {port} is not allowed."

    @pytest.mark.parametrize("port", [80, 443, 8080, 8443])
    def test_web_ports_are_allowed(self, port):
        assert netguard.check_url(f"http://example.com:{port}/") is None

    def test_malformed_port_is_refused(self):
        assert netguard.check_url("http://example.com:notaport/") == "URL is not valid."

    def test_credentials_in_url_do_not_hide_the_real_host(self, monkeypatch):
        seen = []

        def fake(host, *args, **kwargs):
            seen.append(host)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

        monkeypatch.setattr(netguard.socket, "getaddrinfo", fake)
        assert netguard.check_url("http://example.com@localhost/") is not None
        assert seen == ["localhost"]
