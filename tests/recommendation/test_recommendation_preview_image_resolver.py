"""Tests for Module 8.1: restricted image resolver, image/prompt mapping, output robustness,
configuration validation and explicit env-file loading.

All HTTP and DNS traffic goes through fake sessions and fake DNS resolvers; nothing here
touches the Internet. Tests that create images need Pillow (the `preview` extra).
"""

import io
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import travel_agent.recommendation.student5.preview_adapter as preview_adapter
from travel_agent.recommendation.student5.models import (
    ConfirmedOutfit,
    PreviewMode,
    PreviewStatus,
    ProductCategory,
)
from travel_agent.recommendation.student5.preview_adapter import (
    LocalAndHttpsImageResolver,
    OpenAIImagePreviewProvider,
    PreviewConfigurationError,
    PreviewInputError,
    PreviewOutputError,
    RemoteImagePolicy,
    ResolvedImage,
    build_preview_prompt,
    build_preview_request,
    build_prompt_with_image_mapping,
    generate_preview_with_fallback,
    parse_allowed_hosts,
)

PUBLIC_IP = "93.184.215.14"
HOST = "img.example.com"
SECRET_KEY = "test-key-not-a-real-credential"


@pytest.fixture
def pil():
    return pytest.importorskip("PIL.Image")


def png_bytes(pil, size, fmt="PNG"):
    buffer = io.BytesIO()
    pil.new("RGB", size, (40, 90, 160)).save(buffer, format=fmt)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, status=200, headers=None, chunks=()):
        self.status_code = status
        self.headers = headers if headers is not None else {}
        self.chunks = list(chunks)
        self.iterated = False
        self.closed = False

    def iter_content(self, chunk_size):
        self.iterated = True
        yield from self.chunks

    def close(self):
        self.closed = True


def image_response(pil, size, content_type="image/png"):
    return FakeResponse(200, {"Content-Type": content_type}, [png_bytes(pil, size)])


def redirect(location, status=302):
    return FakeResponse(status, {"Location": location} if location is not None else {})


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.routes[url]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


class FakeDNS:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, host, port):
        self.calls.append((host, port))
        outcome = self.table[host]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_resolver(routes, hosts=(HOST,), dns_table=None, **policy):
    session = FakeSession(routes)
    dns = FakeDNS(dns_table if dns_table is not None else {h: [PUBLIC_IP] for h in hosts})
    resolver = LocalAndHttpsImageResolver(
        RemoteImagePolicy(allowed_hosts=hosts, **policy),
        session_factory=lambda: session,
        dns_resolver=dns,
    )
    return resolver, session, dns


def resolve(resolver, tmp_path, url, **kwargs):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return resolver.resolve(url, work, "01_top", **kwargs), work


def rejected(resolver, tmp_path, url, reason, **kwargs):
    with pytest.raises(PreviewInputError) as exc:
        resolve(resolver, tmp_path, url, **kwargs)
    assert exc.value.reason == reason
    return exc.value


# ---------------------------------------------------------------------------
# Policy and host configuration
# ---------------------------------------------------------------------------


class TestRemoteImagePolicy:
    def test_defaults(self):
        policy = RemoteImagePolicy()
        assert policy.allowed_hosts == frozenset()
        assert (policy.max_redirects, policy.max_bytes) == (3, 8 * 1024 * 1024)
        assert (policy.max_pixels, policy.max_dimension) == (20_000_000, 1024)

    def test_hosts_normalised(self):
        assert RemoteImagePolicy(allowed_hosts=["I.EbayImg.com"]).allowed_hosts == {"i.ebayimg.com"}

    @pytest.mark.parametrize(
        "host",
        [
            "*.ebayimg.com",
            "https://i.ebayimg.com",
            "i.ebayimg.com/x",
            "i.ebayimg.com:443",
            "",
            "user@host.com",
            ".example.com",
        ],
    )
    def test_invalid_hosts_rejected(self, host):
        with pytest.raises(PreviewConfigurationError, match="allowed host"):
            RemoteImagePolicy(allowed_hosts=[host])

    def test_single_string_hosts_rejected(self):
        with pytest.raises(PreviewConfigurationError):
            RemoteImagePolicy(allowed_hosts="i.ebayimg.com")

    @pytest.mark.parametrize("field", ["max_redirects", "max_bytes", "max_pixels", "max_dimension"])
    @pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
    def test_invalid_integer_limits_rejected(self, field, value):
        with pytest.raises(PreviewConfigurationError, match=field):
            RemoteImagePolicy(**{field: value})

    @pytest.mark.parametrize("field", ["connect_timeout", "read_timeout", "total_timeout"])
    @pytest.mark.parametrize("value", [0, -1.0, math.nan, math.inf, True, "5"])
    def test_invalid_timeouts_rejected(self, field, value):
        with pytest.raises(PreviewConfigurationError, match=field):
            RemoteImagePolicy(**{field: value})

    def test_parse_allowed_hosts(self):
        assert parse_allowed_hosts(" i.ebayimg.com, Images.Unsplash.com ,") == {
            "i.ebayimg.com",
            "images.unsplash.com",
        }
        assert parse_allowed_hosts("") == frozenset()
        assert parse_allowed_hosts(None) == frozenset()
        with pytest.raises(PreviewConfigurationError):
            parse_allowed_hosts("*.ebayimg.com")


# ---------------------------------------------------------------------------
# URL rules
# ---------------------------------------------------------------------------


class TestUrlRules:
    @pytest.mark.parametrize(
        "url",
        [
            "http://img.example.com/a.png",
            "ftp://img.example.com/a.png",
            "https://other.example.com/a.png",
            "https://user:pw@img.example.com/a.png",
            "https://user@img.example.com/a.png",
            "https://img.example.com:8443/a.png",
            "https://img.example.com:80/a.png",
            "https://img.example.com:bad/a.png",
            "https:///a.png",
        ],
    )
    def test_rejected_without_request(self, pil, tmp_path, url):
        resolver, session, _ = make_resolver({})
        rejected(resolver, tmp_path, url, "not_allowed")
        assert session.calls == []

    def test_port_443_allowed(self, pil, tmp_path):
        url = "https://img.example.com:443/a.png"
        resolver, session, _ = make_resolver({url: image_response(pil, (20, 20))})
        source, _ = resolve(resolver, tmp_path, url)
        assert source.source == "remote"

    def test_host_matching_is_exact(self, pil, tmp_path):
        resolver, session, _ = make_resolver({})
        for url in ("https://evil.img.example.com/a.png", "https://img.example.com.evil.io/a.png"):
            rejected(resolver, tmp_path, url, "not_allowed")
        assert session.calls == []

    def test_empty_allow_list_rejects_everything(self, pil, tmp_path):
        resolver, session, _ = make_resolver({}, hosts=())
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "not_allowed")

    def test_remote_disallowed_when_not_permitted(self, pil, tmp_path):
        url = "https://img.example.com/me.png"
        resolver, session, _ = make_resolver({url: image_response(pil, (20, 20))})
        rejected(resolver, tmp_path, url, "not_allowed", allow_remote=False)
        assert session.calls == []

    @pytest.mark.parametrize(
        "host,url",
        [
            ("localhost", "https://localhost/a.png"),
            ("127.0.0.1", "https://127.0.0.1/a.png"),
            ("10.0.0.1", "https://10.0.0.1/a.png"),
        ],
    )
    def test_internal_hosts_rejected_even_if_allowed(self, pil, tmp_path, host, url):
        resolver, session, _ = make_resolver(
            {}, hosts=(host,), dns_table={"localhost": ["127.0.0.1"]}
        )
        rejected(resolver, tmp_path, url, "not_allowed")
        assert session.calls == []

    def test_ipv6_loopback_literal_rejected(self, pil, tmp_path):
        resolver, session, _ = make_resolver({})
        rejected(resolver, tmp_path, "https://[::1]/a.png", "not_allowed")

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.0.0.5",
            "172.16.1.1",
            "192.168.1.1",
            "169.254.169.254",
            "224.0.0.1",
            "240.0.0.1",
            "0.0.0.0",
            "100.64.0.1",
            "::1",
            "fe80::1",
            "fc00::1",
            "ff02::1",
            "::",
            "::ffff:127.0.0.1",
            "not-an-ip",
        ],
    )
    def test_dns_to_non_public_ip_rejected(self, pil, tmp_path, ip):
        resolver, session, _ = make_resolver({}, dns_table={HOST: [ip]})
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "not_allowed")
        assert session.calls == []

    def test_any_non_public_answer_rejects(self, pil, tmp_path):
        resolver, session, _ = make_resolver({}, dns_table={HOST: [PUBLIC_IP, "10.0.0.7"]})
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "not_allowed")

    def test_empty_dns_answer_rejected(self, pil, tmp_path):
        resolver, session, _ = make_resolver({}, dns_table={HOST: []})
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "not_allowed")

    def test_dns_failure_is_download_error(self, pil, tmp_path):
        resolver, session, _ = make_resolver({}, dns_table={HOST: OSError("no such host")})
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "download")


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------

CDN = "cdn.example.com"


class TestRedirects:
    def test_redirect_to_allowed_host_rechecked(self, pil, tmp_path):
        routes = {
            "https://img.example.com/a.png": redirect("https://cdn.example.com/b.png"),
            "https://cdn.example.com/b.png": image_response(pil, (20, 20)),
        }
        resolver, session, dns = make_resolver(routes, hosts=(HOST, CDN))
        source, _ = resolve(resolver, tmp_path, "https://img.example.com/a.png")
        assert source.source == "remote"
        assert [h for h, _ in dns.calls] == [HOST, CDN]
        assert [u for u, _ in session.calls] == list(routes)
        assert all(kw["allow_redirects"] is False for _, kw in session.calls)

    def test_relative_redirect(self, pil, tmp_path):
        routes = {
            "https://img.example.com/a.png": redirect("/b.png", status=301),
            "https://img.example.com/b.png": image_response(pil, (20, 20)),
        }
        resolver, session, _ = make_resolver(routes)
        resolve(resolver, tmp_path, "https://img.example.com/a.png")
        assert session.calls[-1][0] == "https://img.example.com/b.png"

    @pytest.mark.parametrize(
        "target",
        [
            "https://evil.example.org/b.png",
            "http://img.example.com/b.png",
            "https://img.example.com:8080/b.png",
        ],
    )
    def test_redirect_to_disallowed_target_rejected(self, pil, tmp_path, target):
        routes = {"https://img.example.com/a.png": redirect(target, status=307)}
        resolver, session, _ = make_resolver(routes)
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "not_allowed")
        assert [u for u, _ in session.calls] == ["https://img.example.com/a.png"]
        assert routes["https://img.example.com/a.png"].closed

    def test_redirect_to_internal_address_rejected(self, pil, tmp_path):
        routes = {"https://img.example.com/a.png": redirect("https://cdn.example.com/b.png")}
        resolver, session, _ = make_resolver(
            routes, hosts=(HOST, CDN), dns_table={HOST: [PUBLIC_IP], CDN: ["192.168.0.10"]}
        )
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "not_allowed")
        assert len(session.calls) == 1

    def chain(self, pil, hops):
        urls = [f"https://img.example.com/{n}.png" for n in range(hops + 1)]
        routes = {urls[n]: redirect(urls[n + 1], status=308) for n in range(hops)}
        routes[urls[-1]] = image_response(pil, (20, 20))
        return urls[0], routes

    def test_three_redirects_allowed(self, pil, tmp_path):
        start, routes = self.chain(pil, 3)
        resolver, session, _ = make_resolver(routes)
        resolve(resolver, tmp_path, start)
        assert len(session.calls) == 4

    def test_too_many_redirects_rejected(self, pil, tmp_path):
        start, routes = self.chain(pil, 4)
        resolver, session, _ = make_resolver(routes)
        rejected(resolver, tmp_path, start, "download")
        assert len(session.calls) == 4

    def test_redirect_without_location_rejected(self, pil, tmp_path):
        routes = {"https://img.example.com/a.png": redirect(None)}
        resolver, _, _ = make_resolver(routes)
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "download")


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------

URL = "https://img.example.com/a.png?token=query-secret-123"


class TestResponses:
    def test_request_is_streamed_with_timeouts_and_safe_headers(self, pil, tmp_path):
        resolver, session, _ = make_resolver(
            {URL: image_response(pil, (20, 20))}, connect_timeout=3.0, read_timeout=7.0
        )
        resolve(resolver, tmp_path, URL)
        [(url, kwargs)] = session.calls
        assert kwargs["stream"] is True
        assert kwargs["timeout"] == (3.0, 7.0)
        assert set(kwargs) == {"stream", "allow_redirects", "timeout", "headers"}
        assert {k.lower() for k in kwargs["headers"]} == {"user-agent", "accept"}
        assert SECRET_KEY not in repr(session.calls)
        assert session.closed

    def test_default_session_carries_no_credentials(self):
        pytest.importorskip("requests")
        session = LocalAndHttpsImageResolver()._new_session()
        try:
            assert session.trust_env is False
            assert dict(session.headers) == {}
            assert len(session.cookies) == 0
            assert session.auth is None
        finally:
            session.close()

    @pytest.mark.parametrize("content_type", ["text/html", "application/octet-stream", None])
    def test_non_image_content_type_rejected(self, pil, tmp_path, content_type):
        headers = {} if content_type is None else {"Content-Type": content_type}
        response = FakeResponse(200, headers, [png_bytes(pil, (20, 20))])
        resolver, _, _ = make_resolver({URL: response})
        rejected(resolver, tmp_path, URL, "invalid")
        assert not response.iterated and response.closed

    def test_content_type_with_parameters_accepted(self, pil, tmp_path):
        resolver, _, _ = make_resolver({URL: image_response(pil, (20, 20), "Image/PNG; q=1")})
        assert resolve(resolver, tmp_path, URL)[0].source == "remote"

    def test_http_error_status_rejected(self, pil, tmp_path):
        resolver, _, _ = make_resolver({URL: FakeResponse(404, {"Content-Type": "image/png"})})
        rejected(resolver, tmp_path, URL, "download")

    def test_declared_size_over_limit_rejected_before_reading(self, pil, tmp_path):
        response = FakeResponse(
            200, {"Content-Type": "image/png", "Content-Length": "999999"}, [b"x"]
        )
        resolver, _, _ = make_resolver({URL: response}, max_bytes=1000)
        rejected(resolver, tmp_path, URL, "download")
        assert not response.iterated

    def test_streamed_size_over_limit_aborted(self, pil, tmp_path):
        response = FakeResponse(
            200,
            {"Content-Type": "image/png", "Content-Length": "bogus"},
            [b"x" * 600, b"x" * 600, b"x" * 600],
        )
        resolver, _, _ = make_resolver({URL: response}, max_bytes=1000)
        rejected(resolver, tmp_path, URL, "download")
        assert response.closed

    def test_network_error_is_download_error(self, pil, tmp_path):
        resolver, session, _ = make_resolver({URL: ConnectionError("reset")})
        rejected(resolver, tmp_path, URL, "download")
        assert session.closed

    def test_missing_requests_package(self, pil, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "requests", None)
        resolver = LocalAndHttpsImageResolver(
            RemoteImagePolicy(allowed_hosts=[HOST]), dns_resolver=FakeDNS({HOST: [PUBLIC_IP]})
        )
        rejected(resolver, tmp_path, URL, "download")

    def test_undecodable_image_rejected(self, pil, tmp_path):
        response = FakeResponse(200, {"Content-Type": "image/png"}, [b"not really a png"])
        resolver, _, _ = make_resolver({URL: response})
        rejected(resolver, tmp_path, URL, "invalid")

    def test_pixel_limit(self, pil, tmp_path):
        resolver, _, _ = make_resolver({URL: image_response(pil, (30, 30))}, max_pixels=800)
        rejected(resolver, tmp_path, URL, "invalid")

    def test_downscaled_and_reencoded_as_png(self, pil, tmp_path):
        resolver, _, _ = make_resolver({URL: image_response(pil, (2000, 100))}, max_dimension=1024)
        source, work = resolve(resolver, tmp_path, URL)
        with pil.open(source.path) as img:
            assert img.format == "PNG"
            assert max(img.size) == 1024

    def test_exif_orientation_applied(self, pil, tmp_path):
        exif = pil.Exif()
        exif[0x0112] = 6  # rotate 90° on display
        buffer = io.BytesIO()
        pil.new("RGB", (40, 20)).save(buffer, format="JPEG", exif=exif)
        response = FakeResponse(200, {"Content-Type": "image/jpeg"}, [buffer.getvalue()])
        resolver, _, _ = make_resolver({URL: response})
        source, _ = resolve(resolver, tmp_path, URL)
        with pil.open(source.path) as img:
            assert img.size == (20, 40)
            assert dict(img.getexif()) == {}

    def test_errors_do_not_reveal_url_or_query(self, pil, tmp_path):
        resolver, _, _ = make_resolver({URL: FakeResponse(500)})
        error = rejected(resolver, tmp_path, URL, "download")
        assert "query-secret-123" not in str(error) and HOST not in str(error)


# ---------------------------------------------------------------------------
# Total deadline (whole operation: DNS, every request, every chunk)
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class AdvancingDNS(FakeDNS):
    def __init__(self, table, clock, seconds):
        super().__init__(table)
        self.clock, self.seconds = clock, seconds

    def __call__(self, host, port):
        self.clock.advance(self.seconds)
        return super().__call__(host, port)


class AdvancingSession(FakeSession):
    """Advances the clock by `delays[n]` during the n-th request."""

    def __init__(self, routes, clock, delays):
        super().__init__(routes)
        self.clock, self.delays = clock, list(delays)

    def get(self, url, **kwargs):
        response = super().get(url, **kwargs)
        self.clock.advance(self.delays.pop(0) if self.delays else 0.0)
        return response


class SlowResponse(FakeResponse):
    def __init__(self, clock, chunk_delay, chunks):
        super().__init__(200, {"Content-Type": "image/png"}, chunks)
        self.clock, self.chunk_delay = clock, chunk_delay

    def iter_content(self, chunk_size):
        self.iterated = True
        for chunk in self.chunks:
            self.clock.advance(self.chunk_delay)
            yield chunk


DEADLINE_URL = "https://img.example.com/a.png?token=query-secret-123"
DEADLINE_SECRETS = ("query-secret-123", HOST, "img.example.com", "/a.png")


def deadline_resolver(clock, session, dns, **policy):
    policy = {
        "allowed_hosts": [HOST, CDN],
        "connect_timeout": 5.0,
        "read_timeout": 10.0,
        "total_timeout": 30.0,
        **policy,
    }
    return LocalAndHttpsImageResolver(
        RemoteImagePolicy(**policy), session_factory=lambda: session, dns_resolver=dns, clock=clock
    )


def assert_deadline_error(error):
    assert error.reason == "download"
    for secret in DEADLINE_SECRETS:
        assert secret not in str(error)


class TestTotalDeadline:
    def test_expired_before_dns(self, pil, tmp_path):
        clock = FakeClock()
        session = FakeSession({})
        dns = FakeDNS({HOST: [PUBLIC_IP]})

        def slow_factory():
            clock.advance(30.0)  # deadline reached exactly: remaining <= 0
            return session

        resolver = LocalAndHttpsImageResolver(
            RemoteImagePolicy(allowed_hosts=[HOST], total_timeout=30.0),
            session_factory=slow_factory,
            dns_resolver=dns,
            clock=clock,
        )
        error = rejected(resolver, tmp_path, DEADLINE_URL, "download")
        assert_deadline_error(error)
        assert dns.calls == [] and session.calls == []

    def test_expired_before_first_request(self, pil, tmp_path):
        clock = FakeClock()
        session = FakeSession({DEADLINE_URL: image_response(pil, (20, 20))})
        dns = AdvancingDNS({HOST: [PUBLIC_IP]}, clock, 31.0)
        error = rejected(deadline_resolver(clock, session, dns), tmp_path, DEADLINE_URL, "download")
        assert_deadline_error(error)
        assert len(dns.calls) == 1 and session.calls == []

    def test_request_timeouts_capped_by_remaining_time(self, pil, tmp_path):
        clock = FakeClock()
        session = FakeSession({DEADLINE_URL: image_response(pil, (20, 20))})
        dns = AdvancingDNS({HOST: [PUBLIC_IP]}, clock, 27.0)  # 3 s left
        resolve(deadline_resolver(clock, session, dns), tmp_path, DEADLINE_URL)
        assert session.calls[0][1]["timeout"] == (3.0, 3.0)

    def test_full_timeouts_when_time_remains(self, pil, tmp_path):
        clock = FakeClock()
        session = FakeSession({DEADLINE_URL: image_response(pil, (20, 20))})
        resolve(
            deadline_resolver(clock, session, FakeDNS({HOST: [PUBLIC_IP]})), tmp_path, DEADLINE_URL
        )
        assert session.calls[0][1]["timeout"] == (5.0, 10.0)

    def test_timeouts_recomputed_after_redirect(self, pil, tmp_path):
        clock = FakeClock()
        target = "https://cdn.example.com/b.png"
        routes = {DEADLINE_URL: redirect(target), target: image_response(pil, (20, 20))}
        session = AdvancingSession(routes, clock, [22.0])  # first request takes 22 s
        resolve(
            deadline_resolver(clock, session, FakeDNS({HOST: [PUBLIC_IP], CDN: [PUBLIC_IP]})),
            tmp_path,
            DEADLINE_URL,
        )
        assert [kw["timeout"] for _, kw in session.calls] == [(5.0, 10.0), (5.0, 8.0)]

    def test_expired_after_redirect(self, pil, tmp_path):
        clock = FakeClock()
        target = "https://cdn.example.com/b.png"
        routes = {DEADLINE_URL: redirect(target), target: image_response(pil, (20, 20))}
        session = AdvancingSession(routes, clock, [30.0])
        dns = FakeDNS({HOST: [PUBLIC_IP], CDN: [PUBLIC_IP]})
        error = rejected(deadline_resolver(clock, session, dns), tmp_path, DEADLINE_URL, "download")
        assert_deadline_error(error)
        assert [u for u, _ in session.calls] == [DEADLINE_URL]
        assert [h for h, _ in dns.calls] == [HOST]  # no DNS lookup for the redirect target
        assert routes[DEADLINE_URL].closed

    def test_expired_while_reading(self, pil, tmp_path):
        clock = FakeClock()
        response = SlowResponse(clock, 12.0, [b"a", b"b", b"c"])
        session = FakeSession({DEADLINE_URL: response})
        error = rejected(
            deadline_resolver(clock, session, FakeDNS({HOST: [PUBLIC_IP]})),
            tmp_path,
            DEADLINE_URL,
            "download",
        )
        assert_deadline_error(error)
        assert response.closed

    def test_deadline_covers_ip_literal_hosts(self, pil, tmp_path):
        clock = FakeClock()
        url = "https://93.184.215.14/a.png"
        session = FakeSession({url: image_response(pil, (20, 20))})

        def slow_factory():
            clock.advance(31.0)
            return session

        resolver = LocalAndHttpsImageResolver(
            RemoteImagePolicy(allowed_hosts=[PUBLIC_IP]),
            session_factory=slow_factory,
            dns_resolver=FakeDNS({}),
            clock=clock,
        )
        rejected(resolver, tmp_path, url, "download")
        assert session.calls == []


# ---------------------------------------------------------------------------
# Reference classification (URI vs local path)
# ---------------------------------------------------------------------------


class TestReferenceClassification:
    @pytest.mark.parametrize(
        "reference",
        [
            r"C:\Users\alice\images\top.png",
            "C:/Users/alice/images/top.png",
            r"D:top.png",
            r"\\fileserver\share\top.png",
            "images/top.png",
            "./top.png",
            "/var/data/top.png",
            "top.png",
        ],
    )
    def test_local_paths_not_remote(self, reference):
        assert preview_adapter._is_remote(reference) is False

    @pytest.mark.parametrize(
        "reference",
        [
            "https://img.example.com/a.png",
            "http://img.example.com/a.png",
            "HTTPS://IMG.EXAMPLE.COM/a.png",
            "file:///C:/Users/alice/top.png",
            "ftp://h/a.png",
            "data:image/png;base64,iVBORw0KGgo=",
            "s3://bucket/key.png",
            "https://[::1",
        ],
    )
    def test_uris_are_remote(self, reference):
        assert preview_adapter._is_remote(reference) is True

    def test_windows_drive_path_resolved_locally(self, pil, tmp_path):
        local = tmp_path / "top.png"
        pil.new("RGB", (15, 15)).save(local)
        assert preview_adapter._is_remote(str(local)) is False  # e.g. C:\Users\...\top.png
        resolver, session, dns = make_resolver({})
        source, _ = resolve(resolver, tmp_path, str(local))
        assert source.source == "local"
        assert session.calls == [] and dns.calls == []

    @pytest.mark.parametrize(
        "reference",
        [
            "http://img.example.com/a.png",
            "file:///C:/Users/alice/top.png",
            "data:image/png;base64,iVBORw0KGgo=",
            "ftp://img.example.com/a.png",
            "s3://bucket/key.png",
            "https://[::1",
        ],
    )
    def test_non_https_uris_not_allowed_without_any_request(self, pil, tmp_path, reference):
        def no_session():
            raise AssertionError("no session may be created for a non-https URI")

        dns = FakeDNS({})
        resolver = LocalAndHttpsImageResolver(
            RemoteImagePolicy(allowed_hosts=[HOST]), session_factory=no_session, dns_resolver=dns
        )
        error = rejected(resolver, tmp_path, reference, "not_allowed")
        assert dns.calls == []
        assert "alice" not in str(error) and "bucket" not in str(error)

    def test_https_uppercase_scheme_accepted(self, pil, tmp_path):
        url = "HTTPS://img.example.com/a.png"
        resolver, session, _ = make_resolver({url: image_response(pil, (20, 20))})
        assert resolve(resolver, tmp_path, url)[0].source == "remote"

    def test_non_https_uri_not_allowed_even_without_requests(self, pil, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "requests", None)
        resolver = LocalAndHttpsImageResolver(RemoteImagePolicy(allowed_hosts=[HOST]))
        rejected(resolver, tmp_path, "http://img.example.com/a.png", "not_allowed")


# ---------------------------------------------------------------------------
# Local files and temporary-file clean-up
# ---------------------------------------------------------------------------


class TestLocalAndCleanup:
    def test_local_file_needs_no_http(self, pil, tmp_path):
        local = tmp_path / "product.png"
        pil.new("RGB", (15, 15)).save(local)
        resolver, session, dns = make_resolver({})
        source, _ = resolve(resolver, tmp_path, str(local))
        assert source.source == "local"
        assert session.calls == [] and dns.calls == []
        assert local.read_bytes()  # caller's file untouched

    @pytest.mark.parametrize("kind", ["missing", "too_large", "not_image"])
    def test_bad_local_file(self, pil, tmp_path, kind):
        local = tmp_path / "product.png"
        if kind == "too_large":
            pil.new("RGB", (50, 50)).save(local)
        elif kind == "not_image":
            local.write_bytes(b"plain text")
        resolver = LocalAndHttpsImageResolver(max_input_bytes=10)
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(PreviewInputError) as exc:
            resolver.resolve(str(local), work, "01_top")
        assert exc.value.reason == "invalid"
        assert str(local) not in str(exc.value)

    def test_download_file_removed_after_success(self, pil, tmp_path):
        resolver, _, _ = make_resolver({URL: image_response(pil, (20, 20))})
        source, work = resolve(resolver, tmp_path, URL)
        assert sorted(p.name for p in work.iterdir()) == [source.path.name]

    @pytest.mark.parametrize("response_kind", ["network", "redirect_denied", "bad_image"])
    def test_download_files_removed_after_failure(self, pil, tmp_path, response_kind):
        outcome = {
            "network": OSError("x"),
            "redirect_denied": redirect("https://evil.example.org/x"),
            "bad_image": FakeResponse(200, {"Content-Type": "image/png"}, [b"junk"]),
        }
        resolver, _, _ = make_resolver({URL: outcome[response_kind]})
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(PreviewInputError):
            resolver.resolve(URL, work, "01_top")
        assert list(work.iterdir()) == []

    def test_unexpected_error_still_cleans_up(self, pil, tmp_path):
        class Boom(FakeResponse):
            def iter_content(self, chunk_size):
                raise KeyError("bug")

        resolver, _, _ = make_resolver({URL: Boom(200, {"Content-Type": "image/png"})})
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(KeyError):
            resolver.resolve(URL, work, "01_top")
        assert list(work.iterdir()) == []


# ---------------------------------------------------------------------------
# Provider integration: order, mapping, clean-up
# ---------------------------------------------------------------------------


class FakeImages:
    def __init__(self, pil, error=None):
        self.pil = pil
        self.error = error
        self.calls = []

    def _ok(self):
        if self.error is not None:
            raise self.error
        import base64

        return SimpleNamespace(
            data=[
                SimpleNamespace(b64_json=base64.b64encode(png_bytes(self.pil, (16, 16))).decode())
            ]
        )

    def edit(self, **kwargs):
        images = kwargs["image"] if isinstance(kwargs["image"], list) else [kwargs["image"]]
        sizes = []
        for f in images:
            with self.pil.open(f) as img:
                sizes.append(img.size)
        self.calls.append(("edit", kwargs, sizes))
        return self._ok()

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs, []))
        return self._ok()


def confirmed_with(image_urls, outfit_id="outfit_1"):
    def p(category, owned=False):
        return {
            "product_id": f"p_{category.lower()}",
            "source": "mock",
            "title": f"Mock {category.lower()}",
            "description": f"{category.lower()} text",
            "product_url": None if owned else "https://shop.example.com/x",
            "image_url": image_urls.get(category),
            "price": "10.00",
            "currency": "SGD",
            "availability": True,
            "category": category,
            "product_score": 0.5,
            "is_user_owned": owned,
        }

    return ConfirmedOutfit.model_validate(
        {
            "session_id": "s",
            "selected_outfit_id": outfit_id,
            "recommendation": {
                "outfit_id": outfit_id,
                "display_position": 1,
                "existing_items": [p("BOTTOM", owned=True)],
                "purchase_items": [p("TOP"), p("FOOTWEAR"), p("OUTERWEAR")],
                "total_purchase_cost": "30.00",
                "currency": "SGD",
                "compatibility_score": 0.5,
                "outfit_product_relevance": 0.5,
                "final_outfit_score": 0.5,
            },
        }
    )


def provider_with(pil, tmp_path, routes, images=None, hosts=(HOST,)):
    session = FakeSession(routes)
    dns = FakeDNS({h: [PUBLIC_IP] for h in hosts})
    resolver = LocalAndHttpsImageResolver(
        RemoteImagePolicy(allowed_hosts=hosts), session_factory=lambda: session, dns_resolver=dns
    )
    fake = images or FakeImages(pil)
    (tmp_path / "work").mkdir(exist_ok=True)
    provider = OpenAIImagePreviewProvider(
        model="configured-image-model",
        client=SimpleNamespace(images=fake),
        output_dir=tmp_path / "previews",
        temp_dir_parent=tmp_path / "work",
        api_error_types=(RuntimeError,),
        image_resolver=resolver,
    )
    return provider, fake, session, dns


def remote(name):
    return f"https://{HOST}/{name}.png?sig=query-secret-123"


def reference_lines(prompt):
    return [line for line in prompt.splitlines() if line.startswith("Reference image ")]


class TestProviderWithResolver:
    def test_remote_images_sent_in_item_order(self, pil, tmp_path):
        urls = {c: remote(c.lower()) for c in ("TOP", "BOTTOM", "FOOTWEAR", "OUTERWEAR")}
        routes = {
            urls["TOP"]: image_response(pil, (21, 21)),
            urls["BOTTOM"]: image_response(pil, (22, 22)),
            urls["FOOTWEAR"]: image_response(pil, (23, 23)),
            urls["OUTERWEAR"]: image_response(pil, (24, 24)),
        }
        provider, fake, _, _ = provider_with(pil, tmp_path, routes)
        req = build_preview_request(confirmed_with(urls), PreviewMode.FLAT_LAY)
        result = provider.generate_preview(req)
        assert fake.calls[0][0] == "edit"
        assert fake.calls[0][2] == [(21, 21), (22, 22), (23, 23), (24, 24)]
        assert result.status is PreviewStatus.SUCCESS
        assert not any("not used" in w for w in result.warnings)

    def test_mixed_local_and_remote_order(self, pil, tmp_path):
        top = tmp_path / "in" / "top.png"
        top.parent.mkdir(parents=True)
        pil.new("RGB", (11, 11)).save(top)
        foot = tmp_path / "in" / "foot.png"
        pil.new("RGB", (13, 13)).save(foot)
        urls = {
            "TOP": str(top),
            "BOTTOM": remote("bottom"),
            "FOOTWEAR": str(foot),
            "OUTERWEAR": remote("outer"),
        }
        routes = {
            urls["BOTTOM"]: image_response(pil, (22, 22)),
            urls["OUTERWEAR"]: image_response(pil, (24, 24)),
        }
        provider, fake, session, _ = provider_with(pil, tmp_path, routes)
        provider.generate_preview(
            build_preview_request(confirmed_with(urls), PreviewMode.MANNEQUIN)
        )
        assert fake.calls[0][2] == [(11, 11), (22, 22), (13, 13), (24, 24)]
        assert len(session.calls) == 2  # local files are never fetched

    def test_local_paths_need_no_http(self, pil, tmp_path):
        urls = {}
        for n, c in enumerate(("TOP", "BOTTOM", "FOOTWEAR", "OUTERWEAR")):
            path = tmp_path / "in" / f"{c}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pil.new("RGB", (10 + n, 10 + n)).save(path)
            urls[c] = str(path)
        provider, fake, session, dns = provider_with(pil, tmp_path, {})
        provider.generate_preview(build_preview_request(confirmed_with(urls), PreviewMode.FLAT_LAY))
        assert session.calls == [] and dns.calls == []
        assert fake.calls[0][0] == "edit"

    def test_portrait_first_with_remote_items(self, pil, tmp_path):
        portrait = tmp_path / "user" / "me.png"
        portrait.parent.mkdir(parents=True)
        pil.new("RGB", (30, 40)).save(portrait)
        urls = {"TOP": remote("top"), "FOOTWEAR": remote("foot")}
        routes = {
            urls["TOP"]: image_response(pil, (21, 21)),
            urls["FOOTWEAR"]: image_response(pil, (23, 23)),
        }
        provider, fake, _, _ = provider_with(pil, tmp_path, routes)
        req = build_preview_request(
            confirmed_with(urls),
            PreviewMode.PORTRAIT,
            user_portrait=str(portrait),
            portrait_consent=True,
        )
        provider.generate_preview(req)
        assert fake.calls[0][2] == [(30, 40), (21, 21), (23, 23)]
        assert reference_lines(fake.calls[0][1]["prompt"]) == [
            "Reference image 1 is the user's portrait.",
            "Reference image 2 corresponds to outfit item 1: TOP.",
            "Reference image 3 corresponds to outfit item 3: FOOTWEAR.",
        ]

    def test_remote_portrait_never_downloaded(self, pil, tmp_path):
        url = remote("portrait")
        provider, fake, session, _ = provider_with(
            pil, tmp_path, {url: image_response(pil, (30, 40))}
        )
        req = build_preview_request(
            confirmed_with({}), PreviewMode.PORTRAIT, user_portrait=url, portrait_consent=True
        )
        with pytest.raises(PreviewInputError):
            provider.generate_preview(req)
        assert session.calls == [] and fake.calls == []

    def test_skipped_image_keeps_mapping_correct(self, pil, tmp_path):
        urls = {c: remote(c.lower()) for c in ("TOP", "BOTTOM", "FOOTWEAR", "OUTERWEAR")}
        routes = {
            urls["TOP"]: image_response(pil, (21, 21)),
            urls["BOTTOM"]: FakeResponse(200, {"Content-Type": "text/html"}, [b"<html>"]),
            urls["FOOTWEAR"]: image_response(pil, (23, 23)),
            urls["OUTERWEAR"]: image_response(pil, (24, 24)),
        }
        provider, fake, _, _ = provider_with(pil, tmp_path, routes)
        req = build_preview_request(confirmed_with(urls), PreviewMode.FLAT_LAY)
        result = provider.generate_preview(req)
        prompt = fake.calls[0][1]["prompt"]
        assert fake.calls[0][2] == [(21, 21), (23, 23), (24, 24)]
        assert reference_lines(prompt) == [
            "Reference image 1 corresponds to outfit item 1: TOP.",
            "Reference image 2 corresponds to outfit item 3: FOOTWEAR.",
            "Reference image 3 corresponds to outfit item 4: OUTERWEAR.",
        ]
        assert "2. BOTTOM: bottom text" in prompt  # text description kept
        assert any(
            "1 product image(s) could not be read as valid images" in w for w in result.warnings
        )
        for secret in ("query-secret-123", HOST, "https://", str(tmp_path)):
            assert secret not in prompt
            assert not any(secret in w for w in result.warnings)

    def test_all_downloads_fail_falls_back_to_text_generation(self, pil, tmp_path):
        urls = {"TOP": remote("top")}
        provider, fake, _, _ = provider_with(pil, tmp_path, {urls["TOP"]: OSError("down")})
        result = provider.generate_preview(
            build_preview_request(confirmed_with(urls), PreviewMode.FLAT_LAY)
        )
        assert fake.calls[0][0] == "generate"
        assert any("could not be downloaded" in w for w in result.warnings)
        assert list((tmp_path / "work").iterdir()) == []

    def test_disallowed_host_warning(self, pil, tmp_path):
        urls = {"TOP": "https://other.example.org/top.png?x=query-secret-123"}
        provider, _, session, _ = provider_with(pil, tmp_path, {})
        result = provider.generate_preview(
            build_preview_request(confirmed_with(urls), PreviewMode.FLAT_LAY)
        )
        assert session.calls == []
        assert any("not permitted by the remote image policy" in w for w in result.warnings)
        assert not any("other.example.org" in w for w in result.warnings)

    @pytest.mark.parametrize("failure", ["api", "pillow"])
    def test_cleanup_after_api_or_output_failure(self, pil, tmp_path, failure):
        urls = {"TOP": remote("top")}
        images = FakeImages(pil, error=RuntimeError("api down")) if failure == "api" else None
        provider, fake, _, _ = provider_with(
            pil, tmp_path, {urls["TOP"]: image_response(pil, (21, 21))}, images=images
        )
        if failure == "pillow":
            fake.edit = lambda **kw: SimpleNamespace(data=[SimpleNamespace(b64_json="aGVsbG8=")])
        with pytest.raises(preview_adapter.PreviewProviderError):
            provider.generate_preview(
                build_preview_request(confirmed_with(urls), PreviewMode.FLAT_LAY)
            )
        assert list((tmp_path / "work").iterdir()) == []

    def test_download_requests_carry_no_api_key(self, pil, tmp_path):
        url = remote("top")
        session = FakeSession({url: image_response(pil, (21, 21))})
        resolver = LocalAndHttpsImageResolver(
            RemoteImagePolicy(allowed_hosts=[HOST]),
            session_factory=lambda: session,
            dns_resolver=FakeDNS({HOST: [PUBLIC_IP]}),
        )
        pytest.importorskip("openai")
        provider = OpenAIImagePreviewProvider(
            model="m", api_key=SECRET_KEY, image_resolver=resolver, output_dir=tmp_path / "out"
        )
        provider._client = SimpleNamespace(images=FakeImages(pil))
        provider.generate_preview(
            build_preview_request(confirmed_with({"TOP": url}), PreviewMode.FLAT_LAY)
        )
        assert SECRET_KEY not in repr(session.calls)
        for _, kwargs in session.calls:
            headers = {k.lower() for k in kwargs["headers"]}
            assert not headers & {"authorization", "cookie", "x-api-key"}


class TestPromptMapping:
    def request(self):
        return build_preview_request(confirmed_with({}), PreviewMode.FLAT_LAY)

    def test_mapping_lines(self):
        refs = [
            ResolvedImage(
                path=Path("x.png"),
                source="remote",
                is_portrait=False,
                item_position=2,
                category=ProductCategory.BOTTOM,
            )
        ]
        prompt = build_prompt_with_image_mapping(self.request(), refs)
        assert prompt.startswith(build_preview_prompt(self.request()))
        assert reference_lines(prompt) == [
            "Reference image 1 corresponds to outfit item 2: BOTTOM."
        ]
        assert "x.png" not in prompt

    def test_no_references(self):
        prompt = build_prompt_with_image_mapping(self.request(), [])
        assert "No reference images are provided" in prompt

    def test_product_cards_unchanged(self):
        req = build_preview_request(confirmed_with({}), PreviewMode.PRODUCT_CARDS)
        assert build_prompt_with_image_mapping(req, []) == build_preview_prompt(req)


# ---------------------------------------------------------------------------
# Output robustness
# ---------------------------------------------------------------------------


class TestOutputRobustness:
    def text_request(self):
        return build_preview_request(confirmed_with({}), PreviewMode.FLAT_LAY)

    def test_rename_failure_cleans_part_and_falls_back(self, pil, tmp_path, monkeypatch):
        provider, _, _, _ = provider_with(pil, tmp_path, {})

        def deny(self, target):
            raise PermissionError(f"denied: {target}")

        monkeypatch.setattr(Path, "replace", deny)
        with pytest.raises(PreviewOutputError) as exc:
            provider.generate_preview(self.text_request())
        assert str(tmp_path) not in str(exc.value)
        assert list((tmp_path / "previews").iterdir()) == []
        result = generate_preview_with_fallback(self.text_request(), provider)
        assert result.status is PreviewStatus.FALLBACK
        assert result.delivered_mode is PreviewMode.PRODUCT_CARDS

    def test_write_failure_wrapped(self, pil, tmp_path, monkeypatch):
        provider, _, _, _ = provider_with(pil, tmp_path, {})

        def broken_open(*args, **kwargs):
            raise OSError("disk full at C:/secret/path")

        monkeypatch.setattr(preview_adapter, "open", broken_open, raising=False)
        with pytest.raises(PreviewOutputError) as exc:
            provider.generate_preview(self.text_request())
        assert "secret" not in str(exc.value)
        assert list((tmp_path / "previews").iterdir()) == []

    def test_output_dir_unusable(self, pil, tmp_path):
        provider, _, _, _ = provider_with(pil, tmp_path, {})
        (tmp_path / "previews").write_text("a file, not a directory")
        with pytest.raises(PreviewOutputError) as exc:
            provider.generate_preview(self.text_request())
        assert str(tmp_path) not in str(exc.value)

    def test_programming_errors_not_wrapped(self, pil, tmp_path, monkeypatch):
        provider, _, _, _ = provider_with(pil, tmp_path, {})

        def bug(self, target):
            raise TypeError("bug")

        monkeypatch.setattr(Path, "replace", bug)
        with pytest.raises(TypeError):
            provider.generate_preview(self.text_request())
        assert list((tmp_path / "previews").iterdir()) == []


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


class TestProviderConfiguration:
    @pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True, "10"])
    def test_invalid_timeout(self, pil, value):
        with pytest.raises(PreviewConfigurationError, match="timeout_seconds"):
            OpenAIImagePreviewProvider(
                model="m", client=SimpleNamespace(images=None), timeout_seconds=value
            )

    @pytest.mark.parametrize("value", [0, -5, True, 1.5, "10"])
    def test_invalid_max_input_bytes(self, pil, value):
        with pytest.raises(PreviewConfigurationError, match="max_input_bytes"):
            OpenAIImagePreviewProvider(
                model="m", client=SimpleNamespace(images=None), max_input_bytes=value
            )

    def test_default_resolver_allows_no_remote_hosts(self, pil):
        provider = OpenAIImagePreviewProvider(model="m", client=SimpleNamespace(images=None))
        assert provider.image_resolver.policy.allowed_hosts == frozenset()

    def test_from_env_reads_allowed_hosts(self, pil):
        pytest.importorskip("openai")
        provider = OpenAIImagePreviewProvider.from_env(
            {
                "PREVIEW_PROVIDER": "openai",
                "PREVIEW_API_KEY": SECRET_KEY,
                "PREVIEW_MODEL": "m",
                "PREVIEW_ALLOWED_IMAGE_HOSTS": "i.ebayimg.com, cdn.example.com",
            }
        )
        assert provider.image_resolver.policy.allowed_hosts == {"i.ebayimg.com", "cdn.example.com"}

    def test_from_env_invalid_allowed_hosts(self, pil):
        with pytest.raises(PreviewConfigurationError, match="PREVIEW_ALLOWED_IMAGE_HOSTS"):
            OpenAIImagePreviewProvider.from_env(
                {
                    "PREVIEW_PROVIDER": "openai",
                    "PREVIEW_API_KEY": SECRET_KEY,
                    "PREVIEW_MODEL": "m",
                    "PREVIEW_ALLOWED_IMAGE_HOSTS": "*.ebayimg.com",
                }
            )


class TestInternals:
    def test_system_dns_uses_getaddrinfo(self, monkeypatch):
        seen = {}

        def fake_getaddrinfo(host, port, type=0):
            seen["args"] = (host, port, type)
            return [
                (None, None, None, "", (PUBLIC_IP, port)),
                (None, None, None, "", ("2606:2800:220:1::1", port, 0, 0)),
            ]

        monkeypatch.setattr(preview_adapter.socket, "getaddrinfo", fake_getaddrinfo)
        assert preview_adapter._system_dns(HOST, 443) == [PUBLIC_IP, "2606:2800:220:1::1"]
        assert seen["args"] == (HOST, 443, preview_adapter.socket.SOCK_STREAM)

    def test_default_resolver_uses_system_dns(self):
        resolver = LocalAndHttpsImageResolver()
        assert resolver._dns is preview_adapter._system_dns

    def test_discard_ignores_os_errors(self, tmp_path, monkeypatch):
        def locked(self, missing_ok=False):
            raise PermissionError("locked")

        monkeypatch.setattr(Path, "unlink", locked)
        preview_adapter._discard(tmp_path / "x.part")  # must not raise

    def test_from_env_keeps_injected_resolver(self, pil):
        pytest.importorskip("openai")
        resolver = LocalAndHttpsImageResolver(RemoteImagePolicy(allowed_hosts=[HOST]))
        provider = OpenAIImagePreviewProvider.from_env(
            {
                "PREVIEW_PROVIDER": "openai",
                "PREVIEW_API_KEY": SECRET_KEY,
                "PREVIEW_MODEL": "m",
                "PREVIEW_ALLOWED_IMAGE_HOSTS": "*.ignored.example",
            },
            image_resolver=resolver,
        )
        assert provider.image_resolver is resolver


class TestFromEnvFile:
    def write_env(self, tmp_path, text):
        path = tmp_path / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_preview_values_only_when_called(self, pil, tmp_path, monkeypatch):
        pytest.importorskip("openai")
        for name in (
            "PREVIEW_PROVIDER",
            "PREVIEW_API_KEY",
            "PREVIEW_MODEL",
            "PREVIEW_ALLOWED_IMAGE_HOSTS",
        ):
            monkeypatch.delenv(name, raising=False)
        path = self.write_env(
            tmp_path,
            (
                "# comment\n\nexport PREVIEW_PROVIDER=openai\n"
                f"PREVIEW_API_KEY=\"{SECRET_KEY}\"\nPREVIEW_MODEL='file-model'\n"
                "PREVIEW_ALLOWED_IMAGE_HOSTS=i.ebayimg.com\nUNRELATED_SECRET=zzz\nnot a line\n"
            ),
        )
        provider = OpenAIImagePreviewProvider.from_env_file(path, environ={})
        assert provider.model == "file-model"
        assert provider.image_resolver.policy.allowed_hosts == {"i.ebayimg.com"}
        assert "PREVIEW_MODEL" not in os.environ and "UNRELATED_SECRET" not in os.environ

    def test_real_environment_takes_precedence(self, pil, tmp_path):
        pytest.importorskip("openai")
        path = self.write_env(
            tmp_path,
            f"PREVIEW_PROVIDER=openai\nPREVIEW_API_KEY={SECRET_KEY}\nPREVIEW_MODEL=file-model\n",
        )
        provider = OpenAIImagePreviewProvider.from_env_file(
            path, environ={"PREVIEW_MODEL": "env-model"}
        )
        assert provider.model == "env-model"

    def test_missing_file(self, pil, tmp_path):
        with pytest.raises(PreviewConfigurationError) as exc:
            OpenAIImagePreviewProvider.from_env_file(tmp_path / "nope.env", environ={})
        assert str(tmp_path) not in str(exc.value)

    def test_file_values_still_validated(self, pil, tmp_path):
        path = self.write_env(tmp_path, "PREVIEW_PROVIDER=openai\nPREVIEW_MODEL=m\n")
        with pytest.raises(PreviewConfigurationError, match="PREVIEW_API_KEY"):
            OpenAIImagePreviewProvider.from_env_file(path, environ={})

    def test_import_does_not_read_env_file(self):
        assert "from_env_file" in vars(OpenAIImagePreviewProvider)
        assert not hasattr(preview_adapter, "_LOADED_ENV")
