"""Tests for Module 8.1: restricted image resolver, image/prompt mapping, output robustness,
configuration validation and explicit env-file loading.

All HTTP and DNS traffic goes through fake sessions and fake DNS resolvers; nothing here
touches the Internet. Tests that create images need Pillow (the `preview` extra).
"""

import io
import sys
from types import SimpleNamespace

import pytest

from travel_agent.recommendation.student5.models import (
    ConfirmedOutfit,
    PreviewMode,
)
from travel_agent.recommendation.student5.preview_adapter import (
    LocalAndHttpsImageResolver,
    OpenAIImagePreviewProvider,
    PreviewInputError,
    RemoteImagePolicy,
    build_preview_request,
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


# ---------------------------------------------------------------------------
# URL rules
# ---------------------------------------------------------------------------


class TestUrlRules:
    @pytest.mark.parametrize(
        "url",
        [
            "http://img.example.com/a.png",  # not HTTPS
            "https://other.example.com/a.png",  # host not on the allow-list
            "https://user:pw@img.example.com/a.png",  # credentials in the URL
            "https://img.example.com:8443/a.png",  # non-default port
            "https://img.example.com:bad/a.png",  # unparsable port
        ],
    )
    def test_rejected_without_request(self, pil, tmp_path, url):
        resolver, session, _ = make_resolver({})
        rejected(resolver, tmp_path, url, "not_allowed")
        assert session.calls == []

    def test_host_matching_is_exact(self, pil, tmp_path):
        resolver, session, _ = make_resolver({})
        for url in ("https://evil.img.example.com/a.png", "https://img.example.com.evil.io/a.png"):
            rejected(resolver, tmp_path, url, "not_allowed")
        assert session.calls == []

    @pytest.mark.parametrize(
        "host,url",
        [
            ("localhost", "https://localhost/a.png"),  # name resolving to loopback
            ("127.0.0.1", "https://127.0.0.1/a.png"),  # allow-listed IP literal, no DNS
        ],
    )
    def test_internal_hosts_rejected_even_if_allowed(self, pil, tmp_path, host, url):
        resolver, session, _ = make_resolver(
            {}, hosts=(host,), dns_table={"localhost": ["127.0.0.1"]}
        )
        rejected(resolver, tmp_path, url, "not_allowed")
        assert session.calls == []

    def test_ipv6_loopback_literal_rejected(self, pil, tmp_path):
        # A literal address never goes through DNS, unlike a host name resolving to ::1.
        resolver, session, dns = make_resolver({})
        rejected(resolver, tmp_path, "https://[::1]/a.png", "not_allowed")
        assert session.calls == [] and dns.calls == []

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",  # loopback
            "10.0.0.5",  # private
            "169.254.169.254",  # link-local / cloud metadata
            "::1",  # IPv6 loopback
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
            "not-an-ip",  # unparsable DNS answer
        ],
    )
    def test_dns_to_non_public_ip_rejected(self, pil, tmp_path, ip):
        resolver, session, _ = make_resolver({}, dns_table={HOST: [ip]})
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "not_allowed")
        assert session.calls == []

    def test_any_non_public_answer_rejects(self, pil, tmp_path):
        resolver, session, _ = make_resolver({}, dns_table={HOST: [PUBLIC_IP, "10.0.0.7"]})
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "not_allowed")


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------

CDN = "cdn.example.com"


class TestRedirects:
    @pytest.mark.parametrize(
        "target, cdn_ip",
        [
            ("https://evil.example.org/b.png", PUBLIC_IP),  # public host, not on the allow-list
            ("https://cdn.example.com/b.png", "192.168.0.10"),  # allowed host, internal address
        ],
        ids=["non_allowlisted_public", "internal_address"],
    )
    def test_redirect_to_disallowed_target_rejected(self, pil, tmp_path, target, cdn_ip):
        routes = {"https://img.example.com/a.png": redirect(target)}
        resolver, session, _ = make_resolver(
            routes, hosts=(HOST, CDN), dns_table={HOST: [PUBLIC_IP], CDN: [cdn_ip]}
        )
        rejected(resolver, tmp_path, "https://img.example.com/a.png", "not_allowed")
        assert [u for u, _ in session.calls] == ["https://img.example.com/a.png"]

    def chain(self, pil, hops):
        urls = [f"https://img.example.com/{n}.png" for n in range(hops + 1)]
        routes = {urls[n]: redirect(urls[n + 1], status=308) for n in range(hops)}
        routes[urls[-1]] = image_response(pil, (20, 20))
        return urls[0], routes

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

    @pytest.mark.parametrize("content_type", ["text/html"])
    def test_non_image_content_type_rejected(self, pil, tmp_path, content_type):
        headers = {} if content_type is None else {"Content-Type": content_type}
        response = FakeResponse(200, headers, [png_bytes(pil, (20, 20))])
        resolver, _, _ = make_resolver({URL: response})
        rejected(resolver, tmp_path, URL, "invalid")
        assert not response.iterated and response.closed

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
            # Orientation is applied first (40x20 shown rotated), then EXIF is dropped and the
            # image is re-encoded as PNG.
            assert img.size == (20, 40)
            assert dict(img.getexif()) == {}
            assert img.format == "PNG"

    def test_errors_do_not_reveal_url_or_query(self, pil, tmp_path):
        resolver, _, _ = make_resolver({URL: FakeResponse(500)})
        error = rejected(resolver, tmp_path, URL, "download")
        assert "query-secret-123" not in str(error) and HOST not in str(error)

    def test_missing_requests_package(self, pil, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "requests", None)
        resolver = LocalAndHttpsImageResolver(
            RemoteImagePolicy(allowed_hosts=[HOST]), dns_resolver=FakeDNS({HOST: [PUBLIC_IP]})
        )
        rejected(resolver, tmp_path, URL, "download")


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


# ---------------------------------------------------------------------------
# Reference classification (URI vs local path)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Local files and temporary-file clean-up
# ---------------------------------------------------------------------------


class TestLocalAndCleanup:
    def test_download_file_removed_after_success(self, pil, tmp_path):
        resolver, _, _ = make_resolver({URL: image_response(pil, (20, 20))})
        source, work = resolve(resolver, tmp_path, URL)
        assert sorted(p.name for p in work.iterdir()) == [source.path.name]

    @pytest.mark.parametrize("response_kind", ["network"])
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


# ---------------------------------------------------------------------------
# Output robustness
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------
