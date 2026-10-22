"""Stage 14 — AI-Generated Outfit Preview (generative visual preview, not a virtual try-on).

Builds a PreviewRequest from a ConfirmedOutfit, turns it into a deterministic prompt, and asks a
replaceable PreviewProvider for an image. If generation is unavailable or fails, the result
falls back to PRODUCT_CARDS. A preview never changes or undoes the confirmation itself.

Providers:
- MockPreviewProvider: deterministic, no network, no credentials (tests, CI, demos).
- OpenAIImagePreviewProvider: optional adapter for the OpenAI Images API. The `openai` SDK,
  Pillow and `requests` (the `preview` extra) are imported lazily, and configuration
  (PREVIEW_PROVIDER, PREVIEW_API_KEY, PREVIEW_MODEL, PREVIEW_ALLOWED_IMAGE_HOSTS) is read only
  when a provider is explicitly created, so importing this module needs no SDK, no key, no file
  and no network.

Image inputs go through an injectable ImageReferenceResolver. The default
LocalAndHttpsImageResolver prefers local files (e.g. upstream-downloaded `image_path`) and
only fetches HTTPS product images from an explicit exact host allow-list, with DNS checks
against internal addresses, per-redirect re-validation, timeouts and size/pixel limits. Every
input is re-encoded to PNG in a module-owned temporary directory (EXIF dropped), which is always
deleted. Caller-owned files are never modified or deleted. Portraits are only accepted as local
files and sent only in PORTRAIT mode with consent (enforced by PreviewRequest). Errors, warnings
and prompts never contain keys, URLs, query strings, paths or image data.

All public names here are PROVISIONAL.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import ipaddress
import math
import os
import re
import shutil
import socket
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urljoin, urlsplit

from travel_agent.recommendation.student5.models import (
    ConfirmedOutfit,
    PreviewDeliveryMode,
    PreviewItemReference,
    PreviewMode,
    PreviewRequest,
    PreviewResult,
    PreviewStatus,
    ProductCategory,
)

CATEGORY_ORDER = (
    ProductCategory.TOP,
    ProductCategory.BOTTOM,
    ProductCategory.FOOTWEAR,
    ProductCategory.OUTERWEAR,
)

PREVIEW_DISCLAIMER = (
    "AI-generated preview for visual reference only. It is not a virtual try-on and does not "
    "guarantee the products' actual colour, texture or details, body proportions, size or fit."
)

PRODUCT_CARDS_PROVIDER_NAME = "product_cards"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "previews"
DEFAULT_MAX_INPUT_BYTES = 20 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120.0

PREVIEW_ENV_KEYS = (
    "PREVIEW_PROVIDER",
    "PREVIEW_API_KEY",
    "PREVIEW_MODEL",
    "PREVIEW_ALLOWED_IMAGE_HOSTS",
)

_GENERATIVE_WARNINGS = {
    PreviewMode.PORTRAIT: [
        "The person's likeness, pose and body proportions may be altered by the generative model.",
        "Garment colours, textures and details may differ from the actual products.",
    ],
    PreviewMode.FLAT_LAY: [
        "Garment colours, textures and details may differ from the actual products.",
    ],
}

_SKIP_WARNINGS = {
    "not_allowed": "{n} product image reference(s) were not used: not permitted by the remote "
    "image policy.",
    "download": "{n} product image(s) could not be downloaded and were not used.",
    "invalid": "{n} product image(s) could not be read as valid images and were not used.",
}

_OUTPUT_EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PreviewProviderError(Exception):
    """PROVISIONAL. The preview provider could not produce a preview (recoverable by fallback)."""


class PreviewConfigurationError(PreviewProviderError):
    """PROVISIONAL. Provider not configured: missing/invalid settings, SDK or Pillow."""


class PreviewInputError(PreviewProviderError):
    """PROVISIONAL. An input image cannot be used.

    `reason` is one of "not_allowed" (policy), "download" (fetch failed) or "invalid" (not a
    usable image); messages never contain the reference itself.
    """

    def __init__(self, message: str, reason: str = "invalid") -> None:
        super().__init__(message)
        self.reason = reason


class PreviewOutputError(PreviewProviderError):
    """PROVISIONAL. The provider's output is missing, invalid or could not be stored."""


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreviewConfigurationError(f"{name} must be a positive integer")
    return value


def _positive_seconds(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise PreviewConfigurationError(f"{name} must be a finite positive number")
    return float(value)


_HOST_PATTERN = re.compile(
    r"(?=.{1,253}\Z)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*"
)


def _normalise_host(host: Any) -> str:
    if not isinstance(host, str) or not _HOST_PATTERN.fullmatch(host.strip().lower()):
        raise PreviewConfigurationError("invalid allowed host entry (exact host names only)")
    return host.strip().lower()


def parse_allowed_hosts(value: str | None) -> frozenset[str]:
    """PROVISIONAL. Parse a comma-separated list of exact host names (no wildcards)."""
    if not value:
        return frozenset()
    return frozenset(_normalise_host(part) for part in value.split(",") if part.strip())


def _uri_scheme(reference: str) -> str | None:
    """Lower-cased URI scheme of `reference`, or None for a local path.

    A one-letter "scheme" is a Windows drive letter (C:\\..., C:/..., D:file), so it is a local
    path. A reference that looks like a URI but cannot be parsed returns "invalid", so it is
    treated as a (disallowed) URI rather than a local file.
    """
    try:
        scheme = urlsplit(reference).scheme
    except ValueError:
        return "invalid"
    return scheme.lower() if len(scheme) > 1 else None


def _is_remote(reference: str) -> bool:
    """True for any URI reference (https, http, file, data, ...); False for local paths."""
    return _uri_scheme(reference) is not None


def _import_optional(module: str, purpose: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError:
        raise PreviewConfigurationError(
            f'{purpose} is not installed; install the optional extra: pip install -e ".[preview]"'
        ) from None


# ---------------------------------------------------------------------------
# Image reference resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteImagePolicy:
    """PROVISIONAL. Limits for downloading remote product images.

    `allowed_hosts` is an exact, case-insensitive host allow-list (no wildcards); empty means no
    remote image is fetched.
    """

    allowed_hosts: frozenset[str] = frozenset()
    max_redirects: int = 3
    max_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 20_000_000
    max_dimension: int = 1024
    connect_timeout: float = 5.0
    read_timeout: float = 10.0
    total_timeout: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.allowed_hosts, str):
            raise PreviewConfigurationError(
                "allowed_hosts must be a collection of host names, not a single string"
            )
        object.__setattr__(
            self, "allowed_hosts", frozenset(_normalise_host(h) for h in self.allowed_hosts)
        )
        for name in ("max_redirects", "max_bytes", "max_pixels", "max_dimension"):
            _positive_int(name, getattr(self, name))
        for name in ("connect_timeout", "read_timeout", "total_timeout"):
            object.__setattr__(self, name, _positive_seconds(name, getattr(self, name)))


@dataclass(frozen=True)
class ResolvedSource:
    """PROVISIONAL. A resolver result: a module-owned PNG file and where it came from."""

    path: Path
    source: Literal["local", "remote"]


@dataclass(frozen=True)
class ResolvedImage:
    """PROVISIONAL. One reference image actually sent to the provider, in send order."""

    path: Path
    source: Literal["local", "remote"]
    is_portrait: bool
    item_position: int | None  # 1-based position in PreviewRequest.items; None for the portrait
    category: ProductCategory | None


class ImageReferenceResolver(Protocol):
    """PROVISIONAL. Turns an image reference into a PNG inside `work_dir`, or raises
    PreviewInputError. Must never modify or delete the caller's file."""

    def resolve(
        self, reference: str, work_dir: Path, stem: str, *, allow_remote: bool = True
    ) -> ResolvedSource:
        """Return the resolved PNG; raise PreviewInputError when the reference is unusable."""


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_FETCH_HEADERS = {"User-Agent": "student5-preview-image-fetcher/0.1", "Accept": "image/*"}


def _is_public_address(text: str) -> bool:
    try:
        ip = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _system_dns(host: str, port: int) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


class LocalAndHttpsImageResolver:
    """PROVISIONAL. Default resolver: local files first, restricted HTTPS fetch as a fallback.

    Local path (anything without a URI scheme; Windows drive letters count as local): size-checked
    and re-encoded. URI (only if `allow_remote`): only `https` is fetched, every other scheme is
    `not_allowed`; no user info, port 443, host in the exact allow-list, every DNS answer public,
    each redirect re-validated (max `max_redirects`), streamed, `image/*` content type, at most
    `max_bytes`, then decoded with Pillow. One `total_timeout` deadline covers DNS, all requests
    and body reading: the remaining time is checked before each DNS lookup, before each request
    (whose connect/read timeouts are capped at the remaining time) and for every chunk. Every
    image is checked against
    `max_pixels`, EXIF-transposed, downscaled to `max_dimension` and saved as PNG. The download
    uses a fresh session without credentials, cookies, proxies-from-environment or caller
    headers.
    """

    def __init__(
        self,
        policy: RemoteImagePolicy | None = None,
        *,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        session_factory: Callable[[], Any] | None = None,
        dns_resolver: Callable[[str, int], Iterable[str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy if policy is not None else RemoteImagePolicy()
        self.max_input_bytes = _positive_int("max_input_bytes", max_input_bytes)
        self._session_factory = session_factory or self._new_session
        self._dns = dns_resolver or _system_dns
        self._clock = clock

    # -- public ------------------------------------------------------------

    def resolve(
        self, reference: str, work_dir: Path, stem: str, *, allow_remote: bool = True
    ) -> ResolvedSource:
        target = Path(work_dir) / f"{stem}.png"
        scheme = _uri_scheme(reference)
        if scheme is not None:
            if not allow_remote:
                raise PreviewInputError(
                    "remote references are not accepted here", reason="not_allowed"
                )
            if scheme != "https":
                raise PreviewInputError(
                    "only https image references can be fetched", reason="not_allowed"
                )
            download = Path(work_dir) / f"{stem}.download"
            try:
                self._download(reference, download)
                self._normalise(download, target)
            finally:
                _discard(download)
            return ResolvedSource(target, "remote")
        source = Path(reference)
        if not source.is_file():
            raise PreviewInputError("image reference is not a readable local file")
        if source.stat().st_size > self.max_input_bytes:
            raise PreviewInputError("image file exceeds the input size limit")
        self._normalise(source, target)
        return ResolvedSource(target, "local")

    # -- remote ------------------------------------------------------------

    @staticmethod
    def _new_session() -> Any:
        try:
            requests = importlib.import_module("requests")
        except ImportError:
            raise PreviewInputError(
                "remote image download needs the optional requests package", reason="download"
            ) from None
        session = requests.Session()
        session.trust_env = False  # no proxy / netrc credentials from the environment
        session.headers.clear()
        session.cookies.clear()
        session.auth = None
        return session

    def _remaining(self, deadline: float) -> float:
        """Seconds left before the total deadline; raises once it has been reached."""
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise PreviewInputError("remote image download took too long", reason="download")
        return remaining

    def _check_url(self, url: str, deadline: float) -> None:
        def deny() -> PreviewInputError:
            return PreviewInputError("remote image reference not permitted", reason="not_allowed")

        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError:
            raise deny() from None
        if (
            parts.scheme.lower() != "https"
            or parts.username is not None
            or parts.password is not None
        ):
            raise deny()
        if port not in (None, 443):
            raise deny()
        host = (parts.hostname or "").lower()
        if not host or host not in self.policy.allowed_hosts:
            raise deny()
        self._remaining(deadline)  # before any DNS lookup
        try:
            ipaddress.ip_address(host)
            addresses = [host]
        except ValueError:
            addresses = list(self._dns(host, 443))
        if not addresses or not all(_is_public_address(a) for a in addresses):
            raise deny()

    def _download(self, url: str, target: Path) -> None:
        policy = self.policy
        deadline = self._clock() + policy.total_timeout
        session = None
        try:
            session = self._session_factory()
            for _ in range(policy.max_redirects + 1):
                self._check_url(url, deadline)
                remaining = self._remaining(deadline)  # recomputed before every request
                response = session.get(
                    url,
                    stream=True,
                    allow_redirects=False,
                    timeout=(
                        min(policy.connect_timeout, remaining),
                        min(policy.read_timeout, remaining),
                    ),
                    headers=dict(_FETCH_HEADERS),
                )
                try:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("Location")
                        if not location:
                            raise PreviewInputError("redirect without a target", reason="download")
                        url = urljoin(url, location)
                        continue
                    if response.status_code != 200:
                        raise PreviewInputError(
                            "remote image request was not successful", reason="download"
                        )
                    self._read_image_body(response, target, deadline)
                    return
                finally:
                    response.close()
            raise PreviewInputError("too many redirects", reason="download")
        except OSError:
            raise PreviewInputError("remote image download failed", reason="download") from None
        finally:
            if session is not None:
                session.close()

    def _read_image_body(self, response: Any, target: Path, deadline: float) -> None:
        policy = self.policy
        content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0]
        if not content_type.strip().lower().startswith("image/"):
            raise PreviewInputError("remote content is not an image", reason="invalid")
        try:
            declared = int(response.headers.get("Content-Length", ""))
        except ValueError:
            declared = None
        if declared is not None and declared > policy.max_bytes:
            raise PreviewInputError("remote image exceeds the size limit", reason="download")
        received = 0
        with open(target, "wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                self._remaining(deadline)
                received += len(chunk)
                if received > policy.max_bytes:
                    raise PreviewInputError(
                        "remote image exceeds the size limit", reason="download"
                    )
                handle.write(chunk)

    # -- decoding ------------------------------------------------------------

    def _normalise(self, source: Path, target: Path) -> None:
        image = _import_optional("PIL.Image", "Pillow")
        ops = _import_optional("PIL.ImageOps", "Pillow")
        try:
            with image.open(source) as img:
                width, height = img.size
                if width * height > self.policy.max_pixels:
                    raise PreviewInputError("image has too many pixels")
                img.load()
                has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
                converted = ops.exif_transpose(img).convert("RGBA" if has_alpha else "RGB")
            converted.thumbnail((self.policy.max_dimension, self.policy.max_dimension))
            converted.save(target, format="PNG")
        except (OSError, ValueError, image.DecompressionBombError):
            _discard(target)
            raise PreviewInputError("image file is not a valid image") from None


# ---------------------------------------------------------------------------
# Request and prompt
# ---------------------------------------------------------------------------


def build_preview_request(
    confirmed_outfit: ConfirmedOutfit,
    preview_mode: PreviewMode | str,
    user_portrait: str | None = None,
    portrait_consent: bool = False,
) -> PreviewRequest:
    """Build a PreviewRequest from a ConfirmedOutfit (input not modified).

    Items come from existing_items and purchase_items, in canonical category order
    TOP → BOTTOM → FOOTWEAR → OUTERWEAR. Portrait and consent rules are enforced by
    PreviewRequest itself.
    """
    if not isinstance(confirmed_outfit, ConfirmedOutfit):
        raise TypeError(f"expected a ConfirmedOutfit, got {type(confirmed_outfit).__name__}")
    recommendation = confirmed_outfit.recommendation
    products = sorted(
        recommendation.existing_items + recommendation.purchase_items,
        key=lambda p: CATEGORY_ORDER.index(p.category),
    )
    items = [
        {
            "product_id": p.product_id,
            "category": p.category,
            "image_reference": p.image_url,
            "title": p.title,
            "description": p.description,
            "is_user_owned": p.is_user_owned,
        }
        for p in products
    ]
    return PreviewRequest.model_validate(
        {
            "selected_outfit_id": confirmed_outfit.selected_outfit_id,
            "user_portrait": user_portrait,
            "portrait_consent": portrait_consent,
            "items": items,
            "preview_mode": preview_mode,
        }
    )


_MODE_INSTRUCTIONS = {
    PreviewMode.PORTRAIT: (
        "The first reference image is the user's portrait. Keep the person's identity, face, "
        "pose and background as close to the portrait as possible, and dress the person in the "
        "outfit items listed below. Any further reference images show those items in the listed "
        "order. Do not claim or imply that the result shows the real size, fit or try-on result."
    ),
    PreviewMode.FLAT_LAY: (
        "Create a coordinated, body-shaped outfit composition with no person or mannequin. "
        "Place the outerwear in the centre, slightly open; layer the top visibly inside it; "
        "align the trousers naturally below the top; and place the pair of shoes below the "
        "trouser legs. The garments should read as one styled outfit on an invisible body. "
        "Do not scatter items, place them side-by-side, or use a catalogue grid."
    ),
}


def _item_text(item: PreviewItemReference) -> str:
    return item.description if item.description is not None else item.title


def build_preview_prompt(request: PreviewRequest) -> str:
    """Deterministic base prompt for a PreviewRequest.

    Uses only the mode, each item's category and its description (or title when there is no
    description), in request order. No brand, colour, material, price, link or product id is
    added.
    """
    if not isinstance(request, PreviewRequest):
        raise TypeError(f"expected a PreviewRequest, got {type(request).__name__}")
    lines = [
        f"Create an AI-generated visual outfit preview. Mode: {request.preview_mode.value}.",
        _MODE_INSTRUCTIONS[request.preview_mode],
        "Outfit items, in order:",
        *(
            f"{n}. {item.category.value}: {_item_text(item)}"
            for n, item in enumerate(request.items, start=1)
        ),
        "Use the product reference images only as a visual reference for these garments. Do not "
        "add logos, text, brands or garment details that are not shown in the references or "
        "described above.",
        "This image is a visual reference only, not a virtual try-on; it does not show exact "
        "colours, details, body proportions, size or fit.",
    ]
    return "\n".join(lines)


def build_prompt_with_image_mapping(
    request: PreviewRequest, references: Sequence[ResolvedImage]
) -> str:
    """PROVISIONAL. Base prompt plus the mapping of the images actually sent.

    Image numbers follow the send order; each product image names its outfit item position
    and category, so skipped images never shift the mapping. No path or URL is included.
    """
    base = build_preview_prompt(request)
    if not references:
        return base + "\nNo reference images are provided; use the item descriptions above."
    lines = [base, "Reference images (in the order they are provided):"]
    for n, ref in enumerate(references, start=1):
        if ref.is_portrait:
            lines.append(f"Reference image {n} is the user's portrait.")
        else:
            lines.append(
                f"Reference image {n} corresponds to outfit item {ref.item_position}: "
                f"{ref.category.value}."
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Results and providers
# ---------------------------------------------------------------------------


def build_product_cards_result(
    request: PreviewRequest,
    status: PreviewStatus = PreviewStatus.SUCCESS,
    provider_name: str = PRODUCT_CARDS_PROVIDER_NAME,
    warnings: list[str] | None = None,
) -> PreviewResult:
    """PRODUCT_CARDS result rendered from the request items; no image is generated."""
    return PreviewResult(
        selected_outfit_id=request.selected_outfit_id,
        status=status,
        requested_mode=request.preview_mode,
        delivered_mode=PreviewDeliveryMode.PRODUCT_CARDS,
        output_reference=None,
        provider_name=provider_name,
        model_name=None,
        disclaimer=PREVIEW_DISCLAIMER,
        warnings=list(warnings or []),
        generation_time_seconds=None,
    )


class PreviewProvider(Protocol):
    """PROVISIONAL provider boundary; implementations are injected, never hard-wired."""

    provider_name: str

    def generate_preview(self, request: PreviewRequest) -> PreviewResult:
        """Return a PreviewResult or raise PreviewProviderError (or a subclass)."""


class MockPreviewProvider:
    """PROVISIONAL. Deterministic provider without network or credentials.

    `failure`, if given, is raised for every request to exercise the fallback path.
    """

    provider_name = "mock"

    def __init__(self, failure: PreviewProviderError | None = None) -> None:
        self.failure = failure
        self.calls: list[PreviewMode] = []

    def generate_preview(self, request: PreviewRequest) -> PreviewResult:
        self.calls.append(request.preview_mode)
        if self.failure is not None:
            raise self.failure
        mode = request.preview_mode
        return PreviewResult(
            selected_outfit_id=request.selected_outfit_id,
            status=PreviewStatus.SUCCESS,
            requested_mode=mode,
            delivered_mode=PreviewDeliveryMode(mode.value),
            output_reference=f"mock://preview/{request.selected_outfit_id}/{mode.value.lower()}",
            provider_name=self.provider_name,
            model_name=None,
            disclaimer=PREVIEW_DISCLAIMER,
            warnings=list(_GENERATIVE_WARNINGS[mode]),
            generation_time_seconds=0.0,
        )


def _parse_env_file(text: str) -> dict[str, str]:
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or key not in PREVIEW_ENV_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


class OpenAIImagePreviewProvider:
    """PROVISIONAL. Preview provider backed by the OpenAI Images API (edit / generate).

    - PORTRAIT: images.edit with the consented local portrait first, then the resolved product
      images in request order.
    - FLAT_LAY: images.edit with the resolved product images, or images.generate
      from the text prompt when no product image is usable.
    The prompt states which outfit item each sent image belongs to. The base64 result is
    validated with Pillow and saved under `output_dir` with a random UUID file name.
    """

    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any = None,
        output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
        temp_dir_parent: str | os.PathLike[str] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        api_error_types: tuple[type[BaseException], ...] | None = None,
        image_resolver: ImageReferenceResolver | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise PreviewConfigurationError("a non-empty image model name is required")
        self.model = model.strip()
        self.timeout_seconds = _positive_seconds("timeout_seconds", timeout_seconds)
        self.max_input_bytes = _positive_int("max_input_bytes", max_input_bytes)
        self._image = _import_optional("PIL.Image", "Pillow")
        if client is None:
            if not api_key:
                raise PreviewConfigurationError("an API key is required to create the client")
            openai = _import_optional("openai", "The openai SDK")
            client = openai.OpenAI(api_key=api_key)
        self._client = client
        if api_error_types is None:
            try:
                api_error_types = (importlib.import_module("openai").OpenAIError,)
            except ImportError:
                api_error_types = ()
        self._api_error_types = api_error_types
        self.image_resolver = (
            image_resolver
            if image_resolver is not None
            else LocalAndHttpsImageResolver(max_input_bytes=self.max_input_bytes)
        )
        self.output_dir = Path(output_dir)
        self.temp_dir_parent = temp_dir_parent

    def __repr__(self) -> str:
        return (
            f"OpenAIImagePreviewProvider(model={self.model!r}, output_dir={str(self.output_dir)!r})"
        )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, **kwargs: Any
    ) -> OpenAIImagePreviewProvider:
        """Create the provider from PREVIEW_* variables (read only when called)."""
        env = os.environ if environ is None else environ
        values = {name: (env.get(name) or "").strip() for name in PREVIEW_ENV_KEYS}
        if values["PREVIEW_PROVIDER"] != "openai":
            raise PreviewConfigurationError("PREVIEW_PROVIDER must be set to 'openai'")
        for name in ("PREVIEW_API_KEY", "PREVIEW_MODEL"):
            if not values[name]:
                raise PreviewConfigurationError(f"{name} is not set")
        if "image_resolver" not in kwargs:
            try:
                hosts = parse_allowed_hosts(values["PREVIEW_ALLOWED_IMAGE_HOSTS"])
            except PreviewConfigurationError:
                raise PreviewConfigurationError(
                    "PREVIEW_ALLOWED_IMAGE_HOSTS contains an invalid host"
                ) from None
            kwargs["image_resolver"] = LocalAndHttpsImageResolver(
                RemoteImagePolicy(allowed_hosts=hosts),
                max_input_bytes=kwargs.get("max_input_bytes", DEFAULT_MAX_INPUT_BYTES),
            )
        return cls(model=values["PREVIEW_MODEL"], api_key=values["PREVIEW_API_KEY"], **kwargs)

    @classmethod
    def from_env_file(
        cls,
        path: str | os.PathLike[str] = ".env",
        environ: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> OpenAIImagePreviewProvider:
        """Create the provider from PREVIEW_* entries of a .env file, only when called.

        Only PREVIEW_* keys are read; the file is never exported into os.environ. Non-empty
        values in the real environment take precedence over the file.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            raise PreviewConfigurationError("the environment file could not be read") from None
        env = os.environ if environ is None else environ
        merged = _parse_env_file(text)
        merged.update({k: env[k] for k in PREVIEW_ENV_KEYS if env.get(k)})
        return cls.from_env(merged, **kwargs)

    # -- public ------------------------------------------------------------

    def generate_preview(self, request: PreviewRequest) -> PreviewResult:
        if not isinstance(request, PreviewRequest):
            raise TypeError(f"expected a PreviewRequest, got {type(request).__name__}")
        started = time.perf_counter()
        work_dir = Path(tempfile.mkdtemp(prefix="student5_preview_", dir=self.temp_dir_parent))
        try:
            references, input_warnings = self._prepare_inputs(request, work_dir)
            prompt = build_prompt_with_image_mapping(request, references)
            b64 = self._call_api(prompt, [ref.path for ref in references])
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        output_path = self._save_output(b64)
        mode = request.preview_mode
        return PreviewResult(
            selected_outfit_id=request.selected_outfit_id,
            status=PreviewStatus.SUCCESS,
            requested_mode=mode,
            delivered_mode=PreviewDeliveryMode(mode.value),
            output_reference=output_path.as_posix(),
            provider_name=self.provider_name,
            model_name=self.model,
            disclaimer=PREVIEW_DISCLAIMER,
            warnings=[*_GENERATIVE_WARNINGS[mode], *input_warnings],
            generation_time_seconds=max(0.0, time.perf_counter() - started),
        )

    # -- inputs ------------------------------------------------------------

    def _prepare_inputs(
        self, request: PreviewRequest, work_dir: Path
    ) -> tuple[list[ResolvedImage], list[str]]:
        references: list[ResolvedImage] = []
        if request.preview_mode is PreviewMode.PORTRAIT:
            portrait = self.image_resolver.resolve(
                request.user_portrait, work_dir, "00_portrait", allow_remote=False
            )
            references.append(ResolvedImage(portrait.path, portrait.source, True, None, None))
        skipped: Counter[str] = Counter()
        for position, item in enumerate(request.items, start=1):
            if item.image_reference is None:
                continue
            try:
                resolved = self.image_resolver.resolve(
                    item.image_reference,
                    work_dir,
                    f"{position:02d}_{item.category.value.lower()}",
                    allow_remote=True,
                )
            except PreviewInputError as exc:
                skipped[exc.reason] += 1
                continue
            references.append(
                ResolvedImage(resolved.path, resolved.source, False, position, item.category)
            )
        warnings = [
            message.format(n=skipped[reason])
            for reason, message in _SKIP_WARNINGS.items()
            if skipped[reason]
        ]
        return references, warnings

    # -- API ---------------------------------------------------------------

    def _call_api(self, prompt: str, images: list[Path]) -> str:
        with ExitStack() as stack:
            files = [stack.enter_context(open(path, "rb")) for path in images]
            try:
                if files:
                    response = self._client.images.edit(
                        model=self.model,
                        image=files if len(files) > 1 else files[0],
                        prompt=prompt,
                        n=1,
                        timeout=self.timeout_seconds,
                    )
                else:
                    response = self._client.images.generate(
                        model=self.model, prompt=prompt, n=1, timeout=self.timeout_seconds
                    )
            except self._api_error_types as exc:
                raise PreviewProviderError(
                    f"image API request failed ({type(exc).__name__})"
                ) from None
        data = getattr(response, "data", None)
        b64 = getattr(data[0], "b64_json", None) if data else None
        if not b64:
            raise PreviewOutputError("image API returned no base64 image data")
        return b64

    # -- output ------------------------------------------------------------

    def _save_output(self, b64: str) -> Path:
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            raise PreviewOutputError("image API returned invalid base64 data") from None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise PreviewOutputError("the preview output directory is not usable") from None
        name = uuid.uuid4().hex
        partial = self.output_dir / f"{name}.part"
        try:
            try:
                with open(partial, "xb") as handle:
                    handle.write(raw)
            except OSError:
                raise PreviewOutputError("the preview output could not be written") from None
            try:
                with self._image.open(partial) as img:
                    img.load()
                    image_format = img.format
            except (OSError, ValueError, self._image.DecompressionBombError):
                raise PreviewOutputError("image API output is not a readable image") from None
            extension = _OUTPUT_EXTENSIONS.get(image_format)
            if extension is None:
                raise PreviewOutputError("image API output has an unsupported image format")
            final = self.output_dir / f"{name}{extension}"
            try:
                partial.replace(final)
            except OSError:
                raise PreviewOutputError("the preview output could not be stored") from None
        except BaseException:
            _discard(partial)
            raise
        return final


# ---------------------------------------------------------------------------
# Fallback entry point
# ---------------------------------------------------------------------------


def _check_provider_result(request: PreviewRequest, result: Any) -> PreviewResult:
    if not isinstance(result, PreviewResult):
        raise PreviewOutputError("provider did not return a PreviewResult")
    if (
        result.selected_outfit_id != request.selected_outfit_id
        or result.requested_mode is not request.preview_mode
    ):
        raise PreviewOutputError("provider result does not match the request")
    if result.status is PreviewStatus.FAILED:
        raise PreviewOutputError("provider reported a failed preview")
    return result


def generate_preview_with_fallback(
    request: PreviewRequest, provider: PreviewProvider | None
) -> PreviewResult:
    """PROVISIONAL. Generate a preview, falling back to PRODUCT_CARDS when generation fails.

    A missing provider or any PreviewProviderError (configuration, input, API or output problem)
    gives a FALLBACK result
    with delivered_mode PRODUCT_CARDS and a warning that contains no sensitive data. Other
    exceptions (programming errors) are not caught.
    """
    if not isinstance(request, PreviewRequest):
        raise TypeError(f"expected a PreviewRequest, got {type(request).__name__}")
    if provider is None:
        reason = "a preview provider is not configured"
    else:
        try:
            return _check_provider_result(request, provider.generate_preview(request))
        except PreviewConfigurationError:
            reason = "the preview provider is not configured correctly"
        except PreviewInputError:
            reason = "the input images could not be used"
        except PreviewOutputError:
            reason = "the provider output could not be validated"
        except PreviewProviderError:
            reason = "the preview provider request failed"
    return build_product_cards_result(
        request,
        status=PreviewStatus.FALLBACK,
        warnings=[f"Generative preview unavailable ({reason}); showing product cards instead."],
    )
