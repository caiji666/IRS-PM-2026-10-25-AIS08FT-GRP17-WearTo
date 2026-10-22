"""Tests for the Stage 14 Generative AI Outfit Preview (Module 8, preview_adapter).

No test here touches the network or needs credentials: the OpenAI adapter is exercised with a
fake client. Tests that create or inspect images need Pillow (part of the `preview` extra) and
are skipped without it.
"""

import base64
import io
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from travel_agent.recommendation.student5.models import (
    ConfirmedOutfit,
    PreviewDeliveryMode,
    PreviewMode,
    PreviewResult,
    PreviewStatus,
)
from travel_agent.recommendation.student5.preview_adapter import (
    PREVIEW_DISCLAIMER,
    MockPreviewProvider,
    OpenAIImagePreviewProvider,
    PreviewConfigurationError,
    PreviewOutputError,
    PreviewProviderError,
    build_preview_prompt,
    build_preview_request,
    generate_preview_with_fallback,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECRET_KEY = "test-key-not-a-real-credential"
GENERATIVE_MODES = [PreviewMode.FLAT_LAY]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def product(category, price="50.00", owned=False, **overrides):
    data = {
        "product_id": f"p_{category.lower()}",
        "source": "mock",
        "title": f"Mock {category.lower()}",
        "description": None,
        "product_url": None if owned else f"https://shop.example.com/{category.lower()}",
        "image_url": f"https://img.example.com/{category.lower()}.jpg",
        "price": price,
        "currency": "SGD",
        "availability": True,
        "category": category,
        "product_score": 0.5,
        "is_user_owned": owned,
    }
    data.update(overrides)
    return data


def make_confirmed(image_urls=None, outfit_id="outfit_202"):
    image_urls = image_urls or {}

    def p(category, **kw):
        if category in image_urls:
            kw["image_url"] = image_urls[category]
        return product(category, **kw)

    purchase = [  # deliberately not in canonical order
        p(
            "OUTERWEAR",
            price="120.00",
            description="Charcoal wool overcoat",
            brand="SecretBrand",
            colour="charcoal",
            material="wool",
        ),
        p("FOOTWEAR", price="80.00"),
        p("TOP", price="40.00", description="Cream cable-knit sweater"),
    ]
    existing = [p("BOTTOM", price="60.00", owned=True)]
    return ConfirmedOutfit.model_validate(
        {
            "session_id": "session_001",
            "selected_outfit_id": outfit_id,
            "recommendation": {
                "outfit_id": outfit_id,
                "display_position": 2,
                "existing_items": existing,
                "purchase_items": purchase,
                "total_purchase_cost": "240.00",
                "currency": "SGD",
                "compatibility_score": 0.75,
                "outfit_product_relevance": 0.5,
                "final_outfit_score": 0.625,
            },
        }
    )


def make_request(mode=PreviewMode.FLAT_LAY, confirmed=None, **kw):
    return build_preview_request(confirmed or make_confirmed(), mode, **kw)


@pytest.fixture
def pil():
    return pytest.importorskip("PIL.Image")


def write_png(pil, path, size, colour=(200, 30, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.new("RGB", size, colour).save(path, format="PNG")
    return path


def png_b64(pil, size=(16, 16), fmt="PNG"):
    buffer = io.BytesIO()
    pil.new("RGB", size, (10, 120, 200)).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeAPIError(Exception):
    """Stands in for an SDK error type."""


class FakeImages:
    def __init__(self, pil, b64=None, error=None, data=None):
        self.pil = pil
        self.b64 = b64
        self.error = error
        self.data = data
        self.calls = []

    def _respond(self):
        if self.error is not None:
            raise self.error
        if self.data is not None:
            return SimpleNamespace(data=self.data)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=self.b64)])

    def edit(self, **kwargs):
        images = kwargs["image"] if isinstance(kwargs["image"], list) else [kwargs["image"]]
        sizes = []
        for f in images:
            with self.pil.open(f) as img:
                sizes.append(img.size)
        self.calls.append(("edit", kwargs, sizes, [Path(f.name).name for f in images]))
        return self._respond()

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs, [], []))
        return self._respond()


def make_provider(pil, tmp_path, images=None, **kwargs):
    fake = images or FakeImages(pil, b64=png_b64(pil))
    provider = OpenAIImagePreviewProvider(
        model="configured-image-model",
        client=SimpleNamespace(images=fake),
        output_dir=tmp_path / "previews",
        temp_dir_parent=tmp_path / "work",
        api_error_types=(FakeAPIError,),
        **kwargs,
    )
    (tmp_path / "work").mkdir(exist_ok=True)
    return provider, fake


def reference_lines(prompt):
    return [line for line in prompt.splitlines() if line.startswith("Reference image ")]


def local_images(pil, tmp_path):
    """Local product images with distinct sizes so their order can be checked."""
    return {
        "TOP": str(write_png(pil, tmp_path / "in" / "top.png", (11, 11))),
        "BOTTOM": str(write_png(pil, tmp_path / "in" / "bottom.png", (12, 12))),
        "FOOTWEAR": str(write_png(pil, tmp_path / "in" / "footwear.png", (13, 13))),
        "OUTERWEAR": str(write_png(pil, tmp_path / "in" / "outerwear.png", (14, 14))),
    }


# ---------------------------------------------------------------------------
# build_preview_request
# ---------------------------------------------------------------------------


class TestBuildPreviewRequest:
    def test_requires_confirmed_outfit(self):
        with pytest.raises(TypeError, match="ConfirmedOutfit"):
            build_preview_request(make_confirmed().model_dump(), PreviewMode.FLAT_LAY)


# ---------------------------------------------------------------------------
# build_preview_prompt
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Product cards and mock provider
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


class RecordingProvider:
    provider_name = "recording"

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def generate_preview(self, request):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class TestGeneratePreviewWithFallback:
    def test_missing_provider_falls_back(self):
        result = generate_preview_with_fallback(make_request(PreviewMode.FLAT_LAY), None)
        assert result.status is PreviewStatus.FALLBACK
        assert result.requested_mode is PreviewMode.FLAT_LAY
        assert result.delivered_mode is PreviewDeliveryMode.PRODUCT_CARDS
        assert result.output_reference is None
        assert result.disclaimer == PREVIEW_DISCLAIMER
        assert "not configured" in result.warnings[0]

    @pytest.mark.parametrize(
        "error,reason",
        [
            (PreviewConfigurationError("x"), "not configured correctly"),
            (PreviewProviderError("x"), "request failed"),
        ],
    )
    def test_provider_errors_fall_back(self, error, reason):
        result = generate_preview_with_fallback(make_request(), RecordingProvider(error))
        assert result.status is PreviewStatus.FALLBACK
        assert result.delivered_mode is PreviewDeliveryMode.PRODUCT_CARDS
        assert reason in result.warnings[0]

    @pytest.mark.parametrize(
        "bad_result",
        [
            "not a result",
            PreviewResult(
                selected_outfit_id="someone_else",
                status="SUCCESS",
                requested_mode="FLAT_LAY",
                delivered_mode="FLAT_LAY",
                output_reference="x.png",
                provider_name="p",
                disclaimer="d",
            ),
            PreviewResult(
                selected_outfit_id="outfit_202",
                status="SUCCESS",
                requested_mode="PORTRAIT",
                delivered_mode="PORTRAIT",
                output_reference="x.png",
                provider_name="p",
                disclaimer="d",
            ),
            PreviewResult(
                selected_outfit_id="outfit_202",
                status="FAILED",
                requested_mode="FLAT_LAY",
                provider_name="p",
                disclaimer="d",
            ),
        ],
        ids=["not_a_result", "other_outfit", "other_mode", "provider_failed"],
    )
    def test_invalid_provider_result_falls_back(self, bad_result):
        result = generate_preview_with_fallback(make_request(), RecordingProvider(bad_result))
        assert result.status is PreviewStatus.FALLBACK
        assert result.delivered_mode is PreviewDeliveryMode.PRODUCT_CARDS

    def test_requires_preview_request(self):
        with pytest.raises(TypeError, match="PreviewRequest"):
            generate_preview_with_fallback(make_request().model_dump(), MockPreviewProvider())


# ---------------------------------------------------------------------------
# OpenAIImagePreviewProvider with a fake client
# ---------------------------------------------------------------------------


class TestOpenAIProviderCalls:
    def test_portrait_first_then_items_in_order(self, pil, tmp_path):
        portrait = write_png(pil, tmp_path / "in" / "portrait.png", (30, 40))
        req = make_request(
            PreviewMode.PORTRAIT,
            confirmed=make_confirmed(local_images(pil, tmp_path)),
            user_portrait=str(portrait),
            portrait_consent=True,
        )
        provider, fake = make_provider(pil, tmp_path)
        result = provider.generate_preview(req)
        [(kind, kwargs, sizes, names)] = fake.calls
        assert kind == "edit"
        assert sizes == [(30, 40), (11, 11), (12, 12), (13, 13), (14, 14)]
        assert names[0].startswith("00_portrait")
        assert kwargs["model"] == "configured-image-model"
        assert kwargs["prompt"].startswith(build_preview_prompt(req))
        assert reference_lines(kwargs["prompt"]) == [
            "Reference image 1 is the user's portrait.",
            "Reference image 2 corresponds to outfit item 1: TOP.",
            "Reference image 3 corresponds to outfit item 2: BOTTOM.",
            "Reference image 4 corresponds to outfit item 3: FOOTWEAR.",
            "Reference image 5 corresponds to outfit item 4: OUTERWEAR.",
        ]
        assert kwargs["n"] == 1
        assert result.status is PreviewStatus.SUCCESS
        assert result.delivered_mode is PreviewDeliveryMode.PORTRAIT

    @pytest.mark.parametrize("mode", GENERATIVE_MODES)
    def test_generate_path_without_usable_images(self, pil, tmp_path, mode):
        req = make_request(mode)  # only remote image URLs
        provider, fake = make_provider(pil, tmp_path)
        result = provider.generate_preview(req)
        [(kind, kwargs, _, _)] = fake.calls
        assert kind == "generate"
        assert kwargs["model"] == "configured-image-model"
        assert kwargs["prompt"].startswith(build_preview_prompt(req))
        assert "No reference images are provided" in kwargs["prompt"]
        assert reference_lines(kwargs["prompt"]) == []
        assert "image" not in kwargs
        assert any("4 product image reference(s)" in w for w in result.warnings)

    def test_portrait_not_sent_in_other_modes(self, pil, tmp_path):
        portrait = write_png(pil, tmp_path / "in" / "portrait.png", (30, 40))
        req = make_request(
            PreviewMode.FLAT_LAY,
            confirmed=make_confirmed(local_images(pil, tmp_path)),
            user_portrait=str(portrait),
            portrait_consent=True,
        )
        provider, fake = make_provider(pil, tmp_path)
        provider.generate_preview(req)
        assert (30, 40) not in fake.calls[0][2]


class TestOpenAIProviderOutput:
    @pytest.mark.parametrize("fmt,ext", [("JPEG", ".jpg")])
    def test_extension_follows_actual_format(self, pil, tmp_path, fmt, ext):
        provider, _ = make_provider(
            pil, tmp_path, images=FakeImages(pil, b64=png_b64(pil, fmt=fmt))
        )
        assert provider.generate_preview(make_request()).output_reference.endswith(ext)

    def test_output_filename_cannot_traverse(self, pil, tmp_path):
        provider, _ = make_provider(pil, tmp_path)
        req = make_request(confirmed=make_confirmed(outfit_id="../../escape"))
        path = Path(provider.generate_preview(req).output_reference)
        assert path.parent == tmp_path / "previews"
        assert re.fullmatch(r"[0-9a-f]{32}\.png", path.name)
        assert not (tmp_path / "escape").exists()

    @pytest.mark.parametrize(
        "kind", ["empty_data", "no_b64", "bad_base64", "not_an_image", "unsupported_format"]
    )
    def test_invalid_output_rejected_and_not_kept(self, pil, tmp_path, kind):
        fake = {
            "empty_data": FakeImages(pil, data=[]),
            "no_b64": FakeImages(pil, data=[SimpleNamespace(b64_json=None)]),
            "bad_base64": FakeImages(pil, b64="***not base64***"),
            "not_an_image": FakeImages(pil, b64=base64.b64encode(b"hello").decode()),
            "unsupported_format": FakeImages(pil, b64=png_b64(pil, fmt="GIF")),
        }[kind]
        provider, _ = make_provider(pil, tmp_path, images=fake)
        with pytest.raises(PreviewOutputError):
            provider.generate_preview(make_request())
        out_dir = tmp_path / "previews"
        assert not out_dir.exists() or list(out_dir.iterdir()) == []

    def test_end_to_end_fallback_on_api_error(self, pil, tmp_path):
        fake = FakeImages(pil, error=FakeAPIError(SECRET_KEY))
        provider, _ = make_provider(pil, tmp_path, images=fake)
        result = generate_preview_with_fallback(make_request(PreviewMode.FLAT_LAY), provider)
        assert result.status is PreviewStatus.FALLBACK
        assert SECRET_KEY not in result.model_dump_json()


class TestPortraitPrivacyAndCleanup:
    def portrait_request(self, pil, tmp_path):
        portrait = write_png(pil, tmp_path / "user" / "portrait.png", (30, 40))
        req = make_request(
            PreviewMode.PORTRAIT,
            confirmed=make_confirmed(local_images(pil, tmp_path)),
            user_portrait=str(portrait),
            portrait_consent=True,
        )
        return req, portrait

    def test_temp_files_removed_after_success(self, pil, tmp_path):
        req, _ = self.portrait_request(pil, tmp_path)
        provider, _ = make_provider(pil, tmp_path)
        provider.generate_preview(req)
        assert list((tmp_path / "work").iterdir()) == []

    @pytest.mark.parametrize("fake_kind", ["api_error"])
    def test_temp_files_removed_after_failure(self, pil, tmp_path, fake_kind):
        req, _ = self.portrait_request(pil, tmp_path)
        fake = (
            FakeImages(pil, error=FakeAPIError("x"))
            if fake_kind == "api_error"
            else FakeImages(pil, b64="***")
        )
        provider, _ = make_provider(pil, tmp_path, images=fake)
        with pytest.raises(PreviewProviderError):
            provider.generate_preview(req)
        assert list((tmp_path / "work").iterdir()) == []

    def test_portrait_metadata_not_forwarded(self, pil, tmp_path):
        portrait = tmp_path / "user" / "portrait.jpg"
        portrait.parent.mkdir(parents=True)
        exif = pil.Exif()
        exif[0x010E] = "private description"
        pil.new("RGB", (30, 40)).save(portrait, format="JPEG", exif=exif)
        req = make_request(PreviewMode.PORTRAIT, user_portrait=str(portrait), portrait_consent=True)
        seen = {}

        class InspectingImages(FakeImages):
            def edit(self, **kwargs):
                first = kwargs["image"][0] if isinstance(kwargs["image"], list) else kwargs["image"]
                with self.pil.open(first) as img:
                    seen["exif"] = dict(img.getexif())
                    seen["format"] = img.format
                return super().edit(**kwargs)

        provider, _ = make_provider(pil, tmp_path, images=InspectingImages(pil, b64=png_b64(pil)))
        provider.generate_preview(req)
        assert seen == {"exif": {}, "format": "PNG"}

    def test_requires_preview_request(self, pil, tmp_path):
        provider, _ = make_provider(pil, tmp_path)
        with pytest.raises(TypeError, match="PreviewRequest"):
            provider.generate_preview(make_request().model_dump())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FULL_ENV = {
    "PREVIEW_PROVIDER": "openai",
    "PREVIEW_API_KEY": SECRET_KEY,
    "PREVIEW_MODEL": "configured-image-model",
}


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_module_imports_without_sdk_pillow_key_or_network():
    code = """
import os, socket, sys
for v in ("PREVIEW_PROVIDER", "PREVIEW_API_KEY", "PREVIEW_MODEL"):
    os.environ.pop(v, None)
sys.modules["openai"] = None
sys.modules["requests"] = None
sys.modules["PIL"] = None
sys.modules["PIL.Image"] = None
sys.modules["PIL.ImageOps"] = None
def _blocked(*a, **k):
    raise RuntimeError("network access attempted")
socket.socket = _blocked
socket.create_connection = _blocked
from travel_agent.recommendation.student5 import preview_adapter as pa
assert "openai" not in sys.modules or sys.modules["openai"] is None
print("ok")
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PREVIEW_")}
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
