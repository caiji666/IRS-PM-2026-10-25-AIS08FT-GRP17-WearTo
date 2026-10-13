"""Tests for the Stage 14 Generative AI Outfit Preview (Module 8, preview_adapter).

No test here touches the network or needs credentials: the OpenAI adapter is exercised with a
fake client. Tests that create or inspect images need Pillow (part of the `preview` extra) and
are skipped without it.
"""

import base64
import io
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import travel_agent.recommendation.student5.preview_adapter as preview_adapter
from travel_agent.recommendation.student5.models import (
    ConfirmedOutfit,
    PreviewMode,
    PreviewRequest,
    PreviewResult,
    PreviewStatus,
    ProductCategory,
)
from travel_agent.recommendation.student5.preview_adapter import (
    CATEGORY_ORDER,
    PREVIEW_DISCLAIMER,
    PRODUCT_CARDS_PROVIDER_NAME,
    MockPreviewProvider,
    OpenAIImagePreviewProvider,
    PreviewConfigurationError,
    PreviewInputError,
    PreviewOutputError,
    PreviewProviderError,
    build_preview_prompt,
    build_preview_request,
    build_product_cards_result,
    generate_preview_with_fallback,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECRET_KEY = "test-key-not-a-real-credential"
GENERATIVE_MODES = [PreviewMode.MANNEQUIN, PreviewMode.FLAT_LAY]


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
    def test_selected_outfit_id_kept(self):
        req = make_request(confirmed=make_confirmed(outfit_id="cand-7f3a"))
        assert isinstance(req, PreviewRequest)
        assert req.selected_outfit_id == "cand-7f3a"

    def test_canonical_category_order(self):
        req = make_request()
        assert [i.category for i in req.items] == [
            ProductCategory.TOP,
            ProductCategory.BOTTOM,
            ProductCategory.FOOTWEAR,
            ProductCategory.OUTERWEAR,
        ]
        assert CATEGORY_ORDER == tuple(i.category for i in req.items)

    def test_existing_and_purchase_items_included(self):
        req = make_request()
        owned = {i.product_id: i.is_user_owned for i in req.items}
        assert owned == {
            "p_top": False,
            "p_bottom": True,
            "p_footwear": False,
            "p_outerwear": False,
        }

    def test_field_mapping(self):
        confirmed = make_confirmed()
        req = build_preview_request(confirmed, PreviewMode.MANNEQUIN)
        products = {
            p.product_id: p
            for p in (
                confirmed.recommendation.existing_items + confirmed.recommendation.purchase_items
            )
        }
        for item in req.items:
            p = products[item.product_id]
            assert (
                item.category,
                item.image_reference,
                item.title,
                item.description,
                item.is_user_owned,
            ) == (p.category, p.image_url, p.title, p.description, p.is_user_owned)
        assert req.preview_mode is PreviewMode.MANNEQUIN

    def test_mode_accepted_as_string(self):
        assert make_request("PRODUCT_CARDS").preview_mode is PreviewMode.PRODUCT_CARDS

    def test_portrait_without_portrait_rejected(self):
        with pytest.raises(ValidationError, match="user_portrait"):
            make_request(PreviewMode.PORTRAIT, portrait_consent=True)

    def test_portrait_without_consent_rejected(self):
        with pytest.raises(ValidationError, match="portrait_consent"):
            make_request(PreviewMode.PORTRAIT, user_portrait="tmp/uploads/me.jpg")

    def test_portrait_with_consent(self):
        req = make_request(
            PreviewMode.PORTRAIT, user_portrait="tmp/uploads/me.jpg", portrait_consent=True
        )
        assert req.user_portrait == "tmp/uploads/me.jpg"

    def test_confirmed_outfit_not_modified(self):
        confirmed = make_confirmed()
        snapshot = confirmed.model_dump()
        req = build_preview_request(confirmed, PreviewMode.FLAT_LAY)
        assert confirmed.model_dump() == snapshot
        assert req.items[0] is not confirmed.recommendation.purchase_items[2]

    def test_requires_confirmed_outfit(self):
        with pytest.raises(TypeError, match="ConfirmedOutfit"):
            build_preview_request(make_confirmed().model_dump(), PreviewMode.FLAT_LAY)


# ---------------------------------------------------------------------------
# build_preview_prompt
# ---------------------------------------------------------------------------


class TestBuildPreviewPrompt:
    def test_deterministic(self):
        assert build_preview_prompt(make_request()) == build_preview_prompt(make_request())

    @pytest.mark.parametrize("mode", ["PORTRAIT", "MANNEQUIN", "FLAT_LAY"])
    def test_states_mode(self, mode):
        kw = {"user_portrait": "me.jpg", "portrait_consent": True} if mode == "PORTRAIT" else {}
        assert f"Mode: {mode}" in build_preview_prompt(make_request(mode, **kw))

    def test_items_in_request_order_with_description_or_title(self):
        prompt = build_preview_prompt(make_request())
        lines = [line for line in prompt.splitlines() if re.match(r"^\d\. ", line)]
        assert lines == [
            "1. TOP: Cream cable-knit sweater",
            "2. BOTTOM: Mock bottom",
            "3. FOOTWEAR: Mock footwear",
            "4. OUTERWEAR: Charcoal wool overcoat",
        ]

    def test_does_not_invent_or_leak_details(self):
        prompt = build_preview_prompt(make_request()).lower()
        for absent in (
            "secretbrand",
            "https://",
            "img.example.com",
            "shop.example.com",
            "sgd",
            "120.00",
            "p_top",
        ):
            assert absent not in prompt, absent

    def test_images_are_only_visual_references(self):
        prompt = build_preview_prompt(make_request())
        assert "visual reference" in prompt
        assert "not a virtual try-on" in prompt

    def test_portrait_instructions(self):
        prompt = build_preview_prompt(
            make_request(PreviewMode.PORTRAIT, user_portrait="me.jpg", portrait_consent=True)
        )
        assert "first reference image is the user's portrait" in prompt
        assert "identity, face, pose and background" in prompt
        assert "do not claim" in prompt.lower() and "fit" in prompt
        assert "me.jpg" not in prompt

    def test_mannequin_and_flat_lay_instructions(self):
        assert "mannequin" in build_preview_prompt(make_request(PreviewMode.MANNEQUIN))
        flat = build_preview_prompt(make_request(PreviewMode.FLAT_LAY))
        assert "flat-lay" in flat and "no person" in flat

    def test_product_cards_fixed_non_generative_text(self):
        text = build_preview_prompt(make_request(PreviewMode.PRODUCT_CARDS))
        assert "no generative image" in text.lower()
        assert text == build_preview_prompt(
            make_request(PreviewMode.PRODUCT_CARDS, confirmed=make_confirmed(outfit_id="x"))
        )

    def test_request_not_modified(self):
        req = make_request()
        snapshot = req.model_dump()
        build_preview_prompt(req)
        assert req.model_dump() == snapshot

    def test_requires_preview_request(self):
        with pytest.raises(TypeError, match="PreviewRequest"):
            build_preview_prompt(make_request().model_dump())


# ---------------------------------------------------------------------------
# Product cards and mock provider
# ---------------------------------------------------------------------------


class TestProductCards:
    def test_product_cards_result(self):
        result = build_product_cards_result(make_request(PreviewMode.PRODUCT_CARDS))
        assert isinstance(result, PreviewResult)
        assert (result.status, result.delivered_mode, result.output_reference) == (
            PreviewStatus.SUCCESS,
            PreviewMode.PRODUCT_CARDS,
            None,
        )
        assert result.provider_name == PRODUCT_CARDS_PROVIDER_NAME
        assert result.disclaimer == PREVIEW_DISCLAIMER

    def test_disclaimer_content(self):
        text = PREVIEW_DISCLAIMER.lower()
        for phrase in (
            "visual reference",
            "not a virtual try-on",
            "colour",
            "texture",
            "detail",
            "body proportions",
            "size",
            "fit",
        ):
            assert phrase in text, phrase


class TestMockPreviewProvider:
    @pytest.mark.parametrize("mode", GENERATIVE_MODES)
    def test_generative_modes(self, mode):
        provider = MockPreviewProvider()
        result = provider.generate_preview(make_request(mode))
        assert (result.status, result.requested_mode, result.delivered_mode) == (
            PreviewStatus.SUCCESS,
            mode,
            mode,
        )
        assert result.output_reference == f"mock://preview/outfit_202/{mode.value.lower()}"
        assert result.provider_name == "mock" and result.model_name is None

    def test_portrait_mode(self):
        req = make_request(PreviewMode.PORTRAIT, user_portrait="me.jpg", portrait_consent=True)
        result = MockPreviewProvider().generate_preview(req)
        assert result.delivered_mode is PreviewMode.PORTRAIT
        assert result.output_reference == "mock://preview/outfit_202/portrait"
        assert "me.jpg" not in result.model_dump_json()

    def test_product_cards_mode(self):
        result = MockPreviewProvider().generate_preview(make_request(PreviewMode.PRODUCT_CARDS))
        assert (result.status, result.delivered_mode, result.output_reference) == (
            PreviewStatus.SUCCESS,
            PreviewMode.PRODUCT_CARDS,
            None,
        )

    def test_deterministic(self):
        req = make_request(PreviewMode.MANNEQUIN)
        a = MockPreviewProvider().generate_preview(req)
        b = MockPreviewProvider().generate_preview(req)
        assert a == b

    def test_injected_failure(self):
        provider = MockPreviewProvider(failure=PreviewProviderError("simulated outage"))
        with pytest.raises(PreviewProviderError):
            provider.generate_preview(make_request(PreviewMode.FLAT_LAY))
        assert provider.calls == [PreviewMode.FLAT_LAY]

    def test_injected_failure_does_not_affect_product_cards(self):
        provider = MockPreviewProvider(failure=PreviewProviderError("x"))
        result = provider.generate_preview(make_request(PreviewMode.PRODUCT_CARDS))
        assert result.status is PreviewStatus.SUCCESS


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
    def test_product_cards_never_calls_provider(self):
        provider = RecordingProvider(AssertionError("must not be called"))
        result = generate_preview_with_fallback(make_request(PreviewMode.PRODUCT_CARDS), provider)
        assert result.status is PreviewStatus.SUCCESS
        assert result.delivered_mode is PreviewMode.PRODUCT_CARDS
        assert provider.calls == 0

    def test_provider_success_returned(self):
        result = generate_preview_with_fallback(make_request(), MockPreviewProvider())
        assert result.status is PreviewStatus.SUCCESS
        assert result.delivered_mode is PreviewMode.FLAT_LAY

    def test_missing_provider_falls_back(self):
        result = generate_preview_with_fallback(make_request(PreviewMode.MANNEQUIN), None)
        assert result.status is PreviewStatus.FALLBACK
        assert result.requested_mode is PreviewMode.MANNEQUIN
        assert result.delivered_mode is PreviewMode.PRODUCT_CARDS
        assert result.output_reference is None
        assert result.disclaimer == PREVIEW_DISCLAIMER
        assert "not configured" in result.warnings[0]

    @pytest.mark.parametrize(
        "error,reason",
        [
            (PreviewConfigurationError("x"), "not configured correctly"),
            (PreviewInputError("x"), "input images could not be used"),
            (PreviewOutputError("x"), "output could not be validated"),
            (PreviewProviderError("x"), "request failed"),
        ],
    )
    def test_provider_errors_fall_back(self, error, reason):
        result = generate_preview_with_fallback(make_request(), RecordingProvider(error))
        assert result.status is PreviewStatus.FALLBACK
        assert result.delivered_mode is PreviewMode.PRODUCT_CARDS
        assert reason in result.warnings[0]

    def test_fallback_does_not_leak_sensitive_data(self, caplog):
        caplog.set_level(logging.DEBUG)
        secret = f"key={SECRET_KEY} portrait=C:/Users/alice/portrait.jpg b64=iVBORw0KGgoAAAANSUhEUg"
        req = make_request(
            PreviewMode.PORTRAIT, user_portrait="C:/Users/alice/portrait.jpg", portrait_consent=True
        )
        result = generate_preview_with_fallback(
            req, RecordingProvider(PreviewProviderError(secret))
        )
        dumped = result.model_dump_json()
        for leaked in (SECRET_KEY, "alice", "portrait.jpg", "iVBORw0KGgo"):
            assert leaked not in dumped
            assert leaked not in caplog.text

    def test_programming_errors_not_swallowed(self):
        with pytest.raises(ZeroDivisionError):
            generate_preview_with_fallback(make_request(), RecordingProvider(ZeroDivisionError()))

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
                requested_mode="MANNEQUIN",
                delivered_mode="MANNEQUIN",
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
        assert result.delivered_mode is PreviewMode.PRODUCT_CARDS

    def test_requires_preview_request(self):
        with pytest.raises(TypeError, match="PreviewRequest"):
            generate_preview_with_fallback(make_request().model_dump(), MockPreviewProvider())

    def test_request_not_modified(self):
        req = make_request()
        snapshot = req.model_dump()
        generate_preview_with_fallback(req, RecordingProvider(PreviewProviderError("x")))
        assert req.model_dump() == snapshot


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
        assert result.delivered_mode is PreviewMode.PORTRAIT

    @pytest.mark.parametrize("mode", GENERATIVE_MODES)
    def test_edit_path_with_product_images(self, pil, tmp_path, mode):
        req = make_request(mode, confirmed=make_confirmed(local_images(pil, tmp_path)))
        provider, fake = make_provider(pil, tmp_path)
        provider.generate_preview(req)
        [(kind, kwargs, sizes, _)] = fake.calls
        assert kind == "edit"
        assert sizes == [(11, 11), (12, 12), (13, 13), (14, 14)]
        assert kwargs["prompt"].startswith(build_preview_prompt(req))
        assert reference_lines(kwargs["prompt"])[0] == (
            "Reference image 1 corresponds to outfit item 1: TOP."
        )

    def test_single_image_passed_as_file(self, pil, tmp_path):
        images = {"TOP": str(write_png(pil, tmp_path / "in" / "top.png", (11, 11)))}
        req = make_request(PreviewMode.MANNEQUIN, confirmed=make_confirmed(images))
        provider, fake = make_provider(pil, tmp_path)
        provider.generate_preview(req)
        [(kind, kwargs, sizes, _)] = fake.calls
        assert not isinstance(kwargs["image"], list)
        assert sizes == [(11, 11)]
        assert any(
            "not permitted by the remote image policy" in w
            for w in provider.generate_preview(req).warnings
        )

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

    def test_no_image_reference_at_all(self, pil, tmp_path):
        confirmed = make_confirmed()
        data = confirmed.model_dump()
        for item in (
            data["recommendation"]["existing_items"] + data["recommendation"]["purchase_items"]
        ):
            item["image_url"] = None
        req = make_request(PreviewMode.FLAT_LAY, confirmed=ConfirmedOutfit.model_validate(data))
        provider, fake = make_provider(pil, tmp_path)
        result = provider.generate_preview(req)
        assert fake.calls[0][0] == "generate"
        assert not any("product image" in w for w in result.warnings)

    def test_portrait_not_sent_in_other_modes(self, pil, tmp_path):
        portrait = write_png(pil, tmp_path / "in" / "portrait.png", (30, 40))
        req = make_request(
            PreviewMode.MANNEQUIN,
            confirmed=make_confirmed(local_images(pil, tmp_path)),
            user_portrait=str(portrait),
            portrait_consent=True,
        )
        provider, fake = make_provider(pil, tmp_path)
        provider.generate_preview(req)
        assert (30, 40) not in fake.calls[0][2]

    def test_product_cards_makes_no_api_call(self, pil, tmp_path):
        provider, fake = make_provider(pil, tmp_path)
        result = provider.generate_preview(make_request(PreviewMode.PRODUCT_CARDS))
        assert fake.calls == []
        assert result.status is PreviewStatus.SUCCESS
        assert result.output_reference is None

    def test_unreadable_local_product_image_skipped(self, pil, tmp_path):
        bad = tmp_path / "in" / "not_an_image.png"
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"definitely not an image")
        images = {
            "TOP": str(bad),
            "BOTTOM": str(tmp_path / "missing.png"),
            "FOOTWEAR": str(write_png(pil, tmp_path / "in" / "f.png", (13, 13))),
        }
        req = make_request(PreviewMode.FLAT_LAY, confirmed=make_confirmed(images))
        provider, fake = make_provider(pil, tmp_path)
        result = provider.generate_preview(req)
        assert fake.calls[0][2] == [(13, 13)]
        assert any(
            "2 product image(s) could not be read as valid images" in w for w in result.warnings
        )
        assert not any(str(tmp_path) in w for w in result.warnings)

    def test_oversized_input_skipped(self, pil, tmp_path):
        images = {"TOP": str(write_png(pil, tmp_path / "in" / "big.png", (200, 200)))}
        req = make_request(PreviewMode.FLAT_LAY, confirmed=make_confirmed(images))
        provider, fake = make_provider(pil, tmp_path, max_input_bytes=10)
        provider.generate_preview(req)
        assert fake.calls[0][0] == "generate"


class TestOpenAIProviderOutput:
    def test_base64_saved_as_readable_image(self, pil, tmp_path):
        provider, _ = make_provider(pil, tmp_path)
        result = provider.generate_preview(make_request())
        path = Path(result.output_reference)
        assert path.is_file()
        assert path.parent == tmp_path / "previews"
        with pil.open(path) as img:
            img.load()
            assert img.size == (16, 16)

    @pytest.mark.parametrize("fmt,ext", [("JPEG", ".jpg"), ("WEBP", ".webp")])
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

    def test_provider_and_model_recorded(self, pil, tmp_path):
        provider, _ = make_provider(pil, tmp_path)
        result = provider.generate_preview(make_request(PreviewMode.MANNEQUIN))
        assert result.provider_name == "openai"
        assert result.model_name == "configured-image-model"
        assert result.disclaimer == PREVIEW_DISCLAIMER
        assert result.generation_time_seconds is not None and result.generation_time_seconds >= 0

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

    def test_api_error_wrapped_without_details(self, pil, tmp_path):
        fake = FakeImages(pil, error=FakeAPIError(f"auth failed for {SECRET_KEY}"))
        provider, _ = make_provider(pil, tmp_path, images=fake)
        with pytest.raises(PreviewProviderError) as exc:
            provider.generate_preview(make_request())
        assert SECRET_KEY not in str(exc.value)
        assert exc.value.__cause__ is None and exc.value.__suppress_context__

    def test_unexpected_client_error_not_swallowed(self, pil, tmp_path):
        provider, _ = make_provider(pil, tmp_path, images=FakeImages(pil, error=KeyError("bug")))
        with pytest.raises(KeyError):
            provider.generate_preview(make_request())

    def test_end_to_end_fallback_on_api_error(self, pil, tmp_path):
        fake = FakeImages(pil, error=FakeAPIError(SECRET_KEY))
        provider, _ = make_provider(pil, tmp_path, images=fake)
        result = generate_preview_with_fallback(make_request(PreviewMode.MANNEQUIN), provider)
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

    @pytest.mark.parametrize("fake_kind", ["api_error", "bad_output"])
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

    def test_temp_files_removed_after_unexpected_error(self, pil, tmp_path):
        req, _ = self.portrait_request(pil, tmp_path)
        provider, _ = make_provider(pil, tmp_path, images=FakeImages(pil, error=KeyError("bug")))
        with pytest.raises(KeyError):
            provider.generate_preview(req)
        assert list((tmp_path / "work").iterdir()) == []

    def test_original_portrait_and_product_images_kept(self, pil, tmp_path):
        req, portrait = self.portrait_request(pil, tmp_path)
        before = {p: p.read_bytes() for p in [portrait, *sorted((tmp_path / "in").iterdir())]}
        provider, _ = make_provider(pil, tmp_path)
        provider.generate_preview(req)
        assert {p: p.read_bytes() for p in before} == before

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

    @pytest.mark.parametrize("portrait_ref", ["https://example.com/me.jpg", "missing.jpg"])
    def test_unusable_portrait_raises_input_error_without_path(self, pil, tmp_path, portrait_ref):
        req = make_request(PreviewMode.PORTRAIT, user_portrait=portrait_ref, portrait_consent=True)
        provider, fake = make_provider(pil, tmp_path)
        with pytest.raises(PreviewInputError) as exc:
            provider.generate_preview(req)
        assert portrait_ref not in str(exc.value)
        assert fake.calls == []
        assert list((tmp_path / "work").iterdir()) == []

    def test_request_not_modified(self, pil, tmp_path):
        req, _ = self.portrait_request(pil, tmp_path)
        snapshot = req.model_dump()
        provider, _ = make_provider(pil, tmp_path)
        provider.generate_preview(req)
        assert req.model_dump() == snapshot

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


class TestOpenAIProviderConfiguration:
    @pytest.mark.parametrize("missing", ["PREVIEW_PROVIDER", "PREVIEW_API_KEY", "PREVIEW_MODEL"])
    def test_from_env_requires_all_variables(self, pil, missing):
        env = {k: v for k, v in FULL_ENV.items() if k != missing}
        with pytest.raises(PreviewConfigurationError, match=missing) as exc:
            OpenAIImagePreviewProvider.from_env(env)
        assert SECRET_KEY not in str(exc.value)

    def test_from_env_rejects_other_provider(self, pil):
        with pytest.raises(PreviewConfigurationError, match="PREVIEW_PROVIDER"):
            OpenAIImagePreviewProvider.from_env({**FULL_ENV, "PREVIEW_PROVIDER": "other"})

    def test_from_env_blank_values_rejected(self, pil):
        with pytest.raises(PreviewConfigurationError, match="PREVIEW_MODEL"):
            OpenAIImagePreviewProvider.from_env({**FULL_ENV, "PREVIEW_MODEL": "  "})

    def test_from_env_builds_real_client_without_network(self, pil):
        openai = pytest.importorskip("openai")
        provider = OpenAIImagePreviewProvider.from_env(FULL_ENV)
        assert provider.model == "configured-image-model"
        assert isinstance(provider._client, openai.OpenAI)
        assert provider._api_error_types == (openai.OpenAIError,)
        assert SECRET_KEY not in repr(provider)

    def test_from_env_reads_os_environ_by_default(self, pil, monkeypatch):
        pytest.importorskip("openai")
        for name, value in FULL_ENV.items():
            monkeypatch.setenv(name, value)
        assert OpenAIImagePreviewProvider.from_env().model == "configured-image-model"

    def test_model_required(self, pil):
        with pytest.raises(PreviewConfigurationError, match="model"):
            OpenAIImagePreviewProvider(model=" ", client=SimpleNamespace(images=None))

    def test_api_key_required_without_client(self, pil):
        with pytest.raises(PreviewConfigurationError, match="API key"):
            OpenAIImagePreviewProvider(model="m")

    def test_missing_openai_sdk_reported_clearly(self, pil, monkeypatch):
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(PreviewConfigurationError, match="preview"):
            OpenAIImagePreviewProvider(model="m", api_key=SECRET_KEY)

    def test_missing_pillow_reported_clearly(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "PIL", None)
        monkeypatch.setitem(sys.modules, "PIL.Image", None)
        with pytest.raises(PreviewConfigurationError, match="Pillow"):
            OpenAIImagePreviewProvider(model="m", client=SimpleNamespace(images=None))

    def test_default_error_types_without_sdk(self, pil, monkeypatch):
        monkeypatch.setitem(sys.modules, "openai", None)
        provider = OpenAIImagePreviewProvider(model="m", client=SimpleNamespace(images=None))
        assert provider._api_error_types == ()

    def test_sdk_error_types_used_by_default(self, pil, tmp_path):
        openai = pytest.importorskip("openai")

        class SdkError(openai.OpenAIError):
            pass

        provider = OpenAIImagePreviewProvider(
            model="m",
            client=SimpleNamespace(images=FakeImages(pil, error=SdkError("boom"))),
            output_dir=tmp_path / "out",
        )
        with pytest.raises(PreviewProviderError):
            provider.generate_preview(make_request())


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


def test_module_does_not_import_sdk_at_top_level():
    assert "openai" not in vars(preview_adapter)
    assert "Image" not in vars(preview_adapter)
