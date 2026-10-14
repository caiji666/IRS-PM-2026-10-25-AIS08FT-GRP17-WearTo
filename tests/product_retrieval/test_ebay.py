import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from travel_agent.products.retrieval.ebay.client import BrowseClient, EbayError, download_photo
from travel_agent.products.retrieval.ebay.pipeline import category_names, map_product, run
from travel_agent.products.retrieval.m5 import vector
from travel_agent.products.retrieval.m5.encoder import MODEL_IDENTITY


class Response:
    def __init__(self, status=200, body=None, headers=None, data=b""):
        self.status_code = status
        self.body = body
        self.headers = headers or {}
        self.data = data

    def json(self):
        return self.body

    def close(self):
        pass

    def iter_content(self, size):
        yield self.data


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)


CREDS = {"environment": "sandbox", "client_id": "test-SBX-app", "client_secret": "test-secret"}


def test_oauth_memory_cache_and_image_search_wire():
    s = Session(
        [
            Response(body={"access_token": "test-token", "expires_in": 7200}),
            Response(body={"itemSummaries": []}),
        ]
    )
    c = BrowseClient(CREDS, session=s)
    c.authenticate()
    c.authenticate()
    c.search_image(b"jpg", 5)
    assert (
        len(s.calls) == 2
        and s.calls[0][1] == "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
    )
    assert s.calls[1][0] == "POST" and s.calls[1][2]["json"] == {"image": "anBn"}
    assert s.calls[1][2]["headers"]["X-EBAY-C-MARKETPLACE-ID"] == "EBAY_US"


def test_error_never_leaks_secret_or_token():
    c = BrowseClient(
        CREDS,
        session=Session(
            [Response(401, {"error": "invalid_client", "error_description": "test-secret"})]
        ),
    )
    with pytest.raises(EbayError) as err:
        c.authenticate()
    assert "test-secret" not in str(err.value) and "401" in str(err.value)


@pytest.mark.parametrize(
    "env,app,market",
    [
        ("sandbox", "x-PRD-a", "EBAY_US"),
        ("production", "x-SBX-a", "EBAY_US"),
        ("sandbox", "x-SBX-a", "EBAY_SG"),
    ],
)
def test_environment_and_market_restrictions(env, app, market):
    with pytest.raises(ValueError):
        BrowseClient(dict(CREDS, environment=env, client_id=app), market)


def test_image_host_and_redirect_validation(tmp_path):
    with pytest.raises(ValueError):
        download_photo("http://images.unsplash.com/a", tmp_path / "x.jpg", {"images.unsplash.com"})
    s = Session([Response(302, headers={"Location": "https://127.0.0.1/x"})])
    with pytest.raises(ValueError):
        download_photo(
            "https://images.unsplash.com/a", tmp_path / "x.jpg", {"images.unsplash.com"}, s
        )
    assert len(s.calls) == 1


def test_api_normalization_provenance_stock_and_currency(tmp_path):
    row = {
        "itemId": "v1|1|0",
        "title": "Black jacket",
        "categories": [{"categoryName": "Coats, Jackets & Vests"}],
        "price": {"value": "20.25", "currency": "USD"},
        "estimatedAvailabilities": [{"estimatedAvailabilityStatus": "IN_STOCK"}],
        "itemWebUrl": "https://www.ebay.com/itm/1",
        "localizedAspects": [{"name": "Color", "value": "Black"}],
    }
    p = map_product(row, "sandbox", "EBAY_US", "2026-10-12", tmp_path / "image.jpg")
    assert p.category.value == "OUTERWEAR" and p.availability.value == "UNKNOWN"
    assert p.price.is_synthetic and not p.evidence.price_verified and p.colors == ["black"]
    assert p.provenance.original["api_item"] == row and p.image_embedding_candidate
    live = map_product(row, "production", "EBAY_US", "2026-10-12")
    assert not live.price.is_synthetic and live.evidence.price_verified
    assert category_names({"categories": [{"categoryName": "Accessories"}]}) == "UNKNOWN"


class Encoder:
    identity = MODEL_IDENTITY
    dimension = 512

    def encode_texts(self, texts):
        return [vector(self.identity, "text", [1] + [0] * 511) for _ in texts]

    def encode_images(self, images):
        return [vector(self.identity, "image", [1] + [0] * 511) for _ in images]


class Client:
    environment = "sandbox"
    marketplace = "EBAY_US"

    def authenticate(self):
        pass

    def search_image(self, image, limit, category_ids):
        return {"itemSummaries": [self.item("v1|1|0"), self.item("v1|1|0")]}

    def item(self, item):
        return {
            "itemId": item,
            "title": "shirt",
            "categories": [{"categoryName": "Shirts"}],
            "image": {"imageUrl": "https://i.ebayimg.com/x.jpg"},
            "price": {"value": "12", "currency": "USD"},
        }


def inputs(tmp_path):
    pt = tmp_path / "input.pt"
    text = tmp_path / "text.txt"
    context = tmp_path / "context.json"
    t = torch.zeros(1, 512)
    t[0, 0] = 1
    torch.save(
        {
            "context_embedding": t,
            "top_image_embeddings": t,
            "top_images": [
                {
                    "image_id": "a",
                    "image_url": "https://images.unsplash.com/a.jpg",
                    "description": "shirt",
                }
            ],
        },
        pt,
    )
    text.write_text("winter shirt")
    context.write_text(
        json.dumps(
            {
                "destination": "Seoul",
                "season": "winter",
                "weather_clothing_requirements": {"top": ["sweater"]},
                "activity_style_requirements": {"walk": ["casual"]},
                "user_style": ["minimalist"],
            }
        )
    )
    return pt, text, context


def photo(url, path, hosts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), "red").save(path)
    return path.read_bytes()


def test_full_mock_pipeline_dedup_image_index_and_owned_separation(tmp_path):
    result = run(
        Client(),
        Encoder(),
        *inputs(tmp_path),
        tmp_path / "out",
        fetch_photo=photo,
        wardrobe_store=tmp_path / "no-store",
    )
    assert result["status"] == "completed" and result["summary"]["products_indexed"] == 1
    assert result["index"]["items"][0]["vector"]["modality"] == "image"
    assert (
        result["bundle"]["owned_items"] == []
        and len(result["bundle"]["top_k_products"]["TOP"]) == 1
    )
    assert result["upstream_metadata"]["status"] == "pending_model_identity"
    assert (
        result["request_used"]["top_social_references"][0]["embedding"]["model"] == MODEL_IDENTITY
    )


def test_imported_unknown_model_blocks_before_network(tmp_path):
    class Reject(Client):
        def authenticate(self):
            raise AssertionError("must validate before auth")

    with pytest.raises(ValueError, match="confirmed compatible"):
        run(Reject(), Encoder(), *inputs(tmp_path), tmp_path / "out", reference_mode="imported")


def test_no_search_results_reported_not_faked(tmp_path):
    class Empty(Client):
        def search_image(self, *args):
            return {"itemSummaries": []}

    result = run(
        Empty(),
        Encoder(),
        *inputs(tmp_path),
        tmp_path / "out",
        fetch_photo=photo,
        wardrobe_store=tmp_path / "none",
    )
    assert result["status"] == "no_usable_api_products" and not result["index"]["items"]


@pytest.mark.skipif(
    __import__("os").environ.get("ITSA_REAL_MODEL_TEST") != "1", reason="requires local FashionCLIP"
)
def test_real_model_mock_api_end_to_end(tmp_path):
    from travel_agent.products.retrieval.m5 import FashionCLIPEncoder

    root = Path(
        __import__("os").environ.get("STUDENT3_DATA_ROOT", str(Path(__file__).resolve().parents[2]))
    )
    encoder = FashionCLIPEncoder(root / ".cache/huggingface", offline=True)

    def historical_photo(url, path, hosts):
        # Explicit test transport double: historical photos, not a live eBay response.
        import shutil

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "data/photos20/images/15970.jpg", path)
        return path.read_bytes()

    result = run(
        Client(),
        encoder,
        *inputs(tmp_path),
        tmp_path / "out",
        fetch_photo=historical_photo,
        wardrobe_store=tmp_path / "none",
    )
    assert result["summary"]["image_vectors"] == 1
    assert len(result["bundle"]["top_k_products"]["TOP"]) == 1
    assert len(result["index"]["items"][0]["vector"]["values"]) == 512


def test_optional_student1_context_override_preserves_social_pairing(tmp_path):
    from travel_agent.products.retrieval.schemas import (
        ExpandedFashionContext,
        ProductRetrievalRequest,
        Trip,
    )

    request = ProductRetrievalRequest(
        context=ExpandedFashionContext(
            trip=Trip(destination="Paris"), query="Paris summer smart casual", activities=["dinner"]
        )
    )
    result = run(
        Client(),
        Encoder(),
        *inputs(tmp_path),
        tmp_path / "override",
        fetch_photo=photo,
        wardrobe_store=tmp_path / "none",
        request_override=request.model_dump(mode="json"),
    )
    assert result["request_used"]["context"]["trip"]["destination"] == "Paris"
    assert result["request_used"]["top_social_references"][0]["reference_id"] == "a"
    assert result["upstream_metadata"]["original_context"]["destination"] == "Seoul"
    assert result["upstream_metadata"]["request_override"]["context"]["activities"] == ["dinner"]


def test_keyword_wire_and_validation():
    session = Session(
        [
            Response(body={"access_token": "test-token", "expires_in": 7200}),
            Response(body={"itemSummaries": []}),
        ]
    )
    client = BrowseClient(CREDS, session=session)
    client.search_keywords("linen shirt", 20)
    assert session.calls[1][0] == "GET"
    assert session.calls[1][1].endswith("/item_summary/search")
    assert session.calls[1][2]["params"] == {"q": "linen shirt", "limit": 20}
    with pytest.raises(ValueError):
        client.search_keywords(" ")


def test_hybrid_empty_images_keyword_products_and_social_scoring(tmp_path):
    class Hybrid(Client):
        def search_image(self, *args):
            return {"itemSummaries": []}

        def search_keywords(self, query, *args):
            return {"itemSummaries": [self.item("v1|1|0")]}

    result = run(
        Hybrid(),
        Encoder(),
        *inputs(tmp_path),
        tmp_path / "hybrid",
        fetch_photo=photo,
        wardrobe_store=tmp_path / "none",
        search_mode="hybrid",
    )
    assert result["summary"]["unique_candidates"] == 1
    assert result["summary"]["candidates_by_source"] == {"image": 0, "keyword": 1}
    ranked = result["bundle"]["top_k_products"]["TOP"][0]
    assert result["summary"]["image_vectors"] == 1
    assert result["request_used"]["top_social_references"][0]["embedding"] is not None
    scores = result["bundle"]["product_ranking"]["ranked_products"]["TOP"][0]
    assert scores["social_similarity"] == pytest.approx(1.0)
    assert scores["context_weight"] == scores["social_weight"] == 0.5
    assert scores["score"] == pytest.approx(1.0)
    assert ranked is not None
    original = result["index"]["items"][0]["product"]["provenance"]["original"]
    assert original["matched_queries"] and original["retrieval_sources"] == ["keyword"]


def test_hybrid_dedup_and_bounded_empty_fallback(tmp_path):
    class Hybrid(Client):
        def search_keywords(self, *args):
            return {"itemSummaries": [self.item("v1|1|0")]}

    result = run(
        Hybrid(),
        Encoder(),
        *inputs(tmp_path),
        tmp_path / "dedup",
        fetch_photo=photo,
        wardrobe_store=tmp_path / "none",
        search_mode="hybrid",
        max_search_requests=4,
    )
    assert result["summary"]["search_requests"] == 4
    assert result["summary"]["unique_candidates"] == 1
    original = result["index"]["items"][0]["product"]["provenance"]["original"]
    assert original["retrieval_sources"] == ["image", "keyword"]
    assert original["matched_reference_ids"] == ["a"]


def test_hybrid_errors_are_not_empty_responses_and_fallback_is_once(tmp_path):
    class Empty(Client):
        def search_image(self, *args):
            raise EbayError("eBay HTTP 403")

        def search_keywords(self, query, *args):
            return {"itemSummaries": []}

    result = run(
        Empty(),
        Encoder(),
        *inputs(tmp_path),
        tmp_path / "empty",
        fetch_photo=photo,
        wardrobe_store=tmp_path / "none",
        search_mode="hybrid",
    )
    assert result["status"] == "no_usable_api_products"
    assert any("403" in issue["reason"] for issue in result["issues"])
    queries = [s["query"] for s in result["raw_search_responses"] if s["source"] == "keyword"]
    assert len(queries) == len(set(queries))
    assert result["summary"]["search_requests"] <= 17


def test_query_plan_uses_upstream_without_inventing_attributes(tmp_path):
    from travel_agent.products.retrieval.ebay.queries import plan_queries
    from travel_agent.products.retrieval.upstream import adapt_files

    request, _, metadata = adapt_files(*inputs(tmp_path))
    plan = plan_queries(request, metadata)
    assert "sweater" in plan["TOP"]["primary"][0]
    assert len(plan) == 4
    assert all(len(p["primary"]) <= 2 for p in plan.values())
    assert not any("linen" in q for p in plan.values() for q in p["primary"])


def test_sandbox_cdn_upgrade_keeps_https_host_validation(tmp_path):
    class Sandbox(Client):
        def item(self, item):
            row = super().item(item)
            row["image"]["imageUrl"] = "http://i.ebayimg.sandbox.ebay.com/images/x.jpg"
            return row

    def sandbox_photo(url, path, hosts):
        if "sandbox.ebay.com" in url:
            assert url.startswith("https://i.ebayimg.sandbox.ebay.com/")
            assert "i.ebayimg.sandbox.ebay.com" in hosts
        return photo(url, path, hosts)

    result = run(
        Sandbox(),
        Encoder(),
        *inputs(tmp_path),
        tmp_path / "cdn",
        fetch_photo=sandbox_photo,
        wardrobe_store=tmp_path / "none",
    )
    assert result["summary"]["products_indexed"] == 1


def test_production_pipeline_registers_artifacts_and_preserves_sandbox(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    from travel_agent.products.notifications.lifecycle import Registry, save_report

    class Production(Client):
        environment = "production"

        def item(self, item):
            row = super().item(item)
            row["seller"] = {"username": "synthetic-seller"}
            return row

    monkeypatch.delenv("EBAY_MANAGED_ROOT", raising=False)
    with pytest.raises(ValueError, match="EBAY_MANAGED_ROOT"):
        run(Production(), Encoder(), *inputs(tmp_path), tmp_path / "runs/api")
    monkeypatch.setenv("EBAY_MANAGED_ROOT", str(tmp_path))
    result = run(
        Production(),
        Encoder(),
        *inputs(tmp_path),
        tmp_path / "runs/api",
        fetch_photo=photo,
        wardrobe_store=tmp_path / "none",
    )
    save_report(result, tmp_path / "runs/api/result.json")
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    assert registry.db.execute("SELECT count(*) FROM accounts").fetchone()[0] == 1
    registry.cleanup({"event_id": "signed-event", "identifiers": {"username": "synthetic-seller"}})
    assert not (tmp_path / "runs/api").exists()
    registry.close()
    with pytest.raises(RuntimeError, match="previously deleted"):
        run(
            Production(),
            Encoder(),
            *inputs(tmp_path),
            tmp_path / "runs/new",
            fetch_photo=photo,
            wardrobe_store=tmp_path / "none",
        )
