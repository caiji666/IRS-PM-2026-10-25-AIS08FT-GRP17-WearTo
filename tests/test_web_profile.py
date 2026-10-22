"""Profile persistence and upload boundaries, isolated from real user records."""

import http.client
import io
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

pytest.importorskip("PIL")
from PIL import Image, PngImagePlugin

from travel_agent.web import __main__ as web
from travel_agent.web.profile import decode_upload, portrait_view, profile_view, save_profile


def image_bytes():
    buffer = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private-metadata", "not for display")
    Image.new("RGB", (40, 60), "beige").save(buffer, format="PNG", pnginfo=metadata)
    return buffer.getvalue()


def test_optional_profile_updates_roundtrip_and_reject_invalid_values(tmp_path):
    assert profile_view(tmp_path)["name"] == ""
    data = {"name": "Test traveler", "age": 25, "height_cm": 165, "weight_kg": 55}
    assert save_profile(tmp_path, data) == data
    assert profile_view(tmp_path) == data
    for updates in (
        {"age": 1.5},
        {"height_cm": -1},
        {"weight_kg": float("nan")},
        {"weight_kg": True},
        {"unknown": "field"},
    ):
        with pytest.raises(ValueError):
            save_profile(tmp_path, {**data, **updates})
        assert profile_view(tmp_path) == data
    assert save_profile(tmp_path, {"name": ""})["age"] is None


def test_upload_is_decoded_normalized_and_stripped_of_metadata(tmp_path):
    target = tmp_path / "managed.png"
    decode_upload(image_bytes(), target)
    with Image.open(target) as image:
        assert image.mode == "RGB" and image.size == (40, 60)
        assert "private-metadata" not in image.info
    for content in (b"<svg>not a photo</svg>", b"", b"GIF89a", b"x" * (10 * 1024 * 1024 + 1)):
        with pytest.raises(ValueError):
            decode_upload(content, target)


@pytest.fixture
def local_server(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    monkeypatch.setattr(web, "JOBS", tmp_path / "runs/web_jobs")
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def call(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    connection.request(method, path, body=body, headers={"X-Trip-UI": "1", **(headers or {})})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data


def test_profile_http_and_private_records_not_served(local_server):
    server = local_server
    assert call(server, "GET", "/profile")[0] == 200
    status, raw = call(server, "POST", "/api/profile", json.dumps({"name": "Preview"}))
    assert status == 200 and json.loads(raw)["name"] == "Preview"
    assert json.loads(call(server, "GET", "/api/profile")[1])["name"] == "Preview"
    assert call(server, "GET", "/runs/web_profile/profile.json")[0] == 404
    assert call(server, "GET", "/wardrobe-media/../../.secrets/key.png")[0] == 404
    assert call(server, "POST", "/api/profile", "{}", {"Origin": "https://other.example"})[0] == 403
    assert call(server, "POST", "/api/wardrobe/upload", b"not an image")[0] == 400
    assert not web.ACTIVE.locked()


def test_portrait_upload_status_media_and_delete(local_server):
    assert json.loads(call(local_server, "GET", "/api/profile/portrait")[1])["uploaded"] is False
    status, raw = call(
        local_server,
        "POST",
        "/api/profile/portrait",
        image_bytes(),
        {"Content-Type": "image/png"},
    )
    saved = json.loads(raw)
    assert status == 200 and saved == {
        "uploaded": True,
        "portrait_ref": "current",
        "image": "/portrait-media/current.png",
        "consent": True,
    }
    assert call(local_server, "GET", "/portrait-media/current.png")[0] == 200
    assert call(local_server, "DELETE", "/api/profile/portrait")[0] == 200
    assert portrait_view(web.ROOT)["uploaded"] is False


def test_upload_dispatches_managed_pixels_not_user_filename(local_server, monkeypatch):
    done = threading.Event()
    observed = {}

    def worker(job, mode):
        try:
            directory = web.JOBS / job
            observed["mode"] = mode
            observed["record"] = web.read(directory / "upload.json")
            with Image.open(directory / "upload.png") as image:
                observed["pixels"] = image.size
                observed["metadata"] = image.info
        finally:
            web.ACTIVE.release()
            done.set()

    monkeypatch.setattr(web, "execute", worker)
    status, raw = call(
        local_server,
        "POST",
        "/api/wardrobe/upload",
        image_bytes(),
        {"Content-Type": "image/png", "X-File-Name": "../../shirt.png"},
    )
    assert status == 202 and len(json.loads(raw)["id"]) == 32
    assert done.wait(2)
    assert observed == {
        "mode": "wardrobe",
        "record": {"title": "shirt"},
        "pixels": (40, 60),
        "metadata": {},
    }
