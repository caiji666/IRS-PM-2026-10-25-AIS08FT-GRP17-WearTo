from pathlib import Path

import pytest

from travel_agent.recommendation.portrait_store import (
    PortraitReferenceError,
    resolve_user_portrait_ref,
)


def test_resolves_one_supported_image_from_the_managed_store(tmp_path):
    store = tmp_path / "portraits"
    store.mkdir()
    portrait = store / "portrait_abc123.webp"
    portrait.write_bytes(b"placeholder")

    assert resolve_user_portrait_ref("portrait_abc123", root=store) == portrait.resolve()


@pytest.mark.parametrize(
    "portrait_ref",
    ["../secret", "folder/portrait", "C:/portrait.jpg", "https://example.com/p.jpg", ""],
)
def test_rejects_paths_urls_and_invalid_references(tmp_path, portrait_ref):
    store = tmp_path / "portraits"
    store.mkdir()

    with pytest.raises(PortraitReferenceError, match="safe portrait identifier"):
        resolve_user_portrait_ref(portrait_ref, root=store)


def test_rejects_missing_or_ambiguous_portraits(tmp_path):
    store = tmp_path / "portraits"
    store.mkdir()
    with pytest.raises(PortraitReferenceError, match="not found"):
        resolve_user_portrait_ref("portrait_abc123", root=store)

    (store / "portrait_abc123.jpg").write_bytes(b"jpg")
    (store / "portrait_abc123.png").write_bytes(b"png")
    with pytest.raises(PortraitReferenceError, match="ambiguous"):
        resolve_user_portrait_ref("portrait_abc123", root=store)


def test_uses_the_configured_environment_store(tmp_path):
    store = tmp_path / "portraits"
    store.mkdir()
    portrait = store / "portrait_user_1.png"
    portrait.write_bytes(b"placeholder")

    resolved = resolve_user_portrait_ref(
        "portrait_user_1", environ={"PORTRAIT_UPLOAD_ROOT": str(store)}
    )

    assert resolved == Path(portrait).resolve()


def test_missing_store_is_rejected(tmp_path):
    with pytest.raises(PortraitReferenceError, match="not available"):
        resolve_user_portrait_ref("portrait_abc", root=tmp_path / "missing")


def test_link_escaping_the_store_is_rejected(tmp_path):
    store = tmp_path / "portraits"
    store.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not in the store")
    try:
        (store / "portrait_abc.png").symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this system")
    with pytest.raises(PortraitReferenceError, match="inside the managed upload store"):
        resolve_user_portrait_ref("portrait_abc", root=store)
