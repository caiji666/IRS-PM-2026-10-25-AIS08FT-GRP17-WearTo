"""Resolve opaque user portrait references inside the server-managed upload store."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

PORTRAIT_UPLOAD_ROOT_ENV = "PORTRAIT_UPLOAD_ROOT"
DEFAULT_PORTRAIT_UPLOAD_ROOT = Path("runtime") / "uploads" / "portraits"
PORTRAIT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
_PORTRAIT_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class PortraitReferenceError(ValueError):
    """A portrait reference is unsafe, missing or ambiguous."""


def resolve_user_portrait_ref(
    portrait_ref: str,
    *,
    root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a safe opaque reference to one image inside the managed portrait store."""
    if not isinstance(portrait_ref, str) or _PORTRAIT_REF.fullmatch(portrait_ref) is None:
        raise PortraitReferenceError("user_portrait_ref must be a safe portrait identifier")

    env = os.environ if environ is None else environ
    configured_root = root or env.get(PORTRAIT_UPLOAD_ROOT_ENV) or DEFAULT_PORTRAIT_UPLOAD_ROOT
    store = Path(configured_root).resolve()
    if not store.is_dir():
        raise PortraitReferenceError("the portrait upload store is not available")

    matches = [store / f"{portrait_ref}{extension}" for extension in PORTRAIT_EXTENSIONS]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        message = "user portrait was not found" if not matches else "user portrait is ambiguous"
        raise PortraitReferenceError(message)

    resolved = matches[0].resolve(strict=True)
    if not resolved.is_relative_to(store):
        raise PortraitReferenceError("user portrait must stay inside the managed upload store")
    return resolved
