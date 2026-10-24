import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 20_000_000
MAX_BYTES = 25_000_000


def load_image(path, root):
    """Local raster only. Paths including symlinks must stay within root."""
    root = Path(root).resolve()
    path = Path(path)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("image path must stay inside image root") from exc
    if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("supported images: PNG/JPEG/WebP; SVG placeholder is not a photo")
    if resolved.stat().st_size > MAX_BYTES:
        raise ValueError("image exceeds 25 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(resolved) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("image exceeds 20 million pixels")
                image.load()
                result = ImageOps.exif_transpose(image).convert("RGB")
                return result.copy()
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
    ) as exc:
        raise ValueError(f"invalid raster image: {exc}") from exc
