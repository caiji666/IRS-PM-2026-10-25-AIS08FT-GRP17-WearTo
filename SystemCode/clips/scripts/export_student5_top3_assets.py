"""Export the Student 5 Top-3 test result as image folders and one contact sheet.

Run from the repository root (the contact sheet needs Pillow, e.g. the `preview` extra):

    uv run --extra preview python -m scripts.export_student5_top3_assets [--top3 FILE] [--out DIR]

Reads `artifacts/student4_real_data/student5_top3.json` (written by
`scripts.run_student4_real_top3`) and writes, next to it:

    top3_images/01_<outfit_id>/{TOP,BOTTOM,FOOTWEAR[,OUTERWEAR]}.<ext>, outfit.json
    top3_images/download_summary.json
    top3_contact_sheet.png

A quick test-result view only, not the team UI. Images are downloaded from each product's
`image_url`: http/https only, a timeout per request, an image Content-Type and image file
signature, and a size limit. A failed image is recorded and skipped; the other images and the
contact sheet are still produced ("Image unavailable" placeholder). `outfit.json` keeps only the
display fields, prices and scores. `artifacts/` is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_TOP3 = Path("artifacts/student4_real_data/student5_top3.json")
CATEGORY_ORDER = ("TOP", "BOTTOM", "FOOTWEAR", "OUTERWEAR")
TIMEOUT_SECONDS = 15.0
MAX_IMAGE_BYTES = 10 * 1024 * 1024
_SIGNATURES = (  # (prefix, extension); WEBP also needs "WEBP" at offset 8
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)
_PRODUCT_FIELDS = (
    "product_id",
    "title",
    "category",
    "price",
    "currency",
    "availability",
    "available_sizes",
    "size_data_reliable",
    "is_user_owned",
    "source",
    "product_score",
    "product_url",
    "image_url",
)


class ImageDownloadError(Exception):
    """One image could not be used; the message is the recorded reason."""


Fetcher = Callable[[str], tuple[str, bytes]]  # url -> (content_type, body)


def fetch(url: str) -> tuple[str, bytes]:
    """GET `url` with a timeout; read at most MAX_IMAGE_BYTES + 1 bytes."""
    request = urllib.request.Request(url, headers={"User-Agent": "student5-top3-export/1.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        if urlsplit(response.geturl()).scheme not in ("http", "https"):
            raise ImageDownloadError("redirected to a non-HTTP(S) URL")
        return response.headers.get("Content-Type", ""), response.read(MAX_IMAGE_BYTES + 1)


def image_extension(body: bytes) -> str | None:
    """File extension from the image signature, or None if it is not a known image."""
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return ".webp"
    for prefix, extension in _SIGNATURES:
        if body.startswith(prefix):
            return extension
    return None


def download_image(url: object, target_stem: Path, fetcher: Fetcher = fetch) -> Path:
    """Save one product image as `<target_stem><ext>`; raise ImageDownloadError on any problem."""
    if not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https"):
        raise ImageDownloadError("not an http/https URL")
    try:
        content_type, body = fetcher(url)
    except ImageDownloadError:
        raise
    except Exception as exc:  # network errors, HTTP errors, timeouts
        raise ImageDownloadError(f"download failed ({type(exc).__name__})") from None
    if not content_type.lower().startswith("image/"):
        raise ImageDownloadError("response is not an image (Content-Type)")
    if len(body) > MAX_IMAGE_BYTES:
        raise ImageDownloadError(f"image larger than {MAX_IMAGE_BYTES} bytes")
    extension = image_extension(body)
    if extension is None:
        raise ImageDownloadError("response body is not a JPEG, PNG, GIF or WEBP image")
    path = target_stem.with_suffix(extension)
    path.write_bytes(body)
    return path


def _products(item: dict) -> list[dict]:
    products = item.get("purchase_items", []) + item.get("existing_items", [])
    return sorted(products, key=lambda p: CATEGORY_ORDER.index(p["category"]))


def outfit_folder_name(rank: int, outfit_id: str) -> str:
    return f"{rank:02d}_{outfit_id}"


def export_images(top3: dict, out_dir: Path, fetcher: Fetcher = fetch) -> dict:
    """Write one folder per outfit and return the download summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"downloaded": 0, "failed": 0, "outfits": []}
    for item in top3["top3"]:
        folder = out_dir / outfit_folder_name(item["display_position"], item["outfit_id"])
        folder.mkdir(exist_ok=True)
        products = []
        for product in _products(item):
            record = {key: product.get(key) for key in _PRODUCT_FIELDS}
            try:
                path = download_image(
                    product.get("image_url"), folder / product["category"], fetcher
                )
            except ImageDownloadError as exc:
                record.update(image_file=None, image_status="failed", image_error=str(exc))
                summary["failed"] += 1
            else:
                record.update(image_file=path.name, image_status="ok", image_error=None)
                summary["downloaded"] += 1
            products.append(record)
        outfit = {
            "rank": item["display_position"],
            "outfit_id": item["outfit_id"],
            "total_purchase_cost": item["total_purchase_cost"],
            "currency": item["currency"],
            "compatibility_score": item["compatibility_score"],
            "outfit_product_relevance": item["outfit_product_relevance"],
            "final_outfit_score": item["final_outfit_score"],
            "products": products,
        }
        _write_json(folder / "outfit.json", outfit)
        summary["outfits"].append(
            {
                "folder": folder.name,
                "images": {
                    p["category"]: p["image_status"]
                    if p["image_status"] == "ok"
                    else p["image_error"]
                    for p in products
                },
            }
        )
    _write_json(out_dir / "download_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Contact sheet (Pillow)
# ---------------------------------------------------------------------------

TILE = 240
INFO_WIDTH = 330
PADDING = 20
LABEL_HEIGHT = 46
ROW_HEIGHT = TILE + LABEL_HEIGHT + 2 * PADDING


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)  # scalable built-in font (Pillow >= 10.1)
    except TypeError:
        return ImageFont.load_default()


def _tile(path: Path | None, label: str):
    """A TILE x TILE square: the image letterboxed (aspect ratio kept) or a placeholder."""
    from PIL import Image, ImageDraw, ImageOps

    tile = Image.new("RGB", (TILE, TILE), (238, 240, 243))
    draw = ImageDraw.Draw(tile)
    image = None
    if path is not None:
        try:
            with Image.open(path) as raw:
                image = ImageOps.contain(ImageOps.exif_transpose(raw).convert("RGB"), (TILE, TILE))
        except (OSError, ValueError):
            image = None
    if image is not None:
        tile.paste(image, ((TILE - image.width) // 2, (TILE - image.height) // 2))
    else:
        draw.rectangle((4, 4, TILE - 5, TILE - 5), outline=(200, 40, 40), width=3)
        _centered(draw, (TILE // 2, TILE // 2), label, _font(18), (200, 40, 40))
    return tile


def _centered(draw, center, text, font, fill) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (center[0] - (right - left) // 2, center[1] - (bottom - top) // 2),
        text,
        font=font,
        fill=fill,
    )


DEFAULT_SHEET_TITLE = "Student 5 Top-3 (test result, temporary 500 USD budget; not the team UI)"


def contact_sheet(
    top3: dict, image_dir: Path, output: Path, title_text: str = DEFAULT_SHEET_TITLE
) -> Path:
    """One row per outfit: info column, then TOP, BOTTOM, FOOTWEAR, OUTERWEAR tiles."""
    from PIL import Image, ImageDraw

    rows = top3["top3"]
    width = INFO_WIDTH + len(CATEGORY_ORDER) * (TILE + PADDING) + PADDING
    height = len(rows) * ROW_HEIGHT + 60
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title, body, small = _font(24), _font(18), _font(16)
    draw.text(
        (PADDING, 18),
        title_text,
        font=title,
        fill=(17, 24, 39),
    )
    for n, item in enumerate(rows):
        y = 60 + n * ROW_HEIGHT
        if n:
            draw.line((PADDING, y, width - PADDING, y), fill=(217, 220, 225), width=1)
        lines = [
            (f"#{item['display_position']}  {item['outfit_id']}", title),
            (f"Total: {item['total_purchase_cost']} {item['currency']}", body),
            (f"compatibility_score: {item['compatibility_score']:.6f}", small),
            (f"outfit_product_relevance: {item['outfit_product_relevance']:.6f}", small),
            (f"final_outfit_score: {item['final_outfit_score']:.6f}", small),
        ]
        ty = y + PADDING
        for text, font in lines:
            draw.text((PADDING, ty), text, font=font, fill=(17, 24, 39))
            ty += draw.textbbox((0, 0), text, font=font)[3] + 12
        folder = image_dir / outfit_folder_name(item["display_position"], item["outfit_id"])
        by_category = {p["category"]: p for p in _products(item)}
        for c, category in enumerate(CATEGORY_ORDER):
            x = INFO_WIDTH + c * (TILE + PADDING)
            product = by_category.get(category)
            path = next(iter(sorted(folder.glob(f"{category}.*"))), None) if product else None
            placeholder = "Image unavailable" if product else f"No {category.lower()}"
            sheet.paste(_tile(path, placeholder), (x, y + PADDING))
            label = f"{category}  {product['price']} {product['currency']}" if product else category
            draw.text((x, y + PADDING + TILE + 10), label, font=body, fill=(55, 65, 81))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top3", type=Path, default=DEFAULT_TOP3)
    parser.add_argument("--out", type=Path, default=None, help="default: next to --top3")
    parser.add_argument("--title", default=DEFAULT_SHEET_TITLE, help="contact sheet heading")
    args = parser.parse_args(argv)
    out = args.out or args.top3.parent
    top3 = json.loads(args.top3.read_text(encoding="utf-8"))

    summary = export_images(top3, out / "top3_images")
    print(f"images: {summary['downloaded']} downloaded, {summary['failed']} failed")
    for outfit in summary["outfits"]:
        print(f"  {outfit['folder']}: {outfit['images']}")
    print(f"image folders: {(out / 'top3_images').resolve()}")
    sheet = contact_sheet(top3, out / "top3_images", out / "top3_contact_sheet.png", args.title)
    print(f"contact sheet: {sheet.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
