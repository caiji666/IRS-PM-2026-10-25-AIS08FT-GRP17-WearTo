"""Bounded eBay Browse HTTP client. Tokens remain in memory only."""

import base64
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import requests
from PIL import Image, ImageOps

MARKETS = {"EBAY_US", "EBAY_GB", "EBAY_DE", "EBAY_AU"}


class EbayError(RuntimeError):
    pass


class BrowseClient:
    def __init__(self, credentials, marketplace="EBAY_US", session=None):
        env = credentials.get("environment")
        if env not in {"sandbox", "production"}:
            raise ValueError("environment must be sandbox or production")
        if marketplace not in MARKETS:
            raise ValueError("image search supports US/GB/DE/AU marketplaces only")
        self.environment = env
        self.marketplace = marketplace
        self.base = "https://api.sandbox.ebay.com" if env == "sandbox" else "https://api.ebay.com"
        self.client_id = credentials.get("client_id", "")
        self.secret = credentials.get("client_secret", "")
        if not self.client_id or not self.secret:
            raise ValueError("Client ID and Client Secret required")
        if ("-SBX-" if env == "sandbox" else "-PRD-") not in self.client_id:
            raise ValueError("keyset/environment mismatch")
        self.session = session or requests.Session()
        self.token = None
        self.expires = 0

    def _json(self, method, path, **kwargs):
        for attempt in range(3):
            try:
                r = self.session.request(
                    method, self.base + path, timeout=(10, 30), allow_redirects=False, **kwargs
                )
            except requests.RequestException:
                raise EbayError("eBay network error; credentials and tokens omitted")
            if r.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            try:
                body = r.json()
            except ValueError:
                raise EbayError(f"eBay HTTP {r.status_code}: non-JSON response")
            if r.status_code >= 300:
                errors = body.get("errors", []) if isinstance(body, dict) else []
                ids = [e.get("errorId") for e in errors if isinstance(e, dict)]
                label = body.get("error", "") if isinstance(body, dict) else ""
                raise EbayError(f"eBay HTTP {r.status_code}; error_ids={ids}; oauth_error={label}")
            if not isinstance(body, dict):
                raise EbayError("unexpected eBay response type")
            return body

    def authenticate(self):
        if self.token and time.monotonic() < self.expires:
            return
        body = self._json(
            "POST",
            "/identity/v1/oauth2/token",
            auth=(self.client_id, self.secret),
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if not isinstance(body.get("access_token"), str) or not body["access_token"]:
            raise EbayError("OAuth response has no token")
        self.token = body["access_token"]
        self.expires = time.monotonic() + max(0, int(body.get("expires_in", 0)) - 60)

    def _browse(self, method, path, **kwargs):
        self.authenticate()
        return self._json(
            method,
            path,
            headers={
                "Authorization": "Bearer " + self.token,
                "X-EBAY-C-MARKETPLACE-ID": self.marketplace,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            **kwargs,
        )

    def search_image(self, image_bytes, limit=20, category_ids=None):
        if not 1 <= limit <= 200:
            raise ValueError("limit must be 1..200")
        params = {"limit": limit}
        if category_ids:
            params["category_ids"] = category_ids
        return self._browse(
            "POST",
            "/buy/browse/v1/item_summary/search_by_image",
            params=params,
            json={"image": base64.b64encode(image_bytes).decode("ascii")},
        )

    def item(self, item_id):
        return self._browse("GET", "/buy/browse/v1/item/" + quote(item_id, safe=""))


def download_photo(url, target, hosts, session=None):
    """No bearer token on CDN requests; validate each HTTPS redirect and byte limits."""
    session = session or requests.Session()
    current = url
    for _ in range(4):
        parsed = urlparse(current)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in hosts
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
        ):
            raise ValueError("photo URL outside allowed HTTPS image hosts")
        try:
            r = session.get(current, timeout=(10, 30), stream=True, allow_redirects=False)
        except requests.RequestException:
            raise EbayError("photo download network error")
        if r.status_code in {301, 302, 303, 307, 308}:
            location = r.headers.get("Location")
            r.close()
            if not location:
                raise ValueError("image redirect missing Location")
            current = urljoin(current, location)
            continue
        try:
            if r.status_code != 200:
                raise EbayError(f"photo HTTP {r.status_code}")
            chunks = []
            size = 0
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > 8_000_000:
                    raise ValueError("downloaded photo exceeds 8 MB")
                chunks.append(chunk)
        finally:
            r.close()
        data = b"".join(chunks)
        with Image.open(BytesIO(data)) as im:
            if im.width * im.height > 20_000_000:
                raise ValueError("photo exceeds 20 million pixels")
            im.load()
            photo = ImageOps.exif_transpose(im).convert("RGB")
            photo.thumbnail((1024, 1024))
            buf = BytesIO()
            photo.save(buf, format="JPEG", quality=90)
            image = buf.getvalue()
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(image)
        return image
    raise ValueError("too many image redirects")
