"""Protocol verification based on eBay's official event notification SDK."""

import base64
import hashlib
import json
import re
import time
from urllib.parse import quote

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from ..retrieval.ebay.client import EbayError


def json_object(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: None)
    if not isinstance(result, dict):
        raise ValueError("JSON object required")
    return result


def challenge(code, token, endpoint):
    if not code or len(code) > 1024:
        raise ValueError("invalid challenge")
    return hashlib.sha256((code + token + endpoint).encode()).hexdigest()


class PublicKeys:
    def __init__(self, client):
        self.client = client
        self.cache = {}

    def __call__(self, kid):
        if kid in self.cache and self.cache[kid][0] > time.monotonic():
            return self.cache[kid][1]
        self.client.authenticate()
        data = self.client._json(
            "GET",
            "/commerce/notification/v1/public_key/" + quote(kid, safe=""),
            headers={"Authorization": "Bearer " + self.client.token},
        )
        if len(self.cache) >= 128:
            self.cache.clear()
        self.cache[kid] = (time.monotonic() + 3600, data)
        return data


class Verifier:
    def __init__(self, keys):
        self.keys = keys

    def verify(self, raw, header):
        if not header or len(header) > 8192:
            raise ValueError("missing/oversized signature")
        signature = json_object(base64.b64decode(header, validate=True))
        kid = signature.get("kid", "")
        if not isinstance(kid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", kid):
            raise ValueError("invalid key ID")
        if str(signature.get("alg", "")).lower() != "ecdsa" or signature.get("digest") != "SHA1":
            raise ValueError("unsupported signature protocol")
        record = self.keys(kid)
        if record.get("algorithm") != "ECDSA" or record.get("digest") != "SHA1":
            raise ValueError("unsupported public key protocol")
        key_text = record["key"]
        key_text = key_text.replace("-----BEGIN PUBLIC KEY-----", "").replace(
            "-----END PUBLIC KEY-----", ""
        )
        encoded = "".join(key_text.split())
        lines = "\n".join(encoded[i : i + 64] for i in range(0, len(encoded), 64))
        pem = "-----BEGIN PUBLIC KEY-----\n" + lines + "\n-----END PUBLIC KEY-----\n"
        key = serialization.load_pem_public_key(pem.encode())
        if not isinstance(key, ec.EllipticCurvePublicKey):
            raise ValueError("EC key required")
        message = json_object(raw)
        sig = base64.b64decode(signature["signature"], validate=True)
        # Preserve raw-body verification; the official Node SDK also verifies JSON.stringify.
        compact = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode()
        for body in dict.fromkeys((raw, compact)):
            try:
                key.verify(sig, body, ec.ECDSA(hashes.SHA1()))
                return message
            except InvalidSignature:
                pass
        raise ValueError("invalid notification signature")


__all__ = ["EbayError", "PublicKeys", "Verifier", "challenge", "json_object"]
