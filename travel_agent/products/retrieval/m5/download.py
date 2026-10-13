"""Optional curl-based downloader for Macs whose HF HTTP client stalls."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .encoder import MODEL_ID, MODEL_REVISION

FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "model.safetensors",
]
WEIGHT_SHA256 = "4977e3a54929eccf065ce449aeaf296f0e5cb6b28e8798c3c97d67cb2f6dafc9"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def download(cache_dir):
    snapshot = (
        Path(cache_dir) / "models--patrickjohncyh--fashion-clip" / "snapshots" / MODEL_REVISION
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = snapshot / name
        valid = target.exists()
        if valid:
            if name == "model.safetensors":
                valid = sha256(target) == WEIGHT_SHA256
            else:
                try:
                    if name.endswith(".json"):
                        json.loads(target.read_text())
                    else:
                        valid = target.stat().st_size > 0
                except (OSError, ValueError):
                    valid = False
        if not valid:
            pending = target.with_suffix(target.suffix + ".download")
            print(f"Downloading {name}", flush=True)
            subprocess.run(
                [
                    "curl",
                    "-sS",
                    "-L",
                    "--fail",
                    "--retry",
                    "2",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "300",
                    f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}",
                    "-o",
                    str(pending),
                ],
                check=True,
            )
            if name == "model.safetensors" and sha256(pending) != WEIGHT_SHA256:
                raise ValueError("FashionCLIP weight SHA-256 mismatch")
            if name.endswith(".json"):
                json.loads(pending.read_text())
            pending.replace(target)
    print(f"Pinned FashionCLIP snapshot ready: {snapshot}")
    return snapshot


def main():
    parser = argparse.ArgumentParser(
        description="Download pinned FashionCLIP with curl and verify weight SHA-256"
    )
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/huggingface"))
    args = parser.parse_args()
    try:
        download(args.model_cache)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        import sys

        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
