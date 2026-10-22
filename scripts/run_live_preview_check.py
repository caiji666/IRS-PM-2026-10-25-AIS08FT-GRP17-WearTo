"""Manual acceptance check of the Stage 14 preview through the main-agent tool (PROVISIONAL).

What this is, and what it is not:

- `pytest` is the automated regression suite. Its opt-in paid test
  (`tests/recommendation/test_recommendation_live_preview.py`) checks one text-only FLAT_LAY
  preview, keeps no image and writes no record.
- This script is a manual acceptance and demo tool. It runs recommend -> confirm -> preview
  through `validate_and_recommend`, the same LangChain tool the shared agent calls, with local
  garment images (and, in PORTRAIT mode, a consented local portrait), then checks the result.
- Generated images (`artifacts/previews/`) and check records (`artifacts/preview_checks/`) are
  git-ignored and are not committed by default.

Checks: tool status `ok`, `preview_source == "openai"`, preview status `SUCCESS`, the image at
`output_reference` opens, the API key is not in the tool result, and the record contains no
secret, environment value, absolute path, base64 data or portrait detail. Any FAIL gives a
non-zero exit code. `--mock` uses the offline mock provider (no network, no cost); the checks
that need a real image are reported as SKIP.

Configuration is read only from the current process environment (`PREVIEW_PROVIDER`,
`PREVIEW_API_KEY`, `PREVIEW_MODEL`, `PREVIEW_ALLOWED_IMAGE_HOSTS`); no `.env` file is read. Only
missing variable names are reported, never values. Each non-mock run sends ONE paid image request.

Set the variables in the terminal that runs the check (they last only for that session). Never
put the API key itself in a command: a command line is kept in the shell history. Type the key at
a masked prompt instead (works in Windows PowerShell 5.1 and PowerShell 7):

    $env:PREVIEW_PROVIDER = "openai"
    $env:PREVIEW_MODEL = "<image model name>"
    $secure = Read-Host "PREVIEW_API_KEY" -AsSecureString
    $env:PREVIEW_API_KEY = [System.Net.NetworkCredential]::new("", $secure).Password
    Remove-Variable secure

Check that it is set without printing it: `[bool]$env:PREVIEW_API_KEY` shows True or False.

PowerShell (the backtick ` continues a line; it must be the last character on the line):

    uv run --locked --extra preview python -m scripts.run_live_preview_check --mode FLAT_LAY `
        --top data\\raw\\preview_samples\\top.jpg --top-title "White T-shirt" `
        --bottom data\\raw\\preview_samples\\bottom.jpg --bottom-title "Blue jeans" `
        --shoes data\\raw\\preview_samples\\shoes.jpg --shoes-title "White sneakers"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_KEYS = ("PREVIEW_PROVIDER", "PREVIEW_API_KEY", "PREVIEW_MODEL", "PREVIEW_ALLOWED_IMAGE_HOSTS")
REQUIRED_ENV = ("PREVIEW_PROVIDER", "PREVIEW_API_KEY", "PREVIEW_MODEL")
RECORD_DIR = ROOT / "artifacts" / "preview_checks"
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
# In serialized JSON: a drive letter (C:\\ or C:/) or a string value starting with "/".
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:(\\\\|/)|\"/[^/]")
REQUEST = {
    "destination": "Seoul",
    "start_date": "2026-12-20",
    "end_date": "2026-12-27",
    "activities": ["sightseeing"],
    "budget": "300",
    "currency": "SGD",
    "style": ["minimalist"],
}


def config_problems() -> list[str]:
    """Missing or invalid settings, by variable name only (values are never shown)."""
    problems = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
    provider = os.environ.get("PREVIEW_PROVIDER", "").strip()
    if provider and provider != "openai":
        problems.append("PREVIEW_PROVIDER (must be exactly 'openai'; value not shown)")
    return problems


def git_version() -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain")
    return {"git_commit": git("rev-parse", "HEAD"), "git_worktree_dirty": bool(status)}


def item(category: str, path: Path, title: str, n: int) -> dict:
    return {
        "product_id": f"check_{category.lower()}",
        "source": "manual_preview_check",
        "title": title,
        "category": category,
        "price_minor": 3000 + n * 1000,
        "currency": "SGD",
        "availability": True,
        "is_user_owned": False,
        "product_score": 0.7,
        "product_url": f"https://shop.example.com/check_{category.lower()}",
        "image_path": str(path),  # tool input only; never written to the record
    }


def call(tool, payload: dict) -> dict:
    result = tool.invoke({"payload": payload})
    if result["status"] != "ok":
        # Only the status and the short reason; no payload, state or path is printed.
        raise SystemExit(f"{payload['action']} returned {result['status']}: {result.get('reason')}")
    return result


def secrets_and_private_strings(portrait: Path | None) -> list[str]:
    """Strings that must never appear in the record."""
    values = [os.environ.get(name, "").strip() for name in ENV_KEYS if name != "PREVIEW_PROVIDER"]
    values += [str(ROOT), str(ROOT).replace("\\", "/"), "base64"]
    if portrait is not None:
        values += [str(portrait), portrait.as_posix(), portrait.name]
    return [v for v in values if v]


def record_is_clean(record: dict, forbidden: list[str]) -> bool:
    text = json.dumps(record, ensure_ascii=False)
    # Compare in JSON-escaped form, so a Windows path's backslashes match the serialized text.
    escaped = [json.dumps(v, ensure_ascii=False)[1:-1] for v in forbidden]
    return not any(v in text for v in escaped) and not ABSOLUTE_PATH.search(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=["FLAT_LAY", "PORTRAIT"])
    parser.add_argument("--mock", action="store_true", help="offline mock provider, no API call")
    parser.add_argument(
        "--hash-inputs", action="store_true", help="record SHA-256 of the garment images"
    )
    parser.add_argument("--portrait", type=Path, help="PORTRAIT mode: local portrait file")
    parser.add_argument(
        "--consent",
        action="store_true",
        help="PORTRAIT mode: the portrait owner agreed to send it to the image API",
    )
    for role in ("top", "bottom", "shoes"):
        parser.add_argument(f"--{role}", type=Path, required=True)
        parser.add_argument(f"--{role}-title", required=True)
    args = parser.parse_args(argv)

    roles = {
        "TOP": (args.top, args.top_title),
        "BOTTOM": (args.bottom, args.bottom_title),
        "FOOTWEAR": (args.shoes, args.shoes_title),
    }
    missing = [p.name for p, _ in roles.values() if not p.is_file()]
    portrait = None
    if args.mode == "PORTRAIT":
        if args.portrait is None or not args.consent:
            raise SystemExit("PORTRAIT mode needs --portrait <file> and --consent")
        if not args.portrait.is_file():
            missing.append("portrait file")
        portrait = args.portrait.resolve()
    if missing:
        raise SystemExit("file(s) not found: " + ", ".join(missing))
    paths = {c: p.resolve() for c, (p, _) in roles.items()}

    if not args.mock and (problems := config_problems()):
        raise SystemExit("preview not configured; missing or invalid: " + ", ".join(problems))

    os.chdir(ROOT)  # the provider saves to the relative artifacts/previews/
    from PIL import Image

    from travel_agent.recommendation.tools import validate_and_recommend

    student4 = {
        "status": "ok",
        "model_version": None,
        "source": "manual_preview_check",
        "outfits": [
            {
                "outfit_id": "check_outfit",
                "compatibility_score": 0.8,
                "items": [
                    item(c, paths[c], t, n) for n, (c, (_, t)) in enumerate(roles.items(), start=1)
                ],
            }
        ],
    }
    started = datetime.now(UTC)
    first = call(
        validate_and_recommend, {"action": "recommend", "request": REQUEST, "student4": student4}
    )
    confirmed = call(
        validate_and_recommend,
        {
            "action": "confirm",
            "state": first["data"]["state"],
            "recommendation_set": first["data"]["recommendation_set"],
            "outfit_id": "check_outfit",
        },
    )
    preview_payload = {
        "action": "preview",
        "confirmed_outfit": confirmed["data"]["confirmed_outfit"],
        "preview_mode": args.mode,
    }
    if portrait is not None:
        extension = portrait.suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise SystemExit("portrait must be a JPEG, PNG or WEBP image")
        portrait_ref = f"live_check_{hashlib.sha256(portrait.read_bytes()).hexdigest()[:16]}"
        portrait_store = Path(
            os.environ.get("PORTRAIT_UPLOAD_ROOT", ROOT / "runtime" / "uploads" / "portraits")
        ).resolve()
        portrait_store.mkdir(parents=True, exist_ok=True)
        shutil.copy2(portrait, portrait_store / f"{portrait_ref}{extension}")
        os.environ["PORTRAIT_UPLOAD_ROOT"] = str(portrait_store)
        preview_payload.update(user_portrait_ref=portrait_ref, portrait_consent=True)
    if args.mock:
        preview_payload["preview_provider"] = "mock"
    result = validate_and_recommend.invoke({"payload": preview_payload})
    elapsed = round((datetime.now(UTC) - started).total_seconds(), 2)

    data = result.get("data") or {}
    preview = dict(data.get("preview") or {})
    preview.pop("model_name", None)  # equals PREVIEW_MODEL, an environment value
    source = data.get("preview_source")
    output = preview.get("output_reference") or ""

    checks = {"tool_status_ok": PASS if result.get("status") == "ok" else FAIL}
    if args.mock:  # the mock never calls OpenAI or writes an image
        checks["preview_source_is_openai"] = SKIP if source == "mock" else FAIL
    else:
        checks["preview_source_is_openai"] = PASS if source == "openai" else FAIL
    checks["status_is_SUCCESS"] = PASS if preview.get("status") == "SUCCESS" else FAIL
    image_size = None
    if args.mock:
        checks["output_image_opens"] = SKIP if output.startswith("mock://") else FAIL
    else:
        checks["output_image_opens"] = FAIL
        image = ROOT / output if output else None
        if image is not None and not Path(output).is_absolute() and image.is_file():
            try:
                with Image.open(image) as img:
                    img.verify()
                with Image.open(image) as img:
                    image_size = list(img.size)
                checks["output_image_opens"] = PASS
            except Exception:
                pass
    key = os.environ.get("PREVIEW_API_KEY", "").strip()
    if key:
        checks["api_key_not_in_result"] = PASS if key not in json.dumps(result) else FAIL
    else:
        checks["api_key_not_in_result"] = SKIP  # no key in this environment (mock)

    record = {
        "run_at_utc": started.replace(microsecond=0).isoformat(),
        **git_version(),
        "mode": args.mode,
        "mock": args.mock,
        "elapsed_seconds": elapsed,
        "garments": [
            {
                "category": c,
                "title": roles[c][1],
                "file_name": paths[c].name,
                **(
                    {"sha256": hashlib.sha256(paths[c].read_bytes()).hexdigest()}
                    if args.hash_inputs
                    else {}
                ),
            }
            for c in roles
        ],
        "portrait_provided": portrait is not None,  # no path, name, hash or content
        "portrait_consent": portrait is not None,
        "tool_status": result.get("status"),
        "preview_source": source,
        "preview": preview,
        "output_image_size": image_size,
        "checks": checks,
    }
    if not record_is_clean(record, secrets_and_private_strings(portrait)):
        # Nothing is written and nothing from the record is printed.
        print("[FAIL] record_is_clean: the record would contain a secret, environment value,")
        print("       absolute path, base64 data or portrait detail; no record was written.")
        print("RESULT: FAILED")
        return 1
    checks["record_is_clean"] = PASS
    record["passed"] = FAIL not in checks.values()

    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    record_path = RECORD_DIR / f"{stamp}_{args.mode}{'_mock' if args.mock else ''}.json"
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"git commit     = {record['git_commit']} (dirty={record['git_worktree_dirty']})")
    print(f"preview_source = {source}")
    print(f"status         = {preview.get('status')}")
    print(f"output         = {output}")
    for warning in preview.get("warnings", []):
        print(f"warning        : {warning}")
    for name, outcome in checks.items():
        print(f"[{outcome}] {name}")
    print(f"record: {record_path.relative_to(ROOT).as_posix()}")
    print("RESULT:", "PASSED" if record["passed"] else "FAILED")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
