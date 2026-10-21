"""Serve the private local trip UI: python -m travel_agent.web --port 8765."""

import argparse
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

ROOT = Path.cwd().resolve()
STATIC = Path(__file__).parent / "static"
JOBS = ROOT / "runs/web_jobs"
LOCK = threading.Lock()
ACTIVE = threading.Lock()
COUNTRIES = {
    "KR": "South Korea",
    "JP": "Japan",
    "SG": "Singapore",
    "FR": "France",
    "IT": "Italy",
    "GB": "United Kingdom",
    "US": "United States",
    "ES": "Spain",
    "TH": "Thailand",
    "AU": "Australia",
    "CN": "China",
}
RUN_ID = re.compile(r"^\d{8}_\d{6}_\d{6}\+0800$")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def update(job, **values):
    with LOCK:
        path = JOBS / job / "status.json"
        data = read(path)
        data.update(values)
        write(path, data)


def validate_form(form):
    if not isinstance(form, dict):
        raise ValueError("Invalid trip form")
    country = form.get("country")
    if country not in COUNTRIES:
        raise ValueError("Choose a country")
    city = str(form.get("city", "")).strip()
    if not 1 <= len(city) <= 80:
        raise ValueError("Enter a city")
    start = datetime.strptime(str(form.get("start_date")), "%Y-%m-%d").date()
    end = datetime.strptime(str(form.get("end_date")), "%Y-%m-%d").date()
    today = datetime.now(ZoneInfo("Asia/Singapore")).date()
    if not today <= start <= end <= today + timedelta(days=15):
        raise ValueError("Live weather supports the next 15 days. Please adjust your travel dates.")
    budget = float(form.get("budget", 0))
    if not 0 < budget <= 1000000:
        raise ValueError("Enter a positive budget")
    activities = form.get("activities")
    if (
        not isinstance(activities, list)
        or not 1 <= len(activities) <= 8
        or any(not isinstance(x, str) or not 1 <= len(x.strip()) <= 60 for x in activities)
    ):
        raise ValueError("Choose 1 to 8 activities, each no longer than 60 characters.")
    form = {**form, "activities": list(dict.fromkeys(x.strip() for x in activities))}
    if form.get("currency") not in {"SGD", "USD", "EUR", "GBP", "JPY", "KRW", "CNY"}:
        raise ValueError("Choose a currency")
    if form.get("style") not in {"minimalist", "casual", "sporty", "streetwear", "romantic"}:
        raise ValueError("Choose a style")
    if form.get("dress_code", "") not in {"", "smart casual", "dressy casual", "festive", "casual"}:
        raise ValueError("Choose a dinner dress code")
    notes = str(form.get("notes", "")).strip()
    if len(notes) > 1500:
        raise ValueError("Notes must be shorter than 1500 characters")
    return {
        **form,
        "city": city,
        "country_name": COUNTRIES[country],
        "budget": budget,
        "notes": notes,
    }


def public_result(stamp):
    if not RUN_ID.fullmatch(stamp):
        raise ValueError("Invalid result")
    folder = ROOT / "runs/product_retrieval" / stamp
    report = read(folder / "result.json")
    products = {}
    for category, items in report["bundle"]["top_k_products"].items():
        products[category] = []
        ranks = report["bundle"]["product_ranking"]["ranked_products"].get(category, [])
        for index, item in enumerate(items):
            relative = item.get("image_path")
            original = item.get("provenance", {}).get("original", {})
            products[category].append(
                {
                    "id": item["product_id"],
                    "title": item["title"],
                    "price": item.get("price"),
                    "image": f"/media/{stamp}/{relative}" if relative else None,
                    "url": item.get("product_url"),
                    "availability": item.get("availability"),
                    "verified": item.get("evidence", {}).get("availability_verified", False),
                    "fetched_at": original.get("fetched_at", report.get("fetched_at")),
                    "sizes": item.get("sizes", []),
                    "colors": item.get("colors", []),
                    "score": ranks[index].get("score") if index < len(ranks) else None,
                }
            )
    raw_path = ROOT / "runs/product_retrieval_diagnostics" / stamp / "run_report.json"
    raw = read(raw_path) if raw_path.is_file() else {}
    upstream = raw.get("upstream_metadata", {})
    trip = upstream.get("student1", {}).get("data", {}).get("trip_request", {})
    refs = upstream.get("top_images", [])
    images = sorted((raw_path.parent / "reference_images").glob("*.jpg"))
    references = []
    for i, image in enumerate(images):
        meta = refs[i] if i < len(refs) else {}
        references.append(
            {
                "image": f"/reference/{stamp}/{image.name}",
                "photographer": meta.get("photographer", ""),
                "source_url": meta.get("source_url", ""),
            }
        )
    return {
        "id": stamp,
        "fetched_at": report.get("fetched_at"),
        "trip": trip,
        "products": products,
        "references": references,
        "owned_count": len(report["bundle"]["owned_items"]),
        "count": sum(len(x) for x in products.values()),
    }


def execute(job, mode):
    directory = JOBS / job
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = (
        [sys.executable, "-m", "travel_agent.web.worker", str(directory)]
        if mode in {"prepare", "chat", "wardrobe"}
        else [
            sys.executable,
            "-m",
            "scripts.run_live_product_pipeline",
            "--input",
            str(directory / "confirmed_input.json"),
        ]
    )
    if mode in {"chat", "wardrobe"}:
        command.append(mode)
    try:
        with (directory / f"{mode}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output = None
            stopped = None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if "Student 1: ready" in line:
                    update(
                        job, stage="inspiration", message="Finding inspiration for your journey…"
                    )
                elif "Student 2:" in line:
                    update(job, stage="products", message="Finding and comparing real eBay pieces…")
                elif line.startswith("{"):
                    try:
                        value = json.loads(line)
                        if value.get("output"):
                            output = value
                        elif value.get("status") in {"needs_input", "failed"}:
                            stopped = value
                    except json.JSONDecodeError:
                        pass
            code = process.wait()
        if stopped and stopped.get("status") == "needs_input":
            review = read(directory / "review.json")
            review["questions"] = stopped.get("questions", [])
            update(
                job,
                status="review",
                stage="review",
                review=review,
                message="More trip details are needed before inspiration or product search.",
            )
            return
        if code:
            stage = (stopped or {}).get(
                "stage",
                "wardrobe"
                if mode == "wardrobe"
                else "understanding"
                if mode in {"prepare", "chat"}
                else "weather",
            )
            messages = {
                "wardrobe": "Try another wardrobe photo or check the local model.",
                "understanding": "Check your message or local Qwen model.",
                "weather": "Weather lookup failed. Check the destination and dates.",
                "inspiration": "Inspiration search failed. Check Unsplash connectivity or quota.",
                "products": "Product search failed. Check eBay Production connectivity or access.",
                "setup": "The local setup is incomplete. Check model and credential files.",
            }
            raise RuntimeError(messages.get(stage, "This search did not finish. Please try again."))
        if mode == "wardrobe":
            result = read(directory / "wardrobe_result.json")
            update(
                job,
                status="completed",
                stage="wardrobe",
                wardrobe=result,
                message="Your wardrobe is ready.",
            )
        elif mode == "chat":
            conversation = read(directory / "conversation.json")
            is_review = conversation["status"] == "review"
            update(
                job,
                status=conversation["status"],
                stage="review" if is_review else "chat",
                message="Review your trip"
                if is_review
                else "Tell me a little more about your trip",
                review=read(directory / "review.json") if is_review else None,
            )
        elif mode == "prepare":
            update(
                job,
                status="review",
                stage="review",
                message="Review your travel plans",
                review=read(directory / "review.json"),
            )
        elif output:
            stamp = Path(output["output"]).parent.name
            public_result(stamp)  # Confirm the downstream artifact exists before marking completed.
            update(
                job,
                status="completed",
                stage="completed",
                result=stamp,
                message="Your travel edit is ready.",
            )
        else:
            raise RuntimeError("No product result was saved. Please try again.")
    except Exception as exc:
        # Never return arbitrary model output, HTTP bodies, logs or file paths to the browser.
        update(
            job,
            status="failed",
            message=str(exc)
            if isinstance(exc, RuntimeError)
            else "The request could not finish. Check the local service and try again.",
        )
    finally:
        ACTIVE.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def safe_host(self):
        return self.headers.get("Host") in {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }

    def send(self, data, status=200, content_type="application/json; charset=utf-8"):
        content = (
            json.dumps(data, ensure_ascii=False).encode() if not isinstance(data, bytes) else data
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(content)

    def file(self, path, scope):
        path = path.resolve()
        path.relative_to(scope.resolve())
        self.send(
            path.read_bytes(),
            content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )

    def do_GET(self):
        if not self.safe_host():
            return self.send({"error": "Local access only"}, 403)
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/profile":
                from .profile import profile_view

                return self.send(profile_view(ROOT))
            if path == "/api/wardrobe":
                from .profile import wardrobe_view

                return self.send(wardrobe_view(ROOT))
            if path.startswith("/wardrobe-media/"):
                name = path.rsplit("/", 1)[1]
                if not re.fullmatch(r"[a-f0-9]{64}\.png", name):
                    raise ValueError("Invalid wardrobe image")
                scope = ROOT / "runs/wardrobe/images"
                return self.file(scope / name, scope)
            if path == "/profile":
                return self.file(STATIC / "profile.html", STATIC)
            if path == "/api/history":
                stamps = sorted(
                    (ROOT / "runs/product_retrieval").glob("*/result.json"), reverse=True
                )
                return self.send(
                    [{"id": p.parent.name} for p in stamps[:20] if RUN_ID.fullmatch(p.parent.name)]
                )
            if path.startswith("/api/result/"):
                return self.send(public_result(path.rsplit("/", 1)[1]))
            if path.startswith("/api/job/"):
                job = path.rsplit("/", 1)[1]
                if not re.fullmatch("[a-f0-9]{32}", job):
                    raise ValueError("Invalid job")
                status = read(JOBS / job / "status.json")
                if status.get("status") == "review":
                    from .review import dress_code_options

                    status["review"]["dress_code_options"] = dress_code_options(
                        status["review"]["request"]
                    )
                conversation_path = JOBS / job / "conversation.json"
                if conversation_path.is_file():
                    conversation = read(conversation_path)
                    status["messages"] = conversation["messages"]
                    status["request"] = conversation.get("request", {})
                return self.send(status)
            if path.startswith("/media/") or path.startswith("/reference/"):
                _, kind, stamp, relative = path.split("/", 3)
                if not RUN_ID.fullmatch(stamp):
                    raise ValueError("Invalid image")
                if kind == "media":
                    scope = ROOT / "runs/product_retrieval" / stamp
                else:
                    scope = ROOT / "runs/product_retrieval_diagnostics" / stamp / "reference_images"
                if Path(relative).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    raise ValueError("Invalid image")
                return self.file(scope / relative, scope)
            return self.file(STATIC / ("index.html" if path == "/" else path.lstrip("/")), STATIC)
        except (OSError, ValueError, KeyError):
            return self.send(
                {"error": "Content unavailable. This saved result may have been cleaned up."},
                404,
            )

    def do_POST(self):
        if not self.safe_host():
            return self.send({"error": "Local access only"}, 403)
        origin = self.headers.get("Origin")
        if origin and origin not in {
            f"http://127.0.0.1:{self.server.server_port}",
            f"http://localhost:{self.server.server_port}",
        }:
            return self.send({"error": "Origin rejected"}, 403)
        if self.headers.get("X-Trip-UI") != "1":
            return self.send({"error": "Missing request header"}, 403)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if self.path == "/api/wardrobe/upload":
                from .profile import MAX_UPLOAD_BYTES, decode_upload

                if not 0 < size <= MAX_UPLOAD_BYTES:
                    raise ValueError("Choose an image smaller than 10 MB.")
                if not ACTIVE.acquire(blocking=False):
                    return self.send({"error": "Another task is running. Please wait."}, 409)
                try:
                    job = secrets.token_hex(16)
                    directory = JOBS / job
                    directory.mkdir(parents=True)
                    decode_upload(self.rfile.read(size), directory / "upload.png")
                    name = unquote(self.headers.get("X-File-Name", "Wardrobe piece"))
                    title = Path(name).stem.replace("_", " ").strip()[:100] or "Wardrobe piece"
                    write(directory / "upload.json", {"title": title})
                    write(
                        directory / "status.json",
                        {
                            "id": job,
                            "status": "running",
                            "stage": "wardrobe",
                            "message": "Getting to know your wardrobe…",
                        },
                    )
                    threading.Thread(target=execute, args=(job, "wardrobe"), daemon=True).start()
                except Exception:
                    ACTIVE.release()
                    raise
                return self.send({"id": job}, 202)
            if not 0 < size < 12000:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            if self.path == "/api/profile":
                from .profile import save_profile

                return self.send(save_profile(ROOT, body))
            if self.path == "/api/chat":
                message = body.get("message", "")
                if not isinstance(message, str) or not 1 <= len(message.strip()) <= 2500:
                    raise ValueError("Write a message between 1 and 2,500 characters.")
                job = body.get("id")
                if job:
                    if not re.fullmatch("[a-f0-9]{32}", job):
                        raise ValueError("Invalid conversation")
                    directory = JOBS / job
                    conversation = read(directory / "conversation.json")
                    if conversation.get("turn", 0) >= 30:
                        raise ValueError("Please start a new journey to continue.")
                else:
                    job = secrets.token_hex(16)
                    directory = JOBS / job
                    conversation = {"messages": [], "request": {}, "turn": 0}
                if not ACTIVE.acquire(blocking=False):
                    return self.send({"error": "Another request is in progress. Please wait."}, 409)
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    conversation["messages"].append({"role": "user", "content": message.strip()})
                    conversation["turn"] += 1
                    write(directory / "conversation.json", conversation)
                    write(
                        directory / "status.json",
                        {
                            "id": job,
                            "status": "running",
                            "stage": "understanding",
                            "message": "Reading your travel plans…",
                        },
                    )
                    threading.Thread(target=execute, args=(job, "chat"), daemon=True).start()
                except Exception:
                    ACTIVE.release()
                    raise
                return self.send({"id": job}, 202)
            if self.path == "/api/prepare":
                form = validate_form(body)
                if not ACTIVE.acquire(blocking=False):
                    return self.send({"error": "Another search is running. Please wait."}, 409)
                job = secrets.token_hex(16)
                directory = JOBS / job
                directory.mkdir(parents=True)
                write(directory / "form.json", form)
                write(
                    directory / "status.json",
                    {
                        "id": job,
                        "status": "running",
                        "stage": "understanding",
                        "message": "Reading your travel plans…",
                    },
                )
                threading.Thread(target=execute, args=(job, "prepare"), daemon=True).start()
                return self.send({"id": job}, 202)
            if self.path == "/api/confirm":
                job = body.get("id", "")
                if not re.fullmatch("[a-f0-9]{32}", job):
                    raise ValueError("Invalid job")
                directory = JOBS / job
                if read(directory / "status.json")["status"] != "review":
                    raise ValueError("Please finish the trip conversation first.")
                draft = read(directory / "review.json")
                location = draft["locations"][int(body.get("location", 0))]
                from .review import confirmed_request

                reviewed_request = confirmed_request(draft["request"], body.get("dress_codes", {}))
                payload = {
                    "request": reviewed_request,
                    "allow_draft": True,
                    "weather_provider": "open_meteo",
                    "confirmed_location": {
                        **location,
                        "destination": draft["request"]["destination"],
                    },
                }
                if not ACTIVE.acquire(blocking=False):
                    return self.send({"error": "Another search is running. Please wait."}, 409)
                write(directory / "confirmed_input.json", payload)
                update(
                    job,
                    status="running",
                    stage="weather",
                    message="Checking live weather at your destination…",
                )
                threading.Thread(target=execute, args=(job, "run"), daemon=True).start()
                return self.send({"id": job}, 202)
            return self.send({"error": "Unknown action"}, 404)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            return self.send(
                {
                    "error": str(exc)
                    if isinstance(exc, ValueError)
                    else "Please check your input and try again."
                },
                400,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    print(f"Local trip UI: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
