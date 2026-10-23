"""Serve the private local trip UI: python -m travel_agent.web --port 8765."""

import argparse
import copy
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .refinement_jobs import PendingRefinementStore

ROOT = Path.cwd().resolve()
STATIC = Path(__file__).parent / "static"
JOBS = ROOT / "runs/web_jobs"
LOCK = threading.Lock()
ACTIVE = threading.Lock()
STUDENT5_LOCK = threading.Lock()
STUDENT5_SESSIONS = {}
STUDENT5_BASE_CACHE = {}
STUDENT5_REFINEMENTS = PendingRefinementStore()
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


def public_student5(session):
    from .live_outfits import public_session

    return public_session(ROOT, session)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cached_file_template(source, cache, calculate):
    """Cache an expensive result while sharing its immutable Student 4 input."""
    source = Path(source)
    stat = source.stat()
    key = (str(source.resolve()), stat.st_mtime_ns, stat.st_size)
    if cache.get("key") != key:
        cache.clear()
        cache.update(key=key, value=calculate())
    value = cache["value"]
    session = copy.deepcopy({key: item for key, item in value.items() if key != "student4"})
    if "student4" in value:
        session["student4"] = value["student4"]
    return session


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    if end < start:
        raise ValueError("The return date cannot precede the departure date.")
    budget = float(form.get("budget", 0))
    if not 0 < budget <= 1000000:
        raise ValueError("Enter a positive budget")
    activities = form.get("activities")
    if activities is None:
        activities = []
    if (
        not isinstance(activities, list)
        or not 0 <= len(activities) <= 8
        or any(not isinstance(x, str) or not 1 <= len(x.strip()) <= 60 for x in activities)
    ):
        raise ValueError("Choose up to 8 activities, each no longer than 60 characters.")
    form = {**form, "activities": list(dict.fromkeys(x.strip() for x in activities))}
    if form.get("currency") not in {"SGD", "USD", "EUR", "GBP", "JPY", "KRW", "CNY"}:
        raise ValueError("Choose a currency")
    style = form.get("style") or ""
    if style not in {"", "minimalist", "casual", "sporty", "streetwear", "romantic"}:
        raise ValueError("Choose a supported style or leave it blank")
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
        "style": style,
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


def pipeline_failure(stopped, mode):
    stopped = stopped or {}
    stage = stopped.get(
        "stage", "setup" if mode == "run" else "wardrobe" if mode == "wardrobe" else "understanding"
    )
    if stage not in {"setup", "weather", "understanding", "inspiration", "products", "wardrobe"}:
        stage = "setup"
    messages = {
        "wardrobe": "Try another wardrobe photo or check the local model.",
        "understanding": "Check your message or local Qwen model.",
        "weather": "Weather lookup failed. Your trip is saved; please try again later.",
        "inspiration": "Inspiration search is unavailable. Your trip details are saved.",
        "products": "Product search is unavailable. Your trip details are saved.",
        "setup": "Search is not configured on this device. Your trip details are saved.",
    }
    if stopped.get("reason_code") == "missing_unsplash_config":
        return (
            "setup",
            "Unsplash inspiration search is not configured. Your trip details are saved.",
        )
    return stage, messages[stage]


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
            "--with-outfits",
        ]
    )
    if mode in {"chat", "wardrobe"}:
        command.append(mode)
    failure_stage = "setup" if mode == "run" else "understanding"
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
                        elif (
                            value.get("status") == "running"
                            and value.get("stage") == "understanding"
                        ):
                            update(
                                job,
                                stage="weather",
                                message="Checking the weather context for your trip…",
                            )
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
            failure_stage, message = pipeline_failure(stopped, mode)
            raise RuntimeError(message)

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
            stage=failure_stage,
            message=str(exc)
            if isinstance(exc, RuntimeError)
            else "The request could not finish. Check the local service and try again.",
        )
    finally:
        ACTIVE.release()


def execute_student5_refinement(job, session_id, pending):
    """Run the real Student 1–4 CLI, then atomically install its Student 5 result."""
    directory = JOBS / job
    try:
        update(job, stage="weather", message="Updating trip context and weather…")
        input_path = directory / "confirmed_input.json"
        write(
            input_path,
            {
                "request": pending.proposal["student1_request"],
                "allow_draft": True,
                "weather_provider": "open_meteo",
            },
        )
        command = [
            sys.executable,
            "-m",
            "scripts.run_live_product_pipeline",
            "--input",
            str(input_path),
            "--with-outfits",
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        record_path = None
        with (directory / "refinement.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                if "Student 1: ready" in line:
                    update(job, stage="inspiration", message="Refreshing style inspiration…")
                elif "Student 2:" in line:
                    update(job, stage="products", message="Refreshing product candidates…")
                elif line.startswith("Student 4:"):
                    update(job, stage="outfits", message="Scoring candidate outfits…")
            code = process.wait()
        records = sorted((ROOT / "runs/live_pipeline").glob("*/pipeline_record.json"))
        if records:
            record_path = records[-1]
        if code or record_path is None:
            raise RuntimeError("The updated trip pipeline could not finish.")
        record = read(record_path)
        student5_path = (record.get("outfits") or {}).get("student5_input")
        if record.get("status") not in {"ok", "completed"} or not student5_path:
            raise RuntimeError("The updated outfits were not ready for final ranking.")

        from travel_agent.recommendation.service import run as student5_run

        payload = read(Path(student5_path))
        update(job, stage="ranking", message="Ranking the new Top-3…")
        result = student5_run(payload)
        if result.status != "ok":
            raise RuntimeError("The updated Top-3 could not be created.")
        data = result.model_dump(mode="json")["data"]
        replacement = {
            "run_id": record_path.parent.name,
            "request": payload["request"],
            "student4": payload["student4"],
            "state": data["state"],
            "recommendation_set": data["recommendation_set"],
            "filter_summary": data["filter_summary"],
            "confirmed_outfit": None,
            "preview": None,
            "checkpoints": {
                "pipeline_record": record_path.parent.name,
                "student1_state": pending.proposal["student1_state"],
            },
        }

        with STUDENT5_LOCK:
            current = STUDENT5_SESSIONS.get(session_id)
            if current is None:
                raise RuntimeError("The recommendation session expired.")
            version = (current.get("state") or {}).get("request_version")
            if version != pending.request_version:
                raise RuntimeError("The recommendation session changed while updating.")
            STUDENT5_SESSIONS[session_id] = replacement
        update(
            job,
            status="completed",
            stage="completed",
            message="Your updated Top-3 is ready.",
            result=record_path.parent.name,
            student5={"id": session_id, **public_student5(replacement)},
        )
    except Exception as exc:
        update(
            job,
            status="failed",
            stage="refinement",
            message=str(exc) if isinstance(exc, RuntimeError) else "The update could not finish.",
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
            if path == "/api/profile/portrait":
                from .profile import portrait_view

                return self.send(portrait_view(ROOT))
            if path == "/api/student5/fixed-top3":
                result = ROOT / (
                    "artifacts/fixed_student4_input_run/budget_500_initial/student5_top3.json"
                )
                if not result.is_file():
                    raise ValueError("Fixed Student 5 result is not available")
                return self.send(read(result))
            if path.startswith("/student5-preview-media/"):
                name = path.rsplit("/", 1)[1]
                if not re.fullmatch(r"[a-f0-9]{32}\.png", name):
                    raise ValueError("Invalid preview image")
                scope = JOBS / "student5-preview-media"
                return self.file(scope / name, scope)
            if path.startswith("/artifacts/previews/"):
                name = path.rsplit("/", 1)[1]
                if not re.fullmatch(r"[a-f0-9]{32}\.png", name):
                    raise ValueError("Invalid preview image")
                scope = ROOT / "artifacts/previews"
                return self.file(scope / name, scope)
            if path == "/portrait-media/current.png":
                from .profile import portrait_path

                target = portrait_path(ROOT)
                return self.file(target, target.parent)
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
                    from .chat import public_messages, public_offer

                    conversation = read(conversation_path)
                    status["messages"] = public_messages(conversation["messages"])
                    status["request"] = conversation.get("request", {})
                    status["optional_offer"] = public_offer(
                        conversation.get("dialogue", {}).get("offered_options")
                    )
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
            if self.path == "/api/profile/portrait":
                from .profile import MAX_UPLOAD_BYTES, save_portrait

                if not 0 < size <= MAX_UPLOAD_BYTES:
                    raise ValueError("Choose an image smaller than 10 MB.")
                return self.send(save_portrait(ROOT, self.rfile.read(size)))
            if not 0 < size < 12000:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            if self.path == "/api/profile":
                from .profile import save_profile

                return self.send(save_profile(ROOT, body))
            if self.path == "/api/student5/start":
                from travel_agent.recommendation.service import run as student5_run

                from .live_outfits import load_session

                stamp = body.get("result")
                with STUDENT5_LOCK:
                    session = load_session(ROOT, stamp, student5_run)
                session_id = secrets.token_hex(16)
                with STUDENT5_LOCK:
                    STUDENT5_SESSIONS[session_id] = session
                return self.send({"id": session_id, **public_student5(session)})
            if self.path == "/api/student5/refine/confirm":
                session_id = body.get("id", "")
                pending_id = body.get("pending_change_id", "")
                if not re.fullmatch("[a-f0-9]{32}", session_id) or not re.fullmatch(
                    "[a-f0-9]{32}", pending_id
                ):
                    raise ValueError("Invalid refinement confirmation")
                with STUDENT5_LOCK:
                    session = STUDENT5_SESSIONS.get(session_id)
                    version = (session or {}).get("state", {}).get("request_version")
                if session is None:
                    raise ValueError("Student 5 session expired; reload the recommendations")
                candidate_job = secrets.token_hex(16)
                pending = STUDENT5_REFINEMENTS.consume_pending(
                    pending_id, session_id, version, candidate_job
                )
                if pending.job_id != candidate_job:
                    return self.send({"id": pending.job_id}, 202)
                if not ACTIVE.acquire(blocking=False):
                    STUDENT5_REFINEMENTS.release_job(pending_id, candidate_job)
                    raise ValueError("Another task is running. Please wait.")
                try:
                    directory = JOBS / candidate_job
                    directory.mkdir(parents=True)
                    write(
                        directory / "status.json",
                        {
                            "id": candidate_job,
                            "status": "running",
                            "stage": "weather",
                            "message": "Updating trip context and weather…",
                        },
                    )
                    threading.Thread(
                        target=execute_student5_refinement,
                        args=(candidate_job, session_id, pending),
                        daemon=True,
                    ).start()
                except Exception:
                    STUDENT5_REFINEMENTS.release_job(pending_id, candidate_job)
                    ACTIVE.release()
                    raise
                return self.send({"id": candidate_job}, 202)
            if self.path in {
                "/api/student5/refine",
                "/api/student5/confirm",
                "/api/student5/preview",
            }:
                from travel_agent.recommendation.service import run as student5_run
                from travel_agent.understanding.service import run as understand

                from .student5_session import (
                    confirm,
                    patch_from_message,
                    preview,
                    propose_refinement,
                    refine,
                )

                session_id = body.get("id", "")
                if not re.fullmatch("[a-f0-9]{32}", session_id):
                    raise ValueError("Invalid Student 5 session")
                with STUDENT5_LOCK:
                    session = STUDENT5_SESSIONS.get(session_id)
                if session is None:
                    raise ValueError("Student 5 session expired; reload the recommendations")
                if self.path.endswith("/refine"):
                    patch = body.get("patch")
                    if patch is None and isinstance(body.get("message"), str):
                        patch = patch_from_message(body["message"], session)
                    if not isinstance(patch, dict):
                        raise ValueError("A structured requirement change is required")
                    if patch.get("kind") == "upstream":
                        proposal = propose_refinement(session, patch["message"], understand)
                        pending = STUDENT5_REFINEMENTS.create_pending(
                            session_id,
                            session["state"]["request_version"],
                            proposal,
                        )
                        return self.send(STUDENT5_REFINEMENTS.public_plan(pending))
                    session = refine(session, patch, student5_run)
                elif self.path.endswith("/confirm"):
                    session = confirm(session, str(body.get("outfit_id", "")), student5_run)
                else:
                    session = preview(session, student5_run)
                with STUDENT5_LOCK:
                    STUDENT5_SESSIONS[session_id] = session
                response = {"id": session_id, **public_student5(session)}
                if self.path.endswith("/preview"):
                    generated = (response.get("preview") or {}).get("preview") or {}
                    reference = generated.get("output_reference")
                    if reference:
                        source = Path(reference).resolve()
                        approved = (ROOT / "artifacts/previews").resolve()
                        source.relative_to(approved)
                        content = source.read_bytes()
                        if (
                            not content.startswith(b"\x89PNG\r\n\x1a\n")
                            or len(content) > 20 * 1024 * 1024
                        ):
                            raise ValueError("Preview provider returned an invalid PNG image")
                        media = JOBS / "student5-preview-media"
                        media.mkdir(parents=True, exist_ok=True)
                        target = media / f"{session_id}.png"
                        target.write_bytes(content)
                        generated["output_reference"] = f"/student5-preview-media/{target.name}"
                return self.send(response)
            if self.path == "/api/chat":
                message = body.get("message", "")
                if not isinstance(message, str) or not 1 <= len(message.strip()) <= 2500:
                    raise ValueError("Write a message between 1 and 2,500 characters.")
                job = body.get("id")
                option_selection = False
                if job:
                    if not re.fullmatch("[a-f0-9]{32}", job):
                        raise ValueError("Invalid conversation")
                    directory = JOBS / job
                    conversation = read(directory / "conversation.json")
                    if body.get("offer_key"):
                        from .chat import public_offer

                        offer = public_offer(
                            conversation.get("dialogue", {}).get("offered_options")
                        )
                        if offer is None or offer["key"] != body["offer_key"]:
                            raise ValueError(
                                "These choices have changed. Refresh the chat and try again."
                            )
                        option_selection = True
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
                    user_message = {"role": "user", "content": message.strip()}
                    if option_selection:
                        user_message["source"] = "option_selection"
                    conversation["messages"].append(user_message)
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
                conversation_path = directory / "conversation.json"
                accepted = None
                if conversation_path.exists():
                    from .chat import accept_review

                    accepted = accept_review(read(conversation_path), reviewed_request)
                if not ACTIVE.acquire(blocking=False):
                    return self.send({"error": "Another search is running. Please wait."}, 409)
                write(directory / "confirmed_input.json", payload)
                if accepted is not None:
                    write(conversation_path, accepted)
                update(
                    job,
                    status="running",
                    stage="setup",
                    message="Preparing your search…",
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

    def do_DELETE(self):
        if not self.safe_host() or self.headers.get("X-Trip-UI") != "1":
            return self.send({"error": "Local access only"}, 403)
        if self.path != "/api/profile/portrait":
            return self.send({"error": "Not found"}, 404)
        from .profile import delete_portrait

        return self.send(delete_portrait(ROOT))


def main():
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    os.environ.setdefault("EBAY_MANAGED_ROOT", str(ROOT))
    JOBS.mkdir(parents=True, exist_ok=True)
    print(f"Local trip UI: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
