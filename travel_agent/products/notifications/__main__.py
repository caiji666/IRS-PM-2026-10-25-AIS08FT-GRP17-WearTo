"""Explicit configuration, local server and cleanup CLI."""

import argparse
import json
import logging
import os
import secrets
import sys
import time
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server


def main():
    p = argparse.ArgumentParser(description="eBay account deletion notification lifecycle")
    subs = p.add_subparsers(dest="command", required=True)
    init = subs.add_parser("init-secrets")
    init.add_argument("--output", type=Path, default=Path(".secrets/ebay-notifications.env"))
    serve = subs.add_parser("serve-local")
    serve.add_argument("--port", type=int, default=8765)
    worker = subs.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument(
        "--apply",
        action="store_true",
        help="Delete registered generated artifacts for verified events",
    )
    worker.add_argument("--watch", action="store_true")
    worker.add_argument(
        "--allow-local", action="store_true", help="HTTP loopback test transport only"
    )
    worker.add_argument("--interval", type=int, default=30)
    worker.add_argument("--max-pending-age", type=int, default=3600)
    export = subs.add_parser("export-report")
    export.add_argument("--root", type=Path, required=True)
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--target", type=Path, required=True)
    status = subs.add_parser("registry-status")
    status.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if a.command == "init-secrets":
            from cryptography.fernet import Fernet

            a.output.parent.mkdir(parents=True, exist_ok=True)
            content = {
                "EBAY_VERIFICATION_TOKEN": secrets.token_urlsafe(40),
                "EBAY_QUEUE_KEY": Fernet.generate_key().decode(),
                "EBAY_WORKER_TOKENS_JSON": json.dumps({"mac": secrets.token_urlsafe(40)}),
            }
            fd = os.open(a.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                for key, value in content.items():
                    f.write(f"{key}='{value}'\n")
            print(
                "Secrets generated locally; existing files are never overwritten. Values omitted."
            )
        elif a.command == "serve-local":
            from .service import create_app

            class QuietHandler(WSGIRequestHandler):
                def log_message(self, *args):
                    pass  # Do not log challenge values, identifiers or bearer tokens.

            print(f"Local-only listener: http://127.0.0.1:{a.port}; not a public eBay endpoint")
            with make_server(
                "127.0.0.1", a.port, create_app(), handler_class=QuietHandler
            ) as server:
                server.serve_forever()
        elif a.command == "export-report":
            from .lifecycle import export_report

            export_report(a.root, a.source, a.target)
            print("Production report copied and registered for deletion.")
        elif a.command == "registry-status":
            from .lifecycle import Registry

            registry = Registry(a.root, a.root / ".cache/ebay-lifecycle/registry.sqlite3")
            try:
                print(
                    json.dumps(
                        {
                            "registered_artifacts": registry.db.execute(
                                "SELECT count(*) FROM artifacts"
                            ).fetchone()[0]
                        }
                    )
                )
            finally:
                registry.close()
        else:
            from .worker import configured_worker

            if a.interval < 5:
                raise ValueError("interval must be at least five seconds")
            instance = configured_worker(json.loads(a.config.read_text()), a.allow_local)
            try:
                while True:
                    try:
                        result = instance.once(a.apply)
                        print(json.dumps(result), flush=True)
                        if (
                            result.get("failed_deliveries", 0)
                            or result.get("oldest_pending_age_seconds", 0) > a.max_pending_age
                        ):
                            logging.error("notification_cleanup_overdue")
                            if not a.watch:
                                return 1
                    except RuntimeError:
                        print(
                            "Worker failed; inspect authenticated queue status. Retry pending.",
                            file=sys.stderr,
                        )
                        if not a.watch:
                            return 2
                    if not a.watch:
                        break
                    time.sleep(a.interval)
            finally:
                instance.close()
        return 0
    except Exception as exc:
        print(
            f"Notification command failed ({type(exc).__name__}); secret values omitted.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
