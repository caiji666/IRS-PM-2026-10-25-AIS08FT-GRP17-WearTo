"""Durable SQLite queue with per-consumer leases; identity payloads encrypted at rest."""

import hashlib
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from cryptography.fernet import Fernet


def identities(data):
    result = {}
    for field in ("username", "userId", "eiasToken"):
        value = data.get(field)
        if value is not None:
            if not isinstance(value, str) or not 1 <= len(value) <= 2048:
                raise ValueError("invalid account identifier")
            result[field] = value
    if not result:
        raise ValueError("notification requires account identifiers")
    return result


@contextmanager
def database(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    os.chmod(path, 0o600)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA secure_delete=ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


class Events:
    def __init__(self, path, key, consumers):
        self.path, self.cipher = path, Fernet(key.encode())
        self.consumers = tuple(consumers)
        if not self.consumers:
            raise ValueError("at least one consumer required")
        with database(path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload BLOB,
                    received REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS deliveries (
                    event TEXT, consumer TEXT, state TEXT, lease TEXT, until REAL,
                    attempts INTEGER NOT NULL DEFAULT 0, error TEXT,
                    PRIMARY KEY(event, consumer));
            """)

    def put(self, message):
        if message.get("metadata", {}).get("topic") != "MARKETPLACE_ACCOUNT_DELETION":
            raise ValueError("unsupported notification topic")
        notification = message.get("notification", {})
        event_id = notification.get("notificationId")
        if not isinstance(event_id, str) or not 1 <= len(event_id) <= 256:
            raise ValueError("invalid notification ID")
        data = identities(notification.get("data", {}))
        encoded = json.dumps(data, sort_keys=True).encode()
        fingerprint = hashlib.sha256(encoded).hexdigest()
        with database(self.path) as db:
            previous = db.execute(
                "SELECT fingerprint FROM events WHERE id=?", (event_id,)
            ).fetchone()
            if previous:
                if previous[0] != fingerprint:
                    raise ValueError("conflicting duplicate notification")
                return
            db.execute(
                "INSERT INTO events VALUES (?,?,?,?)",
                (event_id, fingerprint, self.cipher.encrypt(encoded), time.time()),
            )
            for consumer in self.consumers:
                db.execute(
                    "INSERT INTO deliveries(event,consumer,state,until) VALUES (?,?,?,0)",
                    (event_id, consumer, "pending"),
                )

    def claim(self, consumer):
        if consumer not in self.consumers:
            raise ValueError("unknown consumer")
        with database(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT d.*, e.payload FROM deliveries d JOIN events e ON e.id=d.event
                WHERE d.consumer=? AND d.state!='done' AND d.until<=?
                ORDER BY e.received LIMIT 1""",
                (consumer, time.time()),
            ).fetchone()
            if not row:
                return None
            lease = secrets.token_urlsafe(32)
            db.execute(
                "UPDATE deliveries SET state='leased',lease=?,until=?,attempts=attempts+1 "
                "WHERE event=? AND consumer=?",
                (lease, time.time() + 300, row["event"], consumer),
            )
            return {
                "event_id": row["event"],
                "lease": lease,
                "identifiers": json.loads(self.cipher.decrypt(row["payload"])),
            }

    def finish(self, consumer, event_id, lease, success):
        with database(self.path) as db:
            row = db.execute(
                "SELECT * FROM deliveries WHERE event=? AND consumer=?", (event_id, consumer)
            ).fetchone()
            if (
                not row
                or not secrets.compare_digest(row["lease"] or "", lease)
                or row["until"] < time.time()
            ):
                raise ValueError("invalid/expired lease")
            if success:
                db.execute(
                    "UPDATE deliveries SET state='done',error=NULL WHERE event=? AND consumer=?",
                    (event_id, consumer),
                )
                pending = db.execute(
                    "SELECT count(*) FROM deliveries WHERE event=? AND state!='done'", (event_id,)
                ).fetchone()[0]
                if not pending:
                    db.execute("UPDATE events SET payload=NULL WHERE id=?", (event_id,))
            else:
                db.execute(
                    "UPDATE deliveries SET state='pending',until=?,error='cleanup_failed' "
                    "WHERE event=? AND consumer=?",
                    (
                        time.time() + min(3600, 30 * 2 ** min(row["attempts"], 7)),
                        event_id,
                        consumer,
                    ),
                )

    def health(self):
        with database(self.path) as db:
            counts = dict(
                db.execute("SELECT state,count(*) FROM deliveries GROUP BY state").fetchall()
            )
            oldest = db.execute(
                "SELECT min(e.received) FROM events e JOIN deliveries d ON e.id=d.event "
                "WHERE d.state!='done'"
            ).fetchone()[0]
            failures = db.execute(
                "SELECT count(*) FROM deliveries WHERE error IS NOT NULL"
            ).fetchone()[0]
            expired = db.execute(
                "SELECT count(*) FROM deliveries WHERE state='leased' AND until<=?",
                (time.time(),),
            ).fetchone()[0]
        return {
            "failed_deliveries": failures,
            "expired_leases": expired,
            "queue_counts": counts,
            "oldest_pending_age_seconds": round(time.time() - oldest) if oldest else 0,
        }
