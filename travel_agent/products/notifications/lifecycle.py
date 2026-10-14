"""Local managed-artifact registry. Remote events contain identities, never paths."""

import fcntl
import hashlib
import hmac
import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def sellers(row):
    seller = row.get("seller") or {}
    return {
        field: seller[field]
        for field in ("username", "userId", "eiasToken")
        if isinstance(seller.get(field), str) and seller[field]
    }


class DeletedAccount(RuntimeError):
    pass


@contextmanager
def lifecycle_lock(root):
    directory = Path(root).resolve() / ".cache/ebay-lifecycle"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "operation.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class Registry:
    def __init__(self, root, database):
        if Path(database).is_symlink():
            raise ValueError("registry cannot be a symlink")
        self.input_root = Path(root).absolute()
        self.root, self.path = Path(root).resolve(), Path(database).resolve()
        self.path.relative_to(self.root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("registry cannot be a symlink")
        self.db = sqlite3.connect(self.path)
        os.chmod(self.path, 0o600)
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS artifacts (path TEXT PRIMARY KEY, kind TEXT);
            CREATE TABLE IF NOT EXISTS accounts (
                artifact TEXT, fingerprint TEXT, PRIMARY KEY(artifact,fingerprint));
            CREATE TABLE IF NOT EXISTS blocked (fingerprint TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS completed (event TEXT PRIMARY KEY);
        """)
        self.db.execute("INSERT OR IGNORE INTO config VALUES ('salt',?)", (os.urandom(32).hex(),))
        self.db.execute("INSERT OR IGNORE INTO config VALUES ('generation','0')")
        self.db.commit()
        self.salt = bytes.fromhex(
            self.db.execute("SELECT value FROM config WHERE key='salt'").fetchone()[0]
        )

    def close(self):
        self.db.close()

    def fingerprints(self, identifiers):
        # Cross-match immutable IDs in username (eBay may return one in that field).
        return {
            hmac.new(self.salt, v.encode(), hashlib.sha256).hexdigest()
            for v in identifiers.values()
        }

    def safe(self, relative):
        path = self.root / relative
        path.resolve().relative_to(self.root)
        parts = Path(relative).parts
        if not parts or ".." in parts or Path(relative).is_absolute():
            raise ValueError("unsafe artifact path")
        for i in range(1, len(parts) + 1):
            if self.root.joinpath(*parts[:i]).is_symlink():
                raise ValueError("artifact symlinks forbidden")
        # Only explicit generated output/cache areas; never wardrobe originals or model weights.
        if parts[0] not in {"runs", "artifacts"} and parts[:2] != (".cache", "embeddings"):
            raise ValueError("artifact must be generated output or embedding cache")
        if len(parts) < 2 or (parts[0] == "runs" and parts[1] == "wardrobe"):
            raise ValueError("artifact scope is too broad/protected")
        if path == self.path or path in self.path.parents:
            raise ValueError("artifact cannot contain lifecycle registry")
        return path

    def relative(self, path):
        absolute = Path(path).absolute()
        try:
            return str(absolute.relative_to(self.root))
        except ValueError:
            return str(absolute.relative_to(self.input_root))

    def register(self, path, kind="run", identifiers=None):
        relative = self.relative(path)
        self.safe(relative)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?)", (relative, kind))
            for fingerprint in self.fingerprints(identifiers or {}):
                if self.db.execute(
                    "SELECT 1 FROM blocked WHERE fingerprint=?", (fingerprint,)
                ).fetchone():
                    raise DeletedAccount("account previously deleted; do not persist its data")
                self.db.execute(
                    "INSERT OR IGNORE INTO accounts VALUES (?,?)", (relative, fingerprint)
                )

    def observe(self, run, row):
        self.register(run, identifiers=sellers(row))

    def generation(self):
        return int(self.db.execute("SELECT value FROM config WHERE key='generation'").fetchone()[0])

    def cleanup(self, event):
        with lifecycle_lock(self.root):
            return self._cleanup(event)

    def _cleanup(self, event):
        from .store import identities

        event_id = event["event_id"]
        fingerprints = self.fingerprints(identities(event["identifiers"]))
        if self.db.execute("SELECT 1 FROM completed WHERE event=?", (event_id,)).fetchone():
            return {"status": "already_completed", "removed_artifacts": 0}
        # Conservative global invalidation prevents retaining records with incomplete seller IDs,
        # copies in aggregate reports, and shared content-addressed caches.
        artifacts = list(self.db.execute("SELECT path FROM artifacts"))
        paths = [self.safe(row[0]) for row in artifacts]  # Preflight ALL before mutation.
        with self.db:
            for fingerprint in fingerprints:
                self.db.execute("INSERT OR IGNORE INTO blocked VALUES (?)", (fingerprint,))
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        with self.db:
            self.db.execute(
                "UPDATE config SET value=CAST(value AS INTEGER)+1 WHERE key='generation'"
            )
            self.db.execute("DELETE FROM accounts")
            self.db.execute("DELETE FROM artifacts")
            self.db.execute("INSERT INTO completed VALUES (?)", (event_id,))
        return {"status": "completed", "removed_artifacts": len(paths)}


def production_registry(output_dir, encoder):
    root = os.environ.get("EBAY_MANAGED_ROOT")
    if not root:
        raise ValueError(
            "Production persistence requires EBAY_MANAGED_ROOT and a running lifecycle worker"
        )
    registry = Registry(root, Path(root) / ".cache/ebay-lifecycle/registry.sqlite3")
    registry.register(output_dir)
    cache = getattr(encoder, "cache_dir", None)
    if cache is not None:
        registry.register(cache, "embedding_cache")
    return registry


def export_report(root, source, target):
    """Copy a downstream report under the same lock as deletion and registration."""
    with lifecycle_lock(root):
        registry = Registry(root, Path(root) / ".cache/ebay-lifecycle/registry.sqlite3")
        try:
            source = registry.safe(registry.relative(source))
            parent = registry.relative(source.parent)
            if not registry.db.execute(
                "SELECT 1 FROM artifacts WHERE path=?", (parent,)
            ).fetchone():
                raise ValueError("source must belong to a registered production run")
            result = json.loads(source.read_text())
            if (
                result.get("environment") != "production"
                or result.get("lifecycle_generation") != registry.generation()
            ):
                raise DeletedAccount("source is not a current production report")
            target = registry.safe(registry.relative(target))
            if target.suffix != ".json":
                raise ValueError("export target must be JSON")
            registry.register(target, "export")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        finally:
            registry.close()


def save_report(result, target):
    target = Path(target)
    if result.get("environment") == "production":
        root = os.environ["EBAY_MANAGED_ROOT"]
        with lifecycle_lock(root):
            registry = Registry(root, Path(root) / ".cache/ebay-lifecycle/registry.sqlite3")
            try:
                if result.get("lifecycle_generation") != registry.generation():
                    raise DeletedAccount("result invalidated by deletion; rerun retrieval")
                registry.register(target, "report")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            finally:
                registry.close()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
