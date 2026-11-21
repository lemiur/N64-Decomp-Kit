"""Versioned, content-addressed evidence store behind one ``Repository`` interface.

What the kit learns about a title is spread over the JSON files of a run directory and several
SQLite stores (cases, sessions, verification receipts), each with its own identity rules.  This
module is one durable, auditable place for derived evidence.  It replaces none of them:

* **blobs** by SHA-256.  A blob flagged ``rom_derived`` is kept locally and never exported;
* **records** ``(kind, key, version, payload JSON, provenance, parents, depends_on)``, append-only:
  a new payload for a key is a new version whose parents name the previous version; nothing is
  updated or deleted.  ``record_id`` commits to the content, the version and the parents (a hash
  chain).  Putting what the latest version already holds returns that version (idempotent);
* **schema versions** with ordered migrations (``MIGRATIONS``).  A store newer than this code is
  refused; an older one is migrated forward, one transaction per step, when opened writable;
* **transactions**: SQLite in rollback-journal mode (``journal_mode=DELETE``, never WAL),
  ``BEGIN IMMEDIATE`` ... ``COMMIT`` around every operation or ``transaction()`` block; an
  exception inside the block rolls all of it back, and a writer that dies leaves a journal that
  the next writable open rolls back;
* **read-only inspection** opens ``file:...?mode=ro``: a reader under SQLite's own locks (a SHARED
  lock per statement), so a live writer can neither tear a read nor leave stale cached pages; no
  journal, ``-wal`` or ``-shm`` sidecar is created.  A store with a hot journal (header magic
  written: a writer interrupted or committing) or in WAL mode is refused.  ``immutable=1`` is not
  used: it skips locking and change detection, so a writer starting after the checks could be
  read half-way;
* **dependency invalidation**: a record names the keys it depends on; ``invalidate(key)`` marks
  the record named by ``key`` and every record depending on it, transitively, stale.  A mark is
  appended, never removed: a new version of the record is what makes it current again;
* **deterministic export**: sorted canonical JSON lines without timestamps; refused while any
  blob flagged rom-derived (or record whose provenance says ``rom_derived``) would be included;
* **import** of a finished run directory's JSON files (``import_run``), reading only.

Nothing here is evidence of byte identity by itself: credit stays with the strict verifier.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

FORMAT = "n64dk.evidence.v1"
SCHEMA_VERSION = 2
DATABASE = "evidence.sqlite"
IMPORTER = "n64dk.match.evidence_store.import_run/1"
IMPORT_LIMIT = 64 << 20          # bytes: a run file above this is listed as skipped, not read


class StoreError(RuntimeError):
    """The store cannot do what was asked without risking its integrity."""


class SchemaError(StoreError):
    pass


class ExportRefused(StoreError):
    pass


def canonical(value) -> str:
    """Canonical JSON text: sorted keys, no whitespace, ASCII, no NaN; raises on non-JSON values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def handle(kind: str, key: str) -> str:
    """The dependency key that names a record: ``kind:key``."""
    return f"{kind}:{key}"


def _name(value, what: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{what} must be a non-empty string without NUL")
    return value


def content_digest(kind: str, key: str, payload, provenance: dict, depends_on) -> str:
    return sha256(canonical({"kind": kind, "key": key, "payload": payload, "provenance": provenance,
                             "depends_on": list(depends_on)}).encode())


def record_digest(content: str, version: int, parents) -> str:
    return sha256(canonical({"content": content, "version": version, "parents": list(parents)}).encode())


@dataclass(frozen=True)
class Record:
    kind: str
    key: str
    version: int
    payload: object
    provenance: dict
    parents: tuple[str, ...]
    depends_on: tuple[str, ...]
    content: str
    record_id: str

    @property
    def handle(self) -> str:
        return handle(self.kind, self.key)

    def as_dict(self) -> dict:
        return {"record_id": self.record_id, "kind": self.kind, "key": self.key, "version": self.version,
                "content": self.content, "payload": self.payload, "provenance": self.provenance,
                "parents": list(self.parents), "depends_on": list(self.depends_on)}


@runtime_checkable
class Repository(Protocol):
    """The stable interface every store implements (``MemoryRepository``, ``SQLiteRepository``)."""
    readonly: bool

    def put(self, kind: str, key: str, payload, *, provenance: dict | None = None, parents=(),
            depends_on=()) -> Record: ...

    def latest(self, kind: str, key: str) -> Record | None: ...

    def history(self, kind: str, key: str) -> list[Record]: ...

    def records(self, kind: str | None = None) -> list[Record]: ...

    def put_blob(self, data: bytes, *, rom_derived: bool = False) -> str: ...

    def get_blob(self, digest: str) -> bytes: ...

    def invalidate(self, key: str, cause: str = "") -> list[str]: ...

    def is_stale(self, kind: str, key: str) -> bool: ...

    def stale(self) -> list[str]: ...

    def transaction(self): ...

    def export_lines(self, *, omit_rom_derived: bool = False) -> list[str]: ...

    def close(self) -> None: ...


class _Store:
    """The operations, written once over a handful of storage primitives."""
    readonly = False

    # ---- primitives (implemented by each store)
    def _latest(self, kind: str, key: str) -> Record | None:
        raise NotImplementedError

    def _history(self, kind: str, key: str) -> list[Record]:
        raise NotImplementedError

    def _all(self) -> list[Record]:
        raise NotImplementedError

    def _latest_all(self, kind: str | None = None) -> list[Record]:
        raise NotImplementedError

    def _insert(self, record: Record) -> None:
        raise NotImplementedError

    def _stale(self, record: Record) -> bool:
        raise NotImplementedError

    def _add_stale(self, record: Record, cause: str) -> None:
        raise NotImplementedError

    def _stale_marks(self) -> list[tuple[str, str, int, str]]:
        raise NotImplementedError

    def _blob_row(self, digest: str) -> tuple[bytes, bool] | None:
        raise NotImplementedError

    def _blob_put(self, digest: str, data: bytes, rom_derived: bool) -> None:
        raise NotImplementedError

    def _blob_flag(self, digest: str) -> None:
        raise NotImplementedError

    def _blobs(self) -> list[tuple[str, int, bool, bytes]]:
        raise NotImplementedError

    def transaction(self):
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _writable(self) -> None:
        if self.readonly:
            raise StoreError("the store is open read-only")

    # ---- records
    def put(self, kind: str, key: str, payload, *, provenance: dict | None = None, parents=(),
            depends_on=()) -> Record:
        """Append a version of ``kind:key`` (or return the latest one when it already holds this
        content, is not stale and already names ``parents``)."""
        self._writable()
        _name(kind, "kind")
        _name(key, "key")
        payload = json.loads(canonical(payload))
        provenance = json.loads(canonical(dict(provenance or {})))
        deps = tuple(sorted({_name(dep, "dependency") for dep in depends_on}))
        extra = tuple(dict.fromkeys(_name(parent, "parent") for parent in parents))
        content = content_digest(kind, key, payload, provenance, deps)
        with self.transaction():
            previous = self._latest(kind, key)
            if previous is not None and previous.content == content and set(extra) <= set(previous.parents) \
                    and not self._stale(previous):
                return previous
            chain = (previous.record_id,) if previous is not None else ()
            chain += tuple(parent for parent in extra if parent not in chain)
            version = previous.version + 1 if previous is not None else 1
            record = Record(kind, key, version, payload, provenance, chain, deps, content,
                            record_digest(content, version, chain))
            self._insert(record)
            return record

    def latest(self, kind: str, key: str) -> Record | None:
        return self._latest(kind, key)

    def history(self, kind: str, key: str) -> list[Record]:
        return self._history(kind, key)

    def records(self, kind: str | None = None) -> list[Record]:
        """The latest version of every key (of one kind), sorted by kind and key."""
        return self._latest_all(kind)

    # ---- invalidation
    def invalidate(self, key: str, cause: str = "") -> list[str]:
        """Mark stale the record named by ``key`` (a ``kind:key`` handle) and every record that
        depends on ``key``, transitively.  Returns the handles marked, in breadth-first order."""
        self._writable()
        _name(key, "key")
        why = cause or f"invalidated {key}"
        with self.transaction():
            latest = self._latest_all()
            by_handle = {record.handle: record for record in latest}
            dependents: dict[str, list[Record]] = {}
            for record in latest:
                for dep in record.depends_on:
                    dependents.setdefault(dep, []).append(record)
            marked: list[str] = []
            if key in by_handle:
                marked.append(key)
                if not self._stale(by_handle[key]):
                    self._add_stale(by_handle[key], why)
            seen, queue = {key}, [key]
            while queue:
                current = queue.pop(0)
                for record in sorted(dependents.get(current, ()), key=lambda r: r.handle):
                    if record.handle in seen:
                        continue
                    seen.add(record.handle)
                    if not self._stale(record):
                        self._add_stale(record, why)
                    marked.append(record.handle)
                    queue.append(record.handle)
            return marked

    def is_stale(self, kind: str, key: str) -> bool:
        record = self._latest(kind, key)
        return record is not None and self._stale(record)

    def stale(self) -> list[str]:
        return [record.handle for record in self._latest_all() if self._stale(record)]

    # ---- blobs
    def put_blob(self, data: bytes, *, rom_derived: bool = False) -> str:
        """Store bytes by digest.  ``rom_derived`` can be raised on a stored blob, never lowered."""
        self._writable()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("a blob is bytes")
        data = bytes(data)
        digest = sha256(data)
        with self.transaction():
            row = self._blob_row(digest)
            if row is None:
                self._blob_put(digest, data, bool(rom_derived))
            elif rom_derived and not row[1]:
                self._blob_flag(digest)
        return digest

    def get_blob(self, digest: str) -> bytes:
        row = self._blob_row(digest)
        if row is None:
            raise KeyError(digest)
        if sha256(row[0]) != digest:
            raise StoreError(f"corrupt blob {digest}")
        return row[0]

    def blob_is_rom_derived(self, digest: str) -> bool:
        row = self._blob_row(digest)
        if row is None:
            raise KeyError(digest)
        return row[1]

    # ---- export
    def export_lines(self, *, omit_rom_derived: bool = False) -> list[str]:
        """Every blob, record version and stale mark as sorted canonical JSON lines.  Raises
        ``ExportRefused`` while rom-derived content exists, unless ``omit_rom_derived`` (then its
        digest and size are exported, never its bytes or payload)."""
        blobs = self._blobs()
        records = self._all()
        rom_blobs = sorted(digest for digest, _size, flag, _data in blobs if flag)
        rom_records = sorted(f"{r.handle}@{r.version}" for r in records if (r.provenance or {}).get("rom_derived"))
        if (rom_blobs or rom_records) and not omit_rom_derived:
            raise ExportRefused(f"refusing to export rom-derived content: {len(rom_blobs)} blob(s), "
                                f"{len(rom_records)} record version(s) (first: {(rom_blobs + rom_records)[0]})")
        marks = self._stale_marks()
        stale = {(kind, key, version) for kind, key, version, _cause in marks}
        lines = [canonical({"type": "header", "format": FORMAT, "schema": SCHEMA_VERSION})]
        for digest, size, flag, data in sorted(blobs, key=lambda row: row[0]):
            row = {"type": "blob", "digest": digest, "size": size, "rom_derived": flag}
            if not flag:
                row["data"] = base64.b64encode(data).decode("ascii")
            lines.append(canonical(row))
        for record in sorted(records, key=lambda r: (r.kind, r.key, r.version)):
            row = {"type": "record", **record.as_dict(),
                   "stale": (record.kind, record.key, record.version) in stale}
            if (record.provenance or {}).get("rom_derived"):
                row["payload"], row["omitted"] = None, "rom-derived"
            lines.append(canonical(row))
        for kind, key, version, cause in sorted(marks):
            lines.append(canonical({"type": "stale", "kind": kind, "key": key, "version": version, "cause": cause}))
        return lines

    def export(self, path: str | Path | None = None, *, omit_rom_derived: bool = False) -> str:
        text = "\n".join(self.export_lines(omit_rom_derived=omit_rom_derived)) + "\n"
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".export-", dir=target.parent)
            try:
                with os.fdopen(fd, "w", encoding="ascii") as stream:
                    stream.write(text)
                os.replace(name, target)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(name)
        return text


class MemoryRepository(_Store):
    """The same interface in memory (tests, a world model that is never saved).  Transactions are
    real: an undo log reverses every change of a block that raises."""

    def __init__(self):
        self.readonly = False
        self._versions: dict[tuple[str, str], list[Record]] = {}
        self._blob_data: dict[str, list] = {}
        self._marks: list[tuple[str, str, int, str]] = []
        self._mark_set: set[tuple[str, str, int]] = set()
        self._undo: list | None = None
        self._depth = 0

    def _log(self, undo) -> None:
        if self._undo is not None:
            self._undo.append(undo)

    @contextlib.contextmanager
    def transaction(self) -> Iterator["MemoryRepository"]:
        self._writable()
        if self._depth:
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
            return
        self._depth, self._undo = 1, []
        try:
            yield self
        except BaseException:
            for undo in reversed(self._undo):
                undo()
            raise
        finally:
            self._depth, self._undo = 0, None

    def _latest(self, kind, key):
        rows = self._versions.get((kind, key))
        return rows[-1] if rows else None

    def _history(self, kind, key):
        return list(self._versions.get((kind, key), ()))

    def _all(self):
        return [record for rows in self._versions.values() for record in rows]

    def _latest_all(self, kind=None):
        return sorted((rows[-1] for (k, _key), rows in self._versions.items() if rows and (kind is None or k == kind)),
                      key=lambda r: (r.kind, r.key))

    def _insert(self, record):
        rows = self._versions.setdefault((record.kind, record.key), [])
        rows.append(record)
        self._log(rows.pop)

    def _stale(self, record):
        return (record.kind, record.key, record.version) in self._mark_set

    def _add_stale(self, record, cause):
        mark = (record.kind, record.key, record.version)
        self._marks.append((*mark, cause))
        self._mark_set.add(mark)

        def undo():
            self._marks.pop()
            self._mark_set.discard(mark)
        self._log(undo)

    def _stale_marks(self):
        return sorted(self._marks)

    def _blob_row(self, digest):
        row = self._blob_data.get(digest)
        return (row[0], row[1]) if row else None

    def _blob_put(self, digest, data, rom_derived):
        self._blob_data[digest] = [data, rom_derived]
        self._log(lambda: self._blob_data.pop(digest, None))

    def _blob_flag(self, digest):
        row = self._blob_data[digest]
        row[1] = True
        self._log(lambda: row.__setitem__(1, False))

    def _blobs(self):
        return [(digest, len(row[0]), row[1], row[0]) for digest, row in sorted(self._blob_data.items())]


# --------------------------------------------------------------------------- SQLite

# version -> (note, statements).  A migration is a list of statements run in one transaction
# together with the row that records it; never edit a published step, add the next one.
MIGRATIONS: dict[int, tuple[str, tuple[str, ...]]] = {
    1: ("blobs and append-only versioned records", (
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_ns INTEGER NOT NULL, note TEXT NOT NULL)",
        "CREATE TABLE blobs (digest TEXT PRIMARY KEY, size INTEGER NOT NULL, rom_derived INTEGER NOT NULL DEFAULT 0, "
        "data BLOB NOT NULL)",
        "CREATE TABLE records (record_id TEXT PRIMARY KEY, kind TEXT NOT NULL, key TEXT NOT NULL, "
        "version INTEGER NOT NULL, content TEXT NOT NULL, payload TEXT NOT NULL, provenance TEXT NOT NULL, "
        "parents TEXT NOT NULL, created_ns INTEGER NOT NULL, UNIQUE (kind, key, version))",
    )),
    2: ("record dependencies and invalidation marks", (
        "ALTER TABLE records ADD COLUMN depends_on TEXT NOT NULL DEFAULT '[]'",
        "CREATE TABLE record_deps (record_id TEXT NOT NULL, dep TEXT NOT NULL, PRIMARY KEY (record_id, dep))",
        "CREATE INDEX record_deps_dep ON record_deps (dep)",
        "CREATE TABLE stale (kind TEXT NOT NULL, key TEXT NOT NULL, version INTEGER NOT NULL, cause TEXT NOT NULL, "
        "marked_ns INTEGER NOT NULL, PRIMARY KEY (kind, key, version))",
    )),
}
assert sorted(MIGRATIONS) == list(range(1, SCHEMA_VERSION + 1))


def schema_version(db: sqlite3.Connection) -> int:
    """The highest migration recorded in the store (0: an empty or foreign database)."""
    row = db.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'").fetchone()
    if row is None:
        return 0
    value = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    return int(value or 0)


JOURNAL_MAGIC = bytes.fromhex("d9d505f920a163d7")


def _hot_journal(path: Path) -> bool:
    """Whether a rollback journal can hold pages a reader must not miss.  SQLite writes the header
    magic only when it syncs the journal, before the first change reaches the database file; a
    journal without it (e.g. a writer killed before that sync, or a PERSIST-mode journal zeroed at
    commit) leaves the database file as committed, and its rollback is a no-op.  This says nothing
    about a live writer, which may sync and write the file a moment later: the reader's SHARED lock
    (``open_readonly``) covers that, as SQLite's RESERVED-lock condition does for hot journals."""
    try:
        with path.open("rb") as stream:
            return stream.read(8) == JOURNAL_MAGIC
    except OSError:
        return True                   # unreadable: assume the worst


def open_readonly(path: str | Path, *, timeout: float = 60.0) -> sqlite3.Connection:
    """A sidecar-free read-only connection (``mode=ro``, locking), or StoreError when the file
    cannot be read safely: a hot ``-journal`` (an interrupted or committing writer; a reader
    cannot roll it back), a non-empty ``-wal``, or a WAL-mode header (committed pages may live
    outside the main file).  Every read takes SQLite's SHARED lock, so a writer that starts
    after these checks waits for it (and a read waits up to ``timeout`` seconds for a commit in
    progress) instead of being read half-way; a hot journal that appears later is refused by
    SQLite itself (``SQLITE_READONLY_ROLLBACK``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    for suffix in ("-journal", "-wal"):
        side = Path(f"{path}{suffix}")
        if suffix == "-journal" and side.exists() and not _hot_journal(side):
            continue                  # no magic: the file is as committed; a live writer is the lock's job
        if side.exists() and side.stat().st_size > 0:
            raise StoreError(f"{path.name}{suffix} exists: a writer is active or was interrupted; open the store "
                             "writable once (an interrupted transaction is rolled back) before inspecting it")
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) == 20 and header[18:20] == b"\x02\x02":
        raise StoreError(f"{path.name} is in WAL mode; open it writable once to return it to a rollback journal")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=timeout,
                           isolation_level=None, check_same_thread=False)


class SQLiteRepository(_Store):
    """The store in one SQLite file.  ``readonly`` inspection writes nothing anywhere.  ``schema``
    (writable only) stops the migrations at an older version: for migration tests."""

    def __init__(self, path: str | Path, *, readonly: bool = False, schema: int = SCHEMA_VERSION):
        self.path = Path(path)
        self.readonly = bool(readonly)
        self._depth = 0
        if self.readonly:
            self.db = open_readonly(self.path)
            try:
                self.schema = schema_version(self.db)
                if self.schema != SCHEMA_VERSION:
                    raise SchemaError(f"{self.path.name} has schema {self.schema}, this code reads {SCHEMA_VERSION}"
                                      + ("; open it writable once to migrate it" if self.schema < SCHEMA_VERSION
                                         else ": a newer kit wrote it"))
            except BaseException:
                self.db.close()
                raise
            return
        if not 1 <= schema <= SCHEMA_VERSION:
            raise ValueError(f"schema must be 1..{SCHEMA_VERSION}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60, isolation_level=None, check_same_thread=False)
        try:
            mode = self.db.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            if str(mode).lower() != "delete":
                raise StoreError(f"{self.path.name}: journal mode stays {mode} (another connection holds it?)")
            self.db.execute("PRAGMA synchronous=FULL")
            self.schema = schema_version(self.db)
            if self.schema > SCHEMA_VERSION:
                raise SchemaError(f"{self.path.name} has schema {self.schema}: a newer kit wrote it; this code "
                                  f"knows {SCHEMA_VERSION}")
            self.migrate(schema)
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    def migrate(self, target: int = SCHEMA_VERSION) -> list[int]:
        """Apply the missing migrations up to ``target``, each in its own transaction."""
        self._writable()
        applied = []
        for version in range(self.schema + 1, target + 1):
            note, statements = MIGRATIONS[version]
            with self.transaction():
                for statement in statements:
                    self.db.execute(statement)
                self.db.execute("INSERT INTO schema_version VALUES (?, ?, ?)", (version, time.time_ns(), note))
            self.schema = version
            applied.append(version)
        return applied

    @contextlib.contextmanager
    def transaction(self) -> Iterator["SQLiteRepository"]:
        """One ``BEGIN IMMEDIATE`` ... ``COMMIT``; an exception rolls the whole block back.  A
        nested block joins the outer transaction."""
        self._writable()
        if self._depth:
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
            return
        self.db.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield self
            self.db.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                self.db.execute("ROLLBACK")
            raise
        finally:
            self._depth = 0

    # ---- primitives
    def _select(self) -> str:
        deps = "depends_on" if self.schema >= 2 else "'[]'"
        return f"SELECT record_id, kind, key, version, content, payload, provenance, parents, {deps} FROM records"

    @staticmethod
    def _record(row) -> Record:
        return Record(row[1], row[2], int(row[3]), json.loads(row[5]), json.loads(row[6]), tuple(json.loads(row[7])),
                      tuple(json.loads(row[8])), row[4], row[0])

    def _latest(self, kind, key):
        row = self.db.execute(self._select() + " WHERE kind = ? AND key = ? ORDER BY version DESC LIMIT 1",
                              (kind, key)).fetchone()
        return self._record(row) if row else None

    def _history(self, kind, key):
        rows = self.db.execute(self._select() + " WHERE kind = ? AND key = ? ORDER BY version", (kind, key))
        return [self._record(row) for row in rows]

    def _all(self):
        return [self._record(row) for row in self.db.execute(self._select() + " ORDER BY kind, key, version")]

    def _latest_all(self, kind=None):
        query = (self._select() + " AS r WHERE r.version = (SELECT MAX(i.version) FROM records AS i "
                 "WHERE i.kind = r.kind AND i.key = r.key)")
        if kind is not None:
            rows = self.db.execute(query + " AND r.kind = ? ORDER BY r.kind, r.key", (kind,))
        else:
            rows = self.db.execute(query + " ORDER BY r.kind, r.key")
        return [self._record(row) for row in rows]

    def _insert(self, record):
        values = (record.record_id, record.kind, record.key, record.version, record.content,
                  canonical(record.payload), canonical(record.provenance), canonical(list(record.parents)),
                  time.time_ns())
        if self.schema < 2:
            if record.depends_on:
                raise SchemaError("record dependencies need schema 2")
            self.db.execute("INSERT INTO records (record_id, kind, key, version, content, payload, provenance, "
                            "parents, created_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
            return
        self.db.execute("INSERT INTO records (record_id, kind, key, version, content, payload, provenance, parents, "
                        "created_ns, depends_on) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        values + (canonical(list(record.depends_on)),))
        self.db.executemany("INSERT INTO record_deps VALUES (?, ?)",
                            [(record.record_id, dep) for dep in record.depends_on])

    def _stale(self, record):
        if self.schema < 2:
            return False
        return self.db.execute("SELECT 1 FROM stale WHERE kind = ? AND key = ? AND version = ?",
                               (record.kind, record.key, record.version)).fetchone() is not None

    def _add_stale(self, record, cause):
        self.db.execute("INSERT INTO stale VALUES (?, ?, ?, ?, ?)",
                        (record.kind, record.key, record.version, cause, time.time_ns()))

    def _stale_marks(self):
        if self.schema < 2:
            return []
        return [(row[0], row[1], int(row[2]), row[3]) for row in
                self.db.execute("SELECT kind, key, version, cause FROM stale ORDER BY kind, key, version")]

    def _blob_row(self, digest):
        row = self.db.execute("SELECT data, rom_derived FROM blobs WHERE digest = ?", (digest,)).fetchone()
        return (bytes(row[0]), bool(row[1])) if row else None

    def _blob_put(self, digest, data, rom_derived):
        self.db.execute("INSERT INTO blobs VALUES (?, ?, ?, ?)", (digest, len(data), int(rom_derived), data))

    def _blob_flag(self, digest):
        self.db.execute("UPDATE blobs SET rom_derived = 1 WHERE digest = ?", (digest,))

    def _blobs(self):
        return [(row[0], int(row[1]), bool(row[2]), bytes(row[3])) for row in
                self.db.execute("SELECT digest, size, rom_derived, data FROM blobs ORDER BY digest")]


# --------------------------------------------------------------------------- run directories

RUN_FILES = ("solver/invocation.json", "report.json", "discovery/discovery_audit.json", "discovery/overlays.json")
_NOT_IMAGES = {"solver", "discovery", "state", "queue", "usage", "small_data"}


def run_files(run: str | Path) -> list[str]:
    """The JSON a ``decompile-run`` directory keeps, relative to it: the invocation, the report,
    the discovery audit and repairs, the solver's files, each image's checkpointed stage results
    and toolchain regions, and the rounds log.  Bounded: two directory levels, no case store."""
    run = Path(run)
    found = [rel for rel in RUN_FILES if (run / rel).is_file()]
    discovery = run / "discovery"
    if discovery.is_dir():
        found += sorted(f"discovery/{p.name}" for p in discovery.iterdir()
                        if p.is_file() and p.name.startswith("repair_") and p.suffix == ".json")
    solver = run / "solver"
    if solver.is_dir():
        found += sorted(f"solver/{p.name}" for p in solver.iterdir()
                        if p.is_file() and p.suffix == ".json" and f"solver/{p.name}" not in found)
    for folder in sorted(p for p in run.iterdir() if p.is_dir() and p.name not in _NOT_IMAGES):
        found += sorted(f"{folder.name}/{p.name}" for p in folder.iterdir()
                        if p.is_file() and ((p.name.startswith("stage_") and p.name.endswith(".json")
                                             and not p.name.endswith(".meta.json"))
                                            or p.name == "toolchain_regions.json"))
    if (run / "rounds.jsonl").is_file():
        found.append("rounds.jsonl")
    return found


def import_run(repo: Repository, run: str | Path, *, label: str | None = None, limit: int = IMPORT_LIMIT) -> dict:
    """Import a run directory's JSON as ``run-file`` records (key ``<label>/<relative path>``),
    without writing to the run directory.  Each record depends on ``file:<label>/<path>`` and, but
    for the invocation itself, on the invocation's record, so ``invalidate`` of either marks it
    stale.  Importing an unchanged run again adds nothing (records are content-addressed)."""
    run = Path(run)
    if not (run / "report.json").is_file() and not (run / "solver" / "invocation.json").is_file():
        raise FileNotFoundError(f"{run.name}: not a decompile-run directory (no report.json, no solver/invocation.json)")
    label = _name(label or run.resolve().name, "label")
    invocation = handle("run-file", f"{label}/solver/invocation.json")
    has_invocation = (run / "solver" / "invocation.json").is_file()
    imported, skipped = [], {}
    with repo.transaction():
        for rel in run_files(run):
            path = run / rel
            size = path.stat().st_size
            if size > limit:
                skipped[rel] = f"{size} bytes, above the import limit of {limit}"
                continue
            raw = path.read_bytes()
            try:
                if rel.endswith(".jsonl"):
                    payload, cut = [], 0
                    for line in raw.decode("utf-8").splitlines():
                        if not line.strip():
                            continue
                        try:
                            payload.append(json.loads(line))
                        except ValueError:
                            cut += 1                       # a line cut by an interruption
                else:
                    payload, cut = json.loads(raw), 0
            except ValueError as error:                    # UnicodeDecodeError is a ValueError
                skipped[rel] = f"unreadable: {type(error).__name__}"
                continue
            deps = [f"file:{label}/{rel}"] + ([invocation] if has_invocation and rel != "solver/invocation.json" else [])
            provenance = {"importer": IMPORTER, "run": label, "file": rel, "sha256": sha256(raw), "bytes": len(raw)}
            if cut:
                provenance["unreadable_lines"] = cut
            try:
                record = repo.put("run-file", f"{label}/{rel}", payload, provenance=provenance, depends_on=deps)
            except ValueError as error:                    # NaN or infinity: not canonical JSON
                skipped[rel] = f"not canonical JSON: {error}"[:200]
                continue
            imported.append({"file": rel, "record": record.record_id, "version": record.version})
    return {"run": label, "imported": imported, "skipped": skipped}
