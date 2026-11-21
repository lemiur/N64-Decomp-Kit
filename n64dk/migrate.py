"""Versions, formats and migration (product item 25).

**One version.**  ``n64dk.__version__`` is the release version; ``pyproject.toml`` must say the
same (``version_info`` checks it; a release that disagrees is not cut).  Versions are semantic:
MAJOR changes when a public API or a stored format can no longer be read, MINOR when something is
added, PATCH for fixes (``bump``).

**Every stored format in one registry** (``FORMATS``): what it is, where it lives, the key that
carries its version (or that it has none and is read by shape), the current value (read from the
module that writes it, so the registry cannot drift), who reads it, and what happens to an older or
newer one.  Each format gets one of three answers:

* ``read-old``: older data is read as it is (a tolerant reader, or the format never changed);
* ``migrate``: a defined, atomic, backed-up step brings it to the current version;
* ``refuse``: the kit says why and what to do - never a silent misreading.  Data written by a newer
  kit is always refused.

``check(path)`` inspects a run directory, a content cache, a store (SQLite) or a versioned JSON
document and reports every item; ``migrate(path)`` applies the defined steps (``--check`` is the
dry run).  Read-only inspection opens SQLite ``mode=ro&immutable=1`` when no write-ahead log is
present (``mode=ro`` when one is: a writer owns the sidecars), so a check writes nothing.

``changelog``: release notes from ``git log`` (read-only), grouped under the version.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPORT_SCHEMA = "n64dk.migrate/1"
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")


# --------------------------------------------------------------------------- the version

def version() -> str:
    from . import __version__
    return __version__


def parse_semver(text: str) -> tuple[int, int, int]:
    match = SEMVER.match(str(text))
    if not match:
        raise ValueError(f"{text!r} is not a semantic version (MAJOR.MINOR.PATCH)")
    return int(match[1]), int(match[2]), int(match[3])


def bump(text: str, part: str) -> str:
    major, minor, patch = parse_semver(text)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError("part is major, minor or patch")


def pyproject_version(root: Path | None = None) -> str | None:
    import tomllib
    path = (Path(root) if root else Path(__file__).resolve().parents[1]) / "pyproject.toml"
    if not path.is_file():
        return None
    return (tomllib.loads(path.read_text()).get("project") or {}).get("version")


def version_info(root: Path | None = None) -> dict:
    package, declared = version(), pyproject_version(root)
    problems = []
    try:
        parse_semver(package)
    except ValueError as error:
        problems.append(str(error))
    if declared is not None and declared != package:
        problems.append(f"pyproject.toml says {declared}, n64dk.__version__ says {package}")
    return {"version": package, "pyproject": declared, "consistent": not problems, "problems": problems}


# --------------------------------------------------------------------------- the format registry

@dataclass(frozen=True)
class Format:
    name: str
    kind: str                 # "json", "jsonl", "sqlite", "directory"
    where: str                # where it lives
    key: str | None           # the field holding its version (None: none; read by shape)
    current: str | None       # "module:ATTRIBUTE" holding the current value, or a literal "=value"
    old: str                  # "read-old", "migrate" or "refuse"
    note: str

    def current_value(self):
        if self.current is None:
            return None
        if self.current.startswith("="):
            literal = self.current[1:]
            return int(literal) if literal.isdigit() else literal
        module, _sep, attribute = self.current.partition(":")
        try:
            return getattr(importlib.import_module(module), attribute)
        except (ImportError, AttributeError):
            return None

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "where": self.where, "key": self.key,
                "current": self.current_value(), "old": self.old, "note": self.note}


FORMATS: tuple[Format, ...] = (
    Format("run.invocation", "json", "<run>/solver/invocation.json", "mode_settings (shape)", "=2", "migrate",
           "v0 (before effort modes: no effort) is resumed as normal; v1 (effort, no mode_settings) migrates to v2 "
           "by recording the starting mode's settings, exactly as --resume does"),
    Format("run.report", "json", "<run>/report.json", None, None, "read-old",
           "unversioned; decompile-status and decompile-receipt read every key tolerantly"),
    Format("run.rounds", "jsonl", "<run>/rounds.jsonl", None, None, "read-old",
           "unversioned; a line cut by an interruption is skipped"),
    Format("run.observations", "json", "<run>/solver/observations.json", None, None, "read-old",
           "unversioned; missing sections read as empty"),
    Format("run.cases", "sqlite", "<run>/state/cases.sqlite", None, None, "refuse",
           "unversioned; a case payload with fields this kit does not know was written by a newer kit and is "
           "refused (Case(**payload) would fail); missing fields take their defaults"),
    Format("evidence.store", "sqlite", "evidence.sqlite", "schema_version table",
           "n64dk.match.evidence_store:SCHEMA_VERSION", "migrate", "ordered migrations, one transaction each"),
    Format("verification.receipts", "sqlite", "<project>/.n64dk/verification.sqlite", "columns", "=4 added columns",
           "migrate", "older stores lack image_path, image_name, target_offset, catalogue_vram; they are added "
           "(the writer does the same when it opens the store)"),
    Format("content.cache", "directory", "<cache>/objects, refs, locks", None, None, "read-old",
           "layout v1 (implicit); a reference to a missing object reads as a miss"),
    Format("operations.protocol", "json", "operation-schema output", "protocol", "=n64dk.operations.v1", "refuse",
           "automation clients must check the protocol string"),
    Format("api", "python", "n64dk.api", "API_VERSION", "n64dk.api:API_VERSION", "refuse",
           "see the deprecation policy in n64dk.api"),
    Format("receipt.exhaustive", "json", "decompile-receipt output", "receipt", "n64dk.commands.receipt:RECEIPT",
           "refuse", "a receipt is regenerated, never migrated"),
    Format("receipt.proof", "json", "issue-receipts output", "schema", "n64dk.match.proofreceipt:SCHEMA", "refuse",
           "a receipt is reissued, never migrated"),
    Format("corpus.manifest", "json", "corpus manifest", "schema", "n64dk.governance:SCHEMA", "refuse",
           "written by people: an older manifest is re-validated by hand"),
    Format("blind.receipt", "json", "blind-check output", "schema", "n64dk.governance:RECEIPT_SCHEMA", "refuse",
           "sealed by digest: never changed"),
    Format("provider.record", "jsonl", "provider log", "schema", "n64dk.match.providers:RECORD_SCHEMA", "read-old",
           "rows of another schema are skipped by ProviderLog.read"),
    Format("refine.dossier", "json", "provider request", "schema", "n64dk.match.providers:DOSSIER_SCHEMA", "refuse",
           "requests are rebuilt from the case"),
    Format("toolchain.manifest", "json", "toolmanifest output", "schema", "n64dk.match.toolmanifest:SCHEMA", "refuse",
           "regenerated from the toolchain root"),
    Format("plugin.manifest", "json", "plugin.json", "schema", "n64dk.plugins:SCHEMA", "refuse",
           "a manifest of another schema is refused by the plugin loader"),
    Format("loader.capture", "json", "loader capture receipt", "schema", "n64dk.analysis.loaders_generic:SCHEMA",
           "refuse", "recaptured"),
    Format("devrecords", "json", "dev-records-decode output", "schema", "=n64dk.devrecords/1", "refuse", "re-decoded"),
    Format("tv.report", "json", "translation-validation report", "schema", "=n64dk.tv/1", "refuse", "recomputed"),
    Format("media.report", "json", "media report", "schema", "n64dk.media:SCHEMA", "refuse", "recomputed"),
    Format("doctor.report", "json", "doctor output", "schema", "n64dk.commands.doctor:DOCTOR_SCHEMA", "refuse",
           "recomputed"),
    Format("candidates.report", "json", "candidate-reconcile output", "schema_version", "=1", "refuse", "recomputed"),
    Format("simindex", "json", "similarity index", "format/version", "=1", "refuse", "rebuilt"),
    Format("transplant.index", "json", "donor index", "version", "n64dk.match.transplant:INDEX_VERSION", "refuse",
           "rebuilt by the donor stage when the version differs"),
    Format("image.preflight", "json", "image map preflight", "version", "n64dk.match.layout:PREFLIGHT_VERSION",
           "refuse", "recomputed"),
    Format("sdk.database", "json", "n64dk/sdk/data", "format", "n64dk.sdk.db:FORMAT", "refuse", "shipped with the kit"),
    Format("migration.report", "json", "migrate output", "schema", "=" + REPORT_SCHEMA, "refuse", "recomputed"),
)

# JSON documents recognised by their version key: (format name, key, extra key that must be present)
_DOCUMENTS = (("receipt.exhaustive", "receipt", None), ("receipt.proof", "schema", None),
              ("corpus.manifest", "schema", "entries"), ("blind.receipt", "schema", "violations"),
              ("refine.dossier", "schema", "prompt"), ("toolchain.manifest", "schema", "components"),
              ("plugin.manifest", "schema", "capability"), ("loader.capture", "schema", None),
              ("devrecords", "schema", None), ("tv.report", "schema", None), ("media.report", "schema", "media"),
              ("doctor.report", "schema", "doctor"), ("candidates.report", "schema_version", "candidates"),
              ("migration.report", "schema", "items"), ("operations.protocol", "protocol", "commands"))


def registry() -> list[dict]:
    return [f.as_dict() for f in FORMATS]


def _format(name: str) -> Format:
    return next(f for f in FORMATS if f.name == name)


def _family(value) -> tuple[str, int] | None:
    """``n64dk.tv/1`` -> (``n64dk.tv``, 1); ``n64dk.operations.v1`` -> (``n64dk.operations``, 1); 3 -> ("", 3)."""
    if isinstance(value, int):
        return "", value
    match = re.match(r"^(.*?)(?:[/.]v?|-v?)(\d+)$", str(value))
    return (match[1], int(match[2])) if match else None


def compare_versions(found, current) -> str:
    """``current``, ``older``, ``newer`` or ``foreign`` (another family, or unparseable)."""
    a, b = _family(found), _family(current)
    if a is None or b is None or a[0] != b[0]:
        return "current" if found == current else "foreign"
    return "current" if a[1] == b[1] else "older" if a[1] < b[1] else "newer"


# --------------------------------------------------------------------------- inspection

def _item(fmt: str, file: str, status: str, explanation: str, found=None, current=None, action: str = "") -> dict:
    return {"format": fmt, "file": file, "status": status, "found": found, "current": current, "action": action,
            "explanation": explanation}


def open_readonly(path: Path) -> sqlite3.Connection:
    """Read-only, sidecar-free when no write-ahead log exists (the kit's rule for inspection)."""
    mode = "mode=ro" if Path(f"{path}-wal").exists() else "mode=ro&immutable=1"
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?{mode}", uri=True)


def _tables(db: sqlite3.Connection) -> set[str]:
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


VERIFICATION_COLUMNS = (("image_path", "TEXT"), ("image_name", "TEXT"), ("target_offset", "INTEGER"),
                        ("catalogue_vram", "INTEGER"))
UNVERSIONED_STORES = (({"scripts"}, "script index"), ({"files", "sources", "observations"}, "result index"),
                      ({"jobs", "candidates", "job_events"}, "job store"), ({"verdicts"}, "unit verdict cache"),
                      ({"functions", "abi_evidence"}, "function corpus"), ({"tranches", "tranche_items"}, "sweeps"),
                      ({"sessions", "session_candidates"}, "matching sessions"), ({"sessions"}, "session registry"))


def check_store(path: Path, apply: bool = False) -> list[dict]:
    path = Path(path)
    rel = path.name
    try:
        db = open_readonly(path)
        try:
            tables = _tables(db)
            if {"schema_version", "records", "blobs"} <= tables:
                found = int(db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0)
                kind = "evidence"
            elif "cases" in tables:
                kind = "cases"
                columns = _columns(db, "cases")
                payloads = [row[0] for row in db.execute("SELECT payload FROM cases")] if "payload" in columns else []
            elif "receipts" in tables:
                kind = "verification"
                columns = _columns(db, "receipts")
            else:
                kind = "other"
        finally:
            db.close()
    except sqlite3.DatabaseError as error:
        return [_item("sqlite", rel, "refused", f"not a readable SQLite database: {error}")]
    if kind == "evidence":
        from .match import evidence_store as es
        current = es.SCHEMA_VERSION
        state = compare_versions(found, current)
        if state == "current":
            return [_item("evidence.store", rel, "current", "schema is current", found, current)]
        if state == "newer":
            return [_item("evidence.store", rel, "refused", "written by a newer kit: upgrade the kit to read it",
                          found, current)]
        if not apply:
            return [_item("evidence.store", rel, "migratable",
                          f"schema {found} -> {current}: migrations {list(range(found + 1, current + 1))}", found, current,
                          "open it writable (migrate without --check)")]
        repo = es.SQLiteRepository(path)
        repo.close()
        return [_item("evidence.store", rel, "migrated", f"schema {found} -> {current}", found, current,
                      "migrated in place, one transaction per step")]
    if kind == "cases":
        required = {"case_id", "target_digest", "status", "top_cause", "size", "payload", "updated_ns"}
        if not required <= columns:
            return [_item("run.cases", rel, "refused", f"the cases table lacks {sorted(required - columns)}: not a "
                                                       "case store this kit can read")]
        from dataclasses import fields
        from .match.resolve import Case
        known = {f.name for f in fields(Case)}
        unknown: set[str] = set()
        bad = 0
        for payload in payloads:
            try:
                unknown |= set(json.loads(payload)) - known
            except ValueError:
                bad += 1
        if unknown or bad:
            return [_item("run.cases", rel, "refused",
                          (f"case fields this kit does not know {sorted(unknown)}: written by a newer kit; " if unknown
                           else "") + (f"{bad} unreadable payloads" if bad else ""), len(payloads))]
        return [_item("run.cases", rel, "read-old", f"{len(payloads)} cases readable (unversioned store)", len(payloads))]
    if kind == "verification":
        missing = [(name, kind_) for name, kind_ in VERIFICATION_COLUMNS if name not in columns]
        if not missing:
            return [_item("verification.receipts", rel, "current", "every column present")]
        if not apply:
            return [_item("verification.receipts", rel, "migratable", f"missing columns {[m[0] for m in missing]}: "
                          "verification_store.current() reads nothing from it until they exist", action="add columns")]
        db = sqlite3.connect(path, timeout=60, isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE")
            try:
                for name, declaration in missing:
                    db.execute(f"ALTER TABLE receipts ADD COLUMN {name} {declaration}")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        finally:
            db.close()
        return [_item("verification.receipts", rel, "migrated", f"added {[m[0] for m in missing]}",
                      action="columns added in one transaction")]
    for names, label in UNVERSIONED_STORES:
        if names <= tables:
            return [_item("sqlite", rel, "read-old", f"{label}: unversioned, created and extended on open")]
    return [_item("sqlite", rel, "unknown", f"tables {sorted(tables)} belong to no store this kit knows")]


def check_document(path: Path) -> list[dict]:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return [_item("json", path.name, "refused", f"unreadable JSON: {type(error).__name__}")]
    if not isinstance(data, dict):
        return [_item("json", path.name, "unknown", "not a JSON object")]
    if data.get("format") == "n64dk-simindex":
        found, name = data.get("version"), "simindex"
    else:
        hit = next(((n, k) for n, k, extra in _DOCUMENTS
                    if data.get(k) is not None and (extra is None or extra in data)
                    and compare_versions(data.get(k), _format(n).current_value()) != "foreign"), None)
        if hit is None:
            return [_item("json", path.name, "unknown", "no version this kit recognises")]
        name, found = hit[0], data.get(hit[1])
    fmt = _format(name)
    current = fmt.current_value()
    state = compare_versions(found, current)
    if state == "current":
        return [_item(name, path.name, "current", "version is current", found, current)]
    if state == "newer":
        return [_item(name, path.name, "refused", "written by a newer kit: upgrade the kit", found, current)]
    return [_item(name, path.name, "refused" if fmt.old == "refuse" else "read-old",
                  f"older version: {fmt.note}", found, current)]


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1))
    temporary.replace(path)


def check_run(run: Path, apply: bool = False) -> list[dict]:
    run = Path(run)
    items = []
    path = run / "solver" / "invocation.json"
    if path.is_file():
        try:
            saved = json.loads(path.read_text())
        except ValueError as error:
            saved, items = None, [_item("run.invocation", "solver/invocation.json", "refused",
                                        f"unreadable: {type(error).__name__}; --resume cannot continue this run")]
        if isinstance(saved, dict):
            from .match import solver
            if "mode_settings" in saved:
                items.append(_item("run.invocation", "solver/invocation.json", "current", "v2", 2, 2))
            elif saved.get("effort") in solver.MODES:
                settings = solver.mode_settings(solver.MODES[saved["effort"]])
                if apply:
                    shutil.copyfile(path, path.with_name("invocation.json.v1.bak"))
                    _write_json(path, {**saved, "mode_settings": settings})
                    items.append(_item("run.invocation", "solver/invocation.json", "migrated",
                                       f"v1 -> v2: mode_settings {settings} recorded (the starting mode's, as "
                                       "--resume assumes)", 1, 2, "backup invocation.json.v1.bak"))
                else:
                    items.append(_item("run.invocation", "solver/invocation.json", "migratable",
                                       f"v1: --resume assumes mode_settings {settings}; migrating records them", 1, 2,
                                       "record mode_settings"))
            elif saved.get("effort") is None:
                items.append(_item("run.invocation", "solver/invocation.json", "read-old",
                                   "v0 (before effort modes): --resume treats it as effort normal", 0, 2))
            else:
                items.append(_item("run.invocation", "solver/invocation.json", "refused",
                                   f"effort {saved.get('effort')!r} is not a mode of this kit: written by another "
                                   "version", saved.get("effort"), sorted(solver.MODES)))
    else:
        items.append(_item("run.invocation", "solver/invocation.json", "absent",
                           "no invocation: --resume cannot continue this run (it predates resumable runs)"))
    for rel, fmt in (("report.json", "run.report"), ("solver/observations.json", "run.observations")):
        target = run / rel
        if not target.is_file():
            items.append(_item(fmt, rel, "absent", "not written (an unfinished or older run)"))
            continue
        try:
            ok = isinstance(json.loads(target.read_text()), dict)
        except ValueError:
            ok = False
        items.append(_item(fmt, rel, "read-old" if ok else "refused",
                           "unversioned; read tolerantly" if ok else "unreadable JSON: readers report it as an error"))
    rounds = run / "rounds.jsonl"
    if rounds.is_file():
        bad = 0
        for line in rounds.read_text(errors="replace").splitlines():
            try:
                json.loads(line)
            except ValueError:
                bad += 1
        items.append(_item("run.rounds", "rounds.jsonl", "read-old",
                           "unversioned" + (f"; {bad} cut lines are skipped" if bad else "")))
    cases = run / "state" / "cases.sqlite"
    if cases.is_file():
        items += [{**row, "file": "state/cases.sqlite"} for row in check_store(cases, apply)]
    for name in ("evidence.sqlite",):
        if (run / name).is_file():
            items += check_store(run / name, apply)
    for name in ("receipt.json", "toolchain_manifest.json"):
        if (run / name).is_file():
            items += check_document(run / name)
    return items


def check_cache(root: Path, limit: int = 20_000) -> list[dict]:
    root = Path(root)
    refs = sorted((root / "refs").iterdir()) if (root / "refs").is_dir() else []
    dangling = unreadable = 0
    for ref in refs[:limit]:
        try:
            payload = json.loads(ref.read_text())
            digest = str(payload.get("digest", ""))
            if not (root / "objects" / digest[:2] / digest[2:]).is_file():
                dangling += 1
        except (OSError, ValueError, AttributeError):
            unreadable += 1
    status = "refused" if unreadable else "read-old"
    return [_item("content.cache", root.name, status,
                  f"layout v1 (implicit): {len(refs)} references" + (f" (first {limit} checked)" if len(refs) > limit
                                                                     else "")
                  + f", {dangling} to missing objects (read as misses)"
                  + (f", {unreadable} unreadable references" if unreadable else ""))]


def detect(path: Path) -> str:
    path = Path(path)
    if path.is_dir():
        if (path / "solver" / "invocation.json").is_file() or (path / "report.json").is_file() \
                or (path / "rounds.jsonl").is_file():
            return "run"
        if (path / "objects").is_dir() and (path / "refs").is_dir():
            return "cache"
        return "unknown"
    if path.is_file():
        with path.open("rb") as stream:
            head = stream.read(16)
        if head.startswith(b"SQLite format 3"):
            return "store"
        if path.suffix == ".json":
            return "document"
    return "unknown"


def migrate(path: str | Path, check: bool = True) -> dict:
    """The migration report of ``path``; with ``check=False`` the defined steps are applied."""
    path = Path(path)
    kind = detect(path)
    if kind == "run":
        items = check_run(path, apply=not check)
    elif kind == "cache":
        items = check_cache(path)
    elif kind == "store":
        items = check_store(path, apply=not check)
    elif kind == "document":
        items = check_document(path)
    else:
        items = [_item("unknown", path.name, "refused", "not a run directory, a content cache, a store or a versioned "
                                                        "JSON document")]
    counts: dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"schema": REPORT_SCHEMA, "kit_version": version(), "target": path.name, "kind": kind,
            "check": check, "items": items, "counts": dict(sorted(counts.items())),
            "ok": not any(item["status"] in ("refused", "unknown") for item in items),
            "pending": sum(1 for item in items if item["status"] == "migratable")}


# --------------------------------------------------------------------------- release notes

LOG_FORMAT = "%H%x1f%ad%x1f%s"


def git_log(repo: str | Path, since: str | None = None, runner=subprocess.run) -> str:
    """``git log`` of ``repo`` (read-only), in ``LOG_FORMAT`` with short dates."""
    argv = ["git", "-C", str(repo), "log", f"--format={LOG_FORMAT}", "--date=short"]
    if since:
        argv.append(f"{since}..HEAD")
    proc = runner(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"git log failed: {(proc.stderr or '').strip()[-200:]}")
    return proc.stdout


def parse_log(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 3 and parts[0]:
            out.append({"commit": parts[0][:10], "date": parts[1], "subject": parts[2].strip()})
    return out


def changelog(entries: list[dict], release: str, date: str = "") -> str:
    """A CHANGELOG section: one line per commit, newest first, under the release heading.  A stub:
    the subjects are the authors' words; a release editor groups them."""
    lines = [f"## {release}" + (f" ({date})" if date else ""), ""]
    lines += [f"- {e['subject']} ({e['commit']}, {e['date']})" for e in entries] or ["- no changes"]
    return "\n".join(lines) + "\n"


RELEASE_CHECKLIST = (
    "version_info() consistent (pyproject.toml == n64dk.__version__) and bumped by the semantic rule",
    "the full test suite and tools/ci_gates.sh pass on a clean checkout",
    "tests/test_contracts.py passes: CLI, Python API and operation schema agree; API_VERSION bumped if they changed",
    "every changed stored format has a new version in FORMATS and a migrate, read-old or refuse answer",
    "migrate --check over the benchmark runs, caches and stores reports nothing refused",
    "deprecations past their removal version are removed and listed",
    "CHANGELOG section generated (changelog(parse_log(git_log(...)))) and edited",
    "no ROM bytes, compiler binaries, credentials or machine paths in the package (grep the sdist)",
)
