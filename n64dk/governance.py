"""Corpus governance and technically enforced blind isolation (product item 18).

**Data classes.**  Everything the kit touches belongs to one class, and the class decides by
default what may happen to it (``DATA_CLASSES``): may it leave the machine, appear in a shareable
export, train or evaluate a model, and must a blind run be kept away from it.  A corpus entry can
only be *stricter* than its class for redistribution, export and training; blind exclusion is per
entry (another title's public source is a fair donor, the same title's is the answer).

**The corpus manifest** (``n64dk.corpus/1``, JSON) lists entries: name, class, origin, licence,
date, a digest of the tree (``tree_digest``), the uses it permits, the export / training /
blind-exclusion flags, the titles it concerns and where it lives.  ``validate`` names every
problem; ``permitted(entry, use)`` answers with a reason.

**Blind isolation** (``check_isolation``, ``blind_guard``) is checked, not promised.  Given the
forbidden trees (the manifest's blind-excluded entries for a title, plus explicit paths) and what a
run is configured with (its options or ``solver/invocation.json``, the project's ``n64dk.toml``,
every ``N64DK_*`` environment variable, and the files those name, two levels deep), it verifies:

* no configured path is inside a forbidden tree, contains one, or is a symlink into one;
* no file under a configured directory (bounded walk) is a symlink into a forbidden tree;
* no C source under a configured directory is a byte-identical copy of a forbidden one (other
  identical text files, such as shared SDK headers, are listed, not failed);

**Shared libraries** (opt-in).  A forbidden title tree may vendor a public library (Dr. Mario 64's
``lib/ultralib``) that a blind run may legitimately use from its own copy.  The content rule cannot
tell the two apart, so a run may declare a shared-library manifest (``n64dk.shared-library/1``:
name, provenance URL, and a SHA-256 per file or a tree digest of a named root), loaded only with the
digest of the manifest itself (``SharedLibrary.load(path, pin)``; ``blind-check --shared-library
MANIFEST@sha256:<hex>``).  A content copy is then exempt only when its SHA-256 is one the pinned
manifest lists; nothing is exempt by path or name, and a configured path inside a forbidden tree or a
symlink into one is never exempt.  The receipt records each manifest (name, provenance, digest) and
every exemption used.  Without a manifest the receipt is exactly what it was.

and writes an isolation receipt (``n64dk.blind-isolation/1``) listing everything it checked, sealed
by the SHA-256 of its canonical JSON (``verify_receipt`` recomputes it).  The seal shows the receipt
was not edited after it was made; it is not a signature by anyone.

Limits, stated in every receipt: the check runs when the guard is entered and again when it is
left; a process that opens a forbidden file by a path it computes itself, or a network fetch, is
not seen.  ``refine.known_source`` names the forbidden tree on purpose (to keep it *out* of
prompts) and is exempt.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = "n64dk.corpus/1"
RECEIPT_SCHEMA = "n64dk.blind-isolation/1"
USES = ("analysis", "matching", "donor", "evaluation", "training", "export", "redistribution")
TEXT_SUFFIXES = (".c", ".h", ".s", ".inc", ".txt", ".toml", ".json", ".tsv", ".ld", ".mk", ".py")
ANSWER_SUFFIXES = (".c",)              # a copy of one of these is a violation; other copies are listed
CONTENT_MIN = 64                       # files shorter than this say nothing about their origin
CONTENT_MAX = 1 << 20                  # files longer than this are not hashed for the content check
# exclusion lists name the forbidden trees on purpose; ``blind_forbid`` is decompile-run's own
# (saved in solver/invocation.json, so blind-check --invocation of a blind run reads it); a corpus
# manifest (``decompile-run --corpus-manifest``) is read only for the trees it forbids and the titles
# they concern (``n64dk.runmodes``)
EXEMPT_KEYS = ("known_source", "refine_known_source", "blind_forbid", "corpus_manifest")
# Scratch locations a tool is told to write into are not inputs: they are checked against the
# forbidden trees (a TMPDIR inside one would leak), but never walked or followed.  Walking a shared
# /tmp read unrelated projects' files and wrote their paths into the receipt.
SCRATCH_KEYS = ("TMPDIR", "TMP", "TEMP", "TEMPDIR")
# --- strict mode (``check_isolation(strict=True)``, what ``decompile-run --blind-forbid`` uses).
# Environment variables outside N64DK_* that steer what a run's tools read: include and library
# search paths reach a tree they merely contain; a PATH or TMPDIR entry exposes only its own files.
# Checked as locations, never walked (walking /usr/include or PATH would be slow and say nothing).
STRICT_ENV_SEARCH = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
                     "COMPILER_PATH", "GCC_EXEC_PREFIX", "LIBRARY_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD")
STRICT_ENV_INSIDE_ONLY = ("PATH",) + SCRATCH_KEYS
# a byte copy of one of these forbidden files fails a strict check as a C source does: a linker
# script or a symbol list carries the title's names and layout
STRICT_ANSWER_SUFFIXES = (".ld",)
STRICT_ANSWER_PREFIXES = ("symbol_addrs", "undefined_syms", "undefined_funcs")
INCLUDE_SUFFIXES = (".c", ".h", ".s", ".inc", ".ld")
_INCLUDE = re.compile(r'^[ \t]*(?:#[ \t]*include|\.include|INCLUDE)[ \t]+"?([^"\s<>]+)"?', re.M)
# files never read as configuration by the strict sniff (anything else small is tried as TOML, then JSON)
NOT_CONFIG_SUFFIXES = (".c", ".h", ".s", ".inc", ".ld", ".o", ".a", ".elf", ".bin", ".z64", ".n64", ".v64",
                       ".sqlite", ".py", ".tsv", ".md", ".jsonl")
PROVENANCE_SCHEMA = "n64dk.blind-provenance/1"


def _scratch_dirs() -> set[Path]:
    import tempfile
    out = set()
    for value in ("/tmp", "/var/tmp", "/dev/shm", tempfile.gettempdir()):
        with contextlib.suppress(OSError):
            out.add(Path(value).resolve())
    return out


@dataclass(frozen=True)
class DataClass:
    name: str
    meaning: str
    redistributable: bool        # may the content leave this machine
    export: bool                 # may it appear in a shareable export
    training: bool               # may it train or evaluate a model
    blind_exclude: bool          # kept away from blind runs by default


DATA_CLASSES: dict[str, DataClass] = {c.name: c for c in (
    DataClass("user-rom-derived", "bytes, disassembly and facts recovered from a ROM the user supplied",
              False, False, False, False),
    DataClass("public-source", "a published decompilation or source tree (its licence governs)", True, False, True, True),
    DataClass("sdk-source", "SDK or library source and headers", False, False, False, False),
    DataClass("preserved-toolchain", "compilers, assemblers and linkers as preserved or rebuilt", False, False, False,
              False),
    DataClass("synthetic", "inputs generated by the kit or its tests", True, True, True, False),
    DataClass("model-generated", "text a model produced (verified or not)", False, True, False, False),
    DataClass("donor", "proven sources of other titles offered as donors", False, False, False, False),
    DataClass("private-project-source", "the user's own project sources", False, True, False, True),
    DataClass("redistributable-metadata", "names, addresses, digests and counts without content", True, True, True,
              False),
    DataClass("non-redistributable", "anything that must stay on this machine", False, False, False, False),
)}


class GovernanceError(ValueError):
    pass


class BlindViolation(RuntimeError):
    """A blind run could reach a forbidden tree; ``receipt`` says how."""

    def __init__(self, receipt: dict):
        self.receipt = receipt
        first = (receipt.get("violations") or [{}])[0]
        super().__init__(f"blind isolation violated ({len(receipt.get('violations') or [])}): "
                         f"{first.get('key')} -> {first.get('verdict')} {first.get('forbidden')}")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: str | Path, limit: int = 200_000) -> tuple[str, int]:
    """``sha256:<hex>`` over the sorted (relative path, file digest) pairs of a tree (symlinks are
    recorded by target, not followed), or of one file; and the number of files."""
    root = Path(root)
    if root.is_file():
        return "sha256:" + file_digest(root), 1
    rows = _tree_rows(root, limit)
    return "sha256:" + hashlib.sha256(canonical(rows).encode()).hexdigest(), len(rows)


def _tree_rows(root: Path, limit: int = 200_000) -> list[list[str]]:
    """``[relative path, file digest or "link:<target>"]`` for every file of a tree, sorted."""
    rows = []
    for path in sorted(root.rglob("*")):
        if len(rows) >= limit:
            raise GovernanceError(f"{root.name}: more than {limit} files")
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append([rel, "link:" + os.readlink(path)])
        elif path.is_file():
            rows.append([rel, file_digest(path)])
    return rows


# --------------------------------------------------------------------------- the corpus manifest

@dataclass
class CorpusEntry:
    name: str
    data_class: str
    origin: str
    licence: str = ""
    date: str = ""
    digest: str = ""
    uses: list = field(default_factory=list)
    export: bool | None = None           # None: the class default
    training: bool | None = None
    blind_exclude: bool | None = None
    titles: list = field(default_factory=list)   # titles it concerns (empty: every title)
    roots: list = field(default_factory=list)    # where it lives (relative to the manifest, or absolute)

    @property
    def policy(self) -> DataClass:
        return DATA_CLASSES[self.data_class]

    def flag(self, name: str) -> bool:
        value = getattr(self, name)
        return getattr(self.policy, name) if value is None else bool(value)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Manifest:
    entries: list
    base: Path = Path(".")

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise GovernanceError(f"{path.name}: not a {SCHEMA} manifest (schema {data.get('schema') if isinstance(data, dict) else None})")
        known = set(CorpusEntry.__dataclass_fields__)
        entries = []
        for row in data.get("entries") or []:
            unknown = set(row) - known
            if unknown:
                raise GovernanceError(f"{path.name}: entry {row.get('name')} has unknown fields {sorted(unknown)}")
            entries.append(CorpusEntry(**row))
        manifest = cls(entries, path.parent)
        problems = manifest.validate()
        if problems:
            raise GovernanceError(f"{path.name}: " + "; ".join(problems))
        return manifest

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"schema": SCHEMA, "entries": [e.as_dict() for e in self.entries]},
                                         indent=1, sort_keys=True) + "\n")

    def validate(self) -> list[str]:
        problems, seen = [], set()
        for entry in self.entries:
            label = entry.name or "<unnamed>"
            if not entry.name or entry.name in seen:
                problems.append(f"{label}: a name, unique in the manifest, is required")
            seen.add(entry.name)
            if entry.data_class not in DATA_CLASSES:
                problems.append(f"{label}: unknown class {entry.data_class!r}")
                continue
            if not entry.origin:
                problems.append(f"{label}: origin is required")
            bad = sorted(set(entry.uses) - set(USES))
            if bad:
                problems.append(f"{label}: unknown uses {bad}")
            policy = entry.policy
            for name in ("export", "training"):
                if getattr(entry, name) is True and not getattr(policy, name):
                    problems.append(f"{label}: class {entry.data_class} forbids {name}; an entry may only be stricter")
            if "redistribution" in entry.uses and not policy.redistributable:
                problems.append(f"{label}: class {entry.data_class} is not redistributable")
            if entry.data_class == "public-source" and not entry.licence:
                problems.append(f"{label}: public source needs its licence")
            if entry.digest and not entry.digest.startswith("sha256:"):
                problems.append(f"{label}: digest must be sha256:<hex>")
        return problems

    def root_paths(self, entry: CorpusEntry) -> list[Path]:
        return [(Path(r).expanduser() if Path(r).expanduser().is_absolute() else self.base / r) for r in entry.roots]

    def forbidden_for(self, title: str | None = None) -> list[tuple[str, Path]]:
        """``(label, root)`` of every blind-excluded entry concerning ``title`` (or every title)."""
        out = []
        for entry in self.entries:
            if entry.flag("blind_exclude") and (title is None or not entry.titles or title in entry.titles):
                out += [(entry.name, root) for root in self.root_paths(entry)]
        return out

    def verify_digests(self) -> list[dict]:
        """Each entry's recorded digest against its tree now (entries with one root)."""
        out = []
        for entry in self.entries:
            roots = self.root_paths(entry)
            if not entry.digest or len(roots) != 1 or not roots[0].exists():
                out.append({"name": entry.name, "status": "not checked"})
                continue
            now, _files = tree_digest(roots[0])
            out.append({"name": entry.name, "status": "current" if now == entry.digest else "changed", "digest": now})
        return out


def permitted(entry: CorpusEntry, use: str) -> tuple[bool, str]:
    """May ``entry`` serve ``use``?  The use must be listed, and the class and entry flags allow it."""
    if use not in USES:
        raise GovernanceError(f"unknown use {use!r}")
    if use not in entry.uses:
        return False, f"{entry.name}: {use} is not among its permitted uses"
    if use == "training" and not entry.flag("training"):
        return False, f"{entry.name}: training is not allowed for {entry.data_class}"
    if use == "export" and not entry.flag("export"):
        return False, f"{entry.name}: export is not allowed for {entry.data_class}"
    if use == "redistribution" and not entry.policy.redistributable:
        return False, f"{entry.name}: {entry.data_class} is not redistributable"
    return True, f"{entry.name}: {use} permitted ({entry.data_class}, licence {entry.licence or 'none recorded'})"


# --------------------------------------------------------------------------- shared libraries (opt-in)

SHARED_SCHEMA = "n64dk.shared-library/1"
SHARED_FIELDS = frozenset(("schema", "name", "provenance", "revision", "licence", "note", "files", "root",
                           "tree_digest"))
_HEX64 = re.compile(r"[0-9a-f]{64}")
_URL = re.compile(r"[a-z][a-z0-9+.-]*://[^\s/]+\S*")


def parse_pin(spec: str) -> tuple[str, str | None]:
    """``MANIFEST@sha256:<hex>`` -> ``(MANIFEST, "sha256:<hex>")``; no pin -> ``(spec, None)``."""
    path, sep, digest = str(spec).rpartition("@sha256:")
    return (path, "sha256:" + digest) if sep and path else (str(spec), None)


def _relative(rel) -> bool:
    return isinstance(rel, str) and bool(rel) and not rel.startswith(("/", "\\", "~")) \
        and not re.match(r"[A-Za-z]:", rel) \
        and ".." not in rel.replace("\\", "/").split("/")


@dataclass(frozen=True)
class SharedLibrary:
    """A shared public library declared by a pinned ``n64dk.shared-library/1`` manifest.

    ``files`` maps relative paths to SHA-256 digests; only the digests grant exemptions (the paths
    are a record).  ``form`` is ``files`` (digests listed in the manifest) or ``tree`` (``root`` +
    ``tree_digest``: the tree is hashed now and must match, and its regular files' digests are the
    pinned set)."""
    name: str
    provenance: str
    manifest_digest: str
    files: dict
    form: str
    path: Path
    revision: str = ""
    licence: str = ""
    root: Path | None = None
    tree_digest: str = ""

    @property
    def digests(self) -> frozenset:
        return frozenset(self.files.values())

    @classmethod
    def load(cls, path: str | Path, pin: str | None) -> "SharedLibrary":
        """Load a manifest whose bytes hash to ``pin`` (``sha256:<hex>``); refuse anything else."""
        path = Path(path)
        if not pin:
            raise GovernanceError(f"{path.name}: a shared-library manifest is used only when pinned by its "
                                  f"SHA-256 (MANIFEST@sha256:<hex>)")
        pin = pin.strip().lower()
        if not pin.startswith("sha256:") or not _HEX64.fullmatch(pin[7:]):
            raise GovernanceError(f"{path.name}: the pin must be sha256:<64 hex digits>")
        raw = path.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != pin:
            raise GovernanceError(f"{path.name}: the manifest does not match its pinned digest "
                                  f"(it changed after it was reviewed, or the pin is wrong)")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise GovernanceError(f"{path.name}: not JSON ({exc})") from None
        problems = shared_problems(data)
        if problems:
            raise GovernanceError(f"{path.name}: " + "; ".join(problems))
        common = {"name": data["name"], "provenance": data["provenance"], "manifest_digest": pin,
                  "path": path, "revision": str(data.get("revision") or ""),
                  "licence": str(data.get("licence") or "")}
        if "files" in data:
            return cls(files=dict(data["files"]), form="files", **common)
        root = Path(data["root"]).expanduser()
        root = root if root.is_absolute() else path.parent / root
        if not root.is_dir():
            raise GovernanceError(f"{path.name}: tree root {data['root']} is not a directory")
        rows = _tree_rows(root)
        if "sha256:" + hashlib.sha256(canonical(rows).encode()).hexdigest() != data["tree_digest"]:
            raise GovernanceError(f"{path.name}: tree {data['root']} does not match its pinned tree digest")
        files = {rel: digest for rel, digest in rows if not digest.startswith("link:")}
        if not files:
            raise GovernanceError(f"{path.name}: tree {data['root']} has no files")
        return cls(files=files, form="tree", root=root, tree_digest=data["tree_digest"], **common)

    def receipt_entry(self) -> dict:
        row = {"name": self.name, "provenance": self.provenance, "revision": self.revision,
               "licence": self.licence, "form": self.form, "files": len(self.files),
               "manifest_digest": self.manifest_digest}
        if self.form == "tree":
            row["tree_digest"] = self.tree_digest
        return row


def shared_problems(data) -> list[str]:
    """Every problem with a ``n64dk.shared-library/1`` manifest's content (empty: valid)."""
    if not isinstance(data, dict) or data.get("schema") != SHARED_SCHEMA:
        return [f"not a {SHARED_SCHEMA} manifest"]
    problems = []
    unknown = sorted(set(data) - SHARED_FIELDS)
    if unknown:
        problems.append(f"unknown fields {unknown}")
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        problems.append("a name is required")
    if not isinstance(data.get("provenance"), str) or not _URL.fullmatch(data["provenance"].strip()):
        problems.append("provenance must be the library's public URL (scheme://host/...)")
    has_files, has_tree = "files" in data, "root" in data or "tree_digest" in data
    if has_files == has_tree:
        problems.append("give exactly one of: files (path -> sha256), or root + tree_digest")
    elif has_files:
        files = data["files"]
        if not isinstance(files, dict) or not files:
            problems.append("files must be a non-empty map of relative path -> sha256")
        else:
            bad_paths = sorted(str(rel) for rel in files if not _relative(rel))[:5]
            bad_digests = sorted(str(rel) for rel, digest in files.items()
                                 if not isinstance(digest, str) or not _HEX64.fullmatch(digest))[:5]
            if bad_paths:
                problems.append(f"paths must be relative, without '..': {bad_paths}")
            if bad_digests:
                problems.append(f"digests must be 64 lowercase hex digits: {bad_digests}")
    else:
        if not isinstance(data.get("root"), str) or not data["root"]:
            problems.append("a tree manifest needs root")
        digest = data.get("tree_digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:") or not _HEX64.fullmatch(digest[7:]):
            problems.append("a tree manifest needs tree_digest sha256:<hex>")
    return problems


def pin_shared_library(root: str | Path, *, name: str, provenance: str, revision: str = "",
                       licence: str = "", limit: int = 200_000) -> dict:
    """A ``files`` manifest pinning every regular file under ``root`` (symlinks and ``.git`` skipped)."""
    root = Path(root)
    if not root.is_dir():
        raise GovernanceError(f"{root.name}: not a directory")
    files = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if ".git" in rel.parts or path.is_symlink() or not path.is_file():
            continue
        if len(files) >= limit:
            raise GovernanceError(f"{root.name}: more than {limit} files")
        files[rel.as_posix()] = file_digest(path)
    data = {"schema": SHARED_SCHEMA, "name": name, "provenance": provenance, "files": files}
    if revision:
        data["revision"] = revision
    if licence:
        data["licence"] = licence
    problems = shared_problems(data)
    if problems:
        raise GovernanceError("; ".join(problems))
    return data


def shared_manifest_bytes(data: dict) -> bytes:
    return (json.dumps(data, indent=1, sort_keys=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- blind isolation

FILE_SUFFIXES = (".toml", ".json", ".txt", ".tsv", ".c", ".h", ".s", ".z64", ".n64", ".v64", ".bin", ".sqlite",
                 ".ld", ".py")


def _looks_like_path(text: str) -> bool:
    return bool(text) and "\n" not in text and len(text) < 4096 and " " not in text.strip() and (
        "/" in text or "\\" in text or text.startswith(("~", "./", "../")) or text.lower().endswith(FILE_SUFFIXES))


def _leaf(key: str) -> str:
    """The last name in a key path: ``project/n64dk.toml>run.refine.known_source[0]`` -> ``known_source``."""
    return re.sub(r"(\[\d+\])+$", "", key).replace(">", ".").rsplit(".", 1)[-1]


def _strings(value, key: str = ""):
    """``(key, string)`` for every string in a JSON/TOML value."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(v, f"{key}.{k}" if key else str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _strings(v, f"{key}[{i}]")
    elif isinstance(value, str):
        yield key, value
    elif isinstance(value, Path):
        yield key, str(value)


def _parse(path: Path, sniff: bool = False):
    """A TOML or JSON file's content by its suffix; with ``sniff`` (strict checks) any other small
    file that is not plainly source or binary is tried as TOML, then JSON, because the kit reads some
    configuration whatever its name (``--donors`` is parsed as TOML regardless of suffix)."""
    try:
        if path.suffix == ".toml":
            return tomllib.loads(path.read_text(errors="replace"))
        if path.suffix == ".json":
            return json.loads(path.read_text(errors="replace"))
        if not sniff or path.suffix.lower() in NOT_CONFIG_SUFFIXES or path.stat().st_size > CONTENT_MAX:
            return None
        text = path.read_text(errors="replace")
    except (OSError, ValueError):
        return None
    for loader in (tomllib.loads, json.loads):
        try:
            return loader(text)
        except ValueError:
            continue
    return None


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _strict_answer(where: str) -> bool:
    name = where.rsplit("/", 1)[-1]
    return name.endswith(STRICT_ANSWER_SUFFIXES) or name.startswith(STRICT_ANSWER_PREFIXES)


class _Check:
    def __init__(self, forbidden, *, content: bool, file_limit: int, follow: int, libraries=(),
                 strict: bool = False):
        self.content, self.file_limit, self.follow, self.strict = content, file_limit, follow, strict
        self.libraries = list(libraries)          # pinned SharedLibrary objects (opt-in)
        self.exempt: dict[str, str] = {}          # digest -> library name
        for library in self.libraries:
            for digest in sorted(library.digests):
                self.exempt.setdefault(digest, library.name)
        self.exemptions: list[dict] = []
        self.forbidden = []                       # (label, resolved root)
        self.answers: dict[str, str] = {}         # digest -> forbidden label/relative path (.c)
        self.others: dict[str, str] = {}          # digest -> label/relative path (other text files)
        self.summary = []
        self.checked, self.violations, self.shared = [], [], []
        for label, root in forbidden:
            resolved = Path(root).expanduser().resolve()
            self.forbidden.append((str(label), resolved))
            row = {"label": str(label), "root": str(resolved), "exists": resolved.exists()}
            if resolved.exists():
                row["tree_digest"], row["files"] = tree_digest(resolved)
                if content:
                    self._index(label, resolved)
                    if self.libraries:            # how much of this tree the manifests could exempt
                        prefix = f"{label}/"
                        mine = [d for d, where in self.answers.items() if where.startswith(prefix)]
                        row["sources"] = len(mine)
                        row["sources_pinned_by_shared_libraries"] = sum(1 for d in mine if d in self.exempt)
            elif strict:                          # a typo would make the run "blind" against nothing
                self.violations.append({"key": f"forbidden:{label}", "value": str(root), "resolved": str(resolved),
                                        "verdict": "the forbidden tree does not exist", "forbidden": str(label)})
            self.summary.append(row)
        self.seen: set[Path] = set()
        self.walked = {"directories": 0, "files_hashed": 0, "symlinks": 0, "truncated": False}
        self.outputs: set[Path] = set()           # run outputs: checked, never walked into
        if strict:
            self.walked["includes_checked"] = 0

    def _index(self, label, root: Path) -> None:
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        if self.strict and len(paths) > self.file_limit:
            # the files past the limit would not be recognised as copies
            self.violations.append({"key": f"forbidden:{label}", "value": str(root),
                                    "verdict": f"the forbidden tree has more than {self.file_limit} entries: "
                                               "not all of it is indexed (raise the walk limit)",
                                    "forbidden": str(label)})
        for path in paths[: self.file_limit]:
            if path.is_file() and not path.is_symlink() and path.suffix in TEXT_SUFFIXES \
                    and CONTENT_MIN <= path.stat().st_size <= CONTENT_MAX:
                where = f"{label}/{path.relative_to(root).as_posix() if path != root else path.name}"
                (self.answers if path.suffix in ANSWER_SUFFIXES else self.others)[file_digest(path)] = where

    def hit(self, resolved: Path, inside_only: bool = False) -> tuple[str, str] | None:
        for label, root in self.forbidden:
            if _within(resolved, root):
                return "inside", label
            if not inside_only and resolved.is_dir() and _within(root, resolved):
                return "contains", label
        return None

    def location(self, key: str, value: str, base: Path, depth: int = 0, *, walk: bool = True,
                 inside_only: bool = False) -> None:
        """Check one configured path.  ``walk=False`` (strict environment search paths): checked
        against the trees, never walked or followed; ``inside_only``: a directory that merely
        contains a tree is not a route (PATH, TMPDIR)."""
        if _leaf(key) in EXEMPT_KEYS:
            self.checked.append({"key": key, "value": value, "verdict": "exempt (an exclusion list)"})
            return
        raw = Path(os.path.expanduser(value))
        path = raw if raw.is_absolute() else base / raw
        resolved = path.resolve()
        found = self.hit(resolved, inside_only)
        if found is None and path.is_symlink():
            found = self.hit(Path(os.path.realpath(path)), inside_only)
        by_key = _leaf(key) in SCRATCH_KEYS
        scratch = by_key or resolved in _scratch_dirs()
        verdict = (f"{found[0]} forbidden" if found else
                   "outside (scratch: checked, not walked)" if scratch and resolved.exists() else
                   "outside (checked, not walked)" if not walk and resolved.exists() else
                   "outside" if resolved.exists() else "absent")
        row = {"key": key, "value": value, "resolved": str(resolved), "verdict": verdict}
        self.checked.append(row)
        if found:
            self.violations.append({**row, "forbidden": found[1]})
            return
        if self.strict and scratch and not by_key and walk and resolved.exists():
            # an input that is a shared scratch directory is never walked (other projects' files),
            # so nothing under it is checked: a strict check cannot vouch for it
            self.violations.append({**row, "verdict": "a shared scratch directory named as an input: not checkable",
                                    "forbidden": "(unchecked)"})
            return
        if scratch or not walk:
            return
        if not resolved.exists() or resolved in self.seen:
            return
        self.seen.add(resolved)
        if resolved.is_dir():
            self.walk(key, resolved, depth)
            return
        if self.strict and resolved.is_file():
            # a file named directly is checked like one found by a walk: copies and includes
            self._content(key, resolved)
            self._includes(key, resolved)
        if depth < self.follow and (resolved.suffix in (".toml", ".json") or self.strict):
            self.references(key, resolved, depth + 1)

    def references(self, key: str, path: Path, depth: int) -> None:
        data = _parse(path, sniff=self.strict)
        if data is None:
            return
        for inner, text in _strings(data):
            if _looks_like_path(text):
                self.location(f"{key}>{inner}", text, path.parent, depth)

    def _content(self, where: str, child: Path) -> None:
        """The content rule for one file: a byte copy of forbidden source fails, other copies are listed."""
        if not (self.content and child.suffix in TEXT_SUFFIXES and (self.answers or self.others)):
            return
        try:
            size = child.stat().st_size
        except OSError:
            return
        if not CONTENT_MIN <= size <= CONTENT_MAX:
            return
        self.walked["files_hashed"] += 1
        digest = file_digest(child)
        if digest in self.exempt and (digest in self.answers or digest in self.others):
            # by content only: the SHA-256 is one a pinned shared-library manifest lists
            self.exemptions.append({"key": where, "sha256": digest, "library": self.exempt[digest],
                                    "same_as": self.answers.get(digest) or self.others[digest],
                                    "would_be": ("copy of forbidden source" if digest in self.answers
                                                 else "listed")})
        elif digest in self.answers:
            self.violations.append({"key": where, "value": str(child), "verdict": "copy of forbidden source",
                                    "forbidden": self.answers[digest]})
        elif digest in self.others and self.strict and _strict_answer(self.others[digest]):
            self.violations.append({"key": where, "value": str(child),
                                    "verdict": "copy of a forbidden symbol list or linker script",
                                    "forbidden": self.others[digest]})
        elif digest in self.others:
            self.shared.append({"key": where, "same_as": self.others[digest]})

    def _includes(self, where: str, child: Path) -> None:
        """Strict: an ``#include "..."`` / ``.include`` / linker ``INCLUDE`` that names an absolute
        path or climbs with ``..`` is a route the configuration does not show; each is checked."""
        if not self.strict or child.suffix not in INCLUDE_SUFFIXES:
            return
        try:
            if child.stat().st_size > CONTENT_MAX:
                return
            text = child.read_text(errors="replace")
        except OSError:
            return
        for target in _INCLUDE.findall(text):
            if not (target.startswith(("/", "~")) or ".." in Path(target).parts):
                continue
            self.walked["includes_checked"] += 1
            raw = Path(os.path.expanduser(target))
            path = raw if raw.is_absolute() else child.parent / raw
            found = self.hit(path.resolve()) or (self.hit(Path(os.path.realpath(path))) if path.is_symlink() else None)
            if found:
                self.violations.append({"key": f"{where}#include", "value": target, "resolved": str(path.resolve()),
                                        "verdict": f"include {found[0]} forbidden", "forbidden": found[1]})

    def walk(self, key: str, root: Path, depth: int) -> None:
        self.walked["directories"] += 1
        stack, count = [(root, 0)], 0
        while stack:
            folder, level = stack.pop()
            try:
                children = sorted(folder.iterdir())
            except OSError:
                continue
            for child in children:
                count += 1
                if count > self.file_limit:
                    self.walked["truncated"] = True
                    if self.strict:               # the rest of this tree would go unchecked
                        self.violations.append({"key": key, "value": str(root),
                                                "verdict": f"walk truncated at {self.file_limit} entries: files "
                                                           "left unchecked (raise the walk limit)",
                                                "forbidden": "(unchecked)"})
                    return
                if child.is_symlink():
                    self.walked["symlinks"] += 1
                    found = self.hit(Path(os.path.realpath(child)))
                    if found:
                        self.violations.append({"key": f"{key}/{child.relative_to(root).as_posix()}",
                                                "value": os.readlink(child), "verdict": f"symlink {found[0]} forbidden",
                                                "forbidden": found[1]})
                    continue
                if child.is_dir():
                    if child in self.outputs:
                        continue
                    if level < 3 or self.strict:  # strict: every level (the file limit bounds the walk)
                        stack.append((child, level + 1))
                    continue
                if level == 0 and depth < self.follow and child.suffix in (".toml", ".json") \
                        and child.stat().st_size <= CONTENT_MAX:
                    self.references(f"{key}/{child.name}", child, depth + 1)
                where = f"{key}/{child.relative_to(root).as_posix()}"
                self._content(where, child)
                self._includes(where, child)




def environment_paths(env) -> list[tuple[str, str]]:
    """``(NAME, path)`` for every path-like part of every ``N64DK_*`` variable (``name=path`` pairs and
    path lists split)."""
    out = []
    for name in sorted(env):
        if not name.startswith("N64DK_"):
            continue
        for part in str(env[name]).split(os.pathsep):
            if "=" in part and not part.startswith(("/", ".", "~")):
                part = part.split("=", 1)[1]
            if _looks_like_path(part):
                out.append((f"env.{name}", part))
    return out


def strict_environment_paths(env) -> list[tuple[str, str, bool]]:
    """``(key, path, inside_only)`` for every part of the variables outside ``N64DK_*`` that a strict
    check reads (``STRICT_ENV_SEARCH``: a directory containing a tree is a route; ``PATH`` and the
    scratch variables: only a directory inside one is)."""
    out = []
    for name in sorted(env):
        if name in STRICT_ENV_SEARCH:
            inside_only = False
        elif name in STRICT_ENV_INSIDE_ONLY:
            inside_only = True
        else:
            continue
        parts = str(env[name]).split(os.pathsep)
        if name == "LD_PRELOAD":
            parts = [piece for part in parts for piece in part.split()]
        out += [(f"env.{name}", part, inside_only) for part in parts if part and "\n" not in part]
    return out


def _bare_existing(value: str, base: Path) -> bool:
    """A configured value without a slash or a known suffix (``--donors donors``) that names an
    existing file or directory relative to ``base``: strict checks treat it as a path."""
    if not value or "\n" in value or len(value) >= 4096 or any(c.isspace() for c in value):
        return False
    try:
        return (base / os.path.expanduser(value)).exists()
    except (OSError, ValueError):
        return False


def check_isolation(forbidden, *, config: dict | None = None, project: str | Path | None = None, env=None,
                    base: str | Path | None = None, content: bool = True, file_limit: int = 50_000,
                    follow: int = 2, shared=(), strict: bool = False, outputs: dict | None = None,
                    mode: str | None = None) -> dict:
    """The isolation receipt of a blind run configured by ``config`` (its options, e.g. the
    ``solver/invocation.json`` of the run), ``project`` (its ``n64dk.toml`` is read) and ``env``
    (default: this process's environment), against ``forbidden`` ``(label, path)`` trees.

    ``shared``: pinned ``SharedLibrary`` objects (opt-in).  A content copy whose SHA-256 one of
    them lists is recorded under ``exemptions`` instead of failing (or being listed); the
    manifests and their tree roots are themselves checked as locations (never followed), so a
    manifest or tree root inside a forbidden tree is a violation.  Empty: the receipt is unchanged.

    ``strict`` (what ``decompile-run --blind-forbid`` uses; off: the receipt is unchanged) fails
    closed where the default check only reports or does not look: a forbidden tree that does not
    exist or is larger than the limit, a truncated walk, a directory deeper than three levels
    (walked), an input that is a shared scratch directory, a copy of a forbidden symbol list or
    linker script, an ``#include`` that leaves the tree it is in, a configured file of any name read
    as TOML/JSON, a bare existing name, a directly named file's content, and the search-path
    variables outside ``N64DK_*`` (``STRICT_ENV_SEARCH``, ``PATH``, ``TMPDIR``).

    ``outputs`` (``{key: path}``): directories the run writes, such as its run directory: checked
    against the trees, never walked (a run fills its own directory with tens of thousands of
    candidate sources; what it held before the run is decided by ``run_provenance``).

    ``mode`` (``decompile-run --mode``): recorded in the receipt as ``run_mode``; None leaves the
    receipt unchanged."""
    from . import __version__
    env = dict(os.environ if env is None else env)
    base = Path(base) if base is not None else Path.cwd()
    libraries = list(shared or ())
    for library in libraries:
        if not isinstance(library, SharedLibrary):
            raise GovernanceError("shared libraries must be loaded with SharedLibrary.load(path, pin)")
    check = _Check(forbidden, content=content, file_limit=file_limit, follow=follow, libraries=libraries,
                   strict=strict)
    check.outputs = {Path(os.path.expanduser(str(v))).resolve() for v in (outputs or {}).values()}
    for key, value in _strings(config or {}, "config"):
        if _looks_like_path(value) or (strict and _bare_existing(value, base)):
            check.location(key, value, base)
    if project is not None:
        project = Path(project)
        check.location("project", str(project), base)
        toml = project / "n64dk.toml"
        if toml.is_file():
            check.references("project/n64dk.toml", toml.resolve(), 1)
    for key, value in environment_paths(env):
        check.location(key, value, base)
    if strict:
        for key, value, inside_only in strict_environment_paths(env):
            check.location(key, value, base, walk=False, inside_only=inside_only)
    for key, value in sorted((outputs or {}).items()):
        check.location(f"output.{key}", str(value), base, walk=False)
    for index, library in enumerate(libraries):
        check.location(f"shared_library[{index}]", str(library.path), base, depth=follow)
        if library.root is not None:
            check.location(f"shared_library[{index}].root", str(library.root), base, depth=follow)
    limits = ["checked when the guard is entered and when it is left, not continuously",
              "a path computed inside a process, or a network fetch, is not seen",
              "content copies are matched byte for byte (C sources fail; other text files are listed)",
              "the seal is a digest of the receipt, not a signature"]
    body = {"schema": RECEIPT_SCHEMA, "kit_version": __version__,
            "forbidden": check.summary,
            "checked": check.checked, "env_names": sorted(n for n in env if n.startswith("N64DK_")),
            "walk": check.walked, "shared_files": check.shared, "violations": check.violations,
            "ok": not check.violations, "limits": limits}
    if mode:
        body["run_mode"] = mode
    if strict:
        body["strict"] = True
        body["file_limit"] = file_limit
        limits.append("strict: incomplete checks fail (missing or oversized trees, truncated walks, scratch "
                      "directories as inputs); symbol lists and linker scripts copied from a forbidden tree "
                      "fail; knowledge copied by hand into configuration, or derived output of an earlier "
                      "run, is not recognisable by any path or content rule (see run_provenance)")
    if libraries:
        body["shared_libraries"] = [library.receipt_entry() for library in libraries]
        body["exemptions"] = check.exemptions
        limits.append("a content copy is exempt only when its SHA-256 is listed by a shared-library manifest "
                      "pinned by digest (shared_libraries); every exemption is listed; paths inside a "
                      "forbidden tree and symlinks into one are never exempt")
    body["digest"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return body


def verify_receipt(receipt: dict) -> list[str]:
    """Problems with a receipt: wrong schema, or a seal that does not match its content."""
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        return [f"not a {RECEIPT_SCHEMA} receipt"]
    body = {k: v for k, v in receipt.items() if k != "digest"}
    if hashlib.sha256(canonical(body).encode()).hexdigest() != receipt.get("digest"):
        return ["the digest does not match the content: the receipt was changed after it was made"]
    return []


class Guard:
    def __init__(self, forbidden, kwargs):
        self.forbidden, self.kwargs = list(forbidden), kwargs
        self.receipt = check_isolation(self.forbidden, **kwargs)
        self.exit_receipt: dict | None = None

    def recheck(self) -> dict:
        return check_isolation(self.forbidden, **self.kwargs)


@contextlib.contextmanager
def blind_guard(forbidden, **kwargs):
    """Enter a blind run: raise ``BlindViolation`` before anything runs when a forbidden tree is
    reachable; check again on leaving (a violation then means the run's results must not be
    reported as blind)."""
    guard = Guard(forbidden, kwargs)
    if not guard.receipt["ok"]:
        raise BlindViolation(guard.receipt)
    try:
        yield guard
    except BaseException as error:
        if isinstance(error, (GeneratorExit, BlindViolation)):
            raise
        # The run stopped early (an exception, Ctrl-C, a SystemExit from a stage).  Its exit check
        # still runs: a forbidden tree it reached must mark the session violated, or the session
        # stays "entered" and a later blind run in the same directory reuses its state.
        try:
            guard.exit_receipt = guard.recheck()
        except Exception:                               # noqa: BLE001 - the run's own failure is reported
            raise error from None
        if not guard.exit_receipt["ok"]:
            raise BlindViolation(guard.exit_receipt) from error
        raise
    guard.exit_receipt = guard.recheck()
    if not guard.exit_receipt["ok"]:
        raise BlindViolation(guard.exit_receipt)


# --------------------------------------------------------------------------- run-directory provenance
#
# A run directory is a cache: stage checkpoints (stage_*.json, reused when their identity matches),
# the case store (out/state: solved cases are skipped, their sources kept), the donor index, the
# refiner's answer cache.  None of that names a forbidden path or is a byte copy of a forbidden
# file, so the isolation check cannot see that an earlier, non-blind session in the same directory
# was informed by a forbidden tree.  Every decompile-run session therefore records itself in
# ``solver/blind_provenance.json``, and a blind run starts only in a fresh directory or in one whose
# every session was blind against at least the trees it forbids now.  The record is a log the kit
# keeps, not a signature: someone who edits it by hand can lie, as with any file in the run.

def provenance_path(out: str | Path) -> Path:
    return Path(out) / "solver" / "blind_provenance.json"


def _roots(forbidden) -> list[str]:
    return sorted({str(Path(root).expanduser().resolve()) for _label, root in forbidden})


_REFUSAL_RECORDS = frozenset({"blind_violation.json", "blind_entry.json"})


def _refused_attempt_only(out: Path) -> bool:
    """What a blind attempt that never started its work leaves: ``solver/`` holding nothing or only
    its refusal (or entry) receipt, and no provenance record (the run's work is recorded first)."""
    entries = list(out.iterdir())
    if len(entries) != 1 or entries[0].name != "solver" or not entries[0].is_dir() or entries[0].is_symlink():
        return False
    return all(path.name in _REFUSAL_RECORDS and path.is_file() and not path.is_symlink()
               for path in entries[0].iterdir())


def run_provenance(out: str | Path, forbidden) -> dict:
    """May a blind run against ``forbidden`` use the run directory ``out``?  ``ok`` when it does not
    exist or is empty (or holds only what a refused blind attempt wrote), or when its provenance
    record shows only blind sessions, none of them violated, each forbidding at least these trees;
    otherwise the problems are named."""
    out = Path(out)
    roots = _roots(forbidden)
    base = {"schema": PROVENANCE_SCHEMA, "run_dir": str(out), "forbidden_roots": roots}
    if not out.exists():
        return {**base, "ok": True, "verdict": "fresh", "sessions": 0}
    if not out.is_dir():
        return {**base, "ok": False, "verdict": "not a directory", "sessions": 0, "problems": ["not a directory"]}
    if not any(out.iterdir()) or _refused_attempt_only(out):
        return {**base, "ok": True, "verdict": "fresh", "sessions": 0}
    try:
        data = json.loads(provenance_path(out).read_text())
    except FileNotFoundError:
        problem = ("the run directory holds state but no blind provenance record: it was made by a run that "
                   "was not blind (or by a kit without the record); use a fresh --out")
        return {**base, "ok": False, "verdict": "unknown provenance", "sessions": 0, "problems": [problem]}
    except (OSError, ValueError) as error:
        return {**base, "ok": False, "verdict": "unreadable provenance", "sessions": 0,
                "problems": [f"the provenance record cannot be read: {error}"]}
    sessions = data.get("sessions") if isinstance(data, dict) and data.get("schema") == PROVENANCE_SCHEMA else None
    if not isinstance(sessions, list) or not sessions:
        return {**base, "ok": False, "verdict": "unknown provenance", "sessions": 0,
                "problems": ["the provenance record lists no session"]}
    problems = []
    for number, session in enumerate(sessions, 1):
        if _refused(session):                   # refused before any work: it left no state behind
            continue
        if not isinstance(session, dict) or session.get("blind") is not True:
            problems.append(f"session {number} was not blind")
            continue
        if session.get("status") == "violated":
            problems.append(f"session {number} was blind but a forbidden tree became reachable during it")
        missing = [root for root in roots if root not in set(session.get("forbidden_roots") or [])]
        if missing:
            problems.append(f"session {number} was not blind against {', '.join(missing)}")
    interrupted = sum(1 for s in sessions if isinstance(s, dict) and s.get("status") == "entered")
    return {**base, "ok": not problems, "verdict": "blind sessions only" if not problems else "tainted",
            "sessions": len(sessions), "interrupted": interrupted, "problems": problems}


def _write_provenance(out: Path, data: dict) -> None:
    import tempfile
    path = provenance_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".provenance-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(data, indent=1, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def _read_provenance(out: Path) -> dict:
    try:
        data = json.loads(provenance_path(out).read_text())
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict) or data.get("schema") != PROVENANCE_SCHEMA \
            or not isinstance(data.get("sessions"), list):
        # absent or damaged: a fresh log.  If the directory already held state, the first session
        # recorded says so ("prior_state"), and run_provenance refuses it unless it is all blind.
        data = {"schema": PROVENANCE_SCHEMA, "sessions": []}
    return data


def record_session(out: str | Path, *, blind: bool, forbidden=(), status: str = "entered", **fields) -> int:
    """Append one session to the run directory's provenance record; returns its index.  A non-blind
    session is recorded before it does any work (a run killed later still leaves the mark)."""
    import time
    out = Path(out)
    had_state = out.is_dir() and any(p.name != "solver" for p in out.iterdir())
    data = _read_provenance(out)
    session = {"blind": bool(blind), "status": status, "time": round(time.time(), 3), "pid": os.getpid(), **fields}
    if blind:
        session["forbidden_roots"] = _roots(forbidden)
    if not data["sessions"] and had_state:
        session["prior_state"] = True
        if blind:                       # state of unknown origin: never acceptable under a blind session
            session["blind"] = False
            session["status"] = "unknown prior state"
    data["sessions"].append(session)
    _write_provenance(out, data)
    return len(data["sessions"]) - 1


def update_session(out: str | Path, index: int, **fields) -> None:
    out = Path(out)
    data = _read_provenance(out)
    if 0 <= index < len(data["sessions"]) and isinstance(data["sessions"][index], dict):
        data["sessions"][index].update(fields)
        _write_provenance(out, data)


# --------------------------------------------------------------------------- run modes and reuse
#
# ``decompile-run --mode`` (``n64dk.runmodes``) records its mode in every session (``run_mode``).
# What a run directory holds was learnt under the modes of the sessions that filled it, so a run
# may continue in a directory only when every earlier session was at least as strict as it is:
# blind < research < source-tree < anything else (a session without a mode, one that was violated,
# or state of unknown origin).  A session that was refused before it did any work is ignored.

MODE_RANK = {"blind": 0, "research": 1, "source-tree": 2}


def _refused(session) -> bool:
    return isinstance(session, dict) and session.get("status") == "refused" and not session.get("prior_state")


def session_mode(session) -> str:
    """The mode a recorded session counts as: ``blind``, ``research``, ``source-tree``, or a word
    that ranks below all of them (``violated``, ``unknown``, ``blind (no mode)``, ``none``)."""
    if not isinstance(session, dict) or session.get("prior_state"):
        return "unknown"
    if session.get("status") == "violated":
        return "violated"
    mode = session.get("run_mode")
    if mode in MODE_RANK:
        # a blind or research session that the kit did not record as blind did not run under its guard
        return mode if mode == "source-tree" or session.get("blind") is True else "unknown"
    return "blind (no mode)" if session.get("blind") is True else "none"


def mode_reuse(out: str | Path, mode: str) -> dict:
    """May a ``mode`` run use the run directory ``out``?  ``ok`` when it does not exist or is empty,
    or when every recorded session (refused ones aside) ran under ``mode`` or a stricter one."""
    if mode not in MODE_RANK:
        raise GovernanceError(f"unknown run mode {mode!r}")
    out = Path(out)
    base = {"schema": PROVENANCE_SCHEMA, "run_dir": str(out), "run_mode": mode}
    if not out.exists() or (out.is_dir() and not any(out.iterdir())):
        return {**base, "ok": True, "verdict": "fresh", "sessions": 0, "modes": []}
    if not out.is_dir():
        return {**base, "ok": False, "verdict": "not a directory", "sessions": 0, "modes": [],
                "problems": ["not a directory"]}
    try:
        data = json.loads(provenance_path(out).read_text())
    except FileNotFoundError:
        return {**base, "ok": False, "verdict": "unknown provenance", "sessions": 0, "modes": [],
                "problems": ["the run directory holds state but no provenance record; use a fresh --out"]}
    except (OSError, ValueError) as error:
        return {**base, "ok": False, "verdict": "unreadable provenance", "sessions": 0, "modes": [],
                "problems": [f"the provenance record cannot be read: {error}"]}
    sessions = data.get("sessions") if isinstance(data, dict) and data.get("schema") == PROVENANCE_SCHEMA else None
    if not isinstance(sessions, list):
        return {**base, "ok": False, "verdict": "unknown provenance", "sessions": 0, "modes": [],
                "problems": ["the provenance record lists no session"]}
    problems, modes = [], []
    for number, session in enumerate(sessions, 1):
        if _refused(session):
            continue
        found = session_mode(session)
        modes.append(found)
        if MODE_RANK.get(found, len(MODE_RANK)) > MODE_RANK[mode]:
            problems.append(f"session {number} ran as {found}: a {mode} run cannot reuse its state")
    if not modes and any(p.name != "solver" for p in out.iterdir()):
        problems.append("the run directory holds state but every recorded session was refused")
    return {**base, "ok": not problems,
            "verdict": "fresh" if not modes and not problems else "reusable" if not problems else "other mode",
            "sessions": len(modes), "modes": modes, "problems": problems}
