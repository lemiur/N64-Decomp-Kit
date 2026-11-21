"""Explicit run modes (``decompile-run --mode``): blind, research-assisted and source-tree-assisted
runs whose permitted inputs are enforced when the run executes (closeout directive, item 4).

**What each mode may learn from.**

* ``blind``: the ROM, the kit (its SDK signatures, toolchains) and other titles' donors.  Every
  tree of the corpus manifest that concerns this title in a class that carries knowledge of it
  (``BLIND_CLASSES``: its public or private source, donor trees, research metadata, model output,
  anything non-redistributable), and every tree it marks blind-excluded for every title, is
  forbidden through the strict isolation guard (``--blind-forbid``).  A donor that is this title
  or another revision of it is refused (``--donors``: the run is refused; ``--library``: the entry
  is left out and listed).  The project's symbol files and catalogues may hold no name that the
  forbidden trees define, except the kit's own SDK names, placeholders (``func_80001000``) and
  names that appear as text in the ROM itself.  No external refiner (a model's memory may hold
  the title's public tree).
* ``research``: public general knowledge (SDK, other titles' trees, research notes about this
  title, an external refiner) but never this title's source tree (``SOURCE_CLASSES`` entries that
  concern it are forbidden).  A donor that is this title or another revision of it is admitted
  only when its run directory's provenance shows blind or research sessions only.
* ``source-tree``: this title's own tree is allowed; the manifest's trees concerning the title
  are recorded with their digests, and every donor's relation to the title is recorded.

**The title** is identified by the ROM of the project's first image (``main`` when there is one):
its SHA-1 (whose first 16 hex digits are the donor library's key), the internal name and game code
of its header, and the names given with ``--title``.  A manifest entry concerns the title when one
of its ``titles`` is one of those names (compared without case, spaces or punctuation), the game
code, or a prefix of 12+ hex digits of the SHA-1.  A donor is the title itself when its image has
the same SHA-1; another revision when its image's header has the same internal name or the same
two-character game id, when its name is one of the title's names, or when its root or image lies in
a tree that concerns the title; a donor sharing its root with one of these is classed with it.

**Recording.**  The mode is saved in ``solver/invocation.json`` (``run_mode``), recorded in every
provenance session (``governance.record_session``), in the isolation receipts (``run_mode``), in
``report.json`` and in ``solver/run_mode.json`` (``n64dk.run-mode/1``, sealed by the SHA-256 of
its canonical JSON like the isolation receipt).  A refused run writes
``solver/run_mode_violation.json`` and records a ``refused`` session, so the directory stays usable.
``governance.mode_reuse`` refuses a run directory any earlier session of which ran under a less
strict mode (blind after research or source-tree, research after source-tree, any mode after a
run without one).

Stated limits: names are compared as identifiers; knowledge copied by hand into configuration as
anything else (recipes, boundaries, orders) is not recognisable; a donor whose image is not a ROM
and whose name and root say nothing about the title is classed by its SHA-1 alone.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import governance as gov

SCHEMA = "n64dk.run-mode/1"
MODES = ("blind", "research", "source-tree")
# corpus classes that carry knowledge of a title they concern: forbidden in a blind run
BLIND_CLASSES = ("public-source", "private-project-source", "donor", "redistributable-metadata", "model-generated",
                 "non-redistributable")
# a title's source: forbidden in a research run too
SOURCE_CLASSES = ("public-source", "private-project-source", "donor")
ROM_MAGIC = {b"\x80\x37\x12\x40": "z64", b"\x37\x80\x40\x12": "v64", b"\x40\x12\x37\x80": "n64"}
METADATA_SUFFIXES = (".txt", ".toml", ".json", ".tsv", ".ld", ".yaml", ".yml", ".h", ".inc", ".map", ".csv")
TREE_NAME_SUFFIXES = (".c", ".s", ".txt", ".ld", ".map")
SKIP_DIRS = (".git", ".n64dk", "__pycache__")
PERMITTED = {
    "blind": "the ROM, the kit (SDK signatures, toolchains) and other titles' donors; not the title's trees, "
             "not a donor that is the title or another revision of it, not a name the title's trees define",
    "research": "public general knowledge (SDK, other titles' trees, research notes, an external refiner); "
                "not the title's source tree",
    "source-tree": "anything, including the title's own tree (recorded)",
}
# names the kit or any toolchain produces without knowing the title
PLACEHOLDER = re.compile(r"^(?:func|sub|fn|code|D|B|jtbl|jpt|L|lbl|dlabel|jlabel|RO|R)_[0-9A-Fa-f]{6,8}$")
RUNTIME_NAMES = frozenset((
    "main", "_gp", "_start", "start", "entrypoint", "boot", "idle", "memcpy", "memset", "memmove", "strlen",
    "strcpy", "strcmp", "strncpy", "strchr", "sprintf", "printf", "abs", "sqrtf", "__divdi3", "__udivdi3",
    "__moddi3", "__umoddi3", "__lshrdi3", "__ashldi3", "__ashrdi3", "__muldi3", "__cmpdi2", "__ucmpdi2",
    "__fixunssfdi", "__fixunsdfdi", "__fixsfdi", "__fixdfdi", "__floatdisf", "__floatdidf"))
_SYMBOL_DEF = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_.$]*)[ \t]*=[ \t]*(?:0[xX][0-9A-Fa-f]+|[0-9]+)[ \t]*;", re.M)
_QUOTED_NAME = re.compile(r"""\bname["']?[ \t]*[:=][ \t]*["']([A-Za-z_][A-Za-z0-9_.$]*)["']""")
_ASM_LABEL = re.compile(r"^[ \t]*(?:glabel|\.globl|\.global|dlabel|jlabel|leaf|nonleaf)[ \t]+([A-Za-z_][A-Za-z0-9_.$]*)",
                        re.M)
_C_DEFINITION = re.compile(r"^(?![ \t#/*])(?!(?:typedef|extern|return|if|else|while|for|switch|do|case|goto)\b)"
                           r"(?:[A-Za-z_][\w*]*[ \t*]+)+([A-Za-z_]\w*)[ \t]*\([^;\n]*$", re.M)


def norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


# --------------------------------------------------------------------------- identity

@dataclass
class RomIdentity:
    path: str
    sha1: str
    header_name: str = ""
    game_code: str = ""

    @property
    def key(self) -> str:
        """The donor library's key (``match.library.rom_key``)."""
        return self.sha1[:16]

    @property
    def game_id(self) -> str:
        """The two characters of the game code that name the game (not its media or region)."""
        return self.game_code[1:3] if len(self.game_code) == 4 and self.game_code[1:3].isalnum() else ""

    def as_dict(self) -> dict:
        return {"sha1": self.sha1, "key": self.key, "header_name": self.header_name, "game_code": self.game_code}


def _header(head: bytes) -> bytes:
    kind = ROM_MAGIC.get(head[:4])
    if kind == "v64":
        return bytes(b for i in range(0, len(head) - 1, 2) for b in (head[i + 1], head[i]))
    if kind == "n64":
        return b"".join(head[i:i + 4][::-1] for i in range(0, len(head) - 3, 4))
    return head


def rom_identity(path) -> RomIdentity:
    """SHA-1 of the whole file (streamed) and, for a ROM, its header's internal name and game code."""
    digest, head = hashlib.sha1(), b""
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            if len(head) < 64:
                head += block[:64 - len(head)]
            digest.update(block)
    name = code = ""
    if head[:4] in ROM_MAGIC and len(head) >= 64:
        header = _header(head)
        name = header[0x20:0x34].decode("ascii", "replace").strip(" \x00")
        code = header[0x3B:0x3F].decode("ascii", "replace")
        if not code.isalnum():
            code = ""
    return RomIdentity(str(path), digest.hexdigest(), name, code)


@dataclass
class Title:
    rom: RomIdentity
    names: list = field(default_factory=list)          # --title

    def aliases(self) -> set[str]:
        return {norm(n) for n in [*self.names, self.rom.header_name] if norm(n)}

    def matches(self, label) -> str | None:
        """Why a manifest ``titles`` entry names this title, or None."""
        text = str(label).strip()
        low = text.lower().removeprefix("sha1:")
        if re.fullmatch(r"[0-9a-f]{12,40}", low) and self.rom.sha1.startswith(low):
            return "ROM SHA-1"
        if self.rom.game_code and text == self.rom.game_code:
            return "game code"
        if norm(text) and norm(text) in self.aliases():
            return f"title name {text!r}"
        return None

    def relation(self, other: RomIdentity | None) -> tuple[str, str]:
        if other is None:
            return "other", ""
        if other.sha1 == self.rom.sha1:
            return "self", "same ROM (SHA-1)"
        if norm(other.header_name) and norm(other.header_name) in self.aliases():
            return "revision", f"same internal name {other.header_name!r}"
        if self.rom.game_id and other.game_id == self.rom.game_id:
            return "revision", f"same game id {other.game_id!r}"
        return "other", ""

    def as_dict(self) -> dict:
        return {**self.rom.as_dict(), "names": list(self.names)}


# --------------------------------------------------------------------------- the manifest's trees

def select_trees(manifest: "gov.Manifest | None", title: Title, mode: str) -> list[dict]:
    """The manifest's trees a ``mode`` run forbids (blind, research) or records (source-tree)."""
    rows = []
    for entry in (manifest.entries if manifest is not None else []):
        why = next((w for w in map(title.matches, entry.titles or []) if w), None)
        general = not entry.titles and entry.flag("blind_exclude")
        if mode == "blind":
            take = bool(why and (entry.data_class in BLIND_CLASSES or entry.flag("blind_exclude"))) or general
        elif mode == "research":
            take = bool(why or general) and entry.data_class in SOURCE_CLASSES
        else:
            take = bool(why)
        if take:
            for root in manifest.root_paths(entry):
                rows.append({"label": entry.name, "root": str(Path(root).expanduser().resolve()),
                             "class": entry.data_class, "why": why or "blind-excluded for every title"})
    return rows


# --------------------------------------------------------------------------- donors

def donor_provenance(root: Path) -> str:
    """``blind/research only`` when the run directory a donor comes from (its root, or the parent of
    a library entry's ``queue``) shows only blind or research sessions; otherwise why not."""
    for folder in (Path(root), Path(root).parent):
        if gov.provenance_path(folder).is_file():
            verdict = gov.mode_reuse(folder, "research")
            return "blind/research only" if verdict["ok"] else "; ".join(verdict.get("problems") or [verdict["verdict"]])
    return "no provenance record"


def donor_verdict(spec, origin: str, title: Title, mode: str, title_roots: list[dict]) -> dict:
    root = Path(os.path.expanduser(str(spec.root))).resolve()
    image = Path(os.path.expanduser(str(spec.image))).resolve()
    row = {"donor": spec.name, "origin": origin, "root": str(root), "image": str(image)}
    inside = next((t for t in title_roots for p in (root, image) if gov._within(p, Path(t["root"]))), None)
    if inside is not None:
        relation, why = "title tree", f"it lies in {inside['label']}, a tree of this title"
    else:
        try:
            identity = rom_identity(image)
        except OSError:
            identity = None
        relation, why = title.relation(identity)
        if relation == "other" and norm(spec.name) and norm(spec.name) in title.aliases():
            relation, why = "revision", f"its name {spec.name!r} is the title's"
    row.update(relation=relation, why=why)
    return _admit(row, mode)


def _admit(row: dict, mode: str) -> dict:
    """Whether ``mode`` may use a classified donor: any relation in source-tree; only ``other`` in
    blind; in research also this title or a revision when its run directory was blind or research."""
    row.pop("provenance", None)
    if row["relation"] == "other" or mode == "source-tree":
        row["admitted"] = True
    elif mode == "research" and row["relation"] != "title tree":
        row["provenance"] = donor_provenance(Path(row["root"]))
        row["admitted"] = row["provenance"] == "blind/research only"
    else:
        row["admitted"] = False
    return row


def _share_roots(rows: list[dict], mode: str) -> None:
    """Donors that share a root with a donor of this title (a run's other images, whose segment files
    carry no header) are classed with it."""
    flagged = {}
    for row in rows:
        if row.get("root") and row["relation"] != "other":
            flagged.setdefault(row["root"], row)
    for row in rows:
        first = flagged.get(row.get("root"))
        if row["relation"] == "other" and first is not None:
            row.update(relation=first["relation"], why=f"it shares its root with donor {first['donor']} ({first['why']})")
            _admit(row, mode)


def library_entries(library, key: str) -> list[tuple[Path, list]]:
    """``(entry file, donor specs whose image exists)`` for every other title in a donor library,
    exactly what ``match.library.donors`` offers."""
    from .match import library as libmod
    from .match import transplant
    out = []
    for path in libmod.files(library, key):
        try:
            specs = transplant.load_donors(path)
        except Exception:                                    # noqa: BLE001 - skipped, as libmod does
            continue
        out.append((path, [spec for spec in specs if Path(spec.image).is_file()]))
    return out


# --------------------------------------------------------------------------- title-specific names

_GENERAL: frozenset | None = None


def general_names() -> frozenset:
    """Names the kit knows without the title: its SDK signature database."""
    global _GENERAL
    if _GENERAL is None:
        try:
            from .sdk.db import Database
            _GENERAL = frozenset(Database.load().names())
        except Exception:                                    # noqa: BLE001 - no database: nothing general
            _GENERAL = frozenset()
    return _GENERAL


def _read(path: Path, whole: bool = False) -> str | None:
    try:
        if not whole and path.stat().st_size > gov.CONTENT_MAX:
            return None
        return path.read_text(errors="replace")
    except OSError:
        return None


def _walk(root: Path, limit: int, skip=()):
    """Files under ``root`` (not following links), and whether the walk stopped at ``limit``."""
    files, count = [], 0
    if root.is_file():
        return [root], False
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and Path(folder, d).resolve() not in skip)
        for name in sorted(names):
            count += 1
            if count > limit:
                return files, True
            path = Path(folder, name)
            if not path.is_symlink():
                files.append(path)
    return files, False


def tree_names(roots: list[dict], limit: int) -> tuple[dict, dict]:
    """Names the forbidden trees define (symbol lists, linker scripts, assembly labels, C function
    definitions): ``{name: "label/relative path"}`` and the walk's counts."""
    names: dict[str, str] = {}
    stats = {"files": 0, "truncated": []}
    for tree in roots:
        root = Path(tree["root"])
        if not root.exists():
            continue
        files, truncated = _walk(root, limit)
        if truncated:
            stats["truncated"].append(tree["label"])
        for path in files:
            if path.suffix.lower() not in TREE_NAME_SUFFIXES:
                continue
            text = _read(path)
            if text is None:
                continue
            stats["files"] += 1
            where = f"{tree['label']}/{path.relative_to(root).as_posix() if path != root else path.name}"
            found = set(_SYMBOL_DEF.findall(text)) | set(_ASM_LABEL.findall(text))
            if path.suffix.lower() == ".c":
                found |= set(_C_DEFINITION.findall(text))
            for name in found:
                names.setdefault(name, where)
    return names, stats


def metadata_files(project: Path, limit: int, skip=()) -> tuple[list[Path], list[Path], bool]:
    """The title metadata a run reads: the symbol files and catalogues ``n64dk.toml`` names (read
    whole, wherever they are), and every metadata-like file in the project (bounded walk)."""
    named = []
    try:
        data = tomllib.loads((project / "n64dk.toml").read_text())
    except (OSError, ValueError):
        data = {}
    run = data.get("run") or {}
    lists = [run.get("symbols") or []] + [(table or {}).get("symbols") or []
                                          for table in (run.get("images") or {}).values() if isinstance(table, dict)]
    for table in (data.get("images") or {}).values():
        if isinstance(table, dict):
            lists.append([table["catalogue"]] if table.get("catalogue") else [])
            lists.append(table.get("symbols") or [])
    for values in lists:
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, str):
                path = Path(os.path.expanduser(value))
                path = (path if path.is_absolute() else project / path).resolve()
                if path.is_file() and path not in named:
                    named.append(path)
    walked, truncated = _walk(project, limit, skip)
    walked = [p for p in walked if p.suffix.lower() in METADATA_SUFFIXES and p.resolve() not in named]
    return named, walked, truncated


def metadata_names(text: str) -> set[str]:
    return set(_SYMBOL_DEF.findall(text)) | set(_QUOTED_NAME.findall(text))


def name_check(project: Path, rom_path, roots: list[dict], limit: int, skip=()) -> dict:
    """Blind runs: names in the project's metadata that a forbidden tree defines, other than
    placeholders, runtime names, the kit's SDK names and names present as text in the ROM."""
    index, stats = tree_names(roots, limit)
    named, walked, truncated = metadata_files(project, limit, skip)
    seen: dict[str, list[str]] = {}
    skipped = []
    for path in named + walked:
        text = _read(path, whole=path in named)
        if text is None:
            skipped.append(str(path))
            continue
        for name in metadata_names(text):
            if name in index and not PLACEHOLDER.match(name) and name not in RUNTIME_NAMES:
                seen.setdefault(name, []).append(str(path))
    if seen:
        general = general_names()
        seen = {n: w for n, w in seen.items() if n not in general}
    if seen:
        try:
            rom = Path(rom_path).read_bytes()
        except OSError:
            rom = b""
        seen = {n: w for n, w in seen.items() if n.encode() not in rom}
    leaked = [{"name": n, "files": sorted(set(w))[:3], "defined_in": index[n]} for n, w in sorted(seen.items())]
    problems = []
    if stats["truncated"] or truncated:
        problems.append("the name check stopped at the walk limit (" + ", ".join(
            stats["truncated"] + (["project"] if truncated else [])) + "): raise --blind-walk-limit")
    if leaked:
        problems.append(f"{len(leaked)} name(s) in the project's metadata are defined by the title's trees "
                        f"(first: {leaked[0]['name']} in {Path(leaked[0]['files'][0]).name}, from "
                        f"{leaked[0]['defined_in']})")
    return {"checked": True, "tree_names": len(index), "tree_files": stats["files"],
            "metadata_files": len(named) + len(walked) - len(skipped), "skipped_large": skipped[:20],
            "leaked": leaked[:50], "leaked_count": len(leaked), "problems": problems}


# --------------------------------------------------------------------------- the plan

@dataclass(eq=False)
class ModePlan:
    """What a ``--mode`` run may use, decided before it starts.  Not JSON: kept off the saved
    invocation and the isolation guard's configuration (``args._run_mode_plan``)."""
    mode: str
    title: Title
    manifest: str | None = None
    manifest_sha256: str | None = None
    forbidden: list = field(default_factory=list)        # manifest trees forbidden (blind, research)
    explicit_forbid: list = field(default_factory=list)  # --blind-forbid
    recorded: list = field(default_factory=list)         # manifest trees concerning the title (source-tree)
    donors: list = field(default_factory=list)           # one verdict per donor (--donors, --library)
    library_checked: list = field(default_factory=list)  # library entry files the plan classified
    library_excluded: list = field(default_factory=list)
    names: dict = field(default_factory=dict)
    refine: str | None = None
    problems: list = field(default_factory=list)
    run_dir: dict = field(default_factory=dict)

    def forbidden_roots(self) -> list[str]:
        out = []
        for root in [*self.explicit_forbid, *(row["root"] for row in self.forbidden)]:
            if root not in out:
                out.append(root)
        return out

    def check_explicit(self, specs) -> None:
        """At the donor stage: every ``--donors`` entry must be one the plan admitted."""
        admitted = {(row["donor"], row["image"]) for row in self.donors
                    if row["origin"] == "donors" and row["admitted"]}
        for spec in specs:
            image = str(Path(os.path.expanduser(str(spec.image))).resolve())
            if (spec.name, image) not in admitted:
                raise RuntimeError(f"run mode {self.mode}: donor {spec.name} was not admitted when the run started")

    def library_donors(self, library, key: str) -> list:
        """The library's donors this run may use: entries classified when the run started and admitted."""
        out = []
        for path, specs in library_entries(library, key):
            if path.name in self.library_checked and path.name not in self.library_excluded:
                out += specs
        return out

    def library_identity(self, rows: list) -> list:
        return [row for row in rows if row[0] in self.library_checked and row[0] not in self.library_excluded]

    def receipt(self, status: str) -> dict:
        body = {"schema": SCHEMA, "run_mode": self.mode, "status": status, "permitted": PERMITTED[self.mode],
                "title": self.title.as_dict(),
                "corpus_manifest": ({"path": self.manifest, "sha256": self.manifest_sha256}
                                    if self.manifest else None),
                "forbidden": self.forbidden, "explicit_forbid": self.explicit_forbid,
                "recorded_trees": self.recorded, "donors": self.donors,
                "library_excluded": self.library_excluded, "names": self.names, "refine": self.refine,
                "run_dir": self.run_dir, "problems": self.problems,
                "limits": ["names are compared as identifiers; knowledge copied by hand into configuration in "
                           "any other form is not recognisable",
                           "a donor whose image is not a ROM, and whose name and root say nothing about the "
                           "title, is classed by its SHA-1 alone",
                           "the seal is a digest of the receipt, not a signature"]}
        body["digest"] = hashlib.sha256(gov.canonical(body).encode()).hexdigest()
        return body

    def write(self, out: Path, status: str, name: str = "run_mode.json") -> Path:
        path = Path(out) / "solver" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.receipt(status), indent=1, sort_keys=True, default=str) + "\n")
        return path

    def summary(self) -> dict:
        return {"mode": self.mode, "title": self.title.rom.header_name or (self.title.names or [""])[0],
                "rom_sha1": self.title.rom.sha1,
                "forbidden_trees": [row["label"] for row in self.forbidden] + list(self.explicit_forbid),
                "recorded_trees": [row["label"] for row in self.recorded],
                "donors_refused": [row["donor"] for row in self.donors if not row["admitted"]],
                "library_excluded": list(self.library_excluded),
                "names_leaked": self.names.get("leaked_count") if self.names.get("checked") else None}


def _refine_configured(args, project: Path) -> str | None:
    if getattr(args, "refine_command", None):
        return "--refine-command"
    try:
        data = tomllib.loads((project / "n64dk.toml").read_text())
    except (OSError, ValueError):
        return None
    return "[run.refine] command" if ((data.get("run") or {}).get("refine") or {}).get("command") else None


def build_plan(args) -> ModePlan:
    """Classify everything a ``--mode`` run would use; ``plan.problems`` lists what the mode does not
    permit (the caller refuses the run when it is not empty).  A configuration that cannot be a run
    of this mode at all raises ``SystemExit`` before anything is written."""
    from .match import images as imagemod
    from .match import transplant
    mode = getattr(args, "run_mode", None)
    if mode not in MODES:
        raise SystemExit(f"--mode: one of {', '.join(MODES)}")
    project = Path(args.project).resolve()
    if mode == "source-tree" and getattr(args, "blind_forbid", None):
        raise SystemExit("--mode source-tree with --blind-forbid: a source-tree run is not blind; use --mode blind "
                         "or research")
    registry = imagemod.load_registry(project)
    if not registry:
        raise SystemExit("no [images.<name>] tables in the project's n64dk.toml: the title cannot be identified")
    first = registry["main"] if "main" in registry else registry[sorted(registry)[0]]
    title = Title(rom_identity(first.path), list(getattr(args, "run_title", None) or []))
    manifest_path = getattr(args, "corpus_manifest", None)
    manifest = gov.Manifest.load(manifest_path) if manifest_path else None
    plan = ModePlan(mode, title)
    if manifest_path:
        plan.manifest = str(Path(manifest_path).expanduser().resolve())
        plan.manifest_sha256 = gov.file_digest(Path(plan.manifest))
    plan.explicit_forbid = [str(Path(p).expanduser().resolve()) for p in (getattr(args, "blind_forbid", None) or [])]
    title_trees = select_trees(manifest, title, "source-tree")
    limit = getattr(args, "blind_walk_limit", None) or 50_000
    if mode == "source-tree":
        for row in title_trees:
            root = Path(row["root"])
            try:
                row["tree_digest"], row["files"] = gov.tree_digest(root, limit) if root.exists() else (None, 0)
            except gov.GovernanceError as error:
                row["tree_digest"], row["note"] = None, str(error)
        plan.recorded = title_trees
    else:
        plan.forbidden = select_trees(manifest, title, mode)
        if not plan.forbidden and not plan.explicit_forbid:
            tried = sorted(title.aliases()) + [title.rom.sha1[:12]] + ([title.rom.game_code] if title.rom.game_code else [])
            raise SystemExit(f"--mode {mode}: no tree to forbid: the corpus manifest names none for this title "
                             f"(tried {', '.join(tried)}) and no --blind-forbid was given; a {mode} run checked "
                             "against nothing is not one (name the title with --title)")
        if mode == "blind":
            plan.refine = _refine_configured(args, project)
            if plan.refine:
                raise SystemExit(f"--mode blind with an external refiner ({plan.refine}): a model's memory may hold "
                                 "the title's public tree; use --mode research")
    # donors: explicit (--donors after --donor/--exclude-donor) and the library's entries
    # a donor rooted in one of the title's trees is that tree (research: only the trees it forbids)
    roots_of_title = ((plan.forbidden if mode == "research" else title_trees)
                      + [{"label": f"forbid:{Path(p).name}", "root": p} for p in plan.explicit_forbid])
    explicit: list[dict] = []
    if getattr(args, "donors", None) and Path(args.donors).is_file():
        for spec in transplant.load_donors(args.donors, getattr(args, "donor", None) or None,
                                           getattr(args, "exclude_donor", None) or None):
            explicit.append(donor_verdict(spec, "donors", title, mode, roots_of_title))
        _share_roots(explicit, mode)
        plan.donors += explicit
    library = getattr(args, "library", None)
    if library:
        library = Path(library).expanduser().resolve()
        own = library / f"{title.rom.key}.toml"
        if own.is_file():                   # never offered by match.library: recorded for completeness
            plan.donors.append({"donor": own.name, "origin": f"library:{own.name}", "relation": "self",
                                "why": "the library's entry for this ROM", "admitted": False})
        for path, specs in library_entries(library, title.rom.key):
            plan.library_checked.append(path.name)
            rows = [donor_verdict(spec, f"library:{path.name}", title, mode, roots_of_title) for spec in specs]
            _share_roots(rows, mode)
            plan.donors += rows
            if any(not row["admitted"] for row in rows):
                plan.library_excluded.append(path.name)
    refused = [row for row in plan.donors if row["origin"] == "donors" and not row["admitted"]]
    for row in refused:
        plan.problems.append(f"donor {row['donor']} is {'this title' if row['relation'] == 'self' else row['relation']}"
                             f" ({row['why']}){'; ' + row['provenance'] if row.get('provenance') else ''}: "
                             f"not permitted in a {mode} run (leave it out with --exclude-donor)")
    # title-specific names (blind)
    if mode == "blind":
        out = Path(args.out).resolve() if getattr(args, "out", None) else None
        names_roots = plan.forbidden + [{"label": f"forbid:{Path(p).name}", "root": p} for p in plan.explicit_forbid]
        plan.names = name_check(project, first.path, names_roots, limit, skip={out} if out else ())
        plan.problems += plan.names["problems"]
    else:
        plan.names = {"checked": False,
                      "why": ("research may use the title's public research; byte copies of its source tree's "
                              "symbol lists are still refused by the isolation guard") if mode == "research" else
                             "source-tree runs may use the title's names"}
    return plan
