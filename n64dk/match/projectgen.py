"""Reversible project generator: a finished export as a maintainable decompilation project tree.

``decompile-export`` writes what a run settled: one source per matched function, the function
table, recipes, symbols.  This module turns that tree, plus the ``link-layout`` outputs when given,
into a project a person can keep working in:

=====================  ====================================================================
``src/<image>/``       strictly exact C, one function per file, as verified
``src_assumed/``       byte-identical C that leans on an address nothing independent confirms
``donors/``            library and shared units the donor stage verified (sources and headers)
``asm/<image>/``       verified hand-written assembly
``asm/generated/``     assembly that assembles to the function's bytes (disassembly, not decompilation)
``rsp/``               units of the RSP class, kept apart from the CPU code
``raw/``               units nothing reproduces yet: a stub that includes (``.incbin``) bytes cut at build
                       time from the user's own ROM; ``raw/manifest.tsv`` names each by image,
                       offset, size and SHA-256.  No byte of the ROM is stored anywhere in the tree
``include/``           generated headers (unit addresses, assembler macros)
``symbols/``           the export's symbol files
``linker/``            a text placement script per image, and the inferred link order if given
``recipes.tsv``        per unit: pipeline, flags, source, the lane its verification uses
``toolchain.lock.json``  the toolchain manifest (``toolmanifest.build``)
``Makefile``, ``tools/``  ``make extract`` (raw bytes from your ROM), ``make verify`` (each unit
                       rebuilt and strictly compared against your ROM), ``make verify-receipts``
``receipts/``          proof receipts copied from the export when their source digest is this source's
``provenance/``        one record per unit: where it came from and the digest of everything it rests on
``unresolved/``        a dossier per unit that is not settled C
``STATUS.tsv/.md``     exact / assumed / assembly / raw per unit
``GENERATED.json``     the manifest: per-unit input digests and the digest of every generated file
=====================  ====================================================================

**Reversible and incremental.**  Generating again into the same directory rewrites only files
whose content changes; a unit whose inputs changed (its source moved from raw to assembly to C,
its recipe, its receipt) gets new files and its old receipt is dropped unless the new source is
the one the receipt covers; generated files that are no longer produced are removed.  A file
changed by hand since the last generation is never overwritten or removed (it is reported as a
conflict) unless ``force``.  Generation is deterministic: no timestamps, sorted order, no
machine paths, so generating twice gives byte-identical trees.

**Strictness.**  Statuses are copied from the export (which copied the strict verifier's
verdicts); nothing here verifies, promotes or grants credit.  A source file the export kept that
holds raw words (``handasm.as_words``: ROM words as ``.word`` lines) is not copied: that unit is
``raw``.  Binary files are never copied.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "n64dk-generated-project/1"
MANIFEST = "GENERATED.json"
STATUSES = ("exact", "assumed", "assembly", "raw")
RAW_MARK = "raw words"                       # handasm.as_words' header (proofreceipt.RAW_MARK)
TEXT_SUFFIXES = frozenset((".c", ".h", ".s", ".asm", ".inc", ".tsv", ".txt", ".toml", ".md", ".json", ".ld"))
ROM_SUFFIXES = (".z64", ".n64", ".v64", ".rom")
ASSEMBLY_RECIPE = "gnu-as -march=vr4300 -mabi=o64"   # handasm.RECIPE
CREDIT = ("a generated project tree: it copies results the strict verifier settled and grants no credit; "
          "only `make verify` / `make verify-receipts` against your own ROM confirm them again")
# export outcome -> (status, kind)
OUTCOMES = {"exact": ("exact", "c"), "awaiting-symbols": ("assumed", "c"),
            "donor": ("exact", "donor-c"), "donor-awaiting-symbols": ("assumed", "donor-c"),
            "assembly": ("assembly", "handwritten"), "donor-assembly": ("assembly", "donor-asm"),
            "donor-assembly-awaiting-symbols": ("assumed", "donor-asm")}
ASSEMBLY_KINDS = ("handwritten", "donor-asm", "generated")


class GenerationError(ValueError):
    """The generator refuses: a reason, never a partial tree."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _json(value) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()


def receipt_name(image: str, name: str, vram: int) -> str:
    """Where a receipt lives under a receipt root (the layout of ``proofreceipt.receipt_path``)."""
    return f"{_safe(image)}/{_safe(name)}-{vram & 0xFFFFFFFF:08X}.json"


# ---------------------------------------------------------------------------------------------
# inputs


def resolve_export(source) -> Path:
    """The export tree: ``source`` itself or ``source/export`` (a run directory holding its export)."""
    source = Path(source).expanduser()
    for candidate in (source, source / "export"):
        if (candidate / "FUNCTIONS.tsv").is_file():
            return candidate.resolve()
    raise GenerationError(f"{source}: no FUNCTIONS.tsv here or in export/: give a decompile-export tree "
                          "(`n64dk decompile-export --project P --run RUN --out EXPORT` first)")


def check_output(out, protected=(), roms=()) -> Path:
    """Refuse an output directory that is, holds or lies inside an input (the export, the project),
    the kit's own repository, the folder of a ROM in use, or a git work tree (unless it already is
    a generated project being regenerated in place), or a non-empty directory that is not one."""
    out = Path(out).expanduser().resolve()
    if out == Path(out.anchor) or out == Path.home().resolve():
        raise GenerationError(f"{out}: refusing the filesystem root or the home directory")
    for item in protected:
        item = Path(item).expanduser().resolve()
        if out == item or item in out.parents or out in item.parents:
            raise GenerationError(f"{out}: overlaps the input {item}; write the project elsewhere")
    kit = Path(__file__).resolve().parents[2]
    if out == kit or kit in out.parents or out in kit.parents:
        raise GenerationError(f"{out}: overlaps the kit's own tree {kit}")
    for rom in roms:
        folder = Path(rom).expanduser().resolve().parent
        if out == folder or folder in out.parents:
            raise GenerationError(f"{out}: inside {folder}, the folder of a ROM in use; keep the project apart from ROMs")
    generated = (out / MANIFEST).is_file()
    for folder in (out, *out.parents):
        if (folder / ".git").exists() and not generated:
            raise GenerationError(f"{out}: inside the git work tree {folder}; generate elsewhere (a generated project "
                                  "may be committed and regenerated in place afterwards)")
    if out.exists():
        if not out.is_dir():
            raise GenerationError(f"{out}: not a directory")
        if any(p.suffix.lower() in ROM_SUFFIXES for p in out.iterdir()):
            raise GenerationError(f"{out}: holds a ROM image; keep the project apart from ROMs")
        if not generated and any(out.iterdir()):
            raise GenerationError(f"{out}: not empty and not a generated project ({MANIFEST} missing)")
    return out


@dataclass
class Unit:
    image: str
    function: str
    vram: int
    offset: int
    size: int
    code_size: int
    outcome: str
    how: str = ""
    source: str = ""                 # path in the export
    distance: str = ""
    reproduced_by: str = ""
    unit_class: str = ""
    status: str = "raw"
    kind: str = "none"
    path: str = ""                   # path in the generated tree
    content: bytes | None = None     # the source copied (None: nothing to copy)
    recipe: dict = field(default_factory=dict)
    object: str = ""
    raw_sha256: str | None = None
    receipt: str = "none"
    notes: list = field(default_factory=list)
    inputs: str = ""

    @property
    def id(self) -> str:
        return f"{self.image}/{self.function}"


def _hex(text: str, default: int = 0) -> int:
    try:
        return int(str(text).strip(), 16)
    except ValueError:
        return default


def read_functions(export: Path) -> list[Unit]:
    """FUNCTIONS.tsv rows (by header name: older exports lack ``unit_class``)."""
    lines = (Path(export) / "FUNCTIONS.tsv").read_text().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    units = []
    for line in lines[1:]:
        if not line.strip():
            continue
        row = dict(zip(header, line.split("\t")))
        units.append(Unit(row.get("image", ""), row.get("function", ""), _hex(row.get("vram", "")),
                          _hex(row.get("rom_offset", "")), _hex(row.get("size", "")),
                          _hex(row.get("code_size", ""), _hex(row.get("size", ""))), row.get("outcome", ""),
                          row.get("how", ""), row.get("source", ""), row.get("distance", ""),
                          row.get("reproduced_by", ""), row.get("unit_class", "")))
    return sorted(units, key=lambda u: (u.image, u.vram, u.function))


def _text(path: Path) -> bytes | None:
    """A text file's bytes, or None for binaries, unknown suffixes and raw-word listings."""
    if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    data = path.read_bytes()
    if b"\0" in data or RAW_MARK.encode() in data:
        return None
    return data


def classify(unit: Unit, export: Path) -> None:
    """Status, kind and generated path of one unit (see the module docstring)."""
    found = Path(export) / unit.source if unit.source else None
    content = _text(found) if found is not None and found.is_file() else None
    raw_words = found is not None and found.is_file() and RAW_MARK.encode() in found.read_bytes()
    if unit.outcome in OUTCOMES and content is not None:
        unit.status, unit.kind = OUTCOMES[unit.outcome]
    elif unit.reproduced_by.startswith("assembly (not decompiled)") and content is not None:
        unit.status, unit.kind = "assembly", "generated"
    else:
        unit.status, unit.kind = "raw", "raw-words" if raw_words else "none"
        if unit.outcome in OUTCOMES:
            unit.notes.append(f"the run settled it ({unit.outcome}) but the export holds no usable source for it")
        content = None
    unit.content = content
    image, function = _safe(unit.image), _safe(unit.function)
    rsp = unit.unit_class == "rsp"
    if unit.kind == "c":
        unit.path = f"{'src' if unit.status == 'exact' else 'src_assumed'}/{image}/{function}.c"
    elif unit.kind in ("donor-c", "donor-asm"):
        unit.path = unit.source                      # donors/<image>/src/<donor>/...: the donor tree as exported
    elif unit.kind == "handwritten":
        unit.path = f"{'rsp' if rsp else 'asm'}/{image}/{function}.s"
    elif unit.kind == "generated":
        unit.path = f"{'rsp' if rsp else 'asm'}/generated/{image}/{function}.s"
    else:
        unit.path = f"{'rsp/raw' if rsp else 'raw'}/{image}/{function}.s"


def parse_recipe(unit: Unit, donor_recipes: dict) -> dict:
    if unit.kind in ("donor-c", "donor-asm"):
        candidates = donor_recipes.get((unit.image, unit.source), [])
        found = [recipe for start, end, recipe in candidates if start <= unit.offset < end]
        if len(found) == 1:
            return dict(found[0])
        if len(found) > 1:
            # Prefer the narrowest containing unit.  This handles a function
            # source reused at many target addresses without conflating those
            # placements, while retaining whole-TU donor rows.
            found.sort(key=lambda recipe: _hex(recipe["unit_size"]))
            return dict(found[0])
        return {"pipeline": "", "flags": "", "note": f"not found in donors/{unit.image}/proposed_*.tsv"}
    if unit.kind == "c":
        text = unit.how.split(" (", 1)[0].strip()
        pipeline, _sep, flags = text.partition(" ")
        return {"pipeline": pipeline, "flags": flags.strip()}
    if unit.kind in ("handwritten", "generated"):
        return {"pipeline": "gnu-as", "flags": "", "assembler": ASSEMBLY_RECIPE}
    return {"pipeline": "", "flags": "", "note": "raw: cut from your ROM at build time"}


def queue_placements(export: Path) -> dict:
    """(image, vram) -> ``--section-base=`` flags for the sections of a strictly exact unit that the strict
    comparison placed by their contents (``manifests/placed_sections.tsv``: a switch's jump table).  The
    recipe a unit was verified under does not state them; a unit linked without them has that section at a
    scratch address and its code differs from the ROM."""
    out: dict[tuple[str, int], list[str]] = {}
    try:
        lines = (Path(export) / "manifests" / "placed_sections.tsv").read_text(errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        cells = line.split("\t")
        if line.startswith("#") or len(cells) < 4:
            continue
        try:
            key = (cells[0], int(cells[1], 16))
            int(cells[3], 16)
        except ValueError:
            continue
        out.setdefault(key, []).append(f"--section-base={cells[2]}:{cells[3]}")
    return out


def donor_recipes(export: Path) -> dict:
    """(image, donor source) -> target intervals and their recipes.

    A donor source may be transplanted at many target addresses (small SDK
    leaves and no-op callbacks are common).  Keeping only the first row merges
    unrelated placements into one link object and silently turns the rest raw.
    """
    out = {}
    root = Path(export) / "donors"
    if not root.is_dir():
        return out
    for image_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(image_dir.glob("proposed_*.tsv")):
            text = path.read_text(errors="replace")
            lines = text.splitlines()
            first = next((line for line in lines if line.startswith("#")), "# ")
            pipeline = (first[1:].split() or [""])[0]
            donor = re.search(r"relative to\s*\n#\s*src/([^/\s]+)/", text)
            if donor is None:
                continue
            for line in lines:
                if not line.strip() or line.startswith("#"):
                    continue
                cells = line.split("\t")
                if len(cells) < 3:
                    continue
                recipe = {"pipeline": pipeline, "flags": cells[3].strip() if len(cells) > 3 else "",
                          "unit_offset": cells[0], "unit_size": cells[1],
                          "manifest": f"donors/{image_dir.name}/{path.name}"}
                start, end = _hex(cells[0]), _hex(cells[0]) + _hex(cells[1])
                for relative in _donor_relative(cells[2], donor.group(1)):
                    key = (image_dir.name, f"donors/{image_dir.name}/src/{donor.group(1)}/{relative}")
                    out.setdefault(key, []).append((start, end, recipe))
    return out


def _donor_relative(source: str, donor: str) -> list[str]:
    """The path(s) a manifest row's SOURCE has below the donor's directory.  A source-tree row is already
    relative; an SDK assembly row names the original file (``<home>/.../<donor>/asm/os/interrupt.s``) while
    the export holds it as ``src/<donor>/os/interrupt.s``: both tails below the donor directory are offered."""
    if not source.startswith(("<", "/")):
        return [source]
    parts = source.replace("\\", "/").split("/")
    if donor not in parts:
        return []
    tail = parts[len(parts) - parts[::-1].index(donor):]
    return ["/".join(tail[i:]) for i in range(max(1, len(tail) - 1))]


def link_objects(path: Path | None) -> dict:
    """segment -> sorted [(text start, text end, link-layout unit)] from ``objects.tsv``."""
    out: dict[str, list] = {}
    if path is None or not Path(path).is_file():
        return out
    lines = Path(path).read_text().splitlines()
    if not lines:
        return out
    header = lines[0].split("\t")
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t")))
        if not row.get("text_start") or not row.get("text_size"):
            continue
        start = _hex(row["text_start"])
        out.setdefault(row.get("segment", ""), []).append((start, start + _hex(row["text_size"]), row.get("unit", "")))
    for items in out.values():
        items.sort()
    return out


def _object_of(objects: dict, image: str, vram: int) -> str:
    items = objects.get(image) or []
    index = bisect.bisect_right(items, (vram, 0xFFFFFFFFFF, "\U0010FFFF")) - 1
    if index >= 0 and items[index][0] <= vram < items[index][1]:
        return items[index][2]
    return ""


def _image_sources(export: Path) -> dict[str, str]:
    """image -> "rom" when the configured image is the ROM itself, else the file name an
    ``n64dk decompile-init`` of the ROM writes for it."""
    out = {}
    try:
        config = tomllib.loads((Path(export) / "config" / "n64dk.toml").read_text())
    except (OSError, ValueError):
        return out
    for name, table in (config.get("images") or {}).items():
        if isinstance(table, str):
            table = {"path": table}
        path = str(table.get("composite") or table.get("path") or table.get("raw") or table.get("rom") or "")
        out[name] = "rom" if Path(path).suffix.lower() in ROM_SUFFIXES else Path(path).name
    return out


def _rom_sha1(export: Path) -> str | None:
    readme = Path(export) / "README.md"
    found = re.search(r"ROM SHA-1: `([0-9a-f]{40})`", readme.read_text(errors="replace")) if readme.is_file() else None
    return found.group(1) if found else None


def _assumptions(export: Path) -> dict:
    out: dict = {}
    path = Path(export) / "manifests" / "assumptions.tsv"
    if path.is_file():
        for line in path.read_text().splitlines():
            cells = line.split("\t")
            if line.startswith("#") or len(cells) < 5:
                continue
            out.setdefault((cells[0], cells[1]), []).append(f"{cells[2]} {cells[3]} at {cells[4]}")
    return out


# ---------------------------------------------------------------------------------------------
# rendering


EXTRACT_SCRIPT = '''#!/usr/bin/env python3
"""Cut the raw units' bytes from your own ROM (generated by `n64dk decompile-project`).

No byte of the ROM is stored in this project.  raw/manifest.tsv names each raw unit by image,
offset, size and SHA-256; this script reads those ranges from your ROM (or, for an image that is
not the ROM itself, from the image file `n64dk decompile-init` writes for it) and writes
build/raw/<image>/<function>.bin after checking each digest.

    python3 tools/extract_raw.py --rom baserom.z64 [--image NAME=PATH ...] [--out build/raw]
"""
import argparse
import hashlib
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="cut the raw units from your ROM")
    parser.add_argument("--rom", required=True)
    parser.add_argument("--image", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--out", default="build/raw")
    parser.add_argument("--manifest", default="raw/manifest.tsv")
    parser.add_argument("--any-rom", action="store_true", help="do not require the recorded ROM SHA-1")
    args = parser.parse_args()
    rom = Path(args.rom).read_bytes()
    files = {"rom": rom}
    for item in args.image:
        name, _sep, path = item.partition("=")
        files[name] = Path(path).read_bytes()
    expected, rows = None, []
    for line in Path(args.manifest).read_text().splitlines():
        if line.startswith("# rom_sha1 "):
            expected = line.split()[2]
        elif line.strip() and not line.startswith("#"):
            rows.append(line.split("\\t"))
    if expected and hashlib.sha1(rom).hexdigest() != expected and not args.any_rom:
        sys.exit("this ROM's SHA-1 is not the one the project was generated from (" + expected + "); --any-rom to try")
    failed = missing = written = 0
    for image, function, source, offset, size, digest in rows:
        data = files.get("rom" if source == "rom" else image)
        if data is None:
            missing += 1
            print("skip " + image + "/" + function + ": give --image " + image + "=PATH (" + source + ")")
            continue
        start, length = int(offset, 16), int(size, 16)
        piece = data[start:start + length]
        if len(piece) != length or (digest != "-" and hashlib.sha256(piece).hexdigest() != digest):
            failed += 1
            print("FAIL " + image + "/" + function + ": the bytes at " + offset + " are not the recorded ones")
            continue
        target = Path(args.out) / image / (function + ".bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(piece)
        written += 1
    print(str(written) + " written, " + str(failed) + " failed, " + str(missing) + " skipped")
    sys.exit(1 if failed or missing else 0)


if __name__ == "__main__":
    main()
'''

MACRO_INC = """/* Assembler macros of the generated assembly (the export's sources carry their own copy). */
.macro glabel label
    .global \\label
    \\label:
.endm
.macro jlabel label
    \\label:
.endm
"""


def _raw_stub(unit: Unit) -> bytes:
    digest = unit.raw_sha256 or "unknown (generated without the ROM)"
    return (f"/* raw unit {unit.function}: {unit.image} offset 0x{unit.offset:X}, 0x{unit.size:X} bytes, at 0x{unit.vram:08X}.\n"
            f" * SHA-256 {digest}.\n"
            " * No byte of the ROM is stored here: `make extract` cuts these bytes from your own ROM into\n"
            f" * build/raw/{_safe(unit.image)}/{_safe(unit.function)}.bin and checks their digest.  Replace this file with\n"
            " * assembly or C that reproduces the bytes to promote the unit, then generate again. */\n"
            ".set noreorder\n.section .text\n"
            f".global {unit.function}\n{unit.function}:\n"
            f".incbin \"build/raw/{_safe(unit.image)}/{_safe(unit.function)}.bin\"\n").encode()


def _symbol_files(files: dict, image: str, images: list[str]) -> list[str]:
    """Symbol files a unit of ``image`` is verified with: the shared ones (no image suffix) and its own."""
    out = []
    for path in sorted(p for p in files if p.startswith("symbols/") and p.endswith(".txt")):
        owners = [name for name in images if Path(path).stem.endswith("_" + name)]
        if not owners or image in owners:
            out.append(path)
    return out


_DONOR_ROOT = re.compile(r"^(donors/[^/]+/src/[^/]+)/(.+)$")


def _verify_lines(units: list[Unit], files: dict, images: list[str]) -> list[str]:
    lines = ["#!/bin/sh",
             "# Generated by n64dk decompile-project: every unit with a source, rebuilt with its recipe and compared",
             "# strictly against YOUR ROM (through PROJECT, the decompile-init project made from it).  A donor unit",
             "# builds from its donor root (its include paths are relative to it).",
             "#   sh tools/verify.sh [exact|assumed|assembly|all]      results in build/verify/<unit>.json",
             "set -u", 'N64DK=${N64DK:-n64dk}', 'want=${1:-all}', "failed=0", 'ROOT=$(pwd)',
             'PROJECT=$(cd "${PROJECT:-../project}" && pwd) || { echo "set PROJECT to the decompile-init project of your ROM"; exit 2; }',
             'mkdir -p "$ROOT/build/verify"',
             "unit() {",
             '    status=$1; name=$2; dir=$3; shift 3',
             '    [ "$want" = all ] || [ "$want" = "$status" ] || return 0',
             '    (cd "$dir" && $N64DK --format json match --project "$PROJECT" "$@") > "$ROOT/build/verify/$name.json" 2>&1',
             '    if grep -q \'"exact": true\' "$ROOT/build/verify/$name.json"; then echo "ok   $name"; else echo "FAIL $name"; failed=1; fi',
             "}"]
    seen = set()
    for unit in units:
        if unit.status == "raw" or not unit.path:
            continue
        key = (unit.path, unit.recipe.get("unit_offset") or unit.vram)
        if key in seen:
            continue                                   # a donor unit covering several functions: once
        seen.add(key)
        vram, size = unit.vram, unit.code_size
        directory, source = ".", unit.path
        if unit.kind in ("donor-c", "donor-asm"):
            if unit.recipe.get("unit_offset"):
                vram = unit.vram - unit.offset + _hex(unit.recipe["unit_offset"])
                size = _hex(unit.recipe.get("unit_size", ""), unit.code_size)
            found = _DONOR_ROOT.match(unit.path)
            if found:
                directory, source = found.group(1), found.group(2)
        pipeline = unit.recipe.get("pipeline") or "gnu-as"
        args = ["--image-name", unit.image, "--addr", f"{vram:08X}", "--size", f"{size:X}", "--pipeline", pipeline]
        if unit.recipe.get("flags"):
            args.append("--flags=" + unit.recipe["flags"])
        words = [shlex.quote(a) for a in args]
        for symbols in _symbol_files(files, unit.image, images):
            words += ["--symbols", '"$ROOT"/' + shlex.quote(symbols)]
        name = _safe(f"{unit.image}__{unit.function}")
        lines.append(" ".join(["unit", unit.status, name, shlex.quote(directory), *words, shlex.quote(source)]))
    lines += ['exit $failed', ""]
    return lines


def _receipt_lines(units: list[Unit]) -> list[str]:
    lines = ["#!/bin/sh", "# Generated by n64dk decompile-project: replay every proof receipt kept in receipts/ (the export's",
             "# receipts whose source digest is this tree's source).  Each replay rebuilds and verifies strictly again.",
             "set -u", 'N64DK=${N64DK:-n64dk}', 'PROJECT=${PROJECT:-../project}', "failed=0"]
    for unit in units:
        if unit.receipt != "present":
            continue
        receipt = "receipts/" + receipt_name(unit.image, unit.function, unit.vram)
        lines.append(f'$N64DK verify-receipt {shlex.quote(receipt)} --source {shlex.quote(unit.path)} --project "$PROJECT" '
                     f'|| failed=1')
    lines += ['exit $failed', ""]
    return lines


def _makefile(images: dict[str, str]) -> str:
    image_args = " ".join(f"--image {name}=$(PROJECT)/survey/{source}" for name, source in sorted(images.items())
                          if source != "rom")
    return "\n".join([
        "# Generated by n64dk decompile-project.  Nothing of the ROM is in this tree: `make extract` reads",
        "# your own ROM; `make verify` rebuilds every unit and compares it strictly against your ROM.",
        "N64DK ?= n64dk", "PYTHON ?= python3",
        "# PROJECT: an `n64dk decompile-init` project made from your ROM (it holds the ROM; this tree does not)",
        "PROJECT ?= ../project",
        "ROM ?= $(PROJECT)/baserom.z64",
        f"IMAGES ?= {image_args}", "",
        ".PHONY: all extract verify verify-exact verify-assumed verify-assembly verify-receipts status clean",
        "all: extract verify", "",
        "extract:", "\t$(PYTHON) tools/extract_raw.py --rom $(ROM) $(IMAGES) --out build/raw", "",
        "verify:", "\tN64DK=$(N64DK) PROJECT=$(PROJECT) sh tools/verify.sh all", "",
        "verify-exact:", "\tN64DK=$(N64DK) PROJECT=$(PROJECT) sh tools/verify.sh exact", "",
        "verify-assumed:", "\tN64DK=$(N64DK) PROJECT=$(PROJECT) sh tools/verify.sh assumed", "",
        "verify-assembly:", "\tN64DK=$(N64DK) PROJECT=$(PROJECT) sh tools/verify.sh assembly", "",
        "verify-receipts:", "\tN64DK=$(N64DK) PROJECT=$(PROJECT) sh tools/verify_receipts.sh", "",
        "status:", "\t@cat STATUS.md", "",
        "clean:", "\trm -rf build", ""])


def _text_script(image: str, units: list[Unit], source: str) -> str:
    """Every unit's object at its own address (offsets inside one output section): text only."""
    placed = {}
    for unit in units:
        if unit.status == "raw" and not unit.path:
            continue
        obj = "build/" + str(Path(unit.path).with_suffix(".o"))
        vram = unit.vram
        if unit.kind in ("donor-c", "donor-asm") and unit.recipe.get("unit_offset"):
            vram = unit.vram - unit.offset + _hex(unit.recipe["unit_offset"])
        placed[obj] = min(placed.get(obj, vram), vram)
    if not placed:
        return ""
    base = min(placed.values())
    first = min(units, key=lambda u: u.vram)
    at = f" AT(0x{first.offset - (first.vram - base):X})" if source == "rom" else ""
    name = re.sub(r"[^A-Za-z0-9_]", "_", image)
    lines = [f"/* Generated by n64dk decompile-project: the .text of every unit of image {image} at its own address.",
             " * Text only: data, rodata and bss are not units yet, so this does not link a ROM (a starting point",
             " * for a pinned relink; see linker/link_layout.ld for the inferred object order, when present). */",
             "SECTIONS", "{", f"    .{name}_text 0x{base:08X} :{at}", "    {"]
    if source != "rom":
        lines.insert(3, f"/* image {image} is not stored as-is in the ROM ({source}): no load address is given */")
    for obj, vram in sorted(placed.items(), key=lambda item: (item[1], item[0])):
        lines.append(f"        . = 0x{vram - base:X};")
        lines.append(f"        {obj}(.text);")
    lines += ["    }", "}", ""]
    return "\n".join(lines)


def _units_header(image: str, units: list[Unit]) -> str:
    guard = "UNITS_" + re.sub(r"[^A-Za-z0-9_]", "_", image).upper() + "_H"
    lines = [f"/* Generated by n64dk decompile-project: address and size of every unit of image {image}. */",
             f"#ifndef {guard}", f"#define {guard}"]
    for unit in units:
        name = re.sub(r"[^A-Za-z0-9_]", "_", unit.function)
        lines.append(f"#define UNIT_{name}_VRAM 0x{unit.vram:08X}")
        lines.append(f"#define UNIT_{name}_SIZE 0x{unit.code_size:X}")
    lines += [f"#endif /* {guard} */", ""]
    return "\n".join(lines)


NEXT_STEP = {
    "assumed": "declare the addresses it leans on (listed below; manifests/symbols.assumed.txt) and verify again",
    "assembly": "decompile it: write C, verify it with `make verify-exact` (or `n64dk match`), then regenerate",
    "raw": "write assembly or C that reproduces its bytes; until then `make extract` provides them from your ROM",
}


def _dossier(unit: Unit, assumptions: list[str]) -> str:
    lines = [f"# {unit.function} ({unit.image}, 0x{unit.vram:08X}, 0x{unit.code_size:X} bytes)", "",
             f"- status: **{unit.status}** ({unit.kind})", f"- outcome in the run: {unit.outcome or '-'}"
             + (f", distance {unit.distance}" if unit.distance else ""),
             f"- reproduced now by: {unit.reproduced_by or 'nothing'}", f"- recipe or reason: {unit.how or '-'}",
             f"- file: `{unit.path}`", f"- link-layout object: {unit.object or '-'}"]
    lines += [f"- note: {note}" for note in unit.notes]
    if assumptions:
        lines += ["", "Addresses it leans on:", ""] + [f"- {item}" for item in assumptions]
    lines += ["", f"Next step: {NEXT_STEP.get(unit.status, '-')}", ""]
    return "\n".join(lines)


def _readme(counts: dict, total: int, rom_sha1: str | None, layout: bool) -> str:
    return "\n".join([
        "# Generated decompilation project", "",
        "Made by `n64dk decompile-project` from a `decompile-export` tree.  No byte of the ROM is in this tree.", "",
        f"- ROM SHA-1 this project describes: `{rom_sha1 or 'unknown'}`", f"- units: {total:,}",
        *[f"- {status}: {counts.get(status, 0):,}" for status in STATUSES],
        f"- inferred link order: {'linker/link_layout.ld' if layout else 'not given'}", "",
        "## Use", "", "```sh", "n64dk decompile-init YOUR_ROM.z64 --out ../project   # once: your ROM, your machine",
        "make extract    # raw units: bytes cut from your ROM, digests checked", "make verify     # every unit rebuilt and compared strictly",
        "make verify-receipts", "```", "",
        "## Moving a unit up (raw -> assembly -> C)", "",
        "Write the better source, verify it, re-export, and run `n64dk decompile-project EXPORT --out .` again:",
        "only that unit's files (and the shared indices) are rewritten; its old receipt is dropped unless it covers",
        "the new source.  Files you changed by hand are never overwritten (they are reported as conflicts).", "",
        "## Layout", "",
        "`src/` exact C; `src_assumed/` byte-identical C waiting for symbols; `donors/` verified library units;",
        "`asm/` hand-written assembly; `asm/generated/` disassembly that reassembles to the bytes; `rsp/` RSP-class",
        "units; `raw/` stubs and `raw/manifest.tsv`; `include/`; `symbols/`; `linker/`; `recipes.tsv`;",
        "`toolchain.lock.json`; `receipts/`; `provenance/`; `unresolved/`; `STATUS.tsv`; `GENERATED.json`.", "",
        f"Credit: {CREDIT}.", ""])


def build_plan(export: Path, *, link_layout: Path | None = None, images: dict | None = None,
               toolchain_lock: dict | None = None) -> tuple[dict, list[Unit], dict]:
    """Every file of the project (path -> bytes), the units, and the manifest (without file digests)."""
    export = Path(export)
    units = read_functions(export)
    recipes = donor_recipes(export)
    placements = queue_placements(export)
    objects = link_objects(Path(link_layout) / "objects.tsv" if link_layout else None)
    sources = _image_sources(export)
    assumptions = _assumptions(export)
    files: dict[str, bytes] = {}
    for unit in units:
        classify(unit, export)
        unit.recipe = parse_recipe(unit, recipes)
        if unit.kind == "c" and (unit.image, unit.vram) in placements:
            stated = shlex.split(unit.recipe.get("flags", ""))
            unit.recipe["flags"] = " ".join(stated + [f for f in placements[(unit.image, unit.vram)] if f not in stated])
        unit.object = _object_of(objects, unit.image, unit.vram)
        if unit.status == "raw" and images and unit.image in images:
            data, _delta = images[unit.image]
            piece = data[unit.offset:unit.offset + unit.size]
            unit.raw_sha256 = _sha(piece) if len(piece) == unit.size and unit.size else None
    # the export's own text trees: symbols, config, recipe manifests, donor units
    for folder, target in (("symbols", "symbols"), ("config", "config"), ("manifests", "manifests"), ("donors", "donors")):
        root = export / folder
        if not root.is_dir():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            content = _text(path)
            if content is not None:
                files[f"{target}/{path.relative_to(root).as_posix()}"] = content
    unit_files: dict[str, list[str]] = {}
    for unit in units:
        own = []
        if unit.content is not None and unit.kind not in ("donor-c", "donor-asm"):
            files[unit.path] = unit.content
            own.append(unit.path)
        elif unit.status == "raw":
            files[unit.path] = _raw_stub(unit)
            own.append(unit.path)
        source_digest = _sha(unit.content) if unit.content is not None else None
        receipt = export / "receipts" / receipt_name(unit.image, unit.function, unit.vram)
        if receipt.is_file() and source_digest:
            try:
                recorded = (json.loads(receipt.read_text()).get("source") or {}).get("sha256")
            except ValueError:
                recorded = None
            if recorded == source_digest:
                path = "receipts/" + receipt_name(unit.image, unit.function, unit.vram)
                files[path] = receipt.read_bytes()
                own.append(path)
                unit.receipt = "present"
            else:
                unit.receipt = "stale"
        inputs = {"row": [unit.image, unit.function, unit.vram, unit.offset, unit.size, unit.code_size, unit.outcome,
                          unit.how, unit.source, unit.distance, unit.reproduced_by, unit.unit_class],
                  "source_sha256": source_digest, "recipe": unit.recipe, "object": unit.object,
                  "raw_sha256": unit.raw_sha256, "receipt": unit.receipt}
        unit.inputs = _sha(_json(inputs))
        provenance = {"schema": SCHEMA, "credit": CREDIT, "unit": unit.id, "image": unit.image,
                      "function": unit.function, "vram": f"0x{unit.vram:08X}", "offset": f"0x{unit.offset:X}",
                      "size": f"0x{unit.size:X}", "code_size": f"0x{unit.code_size:X}", "status": unit.status,
                      "kind": unit.kind, "outcome": unit.outcome, "how": unit.how, "distance": unit.distance,
                      "reproduced_by": unit.reproduced_by, "unit_class": unit.unit_class,
                      "export_source": unit.source, "file": unit.path, "source_sha256": source_digest,
                      "recipe": unit.recipe, "link_layout_object": unit.object, "receipt": unit.receipt,
                      "raw": ({"offset": f"0x{unit.offset:X}", "size": f"0x{unit.size:X}", "sha256": unit.raw_sha256}
                              if unit.status == "raw" else None),
                      "inputs_sha256": unit.inputs}
        path = f"provenance/{_safe(unit.image)}/{_safe(unit.function)}.json"
        files[path] = _json(provenance)
        own.append(path)
        if unit.status != "exact" and unit.kind != "handwritten":
            path = f"unresolved/{_safe(unit.image)}/{_safe(unit.function)}.md"
            files[path] = _dossier(unit, assumptions.get((unit.image, unit.function), [])).encode()
            own.append(path)
        unit_files[unit.id] = sorted(own)
    image_names = sorted({u.image for u in units})
    counts = {status: sum(1 for u in units if u.status == status) for status in STATUSES}
    # shared indices
    status_rows = ["unit\timage\tfunction\tvram\tsize\tcode_size\tstatus\tkind\tunit_class\tobject\tfile\trecipe\treceipt"]
    recipe_rows = ["unit\timage\tvram\tcode_size\tpipeline\tflags\tfile\tverified_with"]
    raw_rows = ["# raw units: cut from your own ROM at build time by tools/extract_raw.py (no ROM byte is stored)"]
    rom_sha1 = _rom_sha1(export)
    if rom_sha1:
        raw_rows.append(f"# rom_sha1 {rom_sha1}")
    raw_rows.append("# image\tfunction\tsource\toffset\tsize\tsha256")
    for unit in units:
        recipe = f"{unit.recipe.get('pipeline', '')} {unit.recipe.get('flags', '')}".strip()
        status_rows.append("\t".join([unit.id, unit.image, unit.function, f"0x{unit.vram:08X}", f"0x{unit.size:X}",
                                      f"0x{unit.code_size:X}", unit.status, unit.kind, unit.unit_class or "-",
                                      unit.object or "-", unit.path, recipe or "-", unit.receipt]))
        if unit.status != "raw":
            recipe_rows.append("\t".join([unit.id, unit.image, f"0x{unit.vram:08X}", f"0x{unit.code_size:X}",
                                          unit.recipe.get("pipeline", ""), unit.recipe.get("flags", ""), unit.path,
                                          unit.recipe.get("assembler", unit.recipe.get("pipeline", ""))]))
        else:
            raw_rows.append("\t".join([_safe(unit.image), _safe(unit.function), sources.get(unit.image, "rom"),
                                       f"0x{unit.offset:X}", f"0x{unit.size:X}", unit.raw_sha256 or "-"]))
    files["STATUS.tsv"] = ("\n".join(status_rows) + "\n").encode()
    files["recipes.tsv"] = ("\n".join(recipe_rows) + "\n").encode()
    files["raw/manifest.tsv"] = ("\n".join(raw_rows) + "\n").encode()
    by_kind: dict = {}
    for unit in units:
        by_kind[(unit.image, unit.status, unit.kind)] = by_kind.get((unit.image, unit.status, unit.kind), 0) + 1
    status_md = ["# Status", "", f"Credit: {CREDIT}.", "", "| status | units |", "|---|---:|",
                 *[f"| {status} | {counts[status]:,} |" for status in STATUSES], f"| all | {len(units):,} |", "",
                 "| image | status | kind | units |", "|---|---|---|---:|",
                 *[f"| {i} | {s} | {k} | {n:,} |" for (i, s, k), n in sorted(by_kind.items())], "",
                 f"Receipts kept: {sum(1 for u in units if u.receipt == 'present'):,}; "
                 f"stale (source changed since): {sum(1 for u in units if u.receipt == 'stale'):,}.", ""]
    files["STATUS.md"] = "\n".join(status_md).encode()
    for image in image_names:
        members = [u for u in units if u.image == image]
        files[f"include/units_{_safe(image)}.h"] = _units_header(image, members).encode()
        script = _text_script(image, members, sources.get(image, "rom"))
        if script:
            files[f"linker/{_safe(image)}_text.ld"] = script.encode()
    files["include/macro.inc"] = MACRO_INC.encode()
    if link_layout:
        for name, target in (("link.ld", "linker/link_layout.ld"), ("objects.tsv", "linker/link_layout_objects.tsv")):
            content = _text(Path(link_layout) / name)
            if content is not None:
                files[target] = content
    files["toolchain.lock.json"] = _json(toolchain_lock if toolchain_lock is not None else
                                         {"note": "generated without a toolchain manifest"})
    files["Makefile"] = _makefile({image: sources.get(image, "rom") for image in image_names}).encode()
    files["tools/extract_raw.py"] = EXTRACT_SCRIPT.encode()
    files["tools/verify.sh"] = "\n".join(_verify_lines(units, files, image_names)).encode()
    files["tools/verify_receipts.sh"] = "\n".join(_receipt_lines(units)).encode()
    files["unresolved/README.md"] = ("# Unresolved units\n\nOne dossier per unit that is not settled C or hand-written "
                                     "assembly: what the run found, what reproduces it now, the next step.\n").encode()
    files["README.md"] = _readme(counts, len(units), rom_sha1, bool(link_layout)).encode()
    files[".gitignore"] = b"build/\n"
    layout_inputs = {}
    if link_layout:
        for name in ("link.ld", "objects.tsv"):
            path = Path(link_layout) / name
            layout_inputs[name] = _sha(path.read_bytes()) if path.is_file() else None
    manifest = {"schema": SCHEMA, "credit": CREDIT,
                "inputs": {"functions_tsv_sha256": _sha((export / "FUNCTIONS.tsv").read_bytes()),
                           "link_layout": layout_inputs or None,
                           "toolchain_lock_digest": (toolchain_lock or {}).get("digest"), "rom_sha1": rom_sha1},
                "counts": counts,
                "units": {u.id: {"status": u.status, "kind": u.kind, "inputs": u.inputs,
                                 "receipt": u.receipt, "files": unit_files[u.id]} for u in units}}
    return files, units, manifest


# ---------------------------------------------------------------------------------------------
# writing


def _prune(directory: Path, root: Path) -> None:
    while directory != root and root in directory.parents:
        try:
            directory.rmdir()
        except OSError:
            return
        directory = directory.parent


def generate(export, out, *, link_layout=None, images: dict | None = None, toolchain_lock: dict | None = None,
             force: bool = False, protected=(), roms=()) -> dict:
    """Generate (or regenerate in place) the project tree; returns what was written."""
    export = resolve_export(export)
    out = check_output(out, [export, *([link_layout] if link_layout else []), *protected], roms)
    files, units, manifest = build_plan(export, link_layout=link_layout, images=images, toolchain_lock=toolchain_lock)
    previous = {}
    if (out / MANIFEST).is_file():
        try:
            previous = json.loads((out / MANIFEST).read_text())
        except ValueError as error:
            raise GenerationError(f"{out / MANIFEST}: unreadable ({error}); refusing to regenerate over it") from error
    before = previous.get("files") or {}
    report = {"out": str(out), "files": len(files) + 1, "written": [], "unchanged": 0, "removed": [], "conflicts": []}
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(files):
        content, target = files[path], out / path
        if target.is_file():
            current = _sha(target.read_bytes())
            if current == _sha(content):
                report["unchanged"] += 1
                continue
            if current != before.get(path) and not force:
                report["conflicts"].append(path)       # changed by hand (or not ours): never overwritten
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        report["written"].append(path)
    for path in sorted(set(before) - set(files)):
        target = out / path
        if not target.is_file():
            continue
        if _sha(target.read_bytes()) == before[path] or force:
            target.unlink()
            report["removed"].append(path)
            _prune(target.parent, out)
        else:
            report["conflicts"].append(path)
    old_units = previous.get("units") or {}
    report["changed_units"] = sorted(uid for uid, item in manifest["units"].items()
                                     if (old_units.get(uid) or {}).get("inputs") != item["inputs"]) if previous else []
    report["invalidated_receipts"] = sorted(
        uid for uid in report["changed_units"]
        if (old_units.get(uid) or {}).get("receipt") == "present" and manifest["units"][uid]["receipt"] != "present")
    manifest["files"] = {path: _sha(content) for path, content in sorted(files.items())}
    (out / MANIFEST).write_bytes(_json(manifest))
    report["counts"] = manifest["counts"]
    report["units"] = len(units)
    report["written_count"] = len(report["written"])
    report["credit"] = CREDIT
    return report
