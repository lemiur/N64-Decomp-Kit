"""Pinned relink of one image of a generated project (``n64dk relink-image``).

``decompile-project`` writes a tree in which every unit is verified alone (``make verify``: each
unit rebuilt and compared at its own address).  Nothing there checks that the units *link
together* into the image.  This module is the smallest step from there towards level 3 of
WHOLE_ROM_REBUILD (pinned relink), for one image:

1. every unit with a source and a recipe (exact C, assembly; assumed C only with
   ``assumed="link"``) is built with the project's own recipe, through the same
   ``pipeline.build`` the per-unit verification uses;
2. each object's ``.text`` is pinned at the unit's catalogue address (a donor unit at its donor
   unit's address, owning its donor slot).  Its other sections go to a declared base
   (``--section-base`` in the recipe), else to the base its declared global symbols give, else to
   a scratch address outside the image where they own no byte of it;
3. every byte of the image that no compiled section owns (raw units, units that do not build,
   data, padding) is a raw blob cut from the user's ROM at build time into ``<build>/raw/raw.o``,
   the declared local extraction input and the only file that keeps ROM bytes (the linked image
   is kept too only on request, ``keep_image``).  Nothing is written into the project tree or the
   kit;
4. a generated GNU ld script (``link.ld``) places every piece at its address and
   ``mips-linux-gnu-ld`` links it.  ``linker="kit"`` relocates every object with
   :meth:`n64dk.elf.Elf.relocate` instead (the strict verifier's relocator): a cross-check, and
   what the unit tests use;
5. the linked image is compared byte for byte with the image in the user's ROM: identical, or the
   first difference and the piece that owns it.  Coverage says which bytes came from compiled
   source and which are raw, per class and per unit (``units.tsv``).

Symbols: an object's own definitions win; names it leaves undefined are bound to the addresses
the per-unit verification declared (the image's symbol files and the recipes' ``--symbol=``
pins), as ``PROVIDE``.  A name nothing declares is reported and bound to a sentinel, so every
byte that uses it differs visibly.  Nothing is read from the target to make the link work: an
``assumed`` unit (its verification leaned on addresses read from the target) is a raw blob
unless ``assumed="link"``, and then no level is claimed.

**Strictness.**  A measurement: it grants no credit and changes no run, receipt or evidence
store.  An identical image says only that this pinned link of these pieces reproduces it.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import struct
import subprocess
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..elf import GP_RELATIVE, GP_SYMBOL, Elf
from . import projectgen as pg

SCHEMA = "n64dk-relink/1"
MARKER = "RELINK.json"
CREDIT = ("a measurement: the image linked from the project's units and raw blobs cut from your ROM, compared "
          "byte for byte; it grants no credit and changes no run, receipt or evidence store")
SHF_ALLOC = 0x2
# sections placed by name whatever their flags: KMC GCC 2.7.2's assembler can emit `.rodata` as writable but
# not allocatable (flags 0x1); the per-unit verifier compares by name, so the link must place it too
DATA_SECTIONS = (".rodata", ".rdata", ".data", ".sdata", ".bss", ".sbss")
SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, SHT_NOBITS, SHT_REL = 1, 2, 3, 8, 9
SHN_UNDEF, SHN_ABS, SHN_COMMON = 0, 0xFFF1, 0xFFF2
# allocated sections that carry no image bytes of their own (assembler/ABI records): never placed
SKIPPED = (".reginfo", ".MIPS.abiflags", ".options", ".gptab", ".note", ".pdr", ".mdebug", ".comment", ".debug",
           ".gnu.attributes")
DISCARD = " ".join(f"*({name}*)" for name in SKIPPED)
SOURCE = ("exact", "assumed", "assembly")
RAW = ("raw-unit", "raw-other")
CLASSES = SOURCE + RAW
DONOR_KINDS = ("donor-c", "donor-asm")
RANK = {"exact": 0, "assembly": 1, "assumed": 2, "raw": 3}
# inside the image's 256 MB region (so jumps stay encodable), far above any N64 image
SCRATCH_OFFSET, COMMON_OFFSET, SENTINEL_OFFSET = 0x0F000000, 0x0F800000, 0x0FFF0000
OUTPUTS = ("obj", "raw", "work", "link.ld", "link.map", "link.log", "relinked.elf", "relinked.bin", "units.tsv")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_BLOCKING = re.compile(r"error|undefined reference|truncated|multiple definition|overlaps|cannot", re.IGNORECASE)


class RelinkError(ValueError):
    """relink-image refuses: a reason, and nothing linked."""


def _hx(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:08X}"


# ---------------------------------------------------------------------------------------------
# ELF: a minimal big-endian MIPS ELF32 writer (raw blobs; the tests' objects) and the two header
# fields n64dk.elf.Elf does not keep


def write_elf(sections, symbols=(), relocations=None, executable: bool = False) -> bytes:
    """A big-endian MIPS ELF32.  ``sections``: ``(name, bytes | size of a NOBITS section, flags, align,
    address)``.  ``symbols``: ``(name, value, size, section name | None (undefined) | "COMMON" |
    "ABS", bind, type)``, written after the null symbol and one local section symbol per section
    (locals first).  ``relocations``: ``{section: [(offset, type, symbol or section name)]}`` as REL."""
    relocations = relocations or {}
    shstr, strtab = bytearray(b"\0"), bytearray(b"\0")

    def intern(table: bytearray, text: str) -> int:
        offset = len(table)
        table.extend(text.encode() + b"\0")
        return offset

    index = {item[0]: number + 1 for number, item in enumerate(sections)}
    ordered = [s for s in symbols if s[4] == 0] + [s for s in symbols if s[4] != 0]
    entries = [bytes(16)] + [struct.pack(">IIIBBH", 0, 0, 0, 0x03, 0, number + 1) for number in range(len(sections))]
    symbol_index = {}
    for name, value, size, section, bind, kind in ordered:
        shndx = {None: SHN_UNDEF, "COMMON": SHN_COMMON, "ABS": SHN_ABS}.get(section)
        if shndx is None:
            shndx = index[section]
        symbol_index[name] = len(entries)
        entries.append(struct.pack(">IIIBBH", intern(strtab, name), value, size, (bind << 4) | kind, 0, shndx))
    first_global = 1 + len(sections) + sum(1 for s in ordered if s[4] == 0)
    headers = []                                  # [name, type, flags, addr, body, size, link, info, align, entsize]
    for name, body, flags, align, address in sections:
        nobits = isinstance(body, int)
        headers.append([name, SHT_NOBITS if nobits else SHT_PROGBITS, flags, address, b"" if nobits else bytes(body),
                        body if nobits else len(body), 0, 0, align, 0])
    symtab_at = 1 + len(sections) + len(relocations)
    for target, items in relocations.items():
        body = b"".join(struct.pack(">II", offset, (symbol_index.get(symbol, index.get(symbol, 0)) << 8) | rtype)
                        for offset, rtype, symbol in items)
        headers.append([".rel" + target, SHT_REL, 0, 0, body, len(body), symtab_at, index[target], 4, 8])
    symtab = b"".join(entries)
    headers.append([".symtab", SHT_SYMTAB, 0, 0, symtab, len(symtab), symtab_at + 1, first_global, 4, 16])
    headers.append([".strtab", SHT_STRTAB, 0, 0, bytes(strtab), len(strtab), 0, 0, 1, 0])
    headers.append([".shstrtab", SHT_STRTAB, 0, 0, b"", 0, 0, 0, 1, 0])
    names = [intern(shstr, header[0]) for header in headers]
    headers[-1][4], headers[-1][5] = bytes(shstr), len(shstr)
    out = bytearray(52)
    offsets = []
    for header in headers:
        out.extend(bytes(-len(out) % 4))
        offsets.append(len(out))
        out.extend(header[4])
    out.extend(bytes(-len(out) % 4))
    shoff = len(out)
    out.extend(bytes(40))
    for header, name, offset in zip(headers, names, offsets):
        out.extend(struct.pack(">10I", name, header[1], header[2], header[3], offset, header[5], header[6], header[7],
                               header[8], header[9]))
    out[0:52] = (b"\x7fELF\x01\x02\x01" + bytes(9)
                 + struct.pack(">HHIIIIIHHHHHH", 2 if executable else 1, 8, 1, 0, 0, shoff, 0, 52, 0, 0, 40,
                               len(headers) + 1, len(headers)))
    return bytes(out)


def section_table(data: bytes) -> list[tuple[int, int]]:
    """``(flags, address)`` per section index (``n64dk.elf.Elf`` keeps neither)."""
    shoff, = struct.unpack_from(">I", data, 0x20)
    shentsize, shnum = struct.unpack_from(">HH", data, 0x2E)
    return [struct.unpack_from(">II", data, shoff + number * shentsize + 8) for number in range(shnum)]


def _skipped(name: str) -> bool:
    return any(name == item or name.startswith(item) for item in SKIPPED)


# ---------------------------------------------------------------------------------------------
# the project's units


@dataclass
class Member:
    """One unit (a FUNCTIONS.tsv row) as the project's provenance record states it."""
    id: str
    function: str
    vram: int
    offset: int
    size: int
    code_size: int
    status: str
    kind: str
    file: str
    recipe: dict


@dataclass
class Target:
    """One object to link: a unit, or a donor unit covering several rows, or a raw stand-in."""
    key: str
    members: list
    status: str
    kind: str
    file: str
    pipeline: str
    flags: str
    vram: int                       # where its .text goes
    slot: int                       # bytes of the image its .text may own
    reason: str = ""                # why it is a raw stand-in ("" when it is linked)
    reason_code: str = ""
    obj: str = ""                   # its object, relative to the build directory
    pins: dict = field(default_factory=dict)            # --symbol= of the recipe (declared)
    bases: dict = field(default_factory=dict)           # --section-base= of the recipe (declared)
    assumed_pins: dict = field(default_factory=dict)    # manifests/assumptions.tsv (read from the target)
    assumed_bases: dict = field(default_factory=dict)


def read_members(project: Path, image: str) -> list[Member]:
    folder = Path(project) / "provenance" / pg._safe(image)
    if not folder.is_dir():
        raise RelinkError(f"{project}: no provenance/{pg._safe(image)}/ (no such image in this project)")
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except ValueError as error:
            raise RelinkError(f"{path}: unreadable ({error})") from error
        if row.get("image") != image:
            continue
        size = pg._hex(row.get("size", ""))
        out.append(Member(row.get("unit", ""), row.get("function", ""), pg._hex(row.get("vram", "")),
                          pg._hex(row.get("offset", "")), size, pg._hex(row.get("code_size", ""), size),
                          row.get("status", "raw"), row.get("kind", "none"), row.get("file", ""),
                          dict(row.get("recipe") or {})))
    if not out:
        raise RelinkError(f"{folder}: no unit of image {image}")
    return sorted(out, key=lambda m: (m.vram, m.function))


def read_assumptions(project: Path, image: str) -> dict[str, list]:
    """function -> [(kind, name, address)] from ``manifests/assumptions.tsv`` (addresses read from the target)."""
    out: dict[str, list] = {}
    path = Path(project) / "manifests" / "assumptions.tsv"
    if path.is_file():
        for line in path.read_text().splitlines():
            cells = line.split("\t")
            if line.startswith("#") or len(cells) < 5 or cells[0] != image:
                continue
            out.setdefault(cells[1], []).append((cells[2], cells[3], pg._hex(cells[4])))
    return out


def declared_symbols(project: Path, image: str) -> dict[str, int]:
    """The image's symbol files (the shared ones and its own, as ``tools/verify.sh`` passes them),
    explicit entries only: no address is inferred from a name."""
    from .symbols import load
    project = Path(project)
    folder = project / "symbols"
    files = {f"symbols/{path.name}": b"" for path in sorted(folder.glob("*.txt"))} if folder.is_dir() else {}
    images = sorted(path.name for path in (project / "provenance").iterdir() if path.is_dir())
    chosen = pg._symbol_files(files, pg._safe(image), images)
    return dict(load([project / path for path in chosen]).explicit())


def recorded_rom_sha1(project: Path) -> str | None:
    path = Path(project) / "raw" / "manifest.tsv"
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("# rom_sha1 "):
                return line.split()[2]
    return None


def image_extent(project: Path, image: str, size: int, delta: int, is_rom: bool) -> tuple[int, int]:
    """VRAM start and end of the image's loaded bytes: the image's ``config/layout.json`` entry when
    there is one (``main``: the boot segment's text, data and rodata), else the whole image file."""
    path = Path(project) / "config" / "layout.json"
    try:
        layout = json.loads(path.read_text()) if path.is_file() else {}
    except ValueError:
        layout = {}
    entry = layout.get(image) or {}
    if entry.get("vram") and entry.get("end"):
        # any image the layout places (main's boot segment, an overlay): overlays that share a VRAM
        # range are told apart by their ROM offset, which must agree with the image registry's delta
        start, end = pg._hex(entry["vram"]), pg._hex(entry["end"])
        if is_rom and entry.get("rom") and pg._hex(entry["rom"]) != start - delta:
            raise RelinkError(f"config/layout.json places {image} at ROM {entry['rom']}, the image registry at "
                              f"0x{start - delta:X}: give --extent")
        return start, end
    if is_rom:
        raise RelinkError(f"no extent known for image {image} inside the ROM: give --extent VRAM_START:VRAM_END")
    return delta, delta + size


def targets(members: list[Member], assumed: str = "raw", assumptions: dict | None = None) -> list[Target]:
    """The objects to link, in address order.  A donor unit is one object for all its rows."""
    groups: dict[str, Target] = {}
    for m in members:
        if m.kind in DONOR_KINDS and m.recipe.get("unit_offset"):
            vram = m.vram - m.offset + pg._hex(m.recipe["unit_offset"])
            slot = pg._hex(m.recipe.get("unit_size", ""), m.size)
            key = f"{m.file}@{vram:08X}"
        else:
            vram, slot, key = m.vram, m.size, m.id
        found = groups.get(key)
        if found is None:
            # generated and hand-written assembly: the lane the export proved it with; donor assembly
            # (a source tree's .s) keeps the period assembler tools/verify.sh uses for it
            pipeline = m.recipe.get("pipeline", "") or ("gnu-as" if m.kind in ("generated", "handwritten")
                                                        else "as" if m.kind in pg.ASSEMBLY_KINDS else "")
            if pipeline == "as" and m.recipe.get("assembler") == pg.ASSEMBLY_RECIPE:
                pipeline = "gnu-as"         # projects generated before the lane had its own name
            groups[key] = Target(key, [m], m.status, m.kind, m.file, pipeline, m.recipe.get("flags", ""), vram, slot)
        else:
            found.members.append(m)
            if RANK.get(m.status, 3) > RANK.get(found.status, 3):
                found.status = m.status
    out = sorted(groups.values(), key=lambda t: (t.vram, t.key))
    for t in out:
        if t.status not in SOURCE:
            t.reason, t.reason_code = "raw in the project: nothing reproduces it yet", "raw"
        elif t.status == "assumed" and assumed != "link":
            t.reason, t.reason_code = ("assumed: its verification read addresses from the target "
                                       "(--assumed link links it, and no level is claimed)"), "assumed"
        elif not t.pipeline:
            t.reason, t.reason_code = "no recipe in the project", "no-recipe"
        elif not t.file:
            t.reason, t.reason_code = "no source file", "no-source"
        if t.status == "assumed" and assumed == "link" and assumptions:
            for m in t.members:
                for kind, name, address in assumptions.get(m.function, []):
                    (t.assumed_bases if kind == "section" else t.assumed_pins)[name] = address
    return out


def _rooted(flags: list[str], root: Path) -> list[str]:
    """A donor unit's relative include paths are relative to its donor root."""
    def under(value: str) -> str:
        return value if Path(value).is_absolute() else str(root / value)
    out, index = [], 0
    while index < len(flags):
        flag = flags[index]
        if flag == "-I" and index + 1 < len(flags):
            out += ["-I", under(flags[index + 1])]
            index += 2
            continue
        out.append("-I" + under(flag[2:]) if flag.startswith("-I") and len(flag) > 2 else flag)
        index += 1
    return out


def build_targets(units: list[Target], project: Path, build: Path, cfg, build_fn=None) -> dict[int, bytes]:
    """Build every linkable target with its recipe; one that does not build becomes a raw stand-in."""
    from .pipeline import split_flags
    if build_fn is None:
        from .pipeline import build as build_fn
    objects: dict[int, bytes] = {}
    (build / "obj").mkdir(parents=True, exist_ok=True)
    for number, t in enumerate(units):
        if t.reason:
            continue
        source = (Path(project) / t.file).resolve()
        if not source.is_file():
            t.reason, t.reason_code = f"source file {t.file} missing", "no-source"
            continue
        flags = shlex.split(t.flags)
        found = pg._DONOR_ROOT.match(t.file)
        if t.kind in DONOR_KINDS and found:
            flags = _rooted(flags, (Path(project) / found.group(1)).resolve())
        try:
            spec = split_flags(flags, t.pipeline)
            data = Path(build_fn(source, spec, cfg, build / "work" / f"{number + 1:04d}")).read_bytes()
            Elf(data)
        except Exception as error:              # noqa: BLE001 - a unit that does not build is a reported raw stand-in
            t.reason, t.reason_code = f"does not build: {str(error)[-300:]}", "build-failed"
            continue
        t.pins.update(spec.symbols)
        t.bases.update(spec.section_bases)
        t.obj = f"obj/{number + 1:04d}_{re.sub(r'[^A-Za-z0-9_]', '_', t.members[0].function)}.o"
        (build / t.obj).write_bytes(data)
        objects[number] = data
    return objects


# ---------------------------------------------------------------------------------------------
# the plan: every piece at its address, every symbol bound


@dataclass
class Piece:
    out: str                        # its output section in the link
    target: int                     # index of its Target (-1: a raw blob)
    obj: str
    section: str
    index: int                      # input section index (-1: a raw blob)
    vram: int
    size: int
    own: int                        # bytes of the image it owns
    cls: str                        # coverage class, or "unplaced"
    placement: str                  # slot | declared base | declared symbols | assumed base | scratch | raw
    nobits: bool = False
    referenced: bool = False        # unplaced, and placed code refers to a symbol defined in it


@dataclass
class Plan:
    start: int
    end: int
    scratch: int
    pieces: list = field(default_factory=list)
    hard: dict = field(default_factory=dict)          # name = address: _gp, COMMON symbols with a declared address
    provide: dict = field(default_factory=dict)       # PROVIDE(name = address)
    unresolved: list = field(default_factory=list)
    stated: dict = field(default_factory=dict)        # names bound to the address their name states
    resolved: dict = field(default_factory=dict)      # every name the kit linker binds
    duplicates: list = field(default_factory=list)
    disagreements: list = field(default_factory=list)
    pin_conflicts: list = field(default_factory=list)
    commons: list = field(default_factory=list)       # COMMON symbols no declared address places
    derived: list = field(default_factory=list)
    inconsistent: list = field(default_factory=list)
    unscriptable: list = field(default_factory=list)
    assumed_used: bool = False
    uses_gp: bool = False


def _base(bases: dict, name: str):
    if name in bases:
        return bases[name]
    alias = {".rodata": ".rdata", ".rdata": ".rodata"}.get(name)
    return bases.get(alias) if alias else None


def _symbol_base(elf: Elf, index: int, known: dict) -> tuple[int | None, bool]:
    """The section base its declared global symbols agree on (None, and whether they disagreed)."""
    bases = {(known[s.name] - s.value) & 0xFFFFFFFF for s in elf.symbols
             if s.name and s.bind in (1, 2) and s.section == index and s.name in known}
    if len(bases) == 1:
        return bases.pop(), True
    return None, not bases


def raw_unit_pins(units: list[Target], objects: dict[int, bytes], project: Path, image: str) -> dict[str, int]:
    """Addresses that units left as raw stand-ins declare in their recipe, kept only when the donor stage
    recorded an origin independent of the unit's own bytes (``donors/<image>/pin_origins.json``): a unit
    that does not link still states facts about the image that other units' code refers to.  An address
    with no recorded origin, or a self-derived one, is not used: retail bytes never establish an address
    for the comparison they are part of."""
    from .pipeline import split_flags
    from .proofreceipt import INDEPENDENT_ORIGINS
    try:
        origins = json.loads((Path(project) / "donors" / image / "pin_origins.json").read_text())
    except (OSError, ValueError):
        return {}
    out: dict[str, int] = {}
    for number, t in enumerate(units):
        if number in objects or not t.flags:
            continue
        recorded = origins.get(f"0x{t.vram:08X}") or {}
        try:
            spec = split_flags(shlex.split(t.flags), t.pipeline or "as")
        except Exception:                                # noqa: BLE001 - an unreadable recipe states nothing
            continue
        for name, address in spec.symbols.items():
            if str(recorded.get(name, "")).split(":", 1)[0] in INDEPENDENT_ORIGINS:
                out.setdefault(name, address)
    return out


def plan(units: list[Target], objects: dict[int, bytes], extent: tuple[int, int], declared: dict) -> Plan:
    start, end = extent
    region = start & 0xF0000000
    out = Plan(start, end, region | SCRATCH_OFFSET)
    if end > out.scratch:
        raise RelinkError(f"the image reaches the scratch area {_hx(out.scratch)}")
    pins: dict[str, int] = {}
    for number, t in enumerate(units):
        if number not in objects:
            continue
        out.assumed_used = out.assumed_used or t.status == "assumed"
        for name, address in sorted({**t.pins, **t.assumed_pins}.items()):
            if name in pins and pins[name] != address:
                out.pin_conflicts.append(name)
            else:
                pins[name] = address
    known = {**declared, **pins}                      # the recipe's pins win, as in the per-unit verification
    defined: dict[str, tuple[int, int]] = {}          # name -> (address, bind)
    needed: set[str] = set()
    commons: dict[str, tuple[int, int]] = {}
    cursor = out.scratch
    for number, t in enumerate(units):
        data = objects.get(number)
        if data is None:
            continue
        elf, table = Elf(data), section_table(data)
        placed: dict[int, tuple[int, str]] = {}
        for sec in elf.sections:
            if (not table[sec.index][0] & SHF_ALLOC and sec.name not in DATA_SECTIONS) or not sec.size \
                    or _skipped(sec.name):
                continue
            base, how = None, ""
            if sec.name == ".text":
                base, how = t.vram, "slot"
            else:
                for bases, label in ((t.bases, "declared base"), (t.assumed_bases, "assumed base")):
                    base = _base(bases, sec.name)
                    if base is not None:
                        how = label
                        break
                if base is None:
                    base, consistent = _symbol_base(elf, sec.index, known)
                    if base is not None:
                        how = "declared symbols"
                        out.derived.append(f"{t.obj}({sec.name}) at {_hx(base)}")
                    elif not consistent:
                        out.inconsistent.append(f"{t.obj}({sec.name})")
            if base is None:
                align = max(1, sec.align)
                cursor = -(-cursor // align) * align
                base, how = cursor, "scratch"
                cursor += sec.size
            placed[sec.index] = (base, how)
        local_refs: set[int] = set()
        for index, (_base_address, how) in placed.items():
            if how == "scratch":
                continue
            for _offset, rtype, sidx, _addend in elf.relocations(elf.sections[index]):
                sym = elf.symbols[sidx]
                out.uses_gp = out.uses_gp or rtype in GP_RELATIVE
                if 0 < sym.section < len(elf.sections):
                    local_refs.add(sym.section)
        for index, (base, how) in sorted(placed.items()):
            sec = elf.sections[index]
            nobits = sec.type == SHT_NOBITS
            if how == "scratch":
                own, cls = 0, "unplaced"
            else:
                own = 0 if nobits else (min(sec.size, t.slot) if sec.name == ".text" else sec.size)
                cls = t.status
            out.pieces.append(Piece(f".p{len(out.pieces) + 1:05d}", number, t.obj, sec.name, index, base, sec.size,
                                    own, cls, how, nobits, how == "scratch" and index in local_refs))
        for sym in elf.symbols:
            if not sym.name or sym.bind not in (1, 2):
                continue
            if sym.section == SHN_UNDEF:
                needed.add(sym.name)
                continue
            if sym.section == SHN_COMMON:
                commons.setdefault(sym.name, (sym.size, max(1, sym.value)))
                continue
            if sym.section == SHN_ABS:
                address = sym.value
            elif sym.section in placed:
                address = placed[sym.section][0] + sym.value
            else:
                continue
            previous = defined.get(sym.name)
            if previous is not None:
                if previous[1] == 1 and sym.bind == 1:
                    out.duplicates.append(sym.name)
                if not (previous[1] == 2 and sym.bind == 1):
                    continue
            defined[sym.name] = (address, sym.bind)
    common_alloc: dict[str, int] = {}
    cursor = region | COMMON_OFFSET
    for name, (size, align) in sorted(commons.items()):
        if name in defined:
            continue
        if name in known:
            out.hard[name] = known[name]
            continue
        cursor = -(-cursor // align) * align
        common_alloc[name] = cursor
        cursor += max(1, size)
        out.commons.append(name)
    for name, (address, _bind) in sorted(defined.items()):
        if name in known and known[name] != address:
            out.disagreements.append((name, address, known[name]))
    if GP_SYMBOL in known:
        out.hard[GP_SYMBOL] = known[GP_SYMBOL]
    free = sorted(name for name in needed if name not in defined and name not in commons)
    out.provide = {name: known[name] for name in free if name in known and name not in out.hard}
    # a name that states its own address (D_8000E1B0, func_80001234) is bound to it, as the generated
    # project's own build and receipt replay (proofreceipt.stated_pins) do; reported as its own class
    from .handasm import _STATED
    for name in free:
        found = _STATED.match(name) if name not in known else None
        if found:
            out.stated[name] = int(found.group(1), 16)
    out.provide.update(out.stated)
    out.unresolved = [name for name in free if name not in known and name not in out.stated]
    out.resolved = {name: region | SENTINEL_OFFSET for name in out.unresolved}
    out.resolved.update(out.provide)
    out.resolved.update({name: item[0] for name, item in defined.items()})
    out.resolved.update(common_alloc)
    out.resolved.update(out.hard)
    out.unscriptable = sorted(name for name in {*out.hard, *out.provide, *out.unresolved} if not _NAME.fullmatch(name))
    # raw blobs: every byte of the extent no compiled section owns
    length = end - start
    covered, raw_unit = bytearray(length), bytearray(length)
    for p in out.pieces:
        low, high = max(p.vram, start), min(p.vram + p.own, end)
        if p.own and p.cls != "unplaced" and low < high:
            covered[low - start:high - start] = b"\x01" * (high - low)
    for number, t in enumerate(units):
        low, high = max(t.vram, start), min(t.vram + t.slot, end)
        if number not in objects and low < high:
            raw_unit[low - start:high - start] = b"\x01" * (high - low)
    position = 0
    while position < length:
        if covered[position]:
            position += 1
            continue
        kind, stop = raw_unit[position], position + 1
        while stop < length and not covered[stop] and raw_unit[stop] == kind:
            stop += 1
        vram = start + position
        out.pieces.append(Piece(f".p{len(out.pieces) + 1:05d}", -1, "raw/raw.o", f".raw_{vram:08X}", -1, vram,
                                stop - position, stop - position, "raw-unit" if kind else "raw-other", "raw"))
        position = stop
    return out


def linker_script(image: str, layout: Plan) -> str:
    """GNU ld: one output section per piece at its own address, in address order (one load segment for
    the image), ``SUBALIGN(1)`` so an input section's alignment never moves it; names bound as
    :func:`plan` bound them."""
    region = layout.start & 0xF0000000
    skip = set(layout.unscriptable)
    lines = [f"/* Generated by n64dk relink-image: image {image}, every piece pinned at its own address.",
             " * Compiled sections of the project's units; raw blobs cut from your ROM at build time (raw/raw.o,",
             " * never stored in the project); at scratch addresses (NOLOAD) the sections no declared address",
             " * places: they own no byte of the image.  A measurement: nothing here grants credit. */",
             "OUTPUT_ARCH(mips)", "SECTIONS", "{"]
    lines += [f"    {name} = {_hx(value)};" for name, value in sorted(layout.hard.items()) if name not in skip]
    lines += [f"    PROVIDE({name} = {_hx(value)});" for name, value in sorted(layout.provide.items())
              if name not in skip]
    lines += [f"    PROVIDE({name} = {_hx(region | SENTINEL_OFFSET)});   /* unresolved: nothing declares it */"
              for name in layout.unresolved if name not in skip]
    for p in sorted(layout.pieces, key=lambda item: (item.cls == "unplaced", item.vram, item.out)):
        noload = " (NOLOAD)" if p.nobits or p.cls == "unplaced" else ""
        lines.append(f"    {p.out} {_hx(p.vram)}{noload} : SUBALIGN(1) {{ {p.obj}({p.section}) }}")
    lines += [f"    .n64dk_common {_hx(region | COMMON_OFFSET)} (NOLOAD) : {{ *(COMMON) *(.scommon) }}",
              f"    /DISCARD/ : {{ {DISCARD} }}", "}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# linking


def kit_link(objects: dict[int, bytes], layout: Plan, raw: dict[str, bytes]) -> tuple[dict[str, bytes], list[str]]:
    """Every placed section relocated at its address with ``Elf.relocate`` (no external tool)."""
    linked, problems = dict(raw), []
    groups: dict[int, list[Piece]] = {}
    for p in layout.pieces:
        if p.target >= 0:
            groups.setdefault(p.target, []).append(p)
    for number, items in sorted(groups.items()):
        elf = Elf(objects[number])
        bases = {p.section: p.vram for p in items}
        for p in items:
            if p.nobits or p.cls == "unplaced":
                continue
            try:
                linked[p.out] = elf.relocate(p.section, bases, layout.resolved)
            except (KeyError, ValueError) as error:
                problems.append(f"{p.obj}({p.section}): {error}")
                linked[p.out] = elf.sections[p.index].data
    return linked, problems


def locals_after_globals(data: bytes) -> bool:
    """Whether an object's symbol table lists a local symbol at or after ``sh_info`` (the index of the first
    non-local one).  The IDO assembler writes such tables for some hand-written sources; GNU ld refuses the
    whole object (``.symtab local symbol at index N (>= sh_info of M)``)."""
    try:
        shoff, = struct.unpack_from(">I", data, 0x20)
        shentsize, shnum = struct.unpack_from(">HH", data, 0x2E)
        for number in range(shnum):
            kind, = struct.unpack_from(">I", data, shoff + number * shentsize + 4)
            if kind != 2:                                    # SHT_SYMTAB
                continue
            offset, size, _link, first_global = struct.unpack_from(">IIII", data, shoff + number * shentsize + 16)
            return any(data[offset + 16 * k + 12] >> 4 == 0 for k in range(first_global, size // 16))
    except (struct.error, IndexError):
        pass
    return False


def normalise_symbols(cfg, build: Path, objects: list[str]) -> list[str]:
    """Rewrite, with objcopy, the objects whose symbol table GNU ld would refuse (``locals_after_globals``):
    the same sections, relocations and symbols, the table in the order ELF asks for.  Returns their names."""
    from .. import safety
    from .sourcedonor import tool
    rewritten = []
    for name in objects:
        path = build / name
        if not path.is_file() or not locals_after_globals(path.read_bytes()):
            continue
        objcopy, env = tool(cfg, "objcopy")
        fixed = path.with_suffix(".ordered.o")
        done = safety.run_tool([objcopy, str(path), str(fixed)], kind="compile", timeout=120, cwd=str(build),
                               env={**os.environ, **env}, capture_output=True, text=True)
        if done.returncode == 0 and fixed.is_file() and not locals_after_globals(fixed.read_bytes()):
            fixed.replace(path)
            rewritten.append(name)
        else:
            fixed.unlink(missing_ok=True)
    return rewritten


def gnu_link(cfg, build: Path, objects: list[str], timeout: float = 1800.0) -> tuple[Path | None, str, int]:
    """``mips-linux-gnu-ld`` (the binutils the kit already uses) on ``link.ld`` in ``build``."""
    from .. import safety
    from .sourcedonor import tool
    normalise_symbols(cfg, build, objects)
    linker, env = tool(cfg, "ld")
    argv = [linker, "-EB", "-T", "link.ld", "-Map", "link.map", "-o", "relinked.elf", "--no-check-sections",
            "--noinhibit-exec", "--no-warn-mismatch", *objects]
    done = safety.run_tool(argv, kind="compile", timeout=timeout, cwd=str(build), env={**os.environ, **env},
                           capture_output=True, text=True)
    log = (done.stdout or "") + (done.stderr or "")
    (build / "link.log").write_text(log)
    path = build / "relinked.elf"
    return (path if path.is_file() else None), log, done.returncode


def collect_linked(data: bytes, layout: Plan) -> tuple[dict[str, bytes], list[str]]:
    """Each piece's bytes from a linked ELF, checking the linker put it where the plan did."""
    elf, table = Elf(data), section_table(data)
    wanted = {p.out: p for p in layout.pieces}
    linked, problems, misplaced = {}, [], set()
    for sec in elf.sections:
        p = wanted.get(sec.name)
        if p is None:
            continue
        address = table[sec.index][1]
        if address != p.vram:
            problems.append(f"{p.out} {p.obj}({p.section}) linked at {_hx(address)}, planned {_hx(p.vram)}")
            misplaced.add(p.out)
            continue
        linked[p.out] = sec.data
    problems += [f"{p.out} {p.obj}({p.section}): not in the linked output" for p in layout.pieces
                 if p.own and not p.nobits and p.cls != "unplaced" and p.out not in linked and p.out not in misplaced]
    return linked, problems


# ---------------------------------------------------------------------------------------------
# comparison


@dataclass
class Comparison:
    image: bytes
    owner: array
    written: list
    differing: list
    total: int
    first: int | None
    holes: int
    overlaps: list
    spills: list
    outside: list
    yielded: int = 0                # trailing zero padding of a data section given to a declared neighbour


def compare_image(layout: Plan, linked: dict[str, bytes], target: bytes) -> Comparison:
    start, end = layout.start, layout.end
    length = end - start
    image = bytearray(length)
    owner = array("i", [-1]) * length
    written, differing = [0] * len(layout.pieces), [0] * len(layout.pieces)
    overlaps, spills, outside, yielded = [], [], [], 0
    for number, p in enumerate(layout.pieces):
        body = linked.get(p.out)
        if not p.own or p.cls == "unplaced" or body is None:
            continue
        if p.section == ".text" and any(body[p.own:]):
            spills.append(number)                    # non-zero bytes past the unit's slot
        own = min(p.own, len(body))
        low, high = max(p.vram, start), min(p.vram + own, end)
        if low >= high:
            outside.append(number)
            continue
        clash = 0
        for address in range(low, high):
            position = address - start
            previous = owner[position]
            if previous < 0:
                owner[position] = number
                image[position] = body[address - p.vram]
                written[number] += 1
            elif _padding_yields(layout.pieces[previous], p, linked, address):
                # an assembler rounds a data section's size up (KMC as: .rodata to 16 bytes); the zero tail
                # runs into the next unit's declared section, whose own bytes are the image's
                owner[position] = number
                image[position] = body[address - p.vram]
                written[number] += 1
                written[previous] -= 1
                yielded += 1
            else:
                clash += 1
        if clash:
            overlaps.append((number, clash))
    total, first = 0, None
    if bytes(image) != target:
        for position in range(length):
            if image[position] != target[position]:
                total += 1
                first = position if first is None else first
                if owner[position] >= 0:
                    differing[owner[position]] += 1
    return Comparison(bytes(image), owner, written, differing, total, first, owner.count(-1), overlaps, spills, outside,
                      yielded)


def _padding_yields(earlier: Piece, later: Piece, linked: dict[str, bytes], address: int) -> bool:
    """The earlier piece's byte at ``address`` is trailing zero padding of a data section, and the later
    piece is a data section the recipe places there: only then does the later one own the byte."""
    if earlier.section == ".text" or later.section == ".text" or later.placement != "declared base":
        return False
    body = linked.get(earlier.out) or b""
    return not any(body[address - earlier.vram:min(earlier.own, len(body))])


def likely_step(where: dict, linked: bytes, target: bytes, position: int) -> str:
    """The build step a first difference most likely comes from, from the placement class and the
    differing instruction word: a hole or raw blob is extraction/layout; a section placed at an
    assumed or scratch base is placement; in code, a word that differs only in a relocatable field
    (the low 16 bits of an immediate, or the 26-bit target of j/jal) is link (symbol binding or
    section base); any other difference in a compiled unit is compile (recipe, flags or source)."""
    cls, placement = where.get("class", ""), where.get("placement", "")
    if cls.startswith("hole"):
        return "layout: no piece owns these bytes"
    if placement == "raw" or cls.startswith("raw"):
        return "extraction: a raw blob differs from the ROM it was cut from (should not happen)"
    if placement in ("assumed base", "scratch"):
        return "placement: the section has no declared address"
    if where.get("section") != ".text":
        return "data: the unit's data differs (initialisers, layout or a relocated pointer)"
    word = position - position % 4
    a, b = linked[word:word + 4], target[word:word + 4]
    if len(a) == 4 and len(b) == 4:
        x, y = int.from_bytes(a, "big"), int.from_bytes(b, "big")
        if x >> 26 == y >> 26 and x >> 26 in (2, 3) and (x ^ y) & 0xFC000000 == 0:
            return "link: a call or jump target differs (symbol binding)"
        if (x ^ y) & 0xFFFF0000 == 0:
            return "link: an immediate differs (a %hi/%lo symbol address, a section base, or a constant)"
    return "compile: the instruction itself differs (recipe, flags or source)"


def _where(layout: Plan, units: list[Target], members: list[Member], result: Comparison, position: int,
           delta: int, is_rom: bool) -> dict:
    address = layout.start + position
    out = {"vram": _hx(address), "image_offset": f"0x{position:X}",
           ("rom_offset" if is_rom else "image_file_offset"): f"0x{address - delta:X}"}
    number = result.owner[position]
    if number < 0:
        out["class"] = "hole (no piece owns it)"
    else:
        p = layout.pieces[number]
        out.update({"class": p.cls, "piece": p.out, "object": p.obj, "section": p.section, "placement": p.placement,
                    "offset_in_section": f"0x{address - p.vram:X}"})
        if p.target >= 0:
            out["units"] = [m.function for m in units[p.target].members]
    unit = next((m for m in members if m.vram <= address < m.vram + max(m.size, 1)), None)
    if unit is not None:
        out["catalogue_unit"] = f"{unit.function}+0x{address - unit.vram:X} ({unit.status})"
    return out


def _unit_rows(units: list[Target], objects: dict, layout: Plan, result: Comparison) -> tuple[list[str], Counter]:
    by_target: dict[int, list[int]] = {}
    for number, p in enumerate(layout.pieces):
        if p.target >= 0:
            by_target.setdefault(p.target, []).append(number)
    rows = ["object\tfunctions\tstatus\tkind\tvram\tslot\tlinked\ttext_bytes\tdata_bytes\traw_in_slot\tdiffering_bytes"
            "\tunplaced_sections\treason"]
    summary: Counter = Counter()
    for number, t in enumerate(units):
        pieces = by_target.get(number, [])
        text = sum(result.written[i] for i in pieces if layout.pieces[i].section == ".text")
        data = sum(result.written[i] for i in pieces if layout.pieces[i].section != ".text")
        differing = sum(result.differing[i] for i in pieces)
        unplaced = ",".join(layout.pieces[i].section + ("*" if layout.pieces[i].referenced else "")
                            for i in pieces if layout.pieces[i].cls == "unplaced") or "-"
        linked = number in objects
        summary["raw" if not linked else "linked-different" if differing else "linked-identical"] += 1
        rows.append("\t".join([t.obj or "-", ",".join(m.function for m in t.members), t.status, t.kind, _hx(t.vram),
                               f"0x{t.slot:X}", "yes" if linked else "no", str(text), str(data),
                               str(max(0, t.slot - text)), str(differing), unplaced, t.reason or "-"]))
    return rows, summary


# ---------------------------------------------------------------------------------------------
# the command


def check_build(build, project: Path) -> Path:
    """Refuse a build directory that would put ROM bytes into the project tree, the kit, a git work
    tree, the filesystem root or the home directory, or that is a non-empty foreign directory."""
    build = Path(build).expanduser().resolve()
    if build == Path(build.anchor) or build == Path.home().resolve():
        raise RelinkError(f"{build}: refusing the filesystem root or the home directory")
    project = Path(project).resolve()
    if build == project or project in build.parents or build in project.parents:
        raise RelinkError(f"{build}: overlaps the project tree {project}; ROM bytes never go into the project: "
                          "name a build directory outside it")
    kit = Path(__file__).resolve().parents[2]
    if build == kit or kit in build.parents or build in kit.parents:
        raise RelinkError(f"{build}: overlaps the kit's own tree {kit}")
    for folder in (build, *build.parents):
        if (folder / ".git").exists():
            raise RelinkError(f"{build}: inside the git work tree {folder}; the build holds ROM bytes: keep it out of "
                              "anything that is committed")
    if build.exists():
        if not build.is_dir():
            raise RelinkError(f"{build}: not a directory")
        if any(build.iterdir()) and not (build / MARKER).is_file():
            raise RelinkError(f"{build}: not empty and not an earlier relink build ({MARKER} missing)")
    return build


def _reset(build: Path) -> None:
    for name in OUTPUTS:
        path = build / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def _limited(items, limit: int) -> list:
    items = list(items)
    return items[:limit] + ([f"... {len(items) - limit} more"] if len(items) > limit else [])


def relink(project, image: str, build, *, data: bytes, delta: int, is_rom: bool, extent=None, cfg=None,
           assumed: str = "raw", linker: str = "gnu-ld", rom_sha1: str | None = None, any_rom: bool = False,
           build_fn=None, link_fn=None, limit: int = 50, keep_image: bool = False) -> dict:
    """Build, plan, link and compare one image (see the module docstring); returns the report, which
    is also written to ``<build>/RELINK.json``.  ``data`` is the image file (the ROM for ``main``)
    and ``delta`` its VRAM minus file offset.  ROM bytes persist only in ``raw/raw.o``, the declared local
    extraction input; the linked image (``relinked.elf``, ``relinked.bin``) is kept only with ``keep_image``."""
    if assumed not in ("raw", "link"):
        raise RelinkError(f"--assumed {assumed!r}: raw or link")
    if linker not in ("gnu-ld", "kit"):
        raise RelinkError(f"--linker {linker!r}: gnu-ld or kit")
    project = Path(project).expanduser().resolve()
    if not (project / pg.MANIFEST).is_file():
        raise RelinkError(f"{project}: not a decompile-project tree ({pg.MANIFEST} missing)")
    build = check_build(build, project)
    members = read_members(project, image)
    expected = recorded_rom_sha1(project)
    if expected and rom_sha1 and rom_sha1 != expected and not any_rom:
        raise RelinkError(f"this ROM's SHA-1 is not the one the project describes ({expected}); --any-rom to try")
    start, end = extent or image_extent(project, image, len(data), delta, is_rom)
    if end <= start or start - delta < 0 or end - delta > len(data):
        raise RelinkError(f"image extent {_hx(start)}..{_hx(end)} is empty or outside the image file "
                          f"({len(data)} bytes, VRAM - offset {_hx(delta)})")
    target = bytes(data[start - delta:end - delta])
    build.mkdir(parents=True, exist_ok=True)
    _reset(build)
    (build / MARKER).write_text(json.dumps({"schema": SCHEMA, "state": "running"}) + "\n")
    units = targets(members, assumed, read_assumptions(project, image))
    objects = build_targets(units, project, build, cfg, build_fn)
    from_raw = raw_unit_pins(units, objects, project, image)
    layout = plan(units, objects, (start, end), {**from_raw, **declared_symbols(project, image)})
    raw = {p.out: target[p.vram - start:p.vram - start + p.size] for p in layout.pieces if p.target < 0}
    (build / "raw").mkdir(exist_ok=True)
    if raw:                                          # the declared local extraction input: ROM bytes
        (build / "raw" / "raw.o").write_bytes(write_elf(
            [(p.section, raw[p.out], SHF_ALLOC, 1, 0) for p in layout.pieces if p.target < 0]))
    (build / "link.ld").write_text(linker_script(image, layout))
    problems, diagnostics, returncode = [], [], 0
    if linker == "kit":
        linked, problems = kit_link(objects, layout, raw)
    else:
        inputs = [t.obj for number, t in enumerate(units) if number in objects] + (["raw/raw.o"] if raw else [])
        try:
            path, log, returncode = (link_fn or gnu_link)(cfg, build, inputs)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            path, log, returncode = None, f"linker not run: {error}", -1
        diagnostics = [line for line in log.splitlines() if line.strip()]
        if path is None:
            report = {"schema": SCHEMA, "credit": CREDIT, "image": image, "outcome": "link-failed",
                      "linker": linker, "returncode": returncode, "diagnostics": _limited(diagnostics, limit),
                      "level": "unchanged: 2 (unit-verified) at most", "level_blockers": ["the link failed"]}
            (build / MARKER).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
            return report
        linked, problems = collect_linked(path.read_bytes(), layout)
        if not keep_image:
            path.unlink()                            # it holds the raw blobs: not a kept build output
    result = compare_image(layout, linked, target)
    if keep_image:
        (build / "relinked.bin").write_bytes(result.image)
    rows, summary = _unit_rows(units, objects, layout, result)
    (build / "units.tsv").write_text("\n".join(rows) + "\n")
    shutil.rmtree(build / "work", ignore_errors=True)
    coverage = {cls: 0 for cls in CLASSES}
    text_bytes = 0
    for number, p in enumerate(layout.pieces):
        if p.cls in coverage:
            coverage[p.cls] += result.written[number]
        text_bytes += result.written[number] if p.cls in SOURCE and p.section == ".text" else 0
    source_bytes = sum(coverage[cls] for cls in SOURCE)
    by_class: Counter = Counter()
    for number, count in enumerate(result.differing):
        if count:
            by_class[layout.pieces[number].cls] += count
    outcome = "identical" if result.total == 0 and result.holes == 0 else "different"
    raw_members = sum(len(t.members) for number, t in enumerate(units) if number not in objects)
    blocking = [line for line in diagnostics if _BLOCKING.search(line) and "entry symbol" not in line]
    referenced = [f"{p.obj}({p.section})" for p in layout.pieces if p.referenced]
    blockers = []
    if outcome != "identical":
        blockers.append("the linked image is not byte-identical to the image in the ROM")
    if raw_members:
        blockers.append(f"{raw_members} function units are raw stand-ins ({coverage['raw-unit']} bytes): level 3 "
                        "needs every unit linked from its source")
    if layout.assumed_used:
        blockers.append("assumed units were linked: addresses read from the target took part")
    if layout.unresolved:
        blockers.append(f"{len(layout.unresolved)} names nothing declares (bound to a sentinel)")
    if referenced:
        blockers.append(f"{len(referenced)} sections the code refers to have no declared address")
    if layout.duplicates or layout.pin_conflicts:
        blockers.append("conflicting definitions or pins")
    if problems or blocking or returncode or result.spills or result.overlaps or result.holes or layout.unscriptable:
        blockers.append("the link reported problems (see problems, diagnostics, spills, overlaps)")
    level = (f"3 (pinned relink) for image {image} alone: every unit linked from its source (C or assembly) at its "
             "own address, the bytes no unit owns (data, rodata, padding) linked as raw blobs, the image identical; "
             "levels 4-6 (order relink and above) are not tested" if not blockers else
             "unchanged: 2 (unit-verified) at most; this relink moves no level (see level_blockers)")
    first = (_where(layout, units, members, result, result.first, delta, is_rom)
             if result.first is not None else None)
    if first is not None and result.first is not None:
        first["likely_step"] = likely_step(first, result.image, target, result.first)
    if first is None and result.holes:
        first = _where(layout, units, members, result, result.owner.index(-1), delta, is_rom)
    files = sorted(path.relative_to(build).as_posix() for path in build.rglob("*") if path.is_file())
    report = {
        "schema": SCHEMA, "credit": CREDIT, "image": image, "linker": linker, "assumed": assumed,
        "extent": {"vram": _hx(start), "end": _hx(end), "bytes": end - start,
                   ("rom_offset" if is_rom else "image_file_offset"): f"0x{start - delta:X}"},
        "outcome": outcome, "differing_bytes": result.total, "first_difference": first,
        "coverage": coverage, "source_bytes": source_bytes, "source_text_bytes": text_bytes,
        "source_data_bytes": source_bytes - text_bytes, "raw_bytes": coverage["raw-unit"] + coverage["raw-other"],
        "source_share": round(source_bytes / (end - start), 4),
        "units": {"in_image": len(members), "objects": len(units), "linked": len(objects),
                  "raw_members_by_reason": dict(sorted(Counter(t.reason_code for t in units if t.reason
                                                               for _m in t.members).items())),
                  "objects_by_result": dict(sorted(summary.items()))},
        "differing_by_class": dict(sorted(by_class.items())),
        "symbols": {"provided": len(layout.provide), "hard": len(layout.hard),
                    "unresolved": _limited(layout.unresolved, limit), "stated": len(layout.stated),
                    "from_raw_units": len(from_raw), "duplicates": _limited(layout.duplicates, limit),
                    "pin_conflicts": _limited(layout.pin_conflicts, limit),
                    "declared_vs_linked": _limited([f"{n}: linked {_hx(a)}, declared {_hx(d)}"
                                                    for n, a, d in layout.disagreements], limit),
                    "common_without_address": _limited(layout.commons, limit),
                    "unscriptable": _limited(layout.unscriptable, limit), "gp_relative": layout.uses_gp},
        "sections": {"placed_by_declared_symbols": len(layout.derived),
                     "unplaced": sum(1 for p in layout.pieces if p.cls == "unplaced"),
                     "unplaced_referenced": _limited(referenced, limit),
                     "inconsistent_symbol_bases": _limited(layout.inconsistent, limit)},
        "problems": _limited(problems, limit),
        "spills": _limited([f"{layout.pieces[n].obj}(.text) past its slot" for n in result.spills], limit),
        "overlaps": _limited([f"{layout.pieces[n].obj}({layout.pieces[n].section}): {count} bytes already owned"
                              for n, count in result.overlaps], limit),
        "padding_yielded_bytes": result.yielded,
        "outside_extent": _limited([f"{layout.pieces[n].obj}({layout.pieces[n].section})" for n in result.outside],
                                   limit),
        "holes": result.holes, "returncode": returncode, "diagnostics": _limited(diagnostics, limit),
        "level": level, "level_blockers": blockers, "files": _limited(files, limit),
        "rom_bytes_kept_in": ["raw/raw.o (the declared local extraction input)"]
                             + ((["relinked.bin"] + (["relinked.elf"] if linker == "gnu-ld" else []))
                                if keep_image else []),
    }
    (build / MARKER).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report
