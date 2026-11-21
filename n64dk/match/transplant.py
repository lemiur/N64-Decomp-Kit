"""Donor transplant: recover target functions from C already matched in another project.

A function in a target image that is machine-code identical to a function some other local
project has already matched as C (after masking the fields a linker rewrites) is recovered
by compiling the donor's proven source with the donor's proven recipe and verifying it at
the target address.  This is how SDK and library code (libultra, libaudio/libmus, libc
pieces, compiler runtime, shared engine code) is recovered without re-decompiling it.

Stages (each usable on its own):

``build_index``  walk every C row of every donor manifest, cut the retail bytes of the unit
                 into functions (donor catalogue starts), and store per function a
                 relocation-masked signature, the raw bytes, the masked sites with the donor
                 address and donor symbol name at each, and the donor unit recipe.  Persisted
                 as JSON; a unit whose source, manifest row and retail bytes are unchanged is
                 reused without recomputation.
``scan``         exact masked-signature hits (same size) for each target function, plus
                 *near* hits (at most ``near_max`` differing words) that are reported but
                 never transplanted.
``transfer``     align donor and target instruction by instruction and derive the target
                 address of each donor symbol at a masked site; check that every donor symbol
                 maps to one target address across all hits (contradicting hits go to
                 ``needs-review``) and flag different donor symbols landing on one address.
``transplant``   build the donor unit in a scratch workspace (donor include directories,
                 donor flags), relocate it with the transferred symbols and verify it with
                 :func:`n64dk.match.verify.strict_compare`.  Verdicts:
                 ``exact-with-transferred-symbols``, ``mismatch``, ``build-failed``; units
                 only partly present in the target are ``partial-unit``.

Masking rules: the 26-bit target of ``j``/``jal``; the immediate of every ``lui``; the
immediate of each ``%lo`` user found by the same straight-line pairing the m2c bridge uses
(``m2c_bridge._pairs``).  Ordinary immediates (stack offsets, field offsets, small
constants, branch offsets) are kept.  Trailing zero words are not part of a signature, so
alignment padding never prevents a hit.

Nothing here writes into a donor or target project: builds run in a scratch project
directory whose entries are symlinks to the donor's top level, and outputs go to the
caller's directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import tempfile
import time
import tomllib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..elf import R_MIPS_26, Elf
from . import compare as cmp
from . import symbols as symmod
from .m2c_bridge import _pairs
from .pipeline import BuildError, BuildSpec, build, split_flags
from .toolchain import Config

INDEX_VERSION = 3
EXACT_VERDICT = "exact-with-transferred-symbols"
# The unit's code is byte-identical, and an address it uses was read from the target's own
# instructions with nothing independent to confirm the name it was given (a section of the unit
# with no contents to compare, a data symbol no other unit placed).  The same standing as a
# resolved case that ends ``assumed``: identical bytes, awaiting symbols.
ASSUMED_VERDICT = "identical-awaiting-symbols"
CORROBORATING_BYTES = 8         # a section's contents confirm its address from this many bytes on
NEAR_MIN_WORDS = 6
SAME_FUNCTION_MIN = 12          # bytes of a catalogued function before its code can name it
DATA_NONZERO = 8                # non-zero bytes a datum needs before its contents say where it is
AUTO_NAME = re.compile(r"^(?:func|D|B|jtbl|L|sym|data)_[0-9A-Fa-f]{6,8}$|^0x[0-9A-Fa-f]+$")
INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
HEX = re.compile(r"^(0x)?[0-9A-Fa-f]+$")


# --------------------------------------------------------------------------- donors

@dataclass
class Manifest:
    path: str                 # relative to the donor root
    pipeline: str             # kmc+builtin, kmc+as, kmc+asn64, ido5.3, ido7.1, sn


@dataclass
class DonorSpec:
    name: str
    root: Path
    image: Path
    manifests: list[Manifest]
    vram_delta: int | None = None
    symbols: list[Path] = field(default_factory=list)
    catalogue: list[Path] = field(default_factory=list)
    include: list[Path] = field(default_factory=list)
    kmc_lane: str = "kmc+builtin"


def _abs(root: Path, value: str) -> Path:
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else root / path


def donor_from_dict(name: str, data: dict) -> DonorSpec:
    root = Path(os.path.expanduser(data["root"]))
    manifests = []
    for item in data.get("manifests", []):
        if isinstance(item, str):
            path, _, pipe = item.partition("=")
            manifests.append(Manifest(path, pipe or "kmc+builtin"))
        else:
            manifests.append(Manifest(item["path"], item.get("pipeline", "kmc+builtin")))
    delta = data.get("vram_delta")
    return DonorSpec(name, root, _abs(root, data["image"]), manifests,
                     int(delta, 0) if isinstance(delta, str) else delta,
                     [_abs(root, p) for p in data.get("symbols", [])],
                     [_abs(root, p) for p in data.get("catalogue", [])],
                     [_abs(root, p) for p in data.get("include", [])],
                     data.get("kmc_lane", "kmc+builtin"))


def load_donors(path: str | Path, only: list[str] | None = None, exclude: list[str] | None = None) -> list[DonorSpec]:
    """``[donor.<name>]`` tables (see ``donors.example.toml``)."""
    data = tomllib.loads(Path(path).read_text())
    out = [donor_from_dict(name, table) for name, table in data.get("donor", {}).items()]
    if only:
        out = [d for d in out if d.name in only]
    if exclude:
        out = [d for d in out if d.name not in exclude]
    return out


def load_functions(paths, delta: int | None = None) -> list[tuple[str, int, int]]:
    """(name, vram, size) from recomp/splat catalogues or ``ROM SIZE VRAM NAME`` TSVs."""
    from ..analysis.functions import load_catalogue
    out: dict[int, tuple[str, int, int]] = {}
    for path in paths:
        rows = load_catalogue(path)
        if not rows:
            for line in Path(path).read_text().splitlines():
                cols = line.split("\t") if "\t" in line else line.split()
                if line.startswith("#") or len(cols) < 3 or not all(HEX.match(c) for c in cols[:2]):
                    continue
                size = int(cols[1], 16)
                if len(cols) >= 4 and HEX.match(cols[2]) and cols[2].lower().startswith("0x8"):
                    rows.append((cols[3], int(cols[2], 16), size))
                elif delta is not None:
                    rows.append((cols[2] if len(cols) > 2 else f"func_{int(cols[0], 16) + delta:08X}",
                                 int(cols[0], 16) + delta, size))
        for name, vram, size in rows:
            if size > 0:
                out.setdefault(vram, (name, vram, size))
    return sorted(out.values(), key=lambda r: r[1])


def manifest_intervals(paths, delta: int) -> list[tuple[int, int]]:
    """VRAM intervals of manifest rows (first two hex columns are ROM start and size)."""
    out = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split("\t") if "\t" in line else line.split()
            if len(cols) >= 2 and HEX.match(cols[0]) and HEX.match(cols[1]):
                start = int(cols[0], 16) + delta
                out.append((start, start + int(cols[1], 16)))
    return out


# --------------------------------------------------------------------------- masking

def words_of(code: bytes) -> list[int]:
    return [struct.unpack_from(">I", code, i)[0] for i in range(0, len(code) - len(code) % 4, 4)]


def strip_padding(code: bytes) -> bytes:
    end = len(code) - len(code) % 4
    while end >= 4 and code[end - 4:end] == b"\0\0\0\0":
        end -= 4
    return code[:end]


def mask(code: bytes, vram: int) -> tuple[list[int], list[tuple[int, str, int]]]:
    """Masked words and the masked sites ``(word index, kind, full address)``.

    ``kind`` is ``jal``/``j`` (26-bit target), ``hi`` (paired ``lui``), ``lui`` (unpaired
    ``lui``, address ``upper << 16``) or ``lo`` (``%lo`` user)."""
    words = words_of(code)
    pairs = _pairs(words, vram)
    out, sites = [], []
    for i, word in enumerate(words):
        op, site = word >> 26, vram + 4 * i
        if op in (2, 3):
            out.append(word & 0xFC000000)
            sites.append((i, "jal" if op == 3 else "j", ((site + 4) & 0xF0000000) | ((word & 0x3FFFFFF) << 2)))
        elif op == 15:
            out.append(word & 0xFFFF0000)
            if site in pairs:
                sites.append((i, "hi", pairs[site]))
            else:   # unpaired lui: a constant (or an address the tracker lost); masked, never transferred
                sites.append((i, "lui", (word & 0xFFFF) << 16))
        elif site in pairs:
            out.append(word & 0xFFFF0000)
            sites.append((i, "lo", pairs[site]))
        else:
            out.append(word)
    return out, sites


def signature(masked: list[int]) -> str:
    return hashlib.sha1(struct.pack(f">{len(masked)}I", *masked)).hexdigest()


def chunk_keys(masked: list[int], parts: int = 3) -> list[str]:
    """Pigeonhole keys: two masked sequences with at most ``parts - 1`` differing words share
    at least one identical chunk."""
    n = len(masked)
    bounds = [n * k // parts for k in range(parts + 1)]
    return [f"{n}:{k}:" + signature(masked[bounds[k]:bounds[k + 1]]) for k in range(parts)]


# --------------------------------------------------------------------------- index

def _name_at(address: int, by_addr: dict[int, str], sorted_addrs: list[int], sizes: dict[str, int]) -> tuple[str | None, int]:
    """Donor symbol covering ``address``: exact, else the closest lower symbol whose known size covers it."""
    if address in by_addr:
        return by_addr[address], 0
    import bisect
    k = bisect.bisect_right(sorted_addrs, address) - 1
    if k >= 0:
        base = sorted_addrs[k]
        name = by_addr[base]
        if address - base < sizes.get(name, 0):
            return name, address - base
    return None, 0


class DonorContext:
    """Loaded donor: image bytes, address mapping, symbol map and function starts."""

    def __init__(self, spec: DonorSpec):
        from ..rom import Rom
        from .engine import Target
        self.spec = spec
        try:
            self.target = Target.from_rom(Rom.load(spec.image), spec.vram_delta)
        except Exception:
            # Not a cartridge image: a linear one whose load address is declared (a segment a
            # blind run unpacked, a library linked for the purpose).  Offset 0 is that address.
            if spec.vram_delta is None:
                raise
            data = Path(spec.image).read_bytes()
            self.target = Target(data, spec.vram_delta, (0, len(data)))
        self.symbols = symmod.load([p for p in spec.symbols if p.is_file()])
        self.functions = load_functions([p for p in spec.catalogue if p.is_file()], self.target.delta)
        for name, vram, size in self.functions:
            self.symbols.setdefault(name, vram)
            self.symbols.sizes.setdefault(name, size)
        self.by_addr: dict[int, str] = {}
        for name, address in self.symbols.items():   # prefer descriptive names over auto names
            old = self.by_addr.get(address)
            if old is None or (AUTO_NAME.match(old) and not AUTO_NAME.match(name)):
                self.by_addr[address] = name
        self.sorted = sorted(self.by_addr)
        self.starts = sorted(v for _n, v, _s in self.functions)

    def name_at(self, address: int) -> tuple[str | None, int]:
        return _name_at(address, self.by_addr, self.sorted, self.symbols.sizes)


def _unit_rows(donor: DonorSpec, ctx: DonorContext):
    from . import manifest as manmod
    for man in donor.manifests:
        path = donor.root / man.path
        if not path.is_file():
            continue
        lines = path.read_text().splitlines()
        lane = man.pipeline if man.pipeline.startswith("kmc") and man.pipeline != "kmc" else donor.kmc_lane
        pipe = "kmc" if man.pipeline.startswith("kmc") else man.pipeline
        try:
            units = manmod.parse(path, donor.root, ctx.target.rom_to_vram, pipe, lane)
        except Exception:                                   # noqa: BLE001 - foreign manifest layout
            continue
        for unit in units:
            if unit.source.suffix not in (".c", ".s"):       # C, and assembly a library was written in
                continue
            row = lines[unit.line - 1] if 0 < unit.line <= len(lines) else ""
            yield man, unit, row


def _function_spans(ctx: DonorContext, vram: int, size: int) -> list[tuple[str, int, int]]:
    import bisect
    lo = bisect.bisect_left(ctx.starts, vram)
    hi = bisect.bisect_left(ctx.starts, vram + size)
    starts = [vram] + [s for s in ctx.starts[lo:hi] if s != vram]
    out = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else vram + size
        name = ctx.by_addr.get(start) or f"func_{start:08X}"
        out.append((name, start, end - start))
    return out


def function_record(code: bytes, vram: int, name: str, offset: int, span: int, position: int,
                    name_at=None, unit_range: tuple[int, int] | None = None) -> dict:
    """Index record of one donor function (``code`` already stripped of trailing padding)."""
    masked, sites = mask(code, vram)
    lo, hi = unit_range or (vram, vram + len(code))
    site_rows = []
    for i, kind, address in sites:
        sym, addend = name_at(address) if name_at else (None, 0)
        local = lo <= address < hi and kind in ("j", "jal")
        site_rows.append([i, kind, address, sym, addend, local])
    return {"name": name, "vram": vram, "offset": offset, "size": len(code), "span": span,
            "bytes": code.hex(), "masked": signature(masked),
            "chunks": chunk_keys(masked) if len(masked) >= NEAR_MIN_WORDS else [],
            "sites": site_rows, "position": position}


def index_unit(ctx: DonorContext, man: Manifest, unit, row: str) -> dict:
    target = ctx.target
    offset = target.vram_to_rom(unit.vram)
    raw = target.rom[offset:offset + unit.size]
    spans = _function_spans(ctx, unit.vram, unit.size)
    functions = []
    for k, (name, start, size) in enumerate(spans):
        code = strip_padding(raw[start - unit.vram:start - unit.vram + size])
        if code:
            functions.append(function_record(code, start, name, start - unit.vram, size, k, ctx.name_at,
                                             (unit.vram, unit.vram + unit.size)))
    core = strip_padding(raw)
    unit_masked, _ = mask(core, unit.vram)
    flags = row.split("\t")[3] if row.count("\t") >= 3 else ""
    return {"donor": ctx.spec.name, "source": str(unit.source.relative_to(ctx.spec.root)) if unit.source.is_relative_to(ctx.spec.root) else str(unit.source),
            "manifest": man.path, "line": unit.line, "pipeline": unit.spec.pipeline,
            "flags_text": flags.strip(), "flags": unit.spec.flags, "normalizer_args": unit.spec.normalizer_args,
            "section_bases": unit.spec.section_bases, "symbols": unit.spec.symbols,
            "vram": unit.vram, "size": unit.size, "core_size": len(core), "unit_masked": signature(unit_masked),
            "functions": functions}


def _unit_digest(donor: DonorSpec, unit, row: str, raw: bytes) -> str:
    h = hashlib.sha256()
    h.update(f"{INDEX_VERSION}|{donor.name}|{row}|".encode())
    try:
        h.update(unit.source.read_bytes())
    except OSError:
        h.update(b"missing")
    h.update(raw)
    return h.hexdigest()


class Index:
    """Donor functions keyed by masked signature, with chunk keys for near matches."""

    def __init__(self, units: dict[str, dict] | None = None, donors: dict[str, dict] | None = None):
        self.units = units or {}            # digest -> unit record
        self.donors = donors or {}          # donor name -> spec summary
        self._build()

    def _build(self):
        self.by_sig: dict[tuple[str, int], list[tuple[dict, dict]]] = defaultdict(list)
        self.by_chunk: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for unit in self.units.values():
            for fn in unit["functions"]:
                self.by_sig[(fn["masked"], fn["size"])].append((unit, fn))
                for key in fn["chunks"]:
                    self.by_chunk[key].append((unit, fn))

    def stats(self) -> dict:
        per = defaultdict(Counter)
        for unit in self.units.values():
            per[unit["donor"]][unit["pipeline"]] += len(unit["functions"])
            per[unit["donor"]]["units"] += 1
        return {d: dict(c) for d, c in sorted(per.items())}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": INDEX_VERSION, "donors": self.donors, "units": self.units}))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        data = json.loads(Path(path).read_text())
        if data.get("version") != INDEX_VERSION:
            return cls()
        return cls(data["units"], data.get("donors", {}))


def build_index(donors: list[DonorSpec], path: str | Path | None = None) -> tuple[Index, dict]:
    """Index every C unit of ``donors``; reuse records from ``path`` whose digest is unchanged."""
    old = Index.load(path) if path and Path(path).is_file() else Index()
    units: dict[str, dict] = {}
    info = {"reused": 0, "indexed": 0, "skipped_missing": 0}
    donor_info = {}
    for donor in donors:
        ctx = DonorContext(donor)
        donor_info[donor.name] = {"root": str(donor.root), "image": str(donor.image), "delta": ctx.target.delta,
                                  "include": [str(p) for p in donor.include]}
        for man, unit, row in _unit_rows(donor, ctx):
            if not unit.source.is_file():
                info["skipped_missing"] += 1
                continue
            offset = ctx.target.vram_to_rom(unit.vram)
            digest = _unit_digest(donor, unit, row, ctx.target.rom[offset:offset + unit.size])
            if digest in old.units:
                units[digest] = old.units[digest]
                info["reused"] += 1
            else:
                units[digest] = index_unit(ctx, man, unit, row)
                info["indexed"] += 1
    index = Index(units, donor_info)
    if path:
        index.save(path)
    return index, info


# --------------------------------------------------------------------------- scan

@dataclass
class Hit:
    name: str
    vram: int
    size: int
    unit: dict
    function: dict
    kind: str                        # exact | near
    diffs: list[int] = field(default_factory=list)       # word indices that differ (near)
    sites: list[dict] = field(default_factory=list)      # aligned masked sites
    status: str = "candidate"        # candidate | needs-review | local-inconsistent | constant-differs | partial-unit
    notes: list[str] = field(default_factory=list)
    template: bool = False           # generic code shape (one donor function hit by several targets)
    lost_lo: bool = False            # near: only %lo halves the pair tracker could not follow differ
    placed: bool = False             # its place is fixed by its source neighbours (see ``corroborate``)

    @property
    def donor(self) -> str:
        return self.unit["donor"]

    def row(self) -> dict:
        return {"target": self.name, "vram": f"0x{self.vram:08X}", "size": self.size, "kind": self.kind,
                "donor": self.donor, "donor_unit": self.unit["source"], "donor_function": self.function["name"],
                "donor_pipeline": self.unit["pipeline"], "status": self.status, "notes": self.notes,
                **({"differing_words": [f"0x{4 * d:X}" for d in self.diffs]} if self.diffs else {})}


def _align(hit: Hit, target_code: bytes) -> None:
    """Pair each donor masked site with the target address at the same instruction."""
    _masked, tsites = mask(target_code, hit.vram)
    tmap = {i: (kind, address) for i, kind, address in tsites}
    fn = hit.function
    for i, kind, daddr, sym, addend, local in fn["sites"]:
        tk = tmap.get(i)
        if tk is None or tk[0] != kind:
            continue
        taddr = tk[1]
        if kind == "lui":
            # an unpaired lui is usually a constant: a different value is a different function
            if taddr != daddr and not (0x80000000 <= daddr < 0xC0000000 and 0x80000000 <= taddr < 0xC0000000):
                hit.status = "constant-differs"
                hit.notes.append(f"lui constant at +0x{4 * i:X}: donor 0x{daddr >> 16:04X}, target 0x{taddr >> 16:04X}")
            continue
        if local:
            expect = hit.vram + (daddr - fn["vram"])
            if taddr != expect:
                hit.status = "local-inconsistent"
                hit.notes.append(f"local {kind} at +0x{4 * i:X}: donor offset does not carry to the target")
            continue
        hit.sites.append({"offset": 4 * i, "kind": kind, "donor_address": daddr, "target_address": taddr,
                          "donor_symbol": sym, "addend": addend})


def _lost_lo(a: int, b: int) -> bool:
    """Words differing only in imm16 of an addiu/load/store whose base is not $sp."""
    if a >> 16 != b >> 16:
        return False
    op, rs = a >> 26, (a >> 21) & 31
    return rs not in (0, 29) and (op == 9 or 0x20 <= op <= 0x3F)


def only_lost_lo(dmasked: list[int], masked: list[int], code: bytes) -> list[int] | None:
    """The word indices where a donor function and target code of the same length differ, when
    every one of them is a ``%lo`` half the straight-line pair tracker lost (its ``lui`` is on
    another path, so the immediate was not masked): the masked words differ, the code may not.
    ``None`` when the lengths differ or any other word does; ``[]`` when nothing differs."""
    if len(dmasked) != len(masked):
        return None
    diffs = [i for i, (a, b) in enumerate(zip(dmasked, masked)) if a != b]
    bases = {w >> 16 & 31 for w in words_of(code) if w >> 26 == 15}          # registers a lui sets
    if all(_lost_lo(dmasked[i], masked[i]) and (masked[i] >> 21) & 31 in bases for i in diffs):
        return diffs
    return None


def scan(image: bytes, vram_delta: int, functions: list[tuple[str, int, int]], index: Index,
         near_max: int = 2) -> list[Hit]:
    """Exact and near masked-signature hits for each target ``(name, vram, size)``."""
    hits: list[Hit] = []
    for name, vram, size in functions:
        offset = vram - vram_delta
        if offset < 0 or offset + size > len(image):
            continue
        code = strip_padding(image[offset:offset + size])
        if not code:
            continue
        masked, _sites = mask(code, vram)
        exact = index.by_sig.get((signature(masked), len(code)), [])
        seen = set()
        for unit, fn in exact:
            hit = Hit(name, vram, len(code), unit, fn, "exact")
            _align(hit, code)
            hits.append(hit)
            seen.add(id(fn))
        if near_max <= 0 or len(masked) < NEAR_MIN_WORDS or exact:
            continue
        candidates = {}
        for key in chunk_keys(masked):
            for unit, fn in index.by_chunk.get(key, []):
                if fn["size"] == len(code) and id(fn) not in seen:
                    candidates[id(fn)] = (unit, fn)
        for unit, fn in candidates.values():
            dmasked, _ = mask(bytes.fromhex(fn["bytes"]), fn["vram"])
            diffs = [i for i, (a, b) in enumerate(zip(dmasked, masked)) if a != b]
            if 0 < len(diffs) <= near_max:
                hit = Hit(name, vram, len(code), unit, fn, "near", diffs, status="near")
                bases = {w >> 16 & 31 for w in words_of(code) if w >> 26 == 15}      # registers a lui sets
                if all(_lost_lo(dmasked[i], masked[i]) and (masked[i] >> 21) & 31 in bases for i in diffs):
                    hit.lost_lo = True
                    hit.notes.append("only 16-bit immediates of non-stack loads/stores/addiu differ: probably "
                                     "%lo halves the straight-line pair tracker lost (masking too tight)")
                hits.append(hit)
    return hits


# --------------------------------------------------------------------------- symbol transfer

def _key(hit: Hit, site: dict) -> tuple[str, int]:
    """Identity of the donor object a site refers to: (donor, donor base address)."""
    return hit.donor, site["donor_address"] - site["addend"]


def source_order(index: "Index") -> dict[tuple[str, str], tuple[str, int]]:
    """``{(donor, unit source): (translation unit, position)}`` for donors split one function per
    unit (``order.json`` beside their sources, see ``n64dk.match.sourcedonor``)."""
    out = {}
    for name, info in index.donors.items():
        path = Path(info.get("root", "")) / "order.json"
        if not path.is_file():
            continue
        try:
            order = json.loads(path.read_text()).get("order", {})
        except (OSError, ValueError):
            continue
        for unit, labels in order.items():
            for position, label in enumerate(labels):
                out[(name, f"{unit[:-2]}/{label}.c")] = (unit, position)
    return out


def corroborate(hits: list[Hit], functions: list[tuple[str, int, int]], order: dict) -> int:
    """Fix the place of a function whose code stands at several places of the title.

    The same shape at several places (a getter, a setter) says nothing about which of them is
    this function, and such hits are refused.  Source order says more: a compiler lays a unit's
    functions out in the order the unit defines them, so a hit right after the place where the
    unit's previous function was found uniquely (or right before its next one) is this
    function's place.  Places fixed this way can fix their own neighbours.  Returns how many."""
    if not order:
        return 0
    sequence = [vram for _n, vram, _s in sorted(functions, key=lambda row: row[1])]
    position = {vram: k for k, vram in enumerate(sequence)}
    targets: dict[int, set[int]] = defaultdict(set)
    for hit in hits:
        if hit.kind == "exact":
            targets[id(hit.function)].add(hit.vram)
    where: dict[tuple, int] = {}                          # (donor, unit, position) -> title address
    for hit in hits:
        key = order.get((hit.donor, hit.unit["source"]))
        if hit.kind == "exact" and key and len(targets[id(hit.function)]) == 1:
            where[(hit.donor, *key)] = hit.vram
    fixed, changed = 0, True
    while changed:
        changed = False
        for hit in hits:
            key = order.get((hit.donor, hit.unit["source"]))
            if hit.kind != "exact" or hit.placed or not key or len(targets[id(hit.function)]) < 2 \
                    or hit.vram not in position:
                continue
            unit, k = key
            before, after = where.get((hit.donor, unit, k - 1)), where.get((hit.donor, unit, k + 1))
            here = position[hit.vram]
            if (before is not None and position.get(before) == here - 1) or \
                    (after is not None and position.get(after) == here + 1):
                hit.placed = True
                hit.notes.append("placed by the functions written around it in its unit")
                where.setdefault((hit.donor, unit, k), hit.vram)
                fixed += 1
                changed = True
    return fixed


_COPY_KEYS: dict[str, dict] = {}
MAX_COPIES = 4                  # copies of one function a library is expected to hold (the Pi and Si twins: two)


def donor_copies(donor: dict, vram: int, size: int) -> list[int]:
    """The addresses in a donor's image where the masked code of its function at ``vram`` stands (the
    function itself included), ascending.  Found by the code alone, so copies no donor unit lists count."""
    image = _donor_image(donor["image"])
    delta = donor["delta"]
    keys = _COPY_KEYS.get(donor["image"])
    if keys is None:
        keys = defaultdict(list)
        for offset in range(0, len(image) - 11, 4):
            keys[(image[offset] >> 2, image[offset + 4] >> 2, image[offset + 8] >> 2)].append(offset)
        _COPY_KEYS[donor["image"]] = keys
    start = vram - delta
    code = image[start:start + size]
    if start < 0 or len(code) != size or size < 12:
        return []
    want = mask(code, vram)[0]
    places = []
    for offset in keys.get((code[0] >> 2, code[4] >> 2, code[8] >> 2), ()):
        there = image[offset:offset + size]
        if len(there) == size and mask(there, offset + delta)[0] == want:
            places.append(offset + delta)
    return places


def pair_copies(hits: list[Hit], index: "Index") -> tuple[list[Hit], int]:
    """Give each copy of a function that stands several times in the title its donor copy, by their order.

    A library holds some routines twice with other data (the access queues of two devices); the same
    masked code then stands at two places of the title, and a hit says nothing about which is which.  The
    donor holds the copies too, and a library's objects keep their order from one title to the next: when
    the donor's image has as many copies as the title (at most MAX_COPIES), the k-th copy there is the
    k-th here.  The hit at that place is ``placed``; this donor function's hits at the other places are
    dropped (they are its twins').  What the pairing decides is which names the place gets: the code is
    the same at every copy, and each unit is still compared strictly at its own place.  Returns
    (hits, places fixed)."""
    targets: dict[int, set[int]] = defaultdict(set)
    for hit in hits:
        if hit.kind == "exact":
            targets[id(hit.function)].add(hit.vram)
    keep, fixed, decided = [], 0, {}
    for hit in hits:
        places = sorted(targets[id(hit.function)]) if hit.kind == "exact" else []
        if len(places) < 2 or len(places) > MAX_COPIES or hit.placed:
            keep.append(hit)
            continue
        key = id(hit.function)
        if key not in decided:
            donor = index.donors.get(hit.donor)
            try:
                copies = donor_copies(donor, hit.function["vram"], hit.function["size"]) if donor else []
            except (OSError, KeyError, ValueError):
                copies = []
            decided[key] = (places[copies.index(hit.function["vram"])]
                            if len(copies) == len(places) and hit.function["vram"] in copies else None)
        if decided[key] is None:
            keep.append(hit)
        elif decided[key] == hit.vram:
            hit.placed = True
            hit.notes.append(f"placed by the order of its {len(places)} copies in the donor and in the title")
            keep.append(hit)
            fixed += 1
    return keep, fixed


def ambiguous_sites(hits: list[Hit]) -> set[tuple[int, int]]:
    """``(id(hit), site offset)`` of the sites where a target function says nothing.

    One target function can be, masked, several functions of one donor at once (the getters of
    two different variables; the same routine written for two devices).  At a site where those
    donor functions refer to different things, the target's instruction would be given a
    different name by each of them and at most one is right: none of them is a witness for the
    address.  Sites where they all refer to the same thing (a shared callee) still count."""
    groups: dict[tuple[str, int], list[Hit]] = defaultdict(list)
    for hit in hits:
        if hit.kind == "exact":
            groups[(hit.donor, hit.vram)].append(hit)
    out: set[tuple[int, int]] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        referred: dict[int, set[int]] = defaultdict(set)
        for hit in group:
            for site in hit.sites:
                if site.get("offset") is not None:
                    referred[site["offset"]].add(_key(hit, site)[1])
        out.update((id(hit), site["offset"]) for hit in group for site in hit.sites
                   if len(referred.get(site.get("offset"), ())) > 1)
    return out


def transfer(hits: list[Hit]) -> dict:
    """Derive target addresses of donor symbols and check them across all exact hits.

    Returns ``{"symbols": {(donor, donor_addr): {...}}, "contradictions": [...], "suspicious": [...]}``
    and marks contradicting hits ``needs-review``."""
    targets_per_fn: dict[int, set[int]] = defaultdict(set)
    for hit in hits:
        if hit.kind == "exact":
            targets_per_fn[id(hit.function)].add(hit.vram)
    for hit in hits:
        if hit.kind == "exact" and len(targets_per_fn[id(hit.function)]) > 1 and hit.sites and not hit.placed:
            hit.template = True
    votes: dict[tuple[str, int], dict[int, list[Hit]]] = defaultdict(lambda: defaultdict(list))
    names: dict[tuple[str, int], str | None] = {}
    by_name: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    ambiguous = ambiguous_sites(hits)
    for hit in hits:
        if hit.kind != "exact" or hit.template or hit.status != "candidate":
            continue
        for site in hit.sites:
            if site["donor_symbol"] is None and site["donor_address"] == site["target_address"]:
                continue        # constant / hardware address, identical on both sides
            if (id(hit), site.get("offset")) in ambiguous:
                continue        # another function of the same donor names something else here
            key = _key(hit, site)
            taddr = (site["target_address"] - site["addend"]) & 0xFFFFFFFF
            if hit not in votes[key][taddr]:
                votes[key][taddr].append(hit)
            names[key] = site["donor_symbol"]
    contradictions, bad_hits = [], set()
    for key, choices in list(votes.items()):
        if len(choices) > 1:
            counts = sorted(((len({h.vram for h in hs}), a) for a, hs in choices.items()), reverse=True)
            winner = counts[0][1] if counts[0][0] >= 2 and counts[0][0] >= 3 * counts[1][0] else None
            contradictions.append({"donor": key[0], "donor_symbol": names[key], "donor_address": f"0x{key[1]:08X}",
                                   "target_addresses": {f"0x{a:08X}": sorted({h.name for h in hs})[:6]
                                                        for a, hs in choices.items()},
                                   "resolved_to": None if winner is None else f"0x{winner:08X}"})
            for a, hs in choices.items():
                if a != winner:
                    bad_hits.update(id(h) for h in hs)
            if winner is not None:
                votes[key] = {winner: choices[winner]}
    for key, choices in votes.items():
        if names[key] and not AUTO_NAME.match(names[key]):
            for taddr, hs in choices.items():
                if not all(id(h) in bad_hits for h in hs):
                    by_name[names[key]][taddr].add(key)
    for name, choices in by_name.items():
        if len(choices) > 1:
            contradictions.append({"donor_symbol": name, "cross_donor": True,
                                   "target_addresses": {f"0x{a:08X}": sorted(f"{d}:0x{v:08X}" for d, v in keys)
                                                        for a, keys in choices.items()}})
            for a in choices:
                for key in choices[a]:
                    for hs in votes[key].values():
                        bad_hits.update(id(h) for h in hs)
    for hit in hits:
        if id(hit) in bad_hits and hit.status == "candidate":
            hit.status = "needs-review"
    symbols = {}
    by_target: dict[int, set] = defaultdict(set)
    for key, choices in votes.items():
        if len(choices) != 1:
            continue
        (taddr, hs), = choices.items()
        call = any(s["kind"] in ("jal", "j") for h in hs for s in h.sites if _key(h, s) == key)
        symbols[key] = {"donor": key[0], "donor_address": key[1], "donor_symbol": names[key],
                        "target_address": taddr, "hits": len({h.name for h in hs}), "call": call,
                        "functions": sorted({h.name for h in hs}),
                        # where each supporter stands: a supporter inside a unit's own bytes is that
                        # unit, whatever the catalogue calls it (classify_pin ``span``)
                        "supporters": sorted({(h.name, h.vram, h.size) for h in hs})}
        by_target[taddr].add(key)
    suspicious = []
    for taddr, keys in by_target.items():
        per_donor = defaultdict(set)
        for donor, daddr in keys:
            per_donor[donor].add(daddr)
        for donor, daddrs in per_donor.items():
            if len(daddrs) > 1:
                suspicious.append({"target_address": f"0x{taddr:08X}", "donor": donor,
                                   "donor_symbols": sorted(f"{names[(donor, d)] or '?'}@0x{d:08X}" for d in daddrs)})
    # Where each donor function itself was found (independent of who refers to it).
    definitions: dict[tuple[str, int], set[int]] = defaultdict(set)
    for hit in hits:
        if hit.kind == "exact" and hit.status in ("candidate", "partial-unit"):
            definitions[(hit.unit["donor"], hit.function["vram"])].add(hit.vram)
    return {"symbols": symbols, "contradictions": contradictions, "suspicious": suspicious,
            "definitions": dict(definitions)}


def symbol_file(transfer_result: dict) -> str:
    """Reviewable ``symbol_addrs``-style text: one line per target address, best-supported name."""
    by_target: dict[int, list[dict]] = defaultdict(list)
    for item in transfer_result["symbols"].values():
        by_target[item["target_address"]].append(item)
    lines = ["// Proposed by n64dk transplant-scan: donor symbol names transferred by instruction alignment.",
             "// Review before use; names are the donor projects' names."]
    for taddr in sorted(by_target):
        items = sorted(by_target[taddr], key=lambda i: (i["donor_symbol"] is None or bool(AUTO_NAME.match(i["donor_symbol"] or "")), -i["hits"]))
        best = items[0]
        if best["donor_symbol"] is None:
            continue
        kind = "func" if best["call"] else "data"
        others = ", ".join(f"{i['donor']}:{i['donor_symbol']}" for i in items[1:] if i["donor_symbol"])
        lines.append(f"{best['donor_symbol']} = 0x{taddr:08X}; // type:{kind} donor:{best['donor']} "
                     f"donor_addr:0x{best['donor_address']:08X} hits:{sum(i['hits'] for i in items)}"
                     + (f" also:{others}" if others else ""))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- transplant

def scratch_project(donor_root: Path, scratch: Path) -> Path:
    """A writable directory mirroring the donor's top level through symlinks (builds and the
    object cache write here, never into the donor)."""
    scratch.mkdir(parents=True, exist_ok=True)
    for entry in donor_root.iterdir():
        if entry.name in (".n64dk", ".git", "build") or entry.name.startswith("."):
            continue
        link = scratch / entry.name
        if not link.exists() and not link.is_symlink():
            try:
                link.symlink_to(entry)
            except OSError:
                pass
    return scratch


def _spec_of(unit: dict) -> BuildSpec:
    return BuildSpec(unit["pipeline"], list(unit["flags"]), {}, {}, list(unit.get("normalizer_args", [])))


def include_closure(source: Path, roots: list[Path], limit: int = 400) -> list[Path]:
    """Headers reachable from ``source`` through ``#include`` (quote: own directory first)."""
    seen, out, todo = set(), [], [source]
    while todo and len(out) < limit:
        path = todo.pop()
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for quote, name in INCLUDE.findall(text):
            for base in ([path.parent] if quote == '"' else []) + roots:
                found = base / name
                if found.is_file():
                    key = found.resolve()
                    if key not in seen:
                        seen.add(key)
                        out.append(found)
                        todo.append(found)
                    break
    return out


@dataclass
class Job:
    hits: list[Hit]                  # hits covered by this unit transplant (unit order)
    unit: dict
    vram: int                        # target address of the unit start
    size: int                        # bytes compared (unit core size)
    whole: bool = False              # a unit of several functions whose whole masked code stands there


def plan(hits: list[Hit], image: bytes, vram_delta: int,
         starts: set[int] | None = None) -> tuple[list[Job], list[dict]]:
    """Group candidate hits into unit transplants; units only partly present are reported.

    With ``starts`` (target function starts) a unit must begin at a target function start:
    the bytes before a hit can mimic a donor neighbour (a preceding ``jr ra; nop``)."""
    jobs, partial, done = [], [], set()
    for hit in hits:
        if hit.kind == "near" and hit.lost_lo and hit.status == "near" and len(hit.unit["functions"]) == 1:
            # The masking missed a %lo half (its lui is on another path): the masked words
            # differ, the code may not.  The strict comparison of the rebuilt unit decides.
            key = (hit.unit["donor"], hit.unit["source"], hit.unit["line"], hit.vram)
            if key not in done and (starts is None or hit.vram in starts):
                done.add(key)
                hit.status = "candidate"
                jobs.append(Job([hit], hit.unit, hit.vram, hit.unit["core_size"]))
            continue
        if hit.kind != "exact" or hit.status != "candidate":
            continue
        unit, fn = hit.unit, hit.function
        start = hit.vram - fn["offset"]
        key = (unit["donor"], unit["source"], unit["line"], start)
        if key in done:
            continue
        if starts is not None and start not in starts:
            hit.status = "partial-unit"
            hit.notes.append(f"donor unit would start at 0x{start:08X}, which is not a target function start")
            continue
        if len(unit["functions"]) == 1:
            done.add(key)
            jobs.append(Job([hit], unit, start, unit["core_size"]))
            continue
        offset = start - vram_delta
        code = strip_padding(image[offset:offset + unit["core_size"]]) if offset >= 0 else b""
        if len(code) == unit["core_size"] and signature(mask(code, start)[0]) == unit["unit_masked"]:
            done.add(key)
            jobs.append(Job([hit], unit, start, unit["core_size"], whole=True))
            continue
        missing, lost = [], []
        for other in unit["functions"]:
            o = offset + other["offset"]
            ocode = strip_padding(image[o:o + other["size"]]) if o >= 0 else b""
            omasked = mask(ocode, start + other["offset"])[0]
            if len(ocode) != other["size"] or signature(omasked) != other["masked"]:
                missing.append({"donor_function": other["name"], "expected_target_vram": f"0x{start + other['offset']:08X}",
                                "size": other["size"]})
                lost.append(only_lost_lo(mask(bytes.fromhex(other["bytes"]), other["vram"])[0], omasked, ocode)
                            if len(ocode) == other["size"] and other.get("bytes") else None)
        if missing and all(lost) and len(code) == unit["core_size"]:
            # Every function of the unit stands there, and the ones that differ do so only in %lo
            # halves the masking missed (as for a single near function above): the strict
            # comparison of the rebuilt unit decides.  Not ``whole``: the masked code is not equal.
            done.add(key)
            hit.notes.append("unit differs from the donor only in %lo halves the masking missed: "
                             + ", ".join(m["donor_function"] for m in missing))
            jobs.append(Job([hit], unit, start, unit["core_size"]))
            continue
        hit.status = "partial-unit"
        pkey = (hit.vram, unit["donor"], unit["source"])
        if pkey in done or hit.size < 16:      # tiny stubs hit inside many units: count, do not list
            continue
        done.add(pkey)
        partial.append({"target": hit.name, "vram": f"0x{hit.vram:08X}", "donor": unit["donor"],
                        "donor_unit": unit["source"], "functions_in_unit": len(unit["functions"]),
                        "missing_neighbours": missing})
    # attach the other hits that fall inside a whole-unit job
    by_start = {(j.unit["donor"], j.unit["source"], j.vram): j for j in jobs}
    for hit in hits:
        if hit.kind == "exact" and hit.status == "candidate":
            job = by_start.get((hit.unit["donor"], hit.unit["source"], hit.vram - hit.function["offset"]))
            if job and hit not in job.hits:
                job.hits.append(hit)
    return jobs, partial


def unit_definitions(jobs: list[Job], definitions: dict) -> dict:
    """``definitions`` (where each donor function was found, see :func:`transfer`) with the
    functions of every unit found whole.

    A unit of several functions whose whole masked code stands in the target places each of its
    functions, also the ones no catalogued target function matched by itself: a catalogue that
    cuts a function short, or merges two, yields no hit for it, and the callers of that function
    then waited for an address the unit found around it establishes."""
    out = {key: set(places) for key, places in definitions.items()}
    for job in jobs:
        if job.whole:
            for fn in job.unit["functions"]:
                out.setdefault((job.unit["donor"], fn["vram"]), set()).add(job.vram + fn["offset"])
    return out


def unit_label(donor: str, source: str) -> str:
    """``donor:unit`` for notes and pin origins; a unit kept outside the donor's root (generated
    text, an absolute path) is named by its last two components, so no record names a machine."""
    path = Path(str(source))
    return f"{donor}:{'/'.join(path.parts[-2:]) if path.is_absolute() else path.as_posix()}"


def data_defines(elf: Elf, bases: dict[str, int], donor_bases: dict[str, int]) -> list[tuple[str, int, int | None, str]]:
    """``(name, target address, donor address or None, section)`` of every global symbol the
    object defines outside ``.text`` in a section whose target base is in ``bases`` (the bases a
    strict comparison was given: confirmed by the section's contents, or settled by a chain)."""
    out = []
    for symbol in elf.symbols:
        if symbol.bind != 1 or not symbol.name or not 0 < symbol.section < len(elf.sections):
            continue
        section = elf.sections[symbol.section].name
        if section == ".text" or section not in bases:
            continue
        donor_base = donor_bases.get(section)
        out.append((symbol.name, bases[section] + symbol.value,
                    None if donor_base is None else donor_base + symbol.value, section))
    return out


def verified_definitions(rows: list[dict], definitions: dict, verified: dict | None = None,
                         defined_as: dict | None = None) -> int:
    """Add to ``definitions`` what the units verified exact define; returns how many places are new.

    A strictly verified unit is the strongest form of "the donor function of that name is
    identical there": its rebuilt code compared equal with every address it uses established
    independently.  The scan's definitions miss such a function when it was only a near hit (a
    ``%lo`` half the masking lost) or had no hit of its own.  A unit that is only a common shape
    (every function of it stands at several places) defines nothing.

    Such a unit defines its global data as well as its functions: a global symbol in a section
    the strict comparison was given a base for (confirmed by the section's contents, or settled
    by a chain of sections) stands at that base plus its offset, and the units of the same donor
    that refer to it find it under ``(donor, donor address)`` like a function.

    ``verified`` (target address -> what verified units define there: ``kind`` function or data,
    ``name``, ``size`` of a function, ``donor``, ``unit``) is the record a unit of *another*
    donor is settled from (``classify_pin``): ``definitions`` is keyed by donor, and the same
    library function verified from one donor's source was invisible to the units of every other.
    ``defined_as`` keeps a sentence for each data definition (``(donor, donor address, target
    address)``), for the row that uses it."""
    added = 0
    for row in rows:
        unit = row.get("_unit")
        if row.get("verdict") != EXACT_VERDICT or unit is None or row.get("template"):
            continue
        label = unit_label(unit["donor"], unit["source"])
        found = [("function", fn["name"], row["target_vram"] + fn["offset"], fn["vram"], fn["size"], ".text")
                 for fn in unit["functions"]]
        found += [("data", name, taddr, daddr, 0, section) for name, taddr, daddr, section in row.get("_defines") or ()]
        for kind, name, place, daddr, size, section in found:
            if daddr is not None:
                places = definitions.setdefault((unit["donor"], daddr), set())
                if place not in places:
                    places.add(place)
                    added += 1
                if kind == "data" and defined_as is not None:
                    defined_as[(unit["donor"], daddr, place)] = (
                        f"{name} is defined there by the verified unit {label}, whose {section} is at "
                        f"0x{(row.get('_bases') or {}).get(section, place):08X}")
            if verified is not None:
                entry = {"kind": kind, "name": name, "size": size, "donor": unit["donor"], "unit": label,
                         "section": section}
                held = verified.setdefault(place, [])
                if entry not in held:
                    held.append(entry)
                    added += 1
    return added


def same_function_code(dcode: bytes, dvram: int, tcode: bytes, tvram: int) -> bool:
    """Whether donor code and target code of one length are the same function to the scan: equal
    masked words (``mask``), and at the masked sites nothing that marks another function.  An
    unpaired ``lui`` constant must be the same value (as in ``_align``), a fixed address (low
    memory, a hardware register) must be the same address, and a jump or call inside the code
    must land at the same offset."""
    if not dcode or len(dcode) != len(tcode) or len(dcode) % 4:
        return False
    dmasked, dsites = mask(dcode, dvram)
    tmasked, tsites = mask(tcode, tvram)
    if dmasked != tmasked:
        return False
    there = {i: (kind, address) for i, kind, address in tsites}
    for i, kind, daddr in dsites:
        tkind, taddr = there.get(i, (None, None))
        if tkind != kind:
            return False
        if kind == "lui":
            if daddr != taddr and not (0x80000000 <= daddr < 0xC0000000 and 0x80000000 <= taddr < 0xC0000000):
                return False
        elif (fixed_address(daddr) or fixed_address(taddr)) and daddr != taddr:
            return False
        elif dvram <= daddr < dvram + len(dcode) or tvram <= taddr < tvram + len(tcode):
            if daddr - dvram != taddr - tvram:
                return False
    return True


def corroborated_sections(verdict, elf: Elf) -> dict[str, int]:
    """Sections of a unit whose assumed address the unit's own data confirms.

    The address was derived from the code (where its instructions point).  When the section's
    contents, as compiled, are then found at that address, bytes the derivation never looked at
    agree with it.  A section with nothing in it, or next to nothing but zeros, confirms nothing
    and stays an assumption."""
    if not verdict.text_equal:
        return {}
    states = {state.get("section"): state for state in verdict.sections}
    confirmed = {}
    for item in verdict.assumptions:
        state = states.get(item.name) if item.kind == "section" else None
        section = cmp.find_section(elf, item.name) if state else None
        if (state and section is not None and state.get("status") == "equal"
                and state.get("compared") == state.get("size") and (state.get("size") or 0) >= CORROBORATING_BYTES
                and sum(1 for byte in section.data if byte) >= CORROBORATING_BYTES // 2):
            confirmed[item.name] = item.address
    return confirmed


def confirm_sections(compare, elf: Elf, bases: dict[str, int]):
    """(verdict, {section: address}) of a strict comparison whose assumed section addresses the unit's own
    data confirms.  ``compare(bases) -> Verdict`` runs the strict comparison with the given section bases.
    A base already given is never replaced; the rule and its thresholds are ``corroborated_sections``'s.
    This is the one place a resolved C unit, a receipt being issued and a receipt being replayed all
    apply it, so none of them can accept what another would not."""
    verdict = compare(dict(bases))
    confirmed = {name: address for name, address in corroborated_sections(verdict, elf).items()
                 if name not in bases}
    if not confirmed:
        return verdict, {}
    return compare({**bases, **confirmed}), confirmed


def common_shape(hits) -> bool:
    """Whether a unit's place in the target is only a common shape: every one of its functions
    also stands at other addresses (a getter, a setter).  A unit of several functions is pinned
    by the ones that stand nowhere else; one common member does not make its place doubtful."""
    return bool(hits) and all(hit.template for hit in hits)


def fixed_address(address: int) -> bool:
    """Low memory the boot code fills and the hardware registers: the same on every console, so a
    donor's name for one of them cannot belong to a different address in the target."""
    return 0x80000000 <= address < 0x80000400 or 0xA0000000 <= address < 0xC0000000


_CORROBORATION = {"definition": "the donor function of that name is identical there",
                  "fixed-address": "a fixed address, the same in the donor"}


def classify_pin(donor: str, daddr: int | None, taddr: int, table: dict, definitions: dict,
                 own: set[str], *, span: tuple[int, int] | None = None, name: str | None = None,
                 verified: dict | None = None, same_code=None, same_data=None,
                 same_function=None) -> tuple[str, str]:
    """How a unit may use the target address ``taddr`` of one of its relocated symbols.

    ``("declare", origin)``: independent of the unit's own bytes (``definition``: the donor
    function of that name was found identical there, and nowhere else; ``fixed-address``: the
    same fixed address in the donor; ``transfer: …``: aligned instructions of other target
    functions agree on it).  ``("self", "")``: only instructions inside this unit support it, so
    the retail bytes under comparison would establish the address that makes them compare equal;
    never a pin.  ``("unchecked", why)``: nothing transfers it, or the donor function stands at
    several places and only this unit's instruction picks one.  ``("conflict", message)``: the
    transfer table places it elsewhere.

    ``own`` names the unit's functions; ``span`` (``[start, end)`` of the bytes the unit is
    compared over) also makes every supporter or definition inside those bytes the unit itself,
    whatever the catalogue calls it (a catalogue that splits one of the unit's functions in two
    would otherwise let the tail vote for the address its own instruction reads).

    ``verified`` (``verified_definitions``: target address -> what units verified exact define
    there), with the pin's ``name``, settles a pin from the units of *other* donors, where the
    transfer table and ``definitions`` (both keyed by donor) say nothing.  The origin is
    ``definition: …`` when a verified unit that is no common shape defines something starting
    exactly at ``taddr``, outside this unit's bytes, and either it has the pin's name (a name a
    survey made up, ``func_…``/``D_…``, only within one donor), or it is a function and this
    donor's own code at ``daddr`` is the same over that function's extent
    (``same_code(daddr, taddr, size)``, see ``same_function_code``).  It never applies when this
    donor's function is known to stand at another place too: the unit's instruction would pick."""
    def inside(address: int, size: int = 1) -> bool:
        return span is not None and address < span[1] and span[0] < address + max(1, size)

    def by_verified() -> tuple[str, str] | None:
        if not verified or name is None or inside(taddr):
            return None
        if daddr is not None and set(definitions.get((donor, daddr), ())) - {taddr}:
            return None                             # this donor's function stands elsewhere too: a shape
        for entry in verified.get(taddr, ()):
            if entry["name"] == name and (entry["donor"] == donor or not AUTO_NAME.match(name)):
                return "declare", f"definition: the verified unit {entry['unit']} defines {name} there (the same name)"
            if entry["kind"] == "function" and daddr is not None and same_code is not None \
                    and same_code(daddr, taddr, entry["size"]):
                return "declare", (f"definition: the verified unit {entry['unit']} places {entry['name']} there, and "
                                   f"this donor's code for {name} is the same over its {entry['size']} bytes")
        return None

    def by_catalogue() -> tuple[str, str] | None:
        # ``same_function(daddr, taddr)``: the size of the catalogued function that starts at the pin when this
        # donor's code at its address of the symbol is that function, and no other catalogued function is.
        # For a function whose address is taken (an exception vector's target) and that no donor unit holds.
        if same_function is None or daddr is None or name is None or inside(taddr):
            return None
        size = same_function(daddr, taddr)
        if size:
            return "declare", (f"definition: this donor's code for {name} is the {size}-byte function catalogued "
                               f"there, and no other catalogued function")
        return None

    def by_content() -> tuple[str, str] | None:
        # ``same_data(daddr, taddr)``: how many bytes of this donor's datum stand, unchanged, at the pin
        # (0 when too few or too plain to tell).  Bytes the unit's instructions never looked at.
        if same_data is None or daddr is None or name is None or inside(taddr):
            return None
        if daddr == taddr and fixed_address(daddr):
            return "declare", "fixed-address"
        length = same_data(daddr, taddr)
        if length:
            return "declare", f"definition: this donor's {length} bytes of {name} stand at that address"
        return None

    def by_definition() -> tuple[str, str] | None:
        if daddr is None or taddr not in definitions.get((donor, daddr), ()):
            return None
        found = set(definitions[(donor, daddr)])
        if inside(taddr):
            return "self", ""                       # the definition found is this unit's own code
        if found != {taddr}:
            return "unchecked", f"the donor function stands at {len(found)} places; only this unit's code picks one"
        return "declare", "definition"

    item = table.get((donor, daddr)) if daddr is not None else None
    if item is None and by_definition() is not None:
        return by_definition()
    if item is None and daddr is not None and daddr == taddr and fixed_address(daddr):
        return "declare", "fixed-address"
    if item is None:
        return by_verified() or by_content() or by_catalogue() or ("unchecked", "")
    if item["target_address"] != taddr:
        return "conflict", f"transferred 0x{item['target_address']:08X} but this unit needs 0x{taddr:08X}"
    if "functions" not in item:                      # a transfer table written before supporters were kept
        return "declare", f"transfer: {item.get('hits', 0)} function(s), supporters not recorded"
    if span is not None and item.get("supporters") is not None:
        support = sorted({name for name, vram, size in item["supporters"]
                          if name not in own and not inside(vram, size)})
    else:
        support = sorted(set(item["functions"]) - set(own))
    if not support:
        # Every vote for the address is the unit's own instruction.  The votes establish nothing,
        # but the donor function of that name found identical at the address (and nowhere else,
        # and outside the unit's bytes) does: that is evidence the unit's instructions took no
        # part in, exactly as when no vote was recorded at all.
        return by_definition() or by_verified() or by_content() or by_catalogue() or ("self", "")
    return "declare", f"transfer: {len(support)} other function(s), e.g. {support[0]}"


def verify_job(job: Job, index: Index, target, transfer_result: dict, function_starts: set[int] | None,
               workroot: Path, toolchain_root: Path | None, data_symbols: dict | None = None) -> dict:
    from .verify import strict_compare
    unit = job.unit
    donor = index.donors[unit["donor"]]
    root = Path(donor["root"])
    project = scratch_project(root, workroot / "projects" / unit["donor"])
    cfg = Config(project=project, toolchain_root=toolchain_root, include=[Path(p) for p in donor["include"]])
    source = project / unit["source"]
    row = {"target_vram": job.vram, "size": job.size, "donor": unit["donor"], "donor_unit": unit["source"],
           "donor_pipeline": unit["pipeline"], "functions": [h.name for h in job.hits],
           "donor_functions": [f["name"] for f in unit["functions"]], "transferred": {}, "derived_sections": {},
           "conflicts": [], "notes": []}
    spec = _spec_of(unit)
    work = workroot / "objects" / hashlib.sha1(f"{unit['donor']}|{unit['source']}|{unit['flags']}|{job.vram}".encode()).hexdigest()[:12]
    try:
        work.mkdir(parents=True, exist_ok=True)
        elf = Elf.load(build(source, spec, cfg, work))
    except (BuildError, RuntimeError, OSError, ValueError) as error:
        row.update(verdict="build-failed", error=str(error)[-400:])
        return row
    donor_offset = unit["vram"] - donor["delta"]
    # donor addresses of every object symbol/section (from the donor's own retail bytes)
    dimage = _donor_image(donor["image"])
    dcode = dimage[donor_offset:donor_offset + unit["core_size"]]
    dbases, dsyms = cmp.infer_from_rom(elf, dcode, unit["vram"], {}, {})
    toff = target.vram_to_rom(job.vram)
    tcode = target.rom[toff:toff + job.size]
    tbases, tsyms = cmp.infer_from_rom(elf, tcode, job.vram, {}, {})
    row["template"] = common_shape(job.hits)
    table = {} if row["template"] else transfer_result["symbols"]
    definitions = transfer_result.get("definitions") or {}
    verified = transfer_result.get("verified") or {}
    defined_as = transfer_result.get("defined_as") or {}
    declared = symmod.SymbolMap()
    own = {hit.name for hit in job.hits}
    origin = row["pin_origin"] = {}

    def same_code(daddr: int, taddr: int, length: int) -> bool:
        # this donor's own bytes at its address of the symbol against the target's at the pin
        doff, toff_ = daddr - donor["delta"], target.vram_to_rom(taddr)
        if doff < 0 or toff_ < 0:
            return False
        return same_function_code(dimage[doff:doff + length], daddr, target.rom[toff_:toff_ + length], taddr)

    def same_data(daddr: int, taddr: int) -> int:
        # a named datum of the donor found unchanged in the target: the longest of 64, 32 or 16 equal bytes
        # with at least DATA_NONZERO of them non-zero; never inside code the title's catalogue lists
        doff, toff_ = daddr - donor["delta"], target.vram_to_rom(taddr)
        if doff < 0 or toff_ < 0 or (function_starts and taddr in function_starts):
            return 0
        for length in (64, 32, 16):
            there, here = dimage[doff:doff + length], target.rom[toff_:toff_ + length]
            if len(there) == length and there == here and sum(1 for byte in here if byte) >= DATA_NONZERO:
                return length
        return 0

    sizes = transfer_result.get("function_sizes") or {}

    def same_function(daddr: int, taddr: int) -> int:
        size = sizes.get(taddr, 0)
        if size < SAME_FUNCTION_MIN or not same_code(daddr, taddr, size):
            return 0
        others = [start for start, other in sizes.items() if other == size and start != taddr]
        return 0 if any(same_code(daddr, start, size) for start in others) else size

    for name, taddr in tsyms.items():
        decision, why = classify_pin(unit["donor"], dsyms.get(name), taddr, table, definitions, own,
                                     span=(job.vram, job.vram + job.size), name=name, verified=verified,
                                     same_code=same_code, same_data=same_data, same_function=same_function)
        if decision == "declare":
            declared[name] = taddr
            row["transferred"][name] = f"0x{taddr:08X}"
            origin[name] = why
            if why in _CORROBORATION:
                row.setdefault("corroborated", {})[name] = defined_as.get(
                    (unit["donor"], dsyms.get(name), taddr), _CORROBORATION[why])
            elif why.startswith("definition: "):
                row.setdefault("corroborated", {})[name] = why[len("definition: "):]
        elif decision == "conflict":
            row["conflicts"].append(f"{name}: {why}")
        elif (transfer_result.get("stated") or {}).get(name) == taddr:
            # The project's own symbol file gives this name this address: a declaration somebody reviewed
            # (``donors/awaiting_symbols.txt`` asks for exactly that), not the bytes under test speaking
            # for themselves.  Recorded as ``declared``; a receipt replay checks the map still holds it.
            declared[name] = taddr
            row["transferred"][name] = f"0x{taddr:08X}"
            origin[name] = "declared: the project's symbol file gives this name this address"
        else:
            # Target-derived values are kept as diagnostics only.  Supplying them to strict_compare
            # would make the bytes under test their own authority and could prove a wrong symbol.
            row.setdefault("derived_unchecked", {})[name] = f"0x{taddr:08X}"
            if decision == "self":
                row.setdefault("self_derived", {})[name] = f"0x{taddr:08X}"
    # A name only the unit's data refers to (a table of pointers to other units' objects): no instruction of
    # the unit gives its address, so nothing above could settle it, and the data section then cannot be
    # compared at all.  A unit verified exact that defines an object of that name, at one address, does.
    if verified:
        for symbol in elf.symbols:
            if symbol.section != 0 or not symbol.name or symbol.name in declared or symbol.name in tsyms \
                    or AUTO_NAME.match(symbol.name):
                continue
            places = {address: entry for address, entries in verified.items() for entry in entries
                      if entry["name"] == symbol.name}
            if len(places) == 1:
                (address, entry), = places.items()
                if not (job.vram <= address < job.vram + job.size):
                    declared[symbol.name] = address
                    row["transferred"][symbol.name] = f"0x{address:08X}"
                    origin[symbol.name] = (f"definition: the verified unit {entry['unit']} defines {symbol.name} "
                                           f"there (the same name)")
    # ... or, when no verified unit has it under that name (two donors name one object differently): the
    # word standing at the reference in the unit's own data, at the address the unit's code gives that data,
    # when that word is exactly where a verified unit defines an object.  The address comes from the image
    # and the claim it must meet from another unit's verification, not from this unit's instructions.
    if verified:
        for section_name, base in tbases.items():
            section = cmp.find_section(elf, section_name)
            if section is None or section_name == ".text":
                continue
            for offset, rtype, sidx, addend in elf.relocations(section):
                symbol = elf.symbols[sidx]
                if rtype != 2 or symbol.section != 0 or not symbol.name or symbol.name in declared \
                        or symbol.name in tsyms:                 # R_MIPS_32 to a name nothing has placed
                    continue
                at = target.vram_to_rom(base + offset)
                if at < 0 or at + 4 > len(target.rom):
                    continue
                if addend is None:                           # REL: the addend is the word in the object
                    addend = int.from_bytes(section.data[offset:offset + 4], "big") if offset + 4 <= len(section.data) else 0
                address = (int.from_bytes(target.rom[at:at + 4], "big") - addend) & 0xFFFFFFFF
                found = [entry for entry in verified.get(address, ()) if entry["kind"] == "data"]
                if found and not (job.vram <= address < job.vram + job.size):
                    declared[symbol.name] = address
                    row["transferred"][symbol.name] = f"0x{address:08X}"
                    origin[symbol.name] = (f"definition: the unit's {section_name} holds 0x{address:08X} where it "
                                           f"refers to {symbol.name}, and the verified unit {found[0]['unit']} "
                                           f"defines {found[0]['name']} there")
    for name, base in tbases.items():
        # a section of the unit itself: its base follows from the donor base and the aligned sites
        row["derived_sections"][name] = f"0x{base:08X}"
    # A symbol no instruction of the unit refers to (a pointer in its initialized data) has no
    # address read from the unit's code at all.  A verified unit that defines that name gives it
    # one, and the comparison of the section holding the pointer then tests it against the image.
    for symbol in elf.symbols:
        if symbol.bind != 1 or not symbol.name or symbol.section != 0 or symbol.name in tsyms \
                or symbol.name in declared:
            continue
        places = {place: entry for place, entries in verified.items() for entry in entries
                  if entry["name"] == symbol.name and (entry["donor"] == unit["donor"] or not AUTO_NAME.match(symbol.name))
                  and not job.vram <= place < job.vram + job.size}
        if len(places) == 1:
            (place, entry), = places.items()
            declared[symbol.name] = place
            row["transferred"][symbol.name] = f"0x{place:08X}"
            origin[symbol.name] = (f"definition: the verified unit {entry['unit']} defines {symbol.name} there "
                                   f"(the same name); no instruction of this unit reads the address")
            row.setdefault("corroborated", {})[symbol.name] = origin[symbol.name][len("definition: "):]
    if row["conflicts"]:
        row["verdict"] = "needs-review"
        return row
    # Target-inferred section bases are likewise useful evidence but never proof: only a base the
    # section's own contents confirm, or one a chain of sections settled (``settle_sections``, on a
    # later pass), is given to the comparison.  Anything else stays an assumption (fail closed).
    key = (unit["donor"], unit["source"], job.vram)
    chained = {name: item for name, item in ((transfer_result.get("settled") or {}).get(key) or {}).items()
               if tbases.get(name) == item["base"] and cmp.find_section(elf, name) is not None
               and cmp.find_section(elf, name).size == item["size"]}
    given = {name: item["base"] for name, item in chained.items()}
    # A section base the project states for this unit (``n64dk_section_<unit address>_<section>`` in its
    # symbol file, which ``donors/awaiting_symbols.txt`` spells out): a reviewed declaration, like a symbol's.
    # Only a base the unit's own instructions also give is taken, and the comparison still decides.
    stated_bases = {name: base for name, base in tbases.items()
                    if name not in given and cmp.find_section(elf, name) is not None
                    and (transfer_result.get("stated") or {}).get(stated_section_name(job.vram, name)) == base}
    given.update(stated_bases)
    verdict = strict_compare(elf, target.rom, target.vram_to_rom, job.vram, job.size, declared, given,
                             None, function_starts)
    confirmed = {name: base for name, base in corroborated_sections(verdict, elf).items() if name not in given}
    if confirmed:
        verdict = strict_compare(elf, target.rom, target.vram_to_rom, job.vram, job.size, declared,
                                 {**given, **confirmed}, None, function_starts)
        for name, base in confirmed.items():
            row.setdefault("corroborated", {})[name] = f"its {len(cmp.find_section(elf, name).data)} bytes are at 0x{base:08X}"
            origin[f"section:{name}"] = "section-content"      # derived from code, confirmed by its own data bytes
    for name, item in chained.items():
        row.setdefault("corroborated", {})[name] = item["origin"][len(CHAIN_ORIGIN) + 2:]
        origin[f"section:{name}"] = item["origin"]             # derived from code, confirmed by its neighbours
    for name in stated_bases:
        origin[f"section:{name}"] = "declared: the project's symbol file states this unit's section base"
    row["strict"] = verdict.as_dict()
    unverified = [s["section"] for s in verdict.sections if s.get("status") == "unverified"]
    if verdict.verdict == "mismatch" or not verdict.text_equal:
        row["verdict"] = "mismatch"
        row["first_diff"] = None if verdict.first_diff is None else f"0x{verdict.first_diff:X}"
    elif verdict.assumptions and (
            (row["template"] and any(a.kind == "symbol" for a in verdict.assumptions))
            or any(a.kind == "symbol" and dsyms.get(a.name) is not None and fixed_address(dsyms[a.name])
                   and dsyms[a.name] != a.address for a in verdict.assumptions)):
        # The same code stands at several addresses with a different symbol each time (a getter,
        # a setter), or the donor's symbol is a fixed address and the target's is another one:
        # the shape of a function, not this function.  Left to the resolution ladder.
        row["verdict"] = "mismatch"
        row["notes"].append("identical code with other symbols: a common shape, not the donor's function")
    elif verdict.assumptions:
        row["verdict"] = ASSUMED_VERDICT
        row["awaiting"] = {a.name: f"0x{a.address:08X}" for a in verdict.assumptions}
        row["notes"].append("strict oracle still needed retail-derived addresses: "
                            + ", ".join(a.name for a in verdict.assumptions))
    elif unverified:
        row["verdict"] = "needs-review"
        row["notes"].append("strict comparison left owned sections unverified: " + ", ".join(unverified))
    else:
        row["verdict"] = EXACT_VERDICT
    if unverified:
        row["data_unverified"] = unverified
    if row["verdict"] == "mismatch":
        row["donor_self_check"] = _self_check(elf, dimage, donor, unit, dsyms, dbases)
        row["donor_vs_target_first_diff"] = _first_raw_diff(dcode, tcode)
        row["cause"] = _cause(row, elf, dcode, unit["vram"])
    row["_elf_spec"] = (spec, declared, dict(tbases), source)
    row["_object"] = elf
    row["_unit"] = unit
    row["_bases"] = {**given, **confirmed}                     # the bases the comparison was given
    row["_defines"] = data_defines(elf, row["_bases"], dbases)
    return row


def _cause(row: dict, elf: Elf, dcode: bytes, dvram: int) -> str:
    """First-pass classification of a transplant mismatch."""
    if row.get("donor_self_check") != "ok":
        return "donor-recipe-not-reproducible"
    relocated = {r.offset for r in cmp.relocations(elf, ".text")}
    _m, sites = mask(dcode, dvram)
    hard = [4 * i for i, kind, _a in sites if kind in ("jal", "j", "hi", "lo") and 4 * i not in relocated]
    first = row.get("donor_vs_target_first_diff")
    if hard and first is not None and int(first, 16) in hard:
        return "donor-source-hardcodes-address"
    if first is not None and any(int(first, 16) == 4 * i for i, kind, _a in sites if kind == "lui"):
        return "masked-constant-differs"
    if first is None:
        return "data-differs" if row.get("strict", {}).get("text_equal") else "relocation-or-size"
    return "other"


def _first_raw_diff(a: bytes, b: bytes) -> str | None:
    for i in range(0, min(len(a), len(b)) - 3, 4):
        if a[i:i + 4] != b[i:i + 4]:
            return f"0x{i:X}"
    return None


def _self_check(elf: Elf, dimage: bytes, donor: dict, unit: dict, dsyms: dict, dbases: dict) -> str:
    """Does the donor recipe, as indexed, rebuild the donor's own retail bytes?  A ``no``
    means the mismatch is a recipe/lane problem (normalizer, assembler, flags), not the target."""
    from .verify import strict_compare
    syms = symmod.SymbolMap(dsyms)
    result = strict_compare(elf, dimage, lambda v: v - donor["delta"], unit["vram"], unit["core_size"], syms,
                            dict(dbases), None, None)
    return "ok" if result.text_equal else f"fails at 0x{result.first_diff or 0:X}"


_IMAGES: dict[str, bytes] = {}


def _donor_image(path: str) -> bytes:
    if path not in _IMAGES:
        from ..rom import Rom
        try:
            _IMAGES[path] = Rom.load(path).data
        except Exception:                                    # a linear image (see DonorContext)
            _IMAGES[path] = Path(path).read_bytes()
    return _IMAGES[path]


def _manifest_row(job_row: dict, target, out_source: str, include_flags: list[str]) -> str:
    spec, declared, bases, _source = job_row["_elf_spec"]
    out = BuildSpec(spec.pipeline, spec.flags + [f for f in include_flags if f not in spec.flags], {k: v for k, v in bases.items()}, dict(declared), spec.normalizer_args)
    return f"0x{target.vram_to_rom(job_row['target_vram']):X}\t0x{job_row['size']:X}\t{out_source}\t{out.manifest_flags()}"


def write_outputs(out_dir: Path, rows: list[dict], index: Index, target, report: dict, symtext: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposed_symbol_addrs.txt").write_text(symtext)
    waiting = [r for r in rows if r.get("verdict") == ASSUMED_VERDICT]
    if waiting:
        lines = ["# Units whose code is byte-identical and that wait for these addresses.  Each address was",
                 "# read from the target's own instructions; nothing independent confirms the name.  Review,",
                 "# add to the symbol file (sections: to the unit's manifest row), run again.", ""]
        for row in sorted(waiting, key=lambda r: r["target_vram"]):
            lines.append(f"# 0x{row['target_vram']:08X} {row['donor']}:{row['donor_unit']}")
            lines.extend(f"{name} = {address};" if not name.startswith(".") else
                         f"#   section {name} at {address}; to state it: "
                         f"{stated_section_name(row['target_vram'], name)} = {address};"
                         for name, address in sorted((row.get("awaiting") or {}).items()))
        (out_dir / "awaiting_symbols.txt").write_text("\n".join(lines) + "\n")
    manifests: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row.get("verdict") not in (EXACT_VERDICT, ASSUMED_VERDICT) or "_elf_spec" not in row:
            continue
        waits = row.get("verdict") == ASSUMED_VERDICT       # its source is kept too; its manifest rows apart
        donor = index.donors[row["donor"]]
        root = Path(donor["root"])
        src = root / row["donor_unit"]
        unit_path = Path(row["donor_unit"])
        # a unit kept outside the donor's root (generated text) is filed by its last two components
        rel = Path(row["donor"]) / (Path(*unit_path.parts[-2:]) if unit_path.is_absolute() else unit_path)
        dest = out_dir / "src" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        flag_dirs = [root / f[2:] for f in row["_elf_spec"][0].flags if f.startswith("-I") and len(f) > 2]
        roots = flag_dirs + [Path(p) for p in donor["include"]] + [root / "include", root]
        for header in include_closure(src, roots):
            try:
                hrel = header.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            hdest = out_dir / "src" / row["donor"] / hrel
            if not hdest.exists():
                hdest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(header, hdest)
        includes = []
        for path in donor["include"]:
            try:
                includes.append(f"-I{Path(path).resolve().relative_to(root.resolve())}")
            except ValueError:
                includes.append(f"-I{path}")
        manifests[(("awaiting_" if waits else "") + row["donor"], row["donor_pipeline"])].append(
            _manifest_row(row, target, row["donor_unit"], includes))
    for (donor_name, pipe), lines in manifests.items():
        name = donor_name + "_" + pipe.replace("+", "_").replace(".", "")
        (out_dir / f"proposed_{name}.tsv").write_text(
            f"# {pipe} rows proposed by donor transplant from {donor_name}; SOURCE paths are relative to\n"
            f"# src/{donor_name}/ (the source and the headers it includes, in the donor's layout)\n"
            "# ROM_START\tSIZE\tSOURCE\tFLAGS\n" + "\n".join(sorted(lines)) + "\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))


MAX_ALTERNATIVES = 8            # distinct donor units built for one place before giving up
MAX_DEFINITION_ROUNDS = 8       # passes in which units verified exact settle the units calling them
MAX_CHAIN_TRIES = 4             # whole solutions tried while a section chain fails among the selected rows


def _progress(row: dict) -> tuple:
    """How far a row got: exact before waiting before anything else, then fewer addresses waited
    for, then more sections compared equal."""
    rank = {EXACT_VERDICT: 2, ASSUMED_VERDICT: 1}.get(row.get("verdict"), 0)
    equal = sum(1 for state in (row.get("strict") or {}).get("sections", []) if state.get("status") == "equal")
    return rank, -len(row.get("awaiting") or {}), equal
_LIBRARY_DONOR = re.compile(r"^t[0-9a-f]{16}_")


def alternative_order(job) -> tuple:
    """Which donor unit to build first for a place several of them fit.

    Units of decompiled projects and of the SDK come before units of the library of finished
    runs: they carry real names and whole source files, and a library that grows with every
    title must not push them out of the attempts that are made.  Then single-function units,
    then by name, so the order does not depend on how the index was built."""
    unit = job.unit
    return (bool(_LIBRARY_DONOR.match(unit["donor"])), len(unit["functions"]) != 1, unit["donor"], unit["source"])


def places(jobs: list[Job]) -> dict[tuple[int, int], list[Job]]:
    """The jobs grouped by place (unit start and compared size), each group in the order its
    alternatives are tried.

    Units of different sizes that begin at one address are different places, not alternatives of
    each other: an empty function the compiler left at the head of a unit is also, by itself, a
    4-byte unit of some donor, and stopping at that one (exact at once) never tried the unit
    that begins there.  ``select_rows`` keeps the largest exact row of those that overlap."""
    out: dict[tuple[int, int], list[Job]] = defaultdict(list)
    for job in jobs:
        out[(job.vram, job.size)].append(job)
    for alts in out.values():
        alts.sort(key=alternative_order)
    return dict(out)


def distinct_alternatives(alts: list) -> list:
    """``alts`` without repeats of one build: the same source file, compiler and flags (apart from
    macro definitions) whose donor code is identical is the same object, whichever build of the
    library it was indexed from."""
    seen, out = set(), []
    for job in alts:
        unit = job.unit
        flags = tuple(flag for flag in unit.get("flags", []) if not str(flag).startswith("-D"))
        key = (unit.get("unit_masked"), unit.get("pipeline"), Path(str(unit.get("source"))).name, flags)
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out


RODATA = (".rodata", ".rdata")
SETTLE_MIN_PIECES, SETTLE_MIN_BYTES, SETTLE_MIN_NONZERO = 2, 16, 8
CHAIN_ORIGIN = "section-chain"      # pin origin of a section base settled by settle_sections
CHAIN_DATA, CHAIN_BSS = (".data",), (".bss",)


def rodata_chains(pieces: list[dict], rom: bytes, vram_to_rom) -> list[list[dict]]:
    """Chains of rodata pieces that lie end to end in the target.

    A piece is ``{"address", "size", "align", "row"}``: one unit's read-only section at the address
    its own code points to, whose contents the strict comparison found there.  A piece joins the
    chain when it starts exactly where the previous one ends, rounded up to its own alignment,
    and the bytes in between are zero.  Each address was derived from a different unit's code;
    that they abut, which no single derivation implies, is what a chain adds.

    A *partial* piece (``partial``: its section equals the image up to ``size`` bytes and is zero
    padding after, where the image holds something else) ends where the next piece begins: the
    next piece must start inside its verified bytes, after its last non-zero byte (``min_end``)
    and at the next piece's own alignment.  The partial piece then owns exactly the bytes before
    that start (``owned``); everything after belongs to the next unit, whose own code points
    there.  Nothing is forgiven: the owned bytes are equal, and the rest is another unit's."""
    chains: list[list[dict]] = []
    for piece in sorted(pieces, key=lambda p: p["address"]):
        if chains:
            last = chains[-1][-1]
            if last.get("partial"):
                lo, hi = last["address"] + last.get("min_end", 0), last["address"] + last["size"]
                if lo <= piece["address"] <= hi and piece["address"] > last["address"] \
                        and piece["address"] % max(1, piece["align"]) == 0:
                    last["owned"] = piece["address"] - last["address"]
                    chains[-1].append(piece)
                    continue
            else:
                end = last["address"] + last["size"]
                aligned = (end + piece["align"] - 1) // piece["align"] * piece["align"]
                gap = rom[vram_to_rom(end):vram_to_rom(piece["address"])] if piece["address"] >= end else None
                if piece["address"] == aligned and gap is not None and not any(gap):
                    chains[-1].append(piece)
                    continue
        chains.append([piece])
    return chains


def _partial_piece(state: dict, section) -> dict | None:
    """The verified extent of a read-only section that equals the image except for trailing zero
    padding shorter than its alignment (a split unit pads its own section; the original
    translation unit had more data there)."""
    data = getattr(section, "data", None)
    if state.get("status") != "unverified" or not state.get("first_diff") or data is None:
        return None
    size, first = state.get("size") or 0, int(str(state["first_diff"]), 16)
    if not 0 < first < size or len(data) < size or any(data[first:size]) or size - first >= max(1, section.align):
        return None
    nonzero = max((i + 1 for i, byte in enumerate(data[:first]) if byte), default=0)
    return {"size": first, "min_end": nonzero, "partial": True}


def settle_rodata(rows: list[dict], target, function_starts: set[int] | None) -> dict:
    """Settle byte-identical rows that wait only for their read-only section's address.

    Units placed end to end are confirmed together: a chain of at least SETTLE_MIN_PIECES pieces,
    SETTLE_MIN_BYTES bytes and SETTLE_MIN_NONZERO non-zero bytes (a lone 4-byte float confirms
    nothing).  Each awaiting row of an accepted chain is verified again with its section base
    (and, for a partial piece, the size it owns: up to where the next unit's data starts)
    declared, and promoted only when the strict comparison calls it exact."""
    from .verify import strict_compare
    pieces = []
    for row in rows:
        elf = row.get("_object")
        if elf is None or row["verdict"] not in (EXACT_VERDICT, ASSUMED_VERDICT):
            continue
        for state in (row.get("strict") or {}).get("sections", []):
            section = cmp.find_section(elf, state.get("section")) if state.get("section") in RODATA else None
            if section is None or state.get("compared") != state.get("size") or not state.get("size"):
                continue
            piece = {"address": int(state["base"], 16), "size": state["size"], "align": section.align,
                     "name": section.name, "row": row}
            if state.get("status") != "equal":
                partial = _partial_piece(state, section)
                if partial is None:
                    continue
                piece.update(partial)
            pieces.append(piece)
    accepted, report = [], {"pieces": len(pieces), "partial_pieces": sum(1 for p in pieces if p.get("partial")),
                            "chains": 0, "promoted": 0, "promoted_partial": 0, "rows_rechecked": 0}
    for chain in rodata_chains(pieces, target.rom, target.vram_to_rom):
        while chain and chain[-1].get("partial") and "owned" not in chain[-1]:
            chain = chain[:-1]                               # nothing establishes where it ends
        if not chain:
            continue
        start = chain[0]["address"]
        end = chain[-1]["address"] + chain[-1].get("owned", chain[-1]["size"])
        body = target.rom[target.vram_to_rom(start):target.vram_to_rom(end)]
        if len(chain) >= SETTLE_MIN_PIECES and len(body) >= SETTLE_MIN_BYTES and \
                sum(1 for byte in body if byte) >= SETTLE_MIN_NONZERO:
            report["chains"] += 1
            accepted.extend((piece, len(chain), len(body)) for piece in chain)
    for piece, length, size in accepted:
        row = piece["row"]
        if row["verdict"] != ASSUMED_VERDICT or set(row.get("awaiting") or {}) - set(RODATA):
            continue
        spec, declared, _bases, _source = row["_elf_spec"]
        report["rows_rechecked"] += 1
        sizes = {piece["name"]: piece["owned"]} if piece.get("partial") else None
        verdict = strict_compare(row["_object"], target.rom, target.vram_to_rom, row["target_vram"], row["size"],
                                 declared, {piece["name"]: piece["address"]}, sizes, function_starts)
        if verdict.verdict == "exact" and verdict.text_equal and not verdict.assumptions:
            row["verdict"] = EXACT_VERDICT
            row["strict"] = verdict.as_dict()
            row.pop("awaiting", None)
            row.setdefault("corroborated", {})[piece["name"]] = (
                f"one of {length} read-only sections lying end to end over {size} bytes, each at the address "
                f"its own code points to" + (f"; owns {piece['owned']} bytes, up to where the next unit's data "
                                              f"begins" if piece.get("partial") else ""))
            report["promoted"] += 1
            report["promoted_partial"] += bool(piece.get("partial"))
    return report


def selected_rows(rows: list[dict]) -> list[dict]:
    """The rows ``select_rows`` keeps: non-overlapping, every exact row before any waiting one,
    the largest first.  Changes nothing."""
    taken: list[tuple[int, int]] = []
    kept = []
    for wanted in (EXACT_VERDICT, ASSUMED_VERDICT):          # every exact row before any assumed one
        chosen = sorted((r for r in rows if r["verdict"] == wanted), key=lambda r: (-r["size"], r["target_vram"]))
        for row in chosen:
            lo, hi = row["target_vram"], row["target_vram"] + row["size"]
            if not any(a < hi and lo < b for a, b in taken):
                taken.append((lo, hi))
                kept.append(row)
    return kept


def select_rows(rows: list[dict]) -> list[dict]:
    """Keep non-overlapping exact rows (largest first); overlapped exact rows become ``superseded``."""
    kept = {id(row) for row in selected_rows(rows)}
    for row in rows:
        if row["verdict"] in (EXACT_VERDICT, ASSUMED_VERDICT) and id(row) not in kept:
            row["verdict"] = "superseded"
            row["notes"].append("overlaps a larger exact transplant")
    return rows


def section_pieces(units: list[dict]) -> list[dict]:
    """The ``.data`` and ``.bss`` sections of ``units`` as pieces for ``section_chains``.

    A unit is ``{"key", "text": (vram, size), "elf", "bases", "equal"}``: its object, the section
    addresses its own code points to (``bases``: derived from the target's instructions at the
    unit's relocation sites) and the names of the sections whose contents the strict comparison
    found equal at that address (``equal``).  A ``.data`` section is a piece only with equal
    contents; ``.bss`` has none.  The units must not overlap in the code they are compared over
    (rows kept by ``selected_rows``; the manifest rows of a finished run): two of them cannot own
    one address, so where two claim the same one neither claim is a piece."""
    pieces = []
    for unit in units:
        for name in CHAIN_DATA + CHAIN_BSS:
            section, base = unit["elf"].section(name), unit["bases"].get(name)
            if section is None or not section.size or base is None:
                continue
            if name in CHAIN_DATA and name not in unit["equal"]:
                continue
            pieces.append({"address": base, "size": section.size, "align": section.align, "name": name,
                           "kind": "data" if name in CHAIN_DATA else "bss", "key": unit["key"], "text": unit["text"]})
    claims = Counter((piece["kind"], piece["address"]) for piece in pieces)
    return [piece for piece in pieces if claims[(piece["kind"], piece["address"])] == 1]


def section_chains(pieces: list[dict], rom: bytes, vram_to_rom) -> list[list[dict]]:
    """The accepted chains of ``.data`` and of ``.bss`` pieces (``section_pieces``).

    What a chain is, is ``rodata_chains``'s rule: each piece starts exactly where the previous
    one ends, rounded up to its own alignment.  Each address was read from a different unit's
    code; that they abut at the sizes and alignments the units were compiled with, which no
    single derivation implies, is what a chain adds.

    ``.data``: the thresholds of read-only data (at least SETTLE_MIN_PIECES pieces,
    SETTLE_MIN_BYTES bytes and SETTLE_MIN_NONZERO non-zero bytes in the image over the chain),
    every piece's contents equal to the image, the bytes between two pieces zero.

    ``.bss`` holds no bytes in the image, so nothing is compared and the abutting addresses are
    all the evidence there is.  Accepted with at least SETTLE_MIN_PIECES pieces and
    SETTLE_MIN_BYTES bytes, and only where the pieces follow each other in the order of their
    units' code (each unit's code begins at or after the end of the previous one's): a linker
    lays the sections of one kind out in the order of the objects, so sections that abut out of
    that order are a coincidence, not a layout.  (Measured on the title that prompted the rule:
    eleven units' ``.bss`` in three chains of 6, 3 and 2 pieces, none out of order.)

    A chain confirms nothing by itself: ``settle_sections`` gives a waiting unit the base and
    the strict comparison decides."""
    def shifted(piece: dict, origin: int) -> dict:
        return {**piece, "address": piece["address"] - origin, "_piece": piece}

    def unshift(origin: int):
        return lambda address: vram_to_rom(address + origin)

    def better(found: list[list[dict]], chain: list[dict]) -> None:
        # one piece belongs to one chain: the longest found for it under any origin
        members = {id(p) for p in chain}
        rivals = [c for c in found if members & {id(p) for p in c}]
        if all(len(chain) > len(c) for c in rivals):
            found[:] = [c for c in found if c not in rivals] + [chain]

    # The sections of one kind were laid out from the start of their segment, which need not be a multiple of
    # the largest alignment (a library linked in one piece and placed whole: every .data of the title that
    # prompted this is at 8 modulo 16).  The rule is applied with each origin a piece's address allows; under
    # origin 0 it is the rule as written.
    widest = max((max(1, p["align"]) for p in pieces), default=1)
    origins = sorted({p["address"] % widest for p in pieces} | {0})
    accepted: list[list[dict]] = []
    for origin in origins:
        data = sorted((shifted(p, origin) for p in pieces if p["kind"] == "data"), key=lambda p: p["address"])
        for chain in rodata_chains(data, rom, unshift(origin)):
            start, end = chain[0]["address"] + origin, chain[-1]["address"] + chain[-1]["size"] + origin
            body = rom[vram_to_rom(start):vram_to_rom(end)]
            if len(chain) >= SETTLE_MIN_PIECES and len(body) >= SETTLE_MIN_BYTES and \
                    sum(1 for byte in body if byte) >= SETTLE_MIN_NONZERO:
                better(accepted, [p["_piece"] for p in chain])
        chains: list[list[dict]] = []
        for piece in sorted((p for p in pieces if p["kind"] == "bss"), key=lambda p: p["address"]):
            if chains:
                last = chains[-1][-1]
                end = last["address"] + last["size"] - origin
                align = max(1, piece["align"])
                if piece["address"] - origin == (end + align - 1) // align * align and \
                        piece["text"][0] >= last["text"][0] + last["text"][1]:
                    chains[-1].append(piece)
                    continue
            chains.append([piece])
        for chain in chains:
            if len(chain) >= SETTLE_MIN_PIECES and \
                    chain[-1]["address"] + chain[-1]["size"] - chain[0]["address"] >= SETTLE_MIN_BYTES:
                better(accepted, chain)
    return accepted


def _chain_units(rows: list[dict]) -> list[dict]:
    units = []
    for row in rows:
        if row.get("_object") is None or "_elf_spec" not in row or row["verdict"] not in (EXACT_VERDICT, ASSUMED_VERDICT):
            continue
        bases = row["_elf_spec"][2]
        equal = {state["section"] for state in (row.get("strict") or {}).get("sections", [])
                 if state.get("status") == "equal" and state.get("compared") == state.get("size")
                 and state.get("base") is not None and int(state["base"], 16) == bases.get(state["section"])}
        units.append({"key": (row["donor"], row["donor_unit"], row["target_vram"]),
                      "text": (row["target_vram"], row["size"]), "elf": row["_object"], "bases": bases, "equal": equal})
    return units


def chain_origin(piece: dict) -> str:
    if piece["kind"] == "bss":
        return (f"{CHAIN_ORIGIN}: its {piece['name']} abuts the {piece['name']} of other units end to end, each at "
                "the address its own unit's code points to, at the compiled sizes and in the order of the units' "
                "code (it holds no bytes to compare)")
    return (f"{CHAIN_ORIGIN}: its {piece['name']} lies end to end with the {piece['name']} of other units, each at "
            "the address its own unit's code points to and equal to the image there")


def settle_sections(rows: list[dict], target, settled: dict, banned: set | None = None) -> list[tuple]:
    """Record, for the units whose ``.data`` or ``.bss`` stands in an accepted chain, the base the
    chain settles (``settled``: unit key -> section -> ``{"base", "size", "origin"}``); returns the
    keys recorded now.

    The pieces are the sections of the rows ``selected_rows`` keeps, verified or still waiting.
    Nothing is promoted here: a unit is compared again with the base given (``verify_job``), and
    is exact only when that comparison is, with nothing left assumed.  A base given this way has
    the origin ``section-chain``, apart from ``section-content`` (confirmed by the section's own
    bytes), so every unit that rests on a chain can be listed.  ``banned`` names ``(key,
    section)`` pairs not to settle (a chain that did not hold once the rows were selected)."""
    kept = selected_rows(rows)
    # A waiting unit of several functions loses its place to the smaller exact rows of the same code (its
    # functions verified one by one from another donor).  Its sections are still its own, and they are what
    # a chain needs: such a unit contributes pieces too, when every kept row it overlaps is exact and lies
    # inside it.  Settled, it is the larger exact row and takes the place.
    spans = [(row["target_vram"], row["target_vram"] + row["size"], row) for row in kept]
    for row in rows:
        if row["verdict"] != ASSUMED_VERDICT or any(row is other for _a, _b, other in spans):
            continue
        lo, hi = row["target_vram"], row["target_vram"] + row["size"]
        over = [(a, b, other) for a, b, other in spans if a < hi and lo < b]
        if over and all(lo <= a and b <= hi and other["verdict"] == EXACT_VERDICT for a, b, other in over):
            kept = kept + [row]
    pieces = section_pieces(_chain_units(kept))
    recorded = []
    for chain in section_chains(pieces, target.rom, target.vram_to_rom):
        for piece in chain:
            if (piece["key"], piece["name"]) in (banned or ()):
                continue
            held = settled.setdefault(piece["key"], {})
            if held.get(piece["name"], {}).get("base") != piece["address"]:
                held[piece["name"]] = {"base": piece["address"], "size": piece["size"], "origin": chain_origin(piece)}
                recorded.append(piece["key"])
    return recorded


def chain_report(rows: list[dict], target) -> tuple[list[dict], set[tuple]]:
    """The accepted chains over the final rows, for the report, and the ``(key, section)`` pairs
    of rows that are exact on a chain base no accepted chain holds any more."""
    by_key = {(row["donor"], row["donor_unit"], row["target_vram"]): row for row in rows
              if row["verdict"] in (EXACT_VERDICT, ASSUMED_VERDICT)}
    chains = section_chains(section_pieces(_chain_units(list(by_key.values()))), target.rom, target.vram_to_rom)
    held = {(piece["key"], piece["name"]): piece["address"] for chain in chains for piece in chain}
    broken = set()
    for key, row in by_key.items():
        for name, origin in (row.get("pin_origin") or {}).items():
            if name.startswith("section:") and str(origin).startswith(CHAIN_ORIGIN) and row["verdict"] == EXACT_VERDICT:
                section = name[len("section:"):]
                if held.get((key, section)) != (row.get("_bases") or {}).get(section):
                    broken.add((key, section))
    listing = [{"section": chain[0]["name"], "start": f"0x{chain[0]['address']:08X}",
                "bytes": chain[-1]["address"] + chain[-1]["size"] - chain[0]["address"],
                "units": [{"unit": unit_label(piece["key"][0], piece["key"][1]), "target_vram": f"0x{piece['key'][2]:08X}",
                           "address": f"0x{piece['address']:08X}", "size": piece["size"],
                           "verdict": by_key[piece["key"]]["verdict"],
                           "rests_on_it": str((by_key[piece["key"]].get("pin_origin") or {}).get(
                               f"section:{piece['name']}", "")).startswith(CHAIN_ORIGIN)}
                          for piece in chain]} for chain in chains]
    return listing, broken


def functions_inside(vram: int, size: int, functions: list[tuple[str, int, int]], image: bytes,
                     vram_delta: int) -> list[int]:
    """Starts of the catalogued functions that lie wholly in the compared bytes ``[vram, vram + size)``.

    A unit is compared as a whole, so a catalogued function inside it is reproduced whether or
    not it matched a donor function by itself (a catalogue that cuts a function of the unit in
    two gives no hit for either part).  A function may run past the end of the compared bytes
    only over zeros (the unit's alignment padding, which the comparison leaves out); one that
    holds other bytes there is not reproduced by this unit."""
    out = []
    for _name, start, length in functions:
        if not vram <= start < vram + size:
            continue
        end = start + length
        tail = image[vram + size - vram_delta:end - vram_delta] if end > vram + size else b""
        if not any(tail):
            out.append(start)
    return sorted(out)


_STATUS_ORDER = ("candidate", "partial-unit", "needs-review", "local-inconsistent")


def _by_vram(hits: list[Hit]) -> dict[int, list[Hit]]:
    out: dict[int, list[Hit]] = defaultdict(list)
    for hit in hits:
        if hit.kind == "exact":
            out[hit.vram].append(hit)
    return out


def _best_status(hits: list[Hit]) -> str:
    return min((h.status for h in hits), key=lambda s: _STATUS_ORDER.index(s) if s in _STATUS_ORDER else 99)


def stated_section_name(vram: int, section: str) -> str:
    """The symbol-file name under which a project states the base of a donor unit's section."""
    return f"n64dk_section_{vram:08X}_{section.strip('.')}"


def manifest_declared_addresses(paths) -> set[int]:
    """Addresses a project declares in manifest flags (``--symbol=``, ``--section-base=``)."""
    out = set()
    for path in paths:
        for m in re.finditer(r"--(?:symbol|section-base)=[^:\s]+:(0x[0-9A-Fa-f]+)", Path(path).read_text()):
            out.add(int(m.group(1), 16))
    return out


def run(target_image: bytes | None, target, functions: list[tuple[str, int, int]], index: Index,
        skip: list[tuple[int, int]] | None = None, out_dir: Path | None = None, jobs: int = 4,
        toolchain_root: Path | None = None, verify: bool = True, near_max: int = 2,
        target_symbols: symmod.SymbolMap | None = None, truth_addresses: set[int] | None = None,
        other_starts: set[int] | None = None, rodata_settlement: bool = True) -> dict:
    """Scan, transfer, transplant and verify; return the report dict (and write outputs).

    ``other_starts`` are the function starts of the title's other images.  A call out of this
    image into one of them lands on a function start too; without them every unit that calls
    across images was refused.  ``rodata_settlement`` false leaves rows that wait for their
    read-only data's address waiting (decompile-run ablation switch): it can only promote fewer."""
    t0 = time.time()
    skip = sorted(skip or [])
    def skipped(vram, size):
        return any(a < vram + size and vram < b for a, b in skip)
    todo = [f for f in functions if not skipped(f[1], f[2])]
    hits = scan(target.rom, target.delta, todo, index, near_max)
    placed_by_order = corroborate(hits, functions, source_order(index))
    hits, placed_copies = pair_copies(hits, index)
    placed_by_order += placed_copies
    xfer = transfer(hits)
    starts = {v for _n, v, _s in functions}
    jobs_list, partial = plan(hits, target.rom, target.delta, starts)
    xfer["definitions"] = unit_definitions(jobs_list, xfer.get("definitions") or {})
    xfer["function_sizes"] = {vram: size for _name, vram, size in functions}
    xfer["stated"] = dict(target_symbols or {})
    t_scan = time.time() - t0
    rows: list[dict] = []
    settled = None
    section_chain_list: list[dict] = []
    if verify and jobs_list:
        # one job per place: try donors in order until one is exact
        by_addr = places(jobs_list)
        with tempfile.TemporaryDirectory(prefix="n64dk-transplant-") as tmp:
            def attempt(alts):
                tried = []
                for job in distinct_alternatives(alts)[:MAX_ALTERNATIVES]:
                    row = verify_job(job, index, target, xfer, starts | (other_starts or set()), Path(tmp),
                                     toolchain_root)
                    tried.append(row)
                    if row["verdict"] == EXACT_VERDICT:
                        break
                best = next((r for r in tried if r["verdict"] == EXACT_VERDICT),
                            next((r for r in tried if r["verdict"] == ASSUMED_VERDICT), tried[0]))
                best["alternatives_tried"] = len(tried)
                if best["verdict"] != EXACT_VERDICT:
                    best["other_attempts"] = [{"donor": r["donor"], "unit": r["donor_unit"], "verdict": r["verdict"]}
                                              for r in tried[1:]]
                return best
            base_definitions = {key: set(places_) for key, places_ in xfer["definitions"].items()}
            groups = list(by_addr.values())

            def solve(pool, banned):
                """Every place verified, then passes in which what is settled settles more: the
                units verified exact place their functions and data for the units still waiting
                (``verified_definitions``), and sections lying end to end settle each other's
                address (``settle_sections``).  The waiting places are compared again after each
                pass, for as long as one of them gets further."""
                xfer.update(definitions={key: set(places_) for key, places_ in base_definitions.items()},
                            verified={}, defined_as={}, settled={})
                found = list(pool.map(attempt, groups))
                for _round in range(MAX_DEFINITION_ROUNDS):
                    news = verified_definitions(found, xfer["definitions"], xfer["verified"], xfer["defined_as"])
                    news += len(settle_sections(found, target, xfer["settled"], banned)) if rodata_settlement else 0
                    waiting = [k for k, row in enumerate(found) if row["verdict"] != EXACT_VERDICT]
                    if not news or not waiting:
                        break
                    again = list(pool.map(attempt, [groups[k] for k in waiting]))
                    better = [(k, row) for k, row in zip(waiting, again) if _progress(row) > _progress(found[k])]
                    for k, row in better:
                        if row["verdict"] == EXACT_VERDICT:
                            row["notes"].append("settled on a later pass, by what units verified before it define "
                                                "or by sections lying end to end")
                        found[k] = row
                    if not better:
                        break
                return found

            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                banned: set[tuple] = set()
                for _try in range(MAX_CHAIN_TRIES):
                    rows = select_rows(solve(pool, banned))
                    section_chain_list, broken = chain_report(rows, target)
                    if not broken:
                        break
                    banned |= broken                         # the chain did not survive the selection: without it
            for key, section in broken:                      # (never seen; fail closed if it ever is)
                for row in rows:
                    if (row["donor"], row["donor_unit"], row["target_vram"]) == key and row["verdict"] == EXACT_VERDICT:
                        row["verdict"] = ASSUMED_VERDICT
                        row["awaiting"] = {section: f"0x{(row.get('_bases') or {}).get(section, 0):08X}"}
                        row["notes"].append(f"its {section} chain does not hold among the selected rows")
            settled = (settle_rodata(rows, target, starts | (other_starts or set())) if rodata_settlement
                       else {"switched_off": True, "promoted": 0})
        for row in rows:
            for private in ("_object", "_unit", "_bases", "_defines"):
                row.pop(private, None)
    t_verify = time.time() - t0 - t_scan
    # symbol transfer precision against the target catalogue (calls) and symbol file (data)
    precision = {"calls": 0, "calls_at_function_start": 0, "data": 0, "data_at_known_symbol": 0,
                 "confirmed_by_exact_transplant": 0}
    confirmed = {int(a, 16) for r in rows if r["verdict"] == EXACT_VERDICT for a in r["transferred"].values()}
    known = set((target_symbols or {}).values()) | set(truth_addresses or ())
    for item in xfer["symbols"].values():
        precision["confirmed_by_exact_transplant"] += item["target_address"] in confirmed
        if item["call"]:
            precision["calls"] += 1
            precision["calls_at_function_start"] += item["target_address"] in starts
        else:
            precision["data"] += 1
            precision["data_at_known_symbol"] += item["target_address"] in known
    verdicts = Counter(r["verdict"] for r in rows)
    exact_rows = [r for r in rows if r["verdict"] == EXACT_VERDICT]
    sizes = {v: size for _n, v, size in functions}
    new_funcs = set()
    for r in exact_rows:
        lo, hi = r["target_vram"], r["target_vram"] + r["size"]
        inside = [v for v in starts if lo <= v < hi]
        matched = [v for v in inside if skipped(v, sizes[v])]
        if matched:
            r["overlaps_matched"] = [f"0x{v:08X}" for v in sorted(matched)]
            r["notes"].append(f"unit covers {len(matched)} function(s) the target already matched")
        new_funcs.update(v for v in inside if v not in matched)
    report = {
        "functions_considered": len(todo), "functions_skipped": len(functions) - len(todo),
        "exact_hit_functions": len({h.vram for h in hits if h.kind == "exact"}),
        "exact_hit_bytes": sum({h.vram: h.size for h in hits if h.kind == "exact"}.values()),
        "near_hit_functions": len({h.vram for h in hits if h.kind == "near"} - {h.vram for h in hits if h.kind == "exact"}),
        "hit_status_by_function": dict(Counter(_best_status(hs) for hs in _by_vram(hits).values())),
        "unit_transplants": len(rows), "verdicts": dict(verdicts),
        "functions_awaiting_symbols": len({v for r in rows if r["verdict"] == ASSUMED_VERDICT for v in starts
                                           if r["target_vram"] <= v < r["target_vram"] + r["size"]}),
        "functions_recovered_new": len(new_funcs),
        "bytes_recovered_new": sum(sizes[v] for v in new_funcs),
        "rows_overlapping_matched": sum(1 for r in exact_rows if r.get("overlaps_matched")),
        "functions_recovered": len({v for r in exact_rows for v in starts if r["target_vram"] <= v < r["target_vram"] + r["size"]}),
        "mismatch_causes": dict(Counter(r.get("cause", "?") for r in rows if r["verdict"] == "mismatch")),
        "template_units_exact": sum(1 for r in exact_rows if r.get("template")),
        "placed_by_source_order": placed_by_order,
        "near_units_verified": sum(1 for h in hits if h.kind == "near" and h.status == "candidate"),
        "bytes_recovered": sum(r["size"] for r in exact_rows),
        "data_unverified_units": sum(1 for r in exact_rows if r.get("data_unverified")),
        "rodata_settlement": settled,
        "section_chains": section_chain_list,
        "symbol_transfer": {"symbols": len(xfer["symbols"]), "contradictions": len(xfer["contradictions"]),
                            "suspicious": len(xfer["suspicious"]), "precision": precision},
        "timing_s": {"scan": round(t_scan, 2), "verify": round(t_verify, 2)},
        "rows": [{k: (f"0x{v:08X}" if k == "target_vram" else v) for k, v in r.items() if not k.startswith("_")}
                 for r in sorted(rows, key=lambda r: r["target_vram"])],
        "partial_units": partial,
        "near_hits": [h.row() for h in hits if h.kind == "near"][:500],
        "needs_review": [h.row() for h in hits if h.status in ("needs-review", "local-inconsistent", "constant-differs")],
        "contradictions": xfer["contradictions"], "suspicious": xfer["suspicious"],
    }
    symtext = symbol_file(xfer)
    report["proposed_symbol_count"] = symtext.count("\n") - 2
    if out_dir is not None:
        write_outputs(Path(out_dir), rows, index, target, report, symtext)
        # every hit with what became of it: the report lists rows and refusals, and a hit that is
        # neither (a candidate whose unit was not planned) could not be found anywhere
        (Path(out_dir) / "hits.json").write_text(json.dumps(
            [{**h.row(), "template": h.template, "placed": h.placed, "unit_functions": len(h.unit["functions"]),
              "unit_start": f"0x{h.vram - h.function['offset']:08X}"} for h in hits], indent=0))
    report["_symbol_file"] = symtext
    return report
