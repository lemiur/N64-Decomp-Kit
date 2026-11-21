"""Identify RSP microcode and boot code instead of leaving them unresolved (wave 4, workstream 8).

Report only: nothing here changes a unit class, a match, a coverage denominator or a credit.  The
output says *what* an RSP or boot range is (or why that could not be said), so an unresolved-case
explanation can read "gspF3DEX2.fifo 2.08 text, named by its version string" instead of "rsp".

Blocks
------
``text``  RSP code.  The kit's own detector (``sdk.fingerprint.rsp_text_ranges``: 64-byte blocks of
          RSP-legal opcodes with a high share of vector ops) gives ranges; ranges closer than
          ``CLUSTER_GAP`` are merged into clusters, and each cluster is cut at *referenced
          boundaries*: addresses the CPU program takes (a ``lui``/``addiu`` or ``lui``/``ori`` pair,
          or a data word) that fall on or inside the cluster.  ``rspbootTextStart``,
          ``gspF3DEX2_fifoTextStart`` and their ``...TextEnd`` are such addresses: the program puts
          them in an OSTask.  An end with no reference keeps the detector's 64-byte estimate.
``data``  RSP DMEM data, found from its embedded version string (``sdk.fingerprint.find_microcode``).
          Start: the nearest referenced 16-byte-aligned address at or before the string (else the
          kit's zero-padding walk-back, else the string itself); end: the next referenced address
          after the string (else the end of the string, a lower bound).

A block whose two ends are referenced addresses has an ``exact`` extent; only those are compared
by size, and only their digests are comparable across titles.  References are taken only from the
main image (``Mapping``) and never from inside an RSP cluster or a microcode data window (RSP
``lb`` words with base ``$zero`` look like KSEG0 pointers).

Identification (in this order; the first that applies wins)
-----------------------------------------------------------
data  1. the embedded version string against ``KNOWN`` (regexes only)          -> ``named``
      2. a version string no row matches (text and the kit's parse reported)   -> ``named-unlisted``
text  1. the target of a static OSTask template's ``ucode_boot`` field          -> ``role`` (rspboot)
      2. paired with a named data block: an OSTask template's (ucode,
         ucode_data), a (text, data) pointer pair in a table, or the nearest
         references within ``PAIR_DISTANCE`` bytes                              -> ``paired``
      3. exact extent whose size equals a row's text size                       -> ``candidate-by-size``
      4. otherwise                                                              -> ``unidentified`` + reason

Strings first, size second: a size equality never overrides a string, and a size match is only a
candidate (several SDK builds of one microcode differ in size).  ``KNOWN`` holds names, families,
version-string regexes and sizes taken from public sources; it holds no microcode bytes.

Digests
-------
``sha256``      of the block's bytes.
``rel_digest``  relocation- and padding-insensitive: trailing zero words dropped; text words go
                through ``lineage.masked_words`` (``j``/``jal`` targets, ``lui`` immediates and their
                ``%lo`` users blanked: RSP jump targets are IMEM addresses that move with an overlay's
                load origin); data has trailing zero bytes dropped only.

Boot
----
``identify_boot``: IPL3 (ROM 0x40..0x1000) by CRC32 against ``rom.CIC_BY_IPL3_CRC`` (the kit's own
table), with its SHA-256; the entry routine at ROM 0x1000 up to its first ``jr``/``jalr`` and delay
slot, with exact and masked digests.  No entry-routine table: a census groups titles by digest.

``python -m n64dk.analysis.rspid rom <image>...`` or ``... census <project dir>...`` prints one JSON
line per title and a cross-title summary line; ``--out`` writes the full report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import tomllib
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .. import rom as rommod
from ..sdk import fingerprint
from .lineage import masked_words

MASK = 0xFFFFFFFF
KSEG0_LO, KSEG0_HI = 0x80000000, 0x80800000
CLUSTER_GAP = 0x400      # detector ranges closer than this are one cluster (scalar stretches fail the test)
EDGE = 0x40              # a reference this close to a cluster edge is that edge (64-byte detector granularity)
DATA_WINDOW = 0x800      # SP_UCODE_DATA_SIZE: a DMEM image is at most this long
PAIR_DISTANCE = 0x40     # references to a text start and a data start this close are one task's pair
LUI_WINDOW = 32          # instructions a lui value stays live for the %lo pairing
MIN_TEXT = 0x20          # shorter pieces between references are not reported as blocks
MAIN_SPAN = 0x100000     # without a survey layout: the 1 MiB the IPL3 copies from ROM 0x1000
TASK_TYPES = {1: "M_GFXTASK", 2: "M_AUDTASK", 3: "M_VIDTASK", 4: "M_NJPEGTASK"}   # as analysis.runtime
_TASK_POINTERS = (0x08, 0x10, 0x18, 0x20, 0x28, 0x2C, 0x30, 0x38)
# The CIC chips that share one IPL3 (NTSC name first), as media.py documents them.
CIC_CHIPS = {"6101": "CIC-NUS-6101", "7102": "CIC-NUS-7102", "6102": "CIC-NUS-6102/7101",
             "6103": "CIC-NUS-6103/7103", "6105": "CIC-NUS-6105/7105", "6106": "CIC-NUS-6106/7106"}

# --------------------------------------------------------------------------- the identity table


@dataclass(frozen=True)
class Identity:
    name: str               # libultra symbol stem; "{stem}" is filled from the string
    family: str
    task: str               # gfx | audio | boot
    pattern: str | None     # regex over the embedded version string (None: no string is known)
    text_size: int | None   # bytes, when a public layout states it
    data_size: int | None
    source: str             # the public fact the row rests on


GFX_STEMS = "F3DZEX|F3DEX|F3DLX|F3DLP|L3DEX|S2DEX"
_GFX_HEAD = rf"^RSP Gfx ucode\s+(?P<stem>{GFX_STEMS})(?:\.(?P<variant>NoN|Rej))?"
_GFX = _GFX_HEAD + r"\s+(?:(?P<transport>fifo|xbus|dram)\s+)?"
_STRING = ("the version string graphics microcode embeds in its DMEM data, 'RSP Gfx ucode <name>[.NoN|.Rej] "
           "[fifo|xbus|dram] <version> ...', as emulator microcode tables and sdk.fingerprint read it")
_SM64 = ("public SM64 decompilation rsp/fast3d.s: the DMEM data embeds 'RSP SW Version: 2.0D, 04-01-96' "
         "or '2.0H, 02-12-97'")
_DRM = "public Dr. Mario 64 decompilation config/us/drmario64.us.yaml (textbin/databin segment bounds)"

KNOWN: tuple[Identity, ...] = (
    Identity("gspFast3D", "F3D", "gfx", r"^RSP SW Version: (?P<version>\d\.\d[A-Z])\b", None, None, _SM64),
    Identity("gsp{stem}2", "{stem}2", "gfx", _GFX + r"(?P<version>2\.\d\d[A-Za-z0-9]*)", None, None,
             _STRING + "; a 2.xx version is the F3DEX2 generation"),
    Identity("gsp{stem}", "{stem}", "gfx", _GFX + r"(?P<version>[01]\.\d\d[A-Za-z0-9]*)", None, None, _STRING),
    Identity("gsp{stem}", "{stem}", "gfx", _GFX_HEAD + r"(?![A-Za-z0-9])", None, None,
             _STRING + "; this string carries no version number"),
    Identity("rspboot", "rspboot", "boot", None, 0xD0, None, _DRM + ": rsp/rspboot text 0x6C4B0..0x6C580"),
    Identity("aspMain", "aspMain", "audio", None, 0xE20, 0x2C0,
             _DRM + ": rsp/aspMain text 0x6C580..0x6D3A0, data 0x93D60..0x94020"),
    Identity("gspF3DEX2.fifo", "F3DEX2", "gfx", None, 0x1390, 0x420,
             _DRM + ": rsp/f3dex2 text 0x6D3A0..0x6E730, data 0x94020..0x94440 (version not in the layout)"),
    Identity("gspS2DEX.fifo", "S2DEX", "gfx", None, 0x17F0, 0x3C0,
             _DRM + ": rsp/s2dex text 0x6E730..0x6FF20, data 0x94440..0x94800 (ends bounded by the next segment)"),
)
_VARIANTS = {"non": "NoN", "rej": "Rej"}


def match_string(text: str) -> dict | None:
    """The identity an embedded version string names, or None when no ``KNOWN`` row matches."""
    for index, row in enumerate(KNOWN):
        if row.pattern is None:
            continue
        m = re.search(row.pattern, text, re.IGNORECASE)
        if not m:
            continue
        groups = {k: v for k, v in m.groupdict().items() if v}
        stem = groups.get("stem", "").upper()
        variant = _VARIANTS.get(groups.get("variant", "").lower())
        transport = groups.get("transport", "").lower() or None
        name = row.name.format(stem=stem) + (f".{variant}" if variant else "") + (f".{transport}" if transport else "")
        return {"name": name, "family": row.family.format(stem=stem), "task": row.task,
                "version": groups.get("version"), "variant": variant, "transport": transport,
                "row": index, "basis": "version-string", "source": row.source}
    return None


def size_candidates(kind: str, size: int) -> list[str]:
    """Rows whose stated text (or data) size equals ``size``."""
    return [row.name for row in KNOWN if (row.text_size if kind == "text" else row.data_size) == size]


# --------------------------------------------------------------------------- digests

def _strip_words(code: bytes) -> bytes:
    end = len(code) - len(code) % 4
    while end >= 4 and code[end - 4:end] == b"\0\0\0\0":
        end -= 4
    return code[:end]


def rel_digest(block: bytes, kind: str) -> str:
    """Relocation- and padding-insensitive SHA-256 (see the module docstring)."""
    if kind == "data":
        return hashlib.sha256(bytes(block).rstrip(b"\0")).hexdigest()
    words = masked_words(_strip_words(bytes(block)))
    return hashlib.sha256(struct.pack(f">{len(words)}I", *words)).hexdigest()


# --------------------------------------------------------------------------- references


@dataclass(frozen=True)
class Mapping:
    """One linearly loaded image: ROM ``rom..rom+size`` is VRAM ``vram..vram+size``."""
    rom: int
    vram: int
    size: int

    def to_rom(self, vram: int) -> int | None:
        off = (vram - self.vram) & MASK
        return self.rom + off if off < self.size else None

    def to_vram(self, rom: int) -> int | None:
        return self.vram + rom - self.rom if self.rom <= rom < self.rom + self.size else None


def address_constants(data: bytes, start: int, end: int, mapping: Mapping,
                      exclude=()) -> dict[int, list[tuple[int, str]]]:
    """ROM address -> [(site, "word" | "hi/lo")]: every address into ``mapping`` that the bytes in
    ``start..end`` hold as a data word or build with a ``lui`` + ``addiu``/``ori`` pair.  Sites inside
    an ``exclude`` range are skipped."""
    out: dict[int, list[tuple[int, str]]] = defaultdict(list)
    start = max(0, start - start % 4)
    end = min(end, len(data))
    end -= (end - start) % 4
    if end <= start:
        return {}
    base, rom0, size = mapping.vram, mapping.rom, mapping.size

    def add(address: int, site: int, kind: str) -> None:
        off = (address - base) & MASK
        if off < size and not any(lo <= site < hi for lo, hi in exclude):
            out[rom0 + off].append((site, kind))

    lui: dict[int, tuple[int, int]] = {}
    for i, (w,) in enumerate(struct.iter_unpack(">I", data[start:end])):
        site = start + 4 * i
        if (w - base) & MASK < size:
            add(w, site, "word")
        op = w >> 26
        if op == 0x0F:
            lui[(w >> 16) & 31] = (w & 0xFFFF, i)
        elif op in (0x09, 0x0D):
            rs, rt = (w >> 21) & 31, (w >> 16) & 31
            hit = lui.get(rs)
            if hit is not None and i - hit[1] <= LUI_WINDOW:
                lo = w & 0xFFFF
                if op == 0x09:
                    address = ((hit[0] << 16) + (lo - 0x10000 if lo & 0x8000 else lo)) & MASK
                else:
                    address = (hit[0] << 16) | lo
                add(address, site, "hi/lo")
                if rt == rs:
                    lui.pop(rs, None)
    return dict(out)


def ostask_headers(data: bytes, start: int, end: int, mapping: Mapping, exclude=()) -> list[dict]:
    """Static OSTask templates (``sptask.h`` layout, 8-byte aligned) in ``start..end``: a task type,
    small flags, a boot microcode with a size, a microcode pointer, and every pointer field null,
    KSEG0 or inside ``mapping``.  Tasks built at run time are not seen here."""
    out = []
    start = max(0, start + (-start) % 8)
    last = min(end, len(data)) - 0x40

    def pointer(value: int) -> bool:
        return value == 0 or KSEG0_LO <= value < KSEG0_HI or mapping.to_rom(value) is not None

    for off in range(start, last + 1, 8):
        if struct.unpack_from(">I", data, off)[0] not in TASK_TYPES:
            continue
        f = struct.unpack_from(">16I", data, off)
        if f[1] > 0xFF or not f[2] or not f[4] or f[2] == f[4]:
            continue
        if not (0 < f[3] <= 0x1000 and f[3] % 8 == 0 and f[5] <= 0x1000 and f[7] <= 0x800
                and f[9] <= 0x10000 and f[15] <= 0x10000):
            continue
        if not all(pointer(f[p // 4]) for p in _TASK_POINTERS) or any(lo <= off < hi for lo, hi in exclude):
            continue
        out.append({"rom": off, "type": TASK_TYPES[f[0]], "flags": f[1],
                    "ucode_boot": f"0x{f[2]:08X}", "ucode_boot_size": f[3], "ucode": f"0x{f[4]:08X}",
                    "ucode_size": f[5], "ucode_data": f"0x{f[6]:08X}", "ucode_data_size": f[7],
                    "boot_rom": mapping.to_rom(f[2]), "ucode_rom": mapping.to_rom(f[4]),
                    "ucode_data_rom": mapping.to_rom(f[6]) if f[6] else None})
    return out


# --------------------------------------------------------------------------- blocks

def clusters(data: bytes) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in fingerprint.rsp_text_ranges(data):
        if out and a - out[-1][1] <= CLUSTER_GAP:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def text_blocks(found: list[tuple[int, int]], constants: dict) -> list[dict]:
    blocks = []
    for a, b in found:
        refs = [r for r in constants if a - EDGE <= r <= b + EDGE and r % 8 == 0]
        points = {r: "address" for r in refs}
        if not any(abs(r - a) <= EDGE for r in refs):
            points[a] = "rsp-blocks"
        if not any(abs(r - b) <= EDGE for r in refs):
            points[b] = "rsp-blocks"
        ordered = sorted(points)
        for lo, hi in zip(ordered, ordered[1:]):
            if hi - lo >= MIN_TEXT:
                blocks.append({"kind": "text", "rom": lo, "end": hi, "start_basis": points[lo],
                               "end_basis": points[hi], "cluster": [a, b]})
    return blocks


def data_blocks(data: bytes, strings: list, constants: dict) -> list[dict]:
    aligned = sorted(r for r in constants if r % 16 == 0)
    blocks: dict[tuple[int, int], dict] = {}
    for u in strings:
        s = u.rom
        nul = data.find(b"\0", s, s + 0x100)
        string_end = nul if nul >= 0 else s + len(u.text)
        before = [r for r in aligned if s - DATA_WINDOW <= r <= s]
        after = [r for r in aligned if string_end < r <= s + DATA_WINDOW]
        if before:
            start, start_basis = before[-1], "address"
        elif u.data_rom is not None:
            start, start_basis = u.data_rom, "zero-padding"
        else:
            start, start_basis = s - s % 16, "string"
        if after:
            end, end_basis = after[0], "address"
        else:
            end, end_basis = (string_end + 16) & ~15, "string-end"
        block = blocks.get((start, end))
        if block is None:
            blocks[(start, end)] = {"kind": "data", "rom": start, "end": end, "start_basis": start_basis,
                                    "end_basis": end_basis, "strings": [u.text], "string_rom": s,
                                    "kit_parse": [u.family, u.version, u.variant]}
        else:
            block["strings"].append(u.text)
    return list(blocks.values())


def _pairs(texts: list[dict], datas: list[dict], constants: dict, tasks: list[dict]) -> dict[int, tuple[int, str]]:
    """text start -> (data start, basis), for texts and datas whose start is a referenced address."""
    data_starts = {d["rom"] for d in datas if d["start_basis"] == "address"}
    text_starts = {t["rom"] for t in texts if t["start_basis"] == "address"}
    boot_sites = {t["rom"] + 0x08 for t in tasks}
    pairs: dict[int, tuple[int, str]] = {}
    for t in tasks:
        if t["ucode_rom"] in text_starts and t["ucode_data_rom"] in data_starts:
            pairs.setdefault(t["ucode_rom"], (t["ucode_data_rom"], f"OSTask template at ROM 0x{t['rom']:X}"))
    for text in sorted(text_starts - set(pairs)):
        tsites = [(s, k) for s, k in constants.get(text, ()) if s not in boot_sites]
        best = []
        for start in data_starts:
            near = min(((abs(a - b), ka, kb) for a, ka in tsites for b, kb in constants.get(start, ())), default=None)
            if near is not None and near[0] <= PAIR_DISTANCE:
                best.append((near, start))
        best.sort()
        if best and (len(best) == 1 or best[0][0][0] < best[1][0][0]):
            (distance, ka, kb), start = best[0]
            basis = ("adjacent pointer words (text, data)" if distance == 4 and ka == kb == "word"
                     else f"references 0x{distance:X} bytes apart ({ka}, {kb})")
            pairs[text] = (start, basis)
    return pairs


def _finish(data: bytes, block: dict, mapping: Mapping | None) -> None:
    raw = data[block["rom"]:block["end"]]
    block["size"] = len(raw)
    block["extent"] = "exact" if block["start_basis"] == block["end_basis"] == "address" else "estimated"
    block["sha256"] = hashlib.sha256(raw).hexdigest()
    block["rel_digest"] = rel_digest(raw, block["kind"])
    vram = mapping.to_vram(block["rom"]) if mapping else None
    block["vram"] = None if vram is None else f"0x{vram:08X}"
    block["size_candidates"] = size_candidates(block["kind"], len(raw)) if block["extent"] == "exact" else []


def identify(data: bytes, *, mapping: Mapping | None = None, span: tuple[int, int] | None = None) -> dict:
    """RSP blocks of an image (whole ROM, big-endian), each with its extent, digests and identity."""
    data = bytes(data)
    strings = fingerprint.find_microcode(data)
    found = clusters(data)
    constants: dict = {}
    tasks: list[dict] = []
    if mapping is not None:
        lo, hi = span or (mapping.rom, mapping.rom + mapping.size)
        exclude = list(found) + [(u.rom - DATA_WINDOW, u.rom + DATA_WINDOW) for u in strings]
        constants = address_constants(data, lo, hi, mapping, exclude)
        tasks = ostask_headers(data, lo, hi, mapping, found)
        for t in tasks:
            if t["boot_rom"] is not None:
                constants.setdefault(t["boot_rom"] + t["ucode_boot_size"], []).append((t["rom"] + 0x0C, "ostask-size"))
    texts = text_blocks(found, constants)
    datas = data_blocks(data, strings, constants)
    for d in datas:
        ident = match_string(d["strings"][0])
        if ident is not None:
            d["status"], d["identity"] = "named", ident
        else:
            family, version, variant = d["kit_parse"]
            d["status"] = "named-unlisted"
            d["identity"] = {"name": None, "family": family, "version": version, "variant": variant,
                             "basis": "version-string not in KNOWN (sdk.fingerprint's parse)"}
        _finish(data, d, mapping)
    by_start = {d["rom"]: d for d in datas}
    pairs = _pairs(texts, datas, constants, tasks)
    boots = {t["boot_rom"]: t for t in tasks if t["boot_rom"] is not None}
    for t in texts:
        _finish(data, t, mapping)
        if t["rom"] in boots and t["start_basis"] == "address":
            task = boots[t["rom"]]
            t["status"] = "role"
            t["identity"] = {"name": "rspboot", "family": "rspboot", "task": "boot", "version": None,
                             "basis": f"ucode_boot of the OSTask template at ROM 0x{task['rom']:X}"
                                      f" (ucode_boot_size 0x{task['ucode_boot_size']:X})"}
        elif t["rom"] in pairs and by_start[pairs[t["rom"]][0]].get("identity"):
            start, basis = pairs[t["rom"]]
            t["status"] = "paired"
            t["identity"] = dict(by_start[start]["identity"],
                                 basis=f"paired with the data block at ROM 0x{start:X}: {basis}")
            t["data_rom"] = start
        elif t["size_candidates"]:
            t["status"] = "candidate-by-size"
            t["identity"] = {"name": " | ".join(t["size_candidates"]), "family": None, "version": None,
                             "basis": f"exact extent 0x{t['size']:X} bytes equals the KNOWN text size"}
        else:
            t["status"] = "unidentified"
            t["identity"] = None
            t["reason"] = ("no version string pairs with it; " +
                           (f"its exact size 0x{t['size']:X} matches no KNOWN row" if t["extent"] == "exact"
                            else "its extent is the detector's 64-byte estimate (no referenced boundary)"))
    blocks = sorted(texts + datas, key=lambda b: (b["rom"], b["kind"]))
    return {"blocks": blocks, "tasks": tasks, "clusters": [list(c) for c in found],
            "strings": [{"rom": u.rom, "text": u.text} for u in strings],
            "mapping": None if mapping is None else {"rom": mapping.rom, "vram": f"0x{mapping.vram:08X}",
                                                     "size": mapping.size},
            "summary": summarise(blocks, strings, tasks)}


def summarise(blocks: list[dict], strings=(), tasks=()) -> dict:
    counts, text_bytes = Counter(), Counter()
    for b in blocks:
        counts[f"{b['kind']}:{b['status']}"] += 1
        if b["kind"] == "text":
            text_bytes[b["status"]] += b["size"]
    return {"blocks": dict(sorted(counts.items())), "text_bytes": dict(sorted(text_bytes.items())),
            "text_bytes_total": sum(text_bytes.values()), "strings": len(strings), "tasks": len(tasks)}


# --------------------------------------------------------------------------- boot

def identify_boot(data: bytes, entry_rom: int = 0x1000, window: int = 0x100) -> dict:
    """IPL3 by the kit's CIC table, and the entry routine's digests."""
    out: dict = {}
    if len(data) < 0x1000:
        out["ipl3"] = {"status": "absent", "reason": "image shorter than header + IPL3"}
    else:
        ipl3 = bytes(data[0x40:0x1000])
        crc = zlib.crc32(ipl3)
        cic = rommod.CIC_BY_IPL3_CRC.get(crc)
        out["ipl3"] = {"rom": 0x40, "end": 0x1000, "crc32": f"{crc:08X}",
                       "sha256": hashlib.sha256(ipl3).hexdigest(), "cic": cic, "chips": CIC_CHIPS.get(cic),
                       "status": "named" if cic else "unidentified",
                       "basis": "CRC32 of ROM 0x40..0x1000 against rom.CIC_BY_IPL3_CRC" if cic else None,
                       "reason": None if cic else ("CRC32 not in the kit's retail IPL3 table: homebrew, "
                                                   "development or unlisted boot code")}
    end, basis = None, f"window 0x{window:X}"
    for off in range(entry_rom, min(entry_rom + window, len(data) - 7), 4):
        w = struct.unpack_from(">I", data, off)[0]
        if w >> 26 == 0 and w & 0x3F in (8, 9) and w != 0:
            end, basis = off + 8, "first jr/jalr and its delay slot"
            break
    if end is None:
        end = min(entry_rom + window, len(data))
    code = bytes(data[entry_rom:end])
    out["entry_routine"] = {"rom": entry_rom, "end": end, "size": len(code), "end_basis": basis,
                            "sha256": hashlib.sha256(code).hexdigest(), "rel_digest": rel_digest(code, "text")}
    return out


# --------------------------------------------------------------------------- explanations

NO_C = ("not R4300 code: kept as bytes (binary include); C decompilation does not apply, an RSP "
        "disassembler with a byte-identical round trip would be required")


def explain(report: dict) -> list[dict]:
    """One unresolved-case explanation per RSP block and for the boot code."""
    out = []
    for b in report["blocks"]:
        where = f"ROM 0x{b['rom']:X}..0x{b['end']:X} ({b['extent']} extent)"
        ident = b.get("identity") or {}
        label = " ".join(str(x) for x in (ident.get("name") or ident.get("family"), ident.get("version")) if x)
        if b["status"] in ("named", "named-unlisted"):
            cause = f"RSP {b['kind']} of {label or 'unlisted microcode'}, {where}: named by its version string"
            action = NO_C
        elif b["status"] in ("role", "paired"):
            cause = f"RSP {b['kind']} of {label}, {where}: {ident['basis']}"
            action = NO_C
        elif b["status"] == "candidate-by-size":
            cause = f"RSP {b['kind']} {where}: candidate {ident['name']} by size only"
            action = "confirm by a version string, an OSTask, or the same rel_digest in a title where it is named"
        else:
            cause = f"RSP {b['kind']} {where}: unidentified ({b['reason']})"
            action = ("custom or unlisted microcode: compare rel_digest across the corpus; "
                      "add a row from a public source")
        out.append({"rom": b["rom"], "end": b["end"], "unit": "rsp", "status": b["status"],
                    "cause": cause, "next_action": action})
    boot = report.get("boot") or {}
    ipl3 = boot.get("ipl3") or {}
    if ipl3:
        cause = (f"IPL3 boot code ROM 0x40..0x1000: {ipl3.get('chips')} family ({ipl3.get('basis')})"
                 if ipl3.get("status") == "named" else f"IPL3 boot code: {ipl3.get('status')} ({ipl3.get('reason')})")
        out.append({"rom": 0x40, "end": 0x1000, "unit": "boot", "status": ipl3.get("status"), "cause": cause,
                    "next_action": "none: the IPL3 is copied byte for byte, never decompiled"})
    return out


# --------------------------------------------------------------------------- census

def _layout_mapping(project: Path) -> tuple[Mapping | None, str]:
    layout = project / "survey" / "layout.json"
    try:
        main = json.loads(layout.read_text()).get("main") or {}
    except (OSError, ValueError):
        return None, "no survey/layout.json"
    if main.get("kind") not in (None, "rom"):
        return None, f"main image kind {main.get('kind')!r}: not stored plainly, no referenced boundaries"
    try:
        rom, rom_end, vram = int(main["rom"], 16), int(main["rom_end"], 16), int(main["vram"], 16)
    except (KeyError, TypeError, ValueError):
        return None, "survey/layout.json main image without rom/rom_end/vram"
    return Mapping(rom, vram, rom_end - rom), "survey/layout.json main image"


def _project_rom(project: Path) -> Path:
    try:
        config = tomllib.loads((project / "n64dk.toml").read_text())
        return project / config["images"]["main"]["path"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return project / "baserom.z64"


def run_one(title: str, path: Path, project: Path | None = None, span: int = MAIN_SPAN) -> dict:
    image = rommod.Rom.load(path)
    data = image.data
    mapping, basis = _layout_mapping(project) if project is not None else (None, "")
    if project is None:
        mapping = Mapping(0x1000, image.entry_vram, max(0, min(span, len(data) - 0x1000)))
        basis = "header entry, linear from ROM 0x1000"
    report = identify(data, mapping=mapping)
    report["boot"] = identify_boot(data)
    report["title"] = title
    report["mapping_basis"] = basis
    report["explanations"] = explain(report)
    return report


def cross_title(reports: list[dict]) -> dict:
    """Digests shared by titles: exact RSP blocks by rel_digest, entry routines, IPL3s."""
    groups: dict[str, list] = defaultdict(list)
    for r in reports:
        for b in r["blocks"]:
            if b["extent"] == "exact":
                groups[b["rel_digest"]].append([r["title"], b["kind"], b["status"],
                                                (b.get("identity") or {}).get("name")])
    entry = Counter(r["boot"]["entry_routine"]["rel_digest"] for r in reports)
    ipl3 = Counter((r["boot"].get("ipl3") or {}).get("cic") or "unknown" for r in reports)
    totals = Counter()
    for r in reports:
        totals.update(r["summary"]["blocks"])
    text_bytes = Counter()
    for r in reports:
        text_bytes.update(r["summary"]["text_bytes"])
    return {"titles": len(reports), "blocks": dict(sorted(totals.items())),
            "text_bytes": dict(sorted(text_bytes.items())),
            "shared_block_digests": {k[:16]: v for k, v in groups.items() if len({t[0] for t in v}) > 1},
            "entry_routine_groups": sorted(entry.values(), reverse=True), "ipl3_cic": dict(ipl3)}


def _line(report: dict) -> dict:
    named = sorted({" ".join(str(x) for x in ((b.get("identity") or {}).get("name"),
                                                (b.get("identity") or {}).get("version")) if x)
                    for b in report["blocks"] if b["status"] in ("named", "role", "paired")})
    return {"title": report["title"], "cic": (report["boot"].get("ipl3") or {}).get("cic"),
            "mapping": report["mapping_basis"], "summary": report["summary"], "named": named,
            "entry_rel": report["boot"]["entry_routine"]["rel_digest"][:16]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m n64dk.analysis.rspid",
                                 description="Identify RSP microcode and boot code (report only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("rom", help="ROM images; references from the header entry, linear")
    one.add_argument("paths", nargs="+")
    one.add_argument("--span", type=lambda s: int(s, 0), default=MAIN_SPAN)
    one.add_argument("--out")
    many = sub.add_parser("census", help="decompile-init project directories (n64dk.toml, survey/layout.json)")
    many.add_argument("paths", nargs="+")
    many.add_argument("--out")
    args = ap.parse_args(argv)
    reports = []
    for p in args.paths:
        path = Path(p)
        try:
            if args.cmd == "rom":
                report = run_one(path.stem, path, span=args.span)
            else:
                report = run_one(path.name, _project_rom(path), project=path)
        except (OSError, rommod.RomError) as exc:
            print(json.dumps({"title": path.name, "error": str(exc)}), flush=True)
            continue
        reports.append(report)
        print(json.dumps(_line(report)), flush=True)
    summary = cross_title(reports)
    print(json.dumps({"summary": summary}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"titles": reports, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
