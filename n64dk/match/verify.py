"""Strict, non-circular verification of a compiled object against retail bytes.

``compare.compare_object`` is an exploration oracle: when a section base or an external
symbol is unknown it *takes the address from the retail word being compared*.  That makes
matching possible before a symbol map exists, but it also means the comparison can no longer
disagree at those words: a call to the wrong function, a read of the wrong global, or a
static placed in the wrong section slot all compare equal.  Its data comparison likewise
skips sections without a base and forgives a candidate's trailing zeros.

``strict_compare`` is the proof oracle.  It distinguishes three outcomes:

``exact``          every relocation was resolved from independent declared information (symbol
                   files, manifest ``--symbol``/``--section-base``, or symbols defined by other
                   verified units; never candidate source markers or address-bearing names) and every owned
                   byte of ``.text`` and of each non-empty ``.rodata``/``.data`` is equal, and no
                   owned section carries non-zero bytes past its owned size;
``assumed``        the bytes are equal only after taking some addresses from the retail words.
                   Each assumption is listed (symbol or section, address, sites) so it can be
                   reviewed and written into the symbol file, after which the unit is ``exact``;
``mismatch``       bytes differ, or an assumption contradicts declared information (the
                   retail-derived address already belongs to a different declared symbol, two
                   units need different addresses for one symbol, a call lands on an address
                   that is not a function start);

plus ``unverified-data`` notes for non-empty data sections that could not be compared, and for
every other section the object loads into the image (``SHT_PROGBITS`` with ``SHF_ALLOC``, such as
``.ctors`` or a second read-only section): those are never compared, so never ignored either.
``.bss``/``.sbss`` (``SHT_NOBITS``) hold no bytes; nothing is claimed about them.

Nothing here weakens or replaces exploration; it is what integration and progress reporting
should require.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

from ..elf import GP_SYMBOL, R_MIPS_26, Elf
from . import compare as cmp
from .symbols import SymbolMap

# a name that spells its own address (``func_80012340``, ``D_800A1230``): whatever resolves it, declared
# or taken from the retail words, must give that address, or the reference is to the wrong symbol
ADDRESS_NAME = re.compile(r"(?:D|func|jtbl|sym|data|B|L)_([0-9A-Fa-f]{8})")

SHT_PROGBITS, SHF_ALLOC = 1, 0x2


@dataclass
class Assumption:
    kind: str                  # "symbol" | "section"
    name: str
    address: int
    sites: list[int] = field(default_factory=list)      # text offsets that depend on it
    call: bool = False

    def as_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "address": f"0x{self.address:08X}",
                "sites": [f"0x{s:X}" for s in self.sites[:8]], "call": self.call}


@dataclass
class Verdict:
    verdict: str               # exact | assumed | mismatch
    text_equal: bool
    first_diff: int | None = None
    assumptions: list[Assumption] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    compiled_size: int = 0

    @property
    def exact(self) -> bool:
        return self.verdict == "exact"

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "text_equal": self.text_equal,
                "first_diff": None if self.first_diff is None else f"0x{self.first_diff:X}",
                "assumptions": [a.as_dict() for a in self.assumptions], "conflicts": self.conflicts,
                "sections": self.sections, "notes": self.notes, "compiled_size": self.compiled_size}


def _first_diff(a: bytes, b: bytes) -> int | None:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def loaded_sections(elf: Elf) -> list[str]:
    """Names of the object's non-empty sections that put initialized bytes into the image
    (``SHT_PROGBITS`` with ``SHF_ALLOC``; the flags are read from the section headers)."""
    raw = getattr(elf, "data", b"") or b""
    if len(raw) < 0x34:
        return []
    shoff, = struct.unpack_from(">I", raw, 0x20)
    shentsize, = struct.unpack_from(">H", raw, 0x2E)
    out = []
    for section in elf.sections:
        where = shoff + section.index * shentsize + 8
        if section.type == SHT_PROGBITS and section.size and where + 4 <= len(raw) \
                and struct.unpack_from(">I", raw, where)[0] & SHF_ALLOC:
            out.append(section.name)
    return out


def strict_compare(elf: Elf, rom: bytes, vram_to_rom, text_vram: int, size: int, symbols: SymbolMap,
                   bases: dict[str, int] | None = None, section_sizes: dict[str, int] | None = None,
                   function_starts: set[int] | None = None) -> Verdict:
    """``bases`` holds declared section bases only; ``symbols`` declared symbols only."""
    declared = dict(bases or {})
    declared[".text"] = text_vram
    # Strip name-derived pseudo-symbols before *any* dependency or base analysis.
    syms = symbols.explicit()
    have = cmp.complete_bases(elf, ".text", declared, syms)
    rom_offset = vram_to_rom(text_vram)
    target = rom[rom_offset:rom_offset + size]
    text_section = cmp.find_section(elf, ".text")
    result = Verdict("mismatch", False, compiled_size=text_section.size if text_section else 0)
    if text_section is None or len(target) != size:
        result.notes.append("no .text in the object" if text_section is None else "target range is outside the image")
        return result

    relocations = cmp.relocations(elf, ".text")
    need_sections = {r.section for r in relocations if r.section and r.section != ".text" and r.section not in have}
    need_symbols = {r.symbol for r in relocations if r.section is None and r.symbol
                    and syms.resolve(r.symbol) is None}
    if need_sections or need_symbols:
        new_bases, new_symbols = cmp.infer_from_rom(elf, target, text_vram, have, syms, True)
        for name in sorted(need_sections):
            if name in new_bases:
                have[name] = new_bases[name]
                result.assumptions.append(Assumption("section", name, new_bases[name],
                                                     [r.offset for r in relocations if r.section == name]))
            else:
                result.notes.append(f"no address could be derived for section {name}")
        for name in sorted(need_symbols):
            if name in new_symbols:
                syms[name] = new_symbols[name]
                sites = [r for r in relocations if r.section is None and r.symbol == name]
                result.assumptions.append(Assumption("symbol", name, new_symbols[name], [r.offset for r in sites],
                                                     any(r.type == R_MIPS_26 for r in sites)))
            else:
                result.notes.append(f"undefined symbol {name}")
    try:
        text = elf.relocate(text_section.name, have, syms)
    except (KeyError, ValueError) as error:
        result.notes.append(f"relocation: {error}")
        return result
    body = text[:size]
    result.text_equal = body == target and not any(text[size:])
    result.first_diff = _first_diff(body, target)
    if len(text) < size:
        result.notes.append(f"compiled .text is {len(text)} bytes, target is {size}")
    elif any(text[size:]):
        result.notes.append("compiled .text continues with non-zero bytes past the owned size")

    # an assumption may not contradict what is declared
    by_address: dict[int, set[str]] = {}
    for name, address in symbols.items():
        if name != GP_SYMBOL:                        # _gp marks an address; it names no object there
            by_address.setdefault(address, set()).add(name)
    for item in result.assumptions:
        if item.kind != "symbol":
            continue
        others = by_address.get(item.address, set()) - {item.name}
        if others:
            result.conflicts.append(f"{item.name} would have to be at 0x{item.address:08X}, which is declared as "
                                    f"{', '.join(sorted(others))}: wrong symbol, or an alias that must be declared")
        if item.call and function_starts is not None and item.address not in function_starts:
            result.conflicts.append(f"call to {item.name} would land on 0x{item.address:08X}, which is not a "
                                    "function start")

    for name in sorted({r.symbol for r in relocations if r.section is None and r.symbol}):
        spelled = ADDRESS_NAME.fullmatch(name)
        address = syms.resolve(name)
        if spelled and address is not None and address != int(spelled.group(1), 16):
            result.conflicts.append(f"{name} resolves to 0x{address:08X}, not the address its name spells: "
                                    "wrong symbol or wrong addend")

    data_ok, limits = True, section_sizes or {}
    compared = set()
    # any other section the object loads (.ctors, .rodata.<x>, .rdata beside .rodata, ...) is
    # reported, never silently left out of the verdict
    extra = tuple(name for name in loaded_sections(elf)
                   if name != text_section.name and name not in cmp.COMPARED)
    for name in cmp.COMPARED + extra:
        section = cmp.find_section(elf, name)
        if section is None or not section.size or section.name in compared:
            continue                                 # (.sdata falls back to .data: compare it once)
        compared.add(section.name)
        row = {"section": name, "size": section.size}
        base = have.get(section.name, have.get(name))
        if name in extra:
            row["status"] = "unverified"
            result.notes.append(f"{name} ({section.size} bytes) is not a section the strict comparison "
                                "knows: its content was not compared")
            data_ok = False
        elif base is None:
            row["status"] = "unverified"
            result.notes.append(f"{name} ({section.size} bytes) has no declared base and is not referenced by "
                                ".text: its content was not compared")
            data_ok = False
        else:
            assumed = any(a.kind == "section" and a.name in (section.name, name) for a in result.assumptions)
            try:
                data = elf.relocate(section.name, have, syms)
            except (KeyError, ValueError) as error:
                row.update(status="unverified", error=str(error))
                result.notes.append(f"{name}: relocation: {error}")
                data_ok = False
                result.sections.append(row)
                continue
            owned = limits.get(name, len(data))
            offset = vram_to_rom(base)
            row.update(base=f"0x{base:08X}", compared=owned, base_assumed=assumed)
            if offset < 0 or offset + owned > len(rom):
                # the image does not hold these bytes (a range found by code density ends before
                # its data, or the section belongs to another image): nothing to compare, which
                # proves nothing either way
                held = max(0, min(len(rom), offset + owned) - max(0, offset))
                row.update(status="unverified", held=held)
                result.notes.append(f"{name}: the image holds {held} of its {owned} bytes at 0x{base:08X}; "
                                    "its content was not compared")
                data_ok = False
                result.sections.append(row)
                continue
            want = rom[offset:offset + owned]
            if data[:owned] == want and any(data[owned:]):
                # past its owned size the object still holds data, not padding: those bytes would
                # land on another unit's (as for .text, a mismatch)
                row.update(status="different", first_diff=f"0x{owned:X}")
                result.notes.append(f"{name}: the object continues with non-zero bytes past its owned "
                                    f"{owned} bytes")
            elif data[:owned] == want:
                row["status"] = "equal"
            else:
                first = _first_diff(data[:owned], want)
                significant = len(data[:owned].rstrip(b"\0"))
                if name not in limits and first is not None and first >= significant:
                    # only the object's trailing zeros differ: alignment padding, or missing data
                    row.update(status="unverified", first_diff=f"0x{first:X}")
                    result.notes.append(f"{name}: the object's last {owned - first} byte(s) are zero where the "
                                        "image is not; declare the owned size to accept alignment padding")
                else:
                    row.update(status="different", first_diff=None if first is None else f"0x{first:X}")
                data_ok = data_ok and row["status"] == "equal"
        result.sections.append(row)
    different = any(row.get("status") == "different" for row in result.sections)
    if not result.text_equal or different or result.conflicts:
        result.verdict = "mismatch"
    elif result.assumptions or not data_ok or any("undefined symbol" in n or "no address" in n for n in result.notes):
        result.verdict = "assumed"
    else:
        result.verdict = "exact"
    return result


def unit_defines(elf: Elf, text_vram: int, bases: dict[str, int] | None = None) -> dict[str, int]:
    """Global symbols an object defines in sections whose base is declared (``.text`` always is)."""
    known = {".text": text_vram, **(bases or {})}
    out = {}
    for symbol in elf.symbols:
        if symbol.bind == 1 and symbol.name and 0 < symbol.section < len(elf.sections):
            base = known.get(elf.sections[symbol.section].name)
            if base is not None:
                out[symbol.name] = base + symbol.value
    return out


def reconcile_assumptions(rows: list[dict]) -> dict:
    """Cross-unit consistency of retail-derived addresses.

    ``rows`` are ``{"unit": name, "verdict": Verdict}``.  A symbol that needs different
    addresses in different units is a definite error in at least one of them; consistent
    assumptions become a proposed symbol list to review and declare."""
    seen: dict[str, dict[int, list[str]]] = {}
    for row in rows:
        for item in row["verdict"].assumptions:
            if item.kind == "symbol":
                seen.setdefault(item.name, {}).setdefault(item.address, []).append(row["unit"])
    contradictions = {name: {f"0x{address:08X}": units[:6] for address, units in addresses.items()}
                      for name, addresses in seen.items() if len(addresses) > 1}
    proposals = sorted((name, next(iter(addresses))) for name, addresses in seen.items() if len(addresses) == 1)
    return {"contradictions": contradictions,
            "proposed_symbols": [f"{name} = 0x{address:08X};" for name, address in proposals],
            "symbols_assumed": len(seen)}


def jump_targets_outside(code: bytes, vram: int) -> list[tuple[int, int]]:
    """(site, target) of ``j`` instructions leaving ``[vram, vram+len)``: a unit whose retail
    code jumps into another unit cannot be verified in isolation unless that label is declared."""
    out = []
    for offset in range(0, len(code) - 3, 4):
        word, = struct.unpack_from(">I", code, offset)
        if word >> 26 == 2:
            target = ((vram + offset + 4) & 0xF0000000) | ((word & 0x03FFFFFF) << 2)
            if not vram <= target < vram + len(code):
                out.append((vram + offset, target))
    return out
