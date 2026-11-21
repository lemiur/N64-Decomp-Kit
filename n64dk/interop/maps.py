"""Linker map files read into symbol and object-order facts (link layout evidence, not proof).

**GNU ld ``-Map``** (``read_gnu_map``).  After the ``Linker script and memory map`` heading, the
listing has three kinds of line the reader uses::

    .main           0x0000000080246000    0x12340 load address 0x0000000000001000   <- output section
     .text          0x0000000080246000      0x2a0 build/src/game/main.o             <- input section
                    0x0000000080246000                main_func                     <- symbol
                    0x0000000080246120                _mainSegmentEnd = .           <- assignment

A section name too long for its column is printed alone with its numbers on the next line; the
reader joins the two.  ``*fill*``, ``*(...)`` patterns, ``LOAD`` and ``OUTPUT`` lines are skipped;
``COMMON`` input sections are kept (section ``COMMON``).  Addresses printed 64 bits wide by a 64-bit
host are reduced to 32 bits.  The ROM offset of an input section is its output section's load
address plus its offset in the output section (the same arithmetic asm-differ uses).

**Column listings** (``read_column_map``): one symbol per line with an address, a name and an object
(and optionally a section and a size) in whitespace-separated columns; a header line naming the
columns (``address``/``addr``/``value``, ``symbol``/``name``, ``object``/``file``/``module``,
``section``, ``size``) fixes their order, otherwise ``address symbol object`` is assumed.  This is
the shape of IDO-era (IRIX ``ld -m``) and other tools' map output as projects keep it; the exact
IRIX layout was not available to check, so ``format_verified`` is False for these facts.

``read_map`` picks the reader.  ``object_order`` lists objects in link order per output section:
the evidence a translation-unit or link-order reconstruction can weigh.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from . import PROVENANCE
from .symaddrs import SymbolFile, entry as new_entry

_HEX = r"0x[0-9A-Fa-f]+"
_OUTPUT = re.compile(rf"^([^\s*][^\s]*)\s+({_HEX})\s+({_HEX})(?:\s+load address\s+({_HEX}))?\s*$")
_OUTPUT_NUMBERS = re.compile(rf"^\s+({_HEX})\s+({_HEX})(?:\s+load address\s+({_HEX}))?\s*$")
_INPUT = re.compile(rf"^ (\.[^\s]*|COMMON|\.?[A-Za-z_][\w.$]*)\s+({_HEX})\s+({_HEX})\s+(\S.*?)\s*$")
_INPUT_NAME_ONLY = re.compile(r"^ (\.[^\s]+|COMMON)\s*$")
_INPUT_NUMBERS = re.compile(rf"^\s+({_HEX})\s+({_HEX})\s+(\S.*?)\s*$")
_SYMBOL = re.compile(rf"^\s{{8,}}({_HEX})\s+([A-Za-z_.$][\w.$]*)\s*$")
_ASSIGN = re.compile(rf"^\s{{8,}}({_HEX})\s+(?:PROVIDE\s*\(\s*)?([A-Za-z_.$][\w.$]*)\s*=")


def _addr(text: str) -> int:
    return int(text, 16) & 0xFFFFFFFF


@dataclass
class MapSymbol:
    name: str
    address: int
    object: str | None
    section: str | None
    output_section: str | None
    rom: int | None = None
    kind: str = "symbol"                # "symbol" | "assignment" (linker-script defined)
    line: int | None = None


@dataclass
class MapObject:
    object: str
    section: str
    address: int
    size: int
    output_section: str | None
    rom: int | None
    order: int
    line: int | None = None


@dataclass
class MapFacts:
    format: str
    format_verified: bool
    symbols: list[MapSymbol] = field(default_factory=list)
    objects: list[MapObject] = field(default_factory=list)
    output_sections: list[dict] = field(default_factory=list)
    skipped_lines: int = 0
    provenance: str = PROVENANCE

    def as_dict(self) -> dict:
        return {"format": self.format, "format_verified": self.format_verified, "provenance": self.provenance,
                "symbols": len(self.symbols), "objects": len(self.objects),
                "output_sections": self.output_sections, "skipped_lines": self.skipped_lines}

    def symbol_file(self, include_assignments: bool = False) -> SymbolFile:
        """The map's symbols as ``symbol_addrs.txt`` entries (no type or size: a map gives
        neither for a symbol)."""
        out = SymbolFile()
        out.lines.append(f"// symbols read from a {self.format} map file ({self.provenance})")
        seen = set()
        for s in sorted(self.symbols, key=lambda s: (s.address, s.name)):
            if s.kind != "symbol" and not include_assignments:
                continue
            if (s.name, s.address) in seen:
                continue
            seen.add((s.name, s.address))
            out.add(new_entry(s.name, s.address, rom=s.rom))
        return out


def read_gnu_map(text: str) -> MapFacts:
    facts = MapFacts("gnu-ld", True)
    lines = text.splitlines()
    start = next((k + 1 for k, line in enumerate(lines) if line.startswith("Linker script and memory map")), 0)
    output = None                       # (name, vram, size, load)
    current = None                      # MapObject of the input section being listed
    pending_output = pending_input = None
    order = 0
    for number, line in enumerate(lines[start:], start + 1):
        if not line.strip():
            continue
        if pending_output is not None:
            m = _OUTPUT_NUMBERS.match(line)
            name, pending_output = pending_output, None
            if m:
                vram, size = _addr(m.group(1)), int(m.group(2), 16)
                load = _addr(m.group(3)) if m.group(3) else None
                output = (name, vram, size, load)
                facts.output_sections.append({"name": name, "vram": f"0x{vram:08X}", "size": f"0x{size:X}",
                                              "rom": None if load is None else f"0x{load:X}"})
                current = None
                continue
        if pending_input is not None:
            m = _INPUT_NUMBERS.match(line)
            section, pending_input = pending_input, None
            if m:
                current = _input(facts, section, m.group(1), m.group(2), m.group(3), output, order, number)
                order += 1
                continue
        if line.startswith(("LOAD ", "OUTPUT(", "OUTPUT (", "START GROUP", "END GROUP", "Discarded")):
            continue
        if not line[0].isspace():
            m = _OUTPUT.match(line)
            if m:
                vram, size = _addr(m.group(2)), int(m.group(3), 16)
                load = _addr(m.group(4)) if m.group(4) else None
                output = (m.group(1), vram, size, load)
                facts.output_sections.append({"name": m.group(1), "vram": f"0x{vram:08X}", "size": f"0x{size:X}",
                                              "rom": None if load is None else f"0x{load:X}"})
                current = None
            elif re.match(r"^\.?[A-Za-z_][\w.$]*\s*$", line):
                pending_output = line.strip()
            else:
                facts.skipped_lines += 1
            continue
        stripped = line.strip()
        if stripped.startswith(("*fill*", "*(", "*", "FILL", "ASSERT", "[")) or "(*" in stripped.split()[0]:
            continue
        m = _INPUT.match(line)
        if m and not m.group(4).startswith(("0x", "=")):
            current = _input(facts, m.group(1), m.group(2), m.group(3), m.group(4), output, order, number)
            order += 1
            continue
        m = _INPUT_NAME_ONLY.match(line)
        if m:
            pending_input = m.group(1)
            continue
        m = _SYMBOL.match(line)
        if m:
            address = _addr(m.group(1))
            facts.symbols.append(MapSymbol(m.group(2), address, current.object if current else None,
                                           current.section if current else None, output[0] if output else None,
                                           _rom_of(address, current, output), "symbol", number))
            continue
        m = _ASSIGN.match(line)
        if m:
            if m.group(2) != ".":
                address = _addr(m.group(1))
                facts.symbols.append(MapSymbol(m.group(2), address, None, None, output[0] if output else None,
                                               None, "assignment", number))
            continue
        facts.skipped_lines += 1
    return facts


def _input(facts, section, address, size, obj, output, order, number) -> MapObject:
    vram, length = _addr(address), int(size, 16)
    rom = None
    if output is not None and output[3] is not None:
        rom = output[3] + (vram - output[1])
    item = MapObject(obj.strip(), section, vram, length, output[0] if output else None, rom, order, number)
    if length or section == "COMMON":
        facts.objects.append(item)
    return item


def _rom_of(address: int, current: MapObject | None, output) -> int | None:
    if current is not None and current.rom is not None:
        return current.rom + (address - current.address)
    if output is not None and output[3] is not None:
        return output[3] + (address - output[1])
    return None


_COLUMN_NAMES = {"address": "address", "addr": "address", "value": "address", "vaddr": "address",
                 "symbol": "symbol", "name": "symbol", "object": "object", "file": "object",
                 "module": "object", "section": "section", "size": "size"}


def read_column_map(text: str, columns: list[str] | None = None) -> MapFacts:
    facts = MapFacts("columns", False)
    order = columns
    seen_objects: dict[tuple, MapObject] = {}
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", ";")):
            continue
        cells = stripped.split()
        lowered = [c.lower().strip(":") for c in cells]
        if order is None and sum(c in _COLUMN_NAMES for c in lowered) >= 2:
            order = [_COLUMN_NAMES.get(c, c) for c in lowered]
            continue
        layout = order or ["address", "symbol", "object"]
        row = dict(zip(layout, cells))
        try:
            address = _addr(row["address"] if row.get("address", "").lower().startswith("0x")
                            else "0x" + row.get("address", ""))
        except ValueError:
            facts.skipped_lines += 1
            continue
        name = row.get("symbol")
        if not name or not re.fullmatch(r"[A-Za-z_.$][\w.$]*", name):
            facts.skipped_lines += 1
            continue
        obj, section = row.get("object"), row.get("section")
        facts.symbols.append(MapSymbol(name, address, obj, section, None, None, "symbol", number))
        if obj and (obj, section) not in seen_objects:
            item = MapObject(obj, section or "", address, 0, None, None, len(seen_objects), number)
            seen_objects[(obj, section)] = item
            facts.objects.append(item)
        elif obj:
            item = seen_objects[(obj, section)]
            item.address = min(item.address, address)
    return facts


def read_map(text: str) -> MapFacts:
    if "Linker script and memory map" in text or re.search(r"load address\s+0x", text):
        return read_gnu_map(text)
    return read_column_map(text)


def object_order(facts: MapFacts) -> dict[str, list[dict]]:
    """Objects per output section in address order (link layout evidence)."""
    out: dict[str, list[dict]] = {}
    for item in sorted(facts.objects, key=lambda o: (o.output_section or "", o.address, o.order)):
        out.setdefault(item.output_section or "", []).append(
            {"object": item.object, "section": item.section, "vram": f"0x{item.address:08X}",
             "size": f"0x{item.size:X}", "rom": None if item.rom is None else f"0x{item.rom:X}",
             "provenance": facts.provenance})
    return out


def symbols_by_object(facts: MapFacts) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for s in facts.symbols:
        if s.kind == "symbol" and s.object:
            out.setdefault(s.object, []).append({"name": s.name, "vram": f"0x{s.address:08X}",
                                                 "rom": None if s.rom is None else f"0x{s.rom:X}"})
    return out


def as_rows(facts: MapFacts) -> list[dict]:
    return [asdict(s) for s in facts.symbols]
