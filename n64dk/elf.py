"""Minimal ELF32 big-endian reader and relocator for MIPS objects (no external dependency)."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

R_MIPS_32, R_MIPS_26, R_MIPS_HI16, R_MIPS_LO16, R_MIPS_PC16 = 2, 4, 5, 6, 10
R_MIPS_GPREL16, R_MIPS_LITERAL = 7, 8
GP_RELATIVE = (R_MIPS_GPREL16, R_MIPS_LITERAL)
GP_SYMBOL = "_gp"


@dataclass
class Section:
    index: int
    name: str
    type: int
    offset: int
    size: int
    link: int
    info: int
    data: bytes
    align: int = 1


@dataclass
class Symbol:
    name: str
    value: int
    size: int
    section: int
    bind: int
    type: int


class Elf:
    def __init__(self, data: bytes):
        if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 2:
            raise ValueError("not an ELF32 big-endian object")
        self.data = data
        shoff, = struct.unpack_from(">I", data, 0x20)
        shentsize, shnum, shstrndx = struct.unpack_from(">HHH", data, 0x2E)
        raw = [struct.unpack_from(">IIIIIIIIII", data, shoff + i * shentsize) for i in range(shnum)]
        names = raw[shstrndx]
        strtab = data[names[4]:names[4] + names[5]]
        self.sections = []
        for i, s in enumerate(raw):
            name = strtab[s[0]:strtab.index(b"\0", s[0])].decode()
            body = b"" if s[1] == 8 else data[s[4]:s[4] + s[5]]  # SHT_NOBITS
            self.sections.append(Section(i, name, s[1], s[4], s[5], s[6], s[7], body, max(1, s[8])))
        self.symbols = self._symbols()

    @classmethod
    def load(cls, path: str | Path) -> "Elf":
        return cls(Path(path).read_bytes())

    def section(self, name: str) -> Section | None:
        return next((s for s in self.sections if s.name == name), None)

    def _symbols(self) -> list[Symbol]:
        symtab = next((s for s in self.sections if s.type == 2), None)
        if symtab is None:
            return []
        strtab = self.sections[symtab.link].data
        out = []
        for off in range(0, len(symtab.data), 16):
            name, value, size, info, _other, shndx = struct.unpack_from(">IIIBBH", symtab.data, off)
            text = strtab[name:strtab.index(b"\0", name)].decode() if name else ""
            out.append(Symbol(text, value, size, shndx, info >> 4, info & 0xF))
        return out

    def relocations(self, target: Section) -> list[tuple[int, int, int, int]]:
        """(offset, type, symbol index, addend) for REL/RELA sections applying to ``target``."""
        out = []
        for s in self.sections:
            if s.info != target.index or s.type not in (9, 4):  # SHT_REL, SHT_RELA
                continue
            step = 8 if s.type == 9 else 12
            for off in range(0, len(s.data), step):
                if s.type == 9:
                    r_off, r_info = struct.unpack_from(">II", s.data, off)
                    addend = None
                else:
                    r_off, r_info, addend = struct.unpack_from(">IIi", s.data, off)
                out.append((r_off, r_info & 0xFF, r_info >> 8, addend))
        return out

    def gp0(self) -> int:
        """The GP value the assembler assumed (``.reginfo`` ``ri_gp_value``): GP-relative addends of
        local symbols are relative to it (KMC ``as`` 2.6 writes 0x4000; GNU ``as`` 2.9 writes 0)."""
        reginfo = self.section(".reginfo")
        if reginfo is None or len(reginfo.data) < 24:
            return 0
        return struct.unpack_from(">i", reginfo.data, 20)[0]

    def relocate(self, name: str, bases: dict[str, int], symbols: dict[str, int]) -> bytes:
        """Return section ``name`` with MIPS relocations applied.

        ``bases`` maps section names of this object to their final addresses;
        ``symbols`` maps external symbol names to addresses.  HI16/LO16 pairs follow the
        MIPS ABI: in relocation-table order, a HI16 takes its addend from the next LO16
        *of the same symbol* (scheduled IDO code interleaves HI16s of different symbols).
        ``R_MIPS_GPREL16`` and ``R_MIPS_LITERAL`` (small data, ``-G``) resolve against the program's
        ``_gp``, which must be in ``symbols`` (an independently established, declared value): the
        field becomes ``A + S + GP0 - GP`` for a local symbol and ``A + S - GP`` for an external
        one, and a value outside the signed 16-bit range is an error, never truncated.
        """
        sec = self.section(name)
        if sec is None:
            return b""
        buf = bytearray(sec.data)

        def sym_addr(index: int) -> int:
            sym = self.symbols[index]
            if sym.section not in (0, 0xFFF1, 0xFFF2) and sym.section < len(self.sections):
                sec_name = self.sections[sym.section].name
                return bases.get(sec_name, 0) + sym.value
            if sym.section == 0xFFF1:
                return sym.value
            if sym.name not in symbols:
                raise KeyError(f"undefined symbol {sym.name}")
            return symbols[sym.name]

        pending_hi: dict[int, list[int]] = {}
        for off, rtype, sidx, addend in self.relocations(sec):
            word, = struct.unpack_from(">I", buf, off)
            s = sym_addr(sidx)
            if rtype == R_MIPS_32:
                value = (word + s if addend is None else s + addend) & 0xFFFFFFFF
                struct.pack_into(">I", buf, off, value)
            elif rtype == R_MIPS_26:
                target = ((word & 0x3FFFFFF) << 2) if addend is None else addend
                value = (word & 0xFC000000) | (((s + target) >> 2) & 0x3FFFFFF)
                struct.pack_into(">I", buf, off, value)
            elif rtype == R_MIPS_HI16:
                pending_hi.setdefault(sidx, []).append(off)
            elif rtype == R_MIPS_LO16:
                lo = word & 0xFFFF
                lo = lo - 0x10000 if lo & 0x8000 else lo
                for hoff in pending_hi.pop(sidx, []):
                    hword, = struct.unpack_from(">I", buf, hoff)
                    full = ((hword & 0xFFFF) << 16) + lo + s
                    struct.pack_into(">I", buf, hoff, (hword & 0xFFFF0000) | (((full + 0x8000) >> 16) & 0xFFFF))
                value = (word & 0xFFFF0000) | ((s + lo) & 0xFFFF)
                struct.pack_into(">I", buf, off, value)
            elif rtype == R_MIPS_PC16:
                # a branch to a label outside the object (hand-written code branches between routines):
                # the field holds the word offset from the delay slot, with the addend stored in place
                held = word & 0xFFFF
                held = (held - 0x10000 if held & 0x8000 else held) << 2 if addend is None else addend
                place = bases.get(name, 0) + off
                distance = s + held - place
                if distance & 3 or not -0x20000 <= distance < 0x20000:
                    raise ValueError(f"branch out of range at 0x{off:X} in {name}")
                struct.pack_into(">I", buf, off, (word & 0xFFFF0000) | ((distance >> 2) & 0xFFFF))
            elif rtype in GP_RELATIVE:
                if GP_SYMBOL not in symbols:
                    raise KeyError(f"undefined symbol {GP_SYMBOL} (GP-relative relocation at 0x{off:X} in {name})")
                gp = symbols[GP_SYMBOL]
                held = word & 0xFFFF
                held = (held - 0x10000 if held & 0x8000 else held) if addend is None else addend
                symbol = self.symbols[sidx]
                local = symbol.bind == 0 or symbol.type == 3          # STB_LOCAL or a section symbol
                value = held + s + (self.gp0() if local else 0) - gp
                if not -0x8000 <= value < 0x8000:
                    raise ValueError(f"GP-relative relocation out of range at 0x{off:X} in {name}: "
                                     f"0x{(s + held) & 0xFFFFFFFF:08X} is {value} from _gp")
                struct.pack_into(">I", buf, off, (word & 0xFFFF0000) | (value & 0xFFFF))
            else:
                raise ValueError(f"unsupported relocation type {rtype} at 0x{off:X} in {name}")
        if pending_hi:
            raise ValueError(f"unpaired R_MIPS_HI16 in {name}")
        return bytes(buf)
