"""Adversarial tests of the strict verifier's data, .rodata and BSS rules (round 12, workstream 7).

The verifier must never: forgive trailing zeros (a); ignore data, referenced or not (b); take
proof from target bytes (c); claim .bss/.sbss content from ROM bytes (d); merge adjacent objects
or sections without evidence (e).  Objects are hand-built big-endian MIPS relocatables (no
toolchain); the image is a bytearray whose offset is ``vram - BASE``.
"""
import struct
import unittest

from n64dk.elf import R_MIPS_32, R_MIPS_GPREL16, R_MIPS_HI16, R_MIPS_LITERAL, R_MIPS_LO16, Elf
from n64dk.match import compare as cmp
from n64dk.match import transplant
from n64dk.match import verify
from n64dk.match.symbols import SymbolMap

PROGBITS, NOBITS = 1, 8
WRITE, ALLOC, EXEC, MIPS_GPREL = 0x1, 0x2, 0x4, 0x10000000
LOCAL, GLOBAL = 0, 1
NOTYPE, OBJECT, FUNC, SECTION = 0, 1, 2, 3
SCOMMON = 0xFF03                                       # SHN_MIPS_SCOMMON (small .comm symbols)
FLAGS = {".text": ALLOC | EXEC, ".data": ALLOC | WRITE, ".sdata": ALLOC | WRITE | MIPS_GPREL,
         ".lit4": ALLOC | MIPS_GPREL, ".lit8": ALLOC | MIPS_GPREL, ".bss": ALLOC | WRITE,
         ".sbss": ALLOC | WRITE | MIPS_GPREL}

BASE, TEXT = 0x80000000, 0x80000400
GP = 0x80001800
SDATA, SBSS, LIT4, LIT8 = 0x80001700, 0x80001740, 0x80001780, 0x80001788
LUI, LW, LWC1, JR_RA, NOP = 15, 35, 49, 0x03E00008, 0


def i_type(op: int, rs: int, rt: int, imm: int) -> int:
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def words(*values: int) -> bytes:
    return struct.pack(f">{len(values)}I", *values)


PLAIN = words(JR_RA, NOP)                              # references nothing
RO16 = b"\x3f\x80\x00\x00\x40\x20\x00\x00\x41\x00\x00\x00\x42\x00\x00\x00"


def build(sections, symbols=(), relocations=None) -> Elf:
    """A big-endian MIPS ELF32 relocatable.

    ``sections``: ``(name, body[, flags[, align[, type]]])``; ``body`` is bytes, or an int for an
    ``SHT_NOBITS`` section of that size; flags default to FLAGS (else ``SHF_ALLOC``).  Each section
    gets a local section symbol.  ``symbols``: ``(name, value, bind, type, section)`` with
    ``section`` a section name or an index (0 = undefined).  ``relocations``:
    ``{section: [(offset, type, symbol or section name), ...]}`` (REL, addend in place)."""
    relocations = relocations or {}
    specs = []
    for item in sections:
        name, body = item[0], item[1]
        flags = item[2] if len(item) > 2 else FLAGS.get(name, ALLOC)
        align = item[3] if len(item) > 3 else 16
        kind = item[4] if len(item) > 4 else (PROGBITS if isinstance(body, bytes) else NOBITS)
        specs.append((name, body, flags, align, kind))
    names = [spec[0] for spec in specs]
    index = {name: k + 1 for k, name in enumerate(names)}
    targets = [name for name in names if name in relocations]
    symtab_index = len(names) + len(targets) + 1
    every = names + [".rel" + name for name in targets] + [".symtab", ".strtab", ".shstrtab"]
    shstr = b"\0" + b"".join(n.encode() + b"\0" for n in every)

    def name_at(n: str) -> int:
        return shstr.index(b"\0" + n.encode() + b"\0") + 1

    strtab = bytearray(b"\0")
    entries, symbol_index = [bytes(16)], {}
    for name in names:
        symbol_index[name] = len(entries)
        entries.append(struct.pack(">IIIBBH", 0, 0, 0, (LOCAL << 4) | SECTION, 0, index[name]))
    for name, value, bind, kind, where in symbols:
        symbol_index[name] = len(entries)
        shndx = index[where] if isinstance(where, str) else where
        entries.append(struct.pack(">IIIBBH", len(strtab), value, 0, (bind << 4) | kind, 0, shndx))
        strtab += name.encode() + b"\0"
    symtab = b"".join(entries)
    rels = [b"".join(struct.pack(">II", off, (symbol_index[sym] << 8) | rtype)
                     for off, rtype, sym in relocations[name]) for name in targets]
    blobs = [body if isinstance(body, bytes) else b"" for _n, body, _f, _a, _k in specs]
    blobs += rels + [symtab, bytes(strtab), shstr]
    offsets, position = [], 52
    for blob in blobs:
        offsets.append(position)
        position += len(blob)

    def header(n, kind, flags, offset, size, link=0, info=0, align=4, entsize=0) -> bytes:
        return struct.pack(">10I", name_at(n), kind, flags, 0, offset, size, link, info, align, entsize)

    headers = [bytes(40)]
    for k, (name, body, flags, align, kind) in enumerate(specs):
        size = len(body) if isinstance(body, bytes) else body
        headers.append(header(name, kind, flags, offsets[k], size, align=align))
    for j, name in enumerate(targets):
        headers.append(header(".rel" + name, 9, 0, offsets[len(specs) + j], len(rels[j]), symtab_index, index[name],
                              4, 8))
    k = len(specs) + len(targets)
    headers.append(header(".symtab", 2, 0, offsets[k], len(symtab), symtab_index + 1, 1, 4, 16))
    headers.append(header(".strtab", 3, 0, offsets[k + 1], len(strtab), align=1))
    headers.append(header(".shstrtab", 3, 0, offsets[k + 2], len(shstr), align=1))
    elf_header = (b"\x7fELF\x01\x02\x01" + bytes(9)
                  + struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, position, 0, 52, 0, 0, 40, len(headers),
                                len(headers) - 1))
    return Elf(elf_header + b"".join(blobs) + b"".join(headers))


def image(elf: Elf, bases: dict, symbols: dict | None = None, size: int = 0x4000) -> bytearray:
    """The image a correct link gives: each initialized section of ``elf`` relocated at its base."""
    rom = bytearray(size)
    for name, base in bases.items():
        section = elf.section(name)
        if section is not None and section.type == PROGBITS and section.size:
            data = elf.relocate(name, bases, symbols or {})
            rom[base - BASE:base - BASE + len(data)] = data
    return rom


def strict(elf: Elf, rom, symbols: dict | None = None, bases: dict | None = None, sizes: dict | None = None):
    return verify.strict_compare(elf, bytes(rom), lambda v: v - BASE, TEXT, elf.section(".text").size,
                                 SymbolMap(symbols or {}, infer_names=False), bases, sizes, None)


def rows(result) -> dict:
    return {row["section"]: row for row in result.sections}


def gp_program(*references) -> tuple[bytes, list]:
    """``lw``/``lwc1`` through ``$gp`` for each ``(opcode, rtype, symbol, offset)``, then ``jr ra; nop``
    (GP0 = 0: the field in place is the offset)."""
    code, relocs = [], []
    for k, (op, rtype, symbol, offset) in enumerate(references):
        code.append(i_type(op, 28, 2, offset))
        relocs.append((4 * k, rtype, symbol))
    return words(*code, JR_RA, NOP), relocs


class TrailingZerosTest(unittest.TestCase):
    """(a) Data that differs from the image only where the object has trailing zeros is padding or
    missing data; nothing tells them apart, so it is never exact."""

    DATA = b"\x3f\x80\x00\x00\x40\x20\x00\x00" + bytes(8)

    def test_every_compared_section_kind(self):
        where = 0x80001000
        for name in cmp.COMPARED:
            with self.subTest(section=name):
                elf = build([(".text", PLAIN), (name, self.DATA)])
                rom = image(elf, {".text": TEXT, name: where})
                self.assertEqual(strict(elf, rom, bases={name: where}).verdict, "exact")
                rom[where - BASE + 8:where - BASE + 12] = b"\x41\x00\x00\x00"       # the image has more data
                result = strict(elf, rom, bases={name: where})
                self.assertEqual(result.verdict, "assumed")
                self.assertEqual(rows(result)[name]["status"], "unverified")
                owned = strict(elf, rom, bases={name: where}, sizes={name: 16})
                self.assertEqual((owned.verdict, rows(owned)[name]["status"]), ("mismatch", "different"))

    def test_a_relocated_pointer_is_not_trailing_padding(self):
        """A pointer's REL word is zero before relocation; relocated it is data, and a wrong one is a
        difference, never padding a later settlement could trim off."""
        elf = build([(".text", PLAIN), (".rodata", b"\x3f\x80\x00\x00\x40\x20\x00\x00" + bytes(8))],
                    [("other", 0, GLOBAL, FUNC, 0)], {".rodata": [(8, R_MIPS_32, "other")]})
        where = 0x80001000
        rom = image(elf, {".text": TEXT})
        rom[where - BASE:where - BASE + 12] = b"\x3f\x80\x00\x00\x40\x20\x00\x00\x12\x34\x56\x78"
        result = strict(elf, rom, {"other": 0x80000600}, {".rodata": where})
        state = rows(result)[".rodata"]
        self.assertEqual((result.verdict, state["status"]), ("mismatch", "different"))
        self.assertIsNone(transplant._partial_piece(state, elf.section(".rodata")))


class OwnedSizeTest(unittest.TestCase):
    """(a)/(b) A declared owned size accepts zero padding after it, never data."""

    WHERE = 0x80001000

    def test_zero_padding_past_the_owned_size_is_accepted(self):
        elf = build([(".text", PLAIN), (".rodata", RO16[:8] + bytes(8))])
        rom = image(elf, {".text": TEXT})
        rom[self.WHERE - BASE:self.WHERE - BASE + 12] = RO16[:8] + b"\x12\x34\x56\x78"   # the next unit at +8
        self.assertEqual(strict(elf, rom, bases={".rodata": self.WHERE}, sizes={".rodata": 8}).verdict, "exact")

    def test_nonzero_data_past_the_owned_size_is_a_mismatch(self):
        elf = build([(".text", PLAIN), (".rodata", RO16)])
        rom = image(elf, {".text": TEXT})
        rom[self.WHERE - BASE:self.WHERE - BASE + 12] = RO16[:8] + b"\x12\x34\x56\x78"
        result = strict(elf, rom, bases={".rodata": self.WHERE}, sizes={".rodata": 8})
        self.assertEqual((result.verdict, rows(result)[".rodata"]["status"]), ("mismatch", "different"))

    def test_a_relocated_pointer_past_the_owned_size_is_a_mismatch(self):
        elf = build([(".text", PLAIN), (".rodata", RO16[:8] + bytes(8))],
                    [("other", 0, GLOBAL, FUNC, 0)], {".rodata": [(8, R_MIPS_32, "other")]})
        rom = image(elf, {".text": TEXT})
        rom[self.WHERE - BASE:self.WHERE - BASE + 12] = RO16[:8] + b"\x12\x34\x56\x78"
        result = strict(elf, rom, {"other": 0x80000600}, {".rodata": self.WHERE}, {".rodata": 8})
        self.assertEqual(result.verdict, "mismatch")


class UnreferencedDataTest(unittest.TestCase):
    """(b) Data nothing references and nothing places is reported, never left out."""

    def test_every_compared_section_without_a_base(self):
        for name in cmp.COMPARED:
            with self.subTest(section=name):
                elf = build([(".text", PLAIN), (name, RO16)])
                rom = image(elf, {".text": TEXT, name: 0x80001000})   # in the image, but nothing says where
                result = strict(elf, rom)
                self.assertEqual((result.verdict, result.assumptions), ("assumed", []))
                self.assertEqual(rows(result)[name]["status"], "unverified")
                self.assertIn("was not compared", " ".join(result.notes))

    def test_a_loaded_section_outside_the_compared_list_is_not_ignored(self):
        payload = b"\x80\x00\x04\x00\x80\x00\x04\x08"
        for name in (".ctors", ".rodata.str1.4", ".rodata.jtbl_80000400"):
            with self.subTest(section=name):
                elf = build([(".text", PLAIN), (name, payload, ALLOC, 4)])
                rom = image(elf, {".text": TEXT, name: 0x80001000})
                result = strict(elf, rom)
                self.assertNotEqual(result.verdict, "exact")
                self.assertEqual(rows(result)[name]["status"], "unverified")
                rom[0x1000:0x1008] = b"\x80\x00\x05\x00\x80\x00\x05\x08"          # declared, and different
                self.assertNotEqual(strict(elf, rom, bases={name: 0x80001000}).verdict, "exact")
        # two read-only sections are two sections: the second is not compared at the first's base
        elf = build([(".text", PLAIN), (".rodata", RO16), (".rdata", payload)])
        rom = image(elf, {".text": TEXT, ".rodata": 0x80001000, ".rdata": 0x80001010})
        result = strict(elf, rom, bases={".rodata": 0x80001000})
        self.assertNotEqual(result.verdict, "exact")
        self.assertEqual((rows(result)[".rodata"]["status"], rows(result)[".rdata"]["status"]),
                         ("equal", "unverified"))

    def test_sections_that_are_not_loaded_are_not_data(self):
        elf = build([(".text", PLAIN), (".comment", b"GCC: (GNU) 2.7.2\0", 0, 1), (".pdr", bytes(32), 0, 4),
                     (".reginfo", bytes(24), ALLOC, 4, 0x70000006), (".mdebug", bytes(16), 0, 4, 0x70000005)])
        result = strict(elf, image(elf, {".text": TEXT}))
        self.assertEqual((result.verdict, result.sections), ("exact", []))


class TargetDerivedAddressTest(unittest.TestCase):
    """(c) An address read from the image is an assumption, however well the bytes then agree."""

    def sdata_lit4(self):
        code, relocs = gp_program((LW, R_MIPS_GPREL16, ".sdata", 4), (LWC1, R_MIPS_LITERAL, ".lit4", 0))
        elf = build([(".text", code), (".sdata", words(1, 2)), (".lit4", b"\x3f\xc0\x00\x00"), (".sbss", 0x10)],
                    relocations={".text": relocs})
        bases = {".text": TEXT, ".sdata": SDATA, ".lit4": LIT4, ".sbss": SBSS}
        return elf, image(elf, bases, {"_gp": GP})

    def test_gp_relative_section_bases_are_assumptions_even_when_the_bytes_agree(self):
        elf, rom = self.sdata_lit4()
        result = strict(elf, rom, {"_gp": GP})
        self.assertTrue(result.text_equal)
        self.assertEqual(result.verdict, "assumed")
        self.assertEqual({(a.kind, a.name, a.address) for a in result.assumptions},
                         {("section", ".sdata", SDATA), ("section", ".lit4", LIT4)})
        for name in (".sdata", ".lit4"):
            self.assertEqual((rows(result)[name]["status"], rows(result)[name]["base_assumed"]), ("equal", True))
        declared = strict(elf, rom, {"_gp": GP}, {".sdata": SDATA, ".lit4": LIT4})
        self.assertEqual(declared.verdict, "exact", declared.notes)

    def test_a_wrong_gp_is_falsified_by_the_data_not_absorbed(self):
        """A wrong _gp shifts every base inferred through it; the code then agrees with itself, the
        data at the shifted bases does not."""
        elf, rom = self.sdata_lit4()
        shifted = strict(elf, rom, {"_gp": GP + 0x10})
        self.assertTrue(shifted.text_equal)
        self.assertEqual(shifted.verdict, "mismatch")
        declared = strict(elf, rom, {"_gp": GP + 0x10}, {".sdata": SDATA, ".lit4": LIT4})
        self.assertEqual((declared.verdict, declared.text_equal), ("mismatch", False))

    def test_a_wrong_gp_over_bss_only_stays_an_assumption(self):
        code, relocs = gp_program((LW, R_MIPS_GPREL16, ".sbss", 8))
        elf = build([(".text", code), (".sbss", 0x10)], relocations={".text": relocs})
        rom = image(elf, {".text": TEXT, ".sbss": SBSS}, {"_gp": GP})
        result = strict(elf, rom, {"_gp": GP + 0x10})
        self.assertEqual(result.verdict, "assumed")
        self.assertEqual([(a.name, a.address) for a in result.assumptions], [(".sbss", SBSS + 0x10)])
        self.assertEqual(transplant.corroborated_sections(result, elf), {})

    def test_rodata_found_by_searching_the_image_is_not_a_base(self):
        elf = build([(".text", PLAIN), (".rodata", RO16)])
        rom = image(elf, {".text": TEXT, ".rodata": 0x80002000})
        self.assertEqual(cmp.section_candidates(elf, ".rodata", bytes(rom), 0, len(rom), lambda off: off + BASE),
                         [0x80002000])                       # exploration would find it there
        result = strict(elf, rom)
        self.assertEqual((result.verdict, result.assumptions), ("assumed", []))
        self.assertEqual(rows(result)[".rodata"]["status"], "unverified")

    def test_a_lone_literal_or_bss_never_confirms_its_own_address(self):
        code, relocs = gp_program((LWC1, R_MIPS_LITERAL, ".lit4", 0), (LW, R_MIPS_GPREL16, ".sbss", 0))
        elf = build([(".text", code), (".lit4", b"\x3f\xc0\x00\x00"), (".sbss", 0x10)], relocations={".text": relocs})
        rom = image(elf, {".text": TEXT, ".lit4": LIT4, ".sbss": SBSS}, {"_gp": GP})
        rom[SBSS - BASE:SBSS - BASE + 0x10] = b"\x55" * 0x10
        result = strict(elf, rom, {"_gp": GP})
        self.assertEqual(result.verdict, "assumed")
        self.assertEqual(transplant.corroborated_sections(result, elf), {})


class BssTest(unittest.TestCase):
    """(d) .bss/.sbss hold no bytes in the object: nothing is compared or claimed there."""

    def test_bss_is_never_compared_whatever_the_image_holds(self):
        elf = build([(".text", PLAIN), (".bss", 0x40), (".sbss", 0x10)])
        bases = {".bss": 0x80003000, ".sbss": SBSS}
        for fill in (0x55, 0):
            with self.subTest(fill=fill):
                rom = image(elf, {".text": TEXT})
                rom[0x3000:0x3040] = bytes([fill]) * 0x40
                rom[SBSS - BASE:SBSS - BASE + 0x10] = bytes([fill]) * 0x10
                result = strict(elf, rom, bases=bases)
                self.assertEqual((result.verdict, result.sections), ("exact", []))

    def test_a_bss_address_read_from_code_stays_an_assumption(self):
        code = words(i_type(LUI, 0, 2, 0), i_type(LW, 2, 2, 0x10), JR_RA, NOP)
        elf = build([(".text", code), (".bss", 0x40)],
                    relocations={".text": [(0, R_MIPS_HI16, ".bss"), (4, R_MIPS_LO16, ".bss")]})
        rom = image(elf, {".text": TEXT, ".bss": 0x80003000})
        rom[0x3000:0x3040] = bytes(range(1, 0x41))                        # whatever the image holds there
        result = strict(elf, rom)
        self.assertEqual(result.verdict, "assumed")
        self.assertEqual([(a.kind, a.name, a.address) for a in result.assumptions], [("section", ".bss", 0x80003000)])
        self.assertNotIn(".bss", rows(result))
        self.assertEqual(transplant.corroborated_sections(result, elf), {})

    def test_small_common_symbols_are_external_not_bss_content(self):
        code, relocs = gp_program((LW, R_MIPS_GPREL16, "g_small", 0))
        elf = build([(".text", code)], [("g_small", 4, GLOBAL, OBJECT, SCOMMON)], {".text": relocs})
        rom = image(elf, {".text": TEXT}, {"_gp": GP, "g_small": 0x80001790})
        result = strict(elf, rom, {"_gp": GP})
        self.assertEqual(result.verdict, "assumed")
        self.assertEqual([(a.kind, a.name, a.address) for a in result.assumptions],
                         [("symbol", "g_small", 0x80001790)])
        self.assertEqual(strict(elf, rom, {"_gp": GP, "g_small": 0x80001790}).verdict, "exact")


class MergeTest(unittest.TestCase):
    """(e) Sections that lie next to each other are not one section, and two units are not one."""

    def test_adjacent_literal_pools_are_not_one_pool(self):
        code, relocs = gp_program((LWC1, R_MIPS_LITERAL, ".lit4", 0))
        elf = build([(".text", code), (".lit4", b"\x3f\xc0\x00\x00"), (".lit8", bytes.fromhex("400921fb54442d18"))],
                    relocations={".text": relocs})
        rom = image(elf, {".text": TEXT, ".lit4": LIT4, ".lit8": LIT8}, {"_gp": GP})
        result = strict(elf, rom, {"_gp": GP})
        self.assertEqual(result.verdict, "assumed")
        self.assertEqual([a.name for a in result.assumptions], [".lit4"])
        self.assertEqual((rows(result)[".lit4"]["status"], rows(result)[".lit8"]["status"]), ("equal", "unverified"))
        declared = strict(elf, rom, {"_gp": GP}, {".lit4": LIT4})
        self.assertEqual((declared.verdict, declared.assumptions), ("assumed", []))
        self.assertEqual(strict(elf, rom, {"_gp": GP}, {".lit4": LIT4, ".lit8": LIT8}).verdict, "exact")

    def test_small_data_is_not_compared_at_the_data_base(self):
        elf = build([(".text", PLAIN), (".data", words(1, 2)), (".sdata", words(3))])
        rom = image(elf, {".text": TEXT, ".data": 0x80001000, ".sdata": 0x80001008})
        result = strict(elf, rom, bases={".data": 0x80001000})
        self.assertEqual(result.verdict, "assumed")
        self.assertEqual((rows(result)[".data"]["status"], rows(result)[".sdata"]["status"]), ("equal", "unverified"))
        # only .data in the object: a declared .sdata base is not its base
        only = build([(".text", PLAIN), (".data", words(1, 2))])
        result = strict(only, image(only, {".text": TEXT, ".data": 0x80001000}), bases={".sdata": 0x80001000})
        self.assertEqual(result.verdict, "assumed")
        self.assertEqual([(row["section"], row["status"]) for row in result.sections], [(".data", "unverified")])

    def test_one_unit_yields_one_rodata_piece(self):
        """A chain is evidence because each piece's address comes from a different unit's code; one
        unit's two read-only sections must not chain with each other."""
        elf = build([(".text", PLAIN), (".rodata", RO16[:8]), (".rdata", RO16[8:])])
        rom = image(elf, {".text": TEXT, ".rodata": 0x80001000, ".rdata": 0x80001008})
        result = strict(elf, rom, bases={".rodata": 0x80001000, ".rdata": 0x80001008})
        self.assertNotEqual(result.verdict, "exact")
        row = {"verdict": transplant.ASSUMED_VERDICT, "awaiting": {".rodata": "0x80001000"}, "target_vram": TEXT,
               "size": len(PLAIN), "strict": result.as_dict(), "_object": elf,
               "_elf_spec": (None, SymbolMap(), {}, None), "notes": []}

        class Target:
            def __init__(self, data):
                self.rom = data

            @staticmethod
            def vram_to_rom(address):
                return address - BASE

        report = transplant.settle_rodata([row], Target(bytes(rom)), None)
        self.assertEqual((report["pieces"], report["chains"], report["promoted"]), (1, 0, 0))
        self.assertEqual(row["verdict"], transplant.ASSUMED_VERDICT)


if __name__ == "__main__":
    unittest.main()
