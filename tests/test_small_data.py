"""Small data (-G): GP-relative relocations, _gp recovery, the small-data lanes (round 12)."""
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from n64dk.elf import R_MIPS_GPREL16, R_MIPS_LITERAL, Elf

LUI, ADDIU, LW, SW, LWC1, JR_RA, NOP = 15, 9, 35, 43, 49, 0x03E00008, 0


def i_type(op: int, rs: int, rt: int, imm: int) -> int:
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def gp_object(code: bytes, relocations, gp0: int = 0x4000, sbss: int = 0x20, lit4: bytes = b"\x3f\xc0\0\0") -> bytes:
    """A big-endian MIPS relocatable: .text, .sbss (NOBITS), .lit4, .reginfo (GP0), .rel.text,
    .symtab (null, .sbss and .lit4 section symbols, global function f, external small_ext),
    .strtab, .shstrtab.  ``relocations``: (offset, type, symbol index)."""
    shstr = b"\0.text\0.sbss\0.lit4\0.reginfo\0.rel.text\0.symtab\0.strtab\0.shstrtab\0"
    name = {n: shstr.index(n.encode() + b"\0") for n in
            (".text", ".sbss", ".lit4", ".reginfo", ".rel.text", ".symtab", ".strtab", ".shstrtab")}
    strtab = b"\0f\0small_ext\0"
    reginfo = bytes(20) + struct.pack(">i", gp0)
    symtab = (bytes(16)
              + struct.pack(">IIIBBH", 0, 0, 0, 0x03, 0, 2)              # 1: .sbss section symbol (local)
              + struct.pack(">IIIBBH", 0, 0, 0, 0x03, 0, 3)              # 2: .lit4 section symbol (local)
              + struct.pack(">IIIBBH", 1, 0, len(code), 0x12, 0, 1)      # 3: f, global function in .text
              + struct.pack(">IIIBBH", 3, 0, 0, 0x10, 0, 0))             # 4: small_ext, global undefined
    rel = b"".join(struct.pack(">II", off, (sidx << 8) | kind) for off, kind, sidx in relocations)
    body = [code, lit4, reginfo, rel, symtab, strtab, shstr]
    offsets, position = [], 52
    for part in body:
        offsets.append(position)
        position += len(part)
    sh = lambda n, t, flags, off, size, link=0, info=0, align=4, ent=0: struct.pack(  # noqa: E731
        ">10I", name[n], t, flags, 0, off, size, link, info, align, ent)
    sections = [bytes(40),
                sh(".text", 1, 6, offsets[0], len(code)),
                sh(".sbss", 8, 0x10000003, offsets[1], sbss),
                sh(".lit4", 1, 0x10000002, offsets[1], len(lit4)),
                sh(".reginfo", 0x70000006, 2, offsets[2], len(reginfo)),
                sh(".rel.text", 9, 0, offsets[3], len(rel), 6, 1, 4, 8),
                sh(".symtab", 2, 0, offsets[4], len(symtab), 7, 3, 4, 16),
                sh(".strtab", 3, 0, offsets[5], len(strtab), 0, 0, 1),
                sh(".shstrtab", 3, 0, offsets[6], len(shstr), 0, 0, 1)]
    header = (b"\x7fELF\x01\x02\x01" + bytes(9)
              + struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, position, 0, 52, 0, 0, 40, len(sections), 8))
    return header + b"".join(body) + b"".join(sections)


def sample(gp0: int = 0x4000):
    """lw v0,%gp_rel(.sbss+0x10)(gp); sw v0,%gp_rel(small_ext+4)(gp); lwc1 f0,%lit4(.lit4)(gp);
    jr ra; nop — fields as a GP0-assuming assembler writes them (local: offset - GP0)."""
    code = struct.pack(">5I", i_type(LW, 28, 2, 0x10 - gp0), i_type(SW, 28, 2, 4), i_type(LWC1, 28, 0, 0 - gp0),
                       JR_RA, NOP)
    relocs = [(0, R_MIPS_GPREL16, 1), (4, R_MIPS_GPREL16, 4), (8, R_MIPS_LITERAL, 2)]
    return Elf(gp_object(code, relocs, gp0)), code


class RelocateTest(unittest.TestCase):
    GP = 0x80085F60
    BASES = {".text": 0x80001000, ".sbss": 0x80086000, ".lit4": 0x80085F00}

    def fields(self, text: bytes) -> list[int]:
        return [(w & 0xFFFF) - (0x10000 if w & 0x8000 else 0) for w in struct.unpack(">5I", text)[:3]]

    def test_local_external_and_literal(self):
        for gp0 in (0x4000, 0):                       # KMC as 2.6 writes 0x4000, GNU as 2.9 writes 0
            elf, _code = sample(gp0)
            self.assertEqual(elf.gp0(), gp0)
            text = elf.relocate(".text", self.BASES, {"_gp": self.GP, "small_ext": 0x80085F80})
            self.assertEqual(self.fields(text), [0x80086010 - self.GP, 0x80085F84 - self.GP, 0x80085F00 - self.GP])
            # opcode and registers untouched
            for got, was in zip(struct.unpack(">5I", text), struct.unpack(">5I", _code)):
                self.assertEqual(got & 0xFFFF0000, was & 0xFFFF0000)

    def test_without_gp_is_an_error_not_a_guess(self):
        elf, _ = sample()
        with self.assertRaises(KeyError):
            elf.relocate(".text", self.BASES, {"small_ext": 0x80085F80})

    def test_out_of_range_is_an_error_not_truncated(self):
        elf, _ = sample()
        with self.assertRaisesRegex(ValueError, "out of range"):
            elf.relocate(".text", {**self.BASES, ".sbss": self.GP + 0x9000}, {"_gp": self.GP, "small_ext": 0x80085F80})

    def test_inference_reads_sections_and_symbols_through_gp(self):
        from n64dk.match import compare as cmp
        from n64dk.match.symbols import SymbolMap
        elf, _ = sample()
        linked = elf.relocate(".text", self.BASES, {"_gp": self.GP, "small_ext": 0x80085F80})
        syms = SymbolMap({"_gp": self.GP}, infer_names=False)
        bases, found = cmp.infer_from_rom(elf, linked, 0x80001000, {".text": 0x80001000}, syms)
        self.assertEqual(bases.get(".sbss"), 0x80086000)
        self.assertEqual(bases.get(".lit4"), 0x80085F00)
        self.assertEqual(found.get("small_ext"), 0x80085F80)
        # without a declared _gp nothing is inferred from GP-relative fields
        bases, found = cmp.infer_from_rom(elf, linked, 0x80001000, {".text": 0x80001000},
                                          SymbolMap({}, infer_names=False))
        self.assertNotIn(".sbss", bases)
        self.assertNotIn("small_ext", found)

    def test_masks_keep_opcode_and_registers(self):
        from n64dk.match import compare as cmp
        elf, _ = sample()
        types = {r.offset: r.type for r in cmp.relocations(elf, ".text")}
        self.assertEqual(types, {0: R_MIPS_GPREL16, 4: R_MIPS_GPREL16, 8: R_MIPS_LITERAL})
        self.assertEqual([r.local for r in cmp.relocations(elf, ".text")], [True, False, True])

    def test_strict_verification_proves_with_declared_gp_and_assumes_without(self):
        from n64dk.match import verify
        from n64dk.match.symbols import SymbolMap
        elf, _ = sample()
        linked = elf.relocate(".text", self.BASES, {"_gp": self.GP, "small_ext": 0x80085F80})
        rom = bytearray(0x86000)
        rom[0x1000:0x1000 + len(linked)] = linked
        rom[0x85F00:0x85F04] = b"\x3f\xc0\0\0"                                      # the .lit4 literal
        rom = bytes(rom)
        declared = SymbolMap({"_gp": self.GP, "small_ext": 0x80085F80}, infer_names=False)
        to_rom = lambda vram: vram - 0x80000000                                       # noqa: E731
        result = verify.strict_compare(elf, rom, to_rom, 0x80001000, len(linked), declared,
                                       {".sbss": 0x80086000, ".lit4": 0x80085F00}, {}, None)
        self.assertTrue(result.text_equal)
        self.assertEqual(result.verdict, "exact", result.notes)
        result = verify.strict_compare(elf, rom, to_rom, 0x80001000, len(linked),
                                       SymbolMap({"small_ext": 0x80085F80}, infer_names=False),
                                       {".sbss": 0x80086000, ".lit4": 0x80085F00}, {}, None)
        self.assertNotEqual(result.verdict, "exact")


class SmallDataVerificationTest(unittest.TestCase):
    """Adversarial: the small-data sections are held to the same rules as .rodata/.data."""
    GP = RelocateTest.GP
    BASES = RelocateTest.BASES

    def verdict(self, literal: bytes):
        from n64dk.match import verify
        from n64dk.match.symbols import SymbolMap
        elf, _ = sample()
        linked = elf.relocate(".text", self.BASES, {"_gp": self.GP, "small_ext": 0x80085F80})
        rom = bytearray(0x87000)
        rom[0x1000:0x1000 + len(linked)] = linked
        rom[0x85F00:0x85F04] = literal
        rom[0x86000:0x86020] = b"\x55" * 0x20                       # whatever the ROM holds where .sbss is
        declared = SymbolMap({"_gp": self.GP, "small_ext": 0x80085F80}, infer_names=False)
        return verify.strict_compare(elf, bytes(rom), lambda v: v - 0x80000000, 0x80001000, len(linked), declared,
                                     {".sbss": 0x80086000, ".lit4": 0x80085F00}, {}, None)

    def test_a_wrong_literal_pool_byte_is_a_mismatch(self):
        self.assertEqual(self.verdict(b"\x3f\xc0\0\0").verdict, "exact")
        self.assertEqual(self.verdict(b"\x40\xc0\0\0").verdict, "mismatch")     # a significant byte
        # the object's trailing zeros differ: never forgiven (unverified, not exact)
        self.assertEqual(self.verdict(b"\x3f\xc0\0\1").verdict, "assumed")

    def test_bss_content_is_never_claimed_or_compared(self):
        result = self.verdict(b"\x3f\xc0\0\0")
        self.assertNotIn(".sbss", [row.get("section") or row.get("name") for row in result.sections])


class GpRecoveryTest(unittest.TestCase):
    def image(self, *pairs, uses=20):
        words = []
        for value in pairs:
            hi, lo = (value + 0x8000) >> 16 & 0xFFFF, value & 0xFFFF
            words += [i_type(LUI, 0, 28, hi), i_type(ADDIU, 28, 28, lo)]
        words += [i_type(LW, 28, 2, 8)] * uses + [JR_RA, NOP]
        data = struct.pack(f">{len(words)}I", *words)
        return data, 0x80000400, [("start", 0x80000400, len(data))]

    def test_agreeing_sites_give_a_value(self):
        from n64dk.analysis import gp
        record = gp.find({"main": self.image(0x80085F60, 0x80085F60)})
        self.assertEqual(record["value"], "0x80085F60")
        self.assertEqual(record["confidence"], "high")
        self.assertIn("_gp = 0x80085F60;", gp.symbol_line(record))

    def test_negative_value_low_half(self):
        from n64dk.analysis import gp
        self.assertEqual(gp.find({"main": self.image(0x800B8500)})["value"], "0x800B8500")

    def test_disagreeing_cleared_absent_and_weak_evidence_give_none(self):
        from n64dk.analysis import gp
        self.assertIsNone(gp.find({"main": self.image(0x80085F60, 0x80090000)})["value"])
        self.assertIsNone(gp.find({"main": self.image(0)})["value"])                 # Namco-style clear
        self.assertIsNone(gp.find({"main": self.image()})["value"])                  # uses, never set
        weak = gp.find({"main": self.image(0x80085F60, uses=2)})
        self.assertEqual(weak["confidence"], "low")
        self.assertIsNone(gp.symbol_line(weak))

    def test_an_overlay_using_gp_inherits(self):
        from n64dk.analysis import gp
        overlay = self.image(uses=5)
        record = gp.find({"main": self.image(0x80085F60, 0x80085F60), "seg": overlay})
        self.assertEqual(record["inherited"], ["seg"])

    # round-12 review D: fail-safe setting sites
    USE = i_type(LW, 28, 2, 8)

    def rows(self, words, *sizes):
        """``words`` as one image; ``sizes`` (bytes) split it into catalogued functions."""
        data = struct.pack(f">{len(words)}I", *words)
        rows, vram = [], 0x80000400
        for k, size in enumerate(sizes or (len(data),)):
            rows.append((f"f{k}", vram, size))
            vram += size
        return data, 0x80000400, rows

    def test_a_lone_lui_sets_a_value_whose_low_half_is_zero(self):
        from n64dk.analysis import gp
        record = gp.find({"main": self.rows([i_type(LUI, 0, 28, 0x8009)] + [self.USE] * 20 + [JR_RA, NOP])})
        self.assertEqual(record["value"], "0x80090000")
        self.assertEqual(record["confidence"], "low")                  # one site, its uses in one function
        # its low half may be in the next row of a split catalogue: a lui at a function's end is no site
        tail = gp.find({"main": self.rows([self.USE] * 20 + [i_type(LUI, 0, 28, 0x8009), NOP])})
        self.assertIsNone(tail["value"])
        # $gp written again later in the function: the lui alone is not its value
        later = [i_type(LUI, 0, 28, 0x8009)] + [self.USE] * 20 + [i_type(ADDIU, 28, 28, 0x10), JR_RA, NOP]
        self.assertIsNone(gp.find({"main": self.rows(later)})["value"])
        # a lone lui outside RDRAM (a float such as 0x3C1C0000 in data) is no site
        self.assertEqual(gp.find({"main": self.rows([i_type(LUI, 0, 28, 0x3F80)] + [self.USE] * 20)})["sites"], {})

    def test_a_rewrite_between_lui_and_addiu_breaks_the_pair(self):
        from n64dk.analysis import gp
        move_gp_a0 = (4 << 21) | (28 << 11) | 0x25                     # or gp, a0, zero
        words = [i_type(LUI, 0, 28, 0x8009), move_gp_a0, i_type(ADDIU, 28, 28, 0x1234)] + [self.USE] * 20
        record = gp.find({"main": self.rows(words)})
        self.assertIsNone(record["value"])
        self.assertEqual(record["sites"], {})
        self.assertEqual(gp.scan(*self.rows([i_type(ADDIU, 28, 28, 0x10)]))["uses"], 0)   # a write, not a use

    def test_a_pair_split_over_two_functions_is_no_site(self):
        from n64dk.analysis import gp
        words = [self.USE, i_type(LUI, 0, 28, 0x8009), i_type(ADDIU, 28, 28, 0x1234), self.USE]
        self.assertEqual(gp.find({"main": self.rows(words, 8, 8)})["sites"], {})
        self.assertEqual(len(gp.find({"main": self.rows(words, 16)})["sites"]), 1)

    def test_data_decoded_as_a_pair_outside_rdram_gives_no_value(self):
        from n64dk.analysis import gp
        junk = [i_type(LUI, 0, 28, 0x3F80), i_type(ADDIU, 28, 28, 0x10)] + [self.USE] * 20
        record = gp.find({"main": self.rows(junk, 8, 20, 20, 20, 20)})
        self.assertIsNone(record["value"])
        self.assertIn("outside KSEG0 RDRAM", record["reason"])

    def test_one_site_needs_its_uses_in_several_functions(self):
        from n64dk.analysis import gp
        words = [i_type(LUI, 0, 28, 0x8009), i_type(ADDIU, 28, 28, -0xA0)] + [self.USE] * 20
        self.assertEqual(gp.find({"main": self.rows(words)})["confidence"], "low")          # 20 uses, 1 function
        spread = gp.find({"main": self.rows(words, 28, 20, 20, 20)})                        # 4 functions
        self.assertEqual(spread["value"], "0x8008FF60")
        self.assertEqual(spread["confidence"], "high")


class LaneTest(unittest.TestCase):
    def test_threshold_parsing(self):
        from n64dk.match.pipeline import small_data_threshold
        self.assertEqual(small_data_threshold(["-O2", "-G8"]), 8)
        self.assertEqual(small_data_threshold(["-G", "4"]), 4)
        self.assertEqual(small_data_threshold(["-O2", "-G0"]), 0)
        self.assertEqual(small_data_threshold(["-O2"]), 0)

    def test_established_lanes_assemble_at_g0_whatever_the_compiler(self):
        from n64dk.match.pipeline import _as_flags
        for lane in ("kmc+as", "kmc+builtin", "kmc+atload", "kmc+hook", "as", ""):
            self.assertEqual(_as_flags(["-O2", "-G8"], lane), ["-mips3", "-G0"])
        self.assertEqual(_as_flags(["-O2", "-G8"], "kmc+asg"), ["-mips3", "-G8"])
        self.assertEqual(_as_flags(["-O2", "-G8"], "kmc+gas"), ["-mips3", "-G8"])

    def test_lanes_are_registered(self):
        from n64dk.match import components
        from n64dk.match.pipeline import PIPELINES
        for lane in ("kmc+asg", "kmc+gas"):
            self.assertIn(lane, PIPELINES)
            self.assertIn(lane, components.RECIPES)

    def test_gp_rel_operands_for_m2c(self):
        from n64dk.match import m2c_bridge
        code = struct.pack(">4I", i_type(LW, 28, 2, -0x10), i_type(ADDIU, 28, 4, 8), JR_RA, NOP)
        names = {0x80085F50: "gCounter"}
        text = m2c_bridge.emit_asm("f", code, 0x80001000, names, macros=False, gp=0x80085F60)
        self.assertIn("%gp_rel(gCounter)($gp)", text)
        self.assertIn("%gp_rel(D_80085F68)", text)
        plain = m2c_bridge.emit_asm("f", code, 0x80001000, names, macros=False)
        self.assertNotIn("%gp_rel", plain)


@unittest.skipUnless(os.environ.get("N64DK_TOOLCHAIN") and
                     Path(os.environ.get("N64DK_TOOLCHAIN", "/nonexistent"), "kmc-gcc-2.7.2", "as").is_file(),
                     "KMC toolchain not installed")
class KmcAssemblerTest(unittest.TestCase):
    """The real KMC as 2.6 at -G8: .comm symbols and float literals through $gp, relocated
    by the kit to the fields a link at a known _gp gives."""
    SOURCE = "\n".join([
        "\t.comm\tg_half,2,2", "\t.text", "\t.globl\tf", "\t.ent\tf", "f:",
        "\tsh\t$3,g_half", "\tli.s\t$f0,1.5", "\tj\t$31", "\t.end\tf", ""])

    def test_relocated_fields(self):
        kmc = Path(os.environ["N64DK_TOOLCHAIN"], "kmc-gcc-2.7.2")
        work = Path(tempfile.mkdtemp(prefix="n64dk-test-gp-"))
        try:
            (work / "f.s").write_text(self.SOURCE)
            subprocess.run([kmc / "as", "-mips3", "-G8", "-o", work / "f.o", work / "f.s"], check=True,
                           capture_output=True, timeout=60)
            elf = Elf.load(work / "f.o")
            kinds = sorted(r[1] for r in elf.relocations(elf.section(".text")))
            self.assertIn(R_MIPS_GPREL16, kinds)
            self.assertIn(R_MIPS_LITERAL, kinds)
            gp = 0x80090000
            text = elf.relocate(".text", {".text": 0x80001000, ".lit4": 0x80089000}, {"_gp": gp, "g_half": 0x8008A000})
            words = struct.unpack(f">{len(text) // 4}I", text)
            fields = {w >> 26: (w & 0xFFFF) - (0x10000 if w & 0x8000 else 0) for w in words if (w >> 21) & 31 == 28}
            self.assertEqual(fields[41], 0x8008A000 - gp)        # sh: the .comm symbol
            self.assertEqual(fields[49], 0x80089000 - gp)        # lwc1: the .lit4 literal
        finally:
            shutil.rmtree(work, ignore_errors=True)


class PlainDictTest(unittest.TestCase):
    def test_inference_with_a_plain_dict_symbol_table(self):
        """The donor stage passes ``{}`` (round-12 review: ``.resolve`` broke it on every title)."""
        from n64dk.match import compare as cmp
        elf, _ = sample()
        linked = elf.relocate(".text", RelocateTest.BASES, {"_gp": RelocateTest.GP, "small_ext": 0x80085F80})
        bases, found = cmp.infer_from_rom(elf, linked, 0x80001000, {}, {})
        self.assertNotIn(".sbss", bases)                      # no _gp, nothing inferred through it
        bases, found = cmp.infer_from_rom(elf, linked, 0x80001000, {}, {"_gp": RelocateTest.GP})
        self.assertEqual(bases.get(".sbss"), 0x80086000)


class GpNamesNoObjectTest(unittest.TestCase):
    def test_a_global_at_the_gp_address_is_assumed_not_a_conflict(self):
        """_gp marks an address; a variable placed exactly there is not 'declared as _gp'."""
        from n64dk.match import verify
        from n64dk.match.symbols import SymbolMap
        gp = RelocateTest.GP
        elf, _ = sample()
        linked = elf.relocate(".text", RelocateTest.BASES, {"_gp": gp, "small_ext": gp})
        rom = bytearray(0x87000)
        rom[0x1000:0x1000 + len(linked)] = linked
        rom[0x85F00:0x85F04] = b"\x3f\xc0\0\0"
        result = verify.strict_compare(elf, bytes(rom), lambda v: v - 0x80000000, 0x80001000, len(linked),
                                       SymbolMap({"_gp": gp}, infer_names=False),
                                       {".sbss": 0x80086000, ".lit4": 0x80085F00}, {}, None)
        self.assertEqual(result.verdict, "assumed", result.conflicts)
        self.assertFalse(result.conflicts)


if __name__ == "__main__":
    unittest.main()
