"""RSP round trip (wave 4b, item 13): hand-encoded words only, no microcode or ROM bytes.

Expected words are packed here from the public field layouts (not from the module's tables): the
opcode numbers are the RSP manual's octal lists (00 LBV .. 13 LTV; 00 VMULF .. 67 VNOP) converted by
hand.  The CPU fixture is a separate class: CPU-only words must stay ``.word``.
"""
import contextlib
import io
import json
import random
import struct
import tempfile
import unittest
from pathlib import Path

from n64dk.analysis import rspasm, rspid
from n64dk.sdk import fingerprint


def r(fn, rs=0, rt=0, rd=0, sa=0):
    return rs << 21 | rt << 16 | rd << 11 | sa << 6 | fn


def i(op, rs, rt, imm):
    return op << 26 | rs << 21 | rt << 16 | (imm & 0xFFFF)


def vec(fn, vd, vs, vt, e=0):
    return 0x12 << 26 | 1 << 25 | e << 21 | vt << 16 | vs << 11 | vd << 6 | fn


def vmem(op, code, vt, base, e, off7):
    return op << 26 | base << 21 | vt << 16 | code << 11 | e << 7 | (off7 & 0x7F)


# (word, pc, text): one or more per format class
CASES = [
    (0x00000000, 0, "nop"),
    (r(0x21, rs=1, rt=2, rd=3), 0, "addu $3, $1, $2"),
    (r(0x2A, rs=4, rt=5, rd=6), 0, "slt $6, $4, $5"),
    (r(0x00, rt=2, rd=3, sa=4), 0, "sll $3, $2, 4"),
    (r(0x03, rt=2, rd=3, sa=31), 0, "sra $3, $2, 31"),
    (r(0x07, rs=4, rt=2, rd=3), 0, "srav $3, $2, $4"),
    (r(0x08, rs=31), 0, "jr $31"),
    (r(0x09, rs=5, rd=31), 0, "jalr $31, $5"),
    (0x0000000D, 0, "break 0x0"),
    (0x12345 << 6 | 0x0D, 0, "break 0x12345"),
    (i(0x09, 1, 2, -16), 0, "addiu $2, $1, -0x10"),
    (i(0x0B, 1, 2, 0x7FFF), 0, "sltiu $2, $1, 0x7FFF"),
    (i(0x0D, 1, 2, 0xFFFF), 0, "ori $2, $1, 0xFFFF"),
    (i(0x0F, 0, 4, 0x8000), 0, "lui $4, 0x8000"),
    (i(0x23, 29, 31, 0x14), 0, "lw $31, 0x14($29)"),
    (i(0x25, 1, 2, -2), 0, "lhu $2, -0x2($1)"),
    (i(0x28, 1, 2, -1), 0, "sb $2, -0x1($1)"),
    (i(0x04, 1, 2, -4), 0x100, "beq $1, $2, 0x0F4"),
    (i(0x05, 3, 0, 3), 0x080, "bne $3, $0, 0x090"),
    (i(0x06, 5, 0, 2), 0x000, "blez $5, 0x00C"),
    (i(0x07, 6, 0, 0), 0xFFC, "bgtz $6, 0x1000"),
    (0x01 << 26 | 5 << 21 | 0x11 << 16 | 1, 0x010, "bgezal $5, 0x018"),
    (0x01 << 26 | 7 << 21 | 0x00 << 16 | 0xFFFF, 0x020, "bltz $7, 0x020"),
    (0x02 << 26 | 0x04001080 >> 2, 0, "j 0x04001080"),
    (0x03 << 26 | 0xA8, 0, "jal 0x000002A0"),
    (0x40082000, 0, "mfc0 $8, $c4"),
    (0x40890000, 0, "mtc0 $9, $c0"),
    (0x48011300, 0, "mfc2 $1, $v2[6]"),
    (0x48800000 | 5 << 16 | 4 << 11 | 8 << 7, 0, "mtc2 $5, $v4[8]"),
    (0x48400000 | 3 << 16 | 1 << 11, 0, "cfc2 $3, $vcc"),
    (0x48C00000, 0, "ctc2 $0, $vco"),
    (vec(0x00, 1, 2, 3, 8), 0, "vmulf $v1, $v2, $v3[0]"),
    (vec(0x10, 4, 5, 6), 0, "vadd $v4, $v5, $v6"),
    (vec(0x08, 1, 2, 3, 3), 0, "vmacf $v1, $v2, $v3[1q]"),
    (vec(0x2C, 7, 8, 9, 6), 0, "vxor $v7, $v8, $v9[2h]"),
    (vec(0x27, 1, 2, 3, 1), 0, "vmrg $v1, $v2, $v3[e1]"),
    (vec(0x15, 31, 30, 29, 15), 0, "vsubc $v31, $v30, $v29[7]"),
    (vec(0x30, 1, 3, 2, 11), 0, "vrcp $v1[3], $v2[3]"),
    (vec(0x33, 5, 0, 6, 8), 0, "vmov $v5[0], $v6[0]"),
    (0x4A000037, 0, "vnop"),
    (vec(0x1D, 1, 0, 0, 9), 0, "vsar $v1, $v0, $v0[1]"),
    (vmem(0x32, 0o04, 1, 2, 0, 1), 0, "lqv $v1[0], 0x10($2)"),
    (vmem(0x32, 0o03, 3, 4, 8, -1), 0, "ldv $v3[8], -0x8($4)"),
    (vmem(0x32, 0o00, 0, 0, 15, 5), 0, "lbv $v0[15], 0x5($0)"),
    (vmem(0x32, 0o13, 8, 1, 0, -64), 0, "ltv $v8[0], -0x400($1)"),
    (vmem(0x3A, 0o04, 31, 29, 0, 63), 0, "sqv $v31[0], 0x3F0($29)"),
    (vmem(0x3A, 0o12, 2, 3, 4, 2), 0, "swv $v2[4], 0x20($3)"),
    (vmem(0x3A, 0o01, 2, 3, 2, 3), 0, "ssv $v2[2], 0x6($3)"),
]

# the manual's octal opcode lists: name -> (code, offset scale in bytes)
LOADS = {"lbv": (0o00, 1), "lsv": (0o01, 2), "llv": (0o02, 4), "ldv": (0o03, 8), "lqv": (0o04, 16),
         "lrv": (0o05, 16), "lpv": (0o06, 8), "luv": (0o07, 8), "lhv": (0o10, 16), "lfv": (0o11, 16),
         "ltv": (0o13, 16)}
STORES = {"sbv": (0o00, 1), "ssv": (0o01, 2), "slv": (0o02, 4), "sdv": (0o03, 8), "sqv": (0o04, 16),
          "srv": (0o05, 16), "spv": (0o06, 8), "suv": (0o07, 8), "shv": (0o10, 16), "sfv": (0o11, 16),
          "swv": (0o12, 16), "stv": (0o13, 16)}
THREE = {"vmulf": 0o00, "vmulu": 0o01, "vmudl": 0o04, "vmudm": 0o05, "vmudn": 0o06, "vmudh": 0o07,
         "vmacf": 0o10, "vmacu": 0o11, "vmacq": 0o13, "vmadl": 0o14, "vmadm": 0o15, "vmadn": 0o16,
         "vmadh": 0o17, "vadd": 0o20, "vsub": 0o21, "vabs": 0o23, "vaddc": 0o24, "vsubc": 0o25,
         "vsar": 0o35, "vlt": 0o40, "veq": 0o41, "vne": 0o42, "vge": 0o43, "vcl": 0o44, "vch": 0o45,
         "vcr": 0o46, "vmrg": 0o47, "vand": 0o50, "vnand": 0o51, "vor": 0o52, "vnor": 0o53,
         "vxor": 0o54, "vnxor": 0o55}
SINGLE = {"vrcp": 0o60, "vrcpl": 0o61, "vrcph": 0o62, "vmov": 0o63, "vrsq": 0o64, "vrsql": 0o65,
          "vrsqh": 0o66}


class FormatTest(unittest.TestCase):
    def test_every_format_class(self):
        for word, pc, text in CASES:
            self.assertEqual(rspasm.decode(word, pc), text, f"{word:08X}")
            self.assertEqual(rspasm.encode(text, pc), word, text)

    def test_vector_loads_and_stores_scale_their_offsets(self):
        for op, table in ((0x32, LOADS), (0x3A, STORES)):
            for name, (code, scale) in table.items():
                word = vmem(op, code, 1, 3, 2, 1)
                self.assertEqual(rspasm.decode(word), f"{name} $v1[2], 0x{scale:X}($3)")
                word = vmem(op, code, 1, 3, 2, -1)
                self.assertEqual(rspasm.decode(word), f"{name} $v1[2], -0x{scale:X}($3)")

    def test_vector_compute_functions(self):
        for name, fn in THREE.items():
            self.assertEqual(rspasm.decode(vec(fn, 1, 2, 3, 4)), f"{name} $v1, $v2, $v3[0h]")
        for name, fn in SINGLE.items():
            self.assertEqual(rspasm.decode(vec(fn, 1, 7, 3, 12)), f"{name} $v1[7], $v3[4]")

    def test_every_element_name(self):
        names = ["", "[e1]", "[0q]", "[1q]", "[0h]", "[1h]", "[2h]", "[3h]"] + [f"[{n}]" for n in range(8)]
        for e, suffix in enumerate(names):
            word = vec(0x00, 1, 2, 3, e)
            self.assertEqual(rspasm.decode(word), f"vmulf $v1, $v2, $v3{suffix}")
            self.assertEqual(rspasm.encode(f"vmulf $v1, $v2, $v3[e{e}]"), word)

    def test_round_trip(self):
        words = [w for w, _, _ in CASES]
        # re-based at 0x080, every branch still decodes: its target moves with it, printed unmasked
        self.assertEqual(rspasm.round_trip(words, origin=0x080), (True, len(words), 0))
        text = "\n".join(rspasm.disassemble(words, 0x080))
        self.assertEqual(rspasm.assemble(".org 0x080\n" + text), struct.pack(f">{len(words)}I", *words))

    def test_branch_leaving_imem_is_still_exact(self):
        words = [i(0x04, 0, 0, -2), 0]
        self.assertEqual(rspasm.decode(words[0], 0), "beq $0, $0, -0x4")
        report = rspasm.block_report(struct.pack(">2I", *words), "text", origin=0)
        self.assertTrue(report["identical"])
        self.assertEqual(report["branch_targets_outside_imem"], 1)


class RawTest(unittest.TestCase):
    RAW = [
        (vmem(0x32, 0o12, 1, 2, 0, 0), "lwc2 code 0x0A"),            # lwv: no such load
        (vmem(0x32, 0o14, 1, 2, 0, 0), "lwc2 code 0x0C"),
        (vmem(0x3A, 0o17, 1, 2, 0, 0), "swc2 code 0x0F"),
        (vec(0x12, 1, 2, 3), "vector funct 0x12"),
        (vec(0x1E, 1, 2, 3), "vector funct 0x1E"),
        (vec(0x3F, 0, 0, 0), "vector funct 0x3F"),
        (0x48200000, "cop2 rs 0x01"),
        (0x40200000, "cop0 rs 0x01"),
        (0x40088000, "cop0 register 16"),
        (0x48401800, "cop2 control register 3"),
        (vec(0x30, 1, 8, 2, 8), "single-lane destination element 8"),
        (0x01 << 26 | 2 << 16, "regimm 0x02"),
        (r(0x00, rs=1, rt=2, rd=3, sa=4), "fields outside the canonical encoding"),
        (r(0x08, rs=31, rd=1), "fields outside the canonical encoding"),
        (0x4A000037 | 1 << 6, "fields outside the canonical encoding"),
        (0x48011301, "fields outside the canonical encoding"),
        (i(0x06, 5, 1, 2), "fields outside the canonical encoding"),
    ]

    def test_unknown_words_are_kept_as_word(self):
        for word, reason in self.RAW:
            self.assertEqual(rspasm.classify(word), (None, reason), f"{word:08X}")
            self.assertEqual(rspasm.disassemble([word]), [f".word 0x{word:08X}"])
        words = [w for w, _ in self.RAW]
        self.assertEqual(rspasm.round_trip(words), (True, 0, len(words)))

    def test_block_report_counts_raw_preservation(self):
        words = [vec(0x10, 1, 2, 3), vec(0x12, 1, 2, 3), 0, 0]
        report = rspasm.block_report(struct.pack(">4I", *words) + b"\x01\x02", "text", origin=0x80,
                                     with_listing=True)
        self.assertTrue(report["identical"])
        self.assertEqual((report["decoded_words"], report["raw_words"], report["tail_bytes"]), (3, 1, 2))
        self.assertEqual(report["raw_bytes"], 6)
        self.assertEqual(report["raw_preservation_share"], round(6 / 18, 4))
        self.assertEqual(report["raw_reasons"], {"vector funct 0x12": 1})
        self.assertEqual(report["vector_words"], 1)
        self.assertIn(".byte 0x01, 0x02", report["listing"])

    def test_dmem_is_raw(self):
        blob = b"RSP Gfx ucode\0" + bytes([1, 2, 3])
        report = rspasm.block_report(blob, "data", with_listing=True)
        self.assertTrue(report["identical"])
        self.assertEqual((report["decoded_words"], report["raw_words"], report["tail_bytes"]), (0, 4, 1))
        self.assertEqual(report["raw_preservation_share"], 1.0)
        self.assertEqual(rspasm.assemble(report["listing"]), blob)


class CpuFixtureTest(unittest.TestCase):
    """The CPU fixture, kept apart from the RSP cases: VR4300-only words never decode as RSP."""
    CPU_ONLY = {
        "daddu $2,$3,$4": r(0x2D, 3, 4, 2), "dsll $2,$2,4": r(0x38, rt=2, rd=2, sa=4),
        "mult $4,$5": r(0x18, 4, 5), "div $4,$5": r(0x1A, 4, 5), "mflo $2": r(0x12, rd=2),
        "mfhi $2": r(0x10, rd=2), "syscall": 0x0000000C, "sync": 0x0000000F, "teq $1,$2": r(0x34, 1, 2),
        "ld $31,0x18($29)": 0xDFBF0018, "sd $31,0x18($29)": 0xFFBF0018, "lwc1 $f4,0x10($29)": 0xC7A40010,
        "swc1 $f4,0x10($29)": 0xE7A40010, "ldc1 $f4,0x10($29)": 0xD7A40010, "add.s $f0,$f0,$f0": 0x46000000,
        "beql $0,$0,1": 0x50000001, "daddiu $0,$0,1": 0x64000001, "lwl $0,0($0)": 0x88000000,
        "lwu $0,0($0)": 0x9C000000, "ll $0,0($0)": 0xC0000000, "sc $0,0($0)": 0xE0000000,
        "cache 0,0($0)": 0xBC000000,
    }

    def test_cpu_only_words_are_raw(self):
        for name, word in self.CPU_ONLY.items():
            self.assertIsNone(rspasm.decode(word), name)
        words = list(self.CPU_ONLY.values())
        self.assertEqual(rspasm.round_trip(words), (True, 0, len(words)))

    def test_cpu_function_is_not_taken_for_vector_code(self):
        # addiu sp,-0x18; sd ra,0x10(sp); daddu v0,v1,a0; ld ra,0x10(sp); jr ra; addiu sp,0x18
        words = [0x27BDFFE8, 0xFFBF0010, 0x0064102D, 0xDFBF0010, 0x03E00008, 0x27BD0018]
        report = rspasm.block_report(struct.pack(">6I", *words), "text", origin=0)
        self.assertTrue(report["identical"])
        self.assertEqual((report["decoded_words"], report["raw_words"], report["vector_words"]), (3, 3, 0))
        self.assertEqual(rspasm.decode(words[0]), "addiu $29, $29, -0x18")   # shared scalar subset

    def test_only_rsp_primary_opcodes_decode(self):
        rng = random.Random(13)
        words = [rng.getrandbits(32) for _ in range(3000)] + [op << 26 for op in range(64)]
        for word in words:
            text = rspasm.decode(word)
            if text is not None:
                self.assertIn(word >> 26, fingerprint._RSP_OPS, text)
                self.assertEqual(rspasm.encode(text), word, text)
        self.assertTrue(rspasm.round_trip(words[:500], origin=0x080)[0])


class AssemblerTest(unittest.TestCase):
    def test_rejects(self):
        for line, pc in (("lqv $v1[0], 0x8($2)", 0), ("lqv $v1[0], 0x400($2)", 0), ("frob $1", 0),
                         ("addu $1, $2, $32", 0), ("beq $1, $2, 0x102", 0), ("beq $1, $2, 0x40000", 0),
                         ("vadd $v1[0], $v2, $v3", 0), ("vmulf $v1, $v2, $v3[8]", 0),
                         ("vrcp $v1[8], $v2[0]", 0), ("mfc0 $1, $c16", 0), ("cfc2 $1, $vcx", 0),
                         ("j 0x1002", 0), ("addiu $1, $1, 0x8000", 0), ("jr $1, $2", 0)):
            with self.assertRaises(rspasm.AsmError, msg=line):
                rspasm.encode(line, pc)
        with self.assertRaises(rspasm.AsmError):
            rspasm.assemble(".byte 1\nnop\n")
        with self.assertRaises(rspasm.AsmError):
            rspasm.assemble(".fill 4\n")

    def test_directives_and_org(self):
        out = rspasm.assemble(".org 0x080\n  beq $0, $0, 0x080  # loop\n  .word 0x1, 0xFFFFFFFF\n  .byte 0x7F\n")
        self.assertEqual(out, bytes.fromhex("1000FFFF00000001FFFFFFFF7F"))

    def test_origin_guess(self):
        def block(targets):
            words = [0x02 << 26 | t >> 2 for t in targets] + [0] * (16 - len(targets))
            return struct.pack(">16I", *words)
        self.assertEqual(rspasm.guess_origin(block([0xA0, 0xB0]))[0], 0x080)
        self.assertEqual(rspasm.guess_origin(block([0x200, 0x210]))[0], 0x1D8)
        self.assertEqual(rspasm.guess_origin(bytes(64))[0], 0)


BASE = 0x80000400                      # main image: ROM 0x1000 is VRAM 0x80000400
UCODE = (0x4A000010, 0xC8011000, 0x20010001, 0x08000400)   # vadd, llv, addi, j 0x1000
STRING = b"RSP Gfx ucode F3DEX       fifo 2.08  Yoshitaka Yasumoto 1999 Nintendo.\0"


def vram(rom: int) -> int:
    return BASE + rom - 0x1000


def synthetic(size: int = 0x6000) -> bytearray:
    """IMEM text 0x2000..0x2400, DMEM data 0x3000..0x3100 (version string at 0x3040), and a
    (text, data, -, -, data end) pointer table at 0x4000."""
    buf = bytearray(size)
    struct.pack_into(">256I", buf, 0x2000, *(UCODE[n % 4] for n in range(256)))
    buf[0x3000:0x3040] = b"\x11" * 0x40
    buf[0x3040:0x3040 + len(STRING)] = STRING
    struct.pack_into(">5I", buf, 0x4000, vram(0x2000), vram(0x3000), 0, 0, vram(0x3100))
    return buf


class ExtractionTest(unittest.TestCase):
    def test_rspid_blocks_round_trip(self):
        blocks, _ = rspasm.extract_blocks(bytes(synthetic()), rspid.Mapping(0x1000, BASE, 0x5000))
        self.assertEqual([(b["kind"], b["rom"], b["end"]) for b in blocks],
                         [("text", 0x2000, 0x2400), ("data", 0x3000, 0x3100)])
        self.assertEqual(rspasm.decode(UCODE[1]), "llv $v1[0], 0x0($0)")
        text = rspasm.block_report(blocks[0]["bytes"], "text")
        self.assertEqual((text["identical"], text["decoded_words"], text["raw_words"]), (True, 256, 0))
        data = rspasm.block_report(blocks[1]["bytes"], "data")
        self.assertEqual((data["identical"], data["raw_preservation_share"]), (True, 1.0))

    def _rom(self, root: Path) -> Path:
        rom = synthetic(0x101000)
        struct.pack_into(">3I", rom, 0, 0x80371240, 0, BASE)
        path = root / "baserom.z64"
        path.write_bytes(bytes(rom))
        return path

    def test_range_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._rom(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(rspasm.main([str(path), "0x2000", "0x2040"]), 0)
            out = printed.getvalue()
            self.assertEqual(rspasm.assemble(out), bytes(synthetic()[0x2000:0x2040]))
            summary = json.loads(out.splitlines()[-1][2:])
            self.assertEqual((summary["decoded_words"], summary["identical"]), (16, True))

    def test_census_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "synthetic"
            (project / "survey").mkdir(parents=True)
            self._rom(project)
            (project / "n64dk.toml").write_text('[images.main]\npath = "baserom.z64"\n')
            (project / "survey" / "layout.json").write_text(json.dumps(
                {"main": {"kind": "rom", "rom": "0x1000", "rom_end": "0x6000", "vram": f"0x{BASE:08X}"}}))
            out, listings = Path(tmp) / "census.json", Path(tmp) / "listings"
            with contextlib.redirect_stdout(io.StringIO()):
                code = rspasm.main(["census", "--out", str(out), "--listings", str(listings), str(project)])
            self.assertEqual(code, 0)
            summary = json.loads(out.read_text())["summary"]
            self.assertEqual((summary["titles"], summary["all_identical"]), (1, True))
            self.assertEqual((summary["text_decoded_words"], summary["data_raw_bytes"]), (256, 0x100))
            self.assertEqual(len(list(listings.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
