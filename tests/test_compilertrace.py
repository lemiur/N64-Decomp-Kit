"""Compiler traces: dump parsers, collection with an injected build, causal links, reduction."""
import struct
import tempfile
import unittest
from pathlib import Path

from n64dk.elf import Elf
from n64dk.match import compilertrace as ct
from n64dk.match.pipeline import BuildError

RTL = """
;; Function pp_add

(note 1 0 2 "" NOTE_INSN_DELETED)

(note 4 3 5 "probe.c" 3)

(insn 6 4 7 (set (reg/v:SI 77)
        (reg:SI 4 a0)) -1 (nil)
    (nil))

(note 7 6 8 "probe.c" 4)

;; a comment (with parentheses) between insns
(insn 9 7 10 (set (reg:SI 78)
        (plus:SI (reg/v:SI 77)
            (const_int 1))) -1 (nil)
    (nil))

(jump_insn 10 9 11 (return) -1 (nil)
    (nil))

(barrier 11 10 0)
"""
GREG = """
;; Function pp_add

(note 4 3 5 "probe.c" 3)

(insn 9 7 10 (set (reg:SI 2 v0)
        (plus:SI (reg:SI 4 a0)
            (const_int 1))) 33 (nil)
    (nil))

(jump_insn 10 9 11 (return) -1 (nil)
    (nil))
"""
LREG = ("Register 77 used 2 times across 3 insns in block 0; pointer.\n"
        "Register 78 used 1 times across 2 insns in block 0;\n\n"
        ";; Register 77 in 4.\n;; Register 78 in 2.\n")
ASM = ("\t.text\n\t.align\t2\n\t.globl\tpp_add\n\t.ent\tpp_add\npp_add:\n\t.frame\t$sp,0,$31\n"
       "\t.set\tnoreorder\n\t.loc\t1 4\n\tj\t$31\t # 10\treturn_internal\n"
       "\taddu\t$2,$4,1\t # 9\taddsi3/2\n\t.set\treorder\n\t.end\tpp_add\n")
PREPROCESSED = '# 1 "probe.c"\n\nint pp_add(int a)\n{\n# 7 "probe.c"\n  return a + 1;\n}\n'
WORDS = [0x03E00008, 0x24820001]                  # j $31 ; addiu $v0,$a0,1


def elf_bytes(functions) -> bytes:
    """A relocatable with .text holding ``functions`` [(name, words)], .symtab, .strtab, .shstrtab."""
    text = b"".join(struct.pack(">I", w) for _n, words in functions for w in words)
    shstr = b"\0.text\0.symtab\0.strtab\0.shstrtab\0"
    strtab = b"\0" + b"".join(name.encode() + b"\0" for name, _w in functions)
    symbols, position = [bytes(16)], 0
    for name, words in functions:
        symbols.append(struct.pack(">IIIBBH", strtab.index(b"\0" + name.encode() + b"\0") + 1, position,
                                   4 * len(words), 0x12, 0, 1))
        position += 4 * len(words)
    symtab = b"".join(symbols)
    body = [text, symtab, strtab, shstr]
    offsets, cursor = [], 52
    for part in body:
        offsets.append(cursor)
        cursor += len(part)

    def sh(name, kind, offset, size, link=0, info=0, entsize=0):
        return struct.pack(">10I", shstr.index(name.encode() + b"\0"), kind, 0, 0, offset, size, link, info, 4,
                           entsize)
    sections = [bytes(40), sh(".text", 1, offsets[0], len(text)), sh(".symtab", 2, offsets[1], len(symtab), 3, 1, 16),
                sh(".strtab", 3, offsets[2], len(strtab)), sh(".shstrtab", 3, offsets[3], len(shstr))]
    return (b"\x7fELF\x01\x02\x01" + bytes(9) + struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, cursor, 0, 52, 0, 0, 40,
                                                             len(sections), 4)
            + b"".join(body) + b"".join(sections))


def fake_build(refuse=False, traced_words=None, write_dumps=True):
    calls = []

    def build(source, spec, cfg, workdir):
        calls.append(list(spec.flags))
        unit = Path(workdir) / "unit"
        unit.mkdir(parents=True, exist_ok=True)
        traced = "-da" in spec.flags
        if traced and refuse:
            raise BuildError("cc1: Invalid option `-da'")
        words = traced_words if traced and traced_words is not None else WORDS
        obj = unit / "probe.o"
        obj.write_bytes(elf_bytes([("pp_add", words)]))
        if traced and write_dumps:
            (unit / "probe.c.rtl").write_text(RTL)
            (unit / "probe.c.lreg").write_text(LREG)
            (unit / "probe.c.greg").write_text(GREG)
            (unit / "probe.c.weird").write_text("?")
            (unit / "probe.i").write_text(PREPROCESSED)
            (unit / "probe.raw.s").write_text(ASM)
        return obj
    return build, calls


class ParserTest(unittest.TestCase):
    def test_rtl(self):
        insns = ct.parse_rtl(RTL)["pp_add"]
        self.assertEqual([i["uid"] for i in insns], [6, 9, 10])
        self.assertEqual([i["line"] for i in insns], [3, 4, 4])
        self.assertEqual(insns[0]["regs"], [4, 77])
        self.assertEqual(insns[1]["regs"], [77, 78])
        self.assertEqual(insns[2]["kind"], "jump_insn")
        self.assertEqual(ct.parse_rtl(GREG)["pp_add"][0]["code"], 33)

    def test_asm(self):
        rows = ct.parse_asm(ASM)["pp_add"]
        self.assertEqual([r["mnemonic"] for r in rows], ["j", "addu"])
        self.assertEqual([(r["uid"], r["pattern"], r["alternative"]) for r in rows],
                         [(10, "return_internal", None), (9, "addsi3", 2)])
        self.assertEqual(rows[0]["line"], 4)

    def test_multi_line_template_belongs_to_one_insn(self):
        text = "\t.ent\tf\nf:\n\tlui\t$2,%hi(x)\n\tlw\t$2,%lo(x)($2)\t # 12\tmovsi_internal2/1\n\t.end\tf\n"
        rows = ct.parse_asm(text)["f"]
        self.assertEqual([r["uid"] for r in rows], [12, 12])

    def test_line_map_and_regalloc(self):
        mapped = ct.line_map(PREPROCESSED)
        self.assertIn((6, "probe.c", 7), mapped)
        self.assertEqual(mapped[0], (2, "probe.c", 1))
        alloc = ct.parse_regalloc(LREG)
        self.assertEqual(alloc["assignments"], {"77": 4, "78": 2})

    def test_trace_modes(self):
        self.assertEqual(ct.trace_mode("kmc+atload"), "driver")
        self.assertEqual(ct.trace_mode("custom:gcc272snew -O2"), "cc1")
        for lane in ("sn", "custom:gcc272sn0004", "ido5.3", "custom:mipspro744"):
            self.assertEqual(ct.trace_mode(lane), "black-box")


class CollectTest(unittest.TestCase):
    def collect(self, build, lane="kmc+as"):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "probe.c"
            source.write_text("int pp_add(int a) { return a + 1; }\n")
            return ct.collect(source, lane, ["-O2"], None, Path(folder) / "work", build=build)

    def test_white_box(self):
        build, calls = fake_build()
        trace = self.collect(build)
        self.assertEqual(calls, [["-O2"], ["-O2", "-da", "-dp", "-save-temps"]])
        self.assertEqual(trace["schema"], ct.SCHEMA)
        self.assertTrue(trace["identical_code"])
        self.assertEqual([p["name"] for p in trace["passes"]], ["rtl", "lreg", "greg"])
        self.assertEqual(trace["unknown_dumps"], ["probe.c.weird"])
        self.assertEqual(trace["asm"]["name"], "probe.raw.s")
        self.assertEqual(trace["object"]["functions"]["pp_add"]["words"], ["03E00008", "24820001"])
        self.assertEqual(trace["regalloc"]["assignments"]["78"], 2)
        links = ct.causal_links(trace)["functions"]["pp_add"]
        add = next(link for link in links["links"] if link["uid"] == 9)
        self.assertEqual((add["line"], add["pseudos"], add["hard"]), (4, [77, 78], [2, 4]))
        self.assertEqual(add["assigned"], {"77": 4, "78": 2})
        self.assertEqual((add["asm"], add["offsets"]), (["addu $2,$4,1"], [4]))
        ret = next(link for link in links["links"] if link["uid"] == 10)
        self.assertEqual(ret["offsets"], [0])
        self.assertEqual(links["limits"], [])

    def test_refusal_falls_back_to_black_box(self):
        build, _calls = fake_build(refuse=True)
        trace = self.collect(build, lane="custom:egcs112")
        self.assertIn("Invalid option", trace["refused"])
        self.assertEqual(trace["mode"], "black-box")
        self.assertEqual(trace["passes"], [])
        self.assertTrue(trace["identical_code"])
        self.assertIn("pp_add", trace["object"]["functions"])
        self.assertTrue(ct.causal_links(trace)["limits"])

    def test_black_box_lane_gets_no_switches(self):
        build, calls = fake_build()
        trace = self.collect(build, lane="ido5.3")
        self.assertEqual(calls, [["-O2"]])
        self.assertEqual(trace["trace_flags"], [])

    def test_changed_code_is_flagged(self):
        build, _calls = fake_build(traced_words=[0x03E00008, 0x24820002])
        trace = self.collect(build)
        self.assertFalse(trace["identical_code"])
        self.assertTrue(any("differs" in text for text in trace["limits"]))

    def test_no_dumps_is_flagged(self):
        build, _calls = fake_build(write_dumps=False)
        trace = self.collect(build)
        self.assertTrue(any("no pass dump" in text for text in trace["limits"]))

    def test_failed_build(self):
        def broken(source, spec, cfg, workdir):
            raise BuildError("gcc failed")
        trace = self.collect(broken)
        self.assertIn("gcc failed", trace["error"])

    def test_object_summary(self):
        summary = ct.object_summary(Elf(elf_bytes([("a", [0x03E00008, 0]), ("b", [0x24020001])])))
        self.assertEqual(summary["functions"]["b"]["offset"], 8)
        self.assertEqual(len(summary["sha256"]), 64)


class ReduceTest(unittest.TestCase):
    SOURCE = ("int g;\nint unused(int a) { return a + 5; }\nint f(int x)\n{\n    int y = 3;\n    g = y + 7;\n"
              "    return x * 12;\n}\n")

    @staticmethod
    def keeps(text):
        return "int f(int x)" in text and "return x * 12;" in text and text.count("{") == text.count("}")

    def test_units_cover_the_text(self):
        pieces = ct.units(self.SOURCE + '#define S "a;{"\n/* ; */ char *s = "x;y";\n')
        self.assertEqual("".join(pieces), self.SOURCE + '#define S "a;{"\n/* ; */ char *s = "x;y";\n')
        self.assertIn('#define S "a;{"\n', pieces)
        self.assertIn('/* ; */ char *s = "x;y";\n', pieces)

    def test_statements_are_removed(self):
        result = ct.reduce(self.SOURCE, self.keeps, name="mul")
        self.assertTrue(result["kept"])
        self.assertTrue(self.keeps(result["body"]))
        self.assertNotIn("g = y", result["body"])
        self.assertNotIn("int y", result["body"])
        self.assertLess(result["units_after"], result["units_before"])
        self.assertTrue(result["source"].startswith("/* reduced probe mul:"))

    def test_expressions_are_simplified(self):
        result = ct.reduce("int f(int x) { return x + 7 * 3; }",
                           lambda t: "x +" in t and "int f(int x) {" in t and t.rstrip().endswith("}"))
        self.assertEqual(result["body"], "int f(int x) { return x + 0 * 0; }")

    def test_predicate_must_hold_first(self):
        result = ct.reduce(self.SOURCE, lambda text: False)
        self.assertFalse(result["kept"])
        self.assertEqual(result["source"], self.SOURCE)

    def test_budget(self):
        result = ct.reduce(self.SOURCE, self.keeps, max_tests=3)
        self.assertTrue(result["budget_exhausted"])
        self.assertLessEqual(result["tests"], 3)
        self.assertTrue(self.keeps(result["body"]))

    def test_residual_predicate(self):
        target = [0x8C820000, 0x8C830004, 0x00431021, 0x03E00008, 0]
        moved = [0x8C830004, 0x8C820000, 0x00431021, 0x03E00008, 0]
        predicate = ct.residual_predicate(lambda text: moved if "keep" in text else list(target), target,
                                          "scheduling", wrap=False)
        result = ct.reduce("int a;\nint keep;\nint b;\n", predicate)
        self.assertEqual(result["body"], "int keep;\n")


if __name__ == "__main__":
    unittest.main()
