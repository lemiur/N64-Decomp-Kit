"""Code/data classification (n64dk.analysis.codedata) on adversarial hand-built word arrays."""
import struct
import unittest

from n64dk.analysis import codedata
from n64dk.analysis.codedata import AMBIGUOUS, CODE, DATA, Context, Evidence

L = 0x80001000
JR_RA = 0x03E00008
PROLOGUE = [0x27BDFFE8, 0xAFBF0014]                   # addiu sp,sp,-0x18; sw ra,0x14(sp)
EPILOGUE = [0x8FBF0014, JR_RA, 0x27BD0018]            # lw ra,0x14(sp); jr ra; addiu sp,sp,0x18
JAL_F = 0x0C001000                                     # jal 0x80004000


def blob(words):
    return b"".join(struct.pack(">I", w) for w in words)


def run(words, ctx=None, at=L):
    return codedata.classify(blob(words), at, at, at + 4 * len(words), ctx)


def called(at=L):
    return Context(call_sites={at: [0x80009000]})


class CodeTest(unittest.TestCase):
    def test_frameless_leaf_stays_code_through_its_call(self):
        leaf = [JR_RA, 0x8C820008]                     # jr ra; lw v0,8(a0)
        verdict = run(leaf, called())
        self.assertEqual(verdict.verdict, CODE, verdict.as_dict())
        self.assertEqual(run(leaf).verdict, AMBIGUOUS)  # shape alone never makes code

    def test_code_with_embedded_data(self):
        words = PROLOGUE + [JAL_F, 0] + EPILOGUE + [0x3F800000, 0x40000000]
        verdict = run(words, called())
        self.assertEqual(verdict.verdict, CODE, verdict.as_dict())
        self.assertEqual(verdict.extent, (L, L + 0x1C))
        self.assertTrue(any("not reached" in note for note in verdict.notes))
        tail = codedata.classify(blob(words), L, L + 0x1C, L + 0x24)
        self.assertNotEqual(tail.verdict, CODE)


class DataTest(unittest.TestCase):
    # lui t0..t7 with float-looking immediates: every word is a valid instruction and a float
    # of magnitude ~2^-7 with the same exponent
    FLOATS = [0x3C083F80, 0x3C094000, 0x3C0A4040, 0x3C0B4080, 0x3C0C40A0, 0x3C0D40C0, 0x3C0E40E0, 0x3C0F4100]

    def test_float_array_that_decodes_as_instructions(self):
        verdict = run(self.FLOATS, Context(loads={L: {"lwc1"}, L + 4: {"lwc1"}}))
        self.assertEqual(verdict.verdict, DATA, verdict.as_dict())
        self.assertNotEqual(run(self.FLOATS).verdict, CODE)

    def test_display_list_with_prologue_words(self):
        dl = [0xE7000000, 0x00000000,          # G_RDPPIPESYNC
              0xD9000000, 0x27BDFFE8,          # G_GEOMETRYMODE, w1 looks like addiu sp,sp,-0x18
              0xD7000002, 0xFFFFFFFF,          # G_TEXTURE
              0xFC127E03, 0xAFBF0014,          # G_SETCOMBINE, w1 looks like sw ra,0x14(sp)
              0x01020040, 0x06000000,          # G_VTX
              0x06000204, 0x00060402,          # G_TRI2
              0xDF000000, 0x00000000]          # G_ENDDL
        verdict = run(dl)
        self.assertEqual(verdict.verdict, DATA, verdict.as_dict())
        self.assertTrue(any("display list" in e.what for e in verdict.data))

    def test_jump_table_that_looks_executable(self):
        table = [L + 0x20, L + 0x28, L + 0x30, L + 0x20]          # decode as lb $zero, imm($zero)
        at = 0x80002000
        ctx = Context(tables=[(at, 16, L + 0x18)], image=(L, 0x80003000))
        verdict = run(table, ctx, at)
        self.assertEqual(verdict.verdict, DATA, verdict.as_dict())
        self.assertNotEqual(run(table, Context(image=(L, 0x80003000)), at).verdict, CODE)

    def test_random_bytes_with_an_apparent_function(self):
        state, noise = 0x12345678, []
        for _ in range(60):
            state = (state * 1103515245 + 12345) & 0xFFFFFFFF
            noise.append(state ^ (state >> 13))
        words = PROLOGUE + EPILOGUE + noise
        with_call = run(words, called())
        self.assertEqual(with_call.verdict, AMBIGUOUS, with_call.as_dict())
        self.assertNotEqual(run(words).verdict, CODE)

    def test_strings(self):
        raw = b"Hello, world!\0\0\0Another string\0\0"
        verdict = codedata.classify(raw, L, L, L + len(raw))
        self.assertEqual(verdict.verdict, DATA, verdict.as_dict())


class DecideTest(unittest.TestCase):
    def test_strong_evidence_on_both_sides_is_ambiguous(self):
        code = [Evidence("code", 3.0, "called", True)]
        data = [Evidence("data", 4.0, "jump table", True)]
        self.assertEqual(codedata.decide(code, data)[0], AMBIGUOUS)

    def test_code_needs_a_reference(self):
        shape_only = [Evidence("code", 2.0, "closes"), Evidence("code", 1.5, "frame"), Evidence("code", 1.0, "saved")]
        self.assertEqual(codedata.decide(shape_only, [])[0], AMBIGUOUS)
        self.assertEqual(codedata.decide(shape_only + [Evidence("code", 1.0, "la", True)], [])[0], CODE)

    def test_weak_data_does_not_outvote_code(self):
        code = [Evidence("code", 3.0, "called", True), Evidence("code", 2.0, "closes")]
        self.assertEqual(codedata.decide(code, [Evidence("data", 2.0, "floats")])[0], CODE)
        self.assertEqual(codedata.decide(code, [Evidence("data", 2.0, "f"), Evidence("data", 1.0, "w")])[0],
                         AMBIGUOUS)


class ShapeTest(unittest.TestCase):
    def test_float_and_pointer_shapes(self):
        self.assertEqual(codedata.float_shape([0x3F800000, 0x3F000000, 0x3F400000, 0x3F200000]), (1.0, 3))
        self.assertIsNone(codedata.float_shape([0x27BDFFE8, 0xAFBF0014, 0x8FBF0014, JR_RA]))
        self.assertEqual(codedata.pointer_shape([L, L + 4, 0, L + 8], (L, L + 0x100)), (3, 3))
        self.assertIsNone(codedata.pointer_shape([L, 0x3C083F80, L + 8], (L, L + 0x100)))

    def test_context_from_image(self):
        caller = [0x27BDFFE8, 0xAFBF0014, 0x0C000410, 0x00000000] + EPILOGUE + [0]   # jal 0x80001040
        callee = [JR_RA, 0]
        data = bytearray(0x100)
        data[0:len(caller) * 4] = blob(caller)
        data[0x40:0x48] = blob(callee)
        data[0x80:0x84] = struct.pack(">I", L + 0x40)                                # a pointer word in data
        ctx = Context.from_image(bytes(data), L, [("a", L, 0x24), ("b", L + 0x40, 8)])
        self.assertEqual(ctx.call_sites, {L + 0x40: [L + 8]})
        self.assertEqual(ctx.pointer_targets, {L + 0x40: 1})
        verdict = codedata.classify(bytes(data), L, L + 0x40, L + 0x48, ctx)
        self.assertEqual(verdict.verdict, CODE, verdict.as_dict())


if __name__ == "__main__":
    unittest.main()
