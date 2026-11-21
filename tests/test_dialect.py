"""Dialect probe readings on hand-built big-endian MIPS ELF objects (no toolchain)."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from n64dk.elf import Elf
from n64dk.match import dialect
from n64dk.match.passprobe import ObjectView


def r(fn, rs=0, rt=0, rd=0, sa=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | fn


def i(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def cop1(fmt, fn, fs=0, fd=0, ft=0):
    return (17 << 26) | (fmt << 21) | (ft << 16) | (fs << 11) | (fd << 6) | fn


NOP, JR_RA, JAL = 0, r(8, 31), 3 << 26
S, D, W = 16, 17, 20


def SLL(rd, rt, sa):
    return r(0, 0, rt, rd, sa)


def SRL(rd, rt, sa):
    return r(2, 0, rt, rd, sa)


def SRA(rd, rt, sa):
    return r(3, 0, rt, rd, sa)


def ADDU(rd, rs, rt):
    return r(0x21, rs, rt, rd)


def SLTU(rd, rs, rt):
    return r(0x2B, rs, rt, rd)


def ADDIU(rt, rs, imm):
    return i(9, rs, rt, imm)


def ANDI(rt, rs, imm):
    return i(12, rs, rt, imm)


def LB(rt, offset, base):
    return i(32, base, rt, offset)


def LBU(rt, offset, base):
    return i(36, base, rt, offset)


def LW(rt, offset, base):
    return i(35, base, rt, offset)


def SB(rt, offset, base):
    return i(40, base, rt, offset)


def SH(rt, offset, base):
    return i(41, base, rt, offset)


def SW(rt, offset, base):
    return i(43, base, rt, offset)


def MTC1(rt, fs):
    return (17 << 26) | (4 << 21) | (rt << 16) | (fs << 11)


def make_object(functions, data=(), calls=(), sized=True) -> Elf:
    """A relocatable with .text (the functions in order), .bss, .data, .rel.text (R_MIPS_26 for
    ``calls``: (function, word index, callee)), .symtab, .strtab and .shstrtab.  ``data``:
    (name, ".bss" | ".data" | special section index such as 0xFFF2 for COMMON)."""
    text = b"".join(struct.pack(">I", word) for _name, words in functions for word in words)
    shstr = b"\0.text\0.bss\0.data\0.rel.text\0.symtab\0.strtab\0.shstrtab\0"
    callees = sorted({callee for _f, _k, callee in calls})
    names = [name for name, _w in functions] + [name for name, _w in data] + callees
    strtab = b"\0" + b"".join(name.encode() + b"\0" for name in names)

    def at(table, name):
        return table.index(b"\0" + name.encode() + b"\0") + 1

    symbols, index, starts, position = [bytes(16)], {}, {}, 0
    for name, words in functions:
        index[name], starts[name] = len(symbols), position
        symbols.append(struct.pack(">IIIBBH", at(strtab, name), position, 4 * len(words) if sized else 0, 0x12, 0, 1))
        position += 4 * len(words)
    for name, where in data:
        shndx = {".bss": 2, ".data": 3}.get(where, where)
        symbols.append(struct.pack(">IIIBBH", at(strtab, name), 0, 4, 0x11, 0, shndx))
    for callee in callees:
        index[callee] = len(symbols)
        symbols.append(struct.pack(">IIIBBH", at(strtab, callee), 0, 0, 0x10, 0, 0))
    symtab = b"".join(symbols)
    rel = b"".join(struct.pack(">II", starts[f] + 4 * k, (index[callee] << 8) | 4) for f, k, callee in calls)
    data_bytes = bytes(8)
    body = [text, data_bytes, rel, symtab, strtab, shstr]
    offsets, cursor = [], 52
    for part in body:
        offsets.append(cursor)
        cursor += len(part)

    def sh(name, kind, flags, offset, size, link=0, info=0, align=4, entsize=0):
        return struct.pack(">10I", shstr.index(name.encode() + b"\0"), kind, flags, 0, offset, size, link, info, align,
                           entsize)

    sections = [bytes(40),
                sh(".text", 1, 6, offsets[0], len(text)),
                sh(".bss", 8, 3, offsets[1], 16),
                sh(".data", 1, 3, offsets[1], len(data_bytes)),
                sh(".rel.text", 9, 0, offsets[2], len(rel), 5, 1, 4, 8),
                sh(".symtab", 2, 0, offsets[3], len(symtab), 6, 1, 4, 16),
                sh(".strtab", 3, 0, offsets[4], len(strtab), 0, 0, 1),
                sh(".shstrtab", 3, 0, offsets[5], len(shstr), 0, 0, 1)]
    header = (b"\x7fELF\x01\x02\x01" + bytes(9)
              + struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, cursor, 0, 52, 0, 0, 40, len(sections), 7))
    return Elf(header + b"".join(body) + b"".join(sections))


def answers(probe, elf):
    return {fact: value["answer"] for fact, value in dialect.classify(probe, elf).items()}


class ObjectViewTest(unittest.TestCase):
    def test_sized_and_unsized_functions(self):
        functions = [("a", [ADDIU(2, 0, 1), JR_RA, NOP, 0]), ("b", [JR_RA, NOP])]
        sized = ObjectView(make_object(functions))
        self.assertEqual(sized.words("a"), [ADDIU(2, 0, 1), JR_RA, NOP, 0])
        unsized = ObjectView(make_object(functions, sized=False))
        self.assertEqual(unsized.words("a"), [ADDIU(2, 0, 1), JR_RA, 0])
        self.assertEqual(unsized.words("b"), [JR_RA, 0])
        self.assertIsNone(unsized.words("missing"))

    def test_calls_and_places(self):
        view = ObjectView(make_object([("f", [JAL, NOP, JR_RA, NOP])], data=[("c", 0xFFF2), ("b", ".bss")],
                                      calls=[("f", 0, "__muldi3")]))
        self.assertEqual(view.calls("f"), ["__muldi3"])
        self.assertEqual((view.place("c"), view.place("b"), view.place("__muldi3"), view.place("x")),
                         ("COMMON", ".bss", "UNDEF", None))


class ReadingTest(unittest.TestCase):
    def test_call_promotion(self):
        narrow = [SLL(4, 4, 24), SRA(4, 4, 24), SLL(5, 5, 16), JAL, SRA(5, 5, 16), JR_RA, NOP]
        promoted = [cop1(S, 33, fs=14, fd=0), JAL, NOP, JR_RA, NOP]
        found = answers("call_promotion", make_object([("call_p", narrow), ("call_k", promoted)]))
        self.assertEqual(found, {"unprototyped_float_arg": "double", "prototyped_narrow_arg": "caller-narrows"})
        plain = answers("call_promotion", make_object([("call_p", [JAL, NOP, JR_RA, NOP]),
                                                       ("call_k", [JAL, NOP, JR_RA, NOP])]))
        self.assertEqual(plain, {"unprototyped_float_arg": "float", "prototyped_narrow_arg": "caller-passes-int"})

    def test_implicit_declaration(self):
        undeclared = [JAL, NOP, MTC1(2, 4), cop1(W, 32, fs=4, fd=0), JR_RA, NOP]
        reextended = [JAL, NOP, SLL(2, 2, 16), SRA(2, 2, 16), JR_RA, ADDIU(2, 2, 1)]
        found = answers("implicit_decl", make_object([("use_decl", reextended), ("use_undecl_f", undeclared)]))
        self.assertEqual(found, {"implicit_int": "int", "narrow_return": "caller-reextends"})
        trusted = answers("implicit_decl", make_object([("use_decl", [JAL, NOP, JR_RA, ADDIU(2, 2, 1)])]))
        self.assertEqual(trusted["narrow_return"], "caller-trusts-callee")

    def test_plain_char(self):
        controls = [("signed_char", [JR_RA, LB(2, 0, 4)]), ("unsigned_char", [JR_RA, LBU(2, 0, 4)])]
        signed = dialect.classify("char_sign", make_object([("plain_char", [JR_RA, LB(2, 0, 4)])] + controls))
        self.assertEqual(signed["plain_char"]["answer"], "signed")
        self.assertTrue(signed["plain_char"]["conclusive"])
        unsigned = answers("char_sign", make_object([("plain_char", [JR_RA, LBU(2, 0, 4)])] + controls))
        self.assertEqual(unsigned["plain_char"], "unsigned")
        extended = answers("char_sign", make_object([("plain_char", [LBU(2, 0, 4), SLL(2, 2, 24), JR_RA,
                                                                    SRA(2, 2, 24)])] + controls))
        self.assertEqual(extended["plain_char"], "signed")

    def test_enum_size(self):
        found = answers("enum_size", make_object([("size_small", [JR_RA, ADDIU(2, 0, 4)]),
                                                  ("size_big", [JR_RA, ADDIU(2, 0, 4)]),
                                                  ("enum_offset", [JR_RA, LB(2, 8, 4)])]))
        self.assertEqual(found, {"enum_size": "4", "enum_size_large": "4", "enum_in_struct": "4"})
        short = answers("enum_size", make_object([("size_small", [JR_RA, ADDIU(2, 0, 1)]),
                                                  ("enum_offset", [JR_RA, LBU(2, 2, 4)])]))
        self.assertEqual((short["enum_size"], short["enum_in_struct"]), ("1", "1"))

    def test_bitfields(self):
        get_c = [LW(2, 0, 4), NOP, SLL(2, 2, 8), JR_RA, SRA(2, 2, 28)]
        get_d = [LW(2, 0, 4), NOP, SLL(2, 2, 12), JR_RA, SRA(2, 2, 28)]
        msb = dialect.classify("bitfields", make_object([("get_a", [LW(2, 0, 4), NOP, JR_RA, SRL(2, 2, 29)]),
                                                         ("get_c", get_c), ("get_d", get_d)]))
        self.assertEqual(msb["bitfield_order"]["answer"], "msb-first")
        self.assertEqual(msb["plain_int_bitfield"]["answer"], "signed")
        self.assertTrue(msb["plain_int_bitfield"]["conclusive"])
        lsb = answers("bitfields", make_object([("get_a", [LW(2, 0, 4), NOP, JR_RA, ANDI(2, 2, 7)]),
                                                ("get_c", [LW(2, 0, 4), NOP, SRL(2, 2, 8), JR_RA, ANDI(2, 2, 15)])]))
        self.assertEqual(lsb, {"bitfield_order": "lsb-first", "plain_int_bitfield": "unsigned"})
        byte = answers("bitfields", make_object([("get_a", [LBU(2, 0, 4), NOP, JR_RA, SRL(2, 2, 5)])]))
        self.assertEqual(byte["bitfield_order"], "msb-first")

    def test_old_style_parameters(self):
        found = answers("knr_params", make_object([
            ("knr_narrow", [SLL(4, 4, 24), SRA(4, 4, 24), SLL(5, 5, 16), SRA(5, 5, 16), JR_RA, ADDU(2, 4, 5)]),
            ("proto_narrow", [JR_RA, ADDU(2, 4, 5)]),
            ("knr_float", [cop1(D, 32, fs=12, fd=0), cop1(S, 2, fs=0, fd=0, ft=0), JR_RA, NOP])]))
        self.assertEqual(found, {"knr_narrow_params": "callee-narrows", "proto_narrow_params": "trusts-caller",
                                 "knr_float_param": "arrives-as-double"})

    def test_tentative_definitions(self):
        found = answers("tentative", make_object([("use_tent", [JR_RA, NOP])],
                                                 data=[("tent_def", 0xFFF2), ("zero_init", ".bss"),
                                                       ("stat_tent", ".bss")]))
        self.assertEqual(found, {"tentative_definition": "COMMON", "zero_initialized": ".bss",
                                 "static_tentative": ".bss"})
        missing = dialect.classify("tentative", make_object([("use_tent", [JR_RA, NOP])], data=[("tent_def", ".bss")]))
        self.assertEqual(missing["static_tentative"]["answer"], "no-symbol")
        self.assertFalse(missing["static_tentative"]["conclusive"])

    def test_volatile(self):
        found = answers("volatile", make_object([
            ("vol_twice", [LW(2, 0, 8), LW(3, 0, 8), JR_RA, ADDU(2, 2, 3)]),
            ("nonvol_twice", [LW(2, 0, 8), JR_RA, SLL(2, 2, 1)]),
            ("vol_store_call", [ADDIU(29, 29, -24), SW(31, 20, 29), SW(4, 0, 8), JAL, NOP, JR_RA, NOP]),
            ("nonvol_store_call", [ADDIU(29, 29, -24), SW(31, 20, 29), JAL, SW(4, 0, 8), JR_RA, NOP])]))
        self.assertEqual(found, {"volatile_loads": "kept", "nonvolatile_reuse": "reused",
                                 "volatile_in_delay_slot": "kept-out"})

    def test_type_based_aliasing(self):
        reload = [LW(2, 0, 4), SB(0, 0, 5), LW(3, 0, 4), JR_RA, ADDU(2, 2, 3)]
        reuse = [LW(2, 0, 4), SH(0, 0, 5), JR_RA, SLL(2, 2, 1)]
        found = answers("aliasing", make_object([("alias_char", reload), ("alias_short", reuse),
                                                 ("alias_float", reuse), ("alias_int", reload)]))
        self.assertEqual(found, {"type_based_aliasing": "yes"})
        none = answers("aliasing", make_object([("alias_char", reload), ("alias_short", reload),
                                                ("alias_float", reload), ("alias_int", reload)]))
        self.assertEqual(none, {"type_based_aliasing": "no"})
        unsafe = dialect.classify("aliasing", make_object([("alias_char", reuse), ("alias_short", reuse),
                                                           ("alias_float", reuse), ("alias_int", reload)]))
        self.assertFalse(unsafe["type_based_aliasing"]["conclusive"])

    def test_float_precision(self):
        found = answers("float_precision", make_object([
            ("fexpr", [cop1(S, 2, fs=12, fd=0, ft=14), cop1(S, 0, fs=0, fd=0, ft=4), JR_RA, NOP]),
            ("fconst", [cop1(S, 33, fs=12, fd=0), cop1(D, 2, fs=0, fd=0, ft=2), cop1(D, 32, fs=0, fd=0), JR_RA,
                        NOP])]))
        self.assertEqual(found, {"float_arithmetic": "single", "double_constant": "double"})

    def test_long_long(self):
        functions = [("ll_shl", [JAL, NOP, JR_RA, NOP]), ("ll_mul", [JAL, NOP, JR_RA, NOP]),
                     ("ll_div", [JAL, NOP, JR_RA, NOP]),
                     ("ll_add", [ADDU(3, 5, 7), SLTU(8, 3, 7), ADDU(2, 4, 6), JR_RA, ADDU(2, 2, 8)])]
        gnu = answers("long_long", make_object(functions, calls=[("ll_shl", 0, "__ashldi3"), ("ll_mul", 0, "__muldi3"),
                                                                 ("ll_div", 0, "__divdi3")]))
        self.assertEqual(gnu, {"ll_shift": "call:__ashldi3", "ll_multiply": "call:__muldi3",
                               "ll_divide": "call:__divdi3", "ll_add": "inline-32bit", "ll_runtime": "libgcc"})
        ido = answers("long_long", make_object(functions, calls=[("ll_shl", 0, "__ll_lshift"), ("ll_mul", 0, "__ll_mul"),
                                                                 ("ll_div", 0, "__ll_div")]))
        self.assertEqual(ido["ll_runtime"], "ido-ll")

    def test_register_keyword(self):
        loop = [LW(8, 0, 4), JR_RA, NOP]
        self.assertEqual(answers("register_kw", make_object([("no_register", loop), ("with_register", loop)])),
                         {"register_keyword": "no-effect"})
        other = [SW(4, 0, 29), LW(8, 0, 4), JR_RA, NOP]
        self.assertEqual(answers("register_kw", make_object([("no_register", other), ("with_register", loop)])),
                         {"register_keyword": "changes-code"})

    def test_inline_keywords(self):
        self.assertEqual(answers("inline_plain", make_object([("use_inline", [JR_RA, NOP])])),
                         {"inline_keyword": "inlined"})
        self.assertEqual(answers("inline_gnu", make_object([("use_inline", [JAL, NOP, JR_RA, NOP])])),
                         {"inline_gnu_keyword": "accepted-not-inlined"})
        refused = dialect.classify("inline_sgi", None)
        self.assertEqual(refused["inline_sgi_keyword"]["answer"], "rejected")

    def test_failed_and_unreadable(self):
        failed = dialect.classify("char_sign", None)
        self.assertEqual(failed["plain_char"], {"answer": "build-failed", "evidence": ["the probe did not build"],
                                                "conclusive": False})
        unreadable = dialect.classify("enum_size", make_object([("other", [JR_RA, NOP])]))
        self.assertEqual({value["answer"] for value in unreadable.values()}, {"unreadable"})

    def test_corpus_is_consistent(self):
        self.assertEqual(set(dialect.READERS), set(dialect.BY_NAME))
        for probe in dialect.PROBES:
            self.assertTrue(probe.question and probe.reading and probe.source.endswith("\n"))
            self.assertNotIn("//", probe.source)
        self.assertEqual(len(dialect.FACTS), sum(len(probe.facts) for probe in dialect.PROBES))


class MatrixTest(unittest.TestCase):
    def rows(self):
        controls = [("signed_char", [JR_RA, LB(2, 0, 4)]), ("unsigned_char", [JR_RA, LBU(2, 0, 4)])]
        gcc = make_object([("plain_char", [JR_RA, LB(2, 0, 4)])] + controls)
        ido = make_object([("plain_char", [JR_RA, LBU(2, 0, 4)])] + controls)
        return [{"recipe": "kmc+as -O2", "probe": "char_sign", "built": True, "facts": dialect.classify("char_sign", gcc)},
                {"recipe": "ido5.3 -O2", "probe": "char_sign", "built": True, "facts": dialect.classify("char_sign", ido)},
                {"recipe": "ido5.3 -O2", "probe": "inline_plain", "built": False, "error": "cfe: syntax error",
                 "facts": dialect.classify("inline_plain", None)},
                {"recipe": "broken -O2", "probe": "inline_plain", "built": False, "error": "not found",
                 "facts": dialect.classify("inline_plain", None)}]

    def test_matrix_and_differences(self):
        result = dialect.matrix(self.rows())
        self.assertEqual(result["table"]["kmc+as -O2"]["plain_char"], "signed")
        self.assertEqual(result["table"]["ido5.3 -O2"]["plain_char"], "unsigned")
        self.assertEqual(result["table"]["kmc+as -O2"]["inline_keyword"], "-")
        self.assertEqual(result["differing"]["plain_char"], {"ido5.3 -O2": "unsigned", "kmc+as -O2": "signed"})
        self.assertTrue(result["recipes"]["ido5.3 -O2"]["facts"]["inline_keyword"]["conclusive"])
        self.assertFalse(result["recipes"]["broken -O2"]["facts"]["inline_keyword"]["conclusive"])
        self.assertIn("plain_char", result["questions"])
        text = dialect.markdown(result)
        self.assertIn("| plain_char |", text)
        self.assertIn("rejected ?", text)

    def test_facts_for(self):
        result = dialect.matrix(self.rows())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "matrix.json"
            path.write_text(json.dumps(result))
            self.assertEqual(dialect.facts_for("ido5.3 -O2", path)["plain_char"], "unsigned")
            self.assertEqual(dialect.facts_for("kmc+as -O2 -G0 -mips3", path)["plain_char"], "signed")
            self.assertEqual(dialect.facts_for("ido5.3 -g", path)["plain_char"], "unsigned")
            self.assertEqual(dialect.facts_for("sn -O2", path), {})
            self.assertEqual(dialect.facts_for("sn -O2", Path(folder) / "absent.json"), {})


if __name__ == "__main__":
    unittest.main()
