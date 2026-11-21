"""Hand-matching rewrites of the Madden Football 64 campaign (n64dk.match.handforms).

Each test is a reduced (failing C, exact C) pair from the campaign (kitwork/kd_pairs.tsv): the
form computed from the failing C must be the exact C's shape.  No ROM bytes: targets are
synthetic instruction words built here."""
import atexit
import shutil
import struct
import tempfile
import unittest

from n64dk.match import handforms, iterate, resolve

BASE = 0x80001000


def _code(words):
    return b"".join(struct.pack(">I", w) for w in words)


def _scratch():
    path = tempfile.mkdtemp(prefix="n64dk-handforms-")
    atexit.register(shutil.rmtree, path, True)
    return path


# overlay1 800A3294 (reduced): two neighbouring s32 globals; the exact C declared them as one array.
PAIR_GLOBALS = """#include "types.h"
extern s32 D_80087A70;
extern s32 D_80087A74;

void func_800A3294(void) {
    if (D_80087A70 != 0x7FFFFFFF) {
        D_80087A70 = -D_80087A70;
        D_80087A74 = -D_80087A74;
    }
}
"""

# overlay1 800B2968 (reduced): a lone global the exact C wrote as D_8008A664[0].
PAIR_LONE = """extern u16 D_8008A664;

void func_800B2968(s32 on) {
    if (on) {
        D_8008A664 |= 0x100;
    } else {
        D_8008A664 &= ~0x100;
    }
}
"""

# overlay1 8008D1B0 (reduced): two stores of one constant; the exact C chained them.
PAIR_CHAIN = """extern struct { u16 unk_00[0x75]; u16 unk_EA[0x75]; } D_800C5A60;

void func_8008D1B0(s32 i) {
    D_800C5A60.unk_EA[i] = 0xFFFF;
    D_800C5A60.unk_00[i] = 0xFFFF;
}
"""

# overlay1 800975A0 (reduced) and the lesson "length = b1; length <<= 8; length |= b0;".
PAIR_SPLIT = """void func_80040000(u8 *p) {
    s32 length;

    length = (p[1] << 8) | p[0];
    func_80040100(length);
}
"""

# overlay1 800C3D68 (reduced): `s32 limit` where the target sign-extends 16 bits.
PAIR_NARROW = """void func_800C3D68(s32 x) {
    s32 limit;

    limit = x + 0x11B3;
    func_80040100(limit);
}
"""
# addiu a0,a0,0x11B3; sll a0,a0,16; j func; sra a0,a0,16
NARROW_TARGET = _code([0x248411B3, 0x00042400, 0x08010040, 0x00042403])

# overlay2 800AA814 (reduced): the shared tail call the exact C wrote into each branch.
PAIR_TAIL = """void func_800AA814(s32 k) {
    if (k == 1) {
        D_8006BFA0 = 1;
    } else {
        D_8006BFA0 = 2;
    }
    func_80040100(D_8006BFA0);
}
"""

PAIR_DO = """void f(s32 n) {
    s32 i;

    i = n;
    do {
        func_80040100(i);
    } while (--i != 0);
}
"""


class FormTest(unittest.TestCase):
    def test_neighbouring_globals_become_one_array_or_struct(self):
        forms = dict(handforms.global_aggregate(PAIR_GLOBALS, "func_800A3294"))
        array = forms["global-aggregate: D_80087A70..D_80087A74 as s32 D_80087A70[2]"]
        self.assertIn("extern s32 D_80087A70[2];", array)
        self.assertNotIn("D_80087A74", array)
        self.assertIn("D_80087A70[1] = -D_80087A70[1];", array)
        self.assertIn("if (D_80087A70[0] != 0x7FFFFFFF)", array)
        struct_form = forms["global-aggregate: D_80087A70..D_80087A74 as struct"]
        self.assertIn("D_80087A70.unk_04 = -D_80087A70.unk_04;", struct_form)

    def test_lone_global_as_one_element_array(self):
        forms = dict(handforms.global_aggregate(PAIR_LONE, "func_800B2968"))
        text = forms["global-aggregate: D_8008A664 as u16 D_8008A664[1]"]
        self.assertIn("extern u16 D_8008A664[1];", text)
        self.assertIn("D_8008A664[0] |= 0x100;", text)

    def test_volatile_global(self):
        forms = dict(handforms.volatile_global(PAIR_LONE, "func_800B2968"))
        self.assertIn("extern volatile u16 D_8008A664;", forms["volatile-global: D_8008A664"])

    def test_chain_assign_and_split(self):
        forms = dict(handforms.chain_forms(PAIR_CHAIN, "func_8008D1B0"))
        joined = forms["chain-assign: D_800C5A60.unk_00[i] = D_800C5A60.unk_EA[i] = 0xFFFF"]
        self.assertIn("    D_800C5A60.unk_00[i] = D_800C5A60.unk_EA[i] = 0xFFFF;\n", joined)
        back = dict(handforms.chain_forms(joined, "func_8008D1B0"))
        self.assertTrue(any(k.startswith("chain-split") for k in back))
        self.assertIn(PAIR_CHAIN, back.values())

    def test_one_expression_split_into_compound_steps_and_back(self):
        forms = dict(handforms.split_compound(PAIR_SPLIT, "func_80040000"))
        split = forms["split-compound: length = (p[1] << 8) | p[0]"]
        self.assertIn("    length = p[1];\n    length <<= 8;\n    length |= p[0];\n", split)
        fold = dict(handforms.split_compound(split, "func_80040000"))["fold-compound: length"]
        self.assertIn("length = ((p[1]) << 8) | p[0];", fold)

    def test_narrow_local_from_the_target_extension(self):
        self.assertEqual(handforms.extension_widths(NARROW_TARGET), ["s16"])
        forms = dict(handforms.narrow_local(PAIR_NARROW, "func_800C3D68", NARROW_TARGET))
        self.assertIn("    s16 limit;", forms["narrow-local: limit s32 -> s16"])
        self.assertEqual(handforms.narrow_local(PAIR_NARROW, "func_800C3D68", _code([0x03E00008, 0])), [])

    def test_tail_written_per_branch_and_merged_back(self):
        forms = dict(handforms.tail_forms(PAIR_TAIL, "func_800AA814"))
        dup = forms["tail-duplicate: func_80040100(D_8006BFA0);"]
        self.assertEqual(dup.count("func_80040100(D_8006BFA0);"), 2)
        self.assertIn("        D_8006BFA0 = 2;\n        func_80040100(D_8006BFA0);\n    }\n}", dup)
        merged = dict(handforms.tail_forms(dup, "func_800AA814"))["tail-merge: func_80040100(D_8006BFA0);"]
        self.assertEqual(merged.count("func_80040100(D_8006BFA0);"), 1)

    def test_ternary_and_abs(self):
        src = "void f(s32 a) {\n    s32 x;\n\n    x = a > 3 ? 1 : 2;\n    if (a < 0) {\n        a = -a;\n    }\n    g(x, a);\n}\n"
        forms = dict(handforms.tail_forms(src, "f"))
        self.assertIn("    if (a > 3) {\n        x = 1;\n    } else {\n        x = 2;\n    }\n", forms["ternary-to-if: x"])
        self.assertIn("    a = abs(a);\n", forms["abs: a"])
        self.assertTrue(forms["abs: a"].startswith("int abs(int);"))

    def test_top_tested_loop(self):
        forms = handforms.top_tested(PAIR_DO, "f")
        self.assertEqual(len(forms), 1)
        self.assertIn("while (--i != 0) {", forms[0][1])
        self.assertNotIn("do {", forms[0][1])
        back = handforms.top_tested(forms[0][1], "f")
        self.assertIn("do {", back[0][1])

    def test_fixmul_signature(self):
        # mult a0,a1; mfhi t6; mflo t7; sll t6,t6,16; srl t7,t7,16; or v0,t6,t7; jr ra; nop
        fixmul = _code([0x00850018, 0x00007010, 0x00007812, 0x000E7400, 0x000F7C02, 0x01CF1025, 0x03E00008, 0])
        self.assertEqual(handforms.fixmul_sites(fixmul), [0])
        self.assertEqual(handforms.fixmul_sites(NARROW_TARGET), [])

    def test_the_search_has_the_forms_as_its_hand_family(self):
        found = [v for v in iterate.variants(PAIR_GLOBALS, "func_800A3294", ("hand",))]
        self.assertTrue(found and all(v.family == "hand" for v in found))
        self.assertIn("global-aggregate: D_80087A70..D_80087A74 as s32 D_80087A70[2]", [v.label for v in found])
        from n64dk.match import causes
        self.assertIn("hand", causes.FAMILIES_FOR["register.permutation"])

    def test_diagnosis_picks_the_forms(self):
        self.assertEqual(handforms.kinds_for(["schedule.delay_slot"])[0], "global-aggregate")
        self.assertIn("tail", handforms.kinds_for(["control.branch_form"]))
        self.assertIn("top-tested", handforms.kinds_for(["frame.size"]))
        self.assertEqual(handforms.kinds_for(["compiler.family"]), [])


# lw t0,0(a0); lw t1,4(a0); addu v0,t0,t1; jr ra; nop    and the same with t0 <-> t1
PERM_TARGET = _code([0x8C880000, 0x8C890004, 0x01091021, 0x03E00008, 0x00000000])
PERM_BUILT = _code([0x8C890000, 0x8C880004, 0x01281021, 0x03E00008, 0x00000000])


class ResolveStageTest(unittest.TestCase):
    def test_register_near_miss_reaches_the_array_form_by_itself(self):
        store = resolve.CaseStore(_scratch())
        self.addCleanup(store.close)
        seen = []

        def gcc(text):
            seen.append(text)
            exact = "extern s32 D_80087A70[2];" in text
            outcome = {"proof": {"verdict": "exact"}} if exact else None
            return iterate.Evaluation(0 if exact else 3, exact, {"counts": {} if exact else {"register": 3}},
                                      outcome=outcome), PERM_TARGET if exact else PERM_BUILT
        case = resolve.resolve(store, "main", "func_800A3294", BASE, PERM_TARGET, gcc, source=PAIR_GLOBALS,
                               jobs=1, pipeline="kmc+asn64", recipe_label="kmc+asn64 -O2", budget=60,
                               loop_forms=False)
        self.assertEqual(case.status, "exact")
        stage = next(s for s in case.stages if s["stage"] == "hand-forms")
        self.assertTrue(stage["chosen"][0].startswith("global-aggregate"))
        self.assertNotIn("iterate", [s["stage"] for s in case.stages])

    def test_a_closer_form_is_applied_to_the_search_result_not_before_it(self):
        store = resolve.CaseStore(_scratch())
        self.addCleanup(store.close)

        def gcc(text):
            score = 2 if "D_80087A70[2]" in text else 3
            return iterate.Evaluation(score, False, {"counts": {"register": score}}), PERM_BUILT
        case = resolve.resolve(store, "main", "func_800A3294", BASE, PERM_TARGET, gcc, source=PAIR_GLOBALS,
                               jobs=1, pipeline="kmc+asn64", recipe_label="kmc+asn64 -O2", budget=40,
                               loop_forms=False, search=False)
        stages = {s["stage"]: s for s in case.stages}
        self.assertEqual(stages["hand-forms"]["chosen"], [])           # not exact: the start stays the draft
        self.assertTrue(stages["hand-forms-best"]["chosen"][0].startswith("global-aggregate"))
        self.assertEqual(case.best_score, 2)

    def test_switched_off_the_stage_does_not_run(self):
        store = resolve.CaseStore(_scratch())
        self.addCleanup(store.close)

        def gcc(text):
            return iterate.Evaluation(3, False, {"counts": {"register": 3}}), PERM_BUILT
        case = resolve.resolve(store, "main", "func_800A3294", BASE, PERM_TARGET, gcc, source=PAIR_GLOBALS,
                               jobs=1, pipeline="kmc+asn64", recipe_label="kmc+asn64 -O2", budget=10,
                               loop_forms=False, hand_forms=False, search=False)
        self.assertNotIn("hand-forms", [s["stage"] for s in case.stages])


if __name__ == "__main__":
    unittest.main()
