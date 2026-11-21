"""Matching-search lessons of the Sesame Street work (synthetic inputs only).

L1  compiler version per library/region: a near miss is tried under the sibling versions of its
    compiler before the source is permuted; version-insensitive functions are no evidence.
L2  local-free form for IDO: cached field copies are read in place.
L3  GCC 2.7.2 register ranking: goto-built loops <-> structured loops on register permutations.
L4  real names resolve to one address and agree with a known body identity."""
import atexit
import shutil
import struct
import tempfile
import unittest

from n64dk.match import compilers, draftforms, iterate, resolve

BASE = 0x80001000


def _code(words):
    return b"".join(struct.pack(">I", w) for w in words)


TARGET = _code([0x84820004, 0x03E00008, 0x00000000])
WRONG = _code([0x94820004, 0x03E00008, 0x00000000])           # same size, one instruction differs


def _scratch():
    path = tempfile.mkdtemp(prefix="n64dk-lessons-")
    atexit.register(shutil.rmtree, path, True)
    return path


CACHED = """typedef struct Ring { unsigned int base; unsigned int size; unsigned int head; unsigned int tail; int mode; } Ring;
void feed(Ring *r, int n);

void pump(Ring *r, int n) {
    unsigned int head;
    unsigned int size;
    unsigned int used;

    head = r->head;
    used = head - r->tail;
    size = r->size;
    while (used < size - 0xA0 && n > 0) {
        feed(r, (head % size) >> 2);
        head = r->head += 0xA0;
        used = head - r->tail;
        size = r->size;
        n--;
    }
    if (r->mode == 1 && used >= size - 0x140) {
        r->mode = 2;
    }
}
"""


class SiblingVersionTest(unittest.TestCase):
    def setUp(self):
        self.store = resolve.CaseStore(_scratch())
        self.addCleanup(self.store.close)

    def test_siblings(self):
        self.assertEqual(compilers.siblings("ido5.3"), ("ido7.1",))
        self.assertEqual(compilers.siblings("kmc+asn64"), ("kmc+as",))
        self.assertTrue(compilers.sibling_recipe("ido5.3 -O2 -mips2", "ido7.1 -O2 -mips2"))
        self.assertFalse(compilers.sibling_recipe("ido5.3 -O2 -mips2", "ido7.1 -O1 -mips2"))
        self.assertTrue(compilers.sibling_recipe("ido5.3 -O2 -mips2", "ido7.1 -O1 -mips2", levels=("-O1",)))
        self.assertFalse(compilers.sibling_recipe("ido5.3 -O2 -mips2", "kmc+as -O2 -mips2"))

    def test_near_miss_is_tried_under_the_sibling_version_before_permutation(self):
        used = []

        def ido53(text):
            used.append("5.3")
            return iterate.Evaluation(2, False, {"counts": {"register": 2}, "entries": []}), WRONG

        def ido71(text):
            used.append("7.1")
            return iterate.Evaluation(0, True, {"counts": {}}, outcome={"proof": {"verdict": "exact"}}), TARGET

        def gcc(text):
            used.append("gcc")
            return iterate.Evaluation(9, False, {"counts": {}}), WRONG
        source = "s32 read(Thing *thing) {\n    return thing->field;\n}\n"
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, ido53, source=source, jobs=1,
                               pipeline="ido5.3", recipe_label="ido5.3 -O2 -mips2",
                               recipes=lambda family: [("kmc+as -O2 -mips2", "kmc+as", gcc),
                                                       ("ido7.1 -O2 -mips2", "ido7.1", ido71)])
        self.assertEqual(case.status, "exact")
        self.assertEqual(case.recipe, "ido7.1 -O2 -mips2")
        stages = [s["stage"] for s in case.stages]
        self.assertIn("near-siblings", stages)
        self.assertNotIn("iterate", stages)
        self.assertNotIn("gcc", used)

    def test_calibration_sample_records_the_versions_it_matches_under(self):
        def both(text):
            return iterate.Evaluation(0, True, {"counts": {}}, outcome={"proof": {"verdict": "exact"}}), TARGET
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, both, source="s32 read(void) {}\n", jobs=1,
                               pipeline="ido5.3", recipe_label="ido5.3 -O2", provisional=True,
                               recipes=lambda family: [("ido7.1 -O2", "ido7.1", both)])
        check = [s for s in case.stages if s["stage"] == "version-check"][0]
        self.assertEqual(check["exact_under"], ["ido7.1 -O2"])

    def test_version_insensitive_functions_do_not_lock_the_version(self):
        from n64dk.commands.run import recipe_evidence

        class Case:
            def __init__(self, recipe, under):
                self.recipe, self.stages = recipe, [{"stage": "version-check", "exact_under": under}]
        cases = [Case("ido5.3 -O2", ["ido7.1 -O2"]) for _ in range(22)] + [Case("ido7.1 -O2", [])] * 3
        evidence = recipe_evidence(cases)
        self.assertEqual(evidence["votes"], {"ido7.1 -O2": 3})
        self.assertEqual((evidence["insensitive"], evidence["discriminating"]), (22, 3))


class LocalFreeTest(unittest.TestCase):
    def test_cached_field_copies_are_read_in_place(self):
        text, applied = draftforms.local_free(CACHED, "pump", refreshed_only=True)
        self.assertTrue(applied)
        for name in ("head", "used", "size"):
            self.assertNotRegex(text.split("void pump")[1], rf"(?<!->)\b{name}\b")
        self.assertIn("while ((r->head - r->tail) < r->size - 0xA0 && n > 0) {", text)
        self.assertIn("feed(r, (r->head % r->size) >> 2);", text)
        self.assertIn("        r->head += 0xA0;\n        n--;", text)
        self.assertIn("(r->head - r->tail) >= r->size - 0x140", text)

    def test_store_through_a_copy_is_folded(self):
        source = "void f(T *s) {\n    u32 x;\n\n    x = s->a + 4;\n    s->a = x;\n    g(x);\n}\n"
        text, _applied = draftforms.local_free(source, "f")
        self.assertIn("s->a += 4;", text)
        self.assertIn("g(s->a);", text)
        self.assertNotIn("u32 x;", text)

    def test_a_local_read_before_it_is_set_or_written_otherwise_is_kept(self):
        source = "void f(T *s) {\n    u32 x;\n\n    g(x);\n    x = s->a;\n    g(x);\n}\n"
        self.assertEqual(draftforms.local_free(source, "f"), (source, []))
        source = "void f(T *s) {\n    u32 x;\n\n    x = s->a;\n    x++;\n    g(x);\n}\n"
        self.assertEqual(draftforms.local_free(source, "f"), (source, []))

    def test_local_free_form_joins_the_portfolio(self):
        self.assertIn("local-free", [label for label, _t, _r in draftforms.drafts(CACHED, "pump")])

    def test_ido_near_miss_reaches_the_local_free_form_by_itself(self):
        store = resolve.CaseStore(_scratch())
        self.addCleanup(store.close)

        def ido(text):
            exact = "unsigned int used;" not in text
            outcome = {"proof": {"verdict": "exact"}} if exact else None
            return iterate.Evaluation(0 if exact else 4, exact, {"counts": {} if exact else {"register": 4}},
                                      outcome=outcome), TARGET if exact else WRONG
        case = resolve.resolve(store, "main", "pump", BASE, TARGET, ido, source=CACHED, jobs=1,
                               pipeline="ido7.1", recipe_label="ido7.1 -O2", budget=6)
        self.assertEqual(case.status, "exact")
        self.assertNotIn("used", store.source(case.best_digest).split("void pump")[1])


GOTO_COPY = """char *copy(char *dst, const char *src) {
    char *d = dst;

loop:
    {
        u32 w = *(const u32 *)src;
        src += 4;
        if (!(w & 0xFF000000)) { *d = 0; return dst; }
        *(u32 *)d = w;
        d += 4;
        if (w & 0xFF) goto loop;
    }
    return dst;
}
"""

# lw t0,0(a0); lw t1,4(a0); addu v0,t0,t1; jr ra; nop    and the same with t0 <-> t1
PERM_TARGET = _code([0x8C880000, 0x8C890004, 0x01091021, 0x03E00008, 0x00000000])
PERM_BUILT = _code([0x8C890000, 0x8C880004, 0x01281021, 0x03E00008, 0x00000000])


class LoopFormTest(unittest.TestCase):
    def test_goto_loop_becomes_a_structured_loop_and_back(self):
        from n64dk.match import loopforms
        forms = dict(loopforms.variants(GOTO_COPY, "copy"))
        hoisted = forms["loop-form: goto -> while (1) with return, declarations hoisted"]
        self.assertIn("    u32 w;\n", hoisted)
        self.assertIn("while (1) {", hoisted)
        self.assertIn("if (!(w & 0xFF)) return dst;", hoisted)
        self.assertNotIn("goto", hoisted)
        self.assertNotIn("goto", forms["loop-form: goto -> do-while"])
        back = dict(loopforms.variants(hoisted, "copy"))["loop-form: while (1) -> goto"]
        self.assertIn("if (w & 0xFF) goto loop;", back)
        self.assertTrue(back.rstrip().endswith("return dst;\n}"))

    def test_register_permutation_triggers_the_loop_rewrite(self):
        from n64dk.match import causes
        ranked = causes.diagnose(PERM_TARGET, PERM_BUILT, BASE)
        self.assertEqual(ranked[0].id, "register.permutation")
        store = resolve.CaseStore(_scratch())
        self.addCleanup(store.close)

        def gcc(text):
            exact = "while (1)" in text and "goto" not in text
            outcome = {"proof": {"verdict": "exact"}} if exact else None
            return iterate.Evaluation(0 if exact else 3, exact, {"counts": {} if exact else {"register": 3}},
                                      outcome=outcome), PERM_TARGET if exact else PERM_BUILT
        case = resolve.resolve(store, "main", "copy", BASE, PERM_TARGET, gcc, source=GOTO_COPY, jobs=1,
                               pipeline="kmc+asn64", recipe_label="kmc+asn64 -O1", budget=8)
        self.assertEqual(case.status, "exact")
        self.assertIn("loop-form", [s["stage"] for s in case.stages])
        self.assertNotIn("iterate", [s["stage"] for s in case.stages])

    def test_priority_rule_and_the_dossier_says_who_must_gain_a_reference(self):
        from n64dk.match import causes
        self.assertEqual(causes.gcc_priority(4, 8), 2 * 4 / 8)
        self.assertEqual(causes.gcc_priority(4, 8, loop_depth=1), 3 * 8 / 8)      # refs count per loop level
        self.assertEqual(causes.gcc_priority(1, 5), 0.0)
        cause = causes.diagnose(PERM_TARGET, PERM_BUILT, BASE)[0]
        self.assertIn("floor_log2(refs)*refs/live_length", cause.next)
        self.assertIn("goto-built loop is no loop", cause.next)
        self.assertRegex(" ".join(cause.evidence), r"the value in \$t1 .* belongs in \$t0: it must gain priority")
        self.assertRegex(" ".join(cause.evidence), r"the value in \$t0 .* belongs in \$t1: it must lose priority")


class RealNameTest(unittest.TestCase):
    def test_a_name_with_two_addresses_is_flagged(self):
        from n64dk.match import symbols
        syms = symbols.SymbolMap()
        symbols.load_splat("sin = 0x80001000;\ncos = 0x80001100;\nsin = 0x80002000;\n", syms)
        findings = symbols.check_names(syms)
        self.assertEqual([(f["kind"], f["name"]) for f in findings], [("ambiguous-name", "sin")])
        with self.assertRaises(symbols.NameError_):
            symbols.require_consistent_names(syms)

    def test_a_swapped_pair_is_refused_and_a_mismatch_reported(self):
        from n64dk.match import symbols
        syms = symbols.SymbolMap()
        symbols.load_splat("sin = 0x80001100;\ncos = 0x80001000;\nsqrtf = 0x80003000;\n", syms)
        identities = {0x80001000: "sin", 0x80001100: "cos", 0x80003000: "sinf"}
        findings = symbols.check_names(syms, identities)
        self.assertEqual(findings[0]["kind"], "swapped-pair")
        self.assertEqual(findings[0]["names"], ["cos", "sin"])
        with self.assertRaises(symbols.NameError_):
            symbols.require_consistent_names(syms, identities)
        agreeing = symbols.SymbolMap()
        symbols.load_splat("sin = 0x80001000;\ncos = 0x80001100;\n", agreeing)
        self.assertEqual(symbols.require_consistent_names(agreeing, identities), [])
        names = "vram\tcanonical\tdisplay\tdisplay_from\toriginal\tsdk\n0x80001000\tfunc_80001000\tsin\tsdk\t\tsin [sdk signature]\n"
        self.assertEqual(symbols.identities_from_names(names), {0x80001000: "sin"})


if __name__ == "__main__":
    unittest.main()
