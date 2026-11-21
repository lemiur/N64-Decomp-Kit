"""egraph: bounded equality saturation over C expressions (round 12 addendum C).

The rule, saturation and extraction tests parse with pycparser alone (no preprocessor, no
toolchain).  The whole-function tests go through ``astprep`` like the astforms tests and are
skipped when the host preprocessor is missing."""
import os
import shutil
import unittest
from unittest import mock

try:
    import pycparser  # noqa: F401
    HAVE_PYCPARSER = True
except ImportError:                                            # pragma: no cover
    HAVE_PYCPARSER = False

if HAVE_PYCPARSER:
    from n64dk.match import astprep, egraph, iterate

HAVE_CPP = shutil.which(os.environ.get("N64DK_CPP", "cpp")) is not None

PRELUDE = """typedef signed char s8; typedef unsigned char u8; typedef short s16;
typedef unsigned short u16; typedef int s32; typedef unsigned int u32; typedef float f32;
"""


def graph(source, which=0, **options):
    return egraph.statement_graph(PRELUDE + source, "f", which, **options)


def has(fn, eg, root, text, wrapper="ret"):
    return eg.represents(root, (wrapper, None, (egraph.parse_term(fn, text),)))


@unittest.skipUnless(HAVE_PYCPARSER, "pycparser not installed")
class SoundnessTest(unittest.TestCase):
    def test_signed_addition_is_not_reassociated_by_default(self):
        fn, eg, root, _ = graph("s32 f(s32 a, s32 b, s32 c) { return (a + b) + c; }")
        self.assertFalse(has(fn, eg, root, "a + (b + c)"))
        self.assertTrue(has(fn, eg, root, "c + (a + b)"))          # commutation is always sound

    def test_signed_reassociation_only_with_a_wrapping_dialect_and_flagged(self):
        fn, eg, root, _ = graph("s32 f(s32 a, s32 b, s32 c) { return (a + b) + c; }",
                                dialect=egraph.Dialect(signed_wrap=True))
        self.assertTrue(has(fn, eg, root, "a + (b + c)"))
        self.assertIn("assoc~wrap", set(eg.why.values()))

    def test_unsigned_addition_is_reassociated(self):
        fn, eg, root, _ = graph("u32 f(u32 a, u32 b, u32 c) { return (a + b) + c; }")
        self.assertTrue(has(fn, eg, root, "a + (b + c)"))

    def test_float_is_never_reassociated(self):
        fn, eg, root, _ = graph("f32 f(f32 a, f32 b, f32 c) { return (a + b) + c; }",
                                dialect=egraph.Dialect(signed_wrap=True))
        self.assertFalse(has(fn, eg, root, "a + (b + c)"))
        self.assertTrue(has(fn, eg, root, "c + (a + b)"))

    def test_float_comparison_is_not_negated(self):
        fn, eg, root, _ = graph("s32 f(f32 a, f32 b) { if (a < b) { return 1; } return 0; }")
        self.assertFalse(has(fn, eg, root, "!(a >= b)", "cond"))
        self.assertTrue(has(fn, eg, root, "b > a", "cond"))

    def test_signed_multiply_becomes_a_shift_only_when_non_negative(self):
        fn, eg, root, _ = graph("s32 f(s32 x) { return x * 8; }")
        self.assertFalse(has(fn, eg, root, "x << 3"))
        fn, eg, root, _ = graph("s32 f(u16 x) { return x * 8; }")
        self.assertTrue(has(fn, eg, root, "x << 3"))
        fn, eg, root, _ = graph("u32 f(u32 x) { return x << 3; }")
        self.assertTrue(has(fn, eg, root, "x * 8"))

    def test_signed_division_is_not_a_shift(self):
        fn, eg, root, _ = graph("s32 f(s32 x) { return x / 4; }")
        self.assertFalse(has(fn, eg, root, "x >> 2"))
        fn, eg, root, _ = graph("u32 f(u32 x) { return x / 4; }")
        self.assertTrue(has(fn, eg, root, "x >> 2"))
        self.assertTrue(has(fn, eg, root, "x / 4"))

    def test_volatile_expressions_are_untouched(self):
        fn, eg, root, _ = graph("extern volatile s32 v;\ns32 f(s32 x) { return v * 8 + x; }")
        self.assertEqual(egraph.forms(fn, eg, root), ["return v * 8 + x;"])
        fn, eg, root, _ = graph("extern volatile s32 v;\nvoid f(void) { v = v + 1; }")
        self.assertFalse(has(fn, eg, root, "v += 1", "stmt"))

    def test_side_effects_are_not_reordered_or_rewritten(self):
        fn, eg, root, _ = graph("extern s32 g(void);\ns32 f(s32 x) { return g() + x; }")
        self.assertFalse(has(fn, eg, root, "x + g()"))
        fn, eg, root, _ = graph("extern u32 h(void);\nu32 f(void) { return h() * 8; }")
        self.assertFalse(has(fn, eg, root, "h() << 3"))
        fn, eg, root, _ = graph("extern s32 g(void);\nvoid f(s32 *p) { *p = *p + g(); }")
        self.assertFalse(has(fn, eg, root, "*p += g()", "stmt"))

    def test_mask_and_cast_only_when_the_type_is_kept(self):
        fn, eg, root, _ = graph("s32 f(s32 x) { return x & 0xFF; }")
        self.assertTrue(has(fn, eg, root, "(u8) x"))
        fn, eg, root, _ = graph("u32 f(u32 x) { return x & 0xFF; }")
        self.assertFalse(has(fn, eg, root, "(u8) x"))

    def test_value_changing_casts_are_kept(self):
        fn, eg, root, _ = graph("s32 f(s32 x) { return (s16) x; }")
        self.assertFalse(has(fn, eg, root, "x"))
        fn, eg, root, _ = graph("s32 f(s16 x) { return (s32) x; }")
        self.assertTrue(has(fn, eg, root, "x"))

    def test_no_temporary_across_calls_or_out_of_a_guard(self):
        fn, eg, root, _ = graph("extern u32 g(u32);\nu32 f(u32 a, u32 b) { return g(a * b) + g(a * b); }")
        self.assertFalse(any("temp_eg" in t for t in egraph.forms(fn, eg, root)))
        fn, eg, root, _ = graph("s32 f(s32 *p) { return p && *p + *p; }")
        self.assertFalse(any("temp_eg" in t for t in egraph.forms(fn, eg, root)))


@unittest.skipUnless(HAVE_PYCPARSER, "pycparser not installed")
class RuleTest(unittest.TestCase):
    def test_comparison_inversions(self):
        fn, eg, root, _ = graph("s32 f(s32 a, s32 b) { return !(a < b); }")
        self.assertTrue(has(fn, eg, root, "a >= b"))
        self.assertTrue(has(fn, eg, root, "b <= a"))

    def test_boolean_normalisation(self):
        fn, eg, root, _ = graph("s32 f(s32 x) { return !!x; }")
        self.assertTrue(has(fn, eg, root, "x != 0"))
        fn, eg, root, _ = graph("s32 f(s32 x) { return x == 0; }")
        self.assertTrue(has(fn, eg, root, "!x"))

    def test_pointer_and_field_forms(self):
        fn, eg, root, _ = graph("s32 f(s32 *p, s32 i) { return *(p + i); }")
        self.assertTrue(has(fn, eg, root, "p[i]"))
        fn, eg, root, _ = graph("s32 *f(s32 *p, s32 i) { return p + i; }")
        self.assertTrue(has(fn, eg, root, "&p[i]"))
        fn, eg, root, _ = graph("struct S { s32 a; s32 b; };\ns32 f(struct S *p) { return (*p).b; }")
        self.assertTrue(has(fn, eg, root, "p->b"))

    def test_statement_forms(self):
        fn, eg, root, _ = graph("void f(void) { s32 x; x = x + 1; }")
        for text in ("x += 1", "x++", "++x", "x = x + 1"):
            self.assertTrue(has(fn, eg, root, text, "stmt"), text)
        fn, eg, root, _ = graph("void f(s32 *p) { *p = *p + 2; }")
        self.assertTrue(has(fn, eg, root, "*p += 2", "stmt"))

    def test_conditional_expression_and_if_else(self):
        fn, eg, root, _ = graph("void f(s32 c, s32 a, s32 b) { s32 x; x = c ? a : b; }")
        p = lambda t: egraph.parse_term(fn, t)
        branch = ("ifelse", None, (("cond", None, (p("c"),)), ("stmt", None, (p("x = a"),)),
                                   ("stmt", None, (p("x = b"),))))
        self.assertTrue(eg.represents(root, branch))
        fn, eg, root, _ = graph("void f(s32 c, s32 a, s32 b) { s32 x; if (c) { x = a; } else { x = b; } }")
        self.assertTrue(has(fn, eg, root, "x = c ? a : b", "stmt"))
        fn, eg, root, _ = graph("void f(s32 c, f32 a, s32 b) { f32 x; x = c ? a : b; }")
        self.assertFalse(any(text.startswith("if") for text in egraph.forms(fn, eg, root)))

    def test_temporary_for_a_repeated_pure_subexpression(self):
        fn, eg, root, _ = graph("u32 f(u32 a, u32 b) { return (a * b) + ((a * b) >> 2); }")
        self.assertTrue(any(t.startswith("temp_eg = a * b;") for t in egraph.forms(fn, eg, root)))

    def test_constant_construction_is_folded(self):
        fn, eg, root, _ = graph("u32 f(u32 x) { return x & (0x10000 - 1); }")
        self.assertTrue(has(fn, eg, root, "x & 0xFFFF"))


@unittest.skipUnless(HAVE_PYCPARSER, "pycparser not installed")
class SaturationTest(unittest.TestCase):
    SOURCE = "u32 f(u32 a, u32 b, u32 c, u32 d, u32 e, u32 g, u32 h) { return a + b + c + d + e + g + h; }"

    def test_original_is_kept(self):
        fn, eg, root, otree = graph("u32 f(u32 a, u32 b) { return (a * 8) + (b & 0xFF); }")
        self.assertGreater(eg.size, 10)                              # rewrites were added ...
        self.assertTrue(eg.represents(root, egraph.plain(otree)))    # ... and the original is still there
        self.assertTrue(has(fn, eg, root, "(a << 3) + (b & 0xFF)"))

    def test_bounded(self):
        limits = egraph.Limits(max_nodes=120, max_iterations=4)
        _fn, eg, _root, _ = graph(self.SOURCE, limits=limits)
        self.assertLessEqual(eg.size, 120 + 8)
        eg2 = egraph.EGraph()
        stats = egraph.saturate(eg2, limits)
        self.assertTrue(stats["saturated"])

    def test_deterministic(self):
        def snapshot():
            fn, eg, root, _ = graph(self.SOURCE, limits=egraph.Limits(max_nodes=200))
            return [(cid, eg.classes[cid].nodes) for cid in sorted(eg.classes)], egraph.forms(fn, eg, root)
        self.assertEqual(snapshot(), snapshot())

    def test_congruence_closure(self):
        eg = egraph.EGraph()
        x, y = eg.add("var", "x"), eg.add("var", "y")
        fx, fy = eg.add("neg", None, (x,)), eg.add("neg", None, (y,))
        self.assertNotEqual(eg.find(fx), eg.find(fy))
        eg.union(x, y)
        eg.rebuild()
        self.assertEqual(eg.find(fx), eg.find(fy))


@unittest.skipUnless(HAVE_PYCPARSER, "pycparser not installed")
class ExtractionTest(unittest.TestCase):
    def test_prefers_cheaper_forms(self):
        fn, eg, root, _ = graph("s32 f(s32 x) { return !!x; }")
        self.assertEqual(egraph.forms(fn, eg, root)[0], "return x != 0;")
        fn, eg, root, _ = graph("u32 f(u32 x) { return x & (0x10000 - 1); }")
        self.assertEqual(egraph.forms(fn, eg, root)[0], "return x & 0xFFFF;")

    def test_constant_construction_cost(self):
        self.assertEqual(egraph._materialize(0)[0], 0)
        self.assertEqual(egraph._materialize(0x7FFF), (1, {"li": 1}))
        self.assertEqual(egraph._materialize(0x10000), (1, {"lui": 1}))
        self.assertEqual(egraph._materialize(0x12345), (2, {"lui": 1, "or": 1}))
        self.assertEqual(egraph._immediate("&", 1, 0xFFFF, "rv"), (0, {}))
        self.assertEqual(egraph._immediate("+", 1, 0xFFFF, "rv")[0], 1)

    def test_target_profile_and_distance(self):
        words = [0x00041080, 0x3C011234, 0x34210001, 0x00851021, 0x00801021, 0x8FBF0014, 0x8C820000,
                 0x03E00008, 0x00000000]
        expected = {"shl": 1, "lui": 1, "or": 1, "add": 1, "load32s": 1}
        self.assertEqual(egraph.target_profile(words), expected)
        self.assertEqual(egraph.target_profile(b"".join(w.to_bytes(4, "big") for w in words)), expected)
        self.assertEqual(egraph.distance({"mul": 1}, {"shl": 1}), 2)
        self.assertEqual(egraph.distance({"shl": 1}, {"shl": 2}, {"shl": 1, "mul": 1}, {"mul": 1}), 0)

    def test_target_steers_the_choice(self):
        types, analysis, words, fdef = egraph.light_context(PRELUDE + "s32 f(s32 x) { return !!x; }", "f")
        fn = egraph.Function(types, analysis, egraph.Dialect(), egraph.Limits(), words)
        unit = egraph.EUnit("ret", fdef.body.block_items[0], 0, 0, 1, True)
        alts, _size = fn.solve(unit)
        self.assertGreater(len(alts), 1)
        plan = egraph._Plan([(unit, alts)], egraph.target_profile([0x0004102B]), None, egraph.Limits())
        best = min(plan.states(), key=lambda s: (plan.vector(s)["target_distance"], plan.vector(s)["cost"]))
        self.assertIn("x != 0", alts[best[0]].edits[0][2])

    def test_k_distinct_alternatives(self):
        types, analysis, words, fdef = egraph.light_context(
            PRELUDE + "u32 f(u32 a, u32 b, u32 c) { return (a + b) * 4 + c; }", "f")
        fn = egraph.Function(types, analysis, egraph.Dialect(), egraph.Limits(alternatives=4), words)
        unit = egraph.EUnit("ret", fdef.body.block_items[0], 0, 0, 1, True)
        alts, _size = fn.solve(unit)
        texts = [a.edits[0][2] for a in alts[1:]]
        self.assertTrue(1 <= len(texts) <= 4)
        self.assertEqual(len(texts), len(set(texts)))


WHOLE = PRELUDE + """extern s32 g(s32 v);

s32 work(s32 *p, s32 n, s32 c, s32 a, s32 b) {
    s32 x;
    s32 y;

    x = n + 1;
    x = x + 1;
    y = c ? a : b;
    *p = *p + 1;
    if (!(x < y)) {
        g(x);
    }
    return x * 8 + y;
}

void tail(void) {
}
"""

TEMPS = PRELUDE + """u32 twice(u32 a, u32 b) {
    return (a * b) + ((a * b) >> 2);
}
"""


@unittest.skipUnless(HAVE_PYCPARSER and HAVE_CPP, "pycparser or the host preprocessor missing")
class WholeFunctionTest(unittest.TestCase):
    def test_k_distinct_valid_variants(self):
        out = egraph.variants(WHOLE, "work", k=6)
        self.assertTrue(1 <= len(out) <= 6)
        self.assertEqual(len(out), len(set(out)))
        start, end = WHOLE.index("s32 work("), WHOLE.index("void tail")
        for text in out:
            self.assertNotEqual(text, WHOLE)
            self.assertTrue(text.startswith(WHOLE[:start]))
            self.assertTrue(text.endswith(WHOLE[end:]))
            astprep.prepare(text, "work")                      # round trip: valid, mappable C

    def test_statement_rewrites_reach_the_source(self):
        found = egraph.candidates(WHOLE, "work", k=24)
        texts = "\n".join(c.source for c in found)
        self.assertIn("x += 1;", texts)
        self.assertIn("if (c) {", texts)
        self.assertTrue(all(c.label.startswith("egraph: ") for c in found))
        self.assertFalse(any(c.review for c in found))

    def test_temporary_is_declared(self):
        found = egraph.variants(TEMPS, "twice", k=8)
        with_temp = [t for t in found if "temp_eg = a * b;" in t]
        self.assertTrue(with_temp)
        self.assertIn("u32 temp_eg;", with_temp[0])

    def test_opt_in_hook(self):
        with mock.patch.dict(os.environ, {"N64DK_EGRAPH": ""}):
            self.assertFalse([v for v in iterate.variants(WHOLE, "work", ("types",)) if v.family == "egraph"])
            self.assertTrue(all(v.family == "egraph" for v in iterate.variants(WHOLE, "work", ("egraph",))))
        with mock.patch.dict(os.environ, {"N64DK_EGRAPH": "1"}):
            self.assertTrue([v for v in iterate.variants(WHOLE, "work", ("types",)) if v.family == "egraph"])

    def test_unparseable_source_gives_nothing(self):
        self.assertEqual(egraph.variants("int f( {", "f"), [])


if __name__ == "__main__":
    unittest.main()
