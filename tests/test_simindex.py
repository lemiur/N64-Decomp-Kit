import dataclasses
import struct
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from n64dk.match import ir, simindex

BASE = 0x80001000
TARGET = 0x80001234
ZERO, V0, A0, A1 = 0, 2, 4, 5
T0, T1, T6, T7, S0, S1, SP, RA = 8, 9, 14, 15, 16, 17, 29, 31


def R(fn, rs=0, rt=0, rd=0, sa=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | fn


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def J(op, target):
    return (op << 26) | ((target >> 2) & 0x3FFFFFF)


NOP, JR_RA = 0, R(8, rs=RA)


def code(words):
    return b"".join(struct.pack(">I", w) for w in words)


def body(regs=(T6, T7), hi=0x8010, lo=0x1234, call=0x80002000, const=3):
    a, b = regs
    return [I(9, SP, SP, -24), I(43, SP, RA, 20),
            I(15, 0, a, hi),              # lui   a, %hi(sym)
            I(35, a, b, lo),              # lw    b, %lo(sym)(a)
            I(35, A0, A1, 8),             # lw    a1, 8(a0)        (a field of a pointer argument)
            I(9, b, A0, const),           # addiu a0, b, const
            J(3, call), NOP,
            I(35, SP, RA, 20), JR_RA, I(9, SP, SP, 24)]


def original():
    """tests/test_ir.py's function: and the same semantics with another allocation and schedule."""
    return [I(9, SP, SP, -24), I(43, SP, RA, 20), I(43, SP, S0, 16), R(37, A0, ZERO, S0), I(35, A0, T6, 4),
            I(9, T6, T7, 3), I(43, S0, ZERO, 12), I(43, S0, T7, 8), J(3, TARGET), I(9, T6, A0, 1),
            I(10, V0, T0, 10), I(4, T0, ZERO, 2), NOP, I(43, S0, V0, 0), I(35, SP, RA, 20), I(35, SP, S0, 16),
            JR_RA, I(9, SP, SP, 24)]


def rescheduled():
    return [I(9, SP, SP, -32), I(43, SP, S1, 24), I(43, SP, RA, 28), I(35, A0, T1, 4), R(37, A0, ZERO, S1),
            I(9, T1, 10, 3), I(43, S1, 10, 8), I(43, S1, ZERO, 12), J(3, TARGET), I(9, T1, A0, 1),
            I(10, V0, 11, 10), I(4, 11, ZERO, 2), NOP, I(43, S1, V0, 0), I(35, SP, RA, 28), I(35, SP, S1, 24),
            JR_RA, I(9, SP, SP, 32)]


def entry(title, words, name="f", vram=BASE, **kw):
    return simindex.describe(title, name, vram, code(words), **kw)


def best(index, query, **kw):
    rows = index.query(query, **kw)
    return rows[0] if rows else None


class LevelTest(unittest.TestCase):
    def test_exact(self):
        index = simindex.Index()
        index.add(entry("a", body()))
        hit = best(index, entry("b", body()))
        self.assertEqual((hit["class"], hit["level"], hit["confidence"]), ("exact", "exact", "high"))
        self.assertFalse(hit["proof"])

    def test_relocation_only(self):
        a, b = entry("a", body()), entry("b", body(hi=0x8020, lo=0x0040, call=0x80009000))
        self.assertNotEqual(a.exact, b.exact)
        self.assertEqual(a.masked, b.masked)
        index = simindex.Index()
        index.add(a)
        self.assertEqual(best(index, b)["class"], "relocation-only")

    def test_register_renamed_is_equal_only_at_the_renamed_level(self):
        a, b = entry("a", body()), entry("b", body(regs=(T0, T1)))
        self.assertNotEqual(a.exact, b.exact)
        self.assertNotEqual(a.masked, b.masked)
        self.assertEqual(a.renamed, b.renamed)
        index = simindex.Index()
        index.add(a)
        hit = best(index, b)
        self.assertEqual((hit["class"], hit["level"]), ("register-renamed", "register-renamed"))
        # a changed constant is not a renaming
        self.assertNotEqual(a.renamed, entry("c", body(const=5)).renamed)

    def test_same_semantics_other_schedule(self):
        a, b = entry("a", original()), entry("b", rescheduled())
        self.assertTrue(a.lifted and b.lifted)
        self.assertNotEqual(a.renamed, b.renamed)
        self.assertTrue(a.semantic)
        self.assertEqual(a.semantic, b.semantic)
        index = simindex.Index()
        index.add(a)
        hit = best(index, b)
        self.assertEqual(hit["class"], "same-source-other-compiler")
        self.assertEqual(hit["metrics"]["semantic"], 1.0)
        self.assertLess(hit["metrics"]["structural"], 1.0)

    def test_a_failed_lift_falls_back_to_the_other_levels(self):
        with mock.patch.object(ir, "lift", side_effect=RecursionError("deep")):
            e = entry("a", body())
        self.assertFalse(e.lifted)
        self.assertEqual(e.semantic, "")
        self.assertNotIn("sem", e.feats)
        self.assertTrue(e.exact and e.masked and e.renamed and e.feats["mem"])

    def test_without_a_lift_the_semantic_metric_is_absent(self):
        index = simindex.Index()
        index.add(entry("a", body(), lift=False))
        hit = best(index, entry("b", body(const=5), lift=False))
        self.assertIsNone(hit["metrics"]["semantic"])
        self.assertEqual(hit["metrics"]["structural"], 1.0)
        self.assertEqual(hit["class"], "modified")


class PenaltyAndLibraryTest(unittest.TestCase):
    def test_small_functions_are_penalised(self):
        stub = [JR_RA, I(9, ZERO, V0, 1)]                      # return 1
        index = simindex.Index()
        index.add(entry("a", stub))
        hit = best(index, entry("b", stub))
        self.assertEqual(hit["class"], "exact")
        self.assertTrue(hit["small"])
        self.assertIn(hit["confidence"], ("low", "none"))
        self.assertLess(hit["score"], 0.5)

    def test_library_needs_enough_titles(self):
        index = simindex.Index(library_titles=5)
        for title in "abcd":
            index.add(entry(title, body()))
        self.assertEqual(best(index, entry("z", body()))["class"], "exact")      # 4 titles: lineage
        index.add(entry("e", body()))
        hit = best(index, entry("z", body()))
        self.assertEqual((hit["class"], hit["level"]), ("likely-library", "exact"))
        self.assertTrue(hit["library"])
        self.assertEqual(hit["titles_carrying"], 5)


class FeatureTest(unittest.TestCase):
    def test_strings_through_hi_lo(self):
        def data_at(address, length):
            return b"Hello world\0" + bytes(length) if address == 0x80105678 else None

        def user(extra):
            return [I(9, SP, SP, -24), I(43, SP, RA, 20), I(15, 0, A0, 0x8010), J(3, TARGET), I(9, A0, A0, 0x5678),
                    *extra, I(35, SP, RA, 20), JR_RA, I(9, SP, SP, 24)]

        a = entry("a", user([I(35, A0, T6, 4), I(43, A0, T6, 8)]), data_at=data_at)
        b = entry("b", user([I(9, ZERO, T6, 77), I(43, A1, T6, 0), I(43, A1, T6, 4), I(43, A1, T6, 12)]),
                  data_at=data_at)
        self.assertIn(simindex._h("s:Hello world"), a.feats["str"])
        index = simindex.Index()
        index.add(a)
        hit = best(index, b)
        self.assertEqual(hit["metrics"]["strings"], 1.0)
        self.assertGreater(hit["provenance"]["shared_features"]["str"], 0)
        self.assertNotIn("str", entry("c", user([])).feats)    # no lookup, no strings

    def test_memory_shapes(self):
        e = entry("a", body())
        self.assertIn(simindex._h("m:ld4:p8#1"), e.feats["mem"])      # lw a1, 8(a0)
        self.assertIn(simindex._h("m:ld4:abs#1"), e.feats["mem"])     # lw b, %lo(sym)(a)
        self.assertIn(simindex._h("m:st4:sp#1"), e.feats["mem"])

    def test_loops_and_switches(self):
        loop = [I(9, ZERO, T0, 0), I(9, T0, T0, 1), I(10, T0, T1, 10), I(5, T1, ZERO, -3 & 0xFFFF), NOP,
                R(8, rs=T6), NOP, JR_RA, NOP]
        e = entry("a", loop)
        self.assertIn(simindex._h("l:back-edge#1"), e.feats["loop"])
        self.assertIn(simindex._h("l:indirect-jump#1"), e.feats["loop"])

    def test_neighbourhoods_from_an_image(self):
        callee = body(call=0x80003000)
        caller = body(call=0x80000000)
        data = code(callee) + bytes(0x40 - 4 * len(callee)) + code(caller)
        rows = [("callee", 0x80000000, 4 * len(callee)), ("caller", 0x80000040, 4 * len(caller))]
        index = simindex.Index()
        self.assertEqual(simindex.index_image(index, "t", "main", data, 0x80000000, rows), 2)
        by_name = {e.name: e for e in index.entries}
        self.assertTrue(by_name["callee"].feats.get("neigh"))     # its caller
        self.assertTrue(by_name["caller"].feats.get("neigh"))     # its callee
        self.assertEqual(by_name["caller"].callees, 1)


class IndexTest(unittest.TestCase):
    def test_never_returns_the_query_and_respects_title_filters(self):
        index = simindex.Index()
        a = entry("a", body())
        index.add(a)
        index.add(entry("b", body()))
        self.assertEqual([r["title"] for r in index.query(a)], ["b"])
        self.assertEqual(index.query(a, exclude_titles=["b"]), [])

    def test_postings_are_capped_but_frequencies_kept(self):
        index = simindex.Index(max_posting=2)
        for title in "abc":
            index.add(entry(title, body()))
        feature = index.entries[0].feats["mem"][0]
        self.assertIsNone(index.postings[feature])
        self.assertEqual(index.df[feature], 3)
        self.assertEqual(len(index.levels["exact"][index.entries[0].exact]), 2)

    def test_save_and_load_round_trip(self):
        index = simindex.Index(library_titles=3)
        for title, words in (("a", body()), ("b", body(regs=(T0, T1))), ("c", original())):
            index.add(entry(title, words))
        query = entry("q", body(hi=0x8030))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.jsonl"
            index.save(path)
            loaded = simindex.Index.load(path)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded.library_titles, 3)
        self.assertEqual(index.query(query), loaded.query(query))


def loop_and_switch():
    return [I(9, ZERO, T0, 0), I(9, T0, T0, 1), I(10, T0, T1, 10), I(5, T1, ZERO, -3 & 0xFFFF), NOP,
            R(8, rs=T6), NOP, JR_RA, NOP]


def dead_self_loop():
    """``jr ra`` then an unreachable block whose only predecessor is itself (``b .``), then a block
    whose only predecessor is that loop and which reads a0 (in the delay slot of its ``jr ra``)."""
    return [JR_RA, NOP, NOP, I(4, ZERO, ZERO, -2 & 0xFFFF), NOP, JR_RA, R(33, rs=A0, rt=ZERO, rd=V0)]


def dead_two_block_cycle():
    """X: ``j Y`` / Y: ``b X``, both unreachable and each the other's only predecessor; the block
    after Y (only predecessor Y) reads a0."""
    return [JR_RA, NOP, J(2, BASE + 16), NOP, I(4, ZERO, ZERO, -3 & 0xFFFF), NOP, JR_RA,
            R(33, rs=A0, rt=ZERO, rd=V0)]


def reference_lifted(words, vram=BASE):
    """The derivation describe() used before the IR was dropped inside _semantic: straight from a
    Fingerprint (the oracle for the compacted path)."""
    fp = ir.fingerprint(ir.lift(simindex._words(code(words)), vram), symbolic_addresses=True)
    sem = simindex._multiset("e", fp.effects) + simindex._multiset("o", fp.ops) + simindex._multiset("im", fp.memory)
    nodes = Counter((kind, len(t), any(x <= n for x in t)) for n, (kind, t) in enumerate(fp.cfg))
    cfg = simindex._multiset("n", Counter({repr(k): v for k, v in nodes.items()})) + [simindex._h(f"c:ir:{fp.cfg!r}")]
    return fp.digest[:16] if fp.complete else "", sem, cfg


ALLOWED = (str, int, bool, tuple, dict)
FUNCTIONS = {"body": body, "original": original, "rescheduled": rescheduled, "loop": loop_and_switch,
             "dead_self_loop": dead_self_loop, "dead_two_block_cycle": dead_two_block_cycle}


class CompactionTest(unittest.TestCase):
    def test_lifted_features_are_identical_to_the_fingerprint_derivation(self):
        for label, make in FUNCTIONS.items():
            with self.subTest(label):
                words = make()
                semantic, sem, cfg = reference_lifted(words)
                lifted, plain = entry("a", words), entry("a", words, lift=False)
                self.assertTrue(lifted.lifted)
                self.assertEqual(lifted.semantic, semantic)
                self.assertEqual(lifted.feats["sem"], simindex._bottom(sem, simindex.CAPS["sem"]))
                self.assertEqual(lifted.feats["cfg"], simindex._bottom(list(plain.feats["cfg"]) + cfg,
                                                                       simindex.CAPS["cfg"]))

    def test_lifting_leaves_every_unlifted_feature_unchanged(self):
        for label, make in FUNCTIONS.items():
            with self.subTest(label):
                lifted, plain = entry("a", make()), entry("a", make(), lift=False)
                for name in ("key", "n", "exact", "masked", "renamed", "tokens", "callees"):
                    self.assertEqual(getattr(lifted, name), getattr(plain, name))
                for kind in simindex.KINDS:
                    if kind not in ("sem", "cfg"):
                        self.assertEqual(lifted.feats.get(kind), plain.feats.get(kind))
                self.assertFalse(plain.lifted)
                self.assertNotIn("sem", plain.feats)

    def test_the_budget_does_not_change_a_lift_that_fits(self):
        for label, make in FUNCTIONS.items():
            with self.subTest(label):
                words = simindex._words(code(make()))
                free = ir.fingerprint(ir.lift(words, BASE), symbolic_addresses=True)
                bounded = ir.fingerprint(ir.lift(words, BASE, max_values=simindex.LIFT_VALUES_PER_WORD * len(words)
                                                 + simindex.LIFT_VALUES_BASE), symbolic_addresses=True)
                self.assertEqual(free.digest, bounded.digest)

    def test_entries_hold_no_ir_objects(self):
        def walk(value, path):
            self.assertIsInstance(value, ALLOWED, path)
            self.assertNotIsInstance(value, Counter, path)
            if isinstance(value, tuple):
                for n, item in enumerate(value):
                    self.assertIsInstance(item, int, f"{path}[{n}]")
            elif isinstance(value, dict):
                for kind, item in value.items():
                    self.assertIn(kind, simindex.KINDS, path)
                    self.assertIsInstance(item, tuple, f"{path}.{kind}")
                    walk(item, f"{path}.{kind}")

        for label, make in FUNCTIONS.items():
            with self.subTest(label):
                e = entry("a", make())
                self.assertTrue(e.lifted)
                self.assertFalse(hasattr(e, "__dict__"))         # slots: no per-entry attribute dict
                for f in dataclasses.fields(e):
                    walk(getattr(e, f.name), f.name)
                for kind, values in e.feats.items():
                    self.assertLessEqual(len(values), simindex.CAPS[kind])
                self.assertLessEqual(len(e.tokens), simindex.MAX_TOKENS)

    def test_semantic_returns_only_hashes(self):
        semantic, sem, cfg = simindex._semantic(simindex._words(code(original())), BASE)
        self.assertIsInstance(semantic, str)
        self.assertTrue(all(isinstance(x, int) for x in sem + cfg))


class LiftBoundTest(unittest.TestCase):
    def test_a_cycle_of_single_predecessor_blocks_terminates_as_undef(self):
        # Before the fix ir._Lifter.read walked such a cycle forever, appending to a list until
        # memory ran out (run this suite only with the fix applied, inside the capped job wrapper).
        for label, make, reader in (("self-loop", dead_self_loop, 2), ("two-block", dead_two_block_cycle, 3)):
            with self.subTest(label):
                fn = ir.lift(make(), BASE)
                (ret,) = [s for s in fn.blocks[reader].stmts if s.kind == "ret"]
                self.assertEqual(fn.value(ret.args[0]).op, "undef")      # v0 = a0 read through dead code

    def test_the_value_budget_aborts_a_lift_and_the_entry_falls_back(self):
        with self.assertRaises(ir.LiftBudgetExceeded):
            ir.lift(original(), BASE, max_values=8)
        before = simindex.LIFT_STATS["failed:LiftBudgetExceeded"]
        with mock.patch.object(simindex, "LIFT_VALUES_PER_WORD", 0), \
                mock.patch.object(simindex, "LIFT_VALUES_BASE", 8):
            e = entry("a", original())
        self.assertFalse(e.lifted)
        self.assertNotIn("sem", e.feats)
        self.assertEqual(simindex.LIFT_STATS["failed:LiftBudgetExceeded"], before + 1)
        plain = entry("a", original(), lift=False)
        self.assertEqual((e.exact, e.masked, e.renamed, e.feats), (plain.exact, plain.masked, plain.renamed, plain.feats))


if __name__ == "__main__":
    unittest.main()
