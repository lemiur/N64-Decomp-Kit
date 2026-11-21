"""Reason codes (match.reasons): why an exported function is not settled, from the run's records only."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from n64dk.commands import export
from n64dk.match import reasons
from n64dk.match.resolve import Case


def case(status="open", *, score=20, top="", causes=(), stages=(), recipe="ido5.3 -O2", size=0x80, strict=None,
         next_step="", vram=0x80001000):
    return Case(f"main:{vram:08X}:{size:X}", "main", f"func_{vram:08X}", vram, size, "0" * 64, status=status,
                top_cause=top, causes=[{"cause": c, "confidence": conf, "next": f"hint for {c}"} for c, conf in causes],
                best_score=score, stages=list(stages), recipe=recipe, strict=strict or {}, next=next_step)


def code(outcome, item=None, **extra):
    return reasons.code_of(outcome, item, **extra)[0]


class SettledAndAssumedTest(unittest.TestCase):
    def test_settled_outcomes_carry_no_code(self):
        for outcome in ("exact", "donor", "donor-assembly", "assembly"):
            self.assertEqual(reasons.explain(outcome, case("exact", score=0)), ("", ""))

    def test_an_assumed_symbol_names_the_symbol(self):
        item = case("assumed", score=0, strict={"verdict": "assumed", "assumptions": [
            {"kind": "symbol", "name": "D_80123456", "address": "0x80123456"}]})
        found, step = reasons.explain("awaiting-symbols", item)
        self.assertEqual(found, "needs-symbol")
        self.assertIn("D_80123456", step)

    def test_an_assumed_section_base_is_data_not_owned(self):
        item = case("assumed", score=0, strict={"assumptions": [
            {"kind": "symbol", "name": "D_80123456"}, {"kind": "section", "name": ".rodata"}]})
        found, step = reasons.explain("awaiting-symbols", item)
        self.assertEqual(found, "data-not-owned")
        self.assertIn(".rodata", step)

    def test_donor_awaiting_symbols_without_a_case(self):
        self.assertEqual(code("donor-awaiting-symbols"), "needs-symbol")

    def test_a_matched_case_whose_source_is_gone(self):
        self.assertEqual(code("matched-source-missing", case("exact", score=0)), "source-not-kept")


class NotAttemptedTest(unittest.TestCase):
    def test_every_recorded_reason_has_its_code(self):
        expected = {"larger than the run's --max-size 0x400": "too-large",
                    "not in the --only-names set": "excluded-by-settings",
                    "sampled out (every 4th function)": "excluded-by-settings",
                    "beyond the run's --limit 10": "excluded-by-settings",
                    "already matched by a project manifest row": "covered-by-manifest",
                    "no discovered image: the range is outside the image's bytes": "outside-image",
                    "smaller than the resolver's minimum of 4 bytes": "too-small",
                    "unexplained: no case and no recorded reason": "unknown",
                    "": "unknown",
                    "a reason written by a later stage": "excluded-other"}
        for reason, wanted in expected.items():
            self.assertEqual(code("not attempted", not_attempted=reason), wanted, reason)

    def test_the_recorded_reason_is_the_next_steps_detail(self):
        _code, step = reasons.explain("not attempted", None, not_attempted="beyond the run's --limit 10")
        self.assertIn("--limit 10", step)


class PrecedenceTest(unittest.TestCase):
    def test_unit_class_rsp_and_boot_come_before_the_case(self):
        self.assertEqual(code("open", case(top="semantic.unknown"), unit_class="rsp"), "rsp-microcode")
        self.assertEqual(code("blocked-boundary", case("blocked-boundary"), unit_class="boot"), "boot-code")

    def test_target_findings(self):
        self.assertEqual(code("blocked-boundary", case("blocked-boundary", top="boundary.embedded_data")),
                         "embedded-data")
        self.assertEqual(code("blocked-boundary", case("blocked-boundary", top="boundary.false_split")),
                         "boundary-uncertain")
        self.assertEqual(code("open", case(top="boundary.shared_tail")), "boundary-uncertain")
        self.assertEqual(code("probable-handwritten", case("probable-handwritten")), "handwritten-assembly-unproven")
        self.assertEqual(code("needs-tu", case("needs-tu", causes=[("abi.nonstandard", 0.9),
                                                                  ("assembly.handwritten", 0.7)])),
                         "handwritten-assembly-unproven")
        self.assertEqual(code("needs-tu", case("needs-tu", causes=[("abi.nonstandard", 0.9)])), "nonstandard-abi")
        self.assertEqual(code("open", case(top="register.permutation"), unit_class="handwritten"),
                         "handwritten-assembly-unproven")

    def test_drafts(self):
        found, step = reasons.explain("needs-source", case("needs-source", score=None,
                                                           next_step="the automatic draft failed: m2c error"))
        self.assertEqual(found, "decompiler-failed")
        self.assertIn("m2c error", step)
        found, step = reasons.explain("does-not-build", case("does-not-build", score=None,
                                                             next_step="fix the compile error: x.c:3: error"))
        self.assertEqual(found, "draft-does-not-build")
        self.assertIn("x.c:3", step)
        self.assertEqual(code("open", case(score=0, next_step="exact under the exploration oracle but not under "
                                                               "strict verification: reloc")), "strict-verify-failed")

    def test_tool_chain(self):
        far = case(score=300, stages=[{"stage": "iterate-skipped", "score": 300, "ceiling": 40}])
        self.assertEqual(code("open", far, dossier={"primary_cause": "unknown toolchain"}), "style-unrecognised")
        self.assertEqual(code("open", case(top="compiler.family")), "no-recipe-identified")
        self.assertEqual(code("open", case(recipe="")), "no-recipe-identified")

    def test_why_the_search_stopped(self):
        self.assertEqual(code("open", case(size=0x500, score=90)), "too-large")
        self.assertEqual(code("near", case("near", size=0x500, score=6, top="register.permutation",
                                           causes=[("register.permutation", 0.9)])), "register-allocation-only")
        self.assertEqual(code("open", case(score=300, stages=[{"stage": "iterate-skipped", "score": 300,
                                                                "ceiling": 40}])), "far-off-draft")
        improved = case(score=30, stages=[{"stage": "iterate-skipped", "score": 300, "ceiling": 40}],
                        top="semantic.unknown", causes=[("semantic.unknown", 0.5)])
        self.assertEqual(code("open", improved), "unrecognised-difference")      # no longer above the ceiling
        self.assertEqual(code("open", case(stages=[{"stage": "iterate-skipped", "reason": "stratum yields 0",
                                                    "stratum": "s"}])), "search-withheld")
        found, step = reasons.explain("near", case("near", score=5), dossier={
            "next_action": "search", "why_stopped": "round limit: effort normal allows 1 round(s)"})
        self.assertEqual(found, "round-limit")
        self.assertIn("automation's next action: search", step)
        found, step = reasons.explain("near", case("near", score=5, top="register.lifetime", causes=[
            ("register.lifetime", 0.9)]), dossier={"next_action": "search", "why_stopped": "round limit: x"})
        self.assertEqual(found, "round-limit")
        self.assertIn("residual register.lifetime: hint for register.lifetime", step)
        self.assertEqual(code("open", case(), dossier={"next_action": "recipe-sweep",
                                                       "why_stopped": "build budget 400 reached"}), "budget-exhausted")

    def test_residual_classes(self):
        self.assertEqual(code("near", case("near", score=4, top="register.pressure", causes=[
            ("register.pressure", 0.9), ("register.lifetime", 0.6)])), "register-allocation-only")
        found, step = reasons.explain("near", case("near", score=8, top="register.pressure", causes=[
            ("register.pressure", 0.9), ("type.width", 0.6)]))
        self.assertEqual(found, "type-or-declaration")
        self.assertIn("hint for type.width", step)
        for cause, wanted in (("relocation.target", "call-target-differs"), ("frame.size", "stack-frame-differs"),
                              ("control.branch_likely", "control-flow-differs"),
                              ("schedule.delay_slot", "control-flow-differs"),
                              ("schedule.load_hazard", "assembler-or-schedule"), ("tu.context", "needs-tu-context"),
                              ("semantic.unknown", "unrecognised-difference")):
            self.assertEqual(code("open", case(top=cause, causes=[(cause, 0.8)])), wanted, cause)
        self.assertEqual(code("open", case()), "unknown")
        # a run stopped at its fixed point leaves the residual as the reason
        self.assertEqual(code("open", case(top="frame.size", causes=[("frame.size", 0.8)]), dossier={
            "next_action": None, "why_stopped": "search: every variant family is closed"}), "stack-frame-differs")


class VocabularyTest(unittest.TestCase):
    INPUTS = [("open", case(top=c, causes=[(c, 0.8)])) for c in
              ("relocation.target", "type.width", "frame.size", "control.switch", "assembler.variant", "tu.context",
               "semantic.unknown", "register.lifetime", "source.missing", "compiler.family")] + [
        ("near", case("near", score=3)), ("needs-tu", case("needs-tu")), ("not attempted", None),
        ("does-not-build", case("does-not-build", score=None)), ("open", case(score=None))]

    def test_codes_are_closed_and_fields_are_one_line(self):
        for outcome, item in self.INPUTS:
            found, step = reasons.explain(outcome, item, dossier={"next_action": "search",
                                                                  "why_stopped": "x\ty\nz" * 80})
            self.assertIn(found, reasons.CODES)
            self.assertNotIn("\t", step)
            self.assertNotIn("\n", step)
            self.assertLessEqual(len(step), 240)

    def test_deterministic(self):
        for outcome, item in self.INPUTS:
            self.assertEqual(reasons.explain(outcome, item), reasons.explain(outcome, item))

    def test_a_code_never_changes_the_case(self):
        item = case("assumed", score=0, strict={"assumptions": [{"kind": "symbol", "name": "D_1"}]})
        before = json.dumps(item.__dict__, sort_keys=True)
        reasons.explain("awaiting-symbols", item, dossier={"next_action": "declare"})
        self.assertEqual(json.dumps(item.__dict__, sort_keys=True), before)


class FilesTest(unittest.TestCase):
    def test_dossiers_and_coverage_are_read_by_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            solver = Path(tmp) / "solver"
            solver.mkdir()
            (solver / "hard_queue_main.json").write_text(json.dumps([
                {"vram": "0x80001000", "primary_cause": "unknown toolchain", "next_action": None,
                 "why_stopped": "no available compiler", "ladder": [{"technique": "drafts"}]}]))
            (solver / "coverage_main.json").write_text(json.dumps({"entries": [
                {"vram": "0x80002000", "attempted": False, "reason": "sampled out (every 2th function)"},
                {"vram": "0x80003000", "attempted": True, "status": "open"}]}))
            dossiers = reasons.load_dossiers(Path(tmp), "main")
            self.assertEqual(dossiers, {0x80001000: {"primary_cause": "unknown toolchain", "next_action": None,
                                                     "why_stopped": "no available compiler"}})
            self.assertEqual(reasons.load_not_attempted(Path(tmp), "main"),
                             {0x80002000: "sampled out (every 2th function)"})
            self.assertEqual(reasons.load_dossiers(Path(tmp), "other"), {})
            (solver / "hard_queue_bad.json").write_text("{not json")
            self.assertEqual(reasons.load_dossiers(Path(tmp), "bad"), {})


class TableTest(unittest.TestCase):
    def test_summary_and_coverage_table(self):
        tallies = [("", 0x40), ("needs-symbol", 0x10), ("needs-symbol", 0x20), ("unknown", 0x8), ("far-off-draft", 4)]
        found = reasons.summary(tallies)
        self.assertEqual(found["not_settled"], 4)
        self.assertEqual(found["specific"], 3)
        self.assertEqual(found["unknown"], 1)
        self.assertEqual(list(found["by_code"]), ["needs-symbol", "far-off-draft", "unknown"])
        self.assertEqual(found["by_code"]["needs-symbol"], {"functions": 2, "bytes": 0x30})
        lines = reasons.table(tallies)
        self.assertIn("3 of 4 functions have a specific code (75.0%); `unknown`: 1.", lines)
        self.assertIn("| needs-symbol | 2 | 48 | " + reasons.CODES["needs-symbol"][0] + " |", lines)
        self.assertEqual(reasons.table([("", 4)]), [])

    def test_the_export_appends_the_columns_after_the_existing_ones(self):
        source = inspect.getsource(export.export)
        self.assertIn('"\\tunit_class\\treason_code\\tnext_step\\n"', source)
        self.assertIn("image\\tfunction\\tvram\\trom_offset\\tsize\\tcode_size\\toutcome\\thow\\tsource\\tdistance"
                      "\\treproduced_by", source)


if __name__ == "__main__":
    unittest.main()


class ImplausibleSizeTest(unittest.TestCase):
    def test_a_function_over_the_ceiling_is_left_out_with_its_own_code(self):
        from n64dk.commands import resolve as resolvecmd
        from n64dk.match import reasons
        self.assertEqual(resolvecmd.implausible_size(), 0x20000)
        text = "implausibly large for one function (over 0x20000 bytes): most likely a boundary error"
        self.assertEqual(next(code for prefix, code in reasons._NOT_ATTEMPTED if text.startswith(prefix)),
                         "implausible-size")
        self.assertIn("implausible-size", reasons.CODES)


class WorkerBatchTest(unittest.TestCase):
    """Resolve workers are recycled every N64DK_WORKER_BATCH functions (memory, 2026-11-18)."""

    def test_batch_setting(self):
        from unittest import mock
        from n64dk.commands import resolve
        with mock.patch.dict("os.environ", {"N64DK_WORKER_BATCH": ""}):
            self.assertEqual(resolve.worker_batch(), resolve.WORKER_BATCH)
        with mock.patch.dict("os.environ", {"N64DK_WORKER_BATCH": "40"}):
            self.assertEqual(resolve.worker_batch(), 40)
        with mock.patch.dict("os.environ", {"N64DK_WORKER_BATCH": "0"}):
            self.assertGreater(resolve.worker_batch(), 10 ** 6)          # off: one process per shard
        with mock.patch.dict("os.environ", {"N64DK_WORKER_BATCH": "x"}):
            self.assertEqual(resolve.worker_batch(), resolve.WORKER_BATCH)


class BatchesOfTest(unittest.TestCase):
    """Worker batches end at a function count or a code-byte limit (memory, 2026-11-18)."""

    def test_count_and_bytes(self):
        from n64dk.commands.resolve import batches_of
        small = [(f"s{i}", i, 0x10) for i in range(5)]
        big = [(f"b{i}", 0x100 + i, 0x1800) for i in range(3)]
        shards = [small + big, small[:2]]
        got = batches_of(shards, 3, 0x2000)
        self.assertEqual([(i, [f[0] for f in fs]) for i, fs in got],
                         [(0, ["s0", "s1", "s2"]), (1, ["s0", "s1"]),
                          (0, ["s3", "s4", "b0"]), (0, ["b1"]), (0, ["b2"])])

    def test_an_oversize_function_still_forms_a_batch(self):
        from n64dk.commands.resolve import batches_of
        self.assertEqual(batches_of([[("huge", 0, 0x9000)]], 150, 0x1000), [(0, [("huge", 0, 0x9000)])])

    def test_no_limits_keeps_one_batch_per_shard(self):
        from n64dk.commands.resolve import batches_of
        shards = [[("a", 0, 4), ("b", 1, 4)], [("c", 2, 4)]]
        self.assertEqual(batches_of(shards, 10 ** 9, 10 ** 12), [(0, shards[0]), (1, shards[1])])


class TranslatorReasonTest(unittest.TestCase):
    """What the unoptimised-code translator made of a function is the reason, not "far-off draft"."""

    def _case(self, **fields):
        from types import SimpleNamespace
        base = dict(status="open", best_score=251, size=0x800, recipe="kmc+asn64 -O0", top_cause="semantic.unknown",
                    causes=[], stages=[{"stage": "iterate-skipped", "score": 251, "ceiling": 40}], vram=0x80001000,
                    proof=None, strict=None, name="f", image="main")
        base.update(fields)
        return SimpleNamespace(**base)

    def test_a_declined_function_names_the_construct(self):
        from n64dk.match import reasons
        case = self._case(stages=[{"stage": "iterate-skipped", "score": 251, "ceiling": 40},
                                  {"stage": "translator", "declined": "jump table"}])
        self.assertEqual(reasons._case_code(case, "", {}), ("translator-declined", "jump table"))

    def test_a_translated_function_that_differs_says_so(self):
        from n64dk.match import reasons
        case = self._case(stages=[{"stage": "translator", "translated": True}])
        self.assertEqual(reasons._case_code(case, "", {})[0], "translation-differs")

    def test_both_codes_are_in_the_tables(self):
        from n64dk.match import reasons
        for code in ("translator-declined", "translation-differs"):
            self.assertIn(code, reasons.CODES)
            self.assertIn(code, reasons.RETRY)


class DebugTwinTest(unittest.TestCase):
    """A large function is offered exactly one alternative: its unoptimised GCC recipe with -g.  (The first
    version of this named a helper that was not in scope and failed every large function: 2026-11-18.)"""

    def _runner(self, pipeline, flags):
        from types import SimpleNamespace
        from n64dk.commands import resolve as resolvecmd
        from n64dk.match.pipeline import split_flags
        runner = resolvecmd._Runner.__new__(resolvecmd._Runner)
        runner.spec = split_flags(flags, pipeline)
        runner.evaluator = lambda vram, size, filename, spec: ("evaluator", tuple(spec.flags))
        return runner

    def test_unoptimised_gcc_gets_the_twin(self):
        runner = self._runner("kmc+asn64", ["-O0", "-G0", "-mips3", "-mgp32", "-mfp32"])
        found = runner.debug_twin(0x80001000, 0x800, "f.c")(None)
        self.assertEqual(len(found), 1)
        self.assertIn("-g", found[0][0].split())
        self.assertEqual(found[0][1], "kmc+asn64")

    def test_other_recipes_get_nothing(self):
        for pipeline, flags in (("kmc+asn64", ["-O2", "-G0"]), ("kmc+asn64", ["-O0", "-g", "-G0"]),
                                ("ido5.3", ["-O0", "-mips2"])):
            self.assertEqual(self._runner(pipeline, flags).debug_twin(0x80001000, 0x800, "f.c")(None), [], pipeline)


class RuntimeHelperTest(unittest.TestCase):
    """A compiler support routine is declared where its own code stands, and nowhere else."""
    UDIVDI3 = [0x27BDFFE0, 0xAFBF0018, 0x0C00E036, 0xAFA00010, 0x8FBF0018, 0x27BD0020, 0x03E00008, 0]

    def read(self, layout):
        return lambda address: layout.get(address)

    def test_udivdi3_is_found_by_its_code(self):
        from n64dk.match import runtimehelpers
        base = 0x800380B8
        layout = {base + 4 * k: word for k, word in enumerate(self.UDIVDI3)}
        other = {0x80040000 + 4 * k: word for k, word in enumerate([0x27BDFFE0, 0xAFBF0018, 0x0C00E036, 0, 0x8FBF0018,
                                                                    0x27BD0020, 0x03E00008, 0])}
        found = runtimehelpers.helpers(self.read({**layout, **other}), [base, 0x80040000, 0x80050000])
        self.assertEqual(found, {"__udivdi3": {base}})
        self.assertFalse(runtimehelpers.is_udivdi3(None, base))

    def test_umoddi3_is_found_by_its_code(self):
        from n64dk.match import runtimehelpers
        words = [0x27BDFFD8, 0x27A20018, 0xAFBF0020, 0x0C00E036, 0xAFA20010, 0x8FA20018, 0x8FA3001C, 0x8FBF0020,
                 0x27BD0028, 0x03E00008, 0]
        layout = {0x80038718 + 4 * k: word for k, word in enumerate(words)}
        self.assertEqual(runtimehelpers.helpers(self.read(layout), [0x80038718, 0x80038720]), {"__umoddi3": {0x80038718}})
