import shutil
import atexit
import struct
import tempfile
import unittest

from n64dk.match import iterate, resolve


def _scratch(prefix: str = "n64dk-test-") -> str:
    """A scratch directory removed when the test process exits."""
    path = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, path, True)
    return path


BASE = 0x80001000


def code(words):
    return b"".join(struct.pack(">I", w) for w in words)


TARGET = code([0x84820004, 0x03E00008, 0x00000000])        # lh v0,4(a0); jr ra; nop
WRONG = code([0x94820004, 0x03E00008, 0x00000000])         # lhu
FRAGMENT = code([0x96020060, 0x00481821, 0xA6030024, 0x8FBF001C, 0x03E00008, 0x27BD0020])
SOURCE = "s32 read(Thing *thing) {\n    u16 value;\n\n    value = thing->field;\n    return value;\n}\n"


def fake_evaluate(calls, proof=True):
    """Pretend compiler: the function matches when ``value`` is declared s16."""
    def evaluate(text):
        calls.append(text)
        if "does not build" in text:
            return iterate.Evaluation(None, error="x.c:1: parse error"), None
        exact = "s16 value;" in text
        report = {"counts": {} if exact else {"instruction": 1},
                  "entries": [] if exact else [{"compiled": "lhu $v0, 0x4($a0)", "retail": "lh $v0, 0x4($a0)"}]}
        outcome = {"proof": {"verdict": "exact"}} if exact and proof else None
        return iterate.Evaluation(0 if exact else 3, exact, report, outcome=outcome), (TARGET if exact else WRONG)
    return evaluate


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.root = _scratch("n64dk-resolve-test-")
        self.store = resolve.CaseStore(self.root)
        self.addCleanup(self.store.close)

    def test_ladder_reaches_exact_through_the_justified_family(self):
        calls, events = [], []
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate(calls), source=SOURCE,
                               jobs=1, emit=events.append)
        self.assertEqual(case.status, "exact")
        self.assertIn("s16 value;", self.store.source(case.best_digest))
        stages = [s["stage"] for s in case.stages]
        self.assertEqual(stages[:4], ["target", "source", "baseline", "diagnose"])
        self.assertEqual(case.stages[3]["causes"][0], "type.signedness")
        self.assertEqual(stages[4], "iterate")
        self.assertIn("exact-found", [e["event"] for e in events])

    def test_rewrites_are_tried_on_the_best_source_the_search_ends_with(self):
        """The recipe's rewrites match from what the search made of the draft, not only from the draft."""
        def evaluate(text):
            exact = "/* rewritten */" in text
            outcome = {"proof": {"verdict": "exact"}} if exact else None
            return iterate.Evaluation(0 if exact else 5, exact, {"counts": {} if exact else {"register": 2}, "entries": []},
                                      outcome=outcome), (TARGET if exact else WRONG)
        seen = []

        def rewrites(text):
            seen.append(text)
            return [("worse", text + "/* same */"), ("fix", text + "/* rewritten */")]
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, evaluate, source=SOURCE, jobs=1, budget=60,
                               rewrites=rewrites)
        self.assertEqual(case.status, "exact")
        self.assertEqual(len(seen), 1)
        self.assertIn("/* rewritten */", self.store.source(case.best_digest))
        stage = [s for s in case.stages if s["stage"] == "rewrite-best"][0]
        self.assertEqual((stage["tried"], stage["start"], stage["final"]), (2, 5, 0))

    def test_rewrites_that_do_not_help_leave_the_best_source(self):
        def evaluate(text):
            return iterate.Evaluation(7 if "/* worse */" in text else 5, False, {"counts": {"register": 2}, "entries": []}), WRONG
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, evaluate, source=SOURCE, jobs=1, budget=60,
                               rewrites=lambda text: [("worse", text + "/* worse */")])
        self.assertNotIn("/* worse */", self.store.source(case.best_digest))
        self.assertEqual(case.best_score, 5)

    def test_exact_case_is_not_recomputed(self):
        calls = []
        resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate(calls), source=SOURCE, jobs=1)
        before = len(calls)
        again = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate(calls), source=SOURCE, jobs=1)
        self.assertEqual((again.status, len(calls)), ("exact", before))

    def test_fragment_is_blocked_without_spending_a_build(self):
        calls = []
        case = resolve.resolve(self.store, "main", "piece", BASE, FRAGMENT, fake_evaluate(calls), source=SOURCE, jobs=1)
        self.assertEqual((case.status, case.top_cause, calls), ("blocked-boundary", "boundary.fragment", []))

    HANDWRITTEN = code([0x40086000, 0x2401FFFE, 0x01014824, 0x40896000, 0x31020001, 0x00000000,
                        0x03E00008, 0x00000000])          # mfc0/mtc0 Status: no compiler writes this

    def test_handwritten_code_is_kept_as_assembly_when_its_source_assembles_back(self):
        calls = []

        def draft(family=None):
            return None, "not reached"
        draft.assembly = lambda: ("glabel irq_off\n    mfc0 $t0, $12\n", {"verdict": "exact", "assumptions": []})
        case = resolve.resolve(self.store, "main", "irq_off", BASE, self.HANDWRITTEN, fake_evaluate(calls),
                               draft=draft, jobs=1)
        self.assertEqual((case.status, calls), ("assembly", []))
        self.assertIn("mfc0", self.store.source(case.best_digest))
        self.assertEqual(case.strict.get("verdict"), "exact")
        self.assertIn("gnu-as", case.recipe)

    def test_a_section_placed_by_its_contents_is_part_of_the_exported_recipe(self):
        import pathlib
        placed = {"verdict": "exact", "assumptions": [], "corroborated": {".rodata": "0x80000E40"}}
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate([], proof=False), source=SOURCE,
                               jobs=1, strict=lambda text, recipe: dict(placed), recipe_label="kmc+asn64 -O0 -G0",
                               rom_offset=0x1000)
        self.assertEqual(case.status, "exact")
        out = pathlib.Path(_scratch("n64dk-queue-test-"))
        resolve.export_queue(self.store, out)
        row = [line for line in (out / "queue.tsv").read_text().splitlines() if "main/read.c" in line][0]
        # the recipe stays as verified (a receipt rebuilds under it); the placement a link needs is listed beside it
        self.assertEqual(row.split("\t")[3], "-O0 -G0")
        placed = [line.split("\t") for line in (out / "placed_sections.tsv").read_text().splitlines()
                  if not line.startswith("#")]
        self.assertEqual(placed, [["main", f"0x{BASE:08X}", ".rodata", "0x80000E40", "section-content"]])

    def test_assembly_is_exported_beside_the_queue_and_not_in_it(self):
        import pathlib

        def draft(family=None):
            return None, "not reached"
        draft.assembly = lambda: ("glabel irq_off\n    mfc0 $t0, $12\n", {"verdict": "exact", "assumptions": []})
        resolve.resolve(self.store, "main", "irq_off", BASE, self.HANDWRITTEN, fake_evaluate([]), draft=draft,
                        jobs=1, rom_offset=0x1000)
        out = pathlib.Path(_scratch("n64dk-queue-test-"))
        found = resolve.export_queue(self.store, out)
        self.assertEqual((found["assembly"], found["assembly_bytes"], found["exported"]), (1, 32, 0))
        self.assertIn("mfc0", (out / "main" / "irq_off.s").read_text())
        table = (out / "assembly.tsv").read_text()
        self.assertIn("0x1000\t0x20\tmain/irq_off.s\t# 0x80001000", table)
        self.assertFalse((out / "queue.tsv").exists())

    def test_handwritten_code_without_a_proved_source_stays_a_finding(self):
        calls = []

        def draft(family=None):
            return None, "not reached"
        draft.assembly = lambda: None
        case = resolve.resolve(self.store, "main", "irq_off", BASE, self.HANDWRITTEN, fake_evaluate(calls),
                               draft=draft, jobs=1)
        self.assertEqual((case.status, calls), ("probable-handwritten", []))
        plain = resolve.resolve(self.store, "main", "irq_off2", BASE, self.HANDWRITTEN, fake_evaluate(calls), jobs=1)
        self.assertEqual(plain.status, "probable-handwritten")

    def test_changed_target_archives_the_old_case(self):
        calls, events = [], []
        resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate(calls), source=SOURCE, jobs=1)
        other = code([0x84820008, 0x03E00008, 0x00000000])          # same size, different bytes
        case = resolve.resolve(self.store, "main", "read", BASE, other, fake_evaluate(calls), source=SOURCE,
                               jobs=1, emit=events.append)
        self.assertEqual(events[0]["event"], "stale-case-archived")
        self.assertNotEqual(case.target_digest, events[0]["old_target"])
        stale = self.store.db.execute("SELECT COUNT(*) FROM stale").fetchone()[0]
        self.assertEqual(stale, 1)

    def test_missing_source_and_failed_draft(self):
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate([]), jobs=1,
                               draft=lambda family: (None, "m2c not found"))
        self.assertEqual((case.status, case.top_cause), ("needs-source", "source.missing"))
        self.assertIn("m2c not found", case.next)

    def test_build_failure_is_a_status_not_an_exception(self):
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate([]),
                               source="does not build", jobs=1)
        self.assertEqual(case.status, "does-not-build")

    def test_unhelpful_families_are_closed_for_that_source(self):
        def never(text):
            return iterate.Evaluation(5, False, {"counts": {"register": 2}, "entries": []}), WRONG
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, never, source=SOURCE, jobs=1, budget=60)
        self.assertIn(case.status, ("near", "open"))
        self.assertTrue(any(case.closed_families.values()))
        calls = []

        def counting(text):
            calls.append(text)
            return never(text)
        resolve.resolve(self.store, "main", "read", BASE, TARGET, counting, source=SOURCE, jobs=1, budget=60)
        self.assertLessEqual(len(calls), 3)          # baseline (+ final re-check): closed families are skipped

    def test_recipe_stage_switches_to_the_detected_compiler_family(self):
        # retail: move via `or` and an unconditional `b` (IDO); the candidate pipeline is KMC GCC
        target = code([0x00802825, 0x10000001, 0x00000000, 0x00A01025, 0x03E00008, 0x00000000])
        used = []

        def gcc(text):
            used.append("gcc")
            return iterate.Evaluation(9, False, {"counts": {"instruction": 3}, "entries": []}), WRONG

        def ido(text):
            used.append("ido")
            return iterate.Evaluation(0, True, {"counts": {}, "entries": []},
                                      outcome={"proof": {"verdict": "exact"}}), target
        case = resolve.resolve(self.store, "main", "copy", BASE, target, gcc, source=SOURCE, jobs=1,
                               pipeline="kmc+builtin", recipes=lambda family: [("ido5.3 -O2", "ido5.3", ido)])
        self.assertEqual(case.status, "exact")
        self.assertEqual(case.stages[0]["compiler_family"], "ido")
        self.assertEqual([s for s in case.stages if s["stage"] == "recipe"][0]["chosen"], "ido5.3 -O2")
        self.assertEqual(used, ["gcc", "ido"])

    def test_draft_portfolio_continues_from_the_closest_variant(self):
        good = SOURCE.replace("u16 value;", "s16 value;")
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate([]), jobs=1,
                               draft=lambda family: [("plain", SOURCE), ("context", good)])
        self.assertEqual(case.status, "exact")
        drafts = [s for s in case.stages if s["stage"] == "drafts"][0]["variants"]
        self.assertEqual([(d["label"], d["score"]) for d in drafts], [("context", 0), ("plain", 3)])
        self.assertEqual([s for s in case.stages if s["stage"] == "source"][0]["origin"], "draft:context")

    def test_a_draft_that_ranked_second_wins_under_a_sibling_recipe(self):
        """Under the default recipe `plain` is closest and nothing is exact; under the sibling
        recipe only `narrow` matches.  The case must end exact with that draft and that recipe."""
        narrow = SOURCE.replace("u16 value;", "u8 value;")

        def default(text):
            return iterate.Evaluation(6 if "u8 value;" in text else 3, False, {"counts": {"register": 1}}), WRONG

        def sibling(text):
            exact = "u8 value;" in text
            outcome = {"proof": {"verdict": "exact"}} if exact else None
            return iterate.Evaluation(0 if exact else 5, exact, {"counts": {}}, outcome=outcome), \
                (TARGET if exact else WRONG)

        events = []
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, default, jobs=1, emit=events.append,
                               pipeline="ido5.3", recipe_label="ido5.3 -O2",
                               draft=lambda family: [("plain", SOURCE), ("narrow", narrow)],
                               recipes=lambda family: [("ido7.1 -O2", "ido7.1", sibling)])
        self.assertEqual(case.status, "exact")
        self.assertEqual(case.recipe, "ido7.1 -O2")
        self.assertEqual([s for s in case.stages if s["stage"] == "source"][0]["origin"], "draft:narrow")
        cross = [s for s in case.stages if s["stage"] == "drafts-x-recipes"][0]
        self.assertEqual((cross["tried"], cross["chosen"]), (2, "ido7.1 -O2"))
        self.assertIn("recipe-selected", [e["event"] for e in events])

    def test_cross_check_is_skipped_when_a_draft_is_already_exact(self):
        good = SOURCE.replace("u16 value;", "s16 value;")
        calls = []
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate([]), jobs=1,
                               draft=lambda family: [("plain", SOURCE), ("context", good)],
                               recipes=lambda family: calls.append(family) or [])
        self.assertEqual(case.status, "exact")
        self.assertEqual(calls, [])
        self.assertFalse([s for s in case.stages if s["stage"] == "drafts-x-recipes"])

    def test_search_is_skipped_for_a_draft_too_far_from_the_target(self):
        calls = []

        def far(text):
            calls.append(text)
            return iterate.Evaluation(250, False, {"counts": {"instruction": 60}, "entries": []}), WRONG

        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, far, source=SOURCE, jobs=1, budget=50)
        self.assertEqual(case.status, "open")
        self.assertLessEqual(len(calls), 2)                              # baseline (and the final classification)
        self.assertEqual([s for s in case.stages if s["stage"] == "iterate-skipped"][0]["ceiling"],
                         resolve.SEARCH_CEILING)
        self.assertFalse([s for s in case.stages if s["stage"] == "iterate"])
        calls.clear()
        again = resolve.resolve(self.store, "main", "read", BASE, TARGET, far, source=SOURCE, jobs=1, budget=50,
                                search_ceiling=0, force=True)
        self.assertTrue([s for s in again.stages if s["stage"] == "iterate"])   # no ceiling: the search runs
        self.assertGreater(len(calls), 2)

    def test_frame_fit_reserves_the_measured_missing_bytes(self):
        """The candidate's frame is 8 bytes smaller than the target's: the repair is computed from
        the two frames (a pad of exactly that size) and tried before any search."""
        from n64dk.analysis import flow as flowmod
        big = code([0x27BDFFE0, 0xAFBF0014, 0x8FBF0014, 0x03E00008, 0x27BD0020])     # frame 0x20
        small = code([0x27BDFFE8, 0xAFBF0014, 0x8FBF0014, 0x03E00008, 0x27BD0018])   # frame 0x18
        source = "void work(void) {\n    s32 a;\n\n    a = 1;\n    g(a);\n}\n"
        seen = []

        def evaluate(text):
            seen.append(text)
            if "s32 pad[2];" in text:
                return iterate.Evaluation(0, True, {"counts": {}, "entries": []},
                                          outcome={"proof": {"verdict": "exact"}}), big
            return iterate.Evaluation(9, False, {"counts": {"immediate": 3}, "entries": []}), small

        case = resolve.resolve(self.store, "main", "work", BASE, big, evaluate, source=source, jobs=1,
                               flow=flowmod.analyse(big, BASE), budget=40)
        self.assertEqual(case.status, "exact")
        fit = [s for s in case.stages if s["stage"] == "frame-fit"][0]
        self.assertEqual((fit["chosen"], fit["target_frame"]), ("pad 8 top", 0x20))
        self.assertEqual(len(seen), 2)                                   # baseline, then the computed pad
        self.assertFalse([s for s in case.stages if s["stage"] == "iterate"])

    def test_frame_fit_is_not_tried_when_the_frames_agree(self):
        from n64dk.analysis import flow as flowmod
        big = code([0x27BDFFE0, 0xAFBF0014, 0x8FBF0014, 0x03E00008, 0x27BD0020])
        calls = []

        def evaluate(text):
            calls.append(text)
            return iterate.Evaluation(300, False, {"counts": {"instruction": 60}, "entries": []}), big

        case = resolve.resolve(self.store, "main", "work", BASE, big, evaluate, source="void work(void) {\n}\n",
                               jobs=1, flow=flowmod.analyse(big, BASE), budget=40)
        self.assertFalse([s for s in case.stages if s["stage"] == "frame-fit"])
        self.assertEqual(case.status, "open")

    def test_worker_count(self):
        from n64dk.commands.resolve import worker_count
        self.assertEqual(worker_count(1, 500), 1)
        self.assertEqual(worker_count(6, 500), 6)
        self.assertEqual(worker_count(6, 5), 2)              # never more workers than half the functions
        self.assertEqual(worker_count(6, 1), 1)
        self.assertEqual(worker_count(None, 500), 1)
        self.assertGreaterEqual(worker_count(0, 500), 1)     # automatic: bounded by threads and free memory
        import os
        self.assertLessEqual(worker_count(0, 500), max(1, (os.cpu_count() or 4) // 2))

    def test_exact_case_records_the_strict_verdict_and_a_mismatch_demotes_it(self):
        seen = []

        def assumed(text, recipe):
            seen.append(recipe)
            return {"verdict": "assumed", "assumptions": [{"kind": "symbol", "name": "gThing"}]}
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate([], proof=False), source=SOURCE, jobs=1,
                               strict=assumed, recipe_label="ido5.3 -O2")
        self.assertEqual((case.status, case.strict["verdict"], seen), ("assumed", "assumed", ["ido5.3 -O2"]))
        self.assertIn("declare", case.next)
        other = resolve.resolve(self.store, "main", "read", BASE + 0x40, TARGET, fake_evaluate([], proof=False), source=SOURCE,
                                jobs=1, strict=lambda text, recipe: {"verdict": "mismatch",
                                                                     "conflicts": ["wrong callee"]})
        self.assertEqual(other.status, "open")
        self.assertIn("wrong callee", other.next)

    def test_assumed_case_is_reverified_not_recomputed(self):
        verdicts = [{"verdict": "assumed", "assumptions": [{"kind": "symbol", "name": "gThing"}]}, {"verdict": "exact"}]
        calls, seen = [], []

        def strict(text, recipe):
            seen.append(recipe)
            return verdicts[min(len(seen), 2) - 1]
        exact_source = SOURCE.replace("u16 value;", "s16 value;")
        arguments = dict(jobs=1, strict=strict, recipe_label="ido5.3 -O2", draft=lambda family: [("plain", exact_source)])
        first = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate(calls, proof=False), **arguments)
        self.assertEqual(first.status, "assumed")
        builds = len(calls)
        again = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate(calls, proof=False),
                                **{**arguments, "recipe_label": "kmc -O2"})
        self.assertEqual((again.status, len(calls), seen), ("exact", builds, ["ido5.3 -O2", "ido5.3 -O2"]))
        self.assertEqual(self.store.get(first.case_id).status, "exact")

    def test_rerun_continues_under_the_stored_recipe_and_never_gets_worse(self):
        def default(text):
            return iterate.Evaluation(8, False, {"counts": {"register": 8}}), WRONG

        def sibling(text):
            return iterate.Evaluation(2, False, {"counts": {"register": 2}}), WRONG
        arguments = dict(jobs=1, pipeline="kmc", recipe_label="kmc -O2", draft=lambda family: [("plain", SOURCE)])
        first = resolve.resolve(self.store, "main", "read", BASE, TARGET, default, **arguments,
                                recipes=lambda family: [("ido5.3 -O2", "ido5.3", sibling)])
        self.assertEqual((first.status, first.best_score, first.recipe), ("near", 2, "ido5.3 -O2"))
        again = resolve.resolve(self.store, "main", "read", BASE, TARGET, default, **arguments,
                                recipes=lambda family: [("ido5.3 -O2", "ido5.3", sibling)])
        self.assertEqual((again.status, again.best_score, again.recipe), ("near", 2, "ido5.3 -O2"))
        events = []
        kept = resolve.resolve(self.store, "main", "read", BASE, TARGET, default, **arguments, emit=events.append)
        self.assertEqual((kept.status, kept.best_score, kept.recipe), ("near", 2, "ido5.3 -O2"))
        self.assertIn("case-kept-previous", [e["event"] for e in events])
        self.assertEqual(self.store.get(first.case_id).best_score, 2)

    def _two_recipes(self):
        narrow = SOURCE.replace("u16 value;", "u8 value;")

        def recipe_a(text):                     # the configured recipe calibration sampled under
            return iterate.Evaluation(3 if "u16 value;" in text else 6, False, {"counts": {"register": 1}}), WRONG

        def recipe_b(text):                     # the recipe calibration chose: only `narrow` matches
            exact = "u8 value;" in text
            outcome = {"proof": {"verdict": "exact"}} if exact else None
            return iterate.Evaluation(0 if exact else 5, exact, {"counts": {}}, outcome=outcome), \
                (TARGET if exact else WRONG)
        return recipe_a, recipe_b, (lambda family: [("plain", SOURCE), ("narrow", narrow)])

    def test_calibration_sample_ends_like_an_unsampled_function(self):
        """Round 10 (M14): a function calibration had sampled kept its sample's best source and
        never drafted again under the chosen recipe.  Its outcome must not depend on the sample."""
        recipe_a, recipe_b, drafts = self._two_recipes()
        sample = resolve.resolve(self.store, "main", "read", BASE, TARGET, recipe_a, jobs=1, pipeline="kmc",
                                 recipe_label="kmc -O2", draft=drafts, budget=4, provisional=True)
        self.assertNotEqual(sample.status, "exact")
        sampled = resolve.resolve(self.store, "main", "read", BASE, TARGET, recipe_b, jobs=1, pipeline="ido5.3",
                                  recipe_label="ido5.3 -O2", draft=drafts, budget=4)
        other = resolve.CaseStore(_scratch("n64dk-resolve-test-"))
        self.addCleanup(other.close)
        unsampled = resolve.resolve(other, "main", "read", BASE, TARGET, recipe_b, jobs=1, pipeline="ido5.3",
                                    recipe_label="ido5.3 -O2", draft=drafts, budget=4)
        self.assertEqual((sampled.status, sampled.best_score, sampled.recipe),
                         (unsampled.status, unsampled.best_score, unsampled.recipe))
        self.assertEqual(sampled.status, "exact")
        self.assertFalse([s for s in sampled.stages if s["stage"] == "provisional"])

    def test_better_calibration_result_is_kept_and_no_longer_provisional(self):
        recipe_a, _recipe_b, drafts = self._two_recipes()

        def far(text):
            return iterate.Evaluation(9, False, {"counts": {"register": 9}}), WRONG
        resolve.resolve(self.store, "main", "read", BASE, TARGET, recipe_a, jobs=1, pipeline="kmc",
                        recipe_label="kmc -O2", draft=drafts, budget=4, provisional=True)
        calls = []

        def counted(text):
            calls.append(text)
            return far(text)
        kept = resolve.resolve(self.store, "main", "read", BASE, TARGET, counted, jobs=1, pipeline="ido5.3",
                               recipe_label="ido5.3 -O2", draft=drafts, budget=4)
        self.assertEqual((kept.best_score, kept.recipe), (3, "kmc -O2"))
        self.assertTrue(calls)                                   # drafted again under the chosen recipe
        stored = self.store.get(kept.case_id)
        self.assertFalse([s for s in stored.stages if s["stage"] == "provisional"])

    def test_wide_recipe_retry_is_gated_on_evidence(self):
        def far(text):
            return iterate.Evaluation(80, False, {"counts": {"instruction": 20}, "entries": []}), WRONG
        tried = []

        def recipes(family):
            tried.append(family)
            return [("kmc -O1", "kmc", far)]
        arguments = dict(jobs=1, pipeline="kmc", recipe_label="kmc -O2", source=SOURCE, budget=4, recipes=recipes)
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, far, wide_recipes=lambda: False, **arguments)
        self.assertIn("recipe-skipped", [s["stage"] for s in case.stages])
        self.assertEqual(tried, [])
        other = resolve.CaseStore(_scratch("n64dk-resolve-test-"))
        self.addCleanup(other.close)
        case = resolve.resolve(other, "main", "read", BASE, TARGET, far, wide_recipes=lambda: True, **arguments)
        self.assertNotIn("recipe-skipped", [s["stage"] for s in case.stages])
        self.assertTrue(tried)

    def test_exact_without_strict_proof_is_never_persisted(self):
        case = resolve.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate([], proof=False), source=SOURCE,
                               jobs=1, strict=lambda _text, _recipe: {"verdict": "error", "error": "boom"})
        self.assertEqual(case.status, "open")
        self.assertIn("strict verification", case.next)

    def test_twin_source_renames_symbols_site_by_site(self):
        from n64dk.commands.resolve import twin_source
        text = "extern s32 D_80100000;\ns32 func_80001000(void) {\n    helper();\n    return D_80100000 + func_80002000();\n}\n"
        old_sites = [(0, "jal", 0x80003000), (2, "hi", 0x80100000), (3, "lo", 0x80100000), (5, "jal", 0x80002000)]
        new_sites = [(0, "jal", 0x80003000), (2, "hi", 0x80100040), (3, "lo", 0x80100040), (5, "jal", 0x80002200)]
        names = {0x80003000: "helper", 0x80002200: "other_helper"}
        out = twin_source(text, "func_80001000", "func_80001100", old_sites, new_sites, names)
        self.assertEqual(out, "extern s32 D_80100040;\ns32 func_80001100(void) {\n    helper();\n"
                              "    return D_80100040 + func_80002200();\n}\n")
        self.assertIsNone(twin_source(text, "a", "b", old_sites, new_sites[:-1], names))       # different shape
        clash = [(0, "jal", 0x80003000), (2, "hi", 0x80100040), (3, "lo", 0x80100080), (5, "jal", 0x80002200)]
        self.assertIsNone(twin_source(text, "a", "b", old_sites, clash, names))               # one symbol, two addresses

    def test_hard_queue_groups_and_ranks(self):
        resolve.resolve(self.store, "main", "piece", BASE, FRAGMENT, fake_evaluate([]), source=SOURCE, jobs=1)
        resolve.resolve(self.store, "main", "read", BASE + 0x100, TARGET, fake_evaluate([]), jobs=1,
                        draft=lambda family: (None, "no m2c"))
        report = resolve.hard_queue(self.store.all())
        self.assertEqual({g["cause"] for g in report["queue"]}, {"boundary.fragment", "source.missing"})
        self.assertEqual(report["queue"][0]["cause"], "source.missing")      # boundary work has no compile payoff
        self.assertEqual(report["statuses"], {"blocked-boundary": 1, "needs-source": 1})


if __name__ == "__main__":
    unittest.main()
