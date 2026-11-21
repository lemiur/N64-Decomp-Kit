"""``decompile-receipt``: the exhaustive-treatment receipt of a finished run, from synthetic run
directories (a case store, a rounds log, coverage, dossiers and a hand-written ``report.json`` in
the shape ``decompile-run`` writes).  No toolchain, no ROM, no machine lock."""
import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from n64dk import cli
from n64dk.commands import receipt as rc
from n64dk.match import resolve as resolvemod
from n64dk.match import solver

FIXED = "fixed point: a full round brought no new exact function, declaration, evidence or negative fact"


def _case(name, vram, *, status="open", size=0x40, **extra):
    return resolvemod.Case(resolvemod.case_id("main", vram, size), "main", name, vram, size, "t" + name,
                           status=status, best_score=0 if status == "exact" else 12, recipe="ido5.3 -O2", **extra)


CASES = (_case("f1", 0x80001000, status="exact", strict={"verdict": "exact"}),
         _case("f2", 0x80001040, size=32),
         _case("f3", 0x80001060, size=32, status="assumed"))


def _report() -> dict:
    return {
        "project": "<project>", "status": "complete", "seconds": 42.0, "failed_stages": {},
        "resources": {"jobs": 1, "memory": {"total_mb": 8000, "available_mb": 4000, "swap_used_mb": 1033}},
        "profile": {"resolve": {"wall_s": 30.0, "builds_requested": 100, "distinct_objects": 80,
                                "tool_invocations": 120}},
        "discovery": {"verdict_before": "incomplete", "verdict": "complete", "rom_reasons": [], "placed": [],
                      "warning": None,
                      "images": {"main": {"verdict": "incomplete", "verdict_after": "complete", "coverage_after": 1.0,
                                          "functions_after": 4, "repaired_functions": 1}}},
        "small_data": {"value": None, "confidence": None, "reason": "no site sets $gp", "sites": 0},
        "donor": {"functions": 1},
        "images": {"main": {
            "functions": 4,
            "effective_settings": {"recipe_gate": False, "regional_calibration": True},
            "boundaries": {"operations": 0},
            "resolve": {"statuses": {"exact": 1, "open": 1, "assumed": 1}, "exact_bytes": 64},
            "solver_stop": {"kind": "stop", "reason": FIXED, "after_round": 2},
            "solver": {"mode": "exhaustive", "coverage": {"functions": 4, "attempted": 3, "not_attempted": 1,
                                                         "unexplained": 0},
                       "remaining_by_cause": {"register-only residual": {"functions": 1, "bytes": 32}}},
            "toolchain_regions": {"regions": 1, "functions_by_verdict": {"known": 4}},
            "queue": {"strict_exact": 1, "assumed": 1},
            "ledger": {"totals": {"extent_bytes": 256, "sum_equals_extent": True, "c_bytes": 0, "asm_bytes": 0,
                                  "code_bytes": 160},
                       "violations": {}}}}}


def _dossiers() -> list:
    return [{"image": "main", "name": "f2", "case": CASES[1].case_id, "size": 32, "status": "open",
             "primary_cause": "register-only residual", "next_action": "search", "why_stopped": "",
             "excluded_actions": {"refine": "no refiner configured"}},
            {"image": "main", "name": "f3", "case": CASES[2].case_id, "size": 32, "status": "assumed",
             "primary_cause": "link/address mismatch", "next_action": None,
             "why_stopped": "no assumption is a self-describing D_<address> name",
             "excluded_actions": {"declare": "no assumption is a self-describing D_<address> name"}}]


def _snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns,
                                             hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}


class ReceiptTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-receipt-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.run = self.root / "run"
        (self.run / "solver").mkdir(parents=True)
        (self.run / "main").mkdir()
        store = resolvemod.CaseStore(self.run / "state")
        try:
            for case in CASES:
                store.put(case)
        finally:
            store.close()
        self._json("solver/invocation.json", {
            "effort": "exhaustive", "mode_settings": {"recipe_gate": False, "regional_calibration": True},
            "budget": 400, "rounds": 8, "schedule": "order", "search_ceiling": None, "cross_recipes": 12,
            "max_size": 0, "project": "<project>", "skip_stage": [], "toolchain_dir": None})
        log = solver.RoundsLog(self.run / "rounds.jsonl")
        budget = {"seconds": None, "builds": None, "model_calls": 5, "spent_builds": 60, "spent_model_calls": 0}
        log.append({"kind": "round", "session": "s1", "image": "main", "round": 1, "techniques": {}, "cascade": {},
                    "skipped": {}, "budget": budget})
        log.append({"kind": "round", "session": "s1", "image": "main", "round": 2, "techniques": {}, "cascade": {},
                    "skipped": {"refine": {"cases": 2, "reason": "model-call budget reached"}}, "budget": budget})
        log.append({"kind": "stop", "session": "s1", "image": "main", "after_round": 2, "reason": FIXED,
                    "budget": budget})
        self._json("solver/coverage_main.json", {
            "functions": 4, "attempted": 3, "not_attempted": 1, "unexplained": 0,
            "reasons": {"recovered from a donor (proven source of another title)": 1},
            "entries": [{"name": "f1", "vram": "0x80001000", "size": 64, "attempted": True, "status": "exact"},
                        {"name": "f2", "vram": "0x80001040", "size": 32, "attempted": True, "status": "open"},
                        {"name": "f3", "vram": "0x80001060", "size": 32, "attempted": True, "status": "assumed"},
                        {"name": "f4", "vram": "0x80001080", "size": 32, "attempted": False,
                         "reason": "recovered from a donor (proven source of another title)"}]})
        self._json("solver/hard_queue_main.json", _dossiers())
        (self.run / "events.jsonl").write_text('{"event": "image-start", "image": "main"}\n'
                                               '{"event": "image-done", "image": "main"}\n')
        self._json("main/stage_ledger.json", _report()["images"]["main"]["ledger"])
        self._json("report.json", _report())

    def _json(self, rel, value) -> None:
        (self.run / rel).write_text(json.dumps(value, indent=1))

    def _status(self, result) -> dict:
        return {row["clause"]: row["status"] for row in result["contract"]}

    def test_a_finished_exhaustive_run(self):
        before = _snapshot(self.run)
        result = rc.receipt(self.run)
        self.assertEqual(_snapshot(self.run), before)                     # read-only: nothing written
        status = self._status(result)
        self.assertEqual(set(status), set(rc.CLAUSES))
        self.assertEqual(status, {
            "W0": rc.SATISFIED, "W1": rc.SATISFIED, "W2": rc.NOT_SATISFIED, "W3": rc.SATISFIED,
            "W4": rc.SATISFIED, "W5": rc.SATISFIED, "W6": rc.SATISFIED, "W7": rc.SATISFIED, "W8": rc.SATISFIED,
            "W9": rc.SATISFIED, "W10": rc.NOT_VERIFIABLE, "W11": rc.SATISFIED, "W12": rc.SATISFIED,
            "W13": rc.SATISFIED, "W14": rc.NOT_VERIFIABLE, "W15": rc.NOT_VERIFIABLE, "W16": rc.SATISFIED})
        self.assertEqual(result["contract_summary"], {rc.SATISFIED: 13, rc.NOT_SATISFIED: 1, rc.NOT_VERIFIABLE: 3})
        w2 = next(row for row in result["contract"] if row["clause"] == "W2")
        self.assertIn("overlay base inference", w2["detail"])
        self.assertIn("discovery/overlays.json", w2["evidence"])
        self.assertEqual(result["command"]["effort"], "exhaustive")
        self.assertEqual(result["command"]["mode"]["recipe_gate"], False)
        self.assertEqual(result["command"]["sessions"], ["s1"])
        cases = result["counts"]["cases"]
        self.assertEqual((cases["total"], cases["exact_c"], cases["awaiting_declarations"], cases["unresolved"]),
                         (3, 1, 1, 1))
        self.assertEqual(result["counts"]["not_attempted"]["functions"], 1)
        self.assertEqual(result["unresolved"]["by_primary_cause"]["register-only residual"],
                         {"functions": 1, "bytes": 32})
        self.assertEqual(result["stages"]["images"]["main"]["donors"]["state"], "not run")
        self.assertEqual(result["stages"]["images"]["main"]["calibrate"]["state"], "not run")
        self.assertIn("regional calibration", result["stages"]["images"]["main"]["calibrate"]["detail"])
        self.assertEqual(result["stages"]["run"]["overlay-base-inference"]["state"], "unavailable")
        self.assertEqual(result["cache"]["requests_without_a_new_object"], 20)
        self.assertEqual(result["budgets"]["stops"][0]["reason"], FIXED)
        self.assertEqual(result["catalogue"]["images"]["main"]["functions"], 4)
        self.assertTrue(result["image_graph"]["digest"])
        paths = [row["path"] for row in result["files"]]
        self.assertIn("state/cases.sqlite", paths)
        self.assertIn("report.json", paths)
        self.assertEqual(rc.receipt(self.run), result)                     # deterministic
        text = rc.render(result)
        self.assertIn("| W2 | not satisfied |", text)
        self.assertIn("13 satisfied, 1 not satisfied, 3 not verifiable", text)
        json.dumps(result)

    def test_overlay_evidence_satisfies_the_discovery_clause(self):
        (self.run / "discovery").mkdir()
        self._json("discovery/overlays.json", {"main": {"current": "0x80000000", "runs": []}})
        self.assertEqual(self._status(rc.receipt(self.run))["W2"], rc.SATISFIED)

    def test_a_toolchain_manifest_beside_the_report_identifies_the_tools(self):
        self._json("toolchain_manifest.json", {
            "schema": 1, "digest": "d" * 64, "recipes": {"ido5.3": {"identity": "i" * 16, "runnable": True}},
            "components": [{"ident": "ido5.3", "installed": True,
                            "files": [{"file": "cc", "sha256": "a" * 64, "bytes": 4, "format": "elf"}]},
                           {"ident": "sn", "installed": False, "files": []}]})
        result = rc.receipt(self.run)
        self.assertEqual(self._status(result)["W14"], rc.SATISFIED)
        self.assertEqual(result["toolchain"]["recorded_digests"], {"ido5.3/cc": "a" * 64})
        self.assertEqual(result["toolchain"]["manifest"]["components_installed"], 1)

    def test_failures_and_gaps_are_not_satisfied(self):
        report = _report()
        main = report["images"]["main"]
        main["ledger"] = {"error": "boom"}
        main["toolchain_regions"] = {"observation_error": "ValueError: bad marks"}
        main["solver_stop"] = {"reason": "round limit: effort normal allows 1 round(s)"}
        del main["boundaries"]
        report["failed_stages"] = {"main": ["ledger"]}
        report["status"] = "incomplete"
        self._json("report.json", report)
        self._json("solver/invocation.json", {"effort": "normal", "skip_stage": []})
        self._json("solver/hard_queue_main.json", _dossiers()[:1])
        coverage = json.loads((self.run / "solver" / "coverage_main.json").read_text())
        coverage["unexplained"] = 1
        self._json("solver/coverage_main.json", coverage)
        (self.run / "events.jsonl").write_text('{"event": "image-start", "image": "main"}\n'
                                               '{"event": "image-start", "image": "aux"}\n')
        solver.RoundsLog(self.run / "rounds.jsonl").append(
            {"kind": "round", "session": "s1", "image": "main", "round": 3, "skipped": {"search": {"cases": 1}}})
        (self.run / "report.json").touch()
        result = rc.receipt(self.run)
        status = self._status(result)
        for clause in ("W1", "W4", "W5", "W6", "W7", "W8", "W9", "W12"):
            self.assertEqual(status[clause], rc.NOT_SATISFIED, clause)
        detail = {row["clause"]: row["detail"] for row in result["contract"]}
        self.assertIn("observation errors not reflected in report.status", detail["W7"])
        self.assertIn("aux", detail["W8"])
        self.assertIn("main/boundaries", detail["W8"])
        self.assertIn("1 of 2 unresolved cases have no dossier", detail["W5"])
        self.assertEqual(result["stages"]["images"]["main"]["toolchain-regions"]["state"], "failed")

    def test_a_run_without_a_report_is_not_finished(self):
        (self.run / "report.json").unlink()
        status = self._status(rc.receipt(self.run))
        self.assertEqual((status["W0"], status["W7"], status["W8"], status["W16"]),
                         (rc.NOT_SATISFIED, rc.NOT_VERIFIABLE, rc.NOT_SATISFIED, rc.NOT_SATISFIED))

    def test_command_writes_json_and_markdown_and_require_fails_on_a_gap(self):
        out = self.root / "receipt"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main(["--format", "json", "decompile-receipt", str(self.run), "--out", str(out)])
        self.assertEqual(code, 0)
        expected = json.loads(json.dumps(rc.receipt(self.run), default=str))
        self.assertEqual(json.loads(buffer.getvalue()), expected)
        self.assertEqual(json.loads((out / "receipt.json").read_text()), expected)
        self.assertIn("# Exhaustive-treatment receipt: run", (out / "receipt.md").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["decompile-receipt", str(self.run), "--require"]), 4)
        digest = rc.receipt(self.run)["run_digest"]
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(["decompile-receipt", str(self.run), "--out", str(self.run)])
        self.assertTrue((self.run / "receipt.json").is_file())
        self.assertEqual(rc.receipt(self.run)["run_digest"], digest)       # the receipt is not in its own digest


class AssemblyPinTest(unittest.TestCase):
    def test_an_assembly_case_pins_the_labels_that_state_their_address(self):
        from types import SimpleNamespace
        from n64dk.commands import verifyreceipt as vr
        from n64dk.match import handasm
        case = SimpleNamespace(recipe="gnu-as -march=vr4300", status=handasm.STATUS)
        ctx = SimpleNamespace(base=None, symbols={"D_80010000": 0x80010000})
        text = ("glabel func_8000A28C\n    b L_8000A174\n    lui $at, %hi(jtbl_80010CD0)\n"
                "    lw $t0, %lo(D_80010000)($at)\n.L8000A2A0:\n    jal send_mesg\n")
        spec = vr._spec_for(case, ctx, text)
        self.assertEqual(spec.symbols, {"L_8000A174": 0x8000A174, "jtbl_80010CD0": 0x80010CD0})
        from n64dk.match.handasm import _Symbols                     # a map that claims every stated name
        ctx = SimpleNamespace(base=None, symbols=_Symbols({"D_80010000": 0x80010000}))
        self.assertIn("L_8000A174", ctx.symbols)
        self.assertEqual(vr._spec_for(case, ctx, text).symbols,
                         {"L_8000A174": 0x8000A174, "jtbl_80010CD0": 0x80010CD0})
        c_case = SimpleNamespace(recipe="kmc+as -O2", status="exact")
        self.assertEqual(vr._spec_for(c_case, ctx, text).symbols, {})
        # a run or recipe pin keeps its own value; the issuer and the replay share one rule
        pinned = SimpleNamespace(recipe="gnu-as -march=vr4300 --symbol=L_8000A174:0x8000A170", status=handasm.STATUS)
        self.assertEqual(vr._spec_for(pinned, ctx, text).symbols["L_8000A174"], 0x8000A170)
        from n64dk.match import proofreceipt as pr
        self.assertEqual(pr.stated_pins(text, ctx.symbols), {"L_8000A174": 0x8000A174, "jtbl_80010CD0": 0x80010CD0})


class PinJustificationTest(unittest.TestCase):
    """Replay accepts a symbol pin or a section base of a receipt's recipe only when something it
    holds justifies it: the run's declarations, the recipe the result is offered with, the declared
    map, or an address the assembly source states.  ``proofreceipt.issue`` takes any recipe, so
    each forged receipt below is what anyone can write; before the check each one replayed as a
    strict proof.  Fake builder and hand-built objects (tests.test_proofreceipt); no toolchain."""

    def setUp(self):
        from tests import test_proofreceipt as tp
        self.tp = tp
        self.tmp = Path(tempfile.mkdtemp(prefix="n64dk-pin-test-")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.world = tp.World(self.tmp)

    def issue(self, ctx, spec, source=None):
        from n64dk.match import proofreceipt as pr
        return pr.issue(ctx, name="func", vram=self.tp.TEXT, size=0x10, source=source or self.world.source,
                        spec=spec, issued_at=self.tp.STAMP)

    def replay(self, receipt, ctx, source=None, **options):
        from n64dk.match import proofreceipt as pr
        return pr.replay(receipt, ctx, source or self.world.source, **options)

    def assembly(self, stated: str):
        """``func.s`` reading ``stated`` (a name that states an address) and the object it assembles to."""
        from n64dk.elf import R_MIPS_HI16, R_MIPS_LO16
        from tests.test_verify_adversarial import FUNC, GLOBAL, JR_RA, LUI, LW, NOP, NOTYPE, build, i_type, words
        code = words(i_type(LUI, 0, 2, 0), i_type(LW, 2, 2, 0), JR_RA, NOP)
        elf = build([(".text", code), (".data", b"\x00\x00\x00\x2a")],
                    [(stated, 0, GLOBAL, NOTYPE, 0), ("func", 0, GLOBAL, FUNC, ".text")],
                    {".text": [(0, R_MIPS_HI16, stated), (4, R_MIPS_LO16, stated)]})
        source = self.world.project / "src" / "func.s"
        source.write_text(f"glabel func\n    lui $v0, %hi({stated})\n    lw $v0, %lo({stated})($v0)\n"
                          "    jr $ra\n    nop\n")
        return source, self.tp.Builder(elf)

    def failed(self, result, prefix: str) -> None:
        self.assertFalse(result.ok)
        self.assertEqual(result.checks.get("pins"), "failed", result.failures)
        self.assertTrue(any(item.startswith(prefix) for item in result.failures), result.failures)

    def test_a_pin_nothing_justifies_is_rejected(self):
        from n64dk.match import proofreceipt as pr
        ctx = self.world.ctx(symbols={})                              # gValue declared nowhere
        honest = self.issue(ctx, self.world.spec())
        self.assertEqual(honest["verdict"]["class"], "assumed")
        self.assertTrue(self.replay(honest, ctx).ok)
        spec = self.world.spec()
        spec.symbols["gValue"] = self.tp.VALUE                        # the address the image needs
        forged = self.issue(ctx, spec)
        self.assertEqual(pr.check_static(forged), [])
        self.assertTrue(forged["claims"]["strict_proof"])
        for rebuild in (False, True):
            self.failed(self.replay(forged, ctx, rebuild=rebuild), "symbol pin gValue = 0x80001000 is not justified")
        # the same pin carried by the recipe the result is offered with is a declaration of the run
        offered = self.replay(forged, ctx, expect={"recipe": "kmc+as -O2 -Iinclude --symbol=gValue:0x80001000"})
        self.assertTrue(offered.ok, offered.failures)
        self.failed(self.replay(forged, ctx, expect={"recipe": "kmc+as -O2 -Iinclude"}), "symbol pin gValue")

    def test_a_pin_may_not_override_a_declaration(self):
        from n64dk.match import proofreceipt as pr
        wrong = self.world.ctx(symbols={"gValue": self.tp.VALUE + 4})
        with self.assertRaises(pr.ReceiptError):
            self.issue(wrong, self.world.spec())                       # honestly: a mismatch, nothing to receipt
        spec = self.world.spec()
        spec.symbols["gValue"] = self.tp.VALUE
        forged = self.issue(wrong, spec)
        self.assertTrue(forged["claims"]["strict_proof"])
        result = self.replay(forged, wrong)
        self.failed(result, "symbol pin gValue = 0x80001000 overrides its declaration (0x80001004)")

    def test_a_section_base_nothing_justifies_is_rejected(self):
        from n64dk.match.pipeline import BuildSpec
        ctx = self.world.ctx(base=BuildSpec("kmc+as", [], {}))         # the run declares no .data base
        honest = self.issue(ctx, BuildSpec("kmc+as", ["-O2", "-Iinclude"], {}))
        self.assertFalse(honest["claims"]["strict_proof"])              # .data not compared
        self.assertTrue(self.replay(honest, ctx).ok)
        forged = self.issue(ctx, self.world.spec())                    # .data placed where its bytes are
        self.assertTrue(forged["claims"]["strict_proof"])
        self.failed(self.replay(forged, ctx), "section base .data = 0x80000800 is not justified")

    def test_an_assembly_pin_is_the_address_its_source_states(self):
        from n64dk.match import proofreceipt as pr
        from n64dk.match.pipeline import BuildSpec
        source, builder = self.assembly("D_80001000")
        ctx = self.world.ctx(symbols={}, builder=builder)
        stated = pr.stated_pins(source.read_text(), ctx.symbols)
        self.assertEqual(stated, {"D_80001000": 0x80001000})
        receipt = self.issue(ctx, BuildSpec("gnu-as", ["-march=vr4300"], {".data": self.tp.DATA}, stated), source)
        self.assertEqual((receipt["verdict"]["class"], receipt["claims"]["strict_proof"]), ("assembly", True))
        result = self.replay(receipt, ctx, source)
        self.assertTrue(result.ok, result.failures)
        # a pin for a name the source does not state
        extra = self.issue(ctx, BuildSpec("gnu-as", ["-march=vr4300"], {".data": self.tp.DATA},
                                          {**stated, "L_80002000": 0x80002000}), source)
        self.failed(self.replay(extra, ctx, source), "symbol pin L_80002000 = 0x80002000 is not justified")
        # declared later at another address: the declaration decides, the receipt no longer holds
        declared = self.world.ctx(symbols={"D_80001000": 0x80001004}, builder=builder)
        self.failed(self.replay(receipt, declared, source),
                    "symbol pin D_80001000 = 0x80001000 overrides its declaration (0x80001004)")

    def test_a_name_cannot_state_one_address_and_be_pinned_at_another(self):
        from n64dk.match import proofreceipt as pr
        from n64dk.match.pipeline import BuildSpec
        source, builder = self.assembly("D_80001004")                 # states 0x80001004; the image needs 0x80001000
        ctx = self.world.ctx(symbols={}, builder=builder)
        stated = pr.stated_pins(source.read_text(), ctx.symbols)
        with self.assertRaises(pr.ReceiptError):
            self.issue(ctx, BuildSpec("gnu-as", ["-march=vr4300"], {".data": self.tp.DATA}, stated), source)
        # a pin that contradicts the address the name spells is refused by the strict comparison itself
        # (verify.ADDRESS_NAME), at issue as at replay
        with self.assertRaises(pr.ReceiptError):
            self.issue(ctx, BuildSpec("gnu-as", ["-march=vr4300"], {".data": self.tp.DATA},
                                      {"D_80001004": 0x80001000}), source)


if __name__ == "__main__":
    unittest.main()


class ExportedAbsoluteUnitTest(unittest.TestCase):
    """An export writes a donor unit kept outside its donor root as ``<home>/...``; replay files it by its
    last two components, as for the run's absolute path (17 Dr. Mario SDK .s units, cold replay 2026-11-18)."""

    def test_placeholder_roots_are_absolute(self):
        from pathlib import Path
        from n64dk.commands.verifyreceipt import filed_path
        for unit in ("/home/user/.cache/sdk/asm/libc/bcmp.s", "<home>/.cache/sdk/asm/libc/bcmp.s",
                     "<run>/x/libc/bcmp.s", "<project>/libc/bcmp.s"):
            self.assertEqual(filed_path(unit), Path("libc/bcmp.s"), unit)
        self.assertEqual(filed_path("src/os/a.c"), Path("src/os/a.c"))
        self.assertIsNone(filed_path("../a.c"))


class ContentPlacedSectionTest(unittest.TestCase):
    """A section base a generated-C receipt records as confirmed by the unit's own data needs no declaration:
    replay confirms it again.  Any other unexplained base is still refused."""

    def problems(self, earned_by, origins):
        import types
        from n64dk.match import proofreceipt
        receipt = {"earned_by": earned_by, "source": {"kind": "c"},
                   "build": {"symbols": {}, "section_bases": {".rodata": "0x80000508"}, "pin_origins": origins}}
        ctx = types.SimpleNamespace(base=None, symbols={})
        return proofreceipt._pin_problems(receipt, ctx, b"", None)

    def test_content_confirmed_base_of_generated_c_is_accepted(self):
        self.assertEqual(self.problems("generated-c", {"section:.rodata": "section-content"}), [])

    def test_other_bases_are_refused(self):
        self.assertTrue(self.problems("generated-c", {}))
        self.assertTrue(self.problems("source-tree-donor", {"section:.rodata": "section-content"}))



class StatedSectionOriginTest(unittest.TestCase):
    """A section base with origin ``declared`` holds only while the project's symbol file states it."""

    def problems(self, symbols):
        from pathlib import Path
        from n64dk.match import proofreceipt
        build = {"symbols": {}, "section_bases": {".bss": "0x800AFE60"},
                 "pin_origins": {"section:.bss": "declared: the project's symbol file states this unit's section base"}}
        ctx = proofreceipt.Context(Path("."), "main", b"", 0, symbols=symbols)
        return proofreceipt.origin_problems(build, ctx)

    def test_a_stated_base_is_accepted(self):
        self.assertEqual(self.problems({"n64dk_section_80038998_bss": 0x800AFE60}), [])

    def test_a_base_the_file_no_longer_states_is_a_problem(self):
        self.assertTrue(self.problems({}))
        self.assertTrue(self.problems({"n64dk_section_80038998_bss": 0x800AFE64}))
        self.assertTrue(self.problems({"n64dk_section_80038998_data": 0x800AFE60}))

    def test_the_name_a_project_writes(self):
        from n64dk.match import transplant
        self.assertEqual(transplant.stated_section_name(0x80038998, ".bss"), "n64dk_section_80038998_bss")
