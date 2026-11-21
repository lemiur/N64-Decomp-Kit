"""Review W4b_18: credit soundness of the donor pin rules, the receipt origin contract, donor
receipt paths and blind run-directory provenance.  Each test fails on the tree before the fix.
Hand-built inputs only (tiny big-endian ELF objects from tests.test_verify_adversarial through
tests.test_proofreceipt's World); no toolchain, no ROM, no kit run."""
import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from n64dk import governance as gov
from n64dk.commands import run
from n64dk.commands import verifyreceipt as vr
from n64dk.match import proofreceipt as pr
from n64dk.match import transplant as tp
from tests.test_proofreceipt import BASE, HEADER, SOURCE, STAMP, TEXT, Base, reseal

UNIT = (0x80200100, 0x80200180)              # bytes a unit is compared over


class PinSpanTest(unittest.TestCase):
    """A supporter or a definition inside the unit's own bytes is the unit, whatever its name."""

    def test_a_supporter_inside_the_unit_is_the_unit_itself(self):
        # the catalogue split the unit's second function: its tail (t_tail) matched another donor
        # function exactly and voted for the address the unit's own instruction reads
        table = {("D", 0x80123456): {"target_address": 0x80223466, "hits": 2, "functions": ["t_f", "t_tail"],
                                     "supporters": [("t_f", 0x80200100, 0x28), ("t_tail", 0x80200160, 0x10)]}}
        self.assertEqual(tp.classify_pin("D", 0x80123456, 0x80223466, table, {}, {"t_f"}, span=UNIT), ("self", ""))
        # the same supporter outside the unit's bytes is independent
        decision, origin = tp.classify_pin("D", 0x80123456, 0x80223466, table, {}, {"t_f"},
                                           span=(0x80200100, 0x80200140))
        self.assertEqual(decision, "declare")
        self.assertIn("e.g. t_tail", origin)

    def test_a_definition_must_be_unique_and_outside_the_unit(self):
        several = {("D", 0x80101000): {0x80205000, 0x80206000}}
        self.assertEqual(tp.classify_pin("D", 0x80101000, 0x80205000, {}, several, set(), span=UNIT)[0], "unchecked")
        inside = {("D", 0x80101000): {0x80200140}}
        self.assertEqual(tp.classify_pin("D", 0x80101000, 0x80200140, {}, inside, set(), span=UNIT), ("self", ""))
        unique = {("D", 0x80101000): {0x80205000}}
        self.assertEqual(tp.classify_pin("D", 0x80101000, 0x80205000, {}, unique, set(), span=UNIT),
                         ("declare", "definition"))

    def test_transfer_records_where_each_supporter_stands(self):
        from tests.test_transplant import donor_fn, image_with, make_unit
        delta = 0x80200000 - 0x1000
        unit = make_unit([("donor_f", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000)
        target = donor_fn(0x8022, 0x3466, 0x80205000)
        image = image_with(delta, {0x80200100: target})
        index = tp.Index({"0": unit}, {"D": {"root": "/x", "image": "/x", "delta": 0, "include": []}})
        hits = tp.scan(image, delta, [("t_f", 0x80200100, len(target))], index)
        symbols = tp.transfer(hits)["symbols"]
        self.assertTrue(symbols)
        self.assertEqual({tuple(map(tuple, v["supporters"])) for v in symbols.values()},
                         {(("t_f", 0x80200100, len(target)),)})


class OriginContractTest(Base):
    """Only named origins support a strict proof, and the ones the context can check must hold."""

    def donor_run(self, origins, source_cell="src/func.c", donor="sdk_2_0L_libultra", filed="src/func.c") -> Path:
        run_dir = self.tmp / "run"
        shutil.rmtree(run_dir, ignore_errors=True)
        folder = run_dir / "main" / "donors"
        tree = folder / "src" / donor
        (tree / "include").mkdir(parents=True)
        (tree / "include" / "defs.h").write_text(HEADER)
        (tree / filed).parent.mkdir(parents=True, exist_ok=True)
        (tree / filed).write_text(SOURCE)
        flags = f"-O2 -Iinclude --section-base=.data:0x{0x80000800:08X}"
        (folder / f"proposed_{donor}_kmc_as.tsv").write_text(
            f"# kmc+as rows proposed by donor transplant from {donor}; SOURCE paths are relative to\n"
            f"# src/{donor}/\n# ROM_START\tSIZE\tSOURCE\tFLAGS\n0x{TEXT - BASE:X}\t0x10\t{source_cell}\t{flags}\n")
        row = {"target_vram": f"0x{TEXT:08X}", "verdict": "exact-with-transferred-symbols", "pin_origin": origins}
        (folder / "report.json").write_text(json.dumps({"rows": [row]}))
        return run_dir

    def contexts(self):
        world = self.world
        return mock.Mock(get=lambda image: world.ctx(base=None, project=world.project))

    def test_an_unknown_or_empty_origin_supports_nothing(self):
        for origin in ("", "self", "target-derived", "derived from the unit"):
            run_dir = self.donor_run({"section:.data": origin})
            summary = vr.issue_donor_receipts(run_dir, contexts=self.contexts(), issued_at=STAMP)
            self.assertEqual((summary["issued"], summary["strict_proof"]), (1, 0), (origin, summary))

    def test_an_origin_the_context_contradicts_is_refused(self):
        # the donor context has no run declarations: "run-declaration" cannot hold, and a section
        # base is never a fixed address of the declared map
        for origin in ("run-declaration", "fixed-address", "declared"):
            run_dir = self.donor_run({"section:.data": origin})
            summary = vr.issue_donor_receipts(run_dir, contexts=self.contexts(), issued_at=STAMP)
            self.assertEqual((summary["issued"], summary["strict_proof"]), (0, 0), (origin, summary))

    def test_a_relabelled_origin_fails_replay(self):
        forged = copy.deepcopy(self.receipt)
        forged["build"]["symbols"] = {}
        forged["build"]["pin_origins"] = {"section:.data": "fixed-address"}
        forged["claims"], forged["proof"] = pr.derive(forged["verdict"]["class"], forged["verdict"]["strict"],
                                                      None, forged["provenance"], forged["build"])
        reseal(forged)
        self.assertEqual(pr.check_static(forged), [])
        result = self.replay(forged, rebuild=False)
        self.assertFalse(result.ok)
        self.assertEqual(result.checks.get("origins"), "failed", result.failures)

    def test_a_raw_listing_relabelled_as_assembly_is_rejected(self):
        source = self.world.project / "src" / "func.s"
        source.write_text("# raw words: kept as the image has them\n.word 0x3C020000\n")
        receipt = self.world.issue(source=source)
        self.assertEqual(receipt["verdict"]["class"], "raw")
        self.assertFalse(receipt["claims"]["strict_proof"])
        self.assertTrue(self.replay(receipt, source=source, rebuild=False).ok)
        forged = copy.deepcopy(receipt)
        forged["verdict"]["class"] = "assembly"
        forged["claims"], forged["proof"] = pr.derive("assembly", forged["verdict"]["strict"], None,
                                                      forged["provenance"], forged["build"])
        reseal(forged)
        self.assertTrue(forged["claims"]["strict_proof"])
        self.assertEqual(pr.check_static(forged), [])
        result = self.replay(forged, source=source)
        self.assertFalse(result.ok)
        self.assertEqual(result.checks.get("class"), "failed", result.failures)


class DonorSourcePathTest(OriginContractTest):
    """A manifest SOURCE names a file of the donor tree the manifest folder holds, never one elsewhere."""

    def test_filed_paths(self):
        self.assertEqual(vr.filed_path("src/os/x.c"), Path("src/os/x.c"))
        self.assertEqual(vr.filed_path("/x/ultralib/src/os/x.c"), Path("os/x.c"))
        self.assertIsNone(vr.filed_path("../other/x.c"))
        self.assertIsNone(vr.filed_path("src/../../x.c"))
        self.assertIsNone(vr.filed_path("/"))

    def test_an_absolute_source_outside_the_export_does_not_verify(self):
        run_dir = self.donor_run({"section:.data": "section-content"})
        vr.issue_donor_receipts(run_dir, contexts=self.contexts(), issued_at=STAMP)
        folder = run_dir / "main" / "donors"
        elsewhere = self.tmp / "elsewhere" / "func.c"                  # a file the export does not hold
        elsewhere.parent.mkdir(parents=True)
        elsewhere.write_text(SOURCE)
        (folder / "src" / "sdk_2_0L_libultra" / "src" / "func.c").unlink()
        manifest = folder / "proposed_sdk_2_0L_libultra_kmc_as.tsv"
        manifest.write_text(manifest.read_text().replace("\tsrc/func.c\t", f"\t{elsewhere}\t"))
        results, _known = vr.verify_donors([("main", folder)], contexts=self.contexts(),
                                           receipts_dir=run_dir / "receipts")
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][1] is None or not results[0][1].ok, results[0])

    def test_a_unit_kept_outside_its_donor_root_is_receipted_where_it_is_filed(self):
        # transplant.write_outputs writes such a unit's SOURCE as an absolute path and files the
        # source under its last two components
        run_dir = self.donor_run({"section:.data": "section-content"}, source_cell="/x/ultralib/os/func.c",
                                 filed="os/func.c")
        summary = vr.issue_donor_receipts(run_dir, contexts=self.contexts(), issued_at=STAMP)
        self.assertEqual((summary["issued"], summary["strict_proof"]), (1, 1), summary)
        receipt = pr.load(pr.receipt_path(run_dir / "receipts" / "donors", "main", "func", TEXT))
        self.assertEqual(receipt["source"]["name"], "os/func.c")
        results, _known = vr.verify_donors([("main", run_dir / "main" / "donors")], contexts=self.contexts(),
                                           receipts_dir=run_dir / "receipts")
        self.assertTrue(results[0][1] is not None and results[0][1].ok, results[0])

    # the inherited origin tests run once, in OriginContractTest
    test_an_unknown_or_empty_origin_supports_nothing = None
    test_an_origin_the_context_contradicts_is_refused = None
    test_a_relabelled_origin_fails_replay = None
    test_a_raw_listing_relabelled_as_assembly_is_rejected = None


class BlindProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-w4b18-blind-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        environment = mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.truth = self.root / "public_src" / "title"
        (self.truth / "src").mkdir(parents=True)
        (self.truth / "src" / "game.c").write_text("void func_80001000(void) {\n    gCounter += 1;\n}\n")
        self.project = self.root / "proj"
        self.project.mkdir()
        (self.project / "n64dk.toml").write_text('[run]\nsymbols = ["symbols.txt"]\n')
        (self.project / "symbols.txt").write_text("func_80001000 = 0x80001000;\n")
        self.out = self.root / "out"

    def args(self, **extra):
        return SimpleNamespace(project=str(self.project), out=str(self.out), donors=None,
                               blind_forbid=[str(self.truth)], **extra)

    def test_a_run_that_fails_after_reaching_a_forbidden_tree_marks_its_session(self):
        def body(_args):
            (self.project / "n64dk.toml").write_text(f'[run]\nsymbols = ["{self.truth / "src" / "game.c"}"]\n')
            raise RuntimeError("a stage failed")
        with mock.patch.object(run, "_decompile_run", side_effect=body), self.assertRaises(SystemExit) as stop:
            run._blind_run(self.args())
        self.assertEqual(stop.exception.code, 3)
        sessions = json.loads(gov.provenance_path(self.out).read_text())["sessions"]
        self.assertEqual(sessions[0]["status"], "violated")
        (self.project / "n64dk.toml").write_text('[run]\nsymbols = ["symbols.txt"]\n')     # unreachable again
        with mock.patch.object(run, "_decompile_run") as again, self.assertRaises(SystemExit):
            run._blind_run(self.args())
        again.assert_not_called()

    def test_a_clean_failure_still_raises_its_own_error(self):
        with mock.patch.object(run, "_decompile_run", side_effect=RuntimeError("a stage failed")), \
                self.assertRaises(RuntimeError):
            run._blind_run(self.args())

    def test_a_refused_first_attempt_leaves_the_directory_usable(self):
        with mock.patch.object(run, "_decompile_run") as body, self.assertRaises(SystemExit) as stop:
            run._blind_run(self.args(blind_shared_library=["unpinned-manifest.json"]))
        self.assertEqual(stop.exception.code, 2)
        body.assert_not_called()
        with mock.patch.object(run, "_decompile_run", return_value={"images": {}}) as body:
            result = run._blind_run(self.args())
        body.assert_called_once()
        self.assertEqual(result["blind"]["run_dir"]["verdict"], "fresh")

    def test_only_refusal_receipts_count_as_fresh(self):
        forbidden = [("title", self.truth)]
        (self.out / "solver").mkdir(parents=True)
        (self.out / "solver" / "blind_violation.json").write_text("{}")
        self.assertEqual(gov.run_provenance(self.out, forbidden)["verdict"], "fresh")
        (self.out / "solver" / "invocation.json").write_text("{}")
        self.assertFalse(gov.run_provenance(self.out, forbidden)["ok"])


if __name__ == "__main__":
    unittest.main()
