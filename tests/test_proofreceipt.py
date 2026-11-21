"""Proof receipts (n64dk.match.proofreceipt, n64dk.commands.verifyreceipt): issue, replay, and the
ways a receipt must fail.  No toolchain: a fake builder hands back tiny hand-built big-endian ELF
objects (tests.test_verify_adversarial.build), the image is a bytearray whose offset is
``vram - BASE``, and the toolchain and wrapper identities are injected."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from n64dk.elf import R_MIPS_HI16, R_MIPS_LO16
from n64dk.match import proofreceipt as pr
from n64dk.match.pipeline import BuildSpec
from tests.test_verify_adversarial import (BASE, FUNC, GLOBAL, JR_RA, LUI, LW, NOP, NOTYPE, TEXT, build, i_type, image,
                                           words)

DATA, OTHER, VALUE = 0x80000800, 0x80000500, 0x80001000
ADDIU = 9
SOURCE = '#include "defs.h"\nint func(void) { return gValue; }\n'
HEADER = "extern int gValue;\n"
STAMP = "2026-01-01T00:00:00Z"
EVIDENCE = {"kind": "oracle", "result": "equivalent on 64 inputs", "digest": "ab" * 32}


def unit(reg: int = 2):
    """``lui reg, %hi(gValue); lw v0, %lo(gValue)(reg); jr ra; nop`` and four bytes of ``.data``."""
    code = words(i_type(LUI, 0, reg, 0), i_type(LW, reg, 2, 0), JR_RA, NOP)
    return build([(".text", code), (".data", b"\x00\x00\x00\x2a")],
                 [("gValue", 0, GLOBAL, NOTYPE, 0), ("func", 0, GLOBAL, FUNC, ".text")],
                 {".text": [(0, R_MIPS_HI16, "gValue"), (4, R_MIPS_LO16, "gValue")]})


class Builder:
    def __init__(self, elf):
        self.elf, self.calls = elf, 0

    def __call__(self, source, spec, workdir):
        self.calls += 1
        return self.elf


def toolchain(digest: str = "3" * 64):
    return lambda spec: {"pipeline": spec.pipeline, "toolchain_digest": "1" * 64,
                         "recipe": {"lane": spec.pipeline, "identity": "2" * 64,
                                    "stages": [{"stage": "compile", "component": "kmc/cc1",
                                                "files": [["cc1", digest]]}]}}


def wrapper(flags=("-mips3", "-G0")):
    return lambda spec: {"lane_code": {}, "normalizer_hook": None, "assembler_flags": list(flags)}


class World:
    """A project with one header and one symbol file, an image holding ``func`` at TEXT (its
    ``.data`` at DATA) and another function at OTHER."""

    def __init__(self, root: Path, name: str = "project"):
        self.root = root
        self.project = root / name
        (self.project / "include").mkdir(parents=True)
        (self.project / "src").mkdir()
        (self.project / "include" / "defs.h").write_text(HEADER)
        (self.project / "symbols.txt").write_text("gValue = 0x80001000;\n")
        self.source = self.project / "src" / "func.c"
        self.source.write_text(SOURCE)
        self.elf = unit()
        rom = image(self.elf, {".text": TEXT, ".data": DATA}, {"gValue": VALUE})
        rom[OTHER - BASE:OTHER - BASE + 16] = words(i_type(ADDIU, 0, 2, 1), JR_RA, NOP, NOP)
        self.rom = bytes(rom)
        self.builder = Builder(self.elf)

    def ctx(self, **changes) -> pr.Context:
        values = dict(project=self.project, image="main", data=self.rom, delta=BASE, symbols={"gValue": VALUE},
                      symbol_files=[self.project / "symbols.txt"], base=BuildSpec("kmc+as", [], {".data": DATA}),
                      catalogue={TEXT: 0x10, OTHER: 0x10}, function_starts={TEXT, OTHER}, builder=self.builder,
                      toolchain=toolchain(), wrapper=wrapper(), environment={}, include=[])
        values.update(changes)
        return pr.Context(**values)

    def spec(self, flags=("-O2", "-Iinclude")) -> BuildSpec:
        return BuildSpec("kmc+as", list(flags), {".data": DATA})

    def issue(self, ctx=None, **extra) -> dict:
        options = dict(name="func", vram=TEXT, size=0x10, source=self.source, spec=self.spec(), issued_at=STAMP)
        options.update(extra)
        return pr.issue(ctx or self.ctx(), **options)


def reseal(receipt: dict) -> dict:
    """What a forger can always do: recompute the identity after an edit."""
    receipt["identity"] = pr.receipt_identity(receipt)
    return receipt


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="n64dk-receipt-test-")).resolve()
        self.world = World(self.tmp)
        self.receipt = self.world.issue()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def replay(self, receipt=None, ctx=None, source=None, **options):
        return pr.replay(self.receipt if receipt is None else receipt, ctx or self.world.ctx(),
                         source or self.world.source, **options)


class IssueAndReplayTest(Base):
    def test_a_fresh_receipt_replays(self):
        receipt = self.receipt
        self.assertEqual(receipt["verdict"]["class"], "exact")
        self.assertEqual(receipt["verdict"]["strict"]["verdict"], "exact")
        self.assertTrue(receipt["claims"]["strict_proof"])
        self.assertTrue(receipt["claims"]["ownership_complete"])
        self.assertFalse(receipt["claims"]["behavioural_evidence"])
        self.assertTrue(receipt["proof"]["proof_grade"])
        self.assertEqual(receipt["headers"]["headers"], {"<project>/include/defs.h": pr._sha(HEADER.encode())})
        self.assertEqual(receipt["relocations"]["symbols"], {"gValue": "0x80001000"})
        self.assertEqual(set(receipt["owned"]["data"]), {".data"})
        self.assertEqual(pr.check_static(receipt), [])
        result = self.replay()
        self.assertTrue(result.ok, result.failures)
        self.assertTrue(result.rebuilt)
        self.assertEqual(self.world.builder.calls, 2)              # issued once, built again on replay
        self.assertTrue(all(state == "ok" for state in result.checks.values()), result.checks)

    def test_static_replay_checks_digests_without_building(self):
        result = self.replay(rebuild=False)
        self.assertTrue(result.ok, result.failures)
        self.assertFalse(result.rebuilt)
        self.assertEqual(result.checks["rebuild"], "skipped")
        self.assertEqual(self.world.builder.calls, 1)

    def test_receipts_are_deterministic_and_machine_independent(self):
        again = self.world.issue(issued_at="2027-05-05T05:05:05Z")
        self.assertEqual(again["identity"], self.receipt["identity"])
        self.assertNotEqual(again["issued"], self.receipt["issued"])
        strip = lambda r: json.dumps({k: v for k, v in r.items() if k != "issued"}, sort_keys=True)  # noqa: E731
        self.assertEqual(strip(again), strip(self.receipt))
        elsewhere = World(self.tmp / "machine2", "checkout")       # same tree, another place
        moved = elsewhere.issue(ctx=elsewhere.ctx())
        self.assertEqual(moved["identity"], self.receipt["identity"])
        absolute = self.world.issue(spec=self.world.spec(("-O2", f"-I{self.world.project / 'include'}")))
        self.assertEqual(absolute["build"]["flags"], ["-O2", "-Iinclude"])
        self.assertEqual(absolute["identity"], self.receipt["identity"])

    def test_no_machine_paths_or_image_bytes(self):
        text = json.dumps(self.receipt)
        self.assertNotIn(str(self.tmp), text)
        self.assertNotIn("/tmp", text)
        code = self.world.rom[TEXT - BASE:TEXT - BASE + 16]
        self.assertNotIn(code.hex(), text.lower())
        self.assertNotIn(code.hex().upper(), text)
        self.assertEqual(pr.machine_paths(self.receipt), [])
        with self.assertRaises(pr.ReceiptError):
            self.world.issue(spec=self.world.spec(("-O2", "-I/opt/elsewhere/include")))
        planted = reseal(copy.deepcopy(self.receipt))
        planted["provenance"]["origin"] = "/srv/build/src/func.c"
        self.assertTrue(any("machine path" in item for item in pr.check_static(reseal(planted))))


class ForgeryTest(Base):
    def test_a_tampered_digest_is_rejected(self):
        forged = copy.deepcopy(self.receipt)
        forged["source"]["sha256"] = "0" * 64
        result = self.replay(forged)
        self.assertFalse(result.ok)
        self.assertIn("forged or damaged", result.failures[0])
        self.assertEqual(self.world.builder.calls, 1)              # nothing was built for it
        result = self.replay(reseal(forged))                      # identity recomputed: the disk disagrees
        self.assertFalse(result.ok)
        self.assertTrue(any(item.startswith("source changed") for item in result.failures), result.failures)
        self.assertFalse(any("forged" in item for item in result.failures))

    def test_a_copied_receipt_for_another_address_is_rejected(self):
        result = self.replay(expect={"name": "other", "vram": OTHER})
        self.assertFalse(result.ok)
        self.assertTrue(any("offered for 0x80000500" in item for item in result.failures), result.failures)
        moved = copy.deepcopy(self.receipt)
        moved["subject"].update(name="other", vram=pr._hex(OTHER), image_offset=pr._hex(OTHER - BASE))
        result = self.replay(reseal(moved), expect={"name": "other", "vram": OTHER})
        self.assertFalse(result.ok)
        self.assertTrue(any(item.startswith("owned code bytes at 0x80000500") for item in result.failures),
                        result.failures)

    def test_behavioural_evidence_never_yields_a_strict_proof(self):
        wrong = Builder(unit(reg=3))                               # behaves the same, other bytes
        ctx = self.world.ctx(builder=wrong)
        with self.assertRaises(pr.ReceiptError):
            self.world.issue(ctx=ctx)
        receipt = self.world.issue(ctx=ctx, behavioural=EVIDENCE)
        self.assertEqual(receipt["verdict"]["class"], "behavioural")
        claims = receipt["claims"]
        self.assertTrue(claims["behavioural_evidence"])
        self.assertFalse(claims["strict_proof"] or claims["byte_identity"] or claims["proof_grade"])
        self.assertEqual(pr.check_static(receipt), [])
        # claiming more than the evidence gives is caught without a build
        bold = reseal(copy.deepcopy(receipt))
        bold["claims"]["strict_proof"] = True
        self.assertTrue(any(item.startswith("claim strict_proof") for item in pr.check_static(reseal(bold))))
        # a consistent forgery of the evidence itself is caught by the replay
        forged = copy.deepcopy(receipt)
        forged["verdict"]["strict"].update(verdict="exact", text_equal=True)
        forged["verdict"]["class"] = "exact"
        forged["claims"], forged["proof"] = pr.derive("exact", forged["verdict"]["strict"], EVIDENCE,
                                                      forged["provenance"], forged["build"])
        self.assertEqual(pr.check_static(reseal(forged)), [])
        result = self.replay(forged, ctx=ctx)
        self.assertFalse(result.ok)
        self.assertIn("strict verdict on replay is mismatch, the receipt records exact", result.failures)
        # byte-identical but resting on an address read from the image: still no strict proof
        assumed = self.world.issue(ctx=self.world.ctx(symbols={}), behavioural=EVIDENCE)
        self.assertEqual(assumed["verdict"]["class"], "assumed")
        self.assertTrue(assumed["claims"]["byte_identity"] and assumed["claims"]["assumptions_present"])
        self.assertFalse(assumed["claims"]["strict_proof"] or assumed["claims"]["proof_grade"])


class StaleInputTest(Base):
    def test_a_changed_header_is_rejected(self):
        (self.world.project / "include" / "defs.h").write_text(HEADER + "extern int gOther;\n")
        result = self.replay()
        self.assertEqual(result.failures, ["header <project>/include/defs.h changed"])
        self.assertTrue(result.checks["rebuild"].startswith("skipped"))

    def test_an_altered_catalogue_boundary_is_rejected(self):
        result = self.replay(ctx=self.world.ctx(catalogue={TEXT: 0x14, OTHER: 0x10}))
        self.assertIn("catalogue boundary changed at 0x80000400: 0x10 bytes when issued, now 0x14", result.failures)
        result = self.replay(expect={"size": 0x14})
        self.assertTrue(any(item.startswith("catalogue boundary changed: the receipt covers 0x10")
                            for item in result.failures), result.failures)
        grown = copy.deepcopy(self.receipt)
        grown["subject"]["size"] = 0x14
        result = self.replay(reseal(grown))
        self.assertTrue(any(item.startswith("owned code bytes") for item in result.failures), result.failures)

    def test_another_image_is_rejected(self):
        changed = bytearray(self.world.rom)
        changed[0x3000] ^= 0xFF                                    # outside everything the function owns
        result = self.replay(ctx=self.world.ctx(data=bytes(changed)))
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertTrue(result.failures[0].startswith("image identity changed"))

    def test_a_verifier_change_is_rejected(self):
        with mock.patch.object(pr, "VERIFIER_VERSION", "f" * 64):
            result = self.replay()
        self.assertFalse(result.ok)
        self.assertTrue(any(item.startswith("verifier version changed") for item in result.failures))
        old = copy.deepcopy(self.receipt)
        old["verifier"] = "e" * 64
        self.assertTrue(any(item.startswith("verifier version changed") for item in self.replay(reseal(old)).failures))

    def test_a_changed_declaration_is_rejected(self):
        result = self.replay(ctx=self.world.ctx(symbols={"gValue": VALUE + 4}))
        self.assertIn("declaration of gValue changed: 0x80001000 -> 0x80001004", result.failures)
        self.assertIn("strict verdict on replay is mismatch, the receipt records exact", result.failures)
        result = self.replay(ctx=self.world.ctx(base=BuildSpec("kmc+as", [], {".data": DATA + 0x10})))
        self.assertTrue(any(item.startswith("declaration changed: section base .data") for item in result.failures))

    def test_compiler_wrapper_and_environment_changes_are_rejected(self):
        result = self.replay(ctx=self.world.ctx(toolchain=toolchain("4" * 64)))
        self.assertTrue(any(item.startswith("compiler changed") and "compile (kmc/cc1)" in item
                            for item in result.failures), result.failures)
        result = self.replay(ctx=self.world.ctx(wrapper=wrapper(("-mips3", "-G8"))))
        self.assertTrue(any(item.startswith("wrapper changed: assembler flags") for item in result.failures))
        receipt = self.world.issue(ctx=self.world.ctx(environment={"SOURCE_DATE_EPOCH": "1"}))
        result = self.replay(receipt)
        self.assertEqual(result.failures, ["build environment: SOURCE_DATE_EPOCH was set when the receipt was "
                                           "issued and is not set now"])

    def test_a_different_rebuilt_object_is_rejected(self):
        result = self.replay(ctx=self.world.ctx(builder=Builder(unit(reg=3))))
        self.assertTrue(result.rebuilt)
        self.assertTrue(any(item.startswith("the rebuilt object differs") for item in result.failures))
        self.assertIn("strict verdict on replay is mismatch, the receipt records exact", result.failures)


class Contexts:
    def __init__(self, world: World):
        self.world = world

    def get(self, image: str):
        return self.world.ctx()


class RunAndExportTest(Base):
    def make_run(self, text: str = SOURCE) -> Path:
        from n64dk.match import resolve as resolvemod
        run = self.tmp / "run"
        store = resolvemod.CaseStore(run / "state")
        try:
            digest = store.save_source(text)
            store.put(resolvemod.Case(resolvemod.case_id("main", TEXT, 0x10), "main", "func", TEXT, 0x10, "t",
                                      status="exact", recipe="kmc+as -O2 -Iinclude", best_digest=digest))
        finally:
            store.close()
        return run

    def test_a_run_issues_and_verifies_its_receipts(self):
        from n64dk.commands import verifyreceipt as vr
        run = self.make_run()
        issued = vr.issue_receipts(run, contexts=Contexts(self.world), issued_at=STAMP)
        self.assertEqual((issued["issued"], issued["classes"], issued["strict_proof"]), (1, {"exact": 1}, 1))
        written = pr.load(pr.receipt_path(run / "receipts", "main", "func", TEXT))
        self.assertEqual(written["provenance"]["origin"], "draft")         # commands.export._origin
        self.assertEqual(written["identity"], self.world.issue(origin="draft")["identity"])
        summary = vr.verify_run(run, contexts=Contexts(self.world))
        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["replayed"], 1)
        # a receipt the run has no result for
        shutil.copyfile(pr.receipt_path(run / "receipts", "main", "func", TEXT), run / "receipts" / "main" / "x.json")
        self.assertEqual(vr.verify_run(run, contexts=Contexts(self.world))["orphans"], ["main/x.json"])
        (run / "receipts" / "main" / "x.json").unlink()
        # the run's result now has another source
        from n64dk.match import resolve as resolvemod
        store = resolvemod.CaseStore(run / "state")
        try:
            case = store.all()[0]
            case.best_digest = store.save_source(SOURCE + "/* edited */\n")
            store.put(case)
        finally:
            store.close()
        summary = vr.verify_run(run, contexts=Contexts(self.world))
        self.assertFalse(summary["ok"])
        self.assertTrue(any(item.startswith("source changed") for item in summary["rejected"][0]["failures"]))

    def test_an_export_row_verifies_and_a_changed_row_does_not(self):
        from n64dk.commands import verifyreceipt as vr
        export = self.tmp / "export"
        (export / "src" / "main").mkdir(parents=True)
        (export / "src" / "main" / "func.c").write_text(SOURCE)
        pr.write(self.receipt, pr.receipt_path(export / "receipts", "main", "func", TEXT))
        header = ("image\tfunction\tvram\trom_offset\tsize\tcode_size\toutcome\thow\tsource\tdistance\treproduced_by"
                  "\tunit_class\n")

        def table(code_size: str) -> None:
            (export / "FUNCTIONS.tsv").write_text(
                header + f"main\tfunc\t0x80000400\t0x400\t0x10\t{code_size}\texact\tkmc+as -O2 -Iinclude (draft)\t"
                         "src/main/func.c\t\tC\tgame\n"
                         "main\tlib\t0x80000600\t0x600\t0x20\t0x20\tdonor\tdonor unit sdk: x.c\tdonors/main/\t\tC\tsdk\n")
        table("0x10")
        summary = vr.verify_export(export, contexts=Contexts(self.world))
        self.assertTrue(summary["ok"], summary)
        self.assertEqual((summary["replayed"], summary["donor_function_rows"]), (1, 1))
        table("0x14")
        summary = vr.verify_export(export, contexts=Contexts(self.world))
        self.assertFalse(summary["ok"])
        self.assertTrue(any("catalogue boundary changed" in item for item in summary["rejected"][0]["failures"]))


if __name__ == "__main__":
    unittest.main()


class DonorProofTest(Base):
    """Donor units get the same contract: a receipt per unit, built in a private copy of its tree,
    replayed against the manifest row, with the origin of every asserted address recorded."""

    def donor_run(self, origins=None, donor="sdk_2_0L_libultra", flags=None) -> Path:
        run = self.tmp / "run"
        folder = run / "main" / "donors"
        tree = folder / "src" / donor
        (tree / "include").mkdir(parents=True)
        (tree / "src").mkdir()
        (tree / "include" / "defs.h").write_text(HEADER)
        (tree / "src" / "func.c").write_text(SOURCE)
        flags = flags if flags is not None else f"-O2 -Iinclude --section-base=.data:0x{DATA:08X}"
        (folder / f"proposed_{donor}_kmc_as.tsv").write_text(
            f"# kmc+as rows proposed by donor transplant from {donor}; SOURCE paths are relative to\n"
            f"# src/{donor}/\n# ROM_START\tSIZE\tSOURCE\tFLAGS\n0x{TEXT - BASE:X}\t0x10\tsrc/func.c\t{flags}\n")
        row = {"target_vram": f"0x{TEXT:08X}", "verdict": "exact-with-transferred-symbols"}
        if origins is not None:
            row["pin_origin"] = origins
        (folder / "report.json").write_text(json.dumps({"rows": [row]}))
        return run

    def contexts(self):
        world = self.world
        return mock.Mock(get=lambda image: world.ctx(base=None, project=world.project))

    def test_a_donor_unit_is_receipted_and_replays(self):
        from n64dk.commands import verifyreceipt as vr
        run = self.donor_run({"section:.data": "section-content"})
        summary = vr.issue_donor_receipts(run, contexts=self.contexts(), issued_at=STAMP)
        self.assertEqual((summary["issued"], summary["strict_proof"], summary["earned_by"]),
                         (1, 1, {"sdk-donor": 1}), summary)
        receipt = pr.load(pr.receipt_path(run / "receipts" / "donors", "main", "func", TEXT))
        self.assertEqual(receipt["source"]["name"], "src/func.c")
        self.assertEqual(receipt["build"]["pin_origins"], {"section:.data": "section-content"})
        self.assertTrue(receipt["claims"]["pins_independent"])
        results, _known = vr.verify_donors([("main", run / "main" / "donors")], contexts=self.contexts(),
                                           receipts_dir=run / "receipts")
        self.assertTrue(results[0][1].ok, results[0][1].failures)

    def test_an_unrecorded_or_self_derived_address_never_proves(self):
        from n64dk.commands import verifyreceipt as vr
        for origins in (None, {"section:.data": "self-derived"}):
            shutil.rmtree(self.tmp / "run", ignore_errors=True)
            run = self.donor_run(origins)
            summary = vr.issue_donor_receipts(run, contexts=self.contexts(), issued_at=STAMP)
            self.assertEqual((summary["issued"], summary["strict_proof"]), (1, 0), origins)
            receipt = pr.load(pr.receipt_path(run / "receipts" / "donors", "main", "func", TEXT))
            self.assertFalse(receipt["claims"]["pins_independent"])
            self.assertEqual(pr.check_static(receipt), [])

    def test_a_fabricated_pin_in_the_manifest_is_rejected(self):
        from n64dk.commands import verifyreceipt as vr
        run = self.donor_run({"section:.data": "section-content"})
        vr.issue_donor_receipts(run, contexts=self.contexts(), issued_at=STAMP)
        manifest = run / "main" / "donors" / "proposed_sdk_2_0L_libultra_kmc_as.tsv"
        manifest.write_text(manifest.read_text().replace("-O2", "-O2 --symbol=gValue:0x80001004"))
        results, _known = vr.verify_donors([("main", run / "main" / "donors")], contexts=self.contexts(),
                                           receipts_dir=run / "receipts")
        self.assertFalse(results[0][1].ok)

    def test_an_earning_class_must_fit_the_source(self):
        forged = reseal(copy.deepcopy(self.receipt) | {"earned_by": "assembly"})
        self.assertTrue(any("earning class" in item for item in pr.check_static(forged)))
        self.assertEqual(self.receipt["earned_by"], "generated-c")
        self.assertEqual(self.receipt["build"]["pin_origins"], {"section:.data": "run-declaration"})

    def test_a_donor_may_ship_assembly_but_generated_c_may_not(self):
        # libultra ships bcmp.s: an SDK donor's assembly unit is a legitimate receipt (found on the
        # Dr. Mario 64 replay, 2026-11-18, where every such unit was falsely rejected)
        donor_asm = copy.deepcopy(self.receipt)
        donor_asm["earned_by"], donor_asm["source"]["kind"] = "sdk-donor", "assembly"
        self.assertFalse(any("earning class" in item for item in pr._class_problems(donor_asm)))
        c_from_asm = copy.deepcopy(self.receipt)
        c_from_asm["source"]["kind"] = "assembly"                     # still earned_by generated-c
        self.assertTrue(any("earning class" in item for item in pr._class_problems(c_from_asm)))


class DonorOriginRecordTest(DonorProofTest):
    def test_a_laundered_origin_with_recomputed_claims_is_rejected(self):
        from n64dk.commands import verifyreceipt as vr
        run = self.donor_run({"section:.data": "self-derived"})
        vr.issue_donor_receipts(run, contexts=self.contexts(), issued_at=STAMP)
        path = pr.receipt_path(run / "receipts" / "donors", "main", "func", TEXT)
        forged = pr.load(path)
        forged["build"]["pin_origins"]["section:.data"] = "declared"
        forged["claims"], forged["proof"] = pr.derive(forged["verdict"]["class"], forged["verdict"]["strict"],
                                                      forged.get("behavioural"), forged["provenance"], forged["build"])
        self.assertTrue(forged["claims"]["strict_proof"])
        pr.write(reseal(forged), path)
        self.assertEqual(pr.check_static(forged), [])                    # consistent on its own ...
        results, _known = vr.verify_donors([("main", run / "main" / "donors")], contexts=self.contexts(),
                                           receipts_dir=run / "receipts")
        self.assertFalse(results[0][1].ok)                               # ... but not the run's record
        self.assertTrue(any("pin origin" in item for item in results[0][1].failures))
