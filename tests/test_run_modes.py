"""decompile-run --mode blind|research|source-tree (n64dk.runmodes): permitted inputs enforced at
execution time, with negative controls for self-donors, cross-mode reuse of a run directory and
title-specific metadata leaking into a blind run.  Temporary trees and a synthetic 64-byte header
only; no ROM, no toolchain, no kit run (the stages are mocked)."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from n64dk import governance as gov
from n64dk import runmodes
from n64dk.commands import run

TREE_SYMBOLS = ("uvTitleInit = 0x80001000; // type:func\n"
                "uvTitleObjects = 0x80002000; // type:data\n"
                "osCreateThread = 0x80003000; // type:func\n"
                "uvRomString = 0x80004000; // type:func\n")
TREE_C = "void uvTitleUpdate(s32 frame)\n{\n    uvTitleObjects[frame] = 0;\n}\n\nstatic void uvHidden(void);\n"
OTHER_C = "void otherGameMain(void)\n{\n    otherGameCounter = otherGameCounter + 1;\n}\n" * 2


def header(name: str, code: str) -> bytes:
    """A synthetic big-endian (z64) cartridge header: magic, internal name at 0x20, game code at 0x3B."""
    data = bytearray(64)
    data[0:4] = b"\x80\x37\x12\x40"
    data[0x20:0x34] = name.encode().ljust(20, b" ")
    data[0x3B:0x3F] = code.encode()
    return bytes(data)


class ModeFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-run-modes-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        environment = mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        general = mock.patch.object(runmodes, "_GENERAL", frozenset({"osCreateThread"}))   # the kit's SDK names
        general.start()
        self.addCleanup(general.stop)
        # the title's public tree, its research notes and another title's tree
        self.tree = self.root / "public_src" / "title"
        (self.tree / "src").mkdir(parents=True)
        (self.tree / "symbol_addrs.txt").write_text(TREE_SYMBOLS)
        (self.tree / "src" / "game.c").write_text(TREE_C * 3)
        self.notes = self.root / "notes" / "title"
        self.notes.mkdir(parents=True)
        (self.notes / "names.txt").write_text("uvTitleUpdate is the per-frame update\n")
        self.other = self.root / "public_src" / "other"
        (self.other / "src").mkdir(parents=True)
        (self.other / "src" / "other.c").write_text(OTHER_C)
        self.manifest = self.root / "corpus.json"
        gov.Manifest([
            gov.CorpusEntry("title-tree", "public-source", "https://example.invalid/title", licence="CC0",
                            titles=["Test Title"], roots=[str(self.tree)]),
            gov.CorpusEntry("title-notes", "redistributable-metadata", "https://example.invalid/notes",
                            titles=["NTTE"], roots=[str(self.notes)], blind_exclude=False),
            gov.CorpusEntry("other-tree", "public-source", "https://example.invalid/other", licence="CC0",
                            titles=["Other Game"], roots=[str(self.other)]),
        ]).save(self.manifest)
        # the project: a ROM-derived symbol list (placeholders, an SDK name, a name the ROM spells)
        self.rom = header("TEST TITLE", "NTTE") + b"uvRomString\x00" + bytes(range(256)) * 4
        self.project = self.root / "proj"
        self.project.mkdir()
        (self.project / "baserom.z64").write_bytes(self.rom)
        (self.project / "n64dk.toml").write_text('[images.main]\npath = "baserom.z64"\n\n'
                                                 '[run]\nsymbols = ["symbols.txt"]\n')
        (self.project / "symbols.txt").write_text("func_80001000 = 0x80001000;\nosCreateThread = 0x80003000;\n"
                                                  "uvRomString = 0x80004000;\n")
        self.out = self.root / "out"
        self.parser = argparse.ArgumentParser()
        run.decompile_run.arguments(self.parser)

    def args(self, mode, *extra, out=None):
        return self.parser.parse_args(["--project", str(self.project), "--out", str(out or self.out),
                                       "--mode", mode, "--corpus-manifest", str(self.manifest), *extra])

    def rom_file(self, folder: str, data: bytes) -> Path:
        path = self.root / folder / "rom.z64"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def donors_file(self, **donors) -> Path:
        lines = []
        for name, image in donors.items():
            lines += [f"[donor.{name}]", f"root = {json.dumps(str(image.parent))}", f"image = {json.dumps(str(image))}",
                      "manifests = []", ""]
        path = self.root / "donors.toml"
        path.write_text("\n".join(lines))
        return path

    def solved(self, args):
        """What a run leaves behind: a case store with a solved function in it."""
        out = Path(args.out)
        (out / "state").mkdir(parents=True, exist_ok=True)
        (out / "state" / "cases.sqlite").write_bytes(b"solved")
        return {"images": {}}

    def refused(self, args):
        with mock.patch.object(run, "_decompile_run") as body, self.assertRaises(SystemExit) as stop:
            run._mode_run(args)
        body.assert_not_called()
        return stop.exception.code

    def sessions(self, out=None):
        return json.loads(gov.provenance_path(out or self.out).read_text())["sessions"]


class TreeSelectionTest(ModeFixture):
    def test_each_mode_takes_the_titles_trees_and_only_those(self):
        blind = runmodes.build_plan(self.args("blind"))
        self.assertEqual([row["label"] for row in blind.forbidden], ["title-tree", "title-notes"])
        research = runmodes.build_plan(self.args("research"))
        self.assertEqual([row["label"] for row in research.forbidden], ["title-tree"])   # notes are research
        source = runmodes.build_plan(self.args("source-tree"))
        self.assertEqual(source.forbidden, [])
        self.assertEqual([row["label"] for row in source.recorded], ["title-tree", "title-notes"])
        self.assertTrue(all(row["tree_digest"].startswith("sha256:") for row in source.recorded))
        self.assertEqual(blind.title.rom.key, hashlib.sha1(self.rom).hexdigest()[:16])   # the library's key

    def test_configurations_that_cannot_be_a_run_of_their_mode(self):
        bare = self.parser.parse_args(["--project", str(self.project), "--out", str(self.out), "--mode", "blind"])
        with self.assertRaises(SystemExit) as stop:                  # blind against nothing
            runmodes.build_plan(bare)
        self.assertIn("no tree to forbid", str(stop.exception.code))
        with self.assertRaises(SystemExit):                          # a source-tree run is not blind
            runmodes.build_plan(self.args("source-tree", "--blind-forbid", str(self.other)))
        with self.assertRaises(SystemExit):                          # a model's memory is not blind
            runmodes.build_plan(self.args("blind", "--refine-command", "author"))
        self.assertIsNone(runmodes.build_plan(self.args("research", "--refine-command", "author")).refine)


class SelfDonorTest(ModeFixture):
    def test_a_self_donor_is_refused_in_blind_mode(self):
        copy = self.rom_file("selfdonor", self.rom)
        donors = self.donors_file(selfcopy=copy)
        self.assertEqual(self.refused(self.args("blind", "--donors", str(donors))), 3)
        receipt = json.loads((self.out / "solver" / "run_mode_violation.json").read_text())
        self.assertEqual(receipt["status"], "refused")
        self.assertEqual([(d["donor"], d["relation"], d["admitted"]) for d in receipt["donors"]],
                         [("selfcopy", "self", False)])
        self.assertEqual([(s["run_mode"], s["status"]) for s in self.sessions()], [("blind", "refused")])
        # the refused attempt left no state: the same directory serves the corrected run
        with mock.patch.object(run, "_decompile_run", side_effect=self.solved) as body:
            result = run._mode_run(self.args("blind", "--donors", str(donors), "--exclude-donor", "selfcopy"))
        body.assert_called_once()
        self.assertEqual(result["run_mode"]["mode"], "blind")

    def test_another_revision_is_refused_and_left_out_of_the_library(self):
        sibling = self.rom_file("japan", header("TEST TITLE", "NTTJ") + b"another revision" * 8)
        other = self.rom_file("othergame", header("OTHER GAME", "NOGE") + b"another title" * 8)
        segment = sibling.parent / "seg.bin"                         # another image of that run: no header
        segment.write_bytes(bytes(range(64)) * 2)
        self.assertEqual(self.refused(self.args("blind", "--donors", str(self.donors_file(jp=sibling, jpseg=segment)))), 3)
        receipt = json.loads((self.out / "solver" / "run_mode_violation.json").read_text())
        self.assertEqual([(d["donor"], d["relation"], d["admitted"]) for d in receipt["donors"]],
                         [("jp", "revision", False), ("jpseg", "revision", False)])
        self.assertIn("shares its root", receipt["donors"][1]["why"])
        library = self.root / "library"
        library.mkdir()
        for image in (sibling, other):
            key = hashlib.sha1(image.read_bytes()).hexdigest()[:16]
            (library / f"{key}.toml").write_text(f'[donor.t{key}_main]\nroot = {json.dumps(str(image.parent))}\n'
                                                 f'image = {json.dumps(str(image))}\nmanifests = []\n')
        plan = runmodes.build_plan(self.args("blind", "--library", str(library)))
        self.assertEqual(plan.problems, [])                          # a library entry is left out, not fatal
        sibling_entry = hashlib.sha1(sibling.read_bytes()).hexdigest()[:16] + ".toml"
        self.assertEqual(plan.library_excluded, [sibling_entry])
        self.assertEqual([spec.image for spec in plan.library_donors(library, plan.title.rom.key)], [other])
        rows = {row["origin"]: row for row in plan.donors}
        self.assertIn("internal name", rows[f"library:{sibling_entry}"]["why"])
        # an entry added after the run started was never classified: not used
        late = library / "0123456789abcdef.toml"
        late.write_text((library / sibling_entry).read_text())
        self.assertEqual(len(plan.library_donors(library, plan.title.rom.key)), 1)
        self.assertNotIn("0123456789abcdef.toml", [r[0] for r in plan.library_identity(
            [(p.name, 0, 0) for p in library.glob("*.toml")])])

    def test_research_admits_a_revision_only_from_a_blind_or_research_run(self):
        jprun = self.root / "jprun"               # a research run of the other revision, recorded before its state
        gov.record_session(jprun, blind=True, forbidden=[("t", self.tree)], status="clean", run_mode="research")
        image = self.rom_file("jprun/queue", header("TEST TITLE", "NTTJ") + b"jp" * 64)
        donors = self.donors_file(jp=image)
        plan = runmodes.build_plan(self.args("research", "--donors", str(donors)))
        self.assertEqual([(d["relation"], d["admitted"], d["provenance"]) for d in plan.donors],
                         [("revision", True, "blind/research only")])
        gov.record_session(jprun, blind=False, status="not blind", run_mode="source-tree")
        plan = runmodes.build_plan(self.args("research", "--donors", str(donors)))
        self.assertFalse(plan.donors[0]["admitted"])
        self.assertTrue(plan.problems)
        self.assertFalse(runmodes.build_plan(self.args("blind", "--donors", str(donors))).donors[0]["admitted"])
        self.assertTrue(runmodes.build_plan(self.args("source-tree", "--donors", str(donors))).donors[0]["admitted"])

    def test_the_donor_stage_refuses_a_donor_the_plan_did_not_admit(self):
        plan = runmodes.build_plan(self.args("blind"))
        from n64dk.match import transplant
        copy = self.rom_file("late", self.rom)
        spec = transplant.load_donors(self.donors_file(late=copy))[0]
        with self.assertRaises(RuntimeError):
            plan.check_explicit([spec])
        plan.check_explicit([])


class CrossModeReuseTest(ModeFixture):
    def test_a_research_run_cannot_reuse_a_source_tree_runs_directory(self):
        with mock.patch.object(run, "_decompile_run", side_effect=self.solved):
            result = run._mode_run(self.args("source-tree"))
        self.assertEqual(result["run_mode"]["recorded_trees"], ["title-tree", "title-notes"])
        session = self.sessions()[0]
        self.assertEqual((session["run_mode"], session["blind"]), ("source-tree", False))
        self.assertEqual([t["label"] for t in session["trees"]], ["title-tree", "title-notes"])
        self.assertEqual(self.refused(self.args("research")), 3)
        receipt = json.loads((self.out / "solver" / "run_mode_violation.json").read_text())
        self.assertIn("session 1 ran as source-tree", receipt["problems"][0])
        self.assertEqual(self.refused(self.args("blind")), 3)
        with mock.patch.object(run, "_decompile_run", side_effect=self.solved) as body:
            run._mode_run(self.args("source-tree"))                  # the same mode continues
        body.assert_called_once()

    def test_blind_after_research_is_refused_and_research_after_blind_continues(self):
        with mock.patch.object(run, "_decompile_run", side_effect=self.solved):
            run._mode_run(self.args("blind"))
            run._mode_run(self.args("research"))
        self.assertEqual([(s["run_mode"], s["status"]) for s in self.sessions()],
                         [("blind", "clean"), ("research", "clean")])
        self.assertEqual(self.refused(self.args("blind")), 3)

    def test_state_without_a_mode_is_reused_by_no_mode(self):
        plain = self.parser.parse_args(["--project", str(self.project), "--out", str(self.out)])
        with mock.patch.object(run, "_decompile_run", side_effect=self.solved):
            run._unguarded_run(plain)
        for mode in runmodes.MODES:
            self.assertEqual(self.refused(self.args(mode)), 3)
        unrecorded = self.root / "unrecorded"
        (unrecorded / "state").mkdir(parents=True)
        self.assertEqual(gov.mode_reuse(unrecorded, "source-tree")["verdict"], "unknown provenance")

    def test_session_ranks(self):
        out = self.root / "ranks"
        gov.record_session(out, blind=True, forbidden=[("t", self.tree)], status="clean")   # --blind-forbid alone
        (out / "state").mkdir()
        self.assertEqual(gov.mode_reuse(out, "source-tree")["modes"], ["blind (no mode)"])
        self.assertFalse(gov.mode_reuse(out, "source-tree")["ok"])
        fresh = self.root / "fresh"
        gov.record_session(fresh, blind=True, forbidden=[("t", self.tree)], status="refused", run_mode="blind")
        self.assertEqual(gov.mode_reuse(fresh, "blind")["verdict"], "fresh")
        self.assertTrue(gov.run_provenance(fresh, [("t", self.tree)])["ok"])
        gov.record_session(fresh, blind=True, forbidden=[("t", self.tree)], status="violated", run_mode="blind")
        self.assertEqual(gov.mode_reuse(fresh, "source-tree")["modes"], ["violated"])


class MetadataLeakTest(ModeFixture):
    def test_a_symbol_file_copied_from_the_titles_tree_is_refused_in_blind_mode(self):
        shutil.copyfile(self.tree / "symbol_addrs.txt", self.project / "symbols.txt")
        self.assertEqual(self.refused(self.args("blind")), 3)
        receipt = json.loads((self.out / "solver" / "run_mode_violation.json").read_text())
        leaked = [row["name"] for row in receipt["names"]["leaked"]]
        self.assertEqual(leaked, ["uvTitleInit", "uvTitleObjects"])  # not the SDK's name, not the ROM's
        # an edited copy under another name is no byte copy, and is refused just the same
        (self.project / "symbols.txt").write_text("func_80001000 = 0x80001000;\n")
        (self.project / "survey").mkdir()
        (self.project / "survey" / "renamed.txt").write_text("uvTitleUpdate = 0x80001100; // edited\n")
        self.assertEqual(self.refused(self.args("blind")), 3)
        receipt = json.loads((self.out / "solver" / "run_mode_violation.json").read_text())
        self.assertEqual([row["name"] for row in receipt["names"]["leaked"]], ["uvTitleUpdate"])
        self.assertEqual(receipt["names"]["leaked"][0]["defined_in"], "title-tree/src/game.c")

    def test_research_refuses_the_byte_copy_through_the_isolation_guard(self):
        shutil.copyfile(self.tree / "symbol_addrs.txt", self.project / "symbols.txt")
        self.assertEqual(self.refused(self.args("research")), 3)
        receipt = json.loads((self.out / "solver" / "blind_violation.json").read_text())
        self.assertIn("symbol list", receipt["violations"][0]["verdict"])
        self.assertEqual(receipt["run_mode"], "research")
        self.assertEqual(self.sessions()[-1]["status"], "refused")

    def test_a_clean_blind_run_records_its_mode_everywhere(self):
        args = self.args("blind")
        with mock.patch.object(run, "_decompile_run", return_value={"images": {}}) as body:
            result = run._mode_run(args)
        body.assert_called_once()
        self.assertEqual(result["run_mode"]["mode"], "blind")
        self.assertEqual(result["run_mode"]["names_leaked"], 0)
        self.assertTrue(result["blind"]["ok"])
        self.assertIn(str(self.tree.resolve()), args.blind_forbid)
        self.assertNotIn(str(self.other.resolve()), args.blind_forbid)
        for name in ("blind_entry.json", "blind_exit.json"):
            self.assertEqual(json.loads((self.out / "solver" / name).read_text())["run_mode"], "blind")
        self.assertEqual([(s["run_mode"], s["status"]) for s in self.sessions()], [("blind", "clean")])
        receipt = json.loads((self.out / "solver" / "run_mode.json").read_text())
        self.assertEqual(receipt["status"], "complete")
        body_only = {k: v for k, v in receipt.items() if k != "digest"}
        self.assertEqual(hashlib.sha256(gov.canonical(body_only).encode()).hexdigest(), receipt["digest"])
        run.save_invocation(args, self.out)
        saved = json.loads((self.out / "solver" / "invocation.json").read_text())
        self.assertEqual((saved["run_mode"], saved["corpus_manifest"]), ("blind", str(self.manifest.resolve())))
        self.assertNotIn("_run_mode_plan", saved)

    def test_resume_cannot_change_the_mode(self):
        start = self.args("blind")
        self.out.mkdir()
        run.prepare_invocation(start)
        run.save_invocation(start, self.out)
        with self.assertRaises(SystemExit):
            run.prepare_invocation(self.parser.parse_args(["--resume", str(self.out), "--mode", "research"]))
        resumed = self.parser.parse_args(["--resume", str(self.out)])
        run.prepare_invocation(resumed)
        self.assertEqual(resumed.run_mode, "blind")
        plain = self.root / "plain"
        plain.mkdir()
        unmoded = self.parser.parse_args(["--project", str(self.project), "--out", str(plain)])
        run.prepare_invocation(unmoded)
        run.save_invocation(unmoded, plain)
        with self.assertRaises(SystemExit):
            run.prepare_invocation(self.parser.parse_args(["--resume", str(plain), "--mode", "blind"]))


if __name__ == "__main__":
    unittest.main()
