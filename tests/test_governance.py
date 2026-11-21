"""Corpus governance and blind isolation (``n64dk.governance``): data classes, the manifest, permitted
uses, and an isolation check that catches every way a forbidden tree can be reached from a run's
configuration.  Temporary trees only; no ROM, no toolchain."""
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from n64dk import governance as gov

TRUTH_C = "void func_80001000(void) {\n    gSomeGlobalCounter = gSomeGlobalCounter + 1;\n    return;\n}\n"
SDK_C = ("#include \"osint.h\"\n\nvoid osCreateThread(OSThread *t, OSId id, void (*entry)(void *), void *arg) {\n"
         "    t->id = id;\n    t->context.pc = (u32)entry;\n}\n")


class GovernanceTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-governance-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.truth = self.root / "public_src" / "title"
        (self.truth / "src").mkdir(parents=True)
        (self.truth / "src" / "game.c").write_text(TRUTH_C)
        (self.truth / "include").mkdir()
        (self.truth / "include" / "ultra64.h").write_text("/* shared SDK header */\n" + "#define X 1\n" * 8)
        self.project = self.root / "work" / "proj"
        self.project.mkdir(parents=True)
        (self.project / "n64dk.toml").write_text('[run]\nsymbols = ["symbols.txt"]\n')
        (self.project / "symbols.txt").write_text("func_80001000 = 0x80001000;\n")

    def manifest(self, **overrides) -> gov.Manifest:
        entry = {"name": "title-decomp", "data_class": "public-source", "origin": "a public repository",
                 "licence": "MIT", "date": "2026-01-01", "uses": ["analysis", "donor", "training"],
                 "titles": ["title"], "roots": [str(self.truth)], **overrides}
        path = self.root / "corpus.json"
        path.write_text(json.dumps({"schema": gov.SCHEMA, "entries": [entry]}))
        return gov.Manifest.load(path)

    def check(self, config=None, env=None, **kw):
        return gov.check_isolation(self.manifest().forbidden_for("title"), config=config or {},
                                   project=self.project, env=env or {}, base=self.root, **kw)

    def test_data_classes_and_permitted_uses(self):
        self.assertEqual(len(gov.DATA_CLASSES), 10)
        self.assertFalse(gov.DATA_CLASSES["user-rom-derived"].redistributable)
        self.assertFalse(gov.DATA_CLASSES["user-rom-derived"].export)
        entry = self.manifest().entries[0]
        self.assertTrue(gov.permitted(entry, "training")[0])
        allowed, why = gov.permitted(entry, "export")
        self.assertFalse(allowed)
        self.assertIn("not among its permitted uses", why)
        strict = self.manifest(training=False).entries[0]
        self.assertFalse(gov.permitted(strict, "training")[0])
        with self.assertRaises(gov.GovernanceError):
            gov.permitted(entry, "selling")

    def test_manifest_validation_refuses_looser_flags_and_unknown_fields(self):
        for bad in ({"data_class": "user-rom-derived", "export": True}, {"data_class": "sdk-source",
                                                                          "uses": ["redistribution"]},
                    {"licence": ""}, {"data_class": "nonsense"}, {"uses": ["selling"]}, {"digest": "md5:00"},
                    {"origin": ""}):
            with self.subTest(bad=bad), self.assertRaises(gov.GovernanceError):
                self.manifest(**bad)
        path = self.root / "extra.json"
        path.write_text(json.dumps({"schema": gov.SCHEMA, "entries": [{"name": "x", "data_class": "synthetic",
                                                                       "origin": "test", "secret_flag": 1}]}))
        with self.assertRaises(gov.GovernanceError):
            gov.Manifest.load(path)
        path.write_text(json.dumps({"schema": "n64dk.corpus/9", "entries": []}))
        with self.assertRaises(gov.GovernanceError):
            gov.Manifest.load(path)

    def test_tree_digest_and_manifest_digests(self):
        digest, files = gov.tree_digest(self.truth)
        self.assertEqual(files, 2)
        self.assertEqual(gov.tree_digest(self.truth)[0], digest)
        manifest = self.manifest(digest=digest)
        self.assertEqual(manifest.verify_digests()[0]["status"], "current")
        (self.truth / "src" / "game.c").write_text(TRUTH_C + "\n")
        self.assertEqual(manifest.verify_digests()[0]["status"], "changed")
        self.assertEqual(self.manifest(blind_exclude=False).forbidden_for("title"), [])
        self.assertEqual(self.manifest().forbidden_for("another title"), [])

    def test_a_clean_configuration_passes_with_a_sealed_receipt(self):
        receipt = self.check({"project": str(self.project), "toolchain_dir": str(self.root / "tools"),
                              "pipeline": "ido5.3", "flags": "-O2 -mips2"})
        self.assertTrue(receipt["ok"], receipt["violations"])
        self.assertEqual(receipt["schema"], gov.RECEIPT_SCHEMA)
        self.assertEqual(gov.verify_receipt(receipt), [])
        self.assertEqual(receipt["forbidden"][0]["files"], 2)
        keys = [row["key"] for row in receipt["checked"]]
        self.assertIn("project", keys)
        self.assertIn("project/n64dk.toml>run.symbols[0]", keys)
        self.assertNotIn("config.pipeline", keys)                      # not a path
        tampered = dict(receipt, ok=False)
        self.assertTrue(gov.verify_receipt(tampered))

    def test_every_route_to_the_forbidden_tree_is_caught(self):
        library = self.root / "library"
        library.mkdir()
        (library / "other.toml").write_text(f'[donor.t1]\nroot = "{self.truth / "src"}"\n')
        donors = self.root / "donors.toml"
        donors.write_text(f'[donor.a]\nimage = "{self.truth / "game.z64"}"\n')
        link_dir = self.root / "links"
        link_dir.mkdir()
        os.symlink(self.truth / "src", link_dir / "src")
        copies = self.root / "copies"
        copies.mkdir()
        (copies / "game.c").write_text(TRUTH_C)
        (copies / "ultra64.h").write_text((self.truth / "include" / "ultra64.h").read_text())
        cases = {
            "inside": ({"symbols": [str(self.truth / "symbols.txt")]}, {}),
            "contains": ({"library": str(self.root / "public_src")}, {}),
            "library entry": ({"library": str(library)}, {}),
            "donors file": ({"donors": str(donors)}, {}),
            "symlink": ({"context": str(link_dir)}, {}),
            "copy": ({"headers": str(copies)}, {}),
            "environment": ({}, {"N64DK_TRUTH_TITLE": str(self.truth)}),
            "environment pair": ({}, {"N64DK_ROMS": f"title={self.truth / 'x.z64'}"}),
        }
        for label, (config, env) in cases.items():
            with self.subTest(route=label):
                receipt = self.check(config, env)
                self.assertFalse(receipt["ok"])
                self.assertEqual(receipt["violations"][0]["forbidden"].split("/")[0], "title-decomp")
        copied = self.check({"headers": str(copies)})
        self.assertIn("copy of forbidden source", [v["verdict"] for v in copied["violations"]])
        self.assertEqual(len(copied["violations"]), 1)                   # the shared header is listed, not failed
        self.assertEqual(len(copied["shared_files"]), 1)
        self.assertTrue(self.check({"headers": str(copies)}, content=False)["ok"])

    def test_a_scratch_directory_is_checked_but_never_walked(self):
        scratch = self.root / "scratch"
        (scratch / "other_project").mkdir(parents=True)
        # another project's file in a shared temporary folder: its contents and the paths it names
        # must not be read, walked or recorded
        (scratch / "other_project" / "notes.json").write_text(json.dumps({"rom": str(self.root / "private.z64")}))
        (scratch / "stray.json").write_text(json.dumps({"tree": str(self.root / "elsewhere")}))
        (scratch / "game.c").write_text(TRUTH_C)                       # even a copy is not an input there
        receipt = self.check({"pipelines": {"ido": {"env": {"TMPDIR": str(scratch)}}}})
        self.assertTrue(receipt["ok"], receipt["violations"])
        text = json.dumps(receipt)
        self.assertNotIn("private.z64", text)
        self.assertNotIn("elsewhere", text)
        self.assertNotIn("other_project", text)
        [row] = [row for row in receipt["checked"] if row["key"].endswith("TMPDIR")]
        self.assertEqual(row["verdict"], "outside (scratch: checked, not walked)")
        inside = self.check({"pipelines": {"ido": {"env": {"TMPDIR": str(self.truth / "src")}}}})
        self.assertFalse(inside["ok"])                                  # a scratch dir in the tree still fails

    def test_the_exclusion_list_itself_is_exempt(self):
        receipt = self.check({"refine_known_source": [str(self.truth)]})
        self.assertTrue(receipt["ok"])
        self.assertIn("exempt", receipt["checked"][0]["verdict"])

    def test_a_blind_runs_own_forbidden_list_is_exempt(self):
        # decompile-run --blind-forbid saves its list in solver/invocation.json, which is what
        # blind-check --invocation reads: the list must not count as a route to the trees
        receipt = self.check({"blind_forbid": [str(self.truth)], "effort": "normal"})
        self.assertTrue(receipt["ok"], receipt["violations"])
        [row] = [row for row in receipt["checked"] if row["key"].startswith("config.blind_forbid")]
        self.assertIn("exempt", row["verdict"])
        self.assertFalse(self.check({"donors": str(self.truth), "blind_forbid": [str(self.truth)]})["ok"])

    def test_blind_guard_refuses_before_the_run_and_checks_again_after(self):
        forbidden = self.manifest().forbidden_for("title")
        with self.assertRaises(gov.BlindViolation) as caught:
            with gov.blind_guard(forbidden, config={"donors": str(self.truth)}, env={}, project=self.project):
                self.fail("the run must not start")
        self.assertFalse(caught.exception.receipt["ok"])
        env = {}
        with self.assertRaises(gov.BlindViolation):
            with gov.blind_guard(forbidden, config={}, env=env, project=self.project) as guard:
                self.assertTrue(guard.receipt["ok"])
                env["N64DK_TRUTH_TITLE"] = str(self.truth)              # reached during the run
        with gov.blind_guard(forbidden, config={}, env={}, project=self.project) as guard:
            pass
        self.assertTrue(guard.exit_receipt["ok"])

    # ------------------------------------------------------------------ shared libraries (opt-in)

    def sdk(self):
        """A public library vendored inside the forbidden tree, the run's own clone of it, and a
        donor directory holding a copy of the SDK source, the shared header and the title's own source."""
        vendored = self.truth / "lib" / "ultralib" / "src" / "os"
        vendored.mkdir(parents=True)
        (vendored / "thread.c").write_text(SDK_C)
        clone = self.root / "sdk_src" / "ultralib"
        (clone / "src" / "os").mkdir(parents=True)
        (clone / "src" / "os" / "thread.c").write_text(SDK_C)
        (clone / "include").mkdir()
        (clone / "include" / "ultra64.h").write_text((self.truth / "include" / "ultra64.h").read_text())
        (clone / ".git").mkdir()
        (clone / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        donor = self.root / "donor"
        donor.mkdir()
        (donor / "thread.c").write_text(SDK_C)
        return clone, donor

    def pinned(self, clone, data=None, name="ultralib.json"):
        data = data or gov.pin_shared_library(clone, name="ultralib", provenance="https://example.org/ultralib",
                                              revision="0123abc")
        raw = gov.shared_manifest_bytes(data)
        path = self.root / name
        path.write_bytes(raw)
        return path, "sha256:" + hashlib.sha256(raw).hexdigest()

    def test_without_a_shared_library_the_receipt_is_unchanged(self):
        _clone, donor = self.sdk()
        plain = self.check({"donor_root": str(donor)})
        empty = self.check({"donor_root": str(donor)}, shared=[])
        self.assertEqual(plain, empty)                                    # same receipt, same seal
        self.assertFalse(plain["ok"])
        self.assertEqual([v["forbidden"] for v in plain["violations"]], ["title-decomp/lib/ultralib/src/os/thread.c"])
        self.assertNotIn("shared_libraries", plain)
        self.assertNotIn("exemptions", plain)
        self.assertEqual(len(plain["limits"]), 4)
        self.assertNotIn("sources", plain["forbidden"][0])

    def test_a_pinned_shared_library_exempts_its_copies_and_lists_them(self):
        clone, donor = self.sdk()
        (donor / "ultra64.h").write_text((self.truth / "include" / "ultra64.h").read_text())
        path, pin = self.pinned(clone)
        library = gov.SharedLibrary.load(path, pin)
        self.assertEqual(sorted(library.files), ["include/ultra64.h", "src/os/thread.c"])   # .git skipped
        receipt = self.check({"donor_root": str(donor)}, shared=[library])
        self.assertTrue(receipt["ok"], receipt["violations"])
        self.assertEqual(gov.verify_receipt(receipt), [])
        self.assertEqual(receipt["shared_libraries"], [{
            "name": "ultralib", "provenance": "https://example.org/ultralib", "revision": "0123abc", "licence": "",
            "form": "files", "files": 2, "manifest_digest": pin}])
        used = {row["key"].rsplit("/", 1)[-1]: row for row in receipt["exemptions"]}
        self.assertEqual(sorted(used), ["thread.c", "ultra64.h"])
        self.assertEqual(used["thread.c"]["would_be"], "copy of forbidden source")
        self.assertEqual(used["thread.c"]["same_as"], "title-decomp/lib/ultralib/src/os/thread.c")
        self.assertEqual(used["thread.c"]["library"], "ultralib")
        self.assertEqual(used["ultra64.h"]["would_be"], "listed")
        self.assertEqual(receipt["shared_files"], [])
        self.assertEqual(receipt["forbidden"][0]["sources"], 2)
        self.assertEqual(receipt["forbidden"][0]["sources_pinned_by_shared_libraries"], 1)
        self.assertIn("shared_library[0]", [row["key"] for row in receipt["checked"]])
        self.assertEqual(len(receipt["limits"]), 5)

    def test_a_copy_the_manifest_does_not_list_still_fails(self):
        clone, donor = self.sdk()
        (donor / "game.c").write_text(TRUTH_C)                           # the title's own source
        library = gov.SharedLibrary.load(*self.pinned(clone))
        receipt = self.check({"donor_root": str(donor)}, shared=[library])
        self.assertFalse(receipt["ok"])
        self.assertEqual([v["forbidden"] for v in receipt["violations"]], ["title-decomp/src/game.c"])
        self.assertEqual(len(receipt["exemptions"]), 1)
        # naming the title's file in the manifest does nothing: only a digest it lists exempts
        listed = {"src/game.c": hashlib.sha256(b"something else").hexdigest(),
                  "game.c": hashlib.sha256(b"another").hexdigest(),
                  "src/os/thread.c": hashlib.sha256(SDK_C.encode()).hexdigest()}
        by_name = gov.SharedLibrary.load(*self.pinned(clone, {"schema": gov.SHARED_SCHEMA, "name": "ultralib",
                                                              "provenance": "https://example.org/u", "files": listed},
                                                      "named.json"))
        receipt = self.check({"donor_root": str(donor)}, shared=[by_name])
        self.assertEqual([v["forbidden"] for v in receipt["violations"]], ["title-decomp/src/game.c"])
        # a path inside the forbidden tree is never exempt, even to a pinned file
        receipt = self.check({"sdk": str(self.truth / "lib" / "ultralib" / "src" / "os" / "thread.c")},
                             shared=[library])
        self.assertFalse(receipt["ok"])
        self.assertEqual(receipt["violations"][0]["verdict"], "inside forbidden")

    def test_unpinned_mismatching_or_invalid_manifests_are_refused(self):
        clone, _donor = self.sdk()
        path, pin = self.pinned(clone)
        self.assertEqual(gov.parse_pin(f"{path}@{pin}"), (str(path), pin))
        self.assertEqual(gov.parse_pin(str(path)), (str(path), None))
        for bad_pin in (None, "", "sha256:" + "0" * 64, "md5:" + pin[7:], pin[:-1]):
            with self.subTest(pin=bad_pin), self.assertRaises(gov.GovernanceError):
                gov.SharedLibrary.load(path, bad_pin)
        path.write_bytes(path.read_bytes().replace(b"0123abc", b"0123abd"))   # edited after review
        with self.assertRaises(gov.GovernanceError):
            gov.SharedLibrary.load(path, pin)
        good = {"schema": gov.SHARED_SCHEMA, "name": "lib", "provenance": "https://example.org/lib",
                "files": {"a.c": "a" * 64}}
        for bad in ({"schema": "n64dk.shared-library/9"}, {"name": ""}, {"provenance": ""},
                    {"provenance": "lib from somewhere"}, {"files": {}}, {"files": {"/abs/a.c": "a" * 64}},
                    {"files": {"../a.c": "a" * 64}}, {"files": {"a.c": "A" * 64}}, {"files": {"a.c": "a" * 63}},
                    {"extra": 1}, {"tree_digest": "sha256:" + "a" * 64}):
            with self.subTest(bad=bad):
                self.assertTrue(gov.shared_problems({**good, **bad}))
                bad_path, bad_digest = self.pinned(clone, {**good, **bad}, "bad.json")
                with self.assertRaises(gov.GovernanceError):
                    gov.SharedLibrary.load(bad_path, bad_digest)
        self.assertEqual(gov.shared_problems(good), [])
        with self.assertRaises(gov.GovernanceError):
            self.check({}, shared=[{"files": {"a.c": "a" * 64}}])              # not loaded under a pin

    def test_tree_form_is_hashed_now_and_may_not_live_in_a_forbidden_tree(self):
        clone, donor = self.sdk()
        digest, _files = gov.tree_digest(clone)
        tree = {"schema": gov.SHARED_SCHEMA, "name": "ultralib", "provenance": "https://example.org/ultralib",
                "root": "sdk_src/ultralib", "tree_digest": digest}
        library = gov.SharedLibrary.load(*self.pinned(clone, tree, "tree.json"))
        self.assertEqual(library.form, "tree")
        receipt = self.check({"donor_root": str(donor)}, shared=[library])
        self.assertTrue(receipt["ok"], receipt["violations"])
        self.assertEqual(receipt["shared_libraries"][0]["tree_digest"], digest)
        (clone / "src" / "os" / "thread.c").write_text(SDK_C + "/* changed */\n")
        with self.assertRaises(gov.GovernanceError):
            gov.SharedLibrary.load(*self.pinned(clone, tree, "tree.json"))
        inside = self.truth / "lib" / "ultralib"
        tree_inside = dict(tree, root=str(inside), tree_digest=gov.tree_digest(inside)[0])
        library = gov.SharedLibrary.load(*self.pinned(clone, tree_inside, "inside.json"))
        receipt = self.check({"donor_root": str(donor)}, shared=[library])
        self.assertFalse(receipt["ok"])
        self.assertEqual([v["key"] for v in receipt["violations"]], ["shared_library[0].root"])

    def test_blind_check_command_with_and_without_a_pinned_manifest(self):
        import argparse
        import contextlib
        import io
        from unittest import mock
        from n64dk.commands import governance as commands

        clone, donor = self.sdk()
        parser = argparse.ArgumentParser()
        commands._pin_args(parser)
        manifest = self.root / "pinned.json"
        pin_args = parser.parse_args([str(clone), "--name", "ultralib", "--provenance", "https://example.org/u",
                                      "--out", str(manifest)])
        made = commands.shared_library_pin(pin_args)
        self.assertEqual(made["files"], 2)
        self.assertEqual(made["digest"], "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest())
        invocation = self.root / "invocation.json"
        invocation.write_text(json.dumps({"donor_root": str(donor)}))
        parser = argparse.ArgumentParser()
        commands._blind_args(parser)
        base = ["--forbid", str(self.truth), "--invocation", str(invocation), "--project", str(self.project)]
        clean = {k: v for k, v in os.environ.items() if not k.startswith("N64DK_")}

        def run(*extra):
            args = parser.parse_args(base + list(extra))
            args.format = "text"
            with mock.patch.dict(os.environ, clean, clear=True), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                return commands.blind_check(args)

        self.assertEqual(run(), 3)                                             # default: refused as before
        self.assertEqual(run("--shared-library", made["pin"]), 0)
        self.assertEqual(run("--shared-library", str(manifest)), 2)            # unpinned
        self.assertEqual(run("--shared-library", f"{manifest}@sha256:{'0' * 64}"), 2)
        out = self.root / "receipt.json"
        self.assertEqual(run("--shared-library", made["pin"], "--out", str(out)), 0)
        receipt = json.loads(out.read_text())
        self.assertEqual(receipt["shared_libraries"][0]["manifest_digest"], made["digest"])
        self.assertEqual([row["same_as"] for row in receipt["exemptions"]],
                         ["forbid:title/lib/ultralib/src/os/thread.c"])
        self.assertEqual(gov.verify_receipt(receipt), [])


if __name__ == "__main__":
    unittest.main()
