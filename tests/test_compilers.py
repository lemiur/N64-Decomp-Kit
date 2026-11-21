import tempfile
import unittest
from pathlib import Path

from n64dk.match import compilers


class CatalogueTest(unittest.TestCase):
    def test_identifiers_are_unique_and_every_family_is_present(self):
        idents = [c.ident for c in compilers.CATALOGUE]
        self.assertEqual(len(idents), len(set(idents)))
        self.assertEqual({c.family for c in compilers.CATALOGUE}, {"ido", "gcc", "sn", "egcs"})

    def test_an_empty_root_reports_everything_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = compilers.status(Path(tmp))
        self.assertEqual(report["installed"], 0)
        self.assertEqual(len(report["missing"]), report["known"])

    def test_installed_is_not_the_same_as_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ido5.3").mkdir()
            (root / "ido5.3" / "cc").write_text("")
            (root / "egcs_1.1.2-4").mkdir()
            (root / "egcs_1.1.2-4" / "mips-linux-gcc").write_text("")
            (root / "ido7.1").mkdir()                          # directory without its driver: not installed
            report = compilers.status(root)
        self.assertEqual((report["installed"], report["usable"]), (2, 1))
        self.assertEqual(list(report["installed_without_lane"]), ["egcs_1.1.2-4"])
        self.assertIn("32-bit loader", report["installed_without_lane"]["egcs_1.1.2-4"])
        self.assertIn("ido7.1", report["missing"])

    def test_no_root_is_reported_not_raised(self):
        self.assertEqual(compilers.status(None)["installed"], 0)


class LaneTest(unittest.TestCase):
    def _root(self, tmp, *idents, wibo=False):
        root = Path(tmp)
        for ident in idents:
            compiler = next(c for c in compilers.CATALOGUE if c.ident == ident)
            for name in compiler.required:
                path = root / compiler.directories[-1] / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
        if wibo:
            (root / "wibo").mkdir()
            (root / "wibo" / "wibo").write_text("")
        return root

    def test_an_installed_compiler_without_a_built_in_pipeline_gets_a_configured_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, "gcc2.8.1pm", "ido6.0", "ido5.3")
            lanes = compilers.project_pipelines(root)
            self.assertEqual(sorted(lanes), ["gcc281pm", "ido60"])           # ido5.3 has a built-in pipeline
            self.assertEqual((lanes["ido60"]["family"], lanes["gcc281pm"]["family"]), ("ido", "gcc"))
            self.assertTrue(lanes["gcc281pm"]["vars"]["root"].endswith("gcc2.8.1pm"))
            self.assertTrue(all("{object}" in table["steps"][-1] for table in lanes.values()))

    def test_a_lane_whose_helper_is_missing_is_left_out_and_says_why(self):
        import unittest.mock
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, "gcc2.7.2sn0006")
            compiler = next(c for c in compilers.CATALOGUE if c.ident == "gcc2.7.2sn0006")
            with unittest.mock.patch("shutil.which", return_value=None):
                self.assertEqual(compilers.project_pipelines(root), {})
                lane, why = compilers.lane_status(root, compiler)
                self.assertIsNone(lane)
                self.assertIn("wibo", why)
            root = self._root(tmp, "gcc2.7.2sn0006", wibo=True)
            self.assertEqual(compilers.lane_status(root, compiler), ("custom:gcc272sn0006", None))
            self.assertTrue(compilers.project_pipelines(root)["gcc272sn0006"]["vars"]["wibo"].endswith("wibo"))

    def test_the_tables_are_what_the_project_configuration_reads(self):
        import tomllib
        from n64dk.match import custom_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, "gcc2.8.1pm", "egcs_1.1.2-4c")
            project = Path(tmp) / "project"
            project.mkdir()
            (project / "n64dk.toml").write_text(compilers.toml_tables(compilers.project_pipelines(root)))
            self.assertEqual(sorted(tomllib.loads((project / "n64dk.toml").read_text())["pipelines"]),
                             ["egcs112", "gcc281pm"])
            recipes = custom_pipeline.load(project)
            self.assertEqual(recipes["egcs112"].env["COMPILER_PATH"], recipes["egcs112"].variables["root"])
            self.assertEqual(recipes["gcc281pm"].default_flags[0], "-O2")

    def test_lanes_carry_the_settings_real_projects_were_built_with(self):
        from n64dk.match import custom_pipeline
        self.assertIn(("-O2", "-fforce-addr", "-G0", "-mips3", "-mgp32", "-mfp32"), compilers.FLAG_SETS["gcc281pm"])
        self.assertIn(("-O2", "-mips2", "-mcpu=4300", "-mno-abicalls"), compilers.FLAG_SETS["egcs112"])
        self.assertTrue(all(lane in {entry["lane"] for entry in compilers.LANES.values()} for lane in compilers.FLAG_SETS))
        text = compilers.toml_tables({"gcc281pm": {"family": "gcc", "default_flags": ["-O2"], "steps": ["cc {source}"],
                                                   "flag_sets": [["-O2", "-fforce-addr"], ["-O1"]], "vars": {},
                                                   "env": {}, "compiler": "gcc2.8.1pm"}})
        self.assertIn('flag_sets = [["-O2", "-fforce-addr"], ["-O1"]]', text)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "n64dk.toml").write_text(text)
            lane = custom_pipeline.load(Path(tmp))["gcc281pm"]
        self.assertEqual(lane.flag_sets, [["-O2", "-fforce-addr"], ["-O1"]])

    def test_the_ido_sweep_ends_with_the_first_instruction_set(self):
        from n64dk.match.engine import sweep_specs
        specs = sweep_specs(["ido5.3"])
        self.assertEqual([spec.flags[:2] for spec in specs[-2:]], [["-O2", "-mips1"], ["-O1", "-mips1"]])
        self.assertEqual(specs[0].flags[:2], ["-O2", "-mips2"])
        self.assertFalse(any("-mips1" in spec.flags for spec in sweep_specs(["kmc+as"])))

    def test_every_lane_names_a_catalogued_compiler_and_every_compiler_is_accounted_for(self):
        idents = {c.ident for c in compilers.CATALOGUE}
        self.assertLessEqual(set(compilers.LANES), idents)
        self.assertEqual(set(compilers.LANES) | set(compilers.BUILT_IN) | set(compilers.NO_LANE), idents)
        names = [lane["lane"] for lane in compilers.LANES.values()]
        self.assertEqual(len(names), len(set(names)))


class FetchTest(unittest.TestCase):
    """The download path against a registry made of local bytes: no network."""

    def _registry(self, ident="egcs_1.1.2-4", member="compilers/n64/egcs_1.1.2-4/mips-linux-gcc", corrupt=False,
                  link=None):
        import hashlib
        import io
        import json
        import tarfile
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            info = tarfile.TarInfo(member)
            data = b"#!/bin/sh\n"
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            if link:
                extra = tarfile.TarInfo(f"compilers/n64/{ident}/escape")
                extra.type, extra.linkname = tarfile.SYMTYPE, link
                archive.addfile(extra)
        layer = raw.getvalue()
        digest = "sha256:" + hashlib.sha256(layer).hexdigest()
        if corrupt:
            layer = layer[:-1] + bytes([layer[-1] ^ 1])
        seen = []

        def opener(url, headers=None, timeout=60):
            seen.append(url)
            if "/token?" in url:
                return io.BytesIO(json.dumps({"token": "t"}).encode())
            if url.endswith("/manifests/latest"):
                return io.BytesIO(json.dumps({"manifests": [
                    {"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:m"}]}).encode())
            if url.endswith("/manifests/sha256:m"):
                return io.BytesIO(json.dumps({"layers": [{"digest": digest}]}).encode())
            if "/blobs/" in url:
                return io.BytesIO(layer)
            raise AssertionError(url)
        return opener, seen

    def test_a_compiler_is_unpacked_into_its_own_directory(self):
        opener, seen = self._registry()
        with tempfile.TemporaryDirectory() as tmp:
            result = compilers.fetch("egcs_1.1.2-4", Path(tmp), opener=opener)
            self.assertEqual((result["status"], result["missing_expected_files"]), ("fetched", []))
            self.assertTrue((Path(tmp) / "egcs_1.1.2-4" / "mips-linux-gcc").is_file())
            self.assertTrue(all(url.startswith("https://ghcr.io/") for url in seen))
            self.assertEqual(compilers.fetch("egcs_1.1.2-4", Path(tmp), opener=opener)["status"], "present")
            self.assertEqual(compilers.status(Path(tmp))["installed"], 1)

    def test_a_layer_that_does_not_match_its_digest_installs_nothing(self):
        opener, _seen = self._registry(corrupt=True)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(compilers.FetchError):
                compilers.fetch("egcs_1.1.2-4", Path(tmp), opener=opener)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_an_archive_that_reaches_outside_is_refused(self):
        for member, link in (("../outside", None), ("compilers/n64/egcs_1.1.2-4/mips-linux-gcc", "/etc/passwd")):
            opener, _seen = self._registry(member=member, link=link)
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(compilers.FetchError):
                    compilers.fetch("egcs_1.1.2-4", Path(tmp) / "root", opener=opener)
                self.assertFalse((Path(tmp) / "outside").exists())

    def test_a_pinned_manifest_is_required_and_reported(self):
        opener, seen = self._registry()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(compilers.FetchError):
                compilers.fetch("egcs_1.1.2-4", Path(tmp), opener=opener, pin="sha256:other")
            self.assertFalse(any("/blobs/" in url for url in seen))        # refused before any layer
            self.assertEqual(list(Path(tmp).iterdir()), [])
            result = compilers.fetch("egcs_1.1.2-4", Path(tmp), opener=opener, pin="sha256:m")
            self.assertEqual((result["status"], result["manifest_digest"]), ("fetched", "sha256:m"))

    def test_an_unknown_identifier_is_refused_before_any_request(self):
        with self.assertRaises(compilers.FetchError):
            compilers.fetch("not-a-compiler", Path("/nonexistent"), opener=lambda *a, **k: self.fail("requested"))


if __name__ == "__main__":
    unittest.main()
