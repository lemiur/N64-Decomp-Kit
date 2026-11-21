import json
import tempfile
import unittest
from pathlib import Path

from n64dk.match import components as comp
from n64dk.match import toolmanifest as tm


class ManifestTest(unittest.TestCase):
    def test_the_catalogue_alone_builds_a_manifest_without_paths(self):
        m = tm.build(None, probes={"kmc-as-2.6": "aa"})
        self.assertEqual(len(m["components"]), len(comp.COMPONENTS))
        self.assertEqual(m["probes"], {"kmc-as-2.6": "aa"})
        text = json.dumps(m)
        self.assertNotIn("/home/", text)
        self.assertEqual(m["digest"], tm.digest(m))
        self.assertTrue(all(c["supply_class"] in ("preserved-original", "static-recompilation", "source-rebuild",
                                                  "reconstructed", "comparative") for c in m["components"]))

    def test_the_host_does_not_change_the_digest(self):
        m = tm.build(None)
        other = dict(m, host={"system": "Other", "machine": "x", "python": "3.11.0"})
        self.assertEqual(tm.digest(m), tm.digest(other))

    def test_a_file_on_disk_is_digested_and_differences_are_precise(self):
        component = next(c for c in comp.COMPONENTS if c.files and c.directory)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for root, content in ((a, b"one"), (b, b"two")):
                base = Path(root) / component.directory
                base.mkdir(parents=True, exist_ok=True)
                (base / component.files[0]).parent.mkdir(parents=True, exist_ok=True)
                (base / component.files[0]).write_bytes(content)
            ma, mb = tm.build(Path(a)), tm.build(Path(b))
            diffs = tm.compare(ma, mb)
            kinds = {(d["kind"], d.get("component")) for d in diffs}
            self.assertIn(("file-digest", component.ident), kinds)
            self.assertEqual(tm.compare(ma, ma), [])

    def test_a_probe_difference_with_equal_components_names_the_host(self):
        ma, mb = tm.build(None, probes={"p": "1"}), tm.build(None, probes={"p": "2"})
        diffs = tm.compare(ma, mb)
        self.assertEqual([d["kind"] for d in diffs], ["probe-output"])
        self.assertIn("host-side", diffs[0]["note"])

    def test_probe_digests_from_result_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.json").write_text(json.dumps({"expansions": [{"form": "f", "words": [1, 2], "error": ""}]}))
            Path(tmp, "bad.json").write_text("{")
            got = tm.probe_digests(Path(tmp))
            self.assertEqual(list(got), ["x"])

    def test_supply_classes(self):
        self.assertEqual(tm.supply_class({"provenance": "authentic-n64", "emulator": "ido-static-recomp"}),
                         "static-recompilation")
        self.assertEqual(tm.supply_class({"provenance": "authentic-n64", "emulator": "wine"}), "preserved-original")
        self.assertEqual(tm.supply_class({"provenance": "comparative"}), "comparative")
        self.assertEqual(tm.supply_class({"provenance": "authentic-n64", "platform": "x86-64 ELF (recompiled)",
                                          "source_url": "https://github.com/decompals/ido-static-recomp"}),
                         "static-recompilation")
        ido = next(c for c in comp.COMPONENTS if c.ident == "ido5.3/cfe")
        from dataclasses import asdict
        self.assertEqual(tm.supply_class(asdict(ido)), "static-recompilation")
        self.assertEqual(tm.supply_class({"provenance": "reconstructed", "origin": "n64dk/match/normalize.py"}),
                         "reconstructed")


if __name__ == "__main__":
    unittest.main()
