"""decompile-dashboard: static, local, read-only HTML from a synthetic run (round 12, item 15)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from n64dk.commands import dashboard
from tests.test_review import MACHINE, make_run, snapshot

SECTIONS = ("Funnel", "Byte ledger", "Image graph", "Overlays", "Call graph summary", "Toolchain regions",
            "Translation units", "Hard-case clusters", "Stage yield", "Build cost", "Cache", "Resources",
            "Review queue")


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-dashboard-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.run_dir = make_run(self.root)

    def test_page_is_self_contained_and_portable(self):
        result = dashboard.write(self.run_dir, self.root / "out")
        page = (self.root / "out" / "index.html").read_text()
        model = (self.root / "out" / "dashboard.json").read_text()
        for title in SECTIONS:
            self.assertIn(f"<h2>{title}", page)
        audit = result["audit"]
        self.assertEqual((audit["external"], audit["scripts"], audit["control_characters"]), (0, 0, 0))
        self.assertEqual(audit["machine_paths"], [])
        for text in (page, model):
            self.assertNotIn(MACHINE, text)
            self.assertNotIn("/srv/", text)
        self.assertIn("title_base/run", page)

    def test_funnel_and_sections(self):
        model = dashboard.collect(self.run_dir)
        main = model["images"]["main"]
        self.assertEqual(main["funnel"], {"exact": 1, "donor-verified": 20, "assumed": 2, "assembly": 3, "raw": 1,
                                          "open": 13})
        self.assertEqual(main["case_statuses"], {"assumed": 1, "near": 2, "open": 1, "exact": 1})
        self.assertEqual(main["units"]["count"], 2)
        self.assertEqual(model["profile"]["resolve"]["cache_hits"], 30)
        self.assertEqual(model["review"]["by_kind"], {"assumed": 1, "near": 2, "donor": 1})
        self.assertEqual(main["toolchain_regions"]["summary"]["regions"], 1)

    def test_read_only_and_refuses_output_inside_the_run(self):
        before = snapshot(self.run_dir, skip="")
        dashboard.write(self.run_dir, self.root / "elsewhere")
        self.assertEqual(snapshot(self.run_dir, skip=""), before)
        with self.assertRaises(ValueError):
            dashboard.write(self.run_dir, self.run_dir / "dashboard")
        self.assertFalse((self.run_dir / "dashboard").exists())

    def test_sparse_run_still_renders(self):
        run = self.root / "sparse" / "run"
        run.mkdir(parents=True)
        (run / "report.json").write_text(json.dumps({"status": "running", "images": {"main": {"functions": 3}}}))
        page = dashboard.render(dashboard.collect(run))
        self.assertIn("none recorded", page)
        self.assertEqual(dashboard.audit(page)["external"], 0)

    def test_escaping(self):
        page = dashboard.render({"run": "<b>x</b>", "images": {}, "review": {}})
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", page)
        self.assertNotIn("<b>x</b>", page)

    def test_commands_are_registered(self):
        from n64dk import cli
        from n64dk.commands import review as _review_commands  # noqa: F401  (registers on import)
        for name in ("decompile-dashboard", "review-queue", "review-show", "review-decide"):
            self.assertIn(name, cli.COMMANDS)


if __name__ == "__main__":
    unittest.main()
