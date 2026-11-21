"""Toolchain availability checks."""
import unittest
class HostPrerequisiteTest(unittest.TestCase):
    """The SN lanes run the host gcc as their preprocessor: without it they are not usable (Macbook, 2026-11-18)."""

    def test_sn_lanes_are_unusable_without_host_gcc(self):
        from unittest import mock
        from n64dk.match import compilers, toolchain
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(toolchain.missing_host("sn"), ["gcc"])
            rows = {r["id"]: r for r in compilers.status(None)["compilers"]}
        sn = [r for r in rows.values() if r["family"] == "sn"]
        self.assertTrue(sn and all(not r["usable"] for r in sn))
        with mock.patch("shutil.which", return_value="/usr/bin/gcc"):
            self.assertEqual(toolchain.missing_host("sn"), [])


if __name__ == "__main__":
    unittest.main()
