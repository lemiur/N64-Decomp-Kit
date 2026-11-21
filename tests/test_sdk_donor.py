import tempfile
import unittest
from pathlib import Path

from n64dk.sdk import donor


class RecipeTest(unittest.TestCase):
    def test_the_gcc_build_uses_the_librarys_own_levels_and_defines(self):
        pipeline, flags = donor.unit_recipe("J", "libgultra_rom", "os/createthread.c")
        self.assertEqual(pipeline, "kmc+as")
        self.assertEqual(flags[0], "-O3")
        self.assertIn("-D_FINALROM", flags)
        self.assertIn("-DNDEBUG", flags)
        self.assertIn("-funsigned-char", flags)                      # 2.0D .. 2.0J
        self.assertIn('-DBUILD_VERSION_STRING="2.0J"', flags)
        self.assertNotIn("-funsigned-char", donor.unit_recipe("L", "libgultra_rom", "os/createthread.c")[1])
        self.assertEqual(donor.unit_recipe("J", "libgultra_rom", "os/initialize_isv.c")[1][0], "-O2")
        self.assertEqual(donor.unit_recipe("K", "libgultra_rom", "os/seterrorhandler.c")[1][0], "-O0")
        self.assertEqual(donor.unit_recipe("H", "libgultra_rom", "os/seterrorhandler.c")[1][0], "-O3")
        self.assertNotIn("-DNDEBUG", donor.unit_recipe("I", "libgultra_rom", "os/createthread.c")[1])
        self.assertIn("-D_DEBUG", donor.unit_recipe("J", "libgultra_d", "os/createthread.c")[1])

    def test_the_build_that_links_rmon_is_tried_by_default(self):
        """Sesame Street (KMC, 2.0F) links rmon and the debug-port exception code: its library is the
        plain ``libgultra`` build (``_FINALROM`` and ``_DEBUG`` undefined, ``NDEBUG`` defined,
        ``-funsigned-char``), which the ``_rom`` builds compile out."""
        import argparse
        from n64dk.commands import sdkdonor
        parser = argparse.ArgumentParser()
        sdkdonor.sdk_donor.arguments(parser)
        defaults = parser.parse_args(["--source", "s", "--out", "o"]).flavours.split(",")
        self.assertIn("libgultra", defaults)
        self.assertIn("libultra", defaults)
        flags = donor.unit_recipe("F", "libgultra", "rmon/rmonmain.c")[1]
        self.assertIn("-DNDEBUG", flags)
        self.assertIn("-funsigned-char", flags)
        self.assertFalse({"-D_FINALROM", "-D_DEBUG"} & set(flags))

    def test_the_ido_build_varies_by_directory_and_version(self):
        self.assertEqual(donor.unit_recipe("H", "libultra_rom", "os/createthread.c")[0], "ido5.3")
        self.assertEqual(donor.unit_recipe("H", "libultra_rom", "os/createthread.c")[1][0], "-O1")
        self.assertEqual(donor.unit_recipe("H", "libultra_rom", "gu/mtxutil.c")[1][0], "-O3")
        self.assertEqual(donor.unit_recipe("K", "libultra_rom", "gu/mtxutil.c")[1][0], "-O2")
        self.assertEqual(donor.unit_recipe("K", "libultra_rom", "os/createthread.c")[1][0], "-O1")
        wide = donor.unit_recipe("K", "libultra_rom", "libc/ll.c")[1]
        self.assertIn("-mips3", wide)
        self.assertNotIn("-mips2", wide)

    def test_the_gbi_define_follows_the_unit(self):
        self.assertIn("-DF3DEX_GBI", donor.unit_recipe("J", "libgultra_rom", "gu/lookat.c")[1])
        self.assertIn("-DF3D_GBI", donor.unit_recipe("J", "libgultra_rom", "sp/sprite.c")[1])
        flags = donor.unit_recipe("J", "libgultra_rom", "sp/spriteex.c")[1]
        self.assertFalse(any("GBI" in flag for flag in flags))


class UnitListTest(unittest.TestCase):
    def test_the_archive_listing_selects_the_units_and_voice_is_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("os/a.c", "os/b.c", "gu/c.c", "voice/v.c"):
                (root / "src" / name).parent.mkdir(parents=True, exist_ok=True)
                (root / "src" / name).write_text("")
            self.assertEqual(donor.units_of(root, "J", "libgultra_rom"), ["gu/c.c", "os/a.c", "os/b.c"])
            (root / "base" / "J").mkdir(parents=True)
            (root / "base" / "J" / "libgultra_rom.txt").write_text("a.o\nc.o\nmissing.o\n")
            self.assertEqual(donor.units_of(root, "J", "libgultra_rom"), ["gu/c.c", "os/a.c"])
            self.assertEqual(donor.units_of(root, "K", "libgultra_rom"), ["gu/c.c", "os/a.c", "os/b.c"])


class AssemblyUnitTest(unittest.TestCase):
    def test_hand_written_units_are_listed_and_the_matrix_routines_follow_the_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("os/setintmask.s", "os/a.c", "gu/scale.c", "mgu/scale.s", "libc/bcopy.s", "top.s"):
                (root / "src" / name).parent.mkdir(parents=True, exist_ok=True)
                (root / "src" / name).write_text("")
            early = donor.units_of(root, "J", "libgultra_rom")
            late = donor.units_of(root, "L", "libgultra_rom")
            self.assertIn("os/setintmask.s", early)
            self.assertIn("libc/bcopy.s", early)
            self.assertNotIn("top.s", early)                             # not in a library directory
            self.assertEqual(("gu/scale.c" in early, "mgu/scale.s" in early), (True, False))
            self.assertEqual(("gu/scale.c" in late, "mgu/scale.s" in late), (False, True))

    def test_the_ido_recipe_of_an_assembly_unit(self):
        flags = donor.assembly_recipe("J", "libultra_rom", "os/setintmask.s")
        self.assertEqual(flags[0], "-O1")
        self.assertIn("-mips2", flags)
        self.assertIn("-D_FINALROM", flags)
        self.assertIn("-D_LANGUAGE_ASSEMBLY", flags)
        self.assertEqual(donor.assembly_recipe("J", "libultra_rom", "libc/bcopy.s")[0], "-O2")
        self.assertIn("-mips3", donor.assembly_recipe("J", "libultra_rom", "os/exceptasm.s"))

    def test_boot_globals_are_the_fixed_low_memory_addresses(self):
        self.assertEqual(donor.BOOT_GLOBALS["osTvType"], 0x80000300)
        self.assertEqual(donor.BOOT_GLOBALS["osMemSize"], 0x80000318)
        self.assertTrue(all(0x80000300 <= address < 0x80000400 for address in donor.BOOT_GLOBALS.values()))


def _object(symbols: list[tuple[int, int]], relocations: list[int]) -> bytes:
    """A big-endian ELF32 object with a symbol table (``(name offset, binding)`` per symbol after
    the null one) and one REL section whose entries name symbol indices."""
    import struct
    symtab = bytes(16) + b"".join(struct.pack(">IIIBBH", name, 0, 0, binding << 4, 0, 1) for name, binding in symbols)
    rel = b"".join(struct.pack(">II", 4 * n, (index << 8) | 4) for n, index in enumerate(relocations))
    body = symtab + rel
    offset = 0x34
    header = struct.pack(">4sBBB9xHHIIIIIHHHHHH", b"\x7fELF", 1, 2, 1, 1, 8, 1, 0, 0, offset + len(body), 0,
                         0x34, 0, 0, 0x28, 3, 0)
    sections = bytes(40)
    sections += struct.pack(">10I", 0, 2, 0, 0, offset, len(symtab), 0, 0, 4, 16)                # .symtab
    sections += struct.pack(">10I", 0, 9, 0, 0, offset + len(symtab), len(rel), 1, 0, 4, 8)       # .rel -> symtab 1
    return header + body + sections


class SymbolOrderTest(unittest.TestCase):
    def test_locals_move_in_front_and_relocations_follow_their_symbols(self):
        import struct
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unit.o"
            # symbols 1..4: global(10), local(20), global(30), local(40); relocations name 1, 2, 4
            path.write_bytes(_object([(10, 1), (20, 0), (30, 1), (40, 0)], [1, 2, 4]))
            self.assertTrue(donor.sort_symbols(path))
            data = path.read_bytes()
            shoff, = struct.unpack_from(">I", data, 0x20)
            symtab = struct.unpack_from(">10I", data, shoff + 40)
            names = [struct.unpack_from(">I", data, symtab[4] + 16 * n)[0] for n in range(5)]
            self.assertEqual(names, [0, 20, 40, 10, 30])                 # null and locals first, order kept
            self.assertEqual(symtab[7], 3)                               # sh_info: the first global
            rel = struct.unpack_from(">10I", data, shoff + 80)
            indices = [struct.unpack_from(">I", data, rel[4] + 8 * n + 4)[0] >> 8 for n in range(3)]
            self.assertEqual([names[i] for i in indices], [10, 20, 40])  # each still names its symbol
            self.assertFalse(donor.sort_symbols(path))                   # already in order

    def test_something_that_is_not_such_an_object_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.txt"
            path.write_bytes(b"not an object")
            self.assertFalse(donor.sort_symbols(path))
            self.assertEqual(path.read_bytes(), b"not an object")


if __name__ == "__main__":
    unittest.main()
