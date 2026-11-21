import struct
import unittest

from n64dk.analysis import lineage


def words(*ws):
    return b"".join(struct.pack(">I", w) for w in ws)


def body(address_hi, address_lo, call):
    # lui $t6,hi ; lw $t7,lo($t6) ; jal call ; addiu $a0,$t7,4 ; jr $ra ; nop ; addiu $sp,$sp,-24 ; sw $ra,20($sp)
    return words(0x3C0E0000 | address_hi, 0x8DCF0000 | address_lo, 0x0C000000 | (call >> 2 & 0x3FFFFFF),
                 0x25E40004, 0x03E00008, 0x00000000, 0x27BDFFE8, 0xAFBF0014)


class MaskTest(unittest.TestCase):
    def test_relocated_fields_are_blanked_and_constants_kept(self):
        a = lineage.hash_function("t1", "f", 0x80001000, body(0x8010, 0x1234, 0x80002000))
        b = lineage.hash_function("t2", "g", 0x80005000, body(0x8020, 0x5678, 0x80009000))
        self.assertNotEqual(a.exact, b.exact)
        self.assertEqual(a.masked, b.masked)
        c = lineage.hash_function("t3", "h", 0x80001000, body(0x8010, 0x1234, 0x80002000)[:-4] + words(0xAFBF0018))
        self.assertNotEqual(a.masked, c.masked)               # a stack offset is not a relocation

    def test_lo_after_the_base_register_is_overwritten_is_kept(self):
        code = words(0x3C0E8010, 0x01C07021, 0x8DCF0010)    # lui t6 ; addu t6,t6,zero (rd=t6) ; lw t7,16(t6)
        masked = lineage.masked_words(code)
        self.assertEqual(masked[2], 0x8DCF0010)
        code = words(0x3C0E8010, 0x8DCF0010)
        self.assertEqual(lineage.masked_words(code)[1], 0x8DCF0000)

    def test_addiu_carries_the_address_to_another_register(self):
        code = words(0x3C0E8010, 0x25CF1234, 0x8DE40008)    # lui t6 ; addiu t7,t6,lo ; lw a0,8(t7)
        self.assertEqual(lineage.masked_words(code), [0x3C0E0000, 0x25CF0000, 0x8DE40000])


class GraphTest(unittest.TestCase):
    def test_lineage_and_library_classes_are_kept_apart(self):
        engine = body(0x8010, 0x10, 0x80002000)
        lib = words(*([0x27BDFFE0, 0xAFBF001C, 0x24040001] * 4))
        hashes = []
        for i, title in enumerate(["a", "b", "c", "d", "e"]):
            hashes.append(lineage.hash_function(title, "osLib", 0x80000400 + i * 16, lib))
        hashes.append(lineage.hash_function("a", "engine", 0x80003000, engine))
        hashes.append(lineage.hash_function("b", "engine2", 0x80004000, body(0x8030, 0x20, 0x80001000)))
        g = lineage.graph(hashes)
        lineage_edges = [e for e in g["edges"] if e["class"] == "lineage"]
        self.assertEqual([(e["a"], e["b"], e["shared"], e["exact_shared"]) for e in lineage_edges], [("a", "b", 1, 0)])
        self.assertEqual(g["library_hashes"], 1)
        self.assertEqual(len([e for e in g["edges"] if e["class"] == "library"]), 10)
        self.assertEqual(g["titles"]["a"]["lineage_partners"], {"b": 1})

    def test_short_functions_say_nothing(self):
        stub = words(0x03E00008, 0x00000000)
        g = lineage.graph([lineage.hash_function("a", "s", 0, stub), lineage.hash_function("b", "s", 0, stub)])
        self.assertEqual(g["edges"], [])

    def test_proposals_name_the_sources_and_never_credit(self):
        engine = body(0x8010, 0x10, 0x80002000)
        hs = [lineage.hash_function("target", "func_80003000", 0x80003000, engine),
              lineage.hash_function("rev1", "PlayerUpdate", 0x80003100, body(0x8011, 0x14, 0x80002100))]
        p = lineage.proposals("target", hs)
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0]["sources"][0]["name"], "PlayerUpdate")
        self.assertEqual((p[0]["class"], p[0]["same_bytes"]), ("lineage", []))
        self.assertNotIn("exact", p[0])

    def test_hashes_skip_rows_outside_the_image(self):
        image = body(0x8010, 0x10, 0x80002000)
        hs = lineage.hashes("t", image, 0x80000000, [("in", 0x80000000, 32), ("out", 0x80000020, 32)])
        self.assertEqual([h.name for h in hs], ["in"])


if __name__ == "__main__":
    unittest.main()
