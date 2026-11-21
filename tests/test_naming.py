import struct
import unittest

from n64dk.analysis import naming
from n64dk.match import symbols as symmod


def words(*ws):
    return b"".join(struct.pack(">I", w) for w in ws)


class NamingTest(unittest.TestCase):
    def test_fields_stay_apart_and_display_says_where_from(self):
        rows = [("func_80001000", 0x80001000, 8), ("func_80001008", 0x80001008, 8), ("func_80001010", 0x80001010, 8)]
        table = naming.build(rows, sdk={0x80001000: "osPiStartDma"},
                             original={0x80001008: ("PlayerMove", "SN function record @0x1234")},
                             donor={0x80001008: ("player_move", "lineage: other title")},
                             user={0x80001010: "MyName"})
        a, b, c = (t.as_row() for t in table)
        self.assertEqual((a["display"], a["display_from"], a["canonical"]), ("osPiStartDma", "sdk", "func_80001000"))
        self.assertEqual((b["display"], b["display_from"]), ("PlayerMove", "original"))
        self.assertEqual(b["donor"]["name"], "player_move")
        self.assertEqual((c["display"], c["display_from"]), ("MyName", "user"))

    def test_mmio_roles_and_a_generated_name_is_never_original(self):
        code = words(0x3C08A460, 0x8D090010, 0x03E00008, 0x00000000)    # lui t0,0xA460 ; lw t1,16(t0)
        table = naming.build([("f", 0x80002000, 16)], image=code, delta=0x80002000)
        self.assertEqual(table[0].roles, ["pi"])
        self.assertEqual(table[0].semantic["name"], "sem_pi_80002000")
        self.assertIsNone(table[0].original)
        self.assertEqual(table[0].display(), ("sem_pi_80002000", "semantic"))

    def test_a_message_string_names_its_printer_provisionally(self):
        text = b"osPiStartDma: bad length\x00"
        data_at = 0x80003000
        code = words(0x3C048000, 0x24843000, 0x03E00008, 0x00000000)    # lui a0,0x8000 ; addiu a0,a0,0x3000
        image = code + b"\x00" * (0x3000 - len(code)) + text
        read = naming.string_reader(image, 0x80000000)
        table = naming.build([("f", 0x80000000, 16)], image=image, delta=0x80000000, read_string=read)
        self.assertEqual(table[0].semantic["name"], "sem_osPiStartDma")
        self.assertEqual(table[0].semantic["source"], "message string")
        self.assertEqual(read(data_at), "osPiStartDma: bad length")

    def test_names_never_enter_a_symbol_map(self):
        table = naming.build([("func_80001000", 0x80001000, 8)], sdk={0x80001000: "osPiStartDma"})
        tsv = naming.write_tsv(table)
        self.assertIn("osPiStartDma [sdk signature]", tsv)
        # NAMES.tsv is not a symbol file the verifier reads: no `name = 0xADDR;` lines
        self.assertFalse(any(symmod.SPLAT_LINE.match(line) for line in tsv.splitlines()))


if __name__ == "__main__":
    unittest.main()
