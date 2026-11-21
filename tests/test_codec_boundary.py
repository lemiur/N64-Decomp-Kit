"""The loader/codec boundary (n64dk.assets.codecboundary): synthetic streams only, no ROM."""
import struct
import unittest
import zlib

from n64dk.assets import codecboundary as cb
from n64dk.assets import compress as c
from n64dk.image import Segment

KIT = ("mio0", "yay0", "yaz0", "lzari", "zlib", "gzip", "deflate")


def payload(n: int = 600) -> bytes:
    out = bytearray()
    x = 0x1234
    while len(out) < n:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        kind = x % 4
        if kind == 0:
            out += b"texture_" * (1 + (x >> 8) % 4)
        elif kind == 1:
            out += bytes(1 + (x >> 8) % 24)
        elif kind == 2 and len(out) > 8:
            a = (x >> 8) % (len(out) - 4)
            out += out[a:a + 3 + (x >> 16) % 30]
        else:
            out += bytes(((x >> s) & 0xFF) for s in (3, 7, 11, 15, 19))
    return bytes(out[:n])


REPEATS = b"abcdefgh" * 8          # an LZ encoder makes back references of this


def yaz0_literals(data: bytes) -> bytes:
    """A valid Yaz0 stream holding every byte as a literal (no back references)."""
    out = bytearray(b"Yaz0" + struct.pack(">I", len(data)) + bytes(8))
    for g in range(0, len(data), 8):
        out.append(0xFF)
        out += data[g:g + 8]
    return bytes(out)


def mio0_literals(data: bytes) -> bytes:
    """A valid MIO0 stream holding every byte as a literal (all mask bits set, no links)."""
    words = (len(data) + 31) // 32
    mask = bytearray(4 * words)
    for i in range(len(data)):
        mask[i >> 3] |= 0x80 >> (i & 7)
    offset = 16 + len(mask)
    return b"MIO0" + struct.pack(">III", len(data), offset, offset) + bytes(mask) + data


def raw_deflate(data: bytes, level: int) -> bytes:
    z = zlib.compressobj(level, zlib.DEFLATED, -15)
    return z.compress(data) + z.flush()


class EncoderCoverage(unittest.TestCase):
    def test_every_decoded_codec_has_an_encoder_or_a_statement(self):
        for name in c.REGISTRY:
            with self.subTest(codec=name):
                self.assertTrue(cb.default_encoders(name), "a kit codec without an encoder")
                self.assertNotEqual(cb.encoder_statement(name), cb.NO_DECODER)
        self.assertTrue(cb.encoder_statement("game").startswith("none:"))
        self.assertTrue(cb.encoder_statement("game:0x802004DC").startswith("none:"))
        self.assertEqual(cb.encoder_statement("acme"), cb.NO_DECODER)
        self.assertEqual(cb.default_encoders("acme"), ())
        self.assertIsNone(cb.default_decoder("acme"))

    def test_kit_encodings_are_exact_repacks(self):
        data = payload()
        for name in KIT:
            with self.subTest(codec=name):
                stored = c.encode(name, data)
                r = cb.classify_slice(stored, name, rom=0x1000)
                self.assertEqual((r.cls, r.reason), (cb.EXACT, "stored-identical"))
                self.assertFalse(r.preserved)
                self.assertEqual(r.payload_reconstructed, len(data))
                self.assertEqual((r.consumed, r.stored_size, r.end), (len(stored), len(stored), 0x1000 + len(stored)))
                self.assertIsNone(r.first_difference)
                self.assertTrue(r.reencode_roundtrips)

    def test_deflate_parameter_search(self):
        data = payload(2000)
        stored = raw_deflate(data, 1)
        r = cb.classify_slice(stored, "deflate")
        self.assertEqual(r.cls, cb.EXACT)
        self.assertTrue(r.encoder.startswith("zlib level "))
        level, mem = (int(x) for x in r.encoder.split()[2::2])
        self.assertEqual(cb._zlib_encoder(level, mem, -15)(data), stored)
        if raw_deflate(data, 9) != stored:
            self.assertGreater(r.encoders_tried, 1)
            self.assertIsNotNone(r.first_difference)


class DecodedMatchesStoredDiffers(unittest.TestCase):
    """A different but valid encoding of the same payload: never an exact repack."""

    def check(self, codec: str, stored: bytes):
        self.assertEqual(c.decode(codec, stored).data, REPEATS)
        self.assertNotEqual(c.encode(codec, REPEATS), stored)
        r = cb.classify_slice(stored, codec, rom=0x40)
        self.assertEqual((r.cls, r.reason), (cb.PAYLOAD_ONLY, "reencode-differs"))
        self.assertTrue(r.preserved)
        self.assertIsNone(r.encoder)
        self.assertEqual(r.payload_reconstructed, len(REPEATS))
        self.assertIsNotNone(r.first_difference)
        self.assertTrue(r.reencode_roundtrips, "the kit's encoding is valid, just not this one")
        m = cb.metrics([r])
        self.assertEqual(m["exact_repack"]["slices"], 0)
        self.assertEqual(m["preserved"]["slices"], 1)
        self.assertEqual(m["preserved"]["stored_bytes"], len(stored))
        self.assertEqual(m["payload"]["bytes_reconstructed"], len(REPEATS))
        man = cb.manifest([r])
        self.assertEqual(man["repack"], [])
        self.assertEqual([p["rom"] for p in man["preserved"]], ["0x40"])
        return r

    def test_yaz0_literal_only(self):
        self.check("yaz0", yaz0_literals(REPEATS))

    def test_mio0_literal_only(self):
        self.check("mio0", mio0_literals(REPEATS))

    def test_deflate_stored_block_against_a_short_search(self):
        stored = raw_deflate(REPEATS, 0)            # stored blocks: valid, and not what level 9 writes
        r = cb.classify_slice(stored, "deflate", encoders=cb.default_encoders("deflate")[:9])
        self.assertEqual((r.cls, r.reason), (cb.PAYLOAD_ONLY, "reencode-differs"))
        self.assertEqual(cb.classify_slice(stored, "deflate").cls, cb.EXACT)   # level 0 is in the full search

    def test_loader_reference_matches(self):
        r = cb.classify_slice(yaz0_literals(REPEATS), "yaz0", expected=REPEATS, reference="loader")
        self.assertEqual((r.cls, r.reference), (cb.PAYLOAD_ONLY, "loader"))
        self.assertEqual(cb.metrics([r])["payload"]["bytes_by_reference"]["loader"], len(REPEATS))

    def test_stream_one_byte_longer_than_the_reencoding_is_not_exact(self):
        # compress.reencode_status calls this LZARI case "identical"; the boundary does not.
        stored = b"STREAM-BYTES-X"
        r = cb.classify_slice(stored, "toy", decoder=lambda s: (b"payload", len(s)),
                              encoders=(("toy", lambda p: stored[:-1]),))
        self.assertEqual((r.cls, r.reason, r.first_difference), (cb.PAYLOAD_ONLY, "reencode-differs", len(stored) - 1))

    def test_trailing_bytes_in_the_slice_are_never_exact(self):
        data = payload(300)
        for name in KIT:
            with self.subTest(codec=name):
                enc = c.encode(name, data)
                r = cb.classify_slice(enc + b"\0", name)
                self.assertEqual((r.cls, r.reason), (cb.PAYLOAD_ONLY, "trailing-bytes"))
                self.assertEqual((r.consumed, r.first_difference), (len(enc), len(enc)))
                self.assertIsNotNone(r.encoder)

    def test_no_encoder_is_payload_only(self):
        r = cb.classify_slice(b"\x05hello", "game", decoder=lambda s: (s[1:1 + s[0]] * 3, len(s)), encoders=())
        self.assertEqual((r.cls, r.reason, r.payload_reconstructed), (cb.PAYLOAD_ONLY, "no-encoder", 15))


class Opaque(unittest.TestCase):
    def test_payload_mismatch_is_opaque_even_when_the_stream_reencodes(self):
        stored = c.yaz0_encode(REPEATS)
        wrong = REPEATS[:20] + b"?" + REPEATS[21:]
        r = cb.classify_slice(stored, "yaz0", expected=wrong, reference="loader")
        self.assertEqual((r.cls, r.reason, r.payload_difference), (cb.OPAQUE, "payload-mismatch", 20))
        self.assertEqual(r.payload_reconstructed, 0)
        self.assertTrue(r.preserved)
        self.assertEqual(r.encoders_tried, 0)

    def test_decode_failures(self):
        enc = c.yaz0_encode(payload())
        cases = [(enc[:-3], "yaz0", "decode-error"), (b"not a stream at all.", "mio0", "decode-error"),
                 (enc, "acme", "no-decoder"), (b"", "yaz0", "empty-slice"), (None, "yaz0", "source-not-traced")]
        for stored, codec, reason in cases:
            with self.subTest(reason=reason, codec=codec):
                r = cb.classify_slice(stored, codec)
                self.assertEqual((r.cls, r.reason), (cb.OPAQUE, reason))
                self.assertEqual(r.payload_reconstructed, 0)

    def test_decoder_errors_and_bad_claims_fail_closed(self):
        def boom(_s):
            raise IndexError("ran off the end")
        r = cb.classify_slice(b"abcd", "toy", decoder=boom, encoders=())
        self.assertEqual((r.cls, r.reason), (cb.OPAQUE, "decode-error"))
        self.assertIn("IndexError", r.note)
        r = cb.classify_slice(b"abcd", "toy", decoder=lambda s: (b"xyzxyz", 5), encoders=())
        self.assertEqual((r.cls, r.reason), (cb.OPAQUE, "bad-consumed"))
        r = cb.classify_slice(b"abcd", "toy", decoder=lambda s: (b"", 4), encoders=())
        self.assertEqual((r.cls, r.reason), (cb.OPAQUE, "empty-payload"))

    def test_reference_needs_a_payload(self):
        with self.assertRaises(ValueError):
            cb.classify_slice(b"x", "yaz0", reference="loader")
        with self.assertRaises(ValueError):
            cb.classify_slice(b"x", "yaz0", expected=b"x", reference="guess")


class Metrics(unittest.TestCase):
    def test_numerators_and_denominators_are_separate(self):
        data = payload(400)
        exact1 = cb.classify_slice(c.yaz0_encode(data), "yaz0", rom=0x100)
        exact2 = cb.classify_slice(c.mio0_encode(REPEATS), "mio0", rom=0x2000)
        differs = cb.classify_slice(yaz0_literals(REPEATS), "yaz0", rom=0x3000, expected=REPEATS, reference="loader")
        failed = cb.classify_slice(b"Yaz0\0\0\1\0" + bytes(12), "yaz0", rom=0x4000)
        mismatch = cb.classify_slice(c.yaz0_encode(data), "yaz0", rom=0x5000, expected=data[::-1], reference="loader")
        untraced = cb.classify_slice(None, "game", expected=bytes(64), reference="loader")
        results = [exact1, exact2, differs, failed, mismatch, untraced]
        self.assertEqual([r.cls for r in results],
                         [cb.EXACT, cb.EXACT, cb.PAYLOAD_ONLY, cb.OPAQUE, cb.OPAQUE, cb.OPAQUE])
        m = cb.metrics(results)
        stored = [r.stored_size or 0 for r in results]
        self.assertEqual(m["slices"], 6)
        self.assertEqual(m["classes"], {cb.EXACT: 2, cb.PAYLOAD_ONLY: 1, cb.OPAQUE: 3})
        self.assertEqual(m["exact_repack"], {"slices": 2, "of_slices": 6, "stored_bytes": stored[0] + stored[1],
                                             "of_stored_bytes": sum(stored)})
        self.assertEqual(m["preserved"]["slices"], 4)
        self.assertEqual(m["preserved"]["stored_bytes"], sum(stored[2:]))
        self.assertEqual(m["preserved"]["untraced_slices"], 1)
        p = m["payload"]
        self.assertEqual(p["bytes_reconstructed"], len(data) + 2 * len(REPEATS))
        self.assertEqual(p["of_known_payload_bytes"], len(data) + 2 * len(REPEATS) + len(data) + 64)
        self.assertEqual((p["slices_reconstructed"], p["slices_payload_size_unknown"]), (3, 1))
        self.assertEqual(p["bytes_by_reference"], {"loader": len(REPEATS), "declared": 0,
                                                   "none": len(data) + len(REPEATS)})
        self.assertEqual(m["reasons"][cb.OPAQUE], {"decode-error": 1, "payload-mismatch": 1, "source-not-traced": 1})

    def test_manifest_and_overlap(self):
        a = cb.classify_slice(c.yaz0_encode(REPEATS), "yaz0", rom=0x100)
        b = cb.classify_slice(yaz0_literals(REPEATS), "yaz0", rom=0x100 + a.stored_size)
        man = cb.manifest([b, a])
        self.assertEqual([x["rom"] for x in man["repack"]], ["0x100"])
        self.assertEqual(man["repack"][0]["encoder"], "kit yaz0")
        self.assertEqual([x["class"] for x in man["preserved"]], [cb.PAYLOAD_ONLY])
        self.assertFalse(man["strict"])
        overlapping = cb.classify_slice(yaz0_literals(REPEATS), "yaz0", rom=0x100 + a.stored_size - 1)
        with self.assertRaises(ValueError):
            cb.manifest([a, overlapping])


class LoaderBoundary(unittest.TestCase):
    """image.Segment as boot emulation leaves it: ``data`` is the loader's output."""

    def rom_with(self, stream: bytes, at: int = 0x40) -> bytes:
        return bytes(at) + stream + bytes(0x200)

    def segment(self, data: bytes, **kw) -> Segment:
        return Segment(0x80000400, len(data), "packed", data=data, **kw)

    def test_kit_codec_stream_is_cut_at_its_exact_end(self):
        data = payload(500)
        stream = c.yaz0_encode(data)
        seg = self.segment(data, rom=0x40, rom_end=0x40 + len(stream) + 0x80, codec="yaz0")
        r = cb.classify_segment(self.rom_with(stream), seg)
        self.assertEqual((r.cls, r.reference, r.rom, r.stored_size), (cb.EXACT, "loader", 0x40, len(stream)))

    def test_header_before_the_stream(self):
        data = payload(500)
        stream = c.yaz0_encode(data)
        seg = self.segment(data, rom=0x40, rom_end=0x300, codec="yaz0")
        r = cb.classify_segment(self.rom_with(b"HDR!" + stream), seg)
        self.assertEqual((r.cls, r.rom), (cb.EXACT, 0x44))

    def test_loader_output_matches_but_the_stored_stream_differs(self):
        stream = yaz0_literals(REPEATS)
        seg = self.segment(REPEATS, rom=0x40, rom_end=0x100, codec="yaz0")
        r = cb.classify_segment(self.rom_with(stream), seg)
        self.assertEqual((r.cls, r.reason, r.reference), (cb.PAYLOAD_ONLY, "reencode-differs", "loader"))
        self.assertEqual(cb.manifest([r])["preserved"][0]["size"], len(stream))

    def test_trimmed_loader_output(self):
        data = payload(500) + bytes(16)
        seg = self.segment(data[:-16], rom=0x40, rom_end=0x300, codec="yaz0")
        r = cb.classify_segment(self.rom_with(c.yaz0_encode(data)), seg)
        self.assertEqual((r.cls, r.payload_reconstructed, r.unverified_tail), (cb.EXACT, 500, 16))

    def test_loader_output_disagrees_with_the_kit_decode(self):
        data = payload(500)
        seg = self.segment(data[:300] + bytes([data[300] ^ 0xFF]) + data[301:], rom=0x40, rom_end=0x300, codec="yaz0")
        r = cb.classify_segment(self.rom_with(c.yaz0_encode(data)), seg)
        self.assertEqual((r.cls, r.reason, r.payload_difference), (cb.OPAQUE, "payload-mismatch", 300))

    def test_game_codec_and_unknown_and_untraced(self):
        rom = self.rom_with(b"\x05hello")
        seg = self.segment(b"hello" * 3, rom=0x40, rom_end=0x46, game_codec="game routine 0x80200470")
        r = cb.classify_segment(rom, seg)
        self.assertEqual((r.cls, r.reason, r.codec), (cb.OPAQUE, "no-decoder", "game"))
        r = cb.classify_segment(rom, seg, decoder=lambda s: (s[1:1 + s[0]] * 3, len(s)))
        self.assertEqual((r.cls, r.reason, r.payload_reconstructed), (cb.PAYLOAD_ONLY, "no-encoder", 15))
        r = cb.classify_segment(rom, self.segment(b"hello", rom=0x40, rom_end=0x46))
        self.assertEqual((r.cls, r.reason, r.codec), (cb.OPAQUE, "no-decoder", "unknown"))
        r = cb.classify_segment(rom, self.segment(b"hello"))
        self.assertEqual((r.cls, r.reason, r.expected_size), (cb.OPAQUE, "source-not-traced", 5))
        with self.assertRaises(ValueError):
            cb.classify_segment(rom, Segment(0x80000400, 4, "rom", rom=0x40, rom_end=0x44, data=b"\0" * 4))


if __name__ == "__main__":
    unittest.main()
