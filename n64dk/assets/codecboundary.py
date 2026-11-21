"""The loader/codec boundary: what each compressed ROM slice can be rebuilt from (report only).

A compressed slice of the ROM (its *stored stream*) raises three separate questions, answered
separately and never folded into one number:

1. Does the kit decode it, and is the decoded payload the reference payload?  The reference is
   the bytes the game's loader wrote during boot emulation (``reference="loader"``), bytes the
   caller declares (``"declared"``), or nothing but the decode itself (``"none"``).
2. Does an encoder the kit has regenerate the stored slice byte for byte from that payload?
3. If not, the stored slice is a preserved input: a ROM hash match has to take it from the ROM.

Classes (:func:`classify_slice`):

``exact-repack``  the payload matches and a named encoder's output equals the whole stored slice
                  (same bytes, same length).  The slice is a function of its payload.
``payload-only``  the payload matches, but no encoder regenerates the stored bytes: a different
                  but valid encoding of the same payload, bytes in the slice past the stream's
                  end, or no encoder at all (a game's own codec).  The payload is reconstructed;
                  the stored slice is a preserved input.
``opaque``        nothing the kit has decodes the slice to the reference: decode error, no
                  decoder, untraced source, payload mismatch.  A preserved input.

Strictness.  A classification is a measurement: it grants no credit and changes no run,
receipt or ledger.  Equality of the whole stored slice with an encoder's output is the only
test for ``exact-repack``; there is no tolerance.  (``compress.reencode_status`` accepts an
LZARI re-encoding one byte shorter than the stored stream as "identical"; here that is
``payload-only``.)  A payload that differs from the loader's output is ``opaque`` even when
the stored stream re-encodes identically: the kit's decode is then not the game's.
"""
from __future__ import annotations

import hashlib
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from . import compress

EXACT = "exact-repack"
PAYLOAD_ONLY = "payload-only"
OPAQUE = "opaque"
CLASSES = (EXACT, PAYLOAD_ONLY, OPAQUE)
REFERENCES = ("loader", "declared", "none")

Decoder = Callable[[bytes], tuple]           # stored slice -> (payload, consumed)
Encoders = tuple                              # ((label, payload -> stream), ...)

# (a) Every codec the kit decodes, and the encoder this module uses for it.
ENCODERS = {
    "mio0": "kit encoder (assets/compress.py: greedy parse, one-byte lazy look-ahead, farthest offset "
            "on ties, 4 KiB window, matches 3..18); reproduces streams built with that parser",
    "yay0": "kit encoder (the same parser, matches up to 273); agreement with retail Yay0 streams is "
            "pending measurement",
    "yaz0": "kit encoder (the same parser, matches up to 0x111); agreement with retail Yaz0 streams is "
            "pending measurement",
    "lzari": "kit encoder (Okumura's binary-tree LZARI parser and arithmetic coder); a re-encoding "
             "shorter or longer than the stored stream is not exact here",
    "zlib": "zlib at levels 9..0 x memLevel 9..1 (the parameter search of "
            "compress.deflate_reencode_identical), zlib header",
    "gzip": "zlib at levels 9..0 x memLevel 9..1 with zlib's own gzip header (mtime 0, its OS byte); "
            "a stored header with another mtime, OS byte or file name is never reproduced",
    "deflate": "zlib at levels 9..0 x memLevel 9..1, raw deflate",
    "game": "none: the kit runs the game's own decompressor under emulation (n64dk.gamecodec); the "
            "compressor was a development tool and is not in the ROM; an encoder needs the format "
            "identified (the routine decompiled) and its parser reproduced, so a game-codec slice is "
            "at best payload-only",
}
NO_DECODER = "none: the kit has no decoder for this codec, so it has no encoder either"
DEFLATE_WBITS = {"deflate": -15, "zlib": 15, "gzip": 31}
_DEFAULT = object()


def encoder_statement(codec: str) -> str:
    if codec in ENCODERS:
        return ENCODERS[codec]
    if codec.startswith("game"):
        return ENCODERS["game"]
    return NO_DECODER


def _zlib_encoder(level: int, mem: int, wbits: int):
    def run(data: bytes) -> bytes:
        c = zlib.compressobj(level, zlib.DEFLATED, wbits, mem)
        return c.compress(data) + c.flush()
    return run


def default_encoders(codec: str) -> Encoders:
    """The encoders tried, in order.  Deflate family: the first is the kit's own (level 9,
    memLevel 8), then the same order as ``compress.deflate_reencode_identical``."""
    if codec in DEFLATE_WBITS:
        wbits = DEFLATE_WBITS[codec]
        return tuple((f"zlib level {level} memLevel {mem}", _zlib_encoder(level, mem, wbits))
                     for level in range(9, -1, -1) for mem in (8, 9, 7, 6, 5, 4, 3, 2, 1))
    if codec in compress.REGISTRY:
        return ((f"kit {codec}", compress.REGISTRY[codec].encode),)
    return ()


def default_decoder(codec: str, max_output: int = compress.DEFAULT_MAX_OUTPUT) -> Decoder | None:
    if codec not in compress.REGISTRY:
        return None

    def run(stored: bytes) -> tuple[bytes, int]:
        d = compress.decode(codec, stored, 0, max_output)
        return d.data, d.consumed
    return run


def game_codec_decoder(rom, codec, start: int, max_steps: int = 30_000_000) -> Decoder:
    """A decoder that runs the game's own routine (``n64dk.gamecodec``) on the stored slice,
    which must be the ROM's bytes at ``start``.  It has no encoder: such a slice is never
    ``exact-repack``."""
    from .. import gamecodec

    def run(stored: bytes) -> tuple[bytes, int]:
        if bytes(rom.data[start:start + len(stored)]) != bytes(stored):
            raise ValueError("the stored slice is not the ROM's bytes at its start")
        d = gamecodec.decode(rom, codec, start, max_in=len(stored), max_steps=max_steps)
        if d is None:
            raise compress.CodecError("the game routine did not return having expanded its input")
        return d.data, d.consumed
    return run


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _first_difference(a: bytes, b: bytes) -> int | None:
    n = min(len(a), len(b))
    step = 4096
    for base in range(0, n, step):
        end = min(base + step, n)
        if a[base:end] != b[base:end]:                 # equal-length windows: a shorter tail is not a difference
            return next(i for i in range(base, min(base + step, n)) if a[i] != b[i])
    return None if len(a) == len(b) else n


@dataclass
class SliceClass:
    codec: str
    cls: str
    reason: str
    reference: str                        # loader | declared | none
    rom: int | None = None                # ROM offset of the first stored byte (None: untraced)
    stored_size: int | None = None
    stored_sha1: str | None = None
    consumed: int | None = None           # bytes of the slice the decoder read
    payload_size: int | None = None       # decoded bytes
    payload_sha1: str | None = None
    expected_size: int | None = None      # reference payload bytes
    payload_reconstructed: int = 0        # decoded bytes equal to the reference (all of them, with none)
    unverified_tail: int = 0              # decoded bytes past a trimmed reference: not counted
    payload_difference: int | None = None  # first differing payload offset (payload-mismatch)
    encoder: str | None = None            # the encoder whose output equals the stream (or the slice)
    encoders_tried: int = 0
    reencoded_size: int | None = None     # the first encoder's output size
    first_difference: int | None = None   # slice offset where the first encoder's output differs
    reencode_roundtrips: bool | None = None  # the first encoder's output decodes to the payload
    note: str = ""

    @property
    def end(self) -> int | None:
        return None if self.rom is None or self.stored_size is None else self.rom + self.stored_size

    @property
    def preserved(self) -> bool:
        return self.cls != EXACT

    def as_dict(self) -> dict:
        d = asdict(self)
        d["rom"] = None if self.rom is None else f"0x{self.rom:X}"
        d["end"] = None if self.end is None else f"0x{self.end:X}"
        d["preserved"] = self.preserved
        d["encoder_statement"] = encoder_statement(self.codec)
        return d


def classify_slice(stored: bytes | None, codec: str, *, rom: int | None = None,
                   expected: bytes | None = None, expected_prefix: bool = False,
                   reference: str | None = None, decoder=_DEFAULT, encoders=_DEFAULT,
                   max_output: int = compress.DEFAULT_MAX_OUTPUT, note: str = "") -> SliceClass:
    """Classify one stored slice (exactly the ROM bytes the slice owns, ``None`` when the packed
    source was not traced).

    ``expected`` is the reference payload; with ``expected_prefix`` it may be a trimmed prefix of
    the decoded payload (``image.py`` trims a loader-written segment at its BSS or trailing
    zeros): the bytes past it are ``unverified_tail``, never counted as reconstructed.
    ``decoder(stored) -> (payload, consumed)`` and ``encoders`` (``((label, fn), ...)``) default to
    the kit's codec registry; a decoder that raises fails closed (``opaque``).  An encoder must be
    a function of the payload alone: one that reads the ROM would make every slice "exact"."""
    if reference is None:
        reference = "declared" if expected is not None else "none"
    if reference not in REFERENCES:
        raise ValueError(f"reference {reference!r}: one of {', '.join(REFERENCES)}")
    if reference != "none" and expected is None:
        raise ValueError(f"reference {reference!r} needs the expected payload")
    out = SliceClass(codec, OPAQUE, "", reference, rom=rom, note=note)
    if expected is not None:
        expected = bytes(expected)
        out.expected_size = len(expected)
    if stored is None:
        out.reason = "source-not-traced"
        return out
    stored = bytes(stored)
    out.stored_size, out.stored_sha1 = len(stored), _sha1(stored)
    if decoder is _DEFAULT:
        decoder = default_decoder(codec, max_output)
    if encoders is _DEFAULT:
        encoders = default_encoders(codec)
    if decoder is None:
        out.reason = "no-decoder"
        return out
    if not stored:
        out.reason = "empty-slice"
        return out
    try:
        payload, consumed = decoder(stored)
    except Exception as error:                     # noqa: BLE001 - any decode failure fails closed
        out.reason = "decode-error"
        out.note = "; ".join(x for x in (note, f"{type(error).__name__}: {error}"[:200]) if x)
        return out
    payload = bytes(payload)
    out.payload_size, out.payload_sha1, out.consumed = len(payload), _sha1(payload), consumed
    if not payload:
        out.reason = "empty-payload"
        return out
    if not isinstance(consumed, int) or not 0 < consumed <= len(stored):
        out.reason = "bad-consumed"                # the decoder claims bytes the slice does not hold
        return out
    matched = len(payload)
    if expected is not None and payload != expected:
        if expected_prefix and expected and payload[:len(expected)] == expected:
            matched, out.unverified_tail = len(expected), len(payload) - len(expected)
        else:
            out.reason = "payload-mismatch"
            out.payload_difference = _first_difference(payload, expected)
            return out
    out.cls, out.payload_reconstructed = PAYLOAD_ONLY, matched
    encoders = tuple(encoders or ())
    if not encoders:
        out.reason = "no-encoder"
        return out
    stream = stored[:consumed]
    first = None
    stream_label = None
    for label, encode in encoders:
        out.encoders_tried += 1
        try:
            candidate = bytes(encode(payload))
        except Exception:                          # noqa: BLE001 - an encoder that fails is just tried
            continue
        if first is None:
            first = candidate
        if candidate == stored:                    # the whole slice, same length: the only exact test
            out.cls, out.reason, out.encoder = EXACT, "stored-identical", label
            break
        if stream_label is None and candidate == stream:
            stream_label = label
    if first is not None:
        out.reencoded_size = len(first)
        out.first_difference = _first_difference(first, stored)
        try:
            out.reencode_roundtrips = bytes(decoder(first)[0]) == payload
        except Exception:                          # noqa: BLE001 - reported, not raised
            out.reencode_roundtrips = False
    if out.cls == EXACT:
        return out
    if first is None:
        out.reason = "encoder-error"
    elif stream_label is not None:
        out.reason, out.encoder = "trailing-bytes", stream_label   # stream exact, slice holds more
    else:
        out.reason = "reencode-differs"
    return out


def classify_segment(rom_data: bytes, segment, *, decoder: Decoder | None = None, encoders=_DEFAULT,
                     max_output: int | None = None) -> SliceClass:
    """The real loader/codec boundary for a loader-written segment (``image.Segment``, kind
    ``packed``): its ``data`` is what the game's loader wrote in boot emulation and is the
    reference payload; the stored slice is its packed source in the ROM.

    With a kit ``codec`` the stream start is the one ``image._identify_codec`` accepts (the traced
    start or 4, 8, 16 bytes in) and the slice is exactly the bytes the decoder consumes.  Without
    one, the traced range (read-tracking granular) is classified with ``decoder`` (for example
    :func:`game_codec_decoder`), else it is ``opaque``."""
    if segment.kind != "packed":
        raise ValueError(f"segment kind {segment.kind!r}: only a loader-written (packed) segment has a codec boundary")
    expected = bytes(segment.data)
    codec = segment.codec or ("game" if segment.game_codec else "unknown")
    if segment.rom is None:
        return classify_slice(None, codec, expected=expected, reference="loader", note="packed source not traced")
    limit = max_output or len(expected) + 0x10000
    traced_end = segment.rom_end if segment.rom_end is not None else len(rom_data)
    if segment.codec:
        probe = expected[:0x100]
        for start in sorted({segment.rom, segment.rom + 4, segment.rom + 8, segment.rom + 0x10}):
            try:
                d = compress.decode(segment.codec, rom_data, start, max_output=limit)
            except Exception:                      # noqa: BLE001 - this start is not the stream
                continue
            if d.data and probe and d.data[:len(probe)] == probe:
                return classify_slice(rom_data[start:start + d.consumed], segment.codec, rom=start,
                                      expected=expected, expected_prefix=True, reference="loader",
                                      encoders=encoders, max_output=limit,
                                      note=f"stream at traced start +{start - segment.rom}")
        return classify_slice(rom_data[segment.rom:traced_end], segment.codec, rom=segment.rom, expected=expected,
                              expected_prefix=True, reference="loader", encoders=encoders, max_output=limit,
                              note="no candidate start decodes to the loader's first bytes")
    return classify_slice(rom_data[segment.rom:traced_end], codec, rom=segment.rom, expected=expected,
                          expected_prefix=True, reference="loader", decoder=decoder,
                          encoders=() if encoders is _DEFAULT else encoders,
                          note="extent from read tracking (line granular)")


# (c) Metrics: separate numerators and denominators, never one combined ratio.

def metrics(results: Iterable[SliceClass]) -> dict:
    results = list(results)
    n = len(results)
    classes = Counter(r.cls for r in results)
    reasons: dict[str, Counter] = {c: Counter() for c in CLASSES}
    for r in results:
        reasons[r.cls][r.reason] += 1
    stored_total = sum(r.stored_size or 0 for r in results)

    def stored(select) -> int:
        return sum(r.stored_size or 0 for r in results if select(r))

    by_reference = Counter()
    known = unknown = 0
    for r in results:
        by_reference[r.reference] += r.payload_reconstructed
        if r.expected_size is not None:
            known += r.expected_size
        elif r.cls != OPAQUE:
            known += r.payload_reconstructed
        else:
            unknown += 1
    return {
        "slices": n,
        "classes": {c: classes.get(c, 0) for c in CLASSES},
        "reasons": {c: dict(v) for c, v in reasons.items()},
        "exact_repack": {"slices": classes.get(EXACT, 0), "of_slices": n,
                         "stored_bytes": stored(lambda r: r.cls == EXACT), "of_stored_bytes": stored_total},
        "preserved": {"slices": n - classes.get(EXACT, 0), "of_slices": n,
                      "stored_bytes": stored(lambda r: r.preserved), "of_stored_bytes": stored_total,
                      "untraced_slices": sum(1 for r in results if r.rom is None or r.stored_size is None)},
        "payload": {"bytes_reconstructed": sum(r.payload_reconstructed for r in results),
                    "of_known_payload_bytes": known,
                    "slices_reconstructed": classes.get(EXACT, 0) + classes.get(PAYLOAD_ONLY, 0), "of_slices": n,
                    "slices_payload_size_unknown": unknown,
                    "unverified_tail_bytes": sum(r.unverified_tail for r in results),
                    "bytes_by_reference": {ref: by_reference.get(ref, 0) for ref in REFERENCES}},
    }


def check_disjoint(results: Iterable[SliceClass]) -> None:
    """A ROM byte belongs to at most one slice."""
    spans = sorted((r.rom, r.end) for r in results if r.end is not None)
    for (a, b), (c, d) in zip(spans, spans[1:]):
        if c < b:
            raise ValueError(f"slices 0x{a:X}..0x{b:X} and 0x{c:X}..0x{d:X} overlap")


def manifest(results: Iterable[SliceClass]) -> dict:
    """What a cartridge build and its verifier consume: slices that may be regenerated from their
    payload with a named encoder, and slices that are preserved inputs (cut from the ROM as
    declared local extraction inputs, identified by hash)."""
    results = list(results)
    check_disjoint(results)

    def row(r: SliceClass) -> dict:
        return {"rom": f"0x{r.rom:X}", "end": f"0x{r.end:X}", "size": r.stored_size, "codec": r.codec,
                "class": r.cls, "reason": r.reason, "stored_sha1": r.stored_sha1}
    repack = [dict(row(r), encoder=r.encoder, payload_sha1=r.payload_sha1, payload_size=r.payload_size)
              for r in results if r.cls == EXACT]
    preserved = [row(r) for r in results if r.preserved and r.end is not None]
    untraced = [{"codec": r.codec, "reason": r.reason, "expected_size": r.expected_size}
                for r in results if r.end is None]
    return {"repack": sorted(repack, key=lambda x: int(x["rom"], 16)),
            "preserved": sorted(preserved, key=lambda x: int(x["rom"], 16)),
            "untraced": untraced, "strict": False,
            "note": "a measurement: a preserved slice is ROM bytes, never reconstruction; exact repack "
                    "is not C credit"}
