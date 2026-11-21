"""What kind of N64 medium an image is, what the kit can do with it, and why not when it cannot.

``capability_report(path_or_bytes)`` reads at most the first 0x101000 bytes (header, IPL3 and the
checksummed span) plus the file size, and returns one JSON-able report with stable keys.
``require_supported`` raises ``UnsupportedMedia`` carrying that report when the kit cannot take the
image; ``survey_hook`` is the call a survey makes before ``Rom.load``.

Known (encoded from the kit's own tables or the cartridge header layout ``n64dk.rom`` reads):

* byte order: z64 / v64 / n64 by the first word (``rom.to_big_endian``); any other first word is
  unsupported, with the word and the nearest known order in the report;
* boot code: the CIC family from the CRC32 of IPL3 (``rom.CIC_BY_IPL3_CRC``) and, when known, whether
  the header checksum matches (``rom.compute_checksum``);
* header fields: title, category code (byte 0x3B), game id, country code (0x3E), version.

By convention (recorded as such in each field's ``basis``):

* video standard from the country code: the code names the market; the game reads ``osTvType`` at
  run time, so this is the market's standard, not a property of the code;
* category codes N (Game Pak), D (64DD disk), C / E (the Game Pak / 64DD parts of an expandable
  game), Z (Aleck64 arcade board);
* NTSC and PAL boot chips of one family (6102/7101, 6103/7103, 6105/7105, 6106/7106) share one IPL3,
  so the IPL3 names the family, reported under its NTSC name (7102 has its own IPL3 in rom.py);
* the "advanced homebrew" header convention (game id ``ED``, save type in the high nibble of byte
  0x3F) used by flash-cart loaders: a save-type *hint*, never a fact about the code.

Assumed, not verified in this lab (no image of the kind exists here):

* 64DD disk images are recognised by size only: 0x3DEC800 bytes (retail ``.ndd`` dumps) and
  0x435B0C0 bytes (development disks).  The system area is not read: its block layout is not encoded
  in the kit and could not be checked against a reference image, so no signature is claimed.
  Every disk image is unsupported (the kit reads cartridge images only).
* iQue Player content: no reliable static indicator is known to the kit.  ``ique.detected`` is
  always False.  Weak indicators are listed only: a header whose title, category, game id and
  country code are all blank (that is how the public Dr. Mario 64 decompilation writes its iQue
  build's header, ``src/rom_header/rom_header.c``), country code ``C`` (the China convention; not
  what that source writes) and, with either, a non-retail IPL3.

Not determinable from the header: the save type of a retail image (emulators use a game database;
the kit ships none) and Expansion Pak use (``osMemSize`` checks are code, not header).
"""
from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

from . import rom as rommod

SCHEMA = 1
PREFIX = rommod.CHECKSUM_START + rommod.CHECKSUM_LENGTH          # 0x101000: header + IPL3 + checksummed MiB
MAGIC = {b"\x80\x37\x12\x40": "z64", b"\x37\x80\x40\x12": "v64", b"\x40\x12\x37\x80": "n64"}
DISK_SIZES = {0x3DEC800: "retail 64DD disk image (.ndd dump size)",
              0x435B0C0: "development 64DD disk image (size)"}
MAX_CARTRIDGE = 64 * 1024 * 1024

COUNTRY = {  # code: (market, video standard)
    "7": ("beta", "unknown"), "A": ("Asia", "NTSC"), "B": ("Brazil", "PAL-M (60 Hz)"),
    "C": ("China (iQue)", "unknown"), "D": ("Germany", "PAL"), "E": ("North America", "NTSC"),
    "F": ("France", "PAL"), "G": ("Gateway 64 (NTSC)", "NTSC"), "H": ("Netherlands", "PAL"),
    "I": ("Italy", "PAL"), "J": ("Japan", "NTSC"), "K": ("Korea", "NTSC"), "L": ("Gateway 64 (PAL)", "PAL"),
    "N": ("Canada", "NTSC"), "P": ("Europe", "PAL"), "S": ("Spain", "PAL"), "U": ("Australia", "PAL"),
    "W": ("Scandinavia", "PAL"), "X": ("Europe", "PAL"), "Y": ("Europe", "PAL"), "Z": ("Europe", "PAL"),
}
CATEGORY = {"N": "Game Pak", "D": "64DD disk", "C": "expandable game: Game Pak part",
            "E": "expandable game: 64DD disk part", "Z": "Aleck64 arcade board"}
HOMEBREW_SAVE = {0: "none", 1: "EEPROM 4 Kbit", 2: "EEPROM 16 Kbit", 3: "SRAM 256 Kbit",
                 4: "SRAM 768 Kbit (banked)", 5: "FlashRAM 1 Mbit", 6: "SRAM 1 Mbit"}


class UnsupportedMedia(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__("unsupported N64 image: " + "; ".join(report["unsupported"]))


def _read(path_or_bytes) -> tuple[bytes, int, str]:
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        data = bytes(path_or_bytes)
        return data[:PREFIX], len(data), "<bytes>"
    path = Path(path_or_bytes)
    size = os.stat(path).st_size
    with open(path, "rb") as stream:
        return stream.read(PREFIX), size, path.name


def _nearest_order(word: bytes) -> str | None:
    """A known order whose magic shares the most bytes with ``word`` (for the report only)."""
    best = max(MAGIC.items(), key=lambda item: sum(a == b for a, b in zip(item[0], word)))
    return best[1] if sum(a == b for a, b in zip(best[0], word)) >= 2 else None


def _header(be: bytes) -> dict:
    pi, clock, entry, release, crc1, crc2 = struct.unpack_from(">IIIIII", be, 0)
    code = be[0x3B:0x3F].decode("ascii", "replace")
    return {"pi_config": f"0x{pi:08X}", "clock_rate": f"0x{clock:08X}", "entry": f"0x{entry:08X}",
            "release": f"0x{release:08X}", "crc": f"{crc1:08X}/{crc2:08X}",
            "title": be[0x20:0x34].decode("ascii", "replace").rstrip(" \0"),
            "category_code": code[0], "game_id": code[1:3], "country_code": code[3], "version": be[0x3F],
            "_crc": (crc1, crc2)}


def capability_report(path_or_bytes, *, checksum: bool = True) -> dict:
    head, size, source = _read(path_or_bytes)
    report = {"schema": SCHEMA, "source": source, "size": size, "media": "unknown", "supported": False,
              "unsupported": [], "warnings": [], "byte_order": None, "header": None,
              "cic": {"value": None, "basis": "not read"}, "checksum_ok": None,
              "video": {"standard": "unknown", "basis": "no header"},
              "media_category": None,
              "save_type": {"value": "unknown", "basis": "no header"},
              "expansion_pak": {"value": "not determinable statically",
                                "basis": "use of the extra 4 MiB is decided by code (osMemSize), not the header"},
              "ique": {"detected": False, "status": "not detected: unsupported", "weak_indicators": []},
              "homebrew": {"indicators": []},
              "capabilities": {"survey": False, "match": False, "boot_entry": None}}
    if size in DISK_SIZES:
        report["media"] = "64dd-disk"
        report["unsupported"].append(
            f"{DISK_SIZES[size]}, recognised by size 0x{size:X} only (system area not read): the kit reads "
            "cartridge images only; 64DD disk layout, zones and system area are not implemented")
        return report
    word = head[:4]
    order = MAGIC.get(word)
    if order is None:
        nearest = _nearest_order(word) if len(word) == 4 else None
        report["unsupported"].append(
            f"unrecognised first word {word.hex() or '(empty)'}: the kit accepts cartridge images in z64 "
            f"(80371240), v64 (37804012) or n64 (40123780) order" +
            (f"; it resembles {nearest} order with a non-standard PI configuration word" if nearest else ""))
        if size > MAX_CARTRIDGE:
            report["unsupported"].append(f"{size} bytes: larger than any cartridge (64 MiB) and not a known 64DD size")
        return report
    report["media"] = "cartridge"
    report["byte_order"] = order
    usable = len(head) // 4 * 4
    be, _order = rommod.to_big_endian(head[:usable])
    if len(be) < 0x40:
        report["unsupported"].append(f"{size} bytes: shorter than the 64-byte cartridge header")
        return report
    header = _header(be)
    crc = header.pop("_crc")
    report["header"] = header
    category = header["category_code"]
    report["media_category"] = {"code": category, "meaning": CATEGORY.get(category, "unrecognised"),
                                "basis": "header byte 0x3B (convention)"}
    country = header["country_code"]
    market, standard = COUNTRY.get(country, (None, "unknown"))
    report["video"] = {"standard": standard, "country_code": country, "market": market,
                       "basis": "header country code (convention: the game reads osTvType at run time)"}
    # boot code
    if len(be) >= 0x1000:
        cic = rommod.CIC_BY_IPL3_CRC.get(zlib.crc32(be[0x40:0x1000]))
        report["cic"] = {"value": cic, "basis": "CRC32 of IPL3 (0x40..0x1000) against rom.CIC_BY_IPL3_CRC" +
                         ("; the NTSC name of the family (the PAL chip shares the IPL3)" if cic and cic != "7102" else "")}
        if cic and checksum and len(be) >= PREFIX:
            report["checksum_ok"] = rommod.compute_checksum(be, cic) == crc
        if cic is None:
            report["homebrew"]["indicators"].append(
                "IPL3 is not a retail boot code: homebrew boot code (an open IPL3 replacement), a development "
                "cartridge, an iQue image or a damaged dump")
        elif report["checksum_ok"] is False:
            report["homebrew"]["indicators"].append("header checksum does not match the boot code: a modified image")
    # save type: only the homebrew header convention says anything
    if header["game_id"] == "ED":
        report["save_type"] = {"value": HOMEBREW_SAVE.get(header["version"] >> 4, "unrecognised"),
                               "basis": "advanced homebrew header (game id ED, byte 0x3F high nibble): a hint "
                                        "for loaders, assumed convention, not verified against code"}
        report["homebrew"]["indicators"].append("advanced homebrew header (game id ED)")
    else:
        report["save_type"] = {"value": "unknown", "basis": "retail headers do not record the save type; "
                                                           "the kit ships no game-code database"}
    blank = (not header["title"].strip(" \0") and category in ("\0", " ") and
             not header["game_id"].strip(" \0") and country in ("\0", " "))
    if not header["title"].strip(" \0") or not header["game_id"].strip(" \0"):
        report["homebrew"]["indicators"].append("blank title or game id (also how iQue builds are written)")
    # iQue: weak indicators only
    if blank:
        report["ique"]["weak_indicators"].append(
            "title, category, game id and country code all blank (the header of the public Dr. Mario 64 "
            "iQue source is written this way)")
    if country == "C":
        report["ique"]["weak_indicators"].append("country code C (China; a convention, not seen in the iQue source)")
    if report["ique"]["weak_indicators"] and report["cic"]["value"] is None:
        report["ique"]["weak_indicators"].append("non-retail IPL3")
    # what the kit can do
    if size < PREFIX:
        report["unsupported"].append(
            f"{size} bytes: smaller than header + IPL3 + the 1 MiB checksum span (0x{PREFIX:X}); n64dk.rom.Rom "
            "refuses it (pad the image to at least that size to analyse it)")
    if size > MAX_CARTRIDGE:
        report["warnings"].append(f"{size} bytes: larger than retail cartridges (64 MiB)")
    if category in ("D", "E"):
        report["unsupported"].append(f"category code {category} ({CATEGORY[category]}): a 64DD medium in "
                                     "cartridge byte order is not something the kit can analyse")
    if category == "C":
        report["warnings"].append("expandable game: only the Game Pak part is analysed; its 64DD disk part is not supported")
    if category == "Z":
        report["warnings"].append("Aleck64 arcade board: hardware beyond the N64 is not modelled")
    if category not in CATEGORY:
        report["warnings"].append(f"category code {category!r} is not a retail code")
    report["supported"] = not report["unsupported"]
    report["capabilities"] = {
        "survey": report["supported"], "match": report["supported"],
        "boot_entry": (None if not report["supported"] else
                       "known IPL3: entry relocation applied" if report["cic"]["value"] else
                       "unknown IPL3: the header entry is used unrelocated (assumed)")}
    return report


def require_supported(path_or_bytes, **kwargs) -> dict:
    report = capability_report(path_or_bytes, **kwargs)
    if not report["supported"]:
        raise UnsupportedMedia(report)
    return report


def survey_hook(path) -> dict:
    """For ``survey`` before ``Rom.load``: the report, or ``UnsupportedMedia`` with it."""
    return require_supported(path)
