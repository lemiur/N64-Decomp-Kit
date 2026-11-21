"""Debug and development-record archaeology beyond SN/PsyQ objects (round 12, item I).

``devrecords`` decodes SN/PsyQ objects and function records.  This module adds the other record
families a retail N64 ROM can keep, each scanner fail-closed: a candidate is reported only when
its structure is internally consistent, with the reasons for every rejection counted, the offset
of the record and the SHA-256 of its bytes.

``ecoff``   ECOFF symbolic headers (``HDRR``, magic 0x7009, the ``.mdebug`` section of IDO and
            GNU MIPS objects and executables), big- or little-endian: counts and offsets must be
            sane and, where the header's tables lie inside the scanned bytes, the local string
            table must hold identifiers.
``stabs``   runs of STABS entries (12-byte ``n_strx, n_type, n_other, n_desc, n_value``) from
            ``gcc -g``: a run is at least ``minimum`` consecutive entries of known types with
            ``n_other == 0`` and string offsets inside a bounded table.
``build_strings``  compiler and tool identifications (``GCC: (GNU)``, ``cc1``, ``IDO``, ``MIPSpro``,
            ``SN Systems``/PsyQ, KMC, ``@(#)`` and ``$Id`` strings), build dates and times in
            ``__DATE__``/``__TIME__`` form, assertion messages and printf-style format strings.

Recovered facts are evidence with provenance.  They never override byte evidence and are never
applied to a catalogue or symbol file by this module.
"""
from __future__ import annotations

import hashlib
import re
import struct

ECOFF_MAGIC = 0x7009
HDRR_SIZE = 0x60
HDRR_FIELDS = ("magic", "vstamp", "ilineMax", "cbLine", "cbLineOffset", "idnMax", "cbDnOffset", "ipdMax",
               "cbPdOffset", "isymMax", "cbSymOffset", "ioptMax", "cbOptOffset", "iauxMax", "cbAuxOffset",
               "issMax", "cbSsOffset", "issExtMax", "cbSsExtOffset", "ifdMax", "cbFdOffset", "crfd",
               "cbRfdOffset", "iextMax", "cbExtOffset")
# (count field, offset field, entry size) of the tables a HDRR describes
TABLES = (("ipdMax", "cbPdOffset", 52), ("isymMax", "cbSymOffset", 12), ("iauxMax", "cbAuxOffset", 4),
          ("issMax", "cbSsOffset", 1), ("issExtMax", "cbSsExtOffset", 1), ("ifdMax", "cbFdOffset", 72),
          ("iextMax", "cbExtOffset", 16))
COUNT_LIMIT = 1 << 22

STAB_TYPES = {0x20: "GSYM", 0x22: "FNAME", 0x24: "FUN", 0x26: "STSYM", 0x28: "LCSYM", 0x2E: "BNSYM",
              0x3C: "OPT", 0x40: "RSYM", 0x44: "SLINE", 0x4E: "ENSYM", 0x60: "SSYM", 0x64: "SO",
              0x80: "LSYM", 0x82: "BINCL", 0x84: "SOL", 0xA0: "PSYM", 0xA2: "EINCL", 0xC0: "LBRAC",
              0xE0: "RBRAC", 0xE2: "BCOMM", 0xE4: "ECOMM", 0x00: "UNDF"}

TOOL_PATTERNS = (
    ("gcc", rb"GCC: \([^)\x00]{1,40}\)[ -~]{0,40}"),
    ("gcc", rb"(?:gcc|cc1|egcs)[ -]?(?:version )?[0-9]\.[0-9][0-9.]*[ -~]{0,30}"),
    ("ido", rb"(?<![A-Za-z])(?:IDO|MIPSpro|Compiler Development Environment)(?![A-Za-z])[ -~]{0,40}[0-9]\.[0-9][ -~]{0,20}"),
    ("sn", rb"(?:SN Systems|PsyQ|Psy-Q|ccn64|asn64|psylink)[ -~]{0,40}"),
    # a bare "KMC" is three letters any table can hold: a tool name in context only
    ("kmc", rb"(?:Kyoto Micro[ -~]{0,40}|(?<![A-Za-z])KMC[ _-]?(?:GCC|gcc|cc1|as|ld|[Vv]er(?:sion)?)[ -~]{0,40})"),
    ("sccs", rb"@\(#\)[ -~]{2,80}"),
    ("rcs", rb"\$(?:Id|Revision|Header|Date|Author): [ -~]{2,80}\$"),
    ("libultra", rb"(?:libultra|Ultra64|N64 OS)[ -~]{0,40}"),
)
DATE = rb"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [ 0-3][0-9] (?:199[4-9]|200[0-2])"
TIME = rb"[0-2][0-9]:[0-5][0-9]:[0-5][0-9]"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hdrr(data: bytes, offset: int, endian: str) -> dict:
    values = struct.unpack_from(endian + "hH" + "i" * 23, data, offset)
    return dict(zip(HDRR_FIELDS, values))


def _strings_at(data: bytes, start: int, size: int, limit: int = 64) -> list[str]:
    table = data[start:start + min(size, 1 << 16)]
    out = []
    for raw in table.split(b"\x00"):
        if raw:
            out.append(raw.decode("latin-1"))
        if len(out) >= limit:
            break
    return out


def _identifier_share(strings: list[str]) -> float:
    if not strings:
        return 0.0
    ok = sum(bool(re.fullmatch(r"[A-Za-z_.$][A-Za-z0-9_.$/:\-]{0,120}", s)) for s in strings)
    return ok / len(strings)


def ecoff(data: bytes, limit: int = 64) -> dict:
    """Candidate ECOFF symbolic headers.  ``{"found": [...], "rejected": {reason: n}}``.

    A header's table offsets are file offsets of the object or executable it came from; when the
    header was copied into a ROM together with its tables, the tables lie at the same distances
    from the header as in the file, so each candidate is also tested with the file's start placed
    where the header's own offsets put it (``file_start``): the external string table must then
    hold identifiers.  Without that, the header is reported as ``tables: outside`` (evidence of a
    debug build, nothing more)."""
    found, rejected = [], {}

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    magics = (b"\x70\x09", b"\x09\x70")
    for magic, endian in zip(magics, (">", "<")):
        start = 0
        while len(found) < limit:
            at = data.find(magic, start)
            if at < 0:
                break
            start = at + 2
            if at % 4 or at + HDRR_SIZE > len(data):
                reject("misaligned or truncated")
                continue
            h = _hdrr(data, at, endian)
            counts = [h[c] for c, _o, _s in TABLES] + [h["ilineMax"], h["cbLine"], h["crfd"]]
            if any(c < 0 or c > COUNT_LIMIT for c in counts):
                reject("count out of range")
                continue
            if not (h["isymMax"] or h["iextMax"] or h["ipdMax"]):
                reject("no symbols, procedures or externals")
                continue
            offsets = [(h[o], h[c] * size) for c, o, size in TABLES if h[c]]
            if any(off <= 0 for off, _n in offsets):
                reject("table offset not positive")
                continue
            spans = sorted(offsets)
            if any(a_off + a_len > b_off for (a_off, a_len), (b_off, _bl) in zip(spans, spans[1:])):
                reject("tables overlap")
                continue
            entry = {"offset": at, "endian": "big" if endian == ">" else "little", "vstamp": h["vstamp"],
                     "counts": {c: h[c] for c, _o, _s in TABLES}, "sha256": _sha(data[at:at + HDRR_SIZE])}
            # the header follows its tables or precedes them; try the file start that puts the
            # external string table right after the header (the common .mdebug order) and one that
            # puts the header at the file start
            for file_start in sorted({at - min(o for o, _n in offsets) + HDRR_SIZE, at}):
                strings = []
                if h["issExtMax"] and file_start >= 0 and file_start + h["cbSsExtOffset"] < len(data):
                    strings = _strings_at(data, file_start + h["cbSsExtOffset"], h["issExtMax"])
                elif h["issMax"] and file_start >= 0 and file_start + h["cbSsOffset"] < len(data):
                    strings = _strings_at(data, file_start + h["cbSsOffset"], h["issMax"])
                if strings and _identifier_share(strings) >= 0.8:
                    entry.update(tables="inside", file_start=file_start, names=strings[:32])
                    break
            else:
                entry["tables"] = "outside"
            found.append(entry)
    return {"found": found, "rejected": rejected}


def stabs(data: bytes, minimum: int = 8, limit: int = 32) -> dict:
    """Runs of STABS entries.  ``{"found": [{offset, endian, entries, types, sources, functions,
    sha256}], "rejected": {...}}``; ``sources`` and ``functions`` are string-table offsets'
    strings when a string table follows the run (``.stabstr`` usually does)."""
    found, rejected = [], {}
    for endian in (">", "<"):
        i = 0
        end = len(data) - 12
        while i <= end and len(found) < limit:
            n = 0
            types = {}
            strx_max = 0
            j = i
            while j <= end:
                if data[j + 5] != 0 or data[j + 4] not in STAB_TYPES:      # cheap test first
                    break
                strx, typ, _other, _desc, _value = struct.unpack_from(endian + "IBBhI", data, j)
                if strx > (1 << 20):
                    break
                types[STAB_TYPES[typ]] = types.get(STAB_TYPES[typ], 0) + 1
                strx_max = max(strx_max, strx)
                n += 1
                j += 12
            # a run of zeros decodes as UNDF entries: require real structure
            if n >= minimum and len(types) >= 3 and ("SO" in types or "FUN" in types) and strx_max:
                run_end = j
                strings = _strings_at(data, run_end, strx_max + 256, limit=512)
                entry = {"offset": i, "endian": "big" if endian == ">" else "little", "entries": n,
                         "types": types, "sha256": _sha(data[i:run_end])}
                if strings and _identifier_share([s.split(":")[0] for s in strings[:64]]) >= 0.6:
                    entry["sources"] = [s for s in strings if s.endswith((".c", ".s", ".h"))][:32]
                    entry["functions"] = sorted({s.split(":")[0] for s in strings if ":F" in s or ":f" in s})[:64]
                if entry.get("sources") or entry.get("functions"):
                    found.append(entry)
                else:                                    # entries without a string table that names
                    rejected["no string table"] = rejected.get("no string table", 0) + 1   # anything
                i = run_end
                continue
            else:
                if n >= minimum:                     # e.g. zero padding (UNDF entries): skip it whole,
                    rejected["structureless run"] = rejected.get("structureless run", 0) + 1
                    i = j                                # never rescan it from every word inside
                else:
                    i += 4
    return {"found": found, "rejected": rejected}


def build_strings(data: bytes, limit: int = 200) -> dict:
    """Tool identifications, build dates/times, assertion messages and format strings, each with
    its offset.  Counts are complete; lists are capped at ``limit``."""
    tools = []
    for kind, pattern in TOOL_PATTERNS:
        for m in re.finditer(pattern, data):
            text = m.group(0).decode("latin-1").strip()
            tools.append({"kind": kind, "offset": m.start(), "text": text[:100]})
    tools.sort(key=lambda t: t["offset"])
    dates = [{"offset": m.start(), "text": m.group(0).decode()} for m in re.finditer(DATE, data)]
    times = [{"offset": m.start(), "text": m.group(0).decode()} for m in re.finditer(TIME, data)]
    asserts = [{"offset": m.start(), "text": m.group(0).decode("latin-1")[:100]}
               for m in re.finditer(rb"(?:[Aa]ssert(?:ion)?(?: failed)?|ASSERT)[ -~]{0,80}", data)]
    formats = re.findall(rb"[ -~]{0,60}%(?:[-0-9.]*[dxXsfuclp])[ -~]{0,60}\x00", data)
    return {"tools": tools[:limit], "tool_kinds": sorted({t["kind"] for t in tools}),
            "dates": dates[:limit], "times": times[:limit], "assertions": asserts[:limit],
            "counts": {"tools": len(tools), "dates": len(dates), "times": len(times),
                       "assertions": len(asserts), "format_strings": len(formats)}}


def scan(data: bytes) -> dict:
    """Every scanner over one blob (a ROM or a decompressed image)."""
    return {"ecoff": ecoff(data), "stabs": stabs(data), "strings": build_strings(data)}
