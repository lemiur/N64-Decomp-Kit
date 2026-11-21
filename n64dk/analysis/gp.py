"""The program's ``_gp`` (small-data base) from independent evidence in its own code.

Code built with a nonzero ``-G`` addresses small globals relative to ``$gp``, and its startup code
(and often each thread entry) sets ``$gp`` with ``lui $gp, %hi(_gp); addiu $gp, $gp, %lo(_gp)``.
The value is established here only from such setting sites in catalogued functions, never from the
function being verified:

* every ``lui``/``addiu`` (or ``ori``) pair into ``$gp`` within four instructions of each other, in
  one function and with no other write to ``$gp`` between them, forms a value; a ``lui $gp`` alone
  (``li`` of a value whose low half is zero) forms one when nothing writes ``$gp`` within its four
  instructions or later in its function and the value lies in RDRAM; all setting sites must agree
  (a disagreement is reported, and no value is given);
* the value must be nonzero (a program that clears ``$gp`` does not use small data) and lie in
  KSEG0 RDRAM (a pair outside it is data decoded as code, or a mapping this module does not
  model: no value is given);
* GP-relative uses (loads, stores, ``addiu`` with base ``$gp``) are counted per image; an image that
  uses ``$gp`` without setting it (an overlay linked against the program) inherits the program's
  value, which is recorded as such.

``find(...)`` returns a record with the value, its sites, the uses per image and a confidence:
``high`` (set at two or more sites, or at one site with at least 16 uses in at least 4 functions:
data mis-catalogued as one function decodes as uses too), ``low`` otherwise."""
from __future__ import annotations

import struct

GP = 28
USE_OPS = (9, 32, 33, 35, 36, 37, 39, 40, 41, 43, 49, 53, 57, 61)     # addiu, loads, stores
MIN_USES_ONE_SITE = 16
MIN_FUNCTIONS_ONE_SITE = 4
# I-type opcodes that write rt: addi..lui, daddi, daddiu, ldl, ldr, loads, ll, lld, ld, sc, scd
WRITE_RT_OPS = frozenset((8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39,
                          48, 52, 55, 56, 60))
# SPECIAL functs that write no rd: jr, syscall, break, sync, mthi, mtlo, mult/div, traps
NO_RD_FUNCTS = frozenset((8, 12, 13, 15, 17, 19, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 54))
RDRAM_GP = (0x80000000, 0x80808000)          # 8 MiB of KSEG0, plus the 0x7FF0 _gp may lie past .sdata


def plausible(value: int) -> bool:
    return RDRAM_GP[0] <= value < RDRAM_GP[1]


def writes_gp(word: int) -> bool:
    """The instruction writes ``$gp``."""
    op, rs, rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
    if op == 0:
        return (word >> 11) & 31 == GP and word & 0x3F not in NO_RD_FUNCTS
    if op in (16, 17, 18):                                   # mfc/dmfc/cfc
        return rs in (0, 1, 2) and rt == GP
    return op in WRITE_RT_OPS and rt == GP


def _words(data: bytes, delta: int, rows):
    for _name, vram, size in rows:
        start = vram - delta
        if start < 0 or start + size > len(data):
            continue
        yield [(vram + (offset - start), struct.unpack_from(">I", data, offset)[0])
               for offset in range(start, start + size - 3, 4)]


def scan(data: bytes, delta: int, rows) -> dict:
    """Setting sites (address -> value), GP-relative uses and the functions with any, in one image."""
    sites, uses, functions = {}, 0, 0
    for function in _words(data, delta, rows):
        pending = None          # a lui $gp still open for its low half: (address, high half)
        alone = None            # the last lui $gp, while nothing else has written $gp since
        used = 0
        for address, word in function:
            op, rs, rt, imm = word >> 26, (word >> 21) & 31, (word >> 16) & 31, word & 0xFFFF
            if pending and address - pending[0] > 16:
                pending = None                               # its four instructions passed alone
            if op == 15 and rt == GP:
                pending = alone = (address, imm << 16)
                continue
            if pending and op in (9, 13) and rt == GP and rs == GP:
                low = (imm - 0x10000 if imm & 0x8000 else imm) if op == 9 else imm
                sites[address] = (pending[1] + low) & 0xFFFFFFFF
                pending = alone = None
                continue
            if writes_gp(word):
                # rewritten: a pair across it would not give $gp's value, nor a lui before it
                pending = alone = None
                continue
            if rs == GP and op in USE_OPS:
                used += 1
        # a lui alone counts only when its four instructions passed inside this function (a split
        # catalogue may hold its low half in the next row) and its value is an RDRAM address
        if alone and not pending and plausible(alone[1]):
            sites[alone[0]] = alone[1]
        uses += used
        functions += bool(used)
    return {"sites": sites, "uses": uses, "functions": functions}


def find(images: dict) -> dict:
    """``images``: name -> (data, delta, rows), the program's main image first.  Returns
    ``{"value", "confidence", "sites", "uses", "inherited", "reason"}``; ``value`` is None when the
    evidence does not establish one."""
    scans = {name: scan(*spec) for name, spec in images.items()}
    sites = {f"{name}:0x{address:08X}": value for name, found in scans.items()
             for address, value in found["sites"].items()}
    uses = {name: found["uses"] for name, found in scans.items() if found["uses"]}
    record = {"value": None, "confidence": None, "sites": {k: f"0x{v:08X}" for k, v in sites.items()},
              "uses": uses, "inherited": [], "reason": ""}
    values = set(sites.values())
    if not values:
        record["reason"] = "no code sets $gp" + (" although some uses it" if uses else "")
        return record
    if len(values) > 1:
        record["reason"] = "setting sites disagree: " + ", ".join(f"0x{v:08X}" for v in sorted(values))
        return record
    value = values.pop()
    if value == 0:
        record["reason"] = "the program clears $gp (no small data)"
        return record
    if not plausible(value):
        record["reason"] = f"$gp is set to 0x{value:08X}, outside KSEG0 RDRAM (data decoded as code?)"
        return record
    total = sum(uses.values())
    functions = sum(found["functions"] for found in scans.values())
    record["value"] = f"0x{value:08X}"
    record["confidence"] = ("high" if len(sites) >= 2 or (total >= MIN_USES_ONE_SITE and functions >= MIN_FUNCTIONS_ONE_SITE)
                            else "low")
    record["inherited"] = [name for name, found in scans.items() if found["uses"] and not found["sites"]]
    record["reason"] = (f"set to 0x{value:08X} at {len(sites)} site(s); {total} GP-relative uses in {functions} function(s)"
                        + (f"; inherited by {', '.join(record['inherited'])}" if record["inherited"] else ""))
    return record


def for_project(project) -> dict:
    """``find`` over a project's images (main first)."""
    from ..match import images as imagemod
    registry = imagemod.load_registry(project)
    order = sorted(registry, key=lambda name: (name != "main", name))
    images = {}
    for name in order:
        try:
            spec = imagemod.resolve(project, name)
            images[name] = (spec.load(), spec.delta, spec.rows())
        except Exception:                                    # noqa: BLE001 - an image that cannot load gives no evidence
            continue
    return find(images)


def symbol_line(record: dict) -> str | None:
    """``_gp = 0x...;`` for a symbol file, with its evidence, when a value is established."""
    if not record.get("value") or record.get("confidence") != "high":
        return None
    return f"_gp = {record['value']}; // small-data base: {record['reason']}\n"
