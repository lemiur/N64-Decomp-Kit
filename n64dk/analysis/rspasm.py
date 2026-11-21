"""RSP microcode round trip: IMEM/DMEM extraction -> disassembly -> assembly -> byte comparison.

Report only (wave 4b, item 13; closeout directive section 6).  Nothing here changes a unit class, a
match, a coverage denominator or a credit: the output says how much of an RSP block the kit can
write back as instructions and proves that the written text assembles to the same bytes.

Text format (the assembler accepts exactly what the disassembler writes)
-------------------------------------------------------------------------
* One statement per line; ``#`` starts a comment.  ``.org N`` sets the IMEM/DMEM address of the next
  byte (no fill); ``.word W[, W...]`` and ``.byte B[, B...]`` emit raw big-endian data.
* Scalar registers are ``$0``..``$31``; COP0 registers ``$c0``..``$c15``; COP2 control registers
  ``$vco``, ``$vcc``, ``$vce``; vector registers ``$v0``..``$v31``.
* Vector compute ops: ``vmulf $vd, $vs, $vt[e]``.  The element is the hardware's 4-bit field in its
  usual names: none (0), ``[e1]`` (1), ``[0q] [1q]`` (2, 3), ``[0h]``..``[3h]`` (4..7), ``[0]``..``[7]``
  (8..15); ``[eN]`` is accepted for any N.  Single-lane ops: ``vrcp $vd[de], $vt[e]`` with ``de``
  0..7 (the ``vs`` field).  ``vnop`` has no operands.
* ``mfc2/mtc2 $rt, $vs[b]``: ``b`` is the byte element 0..15.  LWC2/SWC2: ``lqv $vt[b], OFF($base)``
  with ``b`` the byte element and ``OFF`` in bytes (the 7-bit field times the access size).
* Branch targets are absolute addresses in the listing's address space (``.org`` + 4 * index),
  printed unmasked so every 16-bit offset is written back exactly; the hardware uses the low 12 bits.
  ``j``/``jal`` print the full 28-bit target (``index << 2``); the RSP uses ``target & 0xFFC``.

Decoding rule
-------------
A word is written as an instruction only if re-assembling that text gives the same word.  Anything
else -- a CPU-only opcode (64-bit, multiply/divide, FPU, LL/SC, likely branches, unaligned loads,
``cache``), an undocumented vector function, a reserved LWC2/SWC2 code, a COP0 register above 15, a
non-zero unused field -- is kept as ``.word 0x........`` with its reason in the comment.  DMEM data is
always written as ``.word``/``.byte``.  ``.word`` bytes are *raw preservation*: identical by
construction, not decoded.

Encodings: the public RSP opcode tables (n64brew "Reality Signal Processor/CPU Core"; the opcode
lists in the GhostlyDark/rsp README, octal; ares' n64/rsp/disassembler.cpp for offset scales).
No microcode bytes are held here.

``python -m n64dk.analysis.rspasm ROM START END [--kind text|data] [--origin N]`` prints a listing
(assemblable as is) and a ``# {json}`` summary line; ``... census DIR... [--out F] [--listings D]``
extracts every ``rspid`` block of each decompile-init project and round-trips it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

from .. import rom as rommod
from . import rspid

MASK = 0xFFFFFFFF
IMEM_SIZE = 0x1000


class AsmError(ValueError):
    """A statement the assembler does not accept."""


class _Raw(Exception):
    """A word kept as ``.word``; the message is the reason."""


def _raw(reason: str):
    raise _Raw(reason)


# --------------------------------------------------------------------------- tables

SPECIAL = {0x00: "sll", 0x02: "srl", 0x03: "sra", 0x04: "sllv", 0x06: "srlv", 0x07: "srav",
           0x08: "jr", 0x09: "jalr", 0x0D: "break", 0x20: "add", 0x21: "addu", 0x22: "sub",
           0x23: "subu", 0x24: "and", 0x25: "or", 0x26: "xor", 0x27: "nor", 0x2A: "slt", 0x2B: "sltu"}
REGIMM = {0x00: "bltz", 0x01: "bgez", 0x10: "bltzal", 0x11: "bgezal"}
PRIMARY = {0x02: "j", 0x03: "jal", 0x04: "beq", 0x05: "bne", 0x06: "blez", 0x07: "bgtz",
           0x08: "addi", 0x09: "addiu", 0x0A: "slti", 0x0B: "sltiu", 0x0C: "andi", 0x0D: "ori",
           0x0E: "xori", 0x0F: "lui", 0x20: "lb", 0x21: "lh", 0x23: "lw", 0x24: "lbu", 0x25: "lhu",
           0x28: "sb", 0x29: "sh", 0x2B: "sw"}
COP0 = {0x00: "mfc0", 0x04: "mtc0"}
COP2 = {0x00: "mfc2", 0x02: "cfc2", 0x04: "mtc2", 0x06: "ctc2"}
VECTOR = {0x00: "vmulf", 0x01: "vmulu", 0x02: "vrndp", 0x03: "vmulq", 0x04: "vmudl", 0x05: "vmudm",
          0x06: "vmudn", 0x07: "vmudh", 0x08: "vmacf", 0x09: "vmacu", 0x0A: "vrndn", 0x0B: "vmacq",
          0x0C: "vmadl", 0x0D: "vmadm", 0x0E: "vmadn", 0x0F: "vmadh", 0x10: "vadd", 0x11: "vsub",
          0x13: "vabs", 0x14: "vaddc", 0x15: "vsubc", 0x1D: "vsar",
          0x20: "vlt", 0x21: "veq", 0x22: "vne", 0x23: "vge", 0x24: "vcl", 0x25: "vch", 0x26: "vcr",
          0x27: "vmrg", 0x28: "vand", 0x29: "vnand", 0x2A: "vor", 0x2B: "vnor", 0x2C: "vxor",
          0x2D: "vnxor", 0x30: "vrcp", 0x31: "vrcpl", 0x32: "vrcph", 0x33: "vmov", 0x34: "vrsq",
          0x35: "vrsql", 0x36: "vrsqh", 0x37: "vnop"}
LWC2 = {0x00: "lbv", 0x01: "lsv", 0x02: "llv", 0x03: "ldv", 0x04: "lqv", 0x05: "lrv", 0x06: "lpv",
        0x07: "luv", 0x08: "lhv", 0x09: "lfv", 0x0B: "ltv"}
SWC2 = {0x00: "sbv", 0x01: "ssv", 0x02: "slv", 0x03: "sdv", 0x04: "sqv", 0x05: "srv", 0x06: "spv",
        0x07: "suv", 0x08: "shv", 0x09: "sfv", 0x0A: "swv", 0x0B: "stv"}
SCALE = {0x00: 1, 0x01: 2, 0x02: 4, 0x03: 8, 0x04: 16, 0x05: 16, 0x06: 8, 0x07: 8, 0x08: 16,
         0x09: 16, 0x0A: 16, 0x0B: 16}
CONTROL = ("$vco", "$vcc", "$vce")
LANE = {0: "", 1: "[e1]", 2: "[0q]", 3: "[1q]", 4: "[0h]", 5: "[1h]", 6: "[2h]", 7: "[3h]",
        **{8 + i: f"[{i}]" for i in range(8)}}
ARITY = {"R3": 3, "SH": 3, "SHV": 3, "JR": 1, "JALR": 2, "BRK": 1, "IS": 3, "IU": 3, "LUI": 2,
         "MEM": 2, "B2": 3, "B1": 2, "J": 1, "C0": 2, "C2M": 2, "C2C": 2, "V3": 3, "V2": 2, "V0": 0,
         "VMEM": 2}


def _table() -> dict[str, tuple[str, int, int]]:
    """mnemonic -> (operand format, fixed bits, LWC2/SWC2 offset scale)."""
    t: dict[str, tuple[str, int, int]] = {}
    for fn, m in SPECIAL.items():
        fmt = ("SH" if m in ("sll", "srl", "sra") else "SHV" if m in ("sllv", "srlv", "srav")
               else "JR" if m == "jr" else "JALR" if m == "jalr" else "BRK" if m == "break" else "R3")
        t[m] = (fmt, fn, 0)
    for code, m in REGIMM.items():
        t[m] = ("B1", 0x01 << 26 | code << 16, 0)
    for op, m in PRIMARY.items():
        fmt = ("J" if op in (0x02, 0x03) else "B2" if op in (0x04, 0x05) else "B1" if op in (0x06, 0x07)
               else "IS" if op <= 0x0B else "IU" if op <= 0x0E else "LUI" if op == 0x0F else "MEM")
        t[m] = (fmt, op << 26, 0)
    for rs, m in COP0.items():
        t[m] = ("C0", 0x10 << 26 | rs << 21, 0)
    for rs, m in COP2.items():
        t[m] = ("C2M" if m in ("mfc2", "mtc2") else "C2C", 0x12 << 26 | rs << 21, 0)
    for fn, m in VECTOR.items():
        t[m] = ("V0" if fn == 0x37 else "V2" if fn >= 0x30 else "V3", 0x12 << 26 | 1 << 25 | fn, 0)
    for op, codes in ((0x32, LWC2), (0x3A, SWC2)):
        for code, m in codes.items():
            t[m] = ("VMEM", op << 26 | code << 11, SCALE[code])
    return t


TABLE = _table()
VECTOR_MNEMONICS = frozenset(m for m, (fmt, _, _) in TABLE.items() if fmt in ("V3", "V2", "V0", "VMEM", "C2M", "C2C"))


# --------------------------------------------------------------------------- disassembler

def _hex(v: int) -> str:
    return f"-0x{-v:X}" if v < 0 else f"0x{v:X}"


def _addr(v: int) -> str:
    return f"-0x{-v:X}" if v < 0 else f"0x{v:03X}"


def _operands(fmt: str, w: int, pc: int) -> str:
    rs, rt, rd, sa = (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31, (w >> 6) & 31
    imm = w & 0xFFFF
    simm = imm - 0x10000 if imm & 0x8000 else imm
    e = (w >> 21) & 15
    if fmt == "R3":
        return f"${rd}, ${rs}, ${rt}"
    if fmt == "SH":
        return f"${rd}, ${rt}, {sa}"
    if fmt == "SHV":
        return f"${rd}, ${rt}, ${rs}"
    if fmt == "JR":
        return f"${rs}"
    if fmt == "JALR":
        return f"${rd}, ${rs}"
    if fmt == "BRK":
        return f"0x{(w >> 6) & 0xFFFFF:X}"
    if fmt == "IS":
        return f"${rt}, ${rs}, {_hex(simm)}"
    if fmt == "IU":
        return f"${rt}, ${rs}, 0x{imm:X}"
    if fmt == "LUI":
        return f"${rt}, 0x{imm:X}"
    if fmt == "MEM":
        return f"${rt}, {_hex(simm)}(${rs})"
    if fmt == "B2":
        return f"${rs}, ${rt}, {_addr(pc + 4 + 4 * simm)}"
    if fmt == "B1":
        return f"${rs}, {_addr(pc + 4 + 4 * simm)}"
    if fmt == "J":
        return f"0x{(w & 0x3FFFFFF) << 2:08X}"
    if fmt == "C0":
        if rd > 15:
            _raw(f"cop0 register {rd}")
        return f"${rt}, $c{rd}"
    if fmt == "C2M":
        return f"${rt}, $v{rd}[{(w >> 7) & 15}]"
    if fmt == "C2C":
        if rd > 2:
            _raw(f"cop2 control register {rd}")
        return f"${rt}, {CONTROL[rd]}"
    if fmt == "V3":
        return f"$v{sa}, $v{rd}, $v{rt}{LANE[e]}"
    if fmt == "V2":
        if rd > 7:
            _raw(f"single-lane destination element {rd}")
        return f"$v{sa}[{rd}], $v{rt}{LANE[e]}"
    if fmt == "V0":
        return ""
    off = w & 0x7F                                                       # VMEM
    off = off - 0x80 if off & 0x40 else off
    return f"$v{rt}[{(w >> 7) & 15}], {_hex(off * SCALE[rd])}(${rs})"


def _text(w: int, pc: int) -> str:
    op, rs, rt, rd = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31
    if w == 0:
        return "nop"
    if op == 0x00:
        name = SPECIAL.get(w & 63) or _raw(f"special funct 0x{w & 63:02X}")
    elif op == 0x01:
        name = REGIMM.get(rt) or _raw(f"regimm 0x{rt:02X}")
    elif op == 0x10:
        name = COP0.get(rs) or _raw(f"cop0 rs 0x{rs:02X}")
    elif op == 0x12:
        if (w >> 25) & 1:
            name = VECTOR.get(w & 63) or _raw(f"vector funct 0x{w & 63:02X}")
        else:
            name = COP2.get(rs) or _raw(f"cop2 rs 0x{rs:02X}")
    elif op == 0x32:
        name = LWC2.get(rd) or _raw(f"lwc2 code 0x{rd:02X}")
    elif op == 0x3A:
        name = SWC2.get(rd) or _raw(f"swc2 code 0x{rd:02X}")
    else:
        name = PRIMARY.get(op) or _raw(f"opcode 0x{op:02X} (not an RSP opcode)")
    ops = _operands(TABLE[name][0], w, pc)
    return f"{name} {ops}" if ops else name


def classify(word: int, pc: int = 0) -> tuple[str | None, str | None]:
    """``(text, None)`` for a word written as an instruction, ``(None, reason)`` for a raw one."""
    word &= MASK
    try:
        text = _text(word, pc)
    except _Raw as exc:
        return None, str(exc)
    try:
        same = encode(text, pc) == word
    except AsmError:
        same = False
    return (text, None) if same else (None, "fields outside the canonical encoding")


def decode(word: int, pc: int = 0) -> str | None:
    """The instruction text of one IMEM word at address ``pc``, or None (kept as ``.word``)."""
    return classify(word, pc)[0]


def disassemble(words, origin: int = 0) -> list[str]:
    """One statement per word: the instruction, or ``.word 0x........``."""
    return [decode(w, origin + 4 * i) or f".word 0x{w & MASK:08X}" for i, w in enumerate(words)]


# --------------------------------------------------------------------------- assembler

_GPR = re.compile(r"\$(\d+)")
_VREG = re.compile(r"\$v(\d+)(?:\[([^\]]+)\])?")
_C0REG = re.compile(r"\$c(\d+)")
_MEMOP = re.compile(r"(.+)\((\$\d+)\)")


def _num(s: str, lo: int, hi: int, what: str = "value") -> int:
    try:
        v = int(s.strip(), 0)
    except ValueError:
        raise AsmError(f"not a number: {s!r}") from None
    if not lo <= v <= hi:
        raise AsmError(f"{what} {s.strip()} out of range")
    return v


def _gpr(s: str) -> int:
    m = _GPR.fullmatch(s)
    if not m or int(m[1]) > 31:
        raise AsmError(f"not a scalar register: {s!r}")
    return int(m[1])


def _vreg(s: str) -> tuple[int, str | None]:
    m = _VREG.fullmatch(s)
    if not m or int(m[1]) > 31:
        raise AsmError(f"not a vector register: {s!r}")
    return int(m[1]), m[2]


def _index(text: str | None, hi: int) -> int:
    return 0 if text is None else _num(text, 0, hi, "element")


def _lane(text: str | None) -> int:
    if text is None:
        return 0
    m = re.fullmatch(r"e(\d+)", text)
    if m and int(m[1]) <= 15:
        return int(m[1])
    m = re.fullmatch(r"([0-7])([qh]?)", text)
    if m:
        n, kind = int(m[1]), m[2]
        if kind == "q" and n <= 1:
            return 2 + n
        if kind == "h" and n <= 3:
            return 4 + n
        if not kind:
            return 8 + n
    raise AsmError(f"not an element: [{text}]")


def _mem(s: str) -> tuple[str, int]:
    m = _MEMOP.fullmatch(s)
    if not m:
        raise AsmError(f"not an offset(base) operand: {s!r}")
    return m[1].strip(), _gpr(m[2])


def _branch(s: str, pc: int) -> int:
    delta = _num(s, -(1 << 40), 1 << 40, "branch target") - pc - 4
    if delta % 4 or not -0x8000 <= delta // 4 <= 0x7FFF:
        raise AsmError(f"branch target {s} not reachable from 0x{pc:X}")
    return (delta // 4) & 0xFFFF


def encode(line: str, pc: int = 0) -> int:
    """One instruction statement (no comment) at address ``pc`` -> its word."""
    parts = line.strip().split(None, 1)
    if not parts:
        raise AsmError("empty statement")
    name = parts[0].lower()
    ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
    if name == "nop":
        fmt, w, scale = "V0", 0, 0
    elif name in TABLE:
        fmt, w, scale = TABLE[name]
    else:
        raise AsmError(f"unknown mnemonic {parts[0]!r}")
    if len(ops) != ARITY[fmt]:
        raise AsmError(f"{name} takes {ARITY[fmt]} operands, got {len(ops)}")
    if fmt == "R3":
        rd, rs, rt = map(_gpr, ops)
        return w | rs << 21 | rt << 16 | rd << 11
    if fmt == "SH":
        return w | _gpr(ops[1]) << 16 | _gpr(ops[0]) << 11 | _num(ops[2], 0, 31, "shift") << 6
    if fmt == "SHV":
        return w | _gpr(ops[2]) << 21 | _gpr(ops[1]) << 16 | _gpr(ops[0]) << 11
    if fmt == "JR":
        return w | _gpr(ops[0]) << 21
    if fmt == "JALR":
        return w | _gpr(ops[1]) << 21 | _gpr(ops[0]) << 11
    if fmt == "BRK":
        return w | _num(ops[0], 0, 0xFFFFF, "break code") << 6
    if fmt == "IS":
        return w | _gpr(ops[1]) << 21 | _gpr(ops[0]) << 16 | (_num(ops[2], -0x8000, 0x7FFF, "immediate") & 0xFFFF)
    if fmt == "IU":
        return w | _gpr(ops[1]) << 21 | _gpr(ops[0]) << 16 | _num(ops[2], 0, 0xFFFF, "immediate")
    if fmt == "LUI":
        return w | _gpr(ops[0]) << 16 | _num(ops[1], 0, 0xFFFF, "immediate")
    if fmt == "MEM":
        off, base = _mem(ops[1])
        return w | base << 21 | _gpr(ops[0]) << 16 | (_num(off, -0x8000, 0x7FFF, "offset") & 0xFFFF)
    if fmt == "B2":
        return w | _gpr(ops[0]) << 21 | _gpr(ops[1]) << 16 | _branch(ops[2], pc)
    if fmt == "B1":
        return w | _gpr(ops[0]) << 21 | _branch(ops[1], pc)
    if fmt == "J":
        target = _num(ops[0], 0, 0x0FFFFFFC, "jump target")
        if target % 4:
            raise AsmError(f"jump target {ops[0]} not word aligned")
        return w | target >> 2
    if fmt == "C0":
        m = _C0REG.fullmatch(ops[1])
        if not m or int(m[1]) > 15:
            raise AsmError(f"not a COP0 register: {ops[1]!r}")
        return w | _gpr(ops[0]) << 16 | int(m[1]) << 11
    if fmt == "C2M":
        v, b = _vreg(ops[1])
        return w | _gpr(ops[0]) << 16 | v << 11 | _index(b, 15) << 7
    if fmt == "C2C":
        if ops[1] not in CONTROL:
            raise AsmError(f"not a COP2 control register: {ops[1]!r}")
        return w | _gpr(ops[0]) << 16 | CONTROL.index(ops[1]) << 11
    if fmt == "V3":
        (vd, x), (vs, y), (vt, e) = map(_vreg, ops)
        if x is not None or y is not None:
            raise AsmError(f"{name}: only the last vector operand takes an element")
        return w | _lane(e) << 21 | vt << 16 | vs << 11 | vd << 6
    if fmt == "V2":
        (vd, de), (vt, e) = map(_vreg, ops)
        return w | _lane(e) << 21 | vt << 16 | _index(de, 7) << 11 | vd << 6
    if fmt == "V0":
        return w
    (vt, b), (off, base) = _vreg(ops[0]), _mem(ops[1])                   # VMEM
    value = _num(off, -64 * scale, 63 * scale, "offset")
    if value % scale:
        raise AsmError(f"{name} offset {off} is not a multiple of {scale}")
    return w | base << 21 | vt << 16 | _index(b, 15) << 7 | ((value // scale) & 0x7F)


def assemble(text: str) -> bytes:
    """The listing format above -> bytes (``.org`` defaults to 0)."""
    out = bytearray()
    pc = 0
    for n, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            if line.startswith("."):
                directive, _, rest = line.partition(" ")
                args = [a.strip() for a in rest.split(",")] if rest.strip() else []
                if directive == ".org" and len(args) == 1:
                    pc = _num(args[0], 0, MASK, ".org")
                elif directive == ".word" and args:
                    for a in args:
                        out += struct.pack(">I", _num(a, 0, MASK, ".word"))
                        pc += 4
                elif directive == ".byte" and args:
                    for a in args:
                        out.append(_num(a, 0, 0xFF, ".byte"))
                        pc += 1
                else:
                    raise AsmError(f"bad directive {line!r}")
                continue
            if pc % 4:
                raise AsmError(f"instruction at unaligned address 0x{pc:X}")
            out += struct.pack(">I", encode(line, pc))
            pc += 4
        except AsmError as exc:
            raise AsmError(f"line {n}: {exc}") from None
    return bytes(out)


# --------------------------------------------------------------------------- round trip and reports

def _words(blob: bytes) -> tuple[int, ...]:
    n = len(blob) // 4
    return struct.unpack_from(f">{n}I", blob) if n else ()


def _render(blob: bytes, origin: int, kind: str) -> tuple[list[str], dict]:
    words = _words(blob)
    lines = [f".org 0x{origin:03X}"]
    stats = {"decoded": 0, "raw": 0, "reasons": Counter(), "outside": 0, "mnemonics": Counter()}
    if kind == "text":
        for i, w in enumerate(words):
            pc = origin + 4 * i
            text, why = classify(w, pc)
            if text is None:
                stats["raw"] += 1
                stats["reasons"][why] += 1
                lines.append(f"    {'.word 0x%08X' % w:<36}# {pc:03X} raw: {why}")
                continue
            stats["decoded"] += 1
            stats["mnemonics"][text.split(None, 1)[0]] += 1
            op = w >> 26
            if op == 0x01 or 0x04 <= op <= 0x07:
                imm = w & 0xFFFF
                target = pc + 4 + 4 * (imm - 0x10000 if imm & 0x8000 else imm)
                stats["outside"] += int(not 0 <= target < IMEM_SIZE)
            lines.append(f"    {text:<36}# {pc:03X} {w:08X}")
    else:
        for i in range(0, len(words), 4):
            chunk = ", ".join(f"0x{w:08X}" for w in words[i:i + 4])
            lines.append(f"    .word {chunk}  # {origin + 4 * i:03X}")
        stats["raw"] = len(words)
        if words:
            stats["reasons"]["dmem data"] = len(words)
    tail = blob[4 * len(words):]
    if tail:
        lines.append("    .byte " + ", ".join(f"0x{b:02X}" for b in tail))
    return lines, stats


def listing(blob: bytes, origin: int = 0, kind: str = "text") -> str:
    """The assemblable listing of an IMEM (``text``) or DMEM (``data``) block."""
    return "\n".join(_render(bytes(blob), origin, kind)[0]) + "\n"


def round_trip(words, origin: int = 0) -> tuple[bool, int, int]:
    """IMEM words -> listing -> assembly: (identical, decoded words, raw ``.word`` words)."""
    blob = struct.pack(f">{len(words)}I", *words)
    lines, stats = _render(blob, origin, "text")
    return assemble("\n".join(lines)) == blob, stats["decoded"], stats["raw"]


def guess_origin(blob: bytes) -> tuple[int, str]:
    """A display origin: the 8-aligned IMEM address that puts the most ``j``/``jal`` targets inside
    the block (0x080, the usual task text origin, wins a tie).  Display only: the round trip is
    byte-exact at any origin."""
    words = _words(blob)
    targets = [((w & 0x3FFFFFF) << 2) & (IMEM_SIZE - 4) for w in words if w >> 26 in (0x02, 0x03)]
    if not targets:
        return 0, "default 0x000 (no j/jal)"
    size = 4 * len(words)
    scores = {o: sum(1 for t in targets if o <= t < o + size) for o in range(0, IMEM_SIZE, 8)}
    best = max(scores.values())
    tied = [o for o, s in scores.items() if s == best]
    origin = 0x080 if 0x080 in tied else tied[0]
    return origin, f"j/jal targets: {best} of {len(targets)} inside the block (display only)"


def block_report(blob: bytes, kind: str = "text", origin: int | None = None, *,
                 with_listing: bool = False) -> dict:
    """Round trip one block; raw ``.word``/``.byte`` bytes are reported as raw preservation."""
    blob = bytes(blob)
    if origin is not None:
        basis = "given"
    elif kind == "text":
        origin, basis = guess_origin(blob)
    else:
        origin, basis = 0, "dmem 0x000"
    lines, stats = _render(blob, origin, kind)
    text = "\n".join(lines) + "\n"
    back = assemble(text)
    raw_bytes = 4 * stats["raw"] + len(blob) % 4
    first = next((i for i, (a, b) in enumerate(zip(blob, back)) if a != b),
                 None if len(blob) == len(back) else min(len(blob), len(back)))
    out = {"kind": kind, "size": len(blob), "origin": f"0x{origin:03X}", "origin_basis": basis,
           "words": len(blob) // 4, "decoded_words": stats["decoded"], "raw_words": stats["raw"],
           "tail_bytes": len(blob) % 4, "raw_bytes": raw_bytes,
           "raw_preservation_share": round(raw_bytes / len(blob), 4) if blob else 0.0,
           "identical": back == blob, "first_difference": first,
           "sha256": hashlib.sha256(blob).hexdigest(), "reassembled_sha256": hashlib.sha256(back).hexdigest(),
           "raw_reasons": dict(stats["reasons"].most_common()),
           "branch_targets_outside_imem": stats["outside"],
           "vector_words": sum(n for m, n in stats["mnemonics"].items() if m in VECTOR_MNEMONICS)}
    if with_listing:
        out["listing"] = text
    return out


# --------------------------------------------------------------------------- extraction (rspid)

def extract_blocks(data: bytes, mapping: rspid.Mapping | None = None) -> tuple[list[dict], dict]:
    """``rspid`` IMEM (``text``) and DMEM (``data``) blocks of an image, each with its bytes."""
    data = bytes(data)
    report = rspid.identify(data, mapping=mapping)
    blocks = []
    for b in report["blocks"]:
        ident = b.get("identity") or {}
        blocks.append({"kind": b["kind"], "rom": b["rom"], "end": b["end"], "extent": b["extent"],
                       "status": b["status"], "name": ident.get("name"), "version": ident.get("version"),
                       "bytes": data[b["rom"]:b["end"]]})
    return blocks, report


def summarise(rows: list[dict]) -> dict:
    s: Counter = Counter()
    reasons: Counter = Counter()
    for r in rows:
        k = r["kind"]
        s[f"{k}_blocks"] += 1
        s[f"{k}_bytes"] += r["size"]
        s[f"{k}_raw_bytes"] += r["raw_bytes"]
        s[f"{k}_identical_blocks"] += int(r["identical"])
        if k == "text":
            s["text_decoded_words"] += r["decoded_words"]
            s["text_raw_words"] += r["raw_words"]
            reasons.update(r["raw_reasons"])
    total = s["text_bytes"] + s["data_bytes"]
    raw = s["text_raw_bytes"] + s["data_raw_bytes"]
    out = dict(sorted(s.items()))
    out["all_identical"] = all(r["identical"] for r in rows)
    out["raw_preservation_share"] = round(raw / total, 4) if total else None
    out["text_raw_preservation_share"] = round(s["text_raw_bytes"] / s["text_bytes"], 4) if s["text_bytes"] else None
    out["text_raw_reasons"] = dict(reasons.most_common(12))
    return out


def census_one(project: Path, listings: Path | None = None) -> dict:
    """Every ``rspid`` block of one decompile-init project, round-tripped."""
    image = rommod.Rom.load(rspid._project_rom(project))
    mapping, basis = rspid._layout_mapping(project)
    blocks, _ = extract_blocks(image.data, mapping)
    rows = []
    for b in blocks:
        rep = block_report(b.pop("bytes"), b["kind"], with_listing=listings is not None)
        text = rep.pop("listing", None)
        if text is not None:
            (listings / f"{project.name}_{b['kind']}_{b['rom']:08X}.s").write_text(text)
        rows.append({**b, "rom": f"0x{b['rom']:X}", "end": f"0x{b['end']:X}", **rep})
    return {"title": project.name, "mapping_basis": basis, "blocks": rows, "summary": summarise(rows)}


def _int0(s: str) -> int:
    return int(s, 0)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["census"]:
        ap = argparse.ArgumentParser(prog="python -m n64dk.analysis.rspasm census",
                                     description="Round-trip every rspid block of decompile-init projects")
        ap.add_argument("paths", nargs="+")
        ap.add_argument("--out")
        ap.add_argument("--listings", help="directory for one .s listing per block")
        args = ap.parse_args(argv[1:])
        listings = Path(args.listings) if args.listings else None
        if listings is not None:
            listings.mkdir(parents=True, exist_ok=True)
        reports = []
        for p in args.paths:
            path = Path(p)
            try:
                report = census_one(path, listings)
            except (OSError, rommod.RomError) as exc:
                print(json.dumps({"title": path.name, "error": str(exc)}), flush=True)
                continue
            reports.append(report)
            print(json.dumps({"title": report["title"], "mapping": report["mapping_basis"],
                              "summary": report["summary"]}), flush=True)
        total = summarise([b for r in reports for b in r["blocks"]])
        total["titles"] = len(reports)
        print(json.dumps({"summary": total}), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps({"titles": reports, "summary": total}, indent=1))
        return 0 if total["all_identical"] else 1
    ap = argparse.ArgumentParser(prog="python -m n64dk.analysis.rspasm",
                                 description="RSP IMEM/DMEM range: listing, assembly and byte comparison")
    ap.add_argument("rom")
    ap.add_argument("start", type=_int0)
    ap.add_argument("end", type=_int0)
    ap.add_argument("--kind", choices=("text", "data"), default="text")
    ap.add_argument("--origin", type=_int0, help="IMEM/DMEM address of START (default: guessed for text)")
    args = ap.parse_args(argv)
    try:
        data = rommod.Rom.load(args.rom).data
    except (OSError, rommod.RomError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not 0 <= args.start < args.end <= len(data):
        ap.error("START..END is not inside the image")
    rep = block_report(data[args.start:args.end], args.kind, args.origin, with_listing=True)
    sys.stdout.write(rep.pop("listing"))
    print("# " + json.dumps(rep))
    return 0 if rep["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
