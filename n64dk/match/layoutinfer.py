"""Competing array and structure-field declarations for one optimised function (opt-in experiment).

Why.  In the round-12 base runs the largest residual class of drafts that build is data layout:
the candidate reads a field at another offset or with another width than the retail code
(``global.declaration``, ``type.width``).  The whole-image recovery (``n64dk.analysis.layouts``)
cannot see two kinds of evidence, because its value tracking (``analysis.values``) drops a
pointer's identity as soon as an index is added to it or it is stepped in a loop:

* **strides** -- ``p + (i << n)``, ``p + i * c`` (``mult``/``mflo`` or shift-and-add) and a pointer
  stepped by a constant in a loop (``addiu r, r, k`` between a backward branch and its target)
  mean an array whose element is ``stride`` bytes; the offsets used off such a pointer are the
  element's fields (offset modulo the stride);
* **one function's own view** -- the whole-image solver merges every function's accesses and turns
  disagreements into byte arrays; one function's own widths and offsets can be the better guess
  for its draft.

What.  ``scan`` reads one function's instruction words (big-endian MIPS, no decoder library) and
records, per incoming argument register used as a base (``$a0``..``$a3``: the calling convention),
the constant-offset accesses (offset, width, signedness, float, load/store), the strided accesses,
the fields loaded and then used as pointers, and what the argument is passed on to (with its
displacement, and its stride when an element is passed).  ``propose`` turns that into at most
``MAX_VARIANTS`` competing declaration sets:

``array``   argument typed ``T *`` where ``T`` is the element (a struct of exactly ``stride`` bytes,
            or a scalar when the element is one field of the stride's width);
``struct``  argument typed as a structure with the fields at the observed offsets (repeated
            accesses off one base), an observed array kept as a member after the fields;
``cross``   ``struct`` plus the fields other functions use on the same object: callers that pass
            something to this argument (``values`` facts of the image) and callees this function
            passes the argument (or one element of it) to;
``alt``     the strongest competing reading: the element of an unrolled loop (accesses spaced
            evenly inside the step), a second observed stride, or the other width at an offset
            accessed with two widths.

Each set is the function's focused context (callee prototypes and globals, when the image's
recovered declarations exist) plus the structures and the function's own prototype with the
typed arguments.  ``draft_modes`` writes one m2c context per set and returns them as extra
entries of the draft portfolio (``commands/resolve.py``), each built and compared like every other
draft.  Nothing here is credited: a declaration is a draft input, the strict comparison decides,
and a set that does not help costs builds from the case's own budget
(``match/resolve.py``: the search allowance is the budget minus the builds already spent).

Opt-in: ``N64DK_LAYOUT_INFER=1``.  ``N64DK_LAYOUT_INFER_MAX`` (1..4) lowers the cap.  Functions
larger than ``MAX_SIZE`` are left alone (the large-function result of round 11 stands).
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ENV, ENV_MAX = "N64DK_LAYOUT_INFER", "N64DK_LAYOUT_INFER_MAX"
MAX_VARIANTS = 4
MAX_SIZE = 0x400
MAX_STRIDE = 0x400
SP, RA, ZERO = 29, 31, 0
ARGS = (4, 5, 6, 7)
CALL_CLOBBERED = tuple(range(1, 16)) + (24, 25, 31)

LOADS = {32: ("lb", 1, True), 36: ("lbu", 1, False), 33: ("lh", 2, True), 37: ("lhu", 2, False),
         35: ("lw", 4, True), 39: ("lwu", 4, False), 55: ("ld", 8, True),
         49: ("lwc1", 4, True), 53: ("ldc1", 8, True)}
STORES = {40: ("sb", 1), 41: ("sh", 2), 43: ("sw", 4), 63: ("sd", 8), 57: ("swc1", 4), 61: ("sdc1", 8)}
FLOAT_OPS = {49, 53, 57, 61}
GPR_LOADS = {32, 33, 35, 36, 37, 39, 55, 26, 27, 34, 38}       # incl. ldl/ldr/lwl/lwr

TYPEDEFS = ("typedef signed char s8; typedef unsigned char u8; typedef signed short s16; "
            "typedef unsigned short u16;\ntypedef signed int s32; typedef unsigned int u32; "
            "typedef signed long long s64; typedef unsigned long long u64;\ntypedef float f32; typedef double f64;\n")


def enabled(environ=None) -> bool:
    return (environ if environ is not None else os.environ).get(ENV) == "1"


def cap(environ=None) -> int:
    value = (environ if environ is not None else os.environ).get(ENV_MAX, "")
    return max(1, min(MAX_VARIANTS, int(value))) if value.isdigit() else MAX_VARIANTS


# ------------------------------------------------------------------------------------- evidence

@dataclass(frozen=True)
class Access:
    offset: int                 # from the object's base (the argument's value at entry)
    width: int
    signed: bool | None         # None for a store
    floating: bool
    store: bool
    site: int


@dataclass(frozen=True)
class Strided:
    stride: int
    offset: int                 # absolute offset from the base at the first element reached
    width: int
    signed: bool | None
    floating: bool
    store: bool
    site: int
    how: str                    # "index", "loop" or "callee"

    @property
    def element_offset(self) -> int:
        return self.offset % self.stride


@dataclass
class ObjectEvidence:
    index: int                                             # argument number
    fields: list[Access] = field(default_factory=list)
    strided: list[Strided] = field(default_factory=list)
    pointers: set[int] = field(default_factory=set)        # offsets loaded and then used as a base
    passed: list[tuple] = field(default_factory=list)      # (callee, argument, displacement, stride or 0)

    @property
    def accesses(self) -> int:
        return len(self.fields) + len(self.strided)


@dataclass
class Evidence:
    vram: int
    size: int
    objects: dict[int, ObjectEvidence] = field(default_factory=dict)
    loops: int = 0
    writes_v0: bool = False

    def obj(self, index: int) -> ObjectEvidence:
        return self.objects.setdefault(index, ObjectEvidence(index))

    def summary(self) -> dict:
        return {"loops": self.loops, "objects": {
            f"a{i}": {"fields": len(o.fields), "strided": len(o.strided),
                      "strides": sorted({s.stride for s in o.strided}), "pointers": len(o.pointers),
                      "passed": len(o.passed)} for i, o in sorted(self.objects.items())}}


def _words(code) -> list[int]:
    if isinstance(code, (bytes, bytearray)):
        return [int.from_bytes(code[i:i + 4], "big") for i in range(0, len(code) - len(code) % 4, 4)]
    return list(code)


def _simm(word: int) -> int:
    value = word & 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def _fields(word: int):
    return word >> 26, (word >> 21) & 31, (word >> 16) & 31, (word >> 11) & 31, (word >> 6) & 31, word & 0x3F


def branch_target(word: int, site: int) -> int | None:
    op, rs, rt, _rd, _sa, _funct = _fields(word)
    if op in (4, 5, 6, 7, 20, 21, 22, 23) or (op == 1 and rt in (0, 1, 2, 3, 16, 17, 18, 19)) \
            or (op == 17 and rs == 8):
        return (site + 4 + (_simm(word) << 2)) & 0xFFFFFFFF
    return None


def jump_target(word: int, site: int) -> int:
    """Target of ``j`` / ``jal`` (same 256 MB region as the delay slot)."""
    return ((site + 4) & 0xF0000000) | ((word & 0x3FFFFFF) << 2)


def _control(word: int, site: int) -> bool:
    op, _rs, _rt, _rd, _sa, funct = _fields(word)
    return branch_target(word, site) is not None or op in (2, 3) or (op == 0 and funct in (8, 9))


def _dest(word: int) -> int | None:
    """The general-purpose register an instruction writes, if any."""
    op, rs, rt, rd, _sa, funct = _fields(word)
    if op == 0:
        if funct in (8, 0x0C, 0x0D, 0x0F, 0x11, 0x13) or 0x18 <= funct <= 0x1F:
            return None                        # jr, syscall, break, sync, mthi/mtlo, mult/div
        return rd or None
    if op == 3:
        return RA
    if 8 <= op <= 15 or op in (24, 25) or op in GPR_LOADS:
        return rt or None
    if op == 17 and rs in (0, 1, 2):           # mfc1 / dmfc1 / cfc1
        return rt or None
    return None


def _loops(words: list[int], vram: int) -> tuple[dict[int, dict[int, int]], int]:
    """Loop headers (instruction index) -> {register: constant step} for registers whose only
    definition inside the loop is ``addiu r, r, step``."""
    heads: dict[int, dict[int, int]] = {}
    count = 0
    for s, word in enumerate(words):
        site = vram + 4 * s
        target = branch_target(word, site)
        if target is None and word >> 26 == 2:                   # a backward ``j`` closes a loop too
            target = jump_target(word, site)
        if target is None or not vram <= target <= site:
            continue
        count += 1
        first = (target - vram) // 4
        steps: dict[int, int] = {}
        bad: set[int] = set()
        for j in range(first, min(s + 2, len(words))):        # the body and the branch's delay slot
            other = words[j]
            op, rs, rt, _rd, _sa, _funct = _fields(other)
            if op in (9, 25) and rs == rt and rt not in (ZERO, SP) and _simm(other):
                if steps.get(rt, _simm(other)) != _simm(other):
                    bad.add(rt)
                steps[rt] = _simm(other)
            else:
                dest = _dest(other)
                if dest is not None:
                    bad.add(dest)
        found = {r: k for r, k in steps.items() if r not in bad and abs(k) <= MAX_STRIDE}
        if found:
            heads.setdefault(first, {}).update(found)
    return heads, count


POINTER = ("p", "ind", "idx")


def scan(code, vram: int) -> Evidence:
    """Array and field evidence of one function's own code (see the module documentation).

    Straight-line over the instructions in address order; a delay slot is applied before its
    branch or call; registers keep their values across forward branches (an approximation: every
    declaration built from this is only a draft input, judged by the build)."""
    words = _words(code)
    evidence = Evidence(vram, 4 * len(words))
    heads, evidence.loops = _loops(words, vram)
    regs: dict[int, tuple] = {a: ("p", a - 4, 0) for a in ARGS}
    defsite: dict[int, int] = {}
    lo: list = [None]

    def put(reg: int, value, site: int):
        if reg == ZERO:
            return
        defsite[reg] = site
        if value is None:
            regs.pop(reg, None)
        else:
            regs[reg] = value

    def lin(value, reg: int):
        """The value as a scaled index ("lin", token, factor, displacement)."""
        if value is not None and value[0] == "lin":
            return value
        if value is not None and value[0] == "c":
            return None
        if value is not None and value[0] == "p" and value[2] == 0:
            return ("lin", ("arg", value[1]), 1, 0)
        if value is not None and value[0] in ("ind", "idx", "p"):
            return None                         # a pointer is not an index
        return ("lin", ("def", reg, defsite.get(reg, -1)), 1, 0)

    def same_index(a, b, breg: int):
        """``a`` is a scaled index and ``b`` the same index (scaled or not): their factors."""
        if a is None or a[0] != "lin" or (b is not None and b[0] not in ("lin", "p", "fld")):
            return None
        other = lin(b, breg)
        return other if other is not None and other[1] == a[1] else None

    def add(left, right, lreg: int, rreg: int):
        for a, b, breg in ((left, right, rreg), (right, left, lreg)):
            other = same_index(a, b, breg)                   # shift-and-add: (i << 1) + i
            if other is not None:
                return ("lin", a[1], a[2] + other[2], a[3] + other[3])
        for a, b in ((left, right), (right, left)):
            if a is None:
                continue
            if a[0] in POINTER and b is not None and b[0] == "c":
                return (a[0], a[1], a[2] + b[1], *a[3:])
            if a[0] == "p" and b is not None and b[0] == "lin":
                return ("idx", a[1], a[2] + b[3], b[2])
            if a[0] == "p" and (b is None or b[0] == "fld"):
                return ("idx", a[1], a[2], 0)          # an index of unknown scale
        if left is not None and right is not None:
            if left[0] == "lin" and right[0] == "c":
                return ("lin", left[1], left[2], left[3] + right[1])
            if left[0] == "c" and right[0] == "lin":
                return ("lin", right[1], right[2], right[3] + left[1])
            if left[0] == "c" and right[0] == "c":
                return ("c", left[1] + right[1])
        return None

    def record(base, offset: int, width: int, signed, floating: bool, store: bool, site: int):
        kind = base[0]
        if kind == "p":
            evidence.obj(base[1]).fields.append(Access(base[2] + offset, width, signed, floating, store, site))
        elif kind == "ind":
            stride = abs(base[3])
            if stride >= width:
                evidence.obj(base[1]).strided.append(
                    Strided(stride, base[2] + offset, width, signed, floating, store, site, "loop"))
        elif kind == "idx":
            stride = base[3] if base[3] > 0 else (1 if width == 1 else 0)
            if width <= stride <= MAX_STRIDE:
                evidence.obj(base[1]).strided.append(
                    Strided(stride, base[2] + offset, width, signed, floating, store, site, "index"))

    def step(index: int):
        word = words[index]
        site = vram + 4 * index
        op, rs, rt, rd, sa, funct = _fields(word)
        a, b = (regs.get(rs) if rs else ("c", 0)), (regs.get(rt) if rt else ("c", 0))
        if _dest(word) == 2:
            evidence.writes_v0 = True
        if op in (9, 25):                                            # addiu / daddiu
            imm = _simm(word)
            if a is not None and rs == rt and a[0] == "ind" and imm == a[3]:
                return                                               # the loop's own step
            if a is not None and rs == rt and a[0] == "lin" and a[1][0] == "loop" and a[1][2] == rt \
                    and imm == a[2]:
                return
            if a is None:
                put(rt, None, site)
            elif a[0] in POINTER or a[0] == "lin":
                put(rt, (*a[:-1], a[-1] + imm) if a[0] == "lin" else (a[0], a[1], a[2] + imm, *a[3:]), site)
            elif a[0] == "c":
                put(rt, ("c", a[1] + imm), site)
            else:
                put(rt, None, site)
            return
        if op == 13:                                                 # ori
            put(rt, ("c", (a[1] | (word & 0xFFFF))) if a is not None and a[0] == "c" else None, site)
            return
        if op == 0 and funct in (0x20, 0x21, 0x25, 0x2C, 0x2D) and rd:   # add(u) / or / dadd(u)
            if funct == 0x25 and rs and rt:
                put(rd, None, site)
            elif not rt:
                put(rd, a, site)
            elif not rs:
                put(rd, b, site)
            else:
                put(rd, add(a, b, rs, rt), site)
            return
        if op == 0 and funct in (0x22, 0x23, 0x2F) and rd:           # sub(u) / dsubu
            value = None
            other = same_index(a, b, rt) if rt else None              # shift-and-subtract: (i << 3) - i
            if other is not None:
                value = ("lin", a[1], a[2] - other[2], a[3] - other[3])
            elif a is not None and b is not None:
                if a[0] == "c" and b[0] == "c":
                    value = ("c", a[1] - b[1])
                elif a[0] in POINTER and b[0] == "c":
                    value = (a[0], a[1], a[2] - b[1], *a[3:])
            elif a is not None and not rt:
                value = a
            put(rd, value, site)
            return
        if op == 0 and funct == 0 and rd:                            # sll
            if b is not None and b[0] == "c":
                put(rd, ("c", (b[1] << sa) & 0xFFFFFFFF), site)
            else:
                scaled = lin(b, rt)
                put(rd, ("lin", scaled[1], scaled[2] << sa, scaled[3] << sa) if scaled else None, site)
            return
        if op == 0 and funct in (0x18, 0x19, 0x1C, 0x1D):            # mult(u) / dmult(u)
            lo[0] = None
            if b is not None and b[0] == "c" and b[1]:
                scaled = lin(a, rs)
                lo[0] = ("lin", scaled[1], scaled[2] * b[1], scaled[3] * b[1]) if scaled else None
            elif a is not None and a[0] == "c" and a[1]:
                scaled = lin(b, rt)
                lo[0] = ("lin", scaled[1], scaled[2] * a[1], scaled[3] * a[1]) if scaled else None
            return
        if op == 0 and funct == 0x12 and rd:                         # mflo
            put(rd, lo[0], site)
            return
        if op in LOADS or op in STORES:
            store = op in STORES
            width = STORES[op][1] if store else LOADS[op][1]
            signed = None if store else LOADS[op][2]
            base = regs.get(rs) if rs else None
            if base is not None and base[0] in POINTER:
                record(base, _simm(word), width, signed, op in FLOAT_OPS, store, site)
            elif base is not None and base[0] == "fld":
                evidence.obj(base[1]).pointers.add(base[2])
            if not store and op in GPR_LOADS:
                loaded = ("fld", base[1], base[2] + _simm(word)) if base is not None and base[0] == "p" \
                    and op == 35 else None
                put(rt, loaded, site)
            return
        dest = _dest(word)
        if dest is not None:
            put(dest, None, site)

    def call(word: int, site: int):
        op = word >> 26
        target = jump_target(word, site) if op in (2, 3) else None
        for reg in ARGS:
            value = regs.get(reg)
            if value is None or value[0] not in POINTER:
                continue
            stride = abs(value[3]) if value[0] in ("ind", "idx") else 0
            if value[0] == "idx" and not stride:
                continue
            evidence.obj(value[1]).passed.append((target, reg - 4, value[2], stride))
        if op != 2:                                                  # a tail jump does not return here
            for reg in CALL_CLOBBERED:
                put(reg, None, site)

    index = 0
    while index < len(words):
        for reg, step_by in heads.get(index, {}).items():
            value = regs.get(reg)
            if value is not None and value[0] == "p":
                regs[reg] = ("ind", value[1], value[2], step_by)
            elif value is not None and value[0] == "c":
                regs[reg] = ("lin", ("loop", index, reg), step_by, value[1])
        word = words[index]
        site = vram + 4 * index
        if _control(word, site):
            if index + 1 < len(words):
                step(index + 1)
            op, _rs, _rt, _rd, _sa, funct = _fields(word)
            local = op == 2 and vram <= jump_target(word, site) < vram + 4 * len(words)
            if (op in (2, 3) and not local) or (op == 0 and funct == 9):
                call(word, site)
            index += 2
            continue
        step(index)
        index += 1
    return evidence


def cross_fields(vram: int, evidence: Evidence, facts: dict | None) -> dict[int, list]:
    """Accesses other functions make to this function's argument objects, from ``values`` facts
    (duck-typed: ``.accesses`` with ``base/offset/width/signed/floating/store/site``, ``.calls`` with
    ``target/args``).  Returns argument -> [Access | Strided]."""
    out: dict[int, list] = {}
    if not facts:
        return out
    for index, obj in evidence.objects.items():                  # callees this function passes to
        for target, argument, displacement, stride in obj.passed:
            callee = facts.get(target) if target is not None else None
            if callee is None or target == vram:
                continue
            for access in callee.accesses:
                if access.base != ("param", argument):
                    continue
                offset = displacement + access.offset
                if stride:
                    if access.offset + access.width <= stride and access.offset >= 0:
                        out.setdefault(index, []).append(Strided(stride, offset, access.width, access.signed,
                                                                 access.floating, access.store, access.site,
                                                                 "callee"))
                elif offset >= 0:
                    out.setdefault(index, []).append(Access(offset, access.width, access.signed,
                                                            access.floating, access.store, access.site))
    for caller_vram, caller in facts.items():                     # callers: what they pass it
        if caller_vram == vram:
            continue
        for site in caller.calls:
            if site.target != vram:
                continue
            for reg, value in site.args.items():
                if value is None or reg not in ARGS or (reg - 4) not in evidence.objects:
                    continue
                key, displacement = value
                for access in caller.accesses:
                    if access.base == key and access.offset - displacement >= 0:
                        out.setdefault(reg - 4, []).append(Access(access.offset - displacement, access.width,
                                                                  access.signed, access.floating, access.store,
                                                                  access.site))
    return out


# ---------------------------------------------------------------------------------- declarations

def ctype(width: int, signed: bool | None, floating: bool) -> str:
    if floating:
        return "f32" if width == 4 else "f64"
    if signed is None:
        return {1: "u8", 2: "u16", 4: "s32", 8: "s64"}[width]
    return {1: "s8" if signed else "u8", 2: "s16" if signed else "u16", 4: "s32" if signed else "u32",
            8: "s64"}[width]


@dataclass
class Member:
    offset: int
    width: int
    ctype: str
    count: int = 1                       # elements of an array member
    element: str | None = None           # struct name of an array member's element
    array: bool = False

    @property
    def end(self) -> int:
        return self.offset + self.width * self.count


def _choose(group: list, prefer_width: int | None = None):
    """(width, signed, floating) for the accesses at one offset: the most used width (or the
    preferred one when it was seen), then signedness and float by majority."""
    widths = Counter(a.width for a in group)
    width = prefer_width if prefer_width in widths else max(widths, key=lambda w: (widths[w], w))
    same = [a for a in group if a.width == width]
    loads = [a for a in same if not a.store]
    signed = None if not loads else sum(bool(a.signed) for a in loads) * 2 >= len(loads)
    if signed is None and width == 4:
        signed = True
    floating = sum(a.floating for a in same) * 2 > len(same)
    return width, signed, floating


def _members(accesses: list, pointers: set[int], limit: int | None = None,
             prefer: dict[int, int] | None = None) -> tuple[list[Member], list[int]]:
    """Non-overlapping, naturally aligned members from accesses; returns (members, conflicted offsets
    where two widths were seen)."""
    by_offset: dict[int, list] = {}
    for access in accesses:
        if access.offset >= 0 and (limit is None or access.offset < limit):
            by_offset.setdefault(access.offset, []).append(access)
    members, conflicts, cursor = [], [], 0
    for offset in sorted(by_offset):
        group = by_offset[offset]
        if len({a.width for a in group}) > 1:
            conflicts.append(offset)
        width, signed, floating = _choose(group, (prefer or {}).get(offset))
        if offset % min(width, 8) or offset < cursor or (limit is not None and offset + width > limit):
            continue                                       # misaligned or overlapping: not a field
        name = "void *" if offset in pointers and width == 4 and not floating else ctype(width, signed, floating)
        members.append(Member(offset, width, name))
        cursor = offset + width
    return members, conflicts


def _align(members: list[Member]) -> int:
    return max((min(m.width, 8) for m in members), default=1)


def struct_text(name: str, members: list[Member], size: int | None = None) -> str | None:
    """C for a structure whose members sit at exactly their offsets; ``size`` forces the total
    (an array element must be exactly its stride).  None when that is impossible."""
    lines = [f"struct {name} {{"]
    cursor = 0
    for member in sorted(members, key=lambda m: m.offset):
        if member.offset < cursor:
            return None
        if member.offset > cursor:
            lines.append(f"    char pad_{cursor:X}[0x{member.offset - cursor:X}];")
        kind = f"struct {member.element}" if member.element else member.ctype
        star = "" if kind.endswith("*") else " "
        suffix = f"[{member.count}]" if member.array else ""
        lines.append(f"    {kind}{star}unk_{member.offset:X}{suffix};")
        cursor = member.end
    if size is not None:
        if cursor > size or size % _align(members):
            return None
        if size > cursor:
            lines.append(f"    char pad_{cursor:X}[0x{size - cursor:X}];")
    elif not members:
        return None
    lines.append("};")
    return "\n".join(lines)


@dataclass
class Typing:
    """How one argument is declared in one variant: its C type and the structures it needs."""
    ctype: str
    structs: list[str]


def _element(prefix: str, strided: list, stride: int, extra: list | None = None) -> Typing | None:
    """``T *`` for an array of ``stride``-byte elements; ``extra`` constant-offset accesses fold in
    modulo the stride."""
    folded = [Access(s.element_offset, s.width, s.signed, s.floating, s.store, s.site)
              for s in strided if s.stride == stride]
    folded += [Access(a.offset % stride, a.width, a.signed, a.floating, a.store, a.site) for a in extra or ()]
    members, _conflicts = _members(folded, set(), limit=stride)
    if not members:
        return None
    if len(members) == 1 and members[0].offset == 0 and members[0].width == stride:
        return Typing(f"{members[0].ctype} *", [])
    name = f"{prefix}_e{stride:X}"
    text = struct_text(name, members, size=stride)
    return Typing(f"struct {name} *", [text]) if text else None


def _struct(prefix: str, obj: ObjectEvidence, extra: list | None = None,
            prefer: dict[int, int] | None = None) -> Typing | None:
    """A structure with the constant-offset fields; an array seen after the last field becomes an
    array member.  Needs repeated accesses off the base."""
    accesses = obj.fields + [a for a in extra or () if isinstance(a, Access)]
    members, _conflicts = _members(accesses, obj.pointers, prefer=prefer)
    structs = []
    strides = Counter(s.stride for s in obj.strided)
    if strides:
        stride = strides.most_common(1)[0][0]
        rows = [s for s in obj.strided if s.stride == stride]
        start = min(s.offset - s.element_offset for s in rows)
        end = max((m.end for m in members), default=0)
        # an array starting at 0 with no field before it is the ``array`` reading, not a structure
        if start >= end and start >= 0 and (members or start > 0):
            element = _element(prefix + "_m", obj.strided, stride)
            if element is not None:
                if element.structs:
                    structs += element.structs
                    name = element.ctype[len("struct "):-2]
                    members.append(Member(start, stride, "", 1, name, array=True))
                else:
                    members.append(Member(start, stride, element.ctype[:-2].strip(), 1, array=True))
    if not members or (len(members) == 1 and members[0].offset == 0 and not members[0].array):
        return None                       # one field at offset 0 is a plain pointer, not a layout
    text = struct_text(prefix, members)
    return Typing(f"struct {prefix} *", structs + [text]) if text else None


@dataclass
class Variant:
    label: str
    typings: dict[int, Typing]
    prototype: str
    structs: str

    def explain(self) -> dict:
        return {"label": self.label, "arguments": {f"a{i}": t.ctype for i, t in sorted(self.typings.items())}}


_PROTOTYPE = re.compile(r"^\s*(?P<ret>.+?)\s*\b(?P<name>[A-Za-z_]\w*)\s*\((?P<args>.*)\)\s*;\s*$")


def _split_args(text: str) -> list[str]:
    text = text.strip()
    if text in ("", "void"):
        return []
    out, depth, current = [], 0, ""
    for char in text:
        if char == "," and depth == 0:
            out.append(current.strip())
            current = ""
            continue
        depth += char == "("
        depth -= char == ")"
        current += char
    out.append(current.strip())
    return out


def prototype(name: str, typings: dict[int, Typing], known: str | None, writes_v0: bool) -> str:
    """The function's own prototype with the typed arguments substituted (a recovered prototype
    keeps its return type and other arguments; without one, untyped arguments are ``s32`` and the
    return is ``s32`` when the code writes ``$v0`` outside a call, else ``void``)."""
    found = _PROTOTYPE.match(known or "")
    if found and found.group("name") == name:
        returns, args = found.group("ret"), _split_args(found.group("args"))
    else:
        returns, args = ("s32" if writes_v0 else "void"), []
    while len(args) <= max(typings, default=-1):
        args.append("s32")
    for index, typing in typings.items():
        if args[index] not in ("f32", "f64"):            # a float argument is not a base pointer
            args[index] = typing.ctype
    return f"{returns} {name}({', '.join(args) or 'void'});"


def _alt(prefix: str, obj: ObjectEvidence) -> Typing | None:
    """The strongest competing reading of one object (see the module documentation)."""
    strides = Counter(s.stride for s in obj.strided)
    if strides:
        stride = strides.most_common(1)[0][0]
        rows = [s for s in obj.strided if s.stride == stride]
        offsets = sorted({s.element_offset for s in rows})
        kinds = {(s.width, s.floating) for s in rows}
        if len(offsets) >= 2 and len(kinds) == 1:
            gap = offsets[1] - offsets[0]
            spaced = all(b - a == gap for a, b in zip(offsets, offsets[1:]))
            if spaced and gap >= rows[0].width and stride % gap == 0 and stride // gap >= 2:
                unrolled = [Strided(gap, s.offset, s.width, s.signed, s.floating, s.store, s.site, s.how)
                            for s in rows]
                return _element(prefix + "_u", unrolled, gap)
        if len(strides) > 1:
            second = strides.most_common(2)[1][0]
            return _element(prefix + "_s", obj.strided, second)
    groups: dict[int, set[int]] = {}
    for access in obj.fields:
        groups.setdefault(access.offset, set()).add(access.width)
    prefer = {}
    for offset, widths in groups.items():
        if len(widths) > 1:
            used = Counter(a.width for a in obj.fields if a.offset == offset)
            prefer[offset] = sorted(widths, key=lambda w: (used[w], w))[0]     # the less used width
    return _struct(prefix + "_w", obj, prefer=prefer) if prefer else None


def propose(name: str, code, vram: int, rendered=None, facts: dict | None = None,
            limit: int = MAX_VARIANTS) -> tuple[list[Variant], Evidence]:
    """At most ``limit`` competing declaration sets for one function (order: array, struct, cross,
    alt), and the evidence they came from."""
    words = _words(code)
    if 4 * len(words) > MAX_SIZE:
        return [], Evidence(vram, 4 * len(words))
    evidence = scan(words, vram)
    known = getattr(rendered, "prototypes", {}).get(vram) if rendered is not None else None
    cross = cross_fields(vram, evidence, facts)
    plans: dict[str, dict[int, Typing]] = {"array": {}, "struct": {}, "cross": {}, "alt": {}}
    for index, obj in sorted(evidence.objects.items()):
        if not obj.accesses or (obj.accesses + len(cross.get(index, ())) < 2 and not obj.strided):
            continue                                         # one access is not a layout
        prefix = f"LI_{name}_a{index}"
        strides = Counter(s.stride for s in obj.strided)
        if strides:
            stride = strides.most_common(1)[0][0]
            typing = _element(prefix + "_arr", obj.strided, stride, obj.fields)
            if typing is not None:
                plans["array"][index] = typing
        typing = _struct(prefix, obj)
        if typing is not None:
            plans["struct"][index] = typing
        others = cross.get(index, [])
        if others:
            widened = ObjectEvidence(index, list(obj.fields), obj.strided + [s for s in others
                                                                              if isinstance(s, Strided)],
                                     set(obj.pointers), list(obj.passed))
            own = {a.offset for a in obj.fields}
            added = [a for a in others if isinstance(a, Access) and a.offset not in own]
            typing = _struct(prefix + "_x", widened, added)
            if typing is not None and (added or len(widened.strided) > len(obj.strided)):
                plans["cross"][index] = typing
        typing = _alt(prefix, obj)
        if typing is not None:
            plans["alt"][index] = typing
    variants, seen = [], set()
    for label in ("array", "struct", "cross", "alt"):
        typings = plans[label]
        if not typings:
            continue
        structs = "\n".join(text for typing in typings.values() for text in typing.structs)
        proto = prototype(name, typings, known, evidence.writes_v0)
        key = (structs, proto)
        if key in seen:
            continue
        seen.add(key)
        variants.append(Variant(label, typings, proto, structs))
        if len(variants) >= limit:
            break
    return variants, evidence


def texts(variant: Variant, base_context: str = "", base_declarations: str = "") -> tuple[str, str]:
    """(m2c context, declarations to prepend to the draft) for one variant.  ``base_*`` are the
    function's focused recovered declarations (callee prototypes, globals), when there are any."""
    ours = (f"/* n64dk layoutinfer ({variant.label}): competing declarations from this function's own "
            f"accesses. Evidence, not truth. */\n{variant.structs}\n{variant.prototype}\n")
    context = (base_context or TYPEDEFS) + ours
    return context, base_declarations + ours


def draft_modes(name: str, code, vram: int, folder: Path, rendered=None, facts: dict | None = None,
                environ=None) -> list[tuple]:
    """Extra draft-portfolio entries ``(label, context path, declarations, None)``: one per
    variant, capped by ``N64DK_LAYOUT_INFER_MAX``.  Empty when nothing qualifies."""
    variants, _evidence = propose(name, code, vram, rendered, facts, cap(environ))
    if not variants:
        return []
    base_context = base_declarations = ""
    if rendered is not None and facts and vram in facts:
        base_context = rendered.focused(facts[vram], vram, False, typedefs=True)
        base_declarations = rendered.focused(facts[vram], vram, False, typedefs=False)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for variant in variants:
        context, declarations = texts(variant, base_context, base_declarations)
        path = folder / f"{name}.layout_{variant.label}.h"
        path.write_text(context)
        out.append((f"layout:{variant.label}", path, declarations, None))
    return out
