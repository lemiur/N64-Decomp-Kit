"""What a function does to bytes: copy, transform, checksum, fixup, PI DMA (static), and what a
game's own routine produced when run (dynamic capture), with a provenance proposal for the image
registry.

Static summarisation (``summarise``) works on the bytes of one function, with no game knowledge:

* **Loops** are back edges (a branch or ``j`` to an earlier address in the function); the body is
  the contiguous range from the target to the delay slot.  Inside a body a flow-insensitive pass
  marks every register whose value derives from a load and records which operations touched it
  (shift, mask, logic, arithmetic).  A loop is then classified:

  - ``transform``: it stores loaded-derived data and either keeps a *bit buffer* (a register shifted
    in place that is a loaded value or is masked against one), copies a *back reference* (a load
    whose base is a store pointer minus a loaded-derived distance: LZ-like; the window is the
    ``andi`` mask that bounds the distance, when there is one), or stores data that went through
    shifts/masks/logic/arithmetic;
  - ``fixup``: read-modify-write of words at the same base and offset, the stored value being the
    loaded one plus something (relocation passes);
  - ``copy``: a load and a store of one width through two different pointers advanced by the same
    stride, the stored value being the loaded value unchanged;
  - ``checksum``: loads through an advancing pointer accumulated into a register (``addu``/``xor``/
    ``or``/``subu`` of itself) and no store of loaded data;
  - ``fill``: stores of non-loaded values through an advancing pointer, no loads;
  - ``other``.

  A function's class is the first of ``PRECEDENCE`` among its loops.
* **PI DMA**: a linear constant/origin propagation finds stores to the PI registers
  (``PI_DRAM_ADDR`` 0xA4600000, ``PI_CART_ADDR`` 0xA4600004, ``PI_RD_LEN`` 0xA4600008,
  ``PI_WR_LEN`` 0xA460000C) and which argument register each stored value came from (through
  ``osVirtualToPhysical``-style calls and stack spills).  ``dma_ranges`` then reads constant
  arguments at the call sites of such functions and gives ROM -> RAM ranges.  (``osPiStartDma``
  proper sends a message to the PI manager thread: its callers are found by
  ``n64dk.assets.loaders`` from ROM-address arguments, not here.)

These are heuristics over bytes: a class is evidence about a function's role, never a proof of it.

Dynamic capture (``capture``) runs a routine on the kit's interpreter (``n64dk.emu``; a bare machine
from ``bare_machine``), under an instruction limit, and returns a deterministic receipt: entry,
inputs (arguments, the hashes of RAM, ROM and registers at entry), the ROM ranges read (PI DMA,
exact; direct cartridge reads, 4 KiB pages; caller-declared preloads), the RAM written with a coarse
taint (``rom-dma``: landed by DMA; ``rom-derived``: written by a run that read ROM data; ``stack``;
``untainted``), the output range and its SHA-256, the instruction count, the limits and a cache key
computed before the run from everything that determines it.  It fails closed: any stop but a return
(budget exhausted, fault, unmapped access), an open-bus read, a wall-clock overrun (checked when the
run ends; the instruction limit is the bound that stops a runaway), no ROM-derived output or an
output above ``max_output`` give ``status: failed`` and no output hash.

``register_image`` turns an ``ok`` receipt into a proposal for ``[images.<name>]`` in ``n64dk.toml``
(returned, never written).  Its ``raw_segments`` transform is ``capture:<key>``, which the image
registry already reports as not linearly representable, so nothing downstream treats the decoded
bytes as a copy of ROM bytes; the proposal carries ``strict: false``.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import time
from dataclasses import dataclass

M32 = 0xFFFFFFFF
SP, RA, V0 = 29, 31, 2
ARG_NAMES = {4: "a0", 5: "a1", 6: "a2", 7: "a3"}
CALLER_SAVED = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 31)
PRECEDENCE = ("transform", "fixup", "copy", "checksum", "fill", "other")
PI_REGS = {0x04600000: "dram", 0x04600004: "cart", 0x04600008: "rd_len", 0x0460000C: "wr_len",
           0x04600010: "status"}
CART_BASE = 0x10000000

# ---------------------------------------------------------------------------------------------
# A small MIPS decoder (only what the analyses need; no external disassembler)

_SPECIAL = {0x00: "sll", 0x02: "srl", 0x03: "sra", 0x04: "sllv", 0x06: "srlv", 0x07: "srav",
            0x08: "jr", 0x09: "jalr", 0x0C: "syscall", 0x0D: "break", 0x0F: "sync",
            0x10: "mfhi", 0x11: "mthi", 0x12: "mflo", 0x13: "mtlo",
            0x14: "dsllv", 0x16: "dsrlv", 0x17: "dsrav",
            0x18: "mult", 0x19: "multu", 0x1A: "div", 0x1B: "divu",
            0x1C: "dmult", 0x1D: "dmultu", 0x1E: "ddiv", 0x1F: "ddivu",
            0x20: "add", 0x21: "addu", 0x22: "sub", 0x23: "subu", 0x24: "and", 0x25: "or",
            0x26: "xor", 0x27: "nor", 0x2A: "slt", 0x2B: "sltu", 0x2C: "dadd", 0x2D: "daddu",
            0x2E: "dsub", 0x2F: "dsubu", 0x38: "dsll", 0x3A: "dsrl", 0x3B: "dsra",
            0x3C: "dsll32", 0x3E: "dsrl32", 0x3F: "dsra32"}
_SHIFT_SA = {0x00, 0x02, 0x03, 0x38, 0x3A, 0x3B, 0x3C, 0x3E, 0x3F}
_SHIFT_V = {0x04, 0x06, 0x07, 0x14, 0x16, 0x17}
_REGIMM = {0x00: "bltz", 0x01: "bgez", 0x02: "bltzl", 0x03: "bgezl",
           0x10: "bltzal", 0x11: "bgezal", 0x12: "bltzall", 0x13: "bgezall"}
_BRANCH = {4: "beq", 5: "bne", 6: "blez", 7: "bgtz", 20: "beql", 21: "bnel", 22: "blezl", 23: "bgtzl"}
_IMM = {8: "addi", 9: "addiu", 10: "slti", 11: "sltiu", 12: "andi", 13: "ori", 14: "xori",
        24: "daddi", 25: "daddiu"}
_LOADS = {32: ("lb", 1, True), 33: ("lh", 2, True), 35: ("lw", 4, True), 36: ("lbu", 1, False),
          37: ("lhu", 2, False), 39: ("lwu", 4, False), 55: ("ld", 8, False)}
_STORES = {40: ("sb", 1), 41: ("sh", 2), 43: ("sw", 4), 63: ("sd", 8)}

OPCLASS = {"sll": "shift", "srl": "shift", "sra": "shift", "sllv": "shift", "srlv": "shift",
           "srav": "shift", "dsll": "shift", "dsrl": "shift", "dsra": "shift", "dsll32": "shift",
           "dsrl32": "shift", "dsra32": "shift", "dsllv": "shift", "dsrlv": "shift", "dsrav": "shift",
           "andi": "mask", "and": "mask", "or": "logic", "ori": "logic", "xor": "logic",
           "xori": "logic", "nor": "logic", "addu": "arith", "addiu": "arith", "subu": "arith",
           "add": "arith", "addi": "arith", "sub": "arith", "daddu": "arith", "daddiu": "arith",
           "dsubu": "arith", "dadd": "arith", "daddi": "arith", "dsub": "arith",
           "slt": "compare", "sltu": "compare", "slti": "compare", "sltiu": "compare"}
_MOVES = {"addu", "or", "daddu", "add", "dadd"}


@dataclass(frozen=True)
class Ins:
    at: int
    word: int
    name: str
    rs: int
    rt: int
    rd: int
    sa: int
    imm: int
    simm: int
    dst: int = 0              # general register written (0: none)
    srcs: tuple = ()          # general registers read ($zero excluded)
    kind: str = ""            # branch | jump | call | jalr | jr | return
    likely: bool = False
    target: int | None = None
    width: int = 0            # bytes accessed by an integer load or store
    store: bool = False
    signed: bool = False

    @property
    def unconditional(self) -> bool:
        if self.kind in ("jump", "jr", "return"):
            return True
        if self.kind != "branch":
            return False
        op = self.word >> 26
        return (op == 4 and self.rs == self.rt) or (op == 1 and self.rt == 1 and self.rs == 0)


def decode(word: int, at: int) -> Ins:
    op = word >> 26
    rs, rt, rd, sa, fn = (word >> 21) & 31, (word >> 16) & 31, (word >> 11) & 31, (word >> 6) & 31, word & 63
    imm = word & 0xFFFF
    simm = imm - 0x10000 if imm & 0x8000 else imm
    name, dst, srcs, kind, likely, target, width, store, signed = "?", 0, (), "", False, None, 0, False, False
    branch_target = (at + 4 + (simm << 2)) & M32
    if op == 0:
        name = _SPECIAL.get(fn, "?")
        if fn in _SHIFT_SA:
            dst, srcs = rd, (rt,)
        elif fn in _SHIFT_V:
            dst, srcs = rd, (rt, rs)
        elif fn == 0x08:
            kind, srcs = ("return" if rs == RA else "jr"), (rs,)
        elif fn == 0x09:
            kind, srcs, dst = "jalr", (rs,), rd
        elif fn in (0x10, 0x12):
            dst = rd
        elif fn in (0x11, 0x13):
            srcs = (rs,)
        elif 0x18 <= fn <= 0x1F:
            srcs = (rs, rt)
        elif 0x20 <= fn <= 0x2F:
            dst, srcs = rd, (rs, rt)
    elif op == 1:
        name = _REGIMM.get(rt, "?")
        if rt in _REGIMM:
            kind = "call" if rt >= 0x10 else "branch"
            likely = rt in (0x02, 0x03, 0x12, 0x13)
            srcs, target = (rs,), branch_target
            dst = RA if rt >= 0x10 else 0
    elif op in (2, 3):
        name, kind = ("j", "jump") if op == 2 else ("jal", "call")
        target = ((at + 4) & 0xF0000000) | ((word & 0x3FFFFFF) << 2)
        dst = RA if op == 3 else 0
    elif op in _BRANCH:
        name, kind, likely, target = _BRANCH[op], "branch", op >= 20, branch_target
        srcs = (rs, rt) if op & 3 < 2 else (rs,)
    elif op in _IMM:
        name, dst, srcs = _IMM[op], rt, (rs,)
    elif op == 15:
        name, dst = "lui", rt
    elif op in _LOADS:
        name, width, signed = _LOADS[op]
        dst, srcs = rt, (rs,)
    elif op in _STORES:
        name, width = _STORES[op]
        store, srcs = True, (rs, rt)
    elif op in (34, 38):
        name, dst, srcs = ("lwl" if op == 34 else "lwr"), rt, (rs, rt)
    elif op in (42, 46):
        name, srcs = ("swl" if op == 42 else "swr"), (rs, rt)
    elif op in (49, 53, 57, 61):
        name, srcs = {49: "lwc1", 53: "ldc1", 57: "swc1", 61: "sdc1"}[op], (rs,)
    elif op == 16 and rs in (0, 1):
        name, dst = "mfc0", rt
    elif op == 17:
        if rs in (0, 1, 2):
            name, dst = ("mfc1", "dmfc1", "cfc1")[rs], rt
        elif rs in (4, 5, 6):
            name, srcs = ("mtc1", "dmtc1", "ctc1")[rs - 4], (rt,)
        elif rs == 8:
            name, kind, likely, target = "bc1", "branch", bool(rt & 2), branch_target
        else:
            name = "cop1"
    elif op == 47:
        name, srcs = "cache", (rs,)
    return Ins(at, word, name, rs, rt, rd, sa, imm, simm, dst, tuple(r for r in srcs if r), kind, likely, target,
               width, store, signed)


def decode_range(code: bytes, vram: int) -> list[Ins]:
    n = len(code) // 4
    return [decode(w, vram + 4 * i) for i, w in enumerate(struct.unpack_from(f">{n}I", code))]


def hx(value: int) -> str:
    return f"0x{value & M32:08X}"


# ---------------------------------------------------------------------------------------------
# Loops and their classification

def find_loops(insns: list[Ins]) -> list[tuple[int, int]]:
    """Back edges as ``(head index, tail index)``; the tail includes the branch's delay slot."""
    if not insns:
        return []
    base, n = insns[0].at, len(insns)
    out = set()
    for i, ins in enumerate(insns):
        if ins.kind in ("branch", "jump") and ins.target is not None and base <= ins.target <= ins.at:
            out.add(((ins.target - base) // 4, min(i + 1, n - 1)))
    return sorted(out)


def is_move(ins: Ins) -> bool:
    """A register copy (``addu``/``or`` with $zero, ``addiu`` of 0); with no source, a move of 0."""
    if ins.name in _MOVES and len(ins.srcs) <= 1 and (ins.rs == 0 or ins.rt == 0):
        return True
    return ins.name in ("addiu", "daddiu") and ins.simm == 0


def derived(body: list[Ins]) -> dict[int, frozenset]:
    """Registers whose value (somewhere in the body) derives from an integer load, with the kinds
    of operation applied on the way (flow-insensitive, to a fixed point)."""
    tags: dict[int, frozenset] = {}
    for _ in range(8):
        changed = False
        for ins in body:
            if not ins.dst:
                continue
            if ins.width and not ins.store:
                new = frozenset({"load"})
            elif ins.name in ("lwl", "lwr"):
                new = frozenset({"load"})
            else:
                parts = [tags[r] for r in ins.srcs if r in tags]
                if not parts:
                    continue
                new = frozenset().union(*parts)
                if not is_move(ins):
                    new |= {OPCLASS.get(ins.name, "other")}
            old = tags.get(ins.dst, frozenset())
            if not new <= old:
                tags[ins.dst] = old | new
                changed = True
        if not changed:
            break
    return tags


def _defs(body: list[Ins], reg: int) -> list[Ins]:
    return [ins for ins in body if ins.dst == reg]


def _def_map(body: list[Ins]) -> dict[int, list[Ins]]:
    """Register -> its definitions in body order (one pass; ``_defs`` for every register)."""
    out: dict[int, list[Ins]] = {}
    for ins in body:
        if ins.dst:
            out.setdefault(ins.dst, []).append(ins)
    return out


def _window(body: list[Ins], reg: int, depth: int = 3, defs: dict | None = None,
            memo: dict | None = None) -> int | None:
    """The power-of-two window an ``andi`` mask gives a back-reference distance."""
    defs = _def_map(body) if defs is None else defs
    memo = {} if memo is None else memo
    if (reg, depth) in memo:
        return memo[(reg, depth)]
    found = None
    for ins in defs.get(reg, ()):
        if ins.name == "andi" and ins.imm and not (ins.imm + 1) & ins.imm:
            found = ins.imm + 1
            break
        if depth and (ins.name in ("addiu", "daddiu") or is_move(ins)) and ins.srcs:
            w = _window(body, ins.srcs[0], depth - 1, defs, memo)
            if w:
                found = w
                break
    memo[(reg, depth)] = found
    return found


def classify_loop(body: list[Ins]) -> dict:
    tags = derived(body)
    loads = [i for i in body if i.width and not i.store and i.dst]
    stores = [i for i in body if i.store and i.width]
    load_bases = {i.rs for i in loads}
    store_bases = {i.rs for i in stores}
    stride: dict[int, int | None] = {}
    for ins in body:
        if ins.dst and ins.dst in load_bases | store_bases:
            if ins.name in ("addiu", "daddiu", "addi") and ins.rs == ins.dst:
                if stride.get(ins.dst, 0) is not None:
                    stride[ins.dst] = stride.get(ins.dst, 0) + ins.simm
            elif ins.dst in (ins.rs, ins.rt) and ins.name in ("addu", "daddu", "subu"):
                stride[ins.dst] = None                       # advanced by a register
    out_stores = [s for s in stores if s.rt in tags]
    facts: dict = {"loads": len(loads), "stores": len(stores)}

    # back reference: load base = store pointer -/+ a loaded-derived distance
    backref, window = False, None
    defs, memo = _def_map(body), {}
    for ld in loads:
        for d in defs.get(ld.rs, ()):
            if d.name in ("subu", "addu", "dsubu", "daddu") and len(d.srcs) == 2:
                ptr, dist = (d.rs, d.rt)
                if d.name in ("addu", "daddu") and ptr not in store_bases:
                    ptr, dist = dist, ptr
                if ptr in store_bases and dist in tags and ld.rs not in store_bases:
                    backref = True
                    window = window or _window(body, dist, defs=defs, memo=memo)
    # bit buffer: shifted in place, and a loaded value or masked against one
    bitbuf = False
    masked = {r for a in body if a.name in ("and", "andi") for r in a.srcs
              if any(s in tags for s in a.srcs if s != r)}
    for ins in body:
        if OPCLASS.get(ins.name) == "shift" and ins.dst and ins.dst == ins.rt:
            r = ins.dst
            if "load" in tags.get(r, ()) or r in masked:
                bitbuf = True
    # read-modify-write at one address
    word_loads = {(ld.rs, ld.simm) for ld in loads if ld.width == 4}
    rmw = any(s.width == 4 and (s.rs, s.simm) in word_loads and s.rt in tags and "arith" in tags[s.rt]
              for s in out_stores)
    data_ops = any(tags[s.rt] & {"shift", "mask", "logic", "arith"} for s in out_stores)
    copy = None
    loads_into: dict[int, list[Ins]] = {}
    for ld in loads:
        loads_into.setdefault(ld.dst, []).append(ld)
    for s in out_stores:
        if tags[s.rt] != {"load"}:
            continue
        for ld in loads_into.get(s.rt, ()):
            if ld.width == s.width and ld.rs != s.rs:
                a, b = stride.get(ld.rs), stride.get(s.rs)
                if a is not None and a == b and a != 0:
                    copy = {"width": s.width, "stride": a}
    accumulator = None
    walking = any(stride.get(ld.rs) for ld in loads)
    for ins in body:
        if (ins.name in ("addu", "xor", "or", "subu", "daddu", "add") and ins.dst and ins.dst in ins.srcs
                and any(r in tags and r != ins.dst for r in ins.srcs) and ins.dst not in load_bases | store_bases):
            accumulator = ins.dst
    if out_stores and (backref or bitbuf):
        cls = "transform"
    elif rmw:
        cls = "fixup"
    elif out_stores and data_ops:
        cls = "transform"
    elif copy:
        cls = "copy"
    elif not out_stores and accumulator is not None and walking:
        cls = "checksum"
    elif stores and not loads and any(stride.get(s.rs) for s in stores):
        cls = "fill"
    else:
        cls = "other"
    facts.update({"class": cls, "lz": backref, "window": window, "bit_buffer": bitbuf})
    if copy:
        facts.update(copy)
    if accumulator is not None and cls == "checksum":
        facts["accumulator"] = accumulator
    return facts


# ---------------------------------------------------------------------------------------------
# Linear constant / origin propagation (PI registers, call arguments)

class Tracker:
    """Straight-line propagation in address order (branches are not followed, as in
    ``n64dk.assets.loaders``): constants, which argument registers a value came from (with an
    additive constant when it is one argument plus a constant), and $sp slots."""

    def __init__(self, named_args: bool = True):
        self.const: dict[int, int] = {}
        self.origin: dict[int, tuple] = {r: (frozenset({n}), 0) for r, n in ARG_NAMES.items()} if named_args else {}
        self.slots: dict[int, tuple] = {}

    def c(self, r: int) -> int | None:
        return 0 if r == 0 else self.const.get(r)

    def _set(self, r: int, const: int | None, origin: tuple | None) -> None:
        if not r:
            return
        if const is None:
            self.const.pop(r, None)
        else:
            self.const[r] = const & M32
        if origin is None:
            self.origin.pop(r, None)
        else:
            self.origin[r] = origin

    def step(self, ins: Ins) -> dict | None:
        """Apply one non-control instruction; a store returns ``{at, addr, width, value, origin}``."""
        name = ins.name
        if ins.store and ins.width:
            base = self.c(ins.rs)
            if ins.rs == SP:
                self.slots[ins.simm] = (self.c(ins.rt), self.origin.get(ins.rt))
            return {"at": ins.at, "addr": None if base is None else (base + ins.simm) & M32, "width": ins.width,
                    "value": self.c(ins.rt), "origin": self.origin.get(ins.rt)}
        if not ins.dst:
            return None
        if name == "lui":
            self._set(ins.dst, ins.imm << 16, None)
        elif name in ("addiu", "addi", "daddiu", "daddi"):
            c, o = self.c(ins.rs), self.origin.get(ins.rs)
            self._set(ins.dst, None if c is None else c + ins.simm,
                      None if o is None else (o[0], None if o[1] is None else o[1] + ins.simm))
        elif name in ("ori", "xori", "andi"):
            c, o = self.c(ins.rs), self.origin.get(ins.rs)
            op = {"ori": int.__or__, "xori": int.__xor__, "andi": int.__and__}[name]
            self._set(ins.dst, None if c is None else op(c, ins.imm),
                      None if o is None else (o[0], o[1] if name == "ori" and ins.imm == 0 else None))
        elif ins.width and not ins.store:
            if ins.rs == SP and ins.simm in self.slots:
                c, o = self.slots[ins.simm]
                self._set(ins.dst, c, o)
            else:
                self._set(ins.dst, None, None)
        elif name in ("addu", "subu", "or", "and", "xor", "nor", "daddu", "dsubu", "add", "sub") and \
                OPCLASS.get(name) in ("arith", "logic", "mask"):
            a, b = self.c(ins.rs), self.c(ins.rt)
            const = None
            if a is not None and b is not None:
                const = {"addu": a + b, "add": a + b, "daddu": a + b, "subu": a - b, "sub": a - b,
                         "dsubu": a - b, "or": a | b, "and": a & b, "xor": a ^ b, "nor": ~(a | b)}[name]
            oa, ob = self.origin.get(ins.rs), self.origin.get(ins.rt)
            if is_move(ins):
                origin = oa if ins.rt == 0 else ob
            elif oa or ob:
                args = (oa[0] if oa else frozenset()) | (ob[0] if ob else frozenset())
                addend = None
                if name in ("addu", "add", "daddu") and oa and b is not None and oa[1] is not None and not ob:
                    addend = oa[1] + b
                elif name in ("addu", "add", "daddu") and ob and a is not None and ob[1] is not None and not oa:
                    addend = ob[1] + a
                origin = (args, addend)
            else:
                origin = None
            self._set(ins.dst, const, origin)
        else:
            self._set(ins.dst, None, None)
        return None

    def call(self) -> None:
        """After a call: caller-saved registers are unknown; $v0 keeps $a0's origin (a translation
        such as ``osVirtualToPhysical`` passes its argument through)."""
        a0 = self.origin.get(4)
        for r in CALLER_SAVED:
            self.const.pop(r, None)
            self.origin.pop(r, None)
        if a0 is not None:
            self.origin[V0] = (a0[0], None)


def walk(insns: list[Ins], tracker: Tracker | None = None):
    """Yield ``(ins, tracker, store_record)`` in address order; a call yields after its delay slot
    has been applied (arguments as the callee sees them) and before the call clobbers anything."""
    t = tracker or Tracker()
    i, n = 0, len(insns)
    while i < n:
        ins = insns[i]
        if ins.kind in ("call", "jalr"):
            rec = t.step(insns[i + 1]) if i + 1 < n and not insns[i + 1].kind else None
            yield ins, t, rec
            t.call()
            i += 2
            continue
        rec = t.step(ins) if not ins.kind else None
        yield ins, t, rec
        i += 1


def _role(origin: tuple | None) -> dict | None:
    if not origin or len(origin[0]) != 1:
        return None
    return {"arg": next(iter(origin[0])), "addend": origin[1]}


def pi_dma(insns: list[Ins]) -> dict | None:
    """Stores to the PI DMA registers in one function, with their argument roles; None when the
    function does not write a DMA address and a length."""
    writes = []
    for _ins, _t, rec in walk(insns):
        if rec and rec["addr"] is not None and rec["width"] == 4:
            reg = PI_REGS.get(rec["addr"] & 0x1FFFFFFF)
            if reg and reg != "status":
                writes.append((reg, rec))
    regs = {reg for reg, _rec in writes}
    if not ({"dram", "cart"} <= regs and regs & {"rd_len", "wr_len"}):
        return None
    last = {reg: rec for reg, rec in writes}
    length = "wr_len" if "wr_len" in last else "rd_len"
    out = {"direction": "cart-to-ram" if length == "wr_len" else "ram-to-cart",
           "writes": [{"reg": reg, "at": hx(rec["at"]), "value": None if rec["value"] is None else hx(rec["value"]),
                       "role": _role(rec["origin"])} for reg, rec in writes],
           "roles": {"dram": _role(last["dram"]["origin"]), "cart": _role(last["cart"]["origin"]),
                     "len": _role(last[length]["origin"])}}
    dram, cart, size = last["dram"]["value"], last["cart"]["value"], last[length]["value"]
    if None not in (dram, cart, size):
        out["range"] = _range(cart, dram, size + 1)
    return out


def _rom_offset(value: int) -> int:
    v = value & M32
    if 0x10000000 <= (v & 0x1FFFFFFF) < 0x20000000 and v >= 0x10000000:
        return (v & 0x1FFFFFFF) - CART_BASE
    return v


def _range(cart: int, dram: int, size: int) -> dict:
    rom = _rom_offset(cart)
    return {"rom": hx(rom), "rom_end": hx(rom + size), "ram": hx(dram | 0x80000000 if dram < 0x00800000 else dram),
            "size": size}


def call_constants(insns: list[Ins]) -> list[tuple[int, int, tuple]]:
    """``(site, target, (a0, a1, a2, a3))`` for every ``jal`` with at least one constant argument."""
    out = []
    for ins, t, _rec in walk(insns, Tracker(named_args=False)):
        if ins.kind == "call" and ins.name == "jal":
            args = tuple(t.c(r) for r in (4, 5, 6, 7))
            if any(a is not None for a in args):
                out.append((ins.at, ins.target, args))
    return out


def dma_ranges(pi_functions: dict[int, dict], calls: list[tuple[int, int, tuple]]) -> list[dict]:
    """ROM -> RAM ranges at call sites of PI DMA functions whose role arguments are constant."""
    out = []
    for site, target, args in calls:
        pi = pi_functions.get(target)
        if not pi:
            continue
        values = {}
        for key in ("dram", "cart", "len"):
            role = pi["roles"].get(key)
            if not role or role["arg"] not in ("a0", "a1", "a2", "a3"):
                break
            value = args[int(role["arg"][1])]
            if value is None:
                break
            values[key] = value + (role["addend"] or 0) if key == "len" else value
        else:
            if pi["roles"]["len"]["addend"] is None:
                continue
            out.append({"site": hx(site), "function": hx(target), **_range(values["cart"], values["dram"],
                                                                            (values["len"] + 1) & M32)})
    return out


def summarise(code: bytes, vram: int, insns: list[Ins] | None = None) -> dict:
    """Loop classes and PI DMA role of one function (``insns``: already decoded ``code``)."""
    insns = insns if insns is not None else decode_range(code, vram)
    loops = []
    for head, tail in find_loops(insns):
        facts = classify_loop(insns[head:tail + 1])
        loops.append({"head": hx(insns[head].at), "tail": hx(insns[tail].at), **facts})
    classes = {loop["class"] for loop in loops}
    out = {"vram": hx(vram), "size": len(code), "class": next((c for c in PRECEDENCE if c in classes), None),
           "loops": loops, "lz": any(loop["lz"] for loop in loops)}
    windows = [loop["window"] for loop in loops if loop["window"]]
    out["window"] = max(windows) if windows else None
    pi = pi_dma(insns)
    if pi:
        out["pi_dma"] = pi
    return out


# ---------------------------------------------------------------------------------------------
# Dynamic capture

SCHEMA = "n64dk.loader-capture/1"
STACK_WINDOW = 0x10000


@dataclass(frozen=True)
class Limits:
    max_instructions: int = 20_000_000
    max_seconds: float = 120.0             # checked when the run ends: a failure, never a result
    max_output: int = 0x400000

    def as_dict(self) -> dict:
        return {"max_instructions": self.max_instructions, "max_seconds": self.max_seconds,
                "max_output": self.max_output}


def bare_machine(rom: bytes = b"", ram_size: int = 0x800000):
    """A machine with no boot state (``Machine(None)``), read tracking on, and ``rom`` as the
    cartridge (readable at 0xB0000000 and by PI DMA)."""
    from .. import emu
    m = emu.Machine(None, ram_size=ram_size, track_reads=True)
    m.rom = bytes(rom)
    return m


def _sha(data) -> str:
    return hashlib.sha256(data).hexdigest()


def _runs(flags: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for lo, hi in sorted(flags):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _changed(before: bytes, after: bytearray, lines: bytes, shift: int) -> list[list[int]]:
    """Byte runs that differ, looked for only in the written lines."""
    runs: list[list[int]] = []
    i, n = 0, len(lines)
    while (i := lines.find(1, i)) >= 0:
        j = lines.find(0, i)
        j = n if j < 0 else j
        lo, hi = i << shift, min(j << shift, len(after))
        if before[lo:hi] != after[lo:hi]:
            k = lo
            while k < hi:
                if before[k] != after[k]:
                    start = k
                    while k < hi and before[k] != after[k]:
                        k += 1
                    if runs and runs[-1][1] == start:
                        runs[-1][1] = k
                    else:
                        runs.append([start, k])
                else:
                    k += 1
        i = j
    return runs


def _split(lo: int, hi: int, cuts: list[tuple[int, int]]) -> list[tuple[int, int, bool]]:
    """``[lo, hi)`` cut into pieces inside / outside the union of ``cuts``."""
    edges = {lo, hi}
    for a, b in cuts:
        edges.update(x for x in (a, b) if lo < x < hi)
    points = sorted(edges)
    return [(a, b, any(c <= a and b <= d for c, d in cuts)) for a, b in zip(points, points[1:])]


def _key(entry: int, inputs: dict, limits: Limits) -> str:
    payload = {"schema": SCHEMA, "entry": hx(entry), "inputs": inputs,
               "limits": {"max_instructions": limits.max_instructions, "max_output": limits.max_output}}
    return _sha(json.dumps(payload, sort_keys=True).encode())


def capture(machine, entry: int, args: tuple = (), limits: Limits | None = None, *, sp: int | None = None,
            output: tuple[int, int] | None = None, preloaded: tuple = ()) -> dict:
    """Call ``entry`` with ``args`` on ``machine`` until it returns; the receipt (see the module
    docstring).  ``output``: the RAM range (virtual, ``[lo, hi)``) holding the result, when the
    caller knows it; otherwise the hull of the ROM-derived bytes written.  ``preloaded``: RAM the
    caller filled with ROM bytes before the call, ``(ram vaddr, rom offset, size)`` triples; they are
    taint sources like DMA.  The machine is left in its final state (its write and read logs keep
    what they held before plus this run)."""
    from .. import emu
    limits = limits or Limits()
    m = machine
    entry &= M32
    sp = (0x80000000 | (m.ram_size - 0x100)) if sp is None else sp & M32
    args = tuple(a & M32 for a in args[:4])
    # the register state the routine starts with, as Machine.call will set it
    g = m.gpr
    for i, value in enumerate(args):
        g[4 + i] = emu.sx32(value)
    g[29], g[31] = emu.sx32(sp), emu.sx32(m.RETURN_TRAP)
    ram_before = bytes(m.ram)
    inputs = {"args": [hx(a) for a in args], "sp": hx(sp),
              "ram_sha256": _sha(ram_before), "rom_sha256": _sha(m.rom), "rom_size": len(m.rom),
              "regs_sha256": _sha(struct.pack(">32Q32Q", *(r & 0xFFFFFFFFFFFFFFFF for r in g),
                                              *(f & 0xFFFFFFFFFFFFFFFF for f in m.fpr))),
              "cop0_status": hx(m.cop0[12]), "count": m.count,
              "preloaded": [{"ram": hx(r), "rom": hx(o), "size": n} for r, o, n in preloaded],
              "output": None if output is None else [hx(output[0]), hx(output[1])]}
    key = _key(entry, inputs, limits)
    receipt = {"schema": SCHEMA, "status": "failed", "reason": "", "entry": hx(entry), "inputs": inputs,
               "rom_ranges": [], "ram_range": None, "ram_written": [], "output_sha256": None, "output_size": 0,
               "instructions": 0, "limits": limits.as_dict(), "cache_key": key, "strict": False,
               "evidence": "behaviour of the routine on the kit's interpreter; never byte evidence"}
    # fresh logs for this run (restored as unions afterwards)
    dma_mark = len(m.dmas)
    cart_before = set(m.cart_pages)
    m.cart_pages.clear()
    probes_before = set(m.probes)
    dirty_saved, reads_saved = bytes(m.dirty), bytes(m.reads)
    m.dirty[:] = bytes(len(m.dirty))
    m.reads[:] = bytes(len(m.reads))
    if not m.track_reads:                       # translated code reads the flag when it is built
        m.track_reads = True
        m.decoded.clear()
        m._templates.clear()
        m.code_pages.clear()
    start_steps = m.steps
    started = time.monotonic()
    reason = m.call(entry, args, sp=sp, max_steps=limits.max_instructions)
    elapsed = time.monotonic() - started
    receipt["instructions"] = m.steps - start_steps
    written_lines, read_lines = bytes(m.dirty), bytes(m.reads)
    m.dirty[:] = bytes(a | b for a, b in zip(dirty_saved, written_lines))
    m.reads[:] = bytes(a | b for a, b in zip(reads_saved, read_lines))
    cart_pages = sorted(m.cart_pages)
    m.cart_pages |= cart_before
    dmas = m.dmas[dma_mark:]
    new_probes = sorted(set(m.probes) - probes_before)

    # ROM inputs
    rom_ranges = [{"rom": hx(d.rom), "rom_end": hx(d.rom + d.size), "via": "pi-dma", "ram": hx(d.ram)}
                  for d in dmas]
    for lo, hi in _runs([(p << 12, (p + 1) << 12) for p in cart_pages]):
        rom_ranges.append({"rom": hx(lo), "rom_end": hx(min(hi, len(m.rom))), "via": "cart-read", "granule": 4096})
    line = emu.LINE
    sources = [(d.ram & 0x1FFFFFFF, (d.ram & 0x1FFFFFFF) + d.size) for d in dmas]
    for ram, rom, size in preloaded:
        p = ram & 0x1FFFFFFF
        was_read = any(read_lines[x >> line] for x in range(p, p + size, 1 << line) if x >> line < len(read_lines))
        rom_ranges.append({"rom": hx(rom), "rom_end": hx(rom + size), "via": "preloaded", "ram": hx(ram),
                           "read": was_read})
        sources.append((p, p + size))
    receipt["rom_ranges"] = rom_ranges
    read_rom = bool(cart_pages) or any(
        read_lines[x >> line] for lo, hi in sources for x in range(lo & ~((1 << line) - 1), hi, 1 << line)
        if x >> line < len(read_lines))

    # RAM written, with the coarse taint
    dma_dest = [(d.ram & 0x1FFFFFFF, (d.ram & 0x1FFFFFFF) + d.size) for d in dmas]
    stack = ((sp & 0x1FFFFFFF) - STACK_WINDOW, (sp & 0x1FFFFFFF) + 0x10)
    written = []
    for lo, hi in _changed(ram_before, m.ram, written_lines, line):
        for a, b, in_dma in _split(lo, hi, dma_dest):
            for c, d, in_stack in _split(a, b, [stack]):
                taint = ("rom-dma" if in_dma else "stack" if in_stack else
                         "rom-derived" if read_rom else "untainted")
                if written and written[-1][2] == taint and written[-1][1] == c:
                    written[-1][1] = d
                else:
                    written.append([c, d, taint])
    receipt["ram_written"] = [{"ram": hx(0x80000000 | a), "end": hx(0x80000000 | b), "taint": t}
                              for a, b, t in written]

    def fail(why: str) -> dict:
        receipt["reason"] = why
        return receipt

    if reason != "returned":
        return fail(f"did not return: {reason} at {hx(m.pc)}")
    if new_probes:
        return fail("read open bus at " + ", ".join(hx(p) for p in new_probes[:4]))
    if elapsed > limits.max_seconds:
        return fail("time limit exceeded")
    if output is not None:
        lo, hi = output[0] & 0x1FFFFFFF, output[1] & 0x1FFFFFFF
    else:
        derived_runs = [(a, b) for a, b, t in written if t in ("rom-dma", "rom-derived")]
        if not derived_runs:
            return fail("no ROM-derived output")
        lo, hi = min(a for a, _b in derived_runs), max(b for _a, b in derived_runs)
    if not 0 <= lo < hi <= m.ram_size:
        return fail("output range outside RAM")
    if hi - lo > limits.max_output:
        return fail(f"output of {hi - lo} bytes exceeds max_output")
    receipt.update({"status": "ok", "reason": "returned", "ram_range": [hx(0x80000000 | lo), hx(0x80000000 | hi)],
                    "output_sha256": _sha(bytes(m.ram[lo:hi])), "output_size": hi - lo})
    return receipt


def capture_output(machine, receipt: dict) -> bytes:
    """The bytes an ``ok`` receipt describes, checked against its hash."""
    if receipt.get("status") != "ok":
        raise ValueError(f"capture failed: {receipt.get('reason')}")
    lo, hi = (int(x, 16) & 0x1FFFFFFF for x in receipt["ram_range"])
    data = bytes(machine.ram[lo:hi])
    if _sha(data) != receipt["output_sha256"]:
        raise ValueError("RAM no longer holds the captured output")
    return data


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def register_image(receipt: dict, name: str, *, path: str | None = None, catalogue: str | None = None) -> dict:
    """A proposal for ``[images.<name>]`` (returned, never written): the captured output as an image
    linked where it was written, its ROM source recorded as a non-copy transform, and the receipt's
    key as provenance.  Refuses a failed capture."""
    if receipt.get("status") != "ok":
        raise ValueError(f"refusing to register a failed capture: {receipt.get('reason')}")
    if not _NAME.match(name):
        raise ValueError(f"not an image name: {name!r}")
    lo, hi = (int(x, 16) for x in receipt["ram_range"])
    sources = sorted(int(r["rom"], 16) for r in receipt["rom_ranges"] if r.get("read", True))
    key = receipt["cache_key"]
    transform = f"capture:{key[:16]}"
    table = {"path": path or f"decoded/{name}.bin", "vram_delta": hx(lo), "extent": [hx(lo), hx(hi)],
             "raw_segments": [[hx(sources[0]), "0x0", f"0x{hi - lo:X}", transform]] if sources else [],
             "provenance": transform}
    if catalogue:
        table["catalogue"] = catalogue
    lines = [f"[images.{name}]                # proposal: output of {receipt['entry']} "
             f"(capture {key[:16]}); not applied",
             f'path = "{table["path"]}"',
             f"vram_delta = {table['vram_delta']}",
             f"extent = [{hx(lo)}, {hx(hi)}]"]
    if sources:
        lines.append(f'raw_segments = [[{hx(sources[0])}, 0x0, 0x{hi - lo:X}, "{transform}"]]')
    if catalogue:
        lines.append(f'catalogue = "{catalogue}"')
    lines.append(f'provenance = "{transform}"')
    return {"name": name, "applied": False, "strict": False, "table": table, "toml": "\n".join(lines) + "\n",
            "bytes": {"size": hi - lo, "sha256": receipt["output_sha256"]},
            "provenance": {"kind": "dynamic-capture", "cache_key": key, "entry": receipt["entry"],
                           "inputs": receipt["inputs"], "rom_ranges": receipt["rom_ranges"],
                           "instructions": receipt["instructions"], "schema": receipt["schema"]},
            "note": "decoded bytes are behavioural output of the game's routine; the registry reports their "
                    "ROM mapping as not representable, and nothing here grants byte or strict evidence"}
