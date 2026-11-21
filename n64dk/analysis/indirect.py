"""Indirect control flow: where ``jr $reg`` and ``jalr`` can go, with the evidence for it.

A function catalogue built from direct ``jal`` targets and linear walks cannot see three things
that compiled N64 code does all the time:

* **switch jump tables**: ``sltiu at, idx, N; beqz at, default; sll t, idx, 2; lui at, %hi(T);
  addu at, at, t; lw t, %lo(T)(at); jr t`` (IDO), or the KMC/GCC form ``lui v1, %hi(T); addiu v1,
  v1, %lo(T); addu v0, v0, v1; lw v0, 0(v0); jr v0``, with an optional lower bound subtracted
  first (``addiu t, idx, -lo``) and the shift either in the branch delay slot or after it;
* **function-pointer tables**: arrays of code addresses in data, stride 4 or a structure stride
  with the pointer at a fixed offset, loaded with ``lw t9, off(base)`` and called with ``jalr``;
* **pointers in registers or memory**: ``la t9, func; jalr t9``, a constant stored to the stack
  or a global and loaded back, or a global another function filled in.

``analyse_function`` runs a small value-set analysis over one function's control flow graph
(:mod:`n64dk.analysis.flow`).  A register holds a constant, a linear function ``scale * s + add``
of an unknown symbol ``s``, a bounded arithmetic progression (an index the code has checked with
``sltiu``/``sltu`` or masked with ``andi 2^k-1``), a word loaded from a constant address, the words
of a bounded table, or an unbounded table scan.  A conditional branch on an ``sltiu`` result
refines the index symbol on the edge where the comparison held, so the bound applies even when
the shift sits in the delay slot.  Joins keep a value only when every predecessor agrees.

Every ``jr``/``jalr`` site gets one :class:`Site` with a status:

``definite``
    a bounded switch table whose every entry lands inside the function (its bytes are then owned
    by the function: ``FunctionResult.owned``), or a single constant target;
``finite``
    a candidate set that is bounded but not proven complete: pointer tables (bounded or scanned
    while the entries are code), switch tables without a bound check, globals whose stored code
    pointers are all known;
``unresolved``
    no value (an argument, a field of an unknown structure, a global written with unknown values);
``impossible``
    every candidate is misaligned or outside code.

``analyse_image`` runs it over every catalogued function of an image, resolves pointers stored to
globals across functions, and propagates non-returning functions (infinite loops, or every path
through a call to a non-returning function) to a fixpoint, flagging the "fall-through" after such
calls as fake.  Everything here is evidence and reporting: it never edits a catalogue and never
grants credit.
"""
from __future__ import annotations

import bisect
import struct
from collections import Counter
from dataclasses import dataclass, field, replace

from .. import mips
from . import flow as flowmod
from .flow import CALL_CLOBBERED, SP
from .values import _reverse_postorder

MASK = 0xFFFFFFFF
MAX_TABLE = 1024
DEFINITE, FINITE, UNRESOLVED, IMPOSSIBLE = "definite", "finite", "unresolved", "impossible"
STATUSES = (DEFINITE, FINITE, UNRESOLVED, IMPOSSIBLE)
_SP = ("lin", ("sp",), 1, 0)          # the stack pointer at entry
_PASSES = 8                            # dataflow passes per round
_ROUNDS = 4                            # re-analyse with the tables found so far (case blocks become reachable)


def hx(value: int) -> str:
    return f"0x{value & MASK:08X}"


# --------------------------------------------------------------------------- abstract values

def _c(value: int) -> tuple:
    return ("c", value & MASK)


def _lin(sid, scale: int, add: int) -> tuple:
    scale &= MASK
    if scale == 0:
        return _c(add)
    return ("lin", sid, scale, add & MASK)


def _set(base: int, step: int, count: int) -> tuple:
    if count <= 1 or step & MASK == 0:
        return _c(base)
    return ("set", base & MASK, step & MASK, count)


def _add_const(value: tuple, k: int):
    kind = value[0]
    if kind == "c":
        return _c(value[1] + k)
    if kind == "lin":
        return _lin(value[1], value[2], value[3] + k)
    if kind == "set":
        return _set(value[1] + k, value[2], value[3])
    return None


def _add(a: tuple, b: tuple):
    if a[0] == "c":
        return _add_const(b, a[1])
    if b[0] == "c":
        return _add_const(a, b[1])
    if a[0] == "lin" and b[0] == "lin" and a[1] == b[1]:
        return _lin(a[1], a[2] + b[2], a[3] + b[3])
    return None


def _shl(value: tuple, sa: int):
    kind = value[0]
    if kind == "c":
        return _c(value[1] << sa)
    if kind == "lin":
        return _lin(value[1], value[2] << sa, value[3] << sa)
    if kind == "set":
        return _set(value[1] << sa, value[2] << sa, value[3])
    return None


def _refine(state, sid, add: int, bound: int):
    """The state on the edge where ``(s + add) <u bound`` held: every ``scale * s + a`` becomes the
    progression ``a - scale * add + scale * k`` for k in [0, bound)."""
    def sub(value):
        if value[0] == "lin" and value[1] == sid:
            return _set(value[3] - value[2] * add, value[2], bound)
        return value
    regs, slots = state
    return {r: sub(v) for r, v in regs.items()}, {k: sub(v) for k, v in slots.items()}


def _merge(states):
    states = [s for s in states if s is not None]
    if not states:
        return None
    regs, slots = dict(states[0][0]), dict(states[0][1])
    for other_regs, other_slots in states[1:]:
        for key in list(regs):
            if other_regs.get(key) != regs[key]:
                del regs[key]
        for key in list(slots):
            if other_slots.get(key) != slots[key]:
                del slots[key]
    return regs, slots


def _forget(slots: dict, keep_stack: bool = True) -> None:
    """Drop what memory may have changed: everything but the function's own stack slots."""
    for key in list(slots):
        if not (keep_stack and key[0] == ("sp",)):
            del slots[key]


# --------------------------------------------------------------------------- results

@dataclass(frozen=True)
class Site:
    site: int
    function: int
    call: bool                         # jalr (True) or jr (False)
    kind: str                          # switch, switch-unbounded, pointer-table, constant, local-jump,
    #                                    stored-pointer, unknown
    status: str                        # definite / finite / unresolved / impossible
    targets: tuple = ()
    rejected: tuple = ()               # candidate words that cannot be instruction addresses
    table: int | None = None           # table address (or the global a stored pointer is loaded from)
    count: int = 0                     # table entries read
    stride: int = 0
    provenance: tuple = ()

    def as_dict(self) -> dict:
        return {"site": hx(self.site), "function": hx(self.function), "call": self.call, "kind": self.kind,
                "status": self.status, "targets": [hx(t) for t in self.targets],
                "rejected": [hx(t) for t in self.rejected], "table": hx(self.table) if self.table is not None
                else None, "count": self.count, "stride": self.stride, "provenance": list(self.provenance)}


@dataclass
class FunctionResult:
    start: int
    end: int
    sites: list[Site] = field(default_factory=list)
    tables: dict[int, list[int]] = field(default_factory=dict)     # jr site -> in-function targets
    owned: list[tuple[int, int]] = field(default_factory=list)     # (address, bytes) of owned tables
    stores: list[tuple[int, int | None, int]] = field(default_factory=list)   # (global, constant|None, site)
    escaped: set[int] = field(default_factory=set)                 # constants stored outside the stack
    shape: "Shape | None" = None


@dataclass
class Shape:
    """What non-return propagation needs of a function: per block, the calls in order, how the
    block leaves the function (``return``, ``tail`` + target, ``open``) and its successors."""
    start: int
    end: int
    blocks: list[tuple[tuple, tuple | None, tuple]]


# --------------------------------------------------------------------------- one function

class _Interp:
    def __init__(self, fl: flowmod.Flow, read_word, max_table: int):
        self.fl, self.read_word, self.max_table = fl, read_word, max_table
        self.block_at = {block.first: block.number for block in fl.blocks}
        self.sites: dict[int, tuple] = {}            # site -> (value, call, block number)
        self.stores: list[tuple[int, int | None, int]] = []
        self.escaped: set[int] = set()

    def index_of(self, address):
        fl = self.fl
        if address is None or address & 3 or not fl.vram <= address < fl.end:
            return None
        return (address - fl.vram) // 4

    # registers ---------------------------------------------------------------------------
    @staticmethod
    def get(regs: dict, reg: int, first: int) -> tuple:
        if reg == 0:
            return ("c", 0)
        value = regs.get(reg)
        if value is None:
            value = regs[reg] = ("lin", ("in", first, reg), 1, 0)
        return value

    @staticmethod
    def put(regs: dict, reg: int, value, site: int) -> None:
        if reg == 0:
            return
        regs[reg] = value if value is not None else ("lin", ("def", site, reg), 1, 0)

    # one instruction -----------------------------------------------------------------------
    def step(self, insn, regs: dict, slots: dict, first: int) -> None:
        if insn.op == ".word" or not insn.word:
            return
        w, site = insn.word, insn.vram
        op, rs, rt, rd = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31
        sa, funct, imm = (w >> 6) & 31, w & 0x3F, w & 0xFFFF
        simm = imm - 0x10000 if imm & 0x8000 else imm
        get, put = self.get, self.put
        if op == 15:                                             # lui
            put(regs, rt, _c(imm << 16), site)
        elif op in (9, 25):                                      # addiu / daddiu
            put(regs, rt, _add_const(get(regs, rs, first), simm), site)
        elif op == 13:                                           # ori
            value = get(regs, rs, first)
            put(regs, rt, _c(value[1] | imm) if value[0] == "c" else None, site)
        elif op == 12:                                           # andi: a 2^k-1 mask bounds an index
            value = get(regs, rs, first)
            if value[0] == "c":
                result = _c(value[1] & imm)
            elif imm and not imm & (imm + 1) and imm + 1 <= self.max_table:
                result = _set(0, 1, imm + 1)
            else:
                result = None
            put(regs, rt, result, site)
        elif op == 11:                                           # sltiu
            put(regs, rt, self._less(get(regs, rs, first), simm & MASK), site)
        elif op == 0 and funct in (0x21, 0x2D):                  # addu / daddu
            put(regs, rd, _add(get(regs, rs, first), get(regs, rt, first)), site)
        elif op == 0 and funct == 0x25:                          # or (move when one side is $zero)
            if rt == 0:
                result = get(regs, rs, first)
            elif rs == 0:
                result = get(regs, rt, first)
            else:
                a, b = get(regs, rs, first), get(regs, rt, first)
                result = _c(a[1] | b[1]) if a[0] == "c" and b[0] == "c" else None
            put(regs, rd, result, site)
        elif op == 0 and funct in (0x23, 0x2F):                  # subu / dsubu
            b = get(regs, rt, first)
            put(regs, rd, _add_const(get(regs, rs, first), -b[1]) if b[0] == "c" else None, site)
        elif op == 0 and funct == 0x2B:                          # sltu against a constant bound
            b = get(regs, rt, first)
            put(regs, rd, self._less(get(regs, rs, first), b[1]) if b[0] == "c" else None, site)
        elif op == 0 and funct == 0:                             # sll
            put(regs, rd, _shl(get(regs, rt, first), sa), site)
        elif op in (35, 39):                                     # lw / lwu
            put(regs, rt, self._load(get(regs, rs, first), simm, slots), site)
        elif op == 43:                                           # sw
            self._store(get(regs, rs, first), simm, get(regs, rt, first), slots, site)
        elif op in (40, 41, 42, 46, 44, 45, 57, 61, 63):         # partial or non-word stores
            self._store(get(regs, rs, first), simm, None, slots, site)
        else:
            for reg in insn.defs:
                put(regs, reg, None, site)

    def _less(self, value: tuple, bound: int):
        if value[0] == "c":
            return _c(1 if value[1] < bound else 0)
        if value[0] == "lin" and value[2] == 1 and 0 < bound <= self.max_table:
            return ("ltu", value[1], 1, value[3], bound)
        return None

    def _load(self, base: tuple, simm: int, slots: dict):
        kind = base[0]
        if kind == "c":
            address = (base[1] + simm) & MASK
            return slots.get(("abs", address)) or ("mem", address)
        if kind == "lin" and base[2] == 1:
            return slots.get((base[1], (base[3] + simm) & MASK))
        if kind == "lin":                                        # table + scale * unbounded index
            first = (base[3] + simm) & MASK
            return ("scan", first, base[2]) if self.read_word(first) is not None else None
        if kind == "set":
            first, step, count = (base[1] + simm) & MASK, base[2], base[3]
            if count > self.max_table:
                return None
            words = tuple(self.read_word((first + k * step) & MASK) for k in range(count))
            return ("tbl", first, step, count, words) if None not in words else None
        return None

    def _store(self, base: tuple, simm: int, value, slots: dict, site: int) -> None:
        key = None
        if base[0] == "c":
            address = (base[1] + simm) & MASK
            key = ("abs", address)
            self.stores.append((address, value[1] if value is not None and value[0] == "c" else None, site))
            for other in [k for k in slots if k[0] not in ("abs", ("sp",))]:
                del slots[other]                                 # a pointer may alias the global
        elif base[0] == "lin" and base[2] == 1:
            key = (base[1], (base[3] + simm) & MASK)
            if base[1] != ("sp",):
                _forget(slots)
        else:
            _forget(slots)
        if value is not None and value[0] == "c" and (key is None or key[0] != ("sp",)):
            self.escaped.add(value[1])
        if key is not None:
            if value is not None:
                slots[key] = value
            else:
                slots.pop(key, None)

    # one block -----------------------------------------------------------------------------
    def block(self, block, state) -> dict:
        regs, slots = dict(state[0]), dict(state[1])
        insns, first = self.fl.insns, block.first
        term = cond = None
        taken = fall = None
        index = block.first
        while index <= block.last:
            insn = insns[index]
            if not insn.control:
                self.step(insn, regs, slots, first)
                index += 1
                continue
            slot = insns[index + 1] if index + 1 <= block.last else None
            source = (insn.word >> 21) & 31
            if insn.kind == "call":
                if insn.target is None:                         # jalr: the register is read before the slot
                    self.sites[insn.vram] = (self.get(regs, source, first), True, block.number)
                if slot is not None:
                    self.step(slot, regs, slots, first)
                for reg in CALL_CLOBBERED:
                    regs[reg] = ("lin", ("call", insn.vram, reg), 1, 0)
                _forget(slots)
                index += 2
                continue
            term = insn
            if insn.kind == "jr":
                self.sites[insn.vram] = (self.get(regs, source, first), False, block.number)
            elif insn.kind == "branch":
                cond = self._condition(insn, regs)
            if insn.likely and slot is not None:                # the slot runs on the taken edge only
                fall = (dict(regs), dict(slots))
                self.step(slot, regs, slots, first)
                taken = (regs, slots)
            else:
                if slot is not None:
                    self.step(slot, regs, slots, first)
                taken = fall = (regs, slots)
            index += 2
        if term is None:
            taken = fall = (regs, slots)
        return self._edges(block, term, cond, taken, fall)

    @staticmethod
    def _condition(insn, regs: dict):
        """(symbol, add, bound, edge on which the comparison held) for ``beq/bne(l) R, $zero`` on an
        ``sltiu``/``sltu`` result."""
        w = insn.word
        op, rs, rt = w >> 26, (w >> 21) & 31, (w >> 16) & 31
        if op not in (4, 5, 20, 21):
            return None
        if rs and not rt:
            reg = rs
        elif rt and not rs:
            reg = rt
        else:
            return None
        value = regs.get(reg)
        if value is None or value[0] != "ltu":
            return None
        return value[1], value[3], value[4], "fall" if op in (4, 20) else "taken"

    def _edges(self, block, term, cond, taken, fall) -> dict:
        out = {}
        if term is None or term.kind != "branch":
            for succ in block.succs:
                out[succ] = taken
            return out
        if cond is not None:
            sid, add, bound, edge = cond
            if edge == "fall":
                fall = _refine(fall, sid, add, bound)
            else:
                taken = _refine(taken, sid, add, bound)
        taken_block = self.block_at.get(self.index_of(term.target))
        fall_block = self.block_at.get(block.last + 1)
        for succ in block.succs:
            if succ == taken_block and succ != fall_block:
                out[succ] = taken
            elif succ == fall_block and succ != taken_block:
                out[succ] = fall
            else:
                out[succ] = _merge([taken, fall])
        return out

    # the function ----------------------------------------------------------------------------
    def run(self) -> None:
        blocks = self.fl.blocks
        if not blocks or blocks[0].last < blocks[0].first:
            return
        order = _reverse_postorder(blocks)
        seen = set(order)
        order += [b.number for b in blocks if b.number not in seen]
        edge_out: dict[tuple[int, int], tuple] = {}
        for _pass in range(_PASSES):
            self.sites.clear()
            self.stores.clear()
            self.escaped.clear()
            changed = False
            for number in order:
                block = blocks[number]
                incoming = [edge_out.get((pred, number)) for pred in block.preds]
                if number == 0:
                    state = _merge([({SP: _SP}, {})] + incoming)
                else:
                    state = _merge(incoming) or ({}, {})
                for succ, out in self.block(block, state).items():
                    if edge_out.get((number, succ)) != out:
                        edge_out[(number, succ)] = out
                        changed = True
            if not changed:
                break


class _Resolver:
    def __init__(self, read_word, starts, is_code, max_table: int):
        self.read_word, self.starts, self.max_table = read_word, starts, max_table
        self.is_code = is_code or (lambda t: read_word(t) is not None)

    def why_not(self, target: int, lo: int, hi: int):
        """None when ``target`` can be an instruction address, else the reason it cannot."""
        if target & 3:
            return "misaligned"
        if lo <= target < hi or target in self.starts or self.is_code(target):
            return None
        return "not code"

    def scan(self, base: int, stride: int, lo: int, hi: int, inside: bool) -> list[int]:
        """Entries from ``base`` while each is a target: inside [lo, hi) for a switch, else a
        catalogued start or code outside the function."""
        out = []
        for k in range(self.max_table):
            word = self.read_word((base + k * stride) & MASK)
            if word is None or not word or word & 3:
                break
            if inside:
                if not lo <= word < hi:
                    break
            elif lo < word < hi or self.why_not(word, lo, hi) is not None:
                break
            out.append(word)
        return out

    def resolve(self, site: int, value: tuple, call: bool, lo: int, hi: int, reached: bool) -> Site:
        prov = [] if reached else ["in a block not reachable from the function's entry"]
        kind = value[0]
        if kind == "c":
            target = value[1]
            local = not call and lo <= target < hi
            why = self.why_not(target, lo, hi)
            if why:
                return Site(site, lo, call, "constant", IMPOSSIBLE, (), (target,),
                            provenance=tuple(prov + [f"constant {hx(target)}: {why}"]))
            note = "constant address (lui/addiu, or stored and reloaded in this function)"
            if not local and target not in self.starts:
                note += "; not a catalogued function start"
            return Site(site, lo, call, "local-jump" if local else "constant", DEFINITE, (target,),
                        provenance=tuple(prov + [note]))
        if kind == "tbl":
            _k, base, step, count, words = value
            bounded = f"{count} entries at {hx(base)} stride {step}, index bounded by sltiu/sltu/andi"
            if not call and step == 4 and all(lo <= t < hi and not t & 3 for t in words):
                return Site(site, lo, call, "switch", DEFINITE, tuple(sorted(set(words))), (), base, count, 4,
                            tuple(prov + [bounded, "every entry lands inside the function"]))
            good = [t for t in words if t and self.why_not(t, lo, hi) is None]
            bad = [t for t in words if t not in good]
            return Site(site, lo, call, "pointer-table", FINITE if good else IMPOSSIBLE, tuple(sorted(set(good))),
                        tuple(sorted(set(bad))), base, count, step,
                        tuple(prov + [bounded, "loaded from data: a candidate set, not proof of completeness"]))
        if kind == "scan":
            _k, base, stride = value
            if not call and stride == 4:
                inside = self.scan(base, stride, lo, hi, True)
                if inside:
                    return Site(site, lo, call, "switch-unbounded", FINITE, tuple(sorted(set(inside))), (), base,
                                len(inside), 4, tuple(prov + [
                                    "no bound check found: entries read while they land inside the function"]))
            entries = self.scan(base, stride, lo, hi, False)
            if entries:
                return Site(site, lo, call, "pointer-table", FINITE, tuple(sorted(set(entries))), (), base,
                            len(entries), stride, tuple(prov + [
                                f"unbounded index, stride {stride}: entries read while they are code addresses"]))
            return Site(site, lo, call, "pointer-table", UNRESOLVED, (), (), base, 0, stride,
                        tuple(prov + [f"table at {hx(base)} stride {stride} holds no code address"]))
        if kind == "mem":
            return Site(site, lo, call, "stored-pointer", UNRESOLVED, (), (), value[1], 0, 0,
                        tuple(prov + [f"loaded from the global {hx(value[1])}"]))
        if kind == "lin":
            sid = value[1]
            if sid[0] == "in" and 4 <= sid[2] <= 7 and sid[1] == 0:
                what = f"argument register ${flowmod.NAMES[sid[2]]}"
            elif sid[0] == "call":
                what = f"a value returned by the call at {hx(sid[1])}"
            elif sid[0] == "def":
                what = f"a value computed at {hx(sid[1])} (load through a pointer, or arithmetic)"
            else:
                what = "a value the analysis does not track"
            return Site(site, lo, call, "unknown", UNRESOLVED, provenance=tuple(prov + [what]))
        return Site(site, lo, call, "unknown", UNRESOLVED, provenance=tuple(prov + [f"value kind {kind}"]))


def _shape(fl: flowmod.Flow, tables: dict) -> Shape:
    blocks = []
    for block in fl.blocks:
        calls, term = [], None
        for index in range(block.first, block.last + 1):
            insn = fl.insns[index]
            if insn.kind == "call":
                calls.append((insn.vram, insn.target))
            elif insn.control:
                term = insn
        after = fl.vram + 4 * (block.last + 1)
        leaves = None
        if term is None:
            if after >= fl.end:
                leaves = ("open", None)                         # runs off the end of the range
        elif term.kind == "return":
            leaves = ("return", None)
        elif term.kind == "jr":
            leaves = None if term.vram in tables else ("open", None)
        elif term.kind == "jump":
            if not fl.vram <= term.target < fl.end:
                leaves = ("tail", term.target)
        elif term.kind == "branch":
            if not fl.vram <= term.target < fl.end or (not flowmod._unconditional(term) and after >= fl.end):
                leaves = ("open", None)
        blocks.append((tuple(calls), leaves, tuple(block.succs)))
    return Shape(fl.vram, fl.end, blocks)


def analyse_function(code: bytes, vram: int, read_word, *, starts=frozenset(), is_code=None,
                     max_table: int = MAX_TABLE) -> FunctionResult:
    """Indirect sites of one function.  ``read_word(address)`` returns the 32-bit word at an
    address or None; ``starts`` are catalogued function starts; ``is_code(address)`` says whether
    an address outside the function may hold code (default: it is readable)."""
    code = code[:len(code) & ~3]
    result = FunctionResult(vram, vram + len(code))
    if len(code) < 4:
        return result
    resolver = _Resolver(read_word, starts, is_code, max_table)
    tables: dict[int, list[int]] = {}
    interp = None
    fl = None
    for _round in range(_ROUNDS):
        fl = flowmod.analyse(code, vram, tables)
        interp = _Interp(fl, read_word, max_table)
        interp.run()
        found = {}
        for site, (value, call, number) in interp.sites.items():
            if value[0] in ("tbl", "scan") and not call:
                resolved = resolver.resolve(site, value, call, vram, result.end, number in fl.reachable)
                if resolved.kind in ("switch", "switch-unbounded"):
                    found[site] = [t for t in _entries(read_word, resolved) if vram <= t < result.end]
        if found == tables or _round == _ROUNDS - 1:
            break
        tables = found
    for site, (value, call, number) in sorted(interp.sites.items()):
        resolved = resolver.resolve(site, value, call, vram, result.end, number in fl.reachable)
        result.sites.append(resolved)
        if resolved.kind == "switch" and resolved.status == DEFINITE:
            result.owned.append((resolved.table, 4 * resolved.count))
    result.tables = tables
    result.stores = list(interp.stores)
    result.escaped = set(interp.escaped)
    result.shape = _shape(fl, tables)
    return result


def _entries(read_word, site: Site) -> list[int]:
    """A table's entries in index order."""
    return [read_word((site.table + k * site.stride) & MASK) or 0 for k in range(site.count)]


# --------------------------------------------------------------------------- non-returning functions

def _returns(shape: Shape, no_return) -> bool:
    if not shape.blocks:
        return True
    seen, stack = {0}, [0]
    while stack:
        calls, leaves, succs = shape.blocks[stack.pop()]
        if any(target is not None and target in no_return for _site, target in calls):
            continue                                            # control never comes back from the call
        if leaves is not None:
            kind, target = leaves
            if kind in ("return", "open") or (kind == "tail" and target not in no_return):
                return True
        for succ in succs:
            if succ not in seen:
                seen.add(succ)
                stack.append(succ)
    return False


def no_return_functions(shapes: dict[int, Shape], seeds=()) -> set[int]:
    """Least fixpoint: a function returns when some path from its entry reaches ``jr $ra``, a tail
    jump to a returning (or unknown) function, or an exit the analysis cannot follow, without
    passing through a call to a function not (yet) known to return.  Calls to addresses outside
    ``shapes`` are assumed to return.  ``seeds`` are known non-returning functions."""
    seeds = set(seeds)
    returning: set[int] = set()
    pending = sorted(set(shapes) - seeds)
    changed = True
    while changed:
        changed = False
        not_yet = (set(shapes) - returning) | seeds
        left = []
        for start in pending:
            if _returns(shapes[start], not_yet):
                returning.add(start)
                changed = True
            else:
                left.append(start)
        pending = left
    return (set(shapes) - returning) | (seeds & set(shapes))


def fake_fallthroughs(shapes: dict[int, Shape], no_return: set[int], starts=frozenset()) -> list[dict]:
    """Calls to non-returning functions: the words after the call's delay slot are not reached
    through it (a walker that assumes calls return would run into them)."""
    out = []
    for start, shape in sorted(shapes.items()):
        for calls, _leaves, _succs in shape.blocks:
            for site, target in calls:
                if target is not None and target in no_return:
                    after = site + 8
                    out.append({"function": hx(start), "site": hx(site), "callee": hx(target), "next": hx(after),
                                "inside": after < shape.end, "next_is_start": after in starts})
    return out


# --------------------------------------------------------------------------- an image

@dataclass
class ImageResult:
    functions: dict[int, FunctionResult]
    sites: list[Site]
    no_return: set[int]
    fake: list[dict]
    starts: frozenset

    def owned(self) -> list[tuple[int, int, int]]:
        """(table address, bytes, owning function) of every definite switch table."""
        return sorted((a, n, f) for f, r in self.functions.items() for a, n in r.owned)

    def summary(self, examples: int = 8) -> dict:
        by_status = Counter(s.status for s in self.sites)
        by_kind = Counter(f"{s.kind}:{s.status}" for s in self.sites)
        ranges = sorted((f, r.end) for f, r in self.functions.items())
        firsts = [lo for lo, _hi in ranges]

        def inside_function(address: int):
            i = bisect.bisect_right(firsts, address) - 1
            return ranges[i][0] if i >= 0 and address < ranges[i][1] else None

        overlaps = []
        for address, size, owner in self.owned():
            hit = inside_function(address) or inside_function(address + size - 1)
            if hit is not None:
                overlaps.append({"table": hx(address), "bytes": size, "owner": hx(owner), "inside": hx(hit)})
        uncatalogued = sorted({t for s in self.sites if s.status in (DEFINITE, FINITE)
                               and (s.call or s.kind in ("pointer-table", "stored-pointer", "constant"))
                               for t in s.targets if t not in self.starts})
        out = {"functions": len(self.functions), "indirect_sites": len(self.sites),
               "jr_sites": sum(1 for s in self.sites if not s.call),
               "jalr_sites": sum(1 for s in self.sites if s.call),
               "by_status": {k: by_status.get(k, 0) for k in STATUSES}, "by_kind": dict(sorted(by_kind.items())),
               "switch_tables": sum(1 for s in self.sites if s.kind == "switch" and s.status == DEFINITE),
               "switch_table_bytes": sum(n for _a, n, _f in self.owned()),
               "tables_inside_catalogued_functions": len(overlaps),
               "tables_inside_examples": overlaps[:examples],
               "uncatalogued_targets": len(uncatalogued),
               "uncatalogued_target_examples": [hx(t) for t in uncatalogued[:examples]],
               "no_return": len(self.no_return), "no_return_examples": [hx(t) for t in sorted(self.no_return)[:examples]],
               "fake_fallthroughs": len(self.fake),
               "fake_fallthroughs_into_code": sum(1 for f in self.fake if f["inside"] and not f["next_is_start"]),
               "fake_fallthrough_examples": self.fake[:examples]}
        out["examples"] = {status: [s.as_dict() for s in self.sites if s.status == status][:examples]
                           for status in STATUSES}
        return out


def image_reader(data: bytes, delta: int):
    """``read_word`` for a linear image (file offset = address - delta)."""
    size = len(data)

    def read_word(address: int):
        offset = address - delta
        return struct.unpack_from(">I", data, offset)[0] if 0 <= offset <= size - 4 else None
    return read_word


def analyse_image(data: bytes, delta: int, rows, *, seeds=(), max_table: int = MAX_TABLE) -> ImageResult:
    """Every catalogued function (``rows`` = (name, vram, size)) of a linear image."""
    read_word = image_reader(data, delta)
    rows = sorted((v, s) for _n, v, s in rows if s > 0 and 0 <= v - delta and v - delta + s <= len(data))
    starts = frozenset(v for v, _s in rows)
    firsts = [v for v, _s in rows]

    def is_code(address: int) -> bool:
        i = bisect.bisect_right(firsts, address) - 1
        if i >= 0 and address < rows[i][0] + rows[i][1]:
            return True
        word = read_word(address)
        if not word:
            return False
        ins = mips.decode(word, address)
        return bool(ins.isValid() and ins.isImplemented())

    functions: dict[int, FunctionResult] = {}
    for vram, size in rows:
        if vram in functions:
            continue
        code = data[vram - delta:vram - delta + size]
        functions[vram] = analyse_function(code, vram, read_word, starts=starts, is_code=is_code,
                                           max_table=max_table)
    # pointers stored to globals by any function, then loaded and called elsewhere
    constant: dict[int, set[int]] = {}
    other: Counter = Counter()
    for result in functions.values():
        for address, value, _site in result.stores:
            if value is None:
                other[address] += 1
            elif value:
                constant.setdefault(address, set()).add(value)
    sites = []
    for result in functions.values():
        for index, site in enumerate(result.sites):
            if site.kind == "stored-pointer" and site.table is not None:
                site = _stored(site, constant.get(site.table, set()), other[site.table], read_word, starts, is_code)
                result.sites[index] = site
            sites.append(site)
    shapes = {start: r.shape for start, r in functions.items() if r.shape is not None}
    no_return = no_return_functions(shapes, seeds)
    return ImageResult(functions, sites, no_return, fake_fallthroughs(shapes, no_return, starts), starts)


def _stored(site: Site, values: set[int], others: int, read_word, starts, is_code) -> Site:
    def code(t):
        return bool(t) and not t & 3 and (t in starts or is_code(t))
    candidates = sorted(t for t in values if code(t))
    prov = list(site.provenance)
    if candidates:
        prov.append(f"{len(candidates)} code address(es) stored to it by catalogued code")
    initial = read_word(site.table)
    if initial and code(initial) and initial not in candidates:
        candidates = sorted(set(candidates) | {initial})
        prov.append("its initial value in the image is a code address")
    if not candidates:
        prov.append("no code address is stored to it by catalogued code")
        return replace(site, provenance=tuple(prov))
    if others:
        prov.append(f"open: {others} store(s) of values the analysis cannot follow")
        return replace(site, targets=tuple(candidates), provenance=tuple(prov))
    return replace(site, status=FINITE, targets=tuple(candidates), provenance=tuple(prov))
