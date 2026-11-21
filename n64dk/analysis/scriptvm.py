"""Script-VM detection: interpreter dispatch loops, their handlers, and the bytecode they run.

A *dispatch* is an indirect jump (``jr``) or call (``jalr``) through a word loaded from a table
indexed by a value that was loaded, a byte or a halfword, through a pointer register (the
instruction pointer, IP)::

    lbu   t0, 0(s0)          # opcode (lbu/lb/lhu/lh), maybe masked (andi) or biased (addiu)
    addiu s0, s0, 1          # IP advance (before or after the fetch)
    sll   t0, t0, 2
    lui   at, %hi(table)
    addu  at, at, t0
    lw    t1, %lo(table)(at)
    jr    t1                 # jump-table flavour: handlers inside the function jump back
                             # call-table flavour: jalr; handlers are functions, the loop follows

``find_interpreters`` recognises the pattern by walking definitions backwards from each ``jr``/
``jalr`` (not by matching fixed sequences), reads the table (bounded by an ``sltiu`` check, an
``andi`` mask or the first entry that is not code), and analyses every handler by bounded path
exploration over an abstract IP domain (IP + k, operand bytes read through IP, IP + operand,
constant, loaded from elsewhere):

* ``plain``: every path returns to the loop with IP advanced by one constant (operand consumption);
* ``jump``: every path assigns IP from an operand (relative, absolute or base + operand);
* ``branch``: some paths assign from an operand, others advance (conditional jump);
* ``call``: a path stores an IP-derived return point through another pointer, then assigns IP from
  an operand; ``return``: IP loaded through a pointer that is not IP;
* ``halt``: the interpreter returns (or a call-table handler returns a null IP);
* ``unknown``: anything else (a path the domain cannot follow, a step or path budget exceeded).

For the call-table flavour the IP reaches the handler in an argument register (``move aN, IP``
in the dispatch) and comes back in ``$v0`` (``move IP, v0`` or ``addu IP, IP, v0`` after the
call); other conventions give ``unknown`` handlers.

``disassemble`` walks a script from a given start with those summaries and returns a listing, a
CFG (basic blocks, call edges) and byte ownership (which bytes each decoded instruction covers,
overlaps reported as conflicts).  Everything is evidence about the program's data, never about
code bytes, and nothing here changes a catalogue.  A compiled ``switch`` on bytes read from a
stream inside a loop (a format-string parser, a text renderer's control codes) has the same
shape and is reported the same way: it *is* a byte-stream interpreter.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

from .loaders_generic import M32, V0, Ins, call_constants, decode, decode_range, hx, is_move

MAX_STEPS = 160          # instructions per handler path
MAX_PATHS = 32           # paths per handler
MAX_ENTRIES = {1: 256, 2: 1024}
BACK = 32                # instructions looked at before a dispatch
CALLER_SAVED = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 31)


@dataclass
class Handler:
    address: int
    kind: str = "unknown"
    advance: int | None = None             # IP advance past the dispatch (fall-through / return point)
    target: dict | None = None             # how a new IP is formed from an operand
    operands: list = field(default_factory=list)   # [start, length, signed], relative to IP after dispatch
    halts: bool = False
    paths: int = 0
    why: str = ""

    def as_dict(self) -> dict:
        return {"address": hx(self.address), "kind": self.kind, "advance": self.advance, "target": self.target,
                "operands": self.operands, "halts": self.halts, "paths": self.paths, "why": self.why}


@dataclass
class Interpreter:
    function: int
    dispatch: int
    fetch: int
    flavour: str                           # jump-table | call-table
    ip_reg: int
    width: int                             # opcode bytes
    signed: bool
    transforms: list                       # ("bias", k) / ("mask", m) in the order they apply
    opcode_offset: int                     # opcode address - IP at the loop head
    dispatch_advance: int                  # IP advance in the dispatch block
    table: int
    entries: list
    handlers: dict
    bound: int | None = None
    ip_arg: int | None = None              # IP initialised from $aN
    ip_update: str | None = None           # call-table: "v0" | "add-v0" | None
    post_advance: int = 0

    @property
    def delta(self) -> int:
        """IP after the dispatch, minus the opcode's address."""
        return self.dispatch_advance - self.opcode_offset

    def index(self, opcode: int) -> int:
        for op, k in self.transforms:
            opcode = opcode + k if op == "bias" else opcode & k
        return opcode

    def handler_for(self, opcode: int) -> Handler | None:
        i = self.index(opcode)
        return self.handlers.get(self.entries[i]) if 0 <= i < len(self.entries) else None

    def as_dict(self) -> dict:
        kinds: dict[str, int] = {}
        for h in self.handlers.values():
            kinds[h.kind] = kinds.get(h.kind, 0) + 1
        return {"function": hx(self.function), "dispatch": hx(self.dispatch), "fetch": hx(self.fetch),
                "flavour": self.flavour, "ip_reg": self.ip_reg, "opcode_width": self.width, "signed": self.signed,
                "transforms": [list(t) for t in self.transforms], "opcode_offset": self.opcode_offset,
                "dispatch_advance": self.dispatch_advance, "table": hx(self.table), "entries": len(self.entries),
                "bound": self.bound, "ip_arg": self.ip_arg, "ip_update": self.ip_update,
                "handler_kinds": kinds, "handlers": [self.handlers[a].as_dict() for a in sorted(self.handlers)]}


# ---------------------------------------------------------------------------------------------
# Recognising a dispatch

def _def_before(insns: list[Ins], idx: int, reg: int, floor: int) -> int | None:
    for k in range(idx - 1, floor - 1, -1):
        if insns[k].dst == reg:
            return k
    return None


def _const_before(insns: list[Ins], idx: int, reg: int, floor: int, depth: int = 4) -> int | None:
    if reg == 0:
        return 0
    k = _def_before(insns, idx, reg, floor)
    if k is None or depth == 0:
        return None
    d = insns[k]
    if d.name == "lui":
        return (d.imm << 16) & M32
    if d.name in ("addiu", "addi", "ori"):
        b = _const_before(insns, k, d.rs, floor, depth - 1)
        if b is None:
            return None
        return (b + d.simm) & M32 if d.name != "ori" else b | d.imm
    if is_move(d):
        return _const_before(insns, k, d.srcs[0], floor, depth - 1) if d.srcs else 0
    return None


def _increments(insns: list[Ins], lo: int, hi: int, reg: int) -> int:
    return sum(x.simm for x in insns[lo:hi] if x.dst == reg and x.rs == reg and x.name in ("addiu", "daddiu"))


def find_interpreters(code: bytes, vram: int, read_word=None, text: tuple[int, int] | None = None,
                      insns: list[Ins] | None = None) -> list[Interpreter]:
    """Dispatch loops in one function.  ``read_word(vram) -> int | None`` reads the table and, for
    call-table handlers, their code (default: this function's bytes); ``text`` bounds where handler
    functions may be (default: the function)."""
    insns = insns if insns is not None else decode_range(code, vram)
    end = vram + 4 * len(insns)
    if read_word is None:
        def read_word(a: int) -> int | None:
            o = a - vram
            return struct.unpack_from(">I", code, o)[0] if 0 <= o <= len(code) - 4 and not o & 3 else None
    text = text or (vram, end)
    out = []
    for i, ins in enumerate(insns):
        if ins.kind in ("jr", "jalr"):
            vm = _match(insns, i, vram, end, read_word, text)
            if vm is not None:
                out.append(vm)
    return out


def _match(insns: list[Ins], i: int, vram: int, end: int, read_word, text) -> Interpreter | None:
    floor = max(0, i - BACK)
    disp = insns[i]
    k_lw = _def_before(insns, i, disp.rs, floor)
    if k_lw is None or insns[k_lw].name != "lw":
        return None
    lw = insns[k_lw]
    k_add = _def_before(insns, k_lw, lw.rs, floor)
    if k_add is None or insns[k_add].name not in ("addu", "daddu", "add") or len(insns[k_add].srcs) != 2:
        return None
    add = insns[k_add]
    scaled = None
    for idx_reg, base_reg in ((add.rs, add.rt), (add.rt, add.rs)):
        k = _def_before(insns, k_add, idx_reg, floor)
        if k is not None and insns[k].name == "sll" and insns[k].sa == 2:
            scaled = (k, base_reg)
            break
    if scaled is None:
        return None
    k_sll, base_reg = scaled
    base = _const_before(insns, k_add, base_reg, floor)
    if base is None:
        return None
    table = (base + lw.simm) & M32
    # the opcode: sll <- [andi | addiu | move]* <- lbu/lb/lhu/lh through the IP
    reg, at, found, chain, backwards = insns[k_sll].rt, k_sll, None, set(), []
    for _ in range(5):
        chain.add(reg)
        k = _def_before(insns, at, reg, floor)
        if k is None:
            return None
        d = insns[k]
        if d.name in ("lbu", "lb", "lhu", "lh"):
            found = k
            break
        if d.name == "andi":
            backwards.append(("mask", d.imm))
        elif d.name in ("addiu", "addi"):
            backwards.append(("bias", d.simm))
        elif not is_move(d):
            return None
        if not d.srcs:
            return None
        reg, at = d.srcs[0], k
    if found is None:
        return None
    fetch = insns[found]
    ip = fetch.rs
    if ip in (0, 29) or not any(x.dst == ip and x.rs == ip and x.name in ("addiu", "daddiu") and x.simm > 0
                                for x in insns):
        return None
    bound = None
    for x in insns[floor:i]:
        if x.name == "sltiu" and x.rs in chain and x.imm:
            bound = x.imm
    # the dispatch block: from the straight-line leader before the fetch to the delay slot
    targets = {x.target for x in insns if x.target is not None}
    lo = found
    while lo > floor and insns[lo].at not in targets and not insns[lo - 1].kind \
            and not (lo >= 2 and insns[lo - 2].kind):
        lo -= 1
    opcode_offset = _increments(insns, lo, found, ip) + fetch.simm
    dispatch_advance = _increments(insns, lo, i + 2, ip)
    k = found
    while k > 0 and not insns[k - 1].unconditional and not (k >= 2 and insns[k - 2].unconditional):
        k -= 1
    cont_lo = insns[k].at
    ip_arg = None
    for x in insns[:lo]:
        if x.dst == ip:
            ip_arg = x.srcs[0] - 4 if is_move(x) and x.srcs and 4 <= x.srcs[0] <= 7 else None
    flavour = "jump-table" if disp.kind == "jr" else "call-table"
    width = fetch.width
    limit = MAX_ENTRIES[width]
    masks = [m for op, m in backwards if op == "mask"]
    if masks:
        limit = min(limit, masks[0] + 1)
    if bound:
        limit = min(limit, bound)
    entries = []
    for n in range(limit):
        w = read_word((table + 4 * n) & M32)
        if w is None or w & 3:
            break
        if not (vram <= w < end if flavour == "jump-table" else text[0] <= w < text[1]):
            break
        entries.append(w)
    if not entries:
        return None
    vm = Interpreter(vram, disp.at, fetch.at, flavour, ip, width, fetch.signed, list(reversed(backwards)),
                     opcode_offset, dispatch_advance, table, entries, {}, bound, ip_arg)
    decoded: dict[int, Ins | None] = {}

    def fetch_ins(a: int) -> Ins | None:
        if a not in decoded:
            w = read_word(a)
            decoded[a] = None if w is None else decode(w, a)
        return decoded[a]

    if flavour == "jump-table":
        init = {ip: ("ip", 0)}
        inside = lambda a: vram <= a < end                                  # noqa: E731
        is_continue = lambda a: cont_lo <= a <= fetch.at                    # noqa: E731

        def final(st):
            return _val(st, ip)
    else:
        init = {}
        for k in range(lo, min(i + 2, len(insns))):
            x = insns[k]
            if 4 <= x.dst <= 7 and x.srcs == (ip,) and (is_move(x) or x.name in ("addiu", "daddiu")):
                off = _increments(insns, lo, k, ip) + (x.simm if x.name in ("addiu", "daddiu") else 0)
                init = {x.dst: ("ip", off - dispatch_advance)}
        # after the call: how the IP is updated, up to the branch back to the loop (conditional
        # branches elsewhere, such as an exit test, are passed on their fall-through side)
        looped, k = False, i + 2
        while k < min(len(insns), i + 18):
            x = insns[k]
            for y in (x, insns[k + 1] if x.kind and k + 1 < len(insns) else None):
                if y is None or y.kind or y.dst != ip:
                    continue
                if is_move(y) and y.srcs == (V0,):
                    vm.ip_update = "v0"
                elif y.name in ("addu", "daddu") and set(y.srcs) == {ip, V0}:
                    vm.ip_update = "add-v0"
                elif y.name in ("addiu", "daddiu") and y.rs == ip:
                    vm.post_advance += y.simm
                else:
                    vm.ip_update = "other"
            if x.kind in ("branch", "jump") and x.target is not None and cont_lo <= x.target <= fetch.at:
                looped = True
                break
            if x.kind and (x.unconditional or x.kind not in ("branch",)):
                break
            k += 2 if x.kind else 1
        if not looped or vm.ip_update == "other":
            return None
        inside = lambda a: text[0] <= a < text[1]                           # noqa: E731
        is_continue = lambda a: False                                       # noqa: E731
        post = vm.post_advance

        def final(st):
            v = _val(st, V0)
            if vm.ip_update == "v0":
                new = v
            elif vm.ip_update == "add-v0":
                new = ("ip", v[1]) if v and v[0] == "const" else ("ipop", 0, v) if v and v[0] == "op" else None
            else:
                new = ("ip", 0)
            if new and new[0] == "ip":
                return ("ip", new[1] + post)
            if new and new[0] == "ipop":
                return ("ipop", new[1] + post, new[2])
            return new
    for address in dict.fromkeys(entries):
        results = _paths(fetch_ins, address, init, inside, is_continue)
        vm.handlers[address] = _summary(address, results, final, flavour)
    if not any(h.kind in ("plain", "jump", "branch", "call", "return") for h in vm.handlers.values()):
        return None                                                         # nothing loops back
    return vm


# ---------------------------------------------------------------------------------------------
# Handler paths over the abstract IP domain
#   ("ip", k)                     IP after dispatch + k
#   ("op", start, len, signed, shift)  operand bytes at IP + start (shifted left)
#   ("ipop", k, op)               IP + k + operand
#   ("based", base, op)           a constant or an entry register + operand
#   ("const", c)  ("entry", reg)  ("mem",)  None (unknown)

def _val(st: dict, r: int):
    if r == 0:
        return ("const", 0)
    return st.get(r, ("entry", r))


def _combine(a, b, name: str):
    for x, y in ((a, b), (b, a)):
        if x is None or y is None:
            continue
        if x[0] == "op" == y[0] and y[4] == 0 and x[4] == 8 * y[2] and y[1] == x[1] + x[2]:
            return ("op", x[1], x[2] + y[2], x[3], 0)
        if name == "or":
            continue
        if x[0] == "ip" and y[0] == "op" and y[4] == 0:
            return ("ipop", x[1], y)
        if x[0] == "ip" and y[0] == "const":
            c = y[1] - 0x100000000 if y[1] & 0x80000000 else y[1]
            return ("ip", x[1] + c)
        if x[0] in ("entry", "const") and y[0] == "op" and y[4] == 0:
            return ("based", x, y)
        if x[0] == "const" == y[0]:
            return ("const", (x[1] + y[1]) & M32)
    return None


def _apply(ins: Ins, st: dict, ev: list) -> None:
    if ins.store:
        if ins.width:
            v, b = _val(st, ins.rt), _val(st, ins.rs)
            if v is not None and v[0] in ("ip", "ipop") and not (b is not None and b[0] == "ip"):
                ev.append(("push", v))
        return
    if not ins.dst:
        return
    name = ins.name
    out = None
    if ins.width:                                                           # integer load
        b = _val(st, ins.rs)
        if b is not None and b[0] == "ip":
            start = b[1] + ins.simm
            ev.append(("read", start, ins.width, ins.signed))
            out = ("op", start, ins.width, ins.signed, 0)
        else:
            out = ("mem",)
    elif is_move(ins):
        out = _val(st, ins.srcs[0]) if ins.srcs else ("const", 0)
    elif name in ("addiu", "addi", "daddiu"):
        a = _val(st, ins.rs)
        if a is not None and a[0] == "ip":
            out = ("ip", a[1] + ins.simm)
        elif a is not None and a[0] == "const":
            out = ("const", (a[1] + ins.simm) & M32)
        elif a is not None and a[0] == "ipop":
            out = ("ipop", a[1] + ins.simm, a[2])
    elif name == "lui":
        out = ("const", (ins.imm << 16) & M32)
    elif name == "ori":
        a = _val(st, ins.rs)
        if a is not None and a[0] == "const":
            out = ("const", a[1] | ins.imm)
    elif name in ("addu", "daddu", "add", "or"):
        out = _combine(_val(st, ins.rs), _val(st, ins.rt), name)
    elif name == "sll":
        x = _val(st, ins.rt)
        if x is not None and x[0] == "op":
            out = ("op", x[1], x[2], x[3], x[4] + ins.sa)
        elif x is not None and x[0] == "const":
            out = ("const", (x[1] << ins.sa) & M32)
    st[ins.dst] = out


def _paths(fetch_ins, start: int, init: dict, inside, is_continue) -> list[tuple[str, dict, list]]:
    """Bounded depth-first exploration of a handler: ``(end, state, events)`` per path, ``end`` one
    of continue (back to the loop), return (``jr $ra``), indirect, unknown."""
    results: list[tuple[str, dict, list]] = []
    work = [(start, dict(init), [], 0)]
    while work and len(results) < MAX_PATHS:
        pc, st, ev, n = work.pop()
        while True:
            ins = fetch_ins(pc) if inside(pc) and n <= MAX_STEPS else None
            if ins is None:
                results.append(("unknown", st, ev))
                break
            if not ins.kind:
                _apply(ins, st, ev)
                pc, n = (pc + 4) & M32, n + 1
                continue
            slot = fetch_ins(pc + 4)
            if slot is None or slot.kind:
                results.append(("unknown", st, ev))
                break
            if ins.kind in ("call", "jalr"):
                _apply(slot, st, ev)
                for r in CALLER_SAVED:
                    st[r] = None
                pc, n = (pc + 8) & M32, n + 2
                continue
            if ins.kind in ("return", "jr"):
                _apply(slot, st, ev)
                results.append(("return" if ins.kind == "return" else "indirect", st, ev))
                break
            if ins.kind == "jump" or ins.unconditional:
                _apply(slot, st, ev)
                if is_continue(ins.target):
                    results.append(("continue", st, ev))
                    break
                pc, n = ins.target, n + 2
                continue
            taken, taken_ev = dict(st), list(ev)                           # conditional branch
            _apply(slot, taken, taken_ev)
            if is_continue(ins.target):
                results.append(("continue", taken, taken_ev))
            else:
                work.append((ins.target, taken, taken_ev, n + 2))
            if not ins.likely:
                _apply(slot, st, ev)
            pc, n = (pc + 8) & M32, n + 2
    if work:
        results.append(("unknown", {}, []))                                 # path budget exceeded
    return results


def _target(v) -> dict | None:
    if v[0] == "ipop" and v[2][4] == 0:
        return {"mode": "rel", "operand": list(v[2][1:4]), "bias": v[1]}
    if v[0] == "op" and v[4] == 0:
        return {"mode": "abs", "operand": list(v[1:4]), "bias": 0}
    if v[0] == "based" and v[2][4] == 0:
        base = hx(v[1][1]) if v[1][0] == "const" else f"entry:{v[1][1]}"
        return {"mode": "based", "base": base, "operand": list(v[2][1:4]), "bias": 0}
    return None


def _summary(address: int, results, final, flavour: str) -> Handler:
    h = Handler(address, paths=len(results))
    advances, targets, returns, unknown, halts = set(), [], 0, 0, 0
    reads = set()
    for end, st, ev in results:
        reads.update(e[1:] for e in ev if e[0] == "read")
        if flavour == "jump-table" and end == "return":
            halts += 1
            continue
        if end != ("continue" if flavour == "jump-table" else "return"):
            unknown += 1
            continue
        v = final(st)
        pushes = [e[1] for e in ev if e[0] == "push"]
        if v is None:
            unknown += 1
        elif v[0] == "ip":
            advances.add(v[1])
        elif v[0] == "const" and v[1] == 0:
            halts += 1
        elif v[0] == "const":
            targets.append(({"mode": "const", "value": hx(v[1])}, pushes))
        elif v[0] == "mem":
            returns += 1
        elif v[0] in ("ipop", "op", "based") and _target(v) is not None:
            targets.append((_target(v), pushes))
        else:
            unknown += 1
    h.operands = [list(r) for r in sorted(reads)]
    h.halts = halts > 0
    if unknown:
        h.why = f"{unknown} of {len(results)} paths not followed"
        return h
    if targets:
        if len({json.dumps(t, sort_keys=True) for t, _p in targets}) > 1 or returns:
            h.why = "paths form the new IP differently"
            return h
        h.target = targets[0][0]
        pushed = [p for _t, ps in targets for p in ps]
        if pushed:
            if pushed[0][0] != "ip" or advances:
                h.why = "call with an unrecognised return point"
                return h
            h.kind, h.advance = "call", pushed[0][1]
        elif advances:
            if len(advances) > 1:
                h.why = "fall-through advances differ by path"
                return h
            h.kind, h.advance = "branch", next(iter(advances))
        else:
            h.kind = "jump"
        return h
    if returns:
        h.kind = "return" if not advances else "unknown"
        h.why = "" if not advances else "returns on some paths only"
        return h
    if advances:
        if len(advances) > 1:
            h.why = "advances differ by path"
            return h
        h.kind, h.advance = "plain", next(iter(advances))
        return h
    if halts:
        h.kind = "halt"
    return h


# ---------------------------------------------------------------------------------------------
# Bytecode

def disassemble(vm: Interpreter, data: bytes, base: int, start: int, *, limit: int = 4096,
                based: int | None = None) -> dict:
    """Walk the script at ``start`` (``data`` holds the bytes from vram ``base``): listing, CFG and
    byte ownership.  ``based``: the base for ``base + operand`` targets formed from a register the
    handler did not set (default: the script start)."""
    end = base + len(data)
    based = start if based is None else based

    def read(addr: int, length: int, signed: bool) -> int | None:
        o = addr - base
        if o < 0 or o + length > len(data):
            return None
        v = int.from_bytes(data[o:o + length], "big")
        return v - (1 << (8 * length)) if signed and v >> (8 * length - 1) else v

    delta = vm.delta
    listing: dict[int, dict] = {}
    owner: dict[int, int] = {}
    conflicts, stopped, calls = [], [], []
    succ_of: dict[int, list[int]] = {}
    work = [start]
    while work and len(listing) < limit:
        pc = work.pop()
        if pc in listing:
            continue
        if not base <= pc < end:
            stopped.append({"at": hx(pc), "why": "outside the data"})
            continue
        op = read(pc, vm.width, vm.signed)
        h = vm.handler_for(op) if op is not None else None
        if h is None:
            stopped.append({"at": hx(pc), "why": "no handler for the opcode" if op is not None else "truncated"})
            continue
        ip_d = pc + delta
        operands = [{"at": s, "value": read(ip_d + s, n, sg)} for s, n, sg in h.operands]
        length = delta + h.advance if h.advance is not None and h.kind in ("plain", "branch", "call") else None
        target = None
        if h.target is not None:
            t = h.target
            if t["mode"] == "const":
                target = int(t["value"], 16)
            else:
                s, n, sg = t["operand"]
                v = read(ip_d + s, n, sg)
                if v is not None:
                    if t["mode"] == "rel":
                        target = ip_d + t["bias"] + v
                    elif t["mode"] == "abs":
                        target = v & M32
                    else:
                        b = int(t["base"], 16) if t["base"].startswith("0x") else based
                        target = (b + v + t["bias"]) & M32
        span = max([vm.width, delta, length or 0] + [delta + s + n for s, n, _sg in h.operands])
        succs: list[int] = []
        if h.kind == "plain":
            succs = [pc + length]
        elif h.kind == "branch":
            succs = [pc + length] + ([target] if target is not None else [])
        elif h.kind == "jump":
            succs = [target] if target is not None else []
        elif h.kind == "call":
            succs = ([target] if target is not None else []) + [pc + length]
            calls.append({"site": hx(pc), "target": None if target is None else hx(target),
                          "returns_to": hx(pc + length)})
        elif h.kind not in ("return", "halt"):
            stopped.append({"at": hx(pc), "why": f"handler {hx(h.address)} not understood: {h.why}"})
        if h.kind in ("jump", "branch", "call") and target is None:
            stopped.append({"at": hx(pc), "why": "target operand unreadable"})
        listing[pc] = {"at": hx(pc), "offset": pc - base, "opcode": op, "handler": hx(h.address), "kind": h.kind,
                       "length": span, "operands": operands, "succs": [hx(s) for s in succs],
                       "text": f"{h.kind} 0x{op & ((1 << (8 * vm.width)) - 1):0{2 * vm.width}X}" +
                               "".join(f" {o['value']}" for o in operands) +
                               (f" -> {hx(target)}" if target is not None else "")}
        succ_of[pc] = succs
        for b in range(pc, pc + span):
            if owner.setdefault(b, pc) != pc:
                conflicts.append({"byte": hx(b), "instructions": [hx(owner[b]), hx(pc)]})
        work.extend(reversed(succs))
    # basic blocks
    leaders = {start}
    for pc, succs in succ_of.items():
        if listing[pc]["kind"] != "plain":
            leaders.update(succs)
    ordered = sorted(listing)
    blocks, current = [], None
    for pc in ordered:
        fallen = current is not None and succ_of[current["last"]] == [pc] and \
            listing[current["last"]]["kind"] == "plain"
        if pc in leaders or not fallen:
            current = {"start": pc, "last": pc, "count": 1}
            blocks.append(current)
        else:
            current["last"], current["count"] = pc, current["count"] + 1
    out_blocks = [{"start": hx(b["start"]), "end": hx(b["last"] + listing[b["last"]]["length"]),
                   "instructions": b["count"], "succs": [hx(s) for s in succ_of[b["last"]]]} for b in blocks]
    owned: list[list[int]] = []
    for b in sorted(owner):
        if owned and owned[-1][1] == b:
            owned[-1][1] = b + 1
        else:
            owned.append([b, b + 1])
    return {"start": hx(start), "instructions": [listing[pc] for pc in ordered], "blocks": out_blocks,
            "calls": calls, "owned": [[hx(a), hx(b)] for a, b in owned], "owned_bytes": len(owner),
            "conflicts": conflicts[:32], "stopped": stopped[:32],
            "complete": not stopped and not work and len(listing) < limit}


def script_starts(vm: Interpreter, calls: list[tuple[int, int, tuple]]) -> list[int]:
    """Constant IP arguments passed to the interpreter function (``call_constants`` output)."""
    if vm.ip_arg is None:
        return []
    return sorted({args[vm.ip_arg] for _site, target, args in calls
                   if target == vm.function and args[vm.ip_arg] is not None})


__all__ = ["Handler", "Interpreter", "find_interpreters", "disassemble", "script_starts", "call_constants"]
