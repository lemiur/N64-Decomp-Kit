"""Cross-function type and structure constraints from machine code, with provenance (MIPS o32, BE).

Pure analysis of image bytes (never of drafted C, so m2c's guesses never become evidence):

1. **Walk** each function once in address order.  A register holds ``("hi", upper, indexed)``
   after ``lui`` or ``("v", cell, delta, indexed)``: the value of a *cell* plus a constant.  A cell
   is ``(root, path)`` with root ``("ga", address)`` (address of a global), ``("a", function, k)``
   (incoming ``$a<k>``) or ``("r", function)`` (a call's result); each path element is one
   dereference at that byte offset: ``(("ga", A), (0,))`` is the global at ``A``,
   ``(("a", F, 0), (0x10, 4))`` is ``arg0->f10->f4``.  At a branch target only values compiled
   code can rely on survive: never-written ``$a``, registers and stack slots written exactly once.
2. **Collect** constraints per cell, each with :class:`Evidence` (function, instruction address,
   opcode, operand): access width/signedness/float, pointer use as a base, ``andi``/``sll+sra``/
   ``mtc1`` narrowing of arguments and results, a global's address stored into a cell.  Value flows
   become *bindings* ``root = value(cell) + delta``: call arguments, ``jr $ra`` results, an
   argument or result stored into a cell.
3. **Propagate** to a fixed point: a constraint on a dereference of a bound root is restated on
   the bound cell (callee's ``*(arg0+4)`` becomes ``*(&gObj+0x24)`` when a caller passes
   ``&gObj+0x20``), the binding appended to ``Evidence.via``; pointer use of an argument flows back
   to the actual.  Flows go callee -> actual only, never between two callers of one callee.

Disagreements are kept as :class:`Conflict` with the evidence of each side, never voted away.
``declarations`` renders C only for globals whose whole type tree is conflict-free.
"""
from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass, field, replace

MASK = 0xFFFFFFFF
RAM = (0x80000000, 0x80800000)
MAX_PATH, MAX_VIA, MIN_OFFSET, MAX_OFFSET, MAX_CONSTRAINTS = 4, 6, -0x100, 0x8000, 250_000
STABLE = frozenset(range(1, 32)) - {2, 3, 4, 5, 6, 7, 29, 31}   # one explicit write is their only def
CLOBBERED = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 31)
LOADS = {32: ("lb", 1, True, False), 36: ("lbu", 1, False, False), 33: ("lh", 2, True, False),
         37: ("lhu", 2, False, False), 35: ("lw", 4, None, False), 39: ("lwu", 4, None, False),
         55: ("ld", 8, None, False), 49: ("lwc1", 4, None, True), 53: ("ldc1", 8, None, True)}
STORES = {40: ("sb", 1, False), 41: ("sh", 2, False), 43: ("sw", 4, False), 63: ("sd", 8, False),
          57: ("swc1", 4, True), 61: ("sdc1", 8, True)}


@dataclass(frozen=True)
class Evidence:
    function: str
    site: int
    op: str
    detail: str
    via: tuple[str, ...] = ()        # bindings the constraint travelled through, nearest first

    def __str__(self) -> str:
        return f"{self.function}@0x{self.site:08X} {self.op} {self.detail}" + "".join(f" <- {s}" for s in self.via)


@dataclass(frozen=True)
class Constraint:
    kind: str                        # access | pointer | address | narrow | actual
    evidence: Evidence
    width: int = 0
    signed: bool | None = None
    floating: bool = False
    store: bool = False
    indexed: bool = False

    def key(self) -> tuple:
        return (self.kind, self.width, self.signed, self.floating, self.store, self.indexed,
                self.evidence.function, self.evidence.site)


@dataclass(frozen=True)
class Conflict:
    kind: str                        # width | float | signedness | narrowing | pointer | overlap
    detail: str
    evidence: tuple[Evidence, ...]


@dataclass
class Inferred:
    type: str                        # C type; alternatives joined by "|" under a conflict; "?" unknown
    size: int | None
    width: int | None = None
    signed: bool | None = None       # None: no narrow load proves it (always None at 4/8 bytes)
    floating: bool = False
    pointer: bool = False
    indexed: bool = False            # accessed with a register index (array)
    fields: dict[int, "Inferred"] = field(default_factory=dict)   # members, or the pointee's fields
    layout: str | None = None        # struct name describing ``fields`` when the type uses one
    evidence: list[Evidence] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def all_conflicts(self) -> list[Conflict]:
        return self.conflicts + [c for f in self.fields.values() for c in f.all_conflicts()]


@dataclass
class TypeReport:
    globals: dict[int, Inferred]
    arguments: dict[tuple[int, int], Inferred]       # (function vram, register 4..7)
    returns: dict[int, Inferred]
    names: dict[int, str]
    constraints: int
    truncated: int                   # projections dropped by the path/offset/via/size limits
    notes: list[str]


def _simm(word: int) -> int:
    return (word & 0xFFFF) - 0x10000 if word & 0x8000 else word & 0xFFFF


def _control(word: int, pc: int):
    """``(kind, target, likely)`` for a control transfer (branch|jump|call|return), else None."""
    op, rs, rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
    rel = (pc + 4 + (_simm(word) << 2)) & MASK
    if op in (2, 3):
        return ("jump" if op == 2 else "call"), ((pc + 4) & 0xF0000000) | (word & 0x3FFFFFF) << 2, False
    if op == 0 and word & 0x3F in (8, 9):
        return ("call" if word & 0x3F == 9 else "return" if rs == 31 else "jump"), None, False
    if op == 4 and rs == 0 and rt == 0:
        return "jump", rel, False
    if 4 <= op <= 7 or 20 <= op <= 23:
        return "branch", rel, op >= 20
    if op == 1 and (rt < 4 or 16 <= rt < 20):
        return ("call" if rt & 0x10 else "branch"), rel, bool(rt & 2)
    if op == 17 and rs == 8:
        return "branch", rel, bool(rt & 2)
    return None


def _dest(word: int) -> int | None:
    """The general-purpose register an instruction writes (0 or None: none)."""
    op, rs, rt, rd, funct = word >> 26, (word >> 21) & 31, (word >> 16) & 31, (word >> 11) & 31, word & 0x3F
    if op == 0:
        return None if funct in (8, 0xC, 0xD, 0xF, 0x11, 0x13) or 0x18 <= funct <= 0x1F or funct >= 0x30 else rd
    if 8 <= op <= 15 or 32 <= op <= 39 or op in (24, 25, 26, 27, 48, 52, 55, 56, 60):
        return rt
    if op in (16, 17, 18) and rs in (0, 1, 2):
        return rt
    return 31 if op == 3 else None


class _Store:
    def __init__(self, names: dict[int, str], functions: dict[int, str]):
        self.cells: dict[tuple, dict[tuple, Constraint]] = {}
        self.bindings: dict[tuple, list] = {}
        self.names, self.functions, self.notes = names, functions, []

    def add(self, cell: tuple, con: Constraint) -> bool:
        bucket = self.cells.setdefault(cell, {})
        if con.key() in bucket:
            return False
        bucket[con.key()] = con
        return True

    def bind(self, root: tuple, cell: tuple, delta: int, ev: Evidence) -> None:
        if cell[0] == root:
            self.notes.append(f"recursive flow not propagated: {ev}")
        else:
            self.bindings.setdefault(root, []).append((cell, delta, ev))

    def fname(self, vram: int) -> str:
        return self.functions.get(vram) or f"func_{vram:08X}"

    def show(self, cell: tuple, delta: int = 0) -> str:
        root, path = cell
        text = ("&" + (self.names.get(root[1]) or f"D_{root[1]:08X}") if root[0] == "ga" else
                f"{self.fname(root[1])}.arg{root[2]}" if root[0] == "a" else f"{self.fname(root[1])}()")
        for offset in path:
            text = f"*({text}{offset:+#x})"
        return text + (f"{delta:+#x}" if delta else "")


def _plain(v) -> bool:
    """An unmodified incoming argument or call result (what narrowing evidence is about)."""
    return bool(v) and v[0] == "v" and v[2] == 0 and not v[3] and v[1][0][0] in ("a", "r") and not v[1][1]


def _scan(name: str, vram: int, code: bytes, gp, ram, store: _Store) -> None:
    n = len(code) // 4
    words = struct.unpack(f">{n}I", code[:4 * n])
    end, code_addrs = vram + 4 * n, store.functions
    labels, writes, slot_writes = set(), {}, {}
    for i, w in enumerate(words):
        c = _control(w, vram + 4 * i)
        if c and c[1] is not None and c[0] != "call" and vram <= c[1] < end:
            labels.add((c[1] - vram) // 4)
        if _dest(w):
            writes[_dest(w)] = i if _dest(w) not in writes else -1
        if w >> 26 in (43, 63) and (w >> 21) & 31 == 29:
            slot_writes[_simm(w)] = i if _simm(w) not in slot_writes else -1
    entry = {4 + k: ("v", (("a", vram, k), ()), 0, False) for k in range(4)}
    kept_regs, kept_slots, fresh = {}, {}, set()
    regs, slots = dict(entry), {}

    def at_label():
        return {**{r: v for r, v in entry.items() if r not in writes}, **kept_regs}, dict(kept_slots)

    def ev(i, op, detail):
        return Evidence(name, vram + 4 * i, op, detail)

    def put(r, v):
        if r and v is None:
            regs.pop(r, None)
        elif r:
            regs[r] = v

    def address(base, imm):
        a = (base + imm) & MASK
        return a if ram[0] <= a < ram[1] and a not in code_addrs else None

    def access(i, w, op, width, signed, floating, is_store):
        rs, rt, imm = (w >> 21) & 31, (w >> 16) & 31, _simm(w)
        gpr_load = not is_store and not floating
        if rs == 29:                                        # stack: word spills and reloads only
            if is_store and width >= 4 and not floating and regs.get(rt) is not None:
                slots[imm] = regs[rt]
            elif is_store:
                slots.pop(imm, None)
            elif gpr_load:
                put(rt, slots.get(imm) if width >= 4 else None)
            return
        b = ("hi", gp, False) if rs == 28 and gp is not None else regs.get(rs)
        cell = indexed = None
        if b and b[0] == "hi" and address(b[1], imm) is not None:
            cell, indexed = (("ga", address(b[1], imm)), (0,)), b[2]
        elif b and b[0] == "v" and len(b[1][1]) < MAX_PATH and MIN_OFFSET <= b[2] + imm < MAX_OFFSET:
            cell, indexed = (b[1][0], b[1][1] + (b[2] + imm,)), b[3]
            if b[1][0][0] != "ga" or b[1][1]:
                store.add(b[1], Constraint("pointer", ev(i, op, f"base of {store.show(cell)}")))
        if cell is None:
            return put(rt, None) if gpr_load else None
        store.add(cell, Constraint("access", ev(i, op, store.show(cell)), width, signed, floating, is_store, indexed))
        v = regs.get(rt)
        if is_store and not floating and v and v[0] == "v" and not v[3]:
            if v[1][0][0] == "ga" and not v[1][1]:
                store.add(cell, Constraint("address", ev(i, op, f"stores {store.show(v[1], v[2])}")))
            elif not v[1][1]:                               # root = value(cell) - delta
                store.bind(v[1][0], cell, -v[2], ev(i, op, f"stores {store.show(v[1], v[2])} into {store.show(cell)}"))
        elif gpr_load:
            put(rt, ("v", cell, 0, False))

    def step(i, w):
        op, rs, rt, rd, sa, funct = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31, (w >> 6) & 31, w & 0x3F
        if op in LOADS:
            return access(i, w, *LOADS[op], False)
        if op in STORES:
            return access(i, w, STORES[op][0], STORES[op][1], None, STORES[op][2], True)
        if op == 15:
            return put(rt, ("hi", (w & 0xFFFF) << 16, False))
        if op in (8, 9, 24, 25):                            # addi(u) / daddi(u)
            b = ("hi", gp, False) if rs == 28 and gp is not None else regs.get(rs)
            if b and b[0] == "hi":
                a = address(b[1], _simm(w))
                return put(rt, ("v", (("ga", a), ()), 0, b[2]) if a is not None else None)
            return put(rt, ("v", b[1], b[2] + _simm(w), b[3]) if b and b[0] == "v" else None)
        if op == 12 or (op == 0 and funct in (2, 3)):       # andi 0xFF/0xFFFF, or the srl/sra of a pair
            v = regs.get(rs) if op == 12 else regs.get(rt)
            if op == 12 and w & 0xFFFF in (0xFF, 0xFFFF) and _plain(v):
                store.add(v[1], Constraint("narrow", ev(i, "andi", f"{store.show(v[1])} & {w & 0xFFFF:#x}"),
                                           1 if w & 0xFFFF == 0xFF else 2, False))
            elif op == 0 and v and v[0] == "shl" and v[2] == sa:
                store.add(v[1], Constraint("narrow", ev(i, "sll+" + ("sra" if funct == 3 else "srl"),
                                           f"{store.show(v[1])} by {sa}"), 2 if sa == 16 else 1, funct == 3))
            return put(rt if op == 12 else rd, None)
        if op == 0 and funct == 0:                          # sll
            v = regs.get(rt)
            return put(rd, ("shl", v[1], sa) if sa in (16, 24) and _plain(v) else None)
        if op == 0 and funct in (0x21, 0x25, 0x2D):         # addu / or / daddu
            left, right = regs.get(rs), regs.get(rt)
            if rs == 0 or rt == 0:
                return put(rd, None if rs == rt else left if rt == 0 else right)
            bases = [x for x in (left, right) if x and funct != 0x25 and
                     (x[0] == "hi" or (x[0] == "v" and x[1][0][0] == "ga" and not x[1][1]))]
            return put(rd, bases[0][:-1] + (True,) if len(bases) == 1 else None)
        if op == 17 and rs == 4 and _plain(regs.get(rt)):  # mtc1: an integer register holds float bits
            store.add(regs[rt][1], Constraint("narrow", ev(i, "mtc1", store.show(regs[rt][1])), 4, None, True))
            return None
        return put(_dest(w), None)

    def run(i, w):
        step(i, w)
        d, imm = _dest(w), _simm(w)
        if d in STABLE and writes.get(d) == i and d in regs:
            kept_regs[d] = regs[d]
        if d in (4, 5, 6, 7):
            fresh.add(d)
        if w >> 26 in (43, 63) and (w >> 21) & 31 == 29 and slot_writes.get(imm) == i and imm in slots:
            kept_slots[imm] = slots[imm]

    def call(i, target):
        for k in range(4):
            v = regs.get(4 + k)
            if target is None or not v or v[0] != "v" or v[3]:
                continue
            e = Evidence(name, vram + 4 * i, "call", f"passes {store.show(v[1], v[2])} as $a{k} to {store.fname(target)}")
            store.bind(("a", target, k), v[1], v[2], e)
            if 4 + k in fresh:                              # set for this call, not left over
                kind = "address" if v[1][0][0] == "ga" and not v[1][1] else "actual"
                store.add((("a", target, k), ()), Constraint(kind, e))
        fresh.clear()
        for r in CLOBBERED:
            regs.pop(r, None)
        if target is not None:
            regs[2] = ("v", (("r", target), ()), 0, False)

    i = 0
    while i < n:
        if i in labels and i:
            regs, slots = at_label()
        c = _control(words[i], vram + 4 * i)
        if c is None:
            run(i, words[i])
            i += 1
            continue
        kind, target, likely = c
        if i + 1 < n and likely:                            # the delay slot runs on the taken path only
            saved = dict(regs), dict(slots)
            run(i + 1, words[i + 1])
            regs, slots = saved
            put(_dest(words[i + 1]), None)
        elif i + 1 < n:
            run(i + 1, words[i + 1])
        if kind == "call" or (kind == "jump" and target is not None and not vram <= target < end):
            call(i, target)                                 # a jump out of the function is a tail call
        v = regs.get(2)
        if kind == "return" and v and v[0] == "v" and not v[3]:
            store.bind(("r", vram), v[1], v[2], ev(i, "jr", f"{name} returns {store.show(v[1], v[2])}"))
        if kind in ("jump", "return"):
            regs, slots = at_label()                        # what follows is reached only by a branch
        i += 2


def _propagate(store: _Store) -> int:
    queue = deque((cell, con) for cell, bucket in store.cells.items() for con in bucket.values())
    truncated = added = 0
    while queue:
        (root, path), con = queue.popleft()
        for target, delta, ev in store.bindings.get(root, ()):
            if not path:                                    # only pointer use of the value itself flows back
                if con.kind != "pointer" or (target[0][0] == "ga" and not target[1]):
                    continue
                new = target
            else:
                new = (target[0], target[1] + (delta + path[0],) + path[1:])
            if len(new[1]) > MAX_PATH or path and not MIN_OFFSET <= delta + path[0] < MAX_OFFSET \
                    or len(con.evidence.via) >= MAX_VIA or added >= MAX_CONSTRAINTS:
                truncated += 1
                continue
            hop = f"{ev.function}@0x{ev.site:08X} {ev.detail}"
            moved = replace(con, evidence=replace(con.evidence, via=con.evidence.via + (hop,)))
            if store.add(new, moved):
                queue.append((new, moved))
                added += 1
    return truncated


def _scalar(width: int, signed, floating: bool) -> str:
    if floating:
        return {4: "f32", 8: "f64"}.get(width, "?")
    return {1: "s8" if signed else "u8", 2: "s16" if signed else "u16", 4: "s32", 8: "s64"}.get(width, "?")


def _forms(cons, key) -> dict:
    out: dict = {}
    for con in cons:
        out.setdefault(key(con), []).append(con)
    return out


def _conflict(kind: str, forms: dict, label) -> Conflict:
    parts = []
    for form, cons in sorted(forms.items(), key=lambda item: repr(item[0])):
        sites = ", ".join(f"{c.evidence.function}@0x{c.evidence.site:08X}" for c in cons[:3])
        parts.append(f"{label(form)} at {sites}" + (" ..." if len(cons) > 3 else ""))
    return Conflict(kind, "; ".join(parts), tuple(c.evidence for cons in forms.values() for c in cons))


def _infer(cell: tuple, label: str, store: _Store, kids: dict) -> Inferred:
    cons = sorted(store.cells.get(cell, {}).values(), key=lambda c: (c.evidence.function, c.evidence.site))
    fields = {o: _infer((cell[0], cell[1] + (o,)), f"{label}_{o:X}", store, kids) for o in sorted(kids.get(cell, ()))}
    acc = [c for c in cons if c.kind == "access"]
    narrow = [c for c in cons if c.kind == "narrow"]
    pointers = [c for c in cons if c.kind in ("pointer", "address")]
    conflicts = []
    shapes = _forms(acc, lambda c: (c.width, c.floating))
    if len({w for w, _f in shapes}) > 1:
        conflicts.append(_conflict("width", shapes, lambda f: f"{f[0]} byte(s){' float' if f[1] else ''}"))
    elif len(shapes) > 1:
        conflicts.append(_conflict("float", shapes, lambda f: "float" if f[1] else "integer"))
    signs = _forms([c for c in acc if c.signed is not None and not c.store and c.width < 4 and not c.floating],
                   lambda c: c.signed)
    if len(signs) > 1:
        conflicts.append(_conflict("signedness", signs, lambda s: "signed load" if s else "unsigned load"))
    narrows = _forms(narrow, lambda c: (c.width, c.signed, c.floating))
    if len(narrows) > 1:
        conflicts.append(_conflict("narrowing", narrows, lambda f: _scalar(*f)))
    pointer = bool(pointers or fields)
    misuse = [c for c in acc if c.width != 4 or c.floating] + narrow
    if pointer and misuse:
        conflicts.append(_conflict("pointer", {"pointer use": pointers[:3] or acc[:1], "non-pointer use": misuse}, str))
    ordered = sorted(fields.items())
    for (o1, f1), (o2, f2) in zip(ordered, ordered[1:]):
        if o1 + (f1.size or 1) > o2:
            conflicts.append(Conflict("overlap", f"0x{o1:X} ({f1.type}) overlaps 0x{o2:X} ({f2.type})",
                                      tuple(f1.evidence[:2] + f2.evidence[:2])))
    signed = next(iter(signs)) if len(signs) == 1 else None
    width = floating = None
    if len(shapes) == 1:
        width, floating = next(iter(shapes))
    elif not shapes and len(narrows) == 1:
        width, signed, floating = next(iter(narrows))
    layout = None
    if pointer:
        size = 4
        if not fields:
            ctype = "void *"
        elif list(fields) == [0]:
            ctype = f"{fields[0].type} *"
        else:
            ctype, layout = f"struct {label}_s *", f"{label}_s"
    elif width:
        ctype, size = _scalar(width, signed, bool(floating)), width
    elif shapes:
        ctype, size = "|".join(_scalar(w, None, f) for w, f in sorted(shapes)), max(w for w, _f in shapes)
    else:
        ctype, size = "?", None
    return Inferred(ctype, size, width, signed, bool(floating), pointer, any(c.indexed for c in acc), fields,
                    layout, [c.evidence for c in cons], conflicts)


def infer(image: bytes, vram_delta: int, functions, *, gp: int | None = None, ram: tuple[int, int] = RAM,
          names: dict[int, str] | None = None) -> TypeReport:
    """``functions`` is ``[(name, vram, size)]``, code at file offset ``vram - vram_delta``.
    ``names`` optionally names data addresses; ``gp`` resolves ``$gp``-relative accesses."""
    fnames = {vram: name for name, vram, _size in functions}
    store = _Store(dict(names or {}), fnames)
    for name, vram, size in functions:
        offset = vram - vram_delta
        if size < 4 or offset < 0 or offset + size > len(image):
            store.notes.append(f"{name}: outside the image, skipped")
        else:
            _scan(name, vram, image[offset:offset + size], gp, ram, store)
    truncated = _propagate(store)
    kids: dict = {}
    for root, path in store.cells:
        for depth in range(len(path)):
            kids.setdefault((root, path[:depth]), set()).add(path[depth])
    report = TypeReport({}, {}, {}, store.names, sum(map(len, store.cells.values())), truncated, store.notes)
    for root in sorted({cell[0] for cell in store.cells}, key=repr):
        if root[0] == "ga":
            label = store.names.get(root[1]) or f"D_{root[1]:08X}"
            obj = _infer((root, ()), label, store, kids)
            members = obj.fields
            if list(members) == [0]:
                only = members[0]
                report.globals[root[1]] = replace(only, type=f"{only.type}[]", size=None) if only.indexed else only
            elif members:
                report.globals[root[1]] = Inferred(
                    f"struct {label}_s", max(o + (f.size or 1) for o, f in members.items()),
                    indexed=any(f.indexed for f in members.values()), fields=members, layout=f"{label}_s",
                    evidence=[e for f in members.values() for e in f.evidence], conflicts=obj.conflicts)
        elif root[1] in fnames:
            label = f"{fnames[root[1]]}_arg{root[2]}" if root[0] == "a" else f"{fnames[root[1]]}_ret"
            info = _infer((root, ()), label, store, kids)
            if root[0] == "a":
                report.arguments[(root[1], 4 + root[2])] = info
            else:
                report.returns[root[1]] = info
    return report


def _declare(ctype: str, name: str) -> str:
    if ctype.endswith("[]"):
        return f"{_declare(ctype[:-2], name)}[]"
    return f"{ctype}{name}" if ctype.endswith("*") else f"{ctype} {name}"


def _complete(info: Inferred, top: bool = True) -> bool:
    if "?" in info.type or "|" in info.type or (info.indexed and not (top and info.type.endswith("[]"))):
        return False
    return all(o >= 0 and _complete(f, False) for o, f in info.fields.items())


def _unproven(info: Inferred) -> str:
    narrow = info.width in (1, 2) and info.signed is None and not info.pointer and not info.floating
    return "   /* signedness unproven */" if narrow else ""


def _structs(info: Inferred, out: list[str], done: set[str]) -> None:
    for item in info.fields.values():
        _structs(item, out, done)
    if not info.layout or info.layout in done:
        return
    done.add(info.layout)
    lines, cursor = [f"struct {info.layout} {{   /* typeflow: observed fields only; true size unknown */"], 0
    for offset, item in sorted(info.fields.items()):
        if offset > cursor:
            lines.append(f"    u8 pad_{cursor:X}[0x{offset - cursor:X}];")
        lines.append(f"    /* 0x{offset:02X} */ {_declare(item.type, f'unk_{offset:X}')};{_unproven(item)}")
        cursor = offset + (item.size or 1)
    out.append("\n".join(lines + ["};"]))


def declarations(report: TypeReport) -> str:
    """C declarations for a draft context: every global whose whole type tree has no conflict."""
    out, done = ["/* typeflow: provisional declarations from machine-code constraints; conflict-free only */"], set()
    for address in sorted(report.globals):
        info = report.globals[address]
        if info.all_conflicts() or not _complete(info):
            continue
        _structs(info, out, done)
        out.append(f"extern {_declare(info.type, report.names.get(address) or f'D_{address:08X}')};{_unproven(info)}")
    return "\n".join(out) + "\n"
