"""Semantic IR for one MIPS (R4300) function: lifting, SSA, fingerprints and similarity.

``lift`` turns the words of one function into basic blocks of statements over hash-consed SSA
values.  Delay slots are resolved (branch conditions and ``jr``/``jalr`` targets are read before
the slot, call arguments and returned values after it; a branch-likely slot moves to a synthetic
block on the taken edge).  Registers, hi/lo, the FP condition, memory and, when the stack pointer
never escapes, stack slots are SSA variables (Braun et al. 2013), so moves, register choice,
spills and callee-saved saves leave no trace; ``lui``/``addiu`` pairs and ``(x+4)+8`` fold.
Anything outside the supported subset is an explicit ``unknown`` statement: never skipped.

``fingerprint`` summarises a function without register names, frame offsets or instruction
order; ``compare`` scores two fingerprints.  Equal digests mean "same semantics up to allocation
and scheduling" under this abstraction; byte equality stays the only proof of a match.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

M32 = 0xFFFFFFFF
SP, RA, FP0 = 29, 31, 32                       # FPR n is SSA variable FP0 + n
GPR = ("zero at v0 v1 a0 a1 a2 a3 t0 t1 t2 t3 t4 t5 t6 t7 s0 s1 s2 s3 s4 s5 s6 s7 "
       "t8 t9 k0 k1 gp sp fp ra").split()
ENTRY = {4: "arg0", 5: "arg1", 6: "arg2", 7: "arg3", FP0 + 12: "farg0", FP0 + 14: "farg1", SP: "sp", RA: "ra"}
ARGS = (4, 5, 6, 7, FP0 + 12, FP0 + 14)
CALLER_SAVED = (*range(1, 16), 24, 25, RA, *range(FP0, FP0 + 20), "hi", "lo", "fcc")
SAVED = frozenset((*range(16, 24), 30, RA, *range(FP0 + 20, FP0 + 32)))   # what a callee saves
OPAQUE = ("undef", "in")                       # shapes without semantics: garbage, untouched inputs
COMMUTATIVE = frozenset("add and or xor nor eq mul mulu mulhi mulhiu dadd dmul dmulu add.s add.d mul.s mul.d "
                        "c.eq.s c.eq.d c.ueq.s c.ueq.d c.seq.s c.seq.d".split())
_SHIFT = {0: ("shl", 0), 2: ("shr", 0), 3: ("sar", 0), 56: ("dshl", 0), 58: ("dshr", 0), 59: ("dsar", 0),
          60: ("dshl", 32), 62: ("dshr", 32), 63: ("dsar", 32)}
_SHIFTV = {4: "shl", 6: "shr", 7: "sar", 20: "dshl", 22: "dshr", 23: "dsar"}
_ALU = {32: "add", 33: "add", 34: "sub", 35: "sub", 36: "and", 37: "or", 38: "xor", 39: "nor",
        42: "lt", 43: "ltu", 44: "dadd", 45: "dadd", 46: "dsub", 47: "dsub"}
_MULDIV = {24: ("mul", "mulhi"), 25: ("mulu", "mulhiu"), 26: ("div", "rem"), 27: ("divu", "remu"),
           28: ("dmul", "dmulhi"), 29: ("dmulu", "dmulhiu"), 30: ("ddiv", "drem"), 31: ("ddivu", "dremu")}
_IMM = {8: "add", 9: "add", 10: "lt", 11: "ltu", 12: "and", 13: "or", 14: "xor", 24: "dadd", 25: "dadd"}
_LOADS = {32: (1, "s"), 33: (2, "s"), 35: (4, "s"), 36: (1, "u"), 37: (2, "u"), 39: (4, "u"), 55: (8, "s"),
          49: (4, "f"), 53: (8, "f"), 34: (4, "l"), 38: (4, "r"), 26: (8, "l"), 27: (8, "r")}
_STORES = {40: (1, ""), 41: (2, ""), 43: (4, ""), 63: (8, ""), 57: (4, "f"), 61: (8, "f"),
           42: (4, "l"), 46: (4, "r"), 44: (8, "l"), 45: (8, "r")}
_FUN = {4: "sqrt", 5: "abs", 7: "neg", 32: "cvt.s", 33: "cvt.d", 36: "cvt.w", 37: "cvt.l",
        **dict(enumerate("round.l trunc.l ceil.l floor.l round.w trunc.w ceil.w floor.w".split(), 8))}
_CONDS = "f un eq ueq olt ult ole ule sf ngle seq ngl lt nge le ngt".split()
_FMT = {16: "s", 17: "d", 20: "w", 21: "l"}


def _s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


_FOLD = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "and": lambda a, b: a & b,
         "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b, "nor": lambda a, b: ~(a | b),
         "shl": lambda a, b: a << (b & 31), "shr": lambda a, b: (a & M32) >> (b & 31),
         "sar": lambda a, b: _s32(a) >> (b & 31), "lt": lambda a, b: int(_s32(a) < _s32(b)),
         "ltu": lambda a, b: int((a & M32) < (b & M32)), "eq": lambda a, b: int(a == b)}


class Value:
    """An SSA value (expression node): ``args`` are value ids, ``attr`` what is not a value
    (a constant, a width, the entry variable, the memory version a load reads)."""
    __slots__ = ("id", "op", "args", "attr", "block", "var")

    def __init__(self, id, op, args, attr, block, var=None):
        self.id, self.op, self.args, self.attr, self.block, self.var = id, op, args, attr, block, var


@dataclass
class Stmt:
    kind: str            # store call tail branch goto ret ijump trap setfcsr unknown stop
    args: tuple = ()     # value ids; call/tail: (target, a0, a1, a2, a3, f12, f14); ret: (v0, f0)
    attr: object = None
    targets: tuple = ()  # block ids; branch: (taken, not taken)
    pc: int = 0


@dataclass
class Block:
    id: int
    start: int           # first instruction index; -1 for a synthetic block
    succs: list = field(default_factory=list)
    preds: list = field(default_factory=list)
    stmts: list = field(default_factory=list)
    values: list = field(default_factory=list)
    body: range = range(0)
    plan: tuple = ()


def _varname(var) -> str:
    if isinstance(var, int):
        return GPR[var] if var < FP0 else f"f{var - FP0}"
    return f"sp{var[1]:+#x}" if isinstance(var, tuple) else str(var)


def _const_text(c: int, symbolic: bool = False) -> str:
    if symbolic and 0x80000000 <= c < 0x80800000:
        return "#addr"
    return f"#-{(1 << 32) - c:x}" if 0xFFFF8000 <= c <= M32 else f"#{c:x}"


def _opname(v: Value) -> str:
    if v.op in ("load", "ext"):
        return f"{v.op}{8 * v.attr[0]}{v.attr[1]}"
    return f"result.{v.attr[1]}" if v.op == "result" else v.op


def _find(replaced: dict, vid: int) -> int:
    while vid in replaced:
        vid = replaced[vid]
    return vid


def _sp_offset(values, replaced, vid):
    """Entry-relative stack offset when ``vid`` is ``sp + constant``, else None."""
    off = 0
    for _ in range(64):
        v = values[_find(replaced, vid)]
        if v.op == "in" and v.attr == SP:
            return _s32(off)
        if v.op != "add" or values[_find(replaced, v.args[1])].op != "const":
            return None
        off, vid = off + values[_find(replaced, v.args[1])].attr, v.args[0]
    return None


def _postorder(start, succs) -> list:
    seen, post, stack = {start}, [], [(start, iter(succs(start)))]
    while stack:
        node, it = stack[-1]
        for nxt in it:
            if nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(succs(nxt))))
                break
        else:
            post.append(node)
            stack.pop()
    return post


def _control(word: int, pc: int):
    """(kind, target, likely) of a control-transfer instruction, else None."""
    op, rs, rt, fn = word >> 26, (word >> 21) & 31, (word >> 16) & 31, word & 63
    imm = word & 0xFFFF
    rel = (pc + 4 + ((imm - 0x10000 if imm & 0x8000 else imm) << 2)) & M32
    if op == 0 and fn in (8, 9):
        return ("jalr" if fn == 9 else "ret" if rs == RA else "jr", None, False)
    if op in (2, 3):
        return ("j" if op == 2 else "jal", ((pc + 4) & 0xF0000000) | ((word & 0x3FFFFFF) << 2), False)
    if 4 <= op <= 7 or 20 <= op <= 23:
        return ("br", rel, op >= 20)
    if op == 1 and (rt < 4 or 16 <= rt < 20):
        return ("br", rel, rt in (2, 3)) if rt < 4 else ("jal" if rt == 17 and rs == 0 else "bad", rel, False)
    return ("br", rel, bool(rt & 2)) if op == 17 and rs == 8 else None


@dataclass
class Function:
    vram: int
    words: tuple
    blocks: list
    values: list
    replaced: dict
    promoted: bool       # stack slots were SSA variables (the stack pointer never escaped)

    def find(self, vid: int) -> int:
        return _find(self.replaced, vid)

    def value(self, vid: int) -> Value:
        return self.values[self.find(vid)]

    def statements(self, kind: str | None = None) -> list:
        return [(b, s) for b in self.blocks for s in b.stmts if kind is None or s.kind == kind]

    def reachable(self) -> set:
        return set(_postorder(0, lambda b: self.blocks[b].stmts[-1].targets))

    def def_use(self) -> dict:
        """value id -> users: ("value", id) or ("stmt", block, index)."""
        uses = defaultdict(list)
        for v in self.values:
            for a in (v.args if v.id not in self.replaced else ()):
                uses[self.find(a)].append(("value", v.id))
        for b in self.blocks:
            for n, s in enumerate(b.stmts):
                for a in s.args:
                    uses[self.find(a)].append(("stmt", b.id, n))
        return dict(uses)

    def ref(self, vid: int) -> str:
        v = self.value(vid)
        if v.op == "const":
            return _const_text(v.attr)
        if v.op == "in":
            return ENTRY.get(v.attr) or f"{_varname(v.attr)}.in"
        return "undef" if v.op == "undef" else f"v{v.id}"

    def expr(self, vid: int) -> str:
        v = self.value(vid)
        if v.op in ("const", "in", "undef"):
            return self.ref(vid)
        return f"{_opname(v)}({', '.join(self.ref(a) for a in v.args)})"

    def stmt_text(self, s: Stmt) -> str:
        r, k = self.ref, s.kind
        if k == "store":
            return f"store{8 * s.attr[0]}{s.attr[1]}({r(s.args[0])}, {r(s.args[1])})"
        if k in ("call", "tail"):
            args = [f"{_varname(var)}={r(a)}" for var, a in zip(ARGS, s.args[1:]) if self.value(a).op != "undef"]
            return f"{k}({r(s.args[0])}, args=[{', '.join(args)}])"
        if k in ("branch", "goto"):
            cond = [self.expr(s.args[0])] if s.args else []
            return f"{k}({', '.join(cond + [f'B{t}' for t in s.targets])})"
        if k == "ret":
            return f"ret(v0={r(s.args[0])}, f0={r(s.args[1])})"
        if k == "unknown":
            word, name, reason = s.attr
            return f"unknown({name} {'-' if word is None else f'0x{word:08X}'} at 0x{s.pc:08X}: {reason})"
        return f"{k}({', '.join([r(a) for a in s.args] + ([] if s.attr is None else [str(s.attr)]))})"

    def text(self) -> str:
        live, out = self.reachable(), []
        for b in self.blocks:
            where = f"0x{self.vram + 4 * b.start:08X}" if b.start >= 0 else "synthetic"
            out.append(f"B{b.id}:  ; {where} preds={b.preds}" + ("" if b.id in live else " unreachable"))
            for vid in b.values:
                v = self.values[vid]
                if vid not in self.replaced and v.op not in ("const", "in", "undef", "mem"):
                    out.append(f"  v{vid} = {self.expr(vid)}" + (f"  ; {_varname(v.var)}" if v.var is not None else ""))
            out.extend("  " + self.stmt_text(s) for s in b.stmts)
        return "\n".join(out)


class LiftBudgetExceeded(ValueError):
    """The lift needed more SSA values than its ``max_values`` budget (a bound on its memory)."""


class _Lifter:
    def __init__(self, words: tuple, vram: int, slots: dict | None = None, max_values: int | None = None):
        self.words, self.vram, self.n, self.slots = words, vram, len(words), slots
        self.limit = sys.maxsize if max_values is None else max_values
        self.values, self.cons, self.replaced = [], {}, {}
        self.defs, self.sealed, self.incomplete = {}, set(), {}
        self.sp_access, self.escaped, self.cur = [], False, None
        self.ctl = [_control(w, vram + 4 * i) for i, w in enumerate(words)]
        self.blocks = self._plan()

    def _index(self, address):
        delta = address - self.vram if address is not None else -1
        return delta // 4 if 0 <= delta < 4 * self.n and not delta & 3 else None

    def _plan(self) -> list:
        n, ctl, leaders = self.n, self.ctl, {0}
        for i, c in enumerate(ctl):
            if c and i + 2 < n:
                leaders.add(i + 2)
            if c and c[0] in ("br", "j") and self._index(c[1]) is not None:
                leaders.add(self._index(c[1]))
        blocks = [Block(k, s) for k, s in enumerate(sorted(leaders))]
        at = {b.start: b.id for b in blocks}

        def synthetic(plan, succs=()):
            blocks.append(Block(len(blocks), -1, succs=list(succs), plan=plan))
            return blocks[-1].id

        for b in list(blocks):
            i = b.start
            while i < n and not ctl[i] and i + 1 not in at:
                i += 1
            if i >= n:
                b.plan, b.body = ("end",), range(b.start, n)
            elif not ctl[i]:
                b.plan, b.body, b.succs = ("fall",), range(b.start, i + 1), [at[i + 1]]
            else:
                kind, target, likely = ctl[i]
                b.plan, b.body, t = ("ctl", i), range(b.start, i), self._index(target)
                if kind == "br":
                    t = at[t] if t is not None else synthetic(("exit", target))
                    after = at[i + 2] if i + 2 < n else synthetic(("exit", self.vram + 4 * (i + 2)))
                    b.succs = [synthetic(("slot", i + 1), [t]) if likely else t, after]
                elif kind == "j" and t is not None:
                    b.succs = [at[t]]
                elif kind in ("jal", "jalr") and i + 2 < n:
                    b.succs = [at[i + 2]]
        for b in blocks:
            for s in b.succs:
                blocks[s].preds.append(b.id)
        return blocks

    # ---- values: hash-consed, folded --------------------------------------------------------
    def find(self, vid: int) -> int:
        return _find(self.replaced, vid)

    def new(self, op, args=(), attr=None, block=None) -> int:
        if block is None:
            block = -1 if self.cur is None or op in ("const", "in") else self.cur.id
        if len(self.values) >= self.limit:
            raise LiftBudgetExceeded(f"more than {self.limit} SSA values for {self.n} instructions")
        v = Value(len(self.values), op, args if op == "phi" else tuple(args), attr, block)
        self.values.append(v)
        if block >= 0:
            self.blocks[block].values.append(v.id)
        return v.id

    def _cons(self, op, args, attr) -> int:
        if (op, args, attr) not in self.cons:
            self.cons[(op, args, attr)] = self.new(op, args, attr)
        return self.cons[(op, args, attr)]

    def k(self, c: int, mask: int = M32) -> int:
        return self._cons("const", (), c & mask)

    def _entry(self, var) -> int:
        vid = self._cons("in", (), var)
        self.values[vid].var = var
        return vid

    def mk(self, op, args, attr=None) -> int:
        V, args = self.values, [self.find(a) for a in args]
        isk = [V[a].op == "const" for a in args]
        if op in COMMUTATIVE and len(args) == 2 and (isk[0], args[0]) > (isk[1], args[1]):
            args.reverse()
            isk.reverse()
        ks = [V[a].attr if c else None for a, c in zip(args, isk)]
        if args and all(isk) and op in _FOLD:
            return self.k(_FOLD[op](*ks))
        if args and all(isk) and op == "pair":
            return self.k((ks[0] << 32) | (ks[1] & M32), (1 << 64) - 1)
        if op in ("mtc1", "mfc1") and (isk[0] or V[args[0]].op == ("mfc1" if op == "mtc1" else "mtc1")):
            return args[0] if isk[0] else V[args[0]].args[0]
        if len(args) == 2 and isk[1]:
            a, c, inner = args[0], ks[1], V[args[0]]
            if op == "sub":
                return self.mk("add", (a, self.k(-c)))
            if c == 0 and op in ("add", "or", "xor", "shl", "shr", "sar", "dadd", "dshl", "dshr", "dsar"):
                return a
            if op == "add" and inner.op == "add" and V[self.find(inner.args[1])].op == "const":
                return self.mk("add", (inner.args[0], self.k(c + V[self.find(inner.args[1])].attr)))
        if op == "eq" and len(args) == 2 and args[0] == args[1]:
            return self.k(1)
        return self._cons(op, tuple(args), attr)

    # ---- SSA construction (Braun et al. 2013; predecessors are all known up front) ----------
    def _preds(self, b: int) -> list:
        return [-1] + self.blocks[0].preds if b == 0 else self.blocks[b].preds

    def write(self, var, vid: int) -> None:
        if var != 0:
            if self.values[vid].var is None:
                self.values[vid].var = var
            self.defs[(self.cur.id, var)] = vid

    def read(self, var, block: int | None = None) -> int:
        if var == 0:
            return self.k(0)
        b, chain, seen = self.cur.id if block is None else block, [], set()
        # ``seen`` ends the walk in a cycle of single-predecessor blocks (unreachable code, e.g. a
        # dead self-loop or a mis-targeted ``j`` of an unlinked object): without it the walk never
        # ends and ``chain`` grows until memory runs out.  The cycle then gets a phi whose only
        # inputs are itself: undef, as Braun et al.'s marker phi gives for unreachable code.
        while (b, var) not in self.defs and b in self.sealed and len(self._preds(b)) == 1 and b not in seen:
            seen.add(b)
            chain.append(b)
            b = self._preds(b)[0]
        vid = self.defs.get((b, var))
        vid = self._read_rec(b, var) if vid is None else vid
        for c in chain:
            self.defs[(c, var)] = vid
        return self.find(vid)

    def _read_rec(self, b: int, var) -> int:
        if b == -1 or (b in self.sealed and not self._preds(b)):
            return self._entry(var)                     # function entry (or an unreachable block)
        vid = self.defs[(b, var)] = self.new("phi", [], var, block=b)
        if b not in self.sealed:
            self.incomplete.setdefault(b, {})[var] = vid
        else:
            vid = self.defs[(b, var)] = self._complete(vid)
        return vid

    def _complete(self, phi: int) -> int:
        v = self.values[phi]
        v.args = [self.read(v.attr, p) for p in self._preds(v.block)]
        return self._trivial(phi)

    def _trivial(self, phi: int) -> int:
        same = None
        for a in map(self.find, self.values[phi].args):
            if a != phi and a != same:
                if same is not None:
                    return phi
                same = a
        self.replaced[phi] = same if same is not None else self.new("undef", (), self.values[phi].attr, block=-1)
        return self.replaced[phi]

    def seal(self, b: int) -> None:
        self.sealed.add(b)
        for phi in self.incomplete.pop(b, {}).values():
            self._complete(phi)

    def run(self) -> "_Lifter":
        order = _postorder(0, lambda b: self.blocks[b].succs)[::-1]
        order += sorted({b.id for b in self.blocks} - set(order))
        filled = {-1}
        ready = lambda b: b not in self.sealed and all(p in filled for p in self._preds(b))
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 10000))          # phi completion recurses through merges
        try:
            for b in order:
                if ready(b):
                    self.seal(b)
                self._fill(self.blocks[b])
                filled.add(b)
                for s in filter(ready, self.blocks[b].succs):
                    self.seal(s)
            for b in self.blocks:
                if b.id not in self.sealed:
                    self.seal(b.id)
            changed = True
            while changed:                               # minimal SSA: drop phis made trivial late
                changed = False
                for v in self.values:
                    if v.op == "phi" and v.id not in self.replaced and self._trivial(v.id) != v.id:
                        changed = True
        finally:
            sys.setrecursionlimit(old)
        return self

    # ---- statements -------------------------------------------------------------------------
    def emit(self, kind, args=(), attr=None, targets=(), pc=0) -> None:
        self.cur.stmts.append(Stmt(kind, tuple(args), attr, tuple(targets), pc))

    def g(self, r: int) -> int:
        self.escaped |= r == SP                          # the stack pointer used as data
        return self.read(r)

    def setg(self, r: int, vid: int) -> None:
        self.escaped |= r == SP
        self.write(r, vid)

    def rdouble(self, var: int) -> int:
        lo, hi = self.read(var), self.read(var + 1)
        h, l = self.values[hi], self.values[lo]
        if (h.op == "hi" and self.find(h.args[0]) == lo) or h.op == l.op == "in":
            return lo
        return self.mk("pair", (hi, lo))

    def wdouble(self, var: int, vid: int) -> None:
        self.write(var, vid)
        self.write(var + 1, self.mk("hi", (vid,)))

    def unknown(self, i: int | None, reason: str = "unsupported instruction") -> None:
        word, pc = (self.words[i], self.vram + 4 * i) if i is not None else (None, self.vram + 4 * self.n)
        name, dest = ("none" if word is None else f"op{word >> 26}"), -1
        try:
            from .. import mips
            ins = mips.decode(word, pc)
            name, dest = ins.getOpcodeName(), mips.reg(ins.getDestinationGpr())
        except Exception:                                # no word, rabbitizer missing or undecodable
            pass
        self.emit("unknown", (), (word, name, reason), pc=pc)
        self.write("mem", self.new("mem", (), pc))
        if 0 < dest < 32:
            self.setg(dest, self.new("undef", (), (pc, dest)))

    def _fill(self, b: Block) -> None:
        self.cur, kind = b, b.plan[0]
        for i in b.body:
            self._insn(i)
        if kind in ("fall", "slot"):
            if kind == "slot":
                self._slot(b.plan[1])
            self.emit("goto", targets=(b.succs[0],))
        elif kind == "exit" and b.plan[1] != self.vram + 4 * self.n:
            self._call(self.k(b.plan[1]), b.plan[1], "tail")
        elif kind in ("exit", "end"):
            self.unknown(None, "falls off the end")
            self.emit("stop")
        else:
            self._transfer(b, b.plan[1])

    def _slot(self, j: int) -> None:
        if j >= self.n:
            self.unknown(None, "delay slot missing")
        elif self.ctl[j]:
            self.unknown(j, "control transfer in a delay slot")
        else:
            self._insn(j)

    def _transfer(self, b: Block, i: int) -> None:
        (kind, target, likely), pc, rs = self.ctl[i], self.vram + 4 * i, (self.words[i] >> 21) & 31
        if kind == "br":
            cond, swap = self._cond(self.words[i])       # read before the delay slot
            if not likely:
                self._slot(i + 1)
            t, c = (b.succs[::-1] if swap else b.succs), self.values[cond]
            if c.op == "const":
                self.emit("goto", targets=(t[0] if c.attr else t[1],), pc=pc)
            else:
                self.emit("branch", (cond,), targets=t, pc=pc)
            return
        tv = self.k(target) if kind in ("j", "jal") else self.g(rs) if kind in ("jalr", "jr") else None
        if kind == "bad":
            self.unknown(i, "conditional call")
            return self.emit("stop", pc=pc)
        self._slot(i + 1)
        if kind == "ret":
            self.emit("ret", (self.read(2), self.read(FP0)), pc=pc)
        elif kind == "jr":
            self.emit("ijump", (tv,), pc=pc)
        elif kind == "j" and not b.succs:
            self._call(tv, pc, "tail")
        else:
            if kind != "j":
                self._call(tv, pc, "call")
            self.emit("goto", targets=b.succs[:1], pc=pc) if b.succs else self.emit("stop", pc=pc)

    def _cond(self, w: int) -> tuple:
        """(condition, swapped): canonical eq/lt/ltu/c.* form; ``swapped`` exchanges the targets."""
        op, rs, rt, zero = w >> 26, (w >> 21) & 31, (w >> 16) & 31, self.k(0)
        if op in (4, 20, 5, 21):
            rel, a, b = ("eq" if op in (4, 20) else "ne"), self.g(rs), self.g(rt)
        elif op in (6, 22, 7, 23):
            rel, a, b = ("le" if op in (6, 22) else "gt"), self.g(rs), zero
        elif op == 1:
            rel, a, b = ("ge" if rt & 1 else "lt"), self.g(rs), zero
        else:
            rel, a, b = ("ne" if rt & 1 else "eq"), self.read("fcc"), zero
        swap = rel in ("ne", "ge", "le")
        a, b = (b, a) if rel in ("le", "gt") else (a, b)
        cond = self.mk("eq" if rel in ("eq", "ne") else "lt", (a, b))
        v = self.values[cond]
        if v.op == "eq" and self.values[v.args[1]].op == "const" and self.values[v.args[1]].attr == 0:
            inner = self.values[v.args[0]].op           # (bool == 0) is !bool
            if inner in ("lt", "ltu", "eq") or inner.startswith("c."):
                cond, swap = v.args[0], not swap
        return cond, swap

    def _call(self, target: int, pc: int, kind: str) -> None:
        self.emit(kind, (target,) + tuple(self.read(r) for r in ARGS), pc=pc)
        if kind != "tail":
            self.write("mem", self.new("mem", (), pc))
            for var in CALLER_SAVED:
                self.write(var, self.new("undef", (), (pc, var)))
            for var, which in ((2, "v0"), (3, "v1"), (FP0, "f0")):
                self.write(var, self.new("result", (target,), (pc, which)))
            self.write(FP0 + 1, self.mk("hi", (self.read(FP0),)))

    def _address(self, i: int, rs: int, simm: int, width) -> tuple:
        addr = self.mk("add", (self.read(rs), self.k(simm)))
        if rs == SP:
            self.sp_access.append((i, addr, width))
            if self.slots is not None:
                return ("slot", self.slots[i]), addr
        return None, addr

    def _insn(self, i: int) -> None:
        w = self.words[i]
        op, rs, rt, rd, sa, fn = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31, (w >> 6) & 31, w & 63
        imm, pc = w & 0xFFFF, self.vram + 4 * i
        simm = imm - 0x10000 if imm & 0x8000 else imm
        g, setg, mk, k = self.g, self.setg, self.mk, self.k
        if op == 0 and fn in _SHIFT:
            setg(rd, mk(_SHIFT[fn][0], (g(rt), k(sa + _SHIFT[fn][1]))))
        elif op == 0 and fn in _SHIFTV:
            setg(rd, mk(_SHIFTV[fn], (g(rt), g(rs))))
        elif op == 0 and fn in _ALU:
            setg(rd, mk(_ALU[fn], (g(rs), g(rt))))
        elif op == 0 and fn in _MULDIV:
            a, b = g(rs), g(rt)
            self.write("lo", mk(_MULDIV[fn][0], (a, b)))
            self.write("hi", mk(_MULDIV[fn][1], (a, b)))
        elif op == 0 and fn in (16, 18):
            setg(rd, self.read("hi" if fn == 16 else "lo"))
        elif op == 0 and fn in (17, 19):
            self.write("hi" if fn == 17 else "lo", g(rs))
        elif op == 0 and fn == 13:
            self.emit("trap", attr=(w >> 6) & 0xFFFFF, pc=pc)
        elif op == 9 and rs == rt == SP:                 # frame allocation and release
            self.write(SP, mk("add", (self.read(SP), k(simm))))
        elif op in _IMM:
            setg(rt, mk(_IMM[op], (g(rs), k(imm if 12 <= op <= 14 else simm))))
        elif op == 15:
            setg(rt, k(imm << 16))
        elif op in _LOADS:
            width, kind = _LOADS[op]
            slot, addr = self._address(i, rs, simm, width if kind in "suf" else None)
            if slot is not None:
                v = self.read(slot)
                v = mk("ext", (v,), (width, kind)) if kind in "su" and width < 4 else v
            else:                                        # lwl/lwr merge with the old register
                old = (g(rt),) if kind in "lr" else ()
                v = mk("load", (addr,) + old, (width, kind, self.read("mem")))
            if kind == "f":
                (self.wdouble if width == 8 else self.write)(FP0 + rt, v)
            else:
                setg(rt, v)
        elif op in _STORES:
            width, kind = _STORES[op]
            slot, addr = self._address(i, rs, simm, width if kind in ("", "f") else None)
            val = g(rt) if kind != "f" else self.rdouble(FP0 + rt) if width == 8 else self.read(FP0 + rt)
            if slot is not None:
                self.write(slot, val)
            else:
                self.emit("store", (addr, val), (width, kind), pc=pc)
                self.write("mem", self.new("mem", (), pc))
        elif op == 17:
            self._cop1(i, rs, rt, rd, sa, fn)
        else:
            self.unknown(i)

    def _cop1(self, i: int, fmt: int, ft: int, fs: int, fd: int, fn: int) -> None:
        mk = self.mk
        if fmt in (0, 1):
            return self.setg(ft, mk("mfc1", (self.read(FP0 + fs),)))
        if fmt in (4, 5):
            return self.write(FP0 + fs, mk("mtc1", (self.g(ft),)))
        if fmt == 2:
            return self.setg(ft, self.read("fcsr"))
        if fmt == 6:
            self.write("fcsr", self.g(ft))
            return self.emit("setfcsr", (self.g(ft),), pc=self.vram + 4 * i)
        f, dbl, arith = _FMT.get(fmt), fmt in (17, 21), fmt in (16, 17)
        src = lambda r: self.rdouble(FP0 + r) if dbl else self.read(FP0 + r)
        put = lambda v, d: (self.wdouble if d else self.write)(FP0 + fd, v)
        if arith and fn < 4:
            put(mk(f"{('add', 'sub', 'mul', 'div')[fn]}.{f}", (src(fs), src(ft))), dbl)
        elif arith and fn == 6:
            put(src(fs), dbl)
        elif arith and fn >= 48:
            self.write("fcc", mk(f"c.{_CONDS[fn - 48]}.{f}", (src(fs), src(ft))))
        elif f and fn in _FUN and (arith or fn in (32, 33)):
            name = _FUN[fn]
            put(mk(f"{name}.{f}", (src(fs),)), dbl if fn in (4, 5, 7) else name.endswith((".d", ".l")))
        else:
            self.unknown(i)

    def promotion(self, fn: Function) -> dict | None:
        """instruction index -> stack offset, when every stack access can be an SSA variable."""
        live = fn.reachable()
        if self.escaped or any(s.kind == "unknown" for b in self.blocks if b.id in live for s in b.stmts):
            return None
        slots, spans = {}, set()
        for i, addr, width in self.sp_access:
            off = _sp_offset(self.values, self.replaced, addr)
            if off is None or width is None or slots.setdefault(i, off) != off:
                return None
            spans.add((off, width))
        ordered = sorted(spans)
        if any(o2 < o1 + w1 for (o1, w1), (o2, _) in zip(ordered, ordered[1:])):
            return None                                  # overlapping slots of different widths
        return slots


def lift(words, vram: int, max_values: int | None = None) -> Function:
    """Lift one function given as instruction words (big-endian values) starting at ``vram``.

    ``max_values`` bounds the SSA values of each pass (``LiftBudgetExceeded`` beyond it); None,
    the default, sets no bound."""
    words = tuple(int(w) & M32 for w in words)
    if not words:
        raise ValueError("empty function")
    first = _Lifter(words, vram, max_values=max_values).run()
    fn = Function(vram, words, first.blocks, first.values, first.replaced, False)
    slots = first.promotion(fn)
    if slots is None:
        return fn
    del fn, first                                        # one pass in memory at a time
    second = _Lifter(words, vram, slots, max_values).run()
    return Function(vram, words, second.blocks, second.values, second.replaced, True)


def lift_bytes(code: bytes, vram: int) -> Function:
    return lift(struct.unpack(f">{len(code) // 4}I", code[:len(code) & ~3]), vram)


# ---- fingerprints ---------------------------------------------------------------------------
class _Shaper:
    """Register-allocation-independent text of a value: its depth-bounded expression tree."""

    def __init__(self, fn: Function, depth: int = 3, symbolic: bool = False):
        self.fn, self.depth, self.symbolic, self.memo = fn, depth, symbolic, {}

    def __call__(self, vid: int, depth: int | None = None) -> str:
        fn, d = self.fn, self.depth if depth is None else depth
        vid = fn.find(vid)
        if (vid, d) not in self.memo:
            v = fn.values[vid]
            off = _sp_offset(fn.values, fn.replaced, vid) if v.op in ("add", "in") else None
            if v.op == "const":
                s = _const_text(v.attr, self.symbolic)
            elif off is not None:                        # the frame layout is allocation
                s = "stack" if off < 0 else f"stack+{off:x}"
            elif v.op == "in":
                s = ENTRY.get(v.attr, "in") if not isinstance(v.attr, tuple) or v.attr[1] < 0 \
                    else f"stackarg+{v.attr[1]:x}"
            elif v.op in ("undef", "mem", "phi") or d <= 0:
                s = f"phi{len(v.args)}" if v.op == "phi" else _opname(v)
            else:
                kids = [self(a, d - 1) for a in v.args]
                s = f"{_opname(v)}({','.join(sorted(kids) if v.op in COMMUTATIVE else kids)})"
            self.memo[(vid, d)] = s
        return self.memo[(vid, d)]


def shape(fn: Function, vid: int, depth: int = 3, symbolic_addresses: bool = False) -> str:
    return _Shaper(fn, depth, symbolic_addresses)(vid)


def _closure(fn: Function, roots) -> set:
    seen, todo = set(), [fn.find(r) for r in roots]
    while todo:
        vid = todo.pop()
        if vid not in seen:
            seen.add(vid)
            if _sp_offset(fn.values, fn.replaced, vid) is None:      # stack addresses are leaves
                todo.extend(fn.find(a) for a in fn.values[vid].args)
    return seen


def live_values(fn: Function) -> set:
    """Values that reach a statement of a reachable block."""
    live = fn.reachable()
    return _closure(fn, [a for b in fn.blocks if b.id in live for s in b.stmts for a in s.args])


def value_numbers(fn: Function, depth: int = 8) -> dict:
    """Live value id -> number; equal across functions exactly when the value shapes are equal."""
    sh = _Shaper(fn, depth)
    return {vid: hashlib.sha1(sh(vid).encode()).hexdigest()[:12] for vid in sorted(live_values(fn))}


@dataclass
class Fingerprint:
    effects: Counter
    ops: Counter
    consts: Counter
    memory: Counter
    calls: Counter
    cfg: tuple
    unknown: int
    profile: dict          # allocation-dependent facts: in neither the digest nor the similarity
    digest: str = ""

    @property
    def complete(self) -> bool:
        return self.unknown == 0


def _skeleton(fn: Function) -> tuple:
    """Blocks in canonical reverse postorder as (terminator, successor numbers).  Blocks that
    only jump on are bypassed, so block splitting and empty fall-throughs do not count."""
    B = fn.blocks

    def fwd(b):
        seen = set()
        while len(B[b].stmts) == 1 and B[b].stmts[0].kind == "goto" and b not in seen:
            seen.add(b)
            b = B[b].stmts[0].targets[0]
        return b

    rpo = _postorder(fwd(0), lambda b: [fwd(t) for t in B[b].stmts[-1].targets])[::-1]
    num = {b: n for n, b in enumerate(rpo)}
    kind = lambda b: "call" if B[b].stmts[-1].kind == "goto" and any(s.kind == "call" for s in B[b].stmts) \
        else B[b].stmts[-1].kind
    return tuple((kind(b), tuple(num[fwd(t)] for t in B[b].stmts[-1].targets)) for b in rpo)


def fingerprint(fn: Function, depth: int = 3, symbolic_addresses: bool = False) -> Fingerprint:
    sh, V = _Shaper(fn, depth, symbolic_addresses), fn.values
    effects, memory, calls, roots, unknown, saves, live = Counter(), Counter(), Counter(), [], 0, 0, fn.reachable()
    for s in (s for b in fn.blocks if b.id in live for s in b.stmts):
        k, shapes = s.kind, [sh(a) for a in s.args]
        if k == "store":
            if _sp_offset(V, fn.replaced, s.args[0]) is not None and fn.value(s.args[1]).op == "in" \
                    and fn.value(s.args[1]).attr in SAVED:
                saves += 1                               # a callee-saved register saved on entry
                continue
            tag = f"store{8 * s.attr[0]}{s.attr[1]}"
            effects[f"{tag}({shapes[0]},{shapes[1]})"] += 1
            memory[tag] += 1
            roots += s.args
        elif k in ("call", "tail"):
            calls[shapes[0]] += 1
            effects[f"{k}({shapes[0]})"] += 1
            roots.append(s.args[0])
            for n, (a, text) in enumerate(zip(s.args[1:], shapes[1:])):
                if text not in OPAQUE:
                    effects[f"{k}.arg{n}({shapes[0]}):{text}"] += 1
                    roots.append(a)
        elif k == "ret":
            effects["ret"] += 1
            for name, a, text in zip(("v0", "f0"), s.args, shapes):
                if text not in OPAQUE:
                    effects[f"ret.{name}:{text}"] += 1
                    roots.append(a)
        elif k in ("branch", "ijump", "setfcsr"):
            effects[f"{k}({shapes[0]})"] += 1
            roots += s.args
        elif k in ("trap", "unknown"):
            effects[f"{k}({s.attr if k == 'trap' else s.attr[1]})"] += 1
            unknown += k == "unknown"
    ops, consts, values = Counter(), Counter(), _closure(fn, roots)
    for v in map(V.__getitem__, values):
        if v.op == "const":
            consts[_const_text(v.attr, symbolic_addresses)] += 1
        else:
            ops[sh(v.id, 0)] += 1
        if v.op == "load":
            memory[_opname(v)] += 1
    cfg = _skeleton(fn)
    frame = max((0x10000 - (w & 0xFFFF) for w in fn.words if w >> 16 == 0x27BD and w & 0x8000), default=0)
    profile = {"promoted": fn.promoted, "saves": saves, "frame": frame, "blocks": len(fn.blocks),
               "values": len(values), "instructions": len(fn.words)}
    payload = repr([sorted(c.items()) for c in (effects, ops, consts, memory, calls)] + [cfg])
    return Fingerprint(effects, ops, consts, memory, calls, cfg, unknown, profile,
                       hashlib.sha256(payload.encode()).hexdigest())


def fingerprint_bytes(code: bytes, vram: int, **options) -> Fingerprint:
    return fingerprint(lift_bytes(code, vram), **options)


WEIGHTS = {"effects": 0.4, "ops": 0.15, "consts": 0.15, "memory": 0.1, "calls": 0.1, "cfg": 0.1}


def _jaccard(a: Counter, b: Counter) -> float:
    union = sum((a | b).values())
    return 1.0 if not union else sum((a & b).values()) / union


def _cfg_similarity(a: tuple, b: tuple) -> float:
    if a == b:
        return 1.0
    nodes = lambda cfg: Counter((kind, len(t), any(x <= n for x in t)) for n, (kind, t) in enumerate(cfg))
    return 0.9 * _jaccard(nodes(a), nodes(b))


def compare(a: Fingerprint, b: Fingerprint) -> dict:
    """Weighted multiset similarity in [0, 1] and a verdict: ``same-semantics`` (equal digests),
    ``different-semantics``, or ``incomplete`` when either side holds an unknown instruction."""
    parts = {key: _jaccard(getattr(a, key), getattr(b, key)) for key in WEIGHTS if key != "cfg"}
    parts["cfg"] = _cfg_similarity(a.cfg, b.cfg)
    verdict = ("incomplete" if a.unknown or b.unknown else
               "same-semantics" if a.digest == b.digest else "different-semantics")
    return {"score": round(sum(WEIGHTS[key] * parts[key] for key in WEIGHTS), 6), "parts": parts,
            "verdict": verdict, "differing_effects": sorted(set(a.effects - b.effects) | set(b.effects - a.effects))}


def similarity(a: Fingerprint, b: Fingerprint) -> float:
    return compare(a, b)["score"]
