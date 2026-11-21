"""Bounded translation validation over the semantic IR (``n64dk.match.ir``).

``validate(words_a, vram_a, words_b, vram_b, limits)`` asks whether two machine-code functions
behave alike on every input, *within stated bounds and under stated assumptions*.  It complements
the behaviour oracle (``oracle.py``): the oracle tests concrete inputs, this module reasons over
symbolic ones.  It never replaces byte identity, never grants strict-exact credit and is not a
universal equivalence proof: an ``equivalent`` verdict always carries its assumption list.

Method (stdlib only, no SMT solver):

* both functions are lifted by ``ir.lift`` (delay slots and branch-likely slots already resolved,
  stack slots promoted), with a few extra statements recorded at lift time: ``ovf`` for trapping
  ``add/addi/sub/dadd/daddi/dsub``, ``ldchk`` for every memory load, and the extra registers
  read at ``ret``/``tail`` (``v1``, callee-saved ``s0-s8``, ``sp``, ``ra``);
* path-wise symbolic execution over fixed-width bit-vector terms.  Terms are normalised
  (constant folding, linear sums ``k1*x1 + ... + c`` in canonical order, so ``x*8 == x<<3`` and
  ``a-b == a+(-b)``; flattened sorted and/or/xor; masked shift amounts; canonical comparisons
  against constants).  Loops are unrolled to ``limits["unroll"]`` iterations per path;
* a path forks on branches and on every case split the model needs: misaligned access (a fault
  outcome), signed overflow of a trapping add/sub (a fault outcome), divisor zero (HI/LO become a
  fresh *undefined* symbol: MIPS leaves them unpredictable), and pointer aliasing between a load
  and an earlier store (same address / each partial overlap / disjoint, bounded);
* each path ends with (path condition, outputs, calls with the memory they see, stores outside
  the stack frame, fault).  Every pair of paths (A, B) whose joint condition is not refuted by
  the constraint store (equalities, disequalities, signed and unsigned intervals) is compared
  after rewriting both under that joint condition.  Differences are searched for a concrete
  witness (boundary values, seeded random values, and values solving the simple conditions on
  the path such as ``x == k``, ``x < k``, alignment, ``p == q``, overflow).

Outcomes: ``equivalent`` (every feasible pair agrees: under the listed assumptions), ``counterexample``
(a concrete witness with both behaviours), ``unsupported`` (outside the modelled subset: floating
point, ``lwl``/``lwr``, jump tables, MMIO or uncached accesses, stack addresses that escape,
unknown instructions), ``timeout`` (a budget ran out, or differences remain without a witness) and
``inconsistent-model`` (an internal contradiction; also when a function compared with itself is
not judged equivalent: that self-check runs before every comparison).

Differences that depend only on undefined values (HI/LO after a division by zero, registers
clobbered by a call, uninitialised stack) are *undefined-dependent*: never counterexamples,
reported as an assumption.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field

from . import ir

M32 = 0xFFFFFFFF
OUTCOMES = ("equivalent", "counterexample", "unsupported", "timeout", "inconsistent-model")
DEFAULT_LIMITS = {
    "unroll": 4,             # loop iterations followed on one path; longer executions are not checked
    "max_paths": 512,        # finished + pending paths of one function
    "max_forks": 48,         # case splits (not branches) on one path
    "max_steps": 200_000,    # statements executed over both explorations
    "max_pairs": 40_000,     # (path of A, path of B) pairs examined
    "samples": 64,           # concrete candidates tried per differing pair
    "seed": 0x5EED,
    "outputs": ("v0", "v1"),  # result registers compared at return ("f0" may be added)
    "callee_saved": True,    # s0-s7, fp, sp, ra must agree at return
    "call_args": 4,          # integer argument registers a call is assumed to read
    "noalias": False,        # True: unknown pointer pairs are assumed disjoint (recorded)
    "fault_domain": ("address-error-load", "address-error-store"),  # A faulting so: input outside domain
}
_SAVED = (("s0", 16), ("s1", 17), ("s2", 18), ("s3", 19), ("s4", 20), ("s5", 21), ("s6", 22),
          ("s7", 23), ("fp", 30), ("sp", ir.SP), ("ra", ir.RA))
_SAVED_NAMES = tuple(n for n, _ in _SAVED)
_EXTRA_RET = (3,) + tuple(r for _, r in _SAVED)          # appended to ret: v1, then the saved registers


class _Unsupported(Exception):
    pass


class _Inconsistent(Exception):
    pass


class _Budget(Exception):
    pass


class _Fork(Exception):
    def __init__(self, atom, kind):
        super().__init__(kind)
        self.atom, self.kind = atom, kind


# ---- terms ------------------------------------------------------------------------------------
# ("c", w, value) ("s", w, name) ("lin", w, const, ((term, coeff), ...)) ("mul", w, a, b)
# ("and"|"or"|"xor", w, (args...)) ("shl"|"lshr"|"ashr", w, x, amount) ("sext"|"zext"|"trunc", w, x)
# ("b2i", w, bool) ("eq"|"slt"|"ult"|"addo"|"subo", 1, a, b) ("not", 1, bool)
# ("sdiv"|"udiv"|"srem"|"urem"|"mulhs"|"mulhu", w, a, b) ("ld", w, base, address)
def _mask(w: int) -> int:
    return (1 << w) - 1


def C(w: int, v: int) -> tuple:
    return ("c", w, v & ((1 << w) - 1))


def S(name: str, w: int = 32) -> tuple:
    return ("s", w, name)


TRUE, FALSE = ("c", 1, 1), ("c", 1, 0)
_SP8 = S("sp/8")                                 # entry $sp = 8 * sp/8: the stack is 8-byte aligned
_SP = ("lin", 32, 0, ((_SP8, 8),))
_ATOMS = frozenset(("eq", "slt", "ult", "addo", "subo", "not"))
_UNK = "unknown"
_KEYS: dict = {}


def _signed(v: int, w: int) -> int:
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


def _key(t) -> str:
    k = _KEYS.get(t)
    if k is None:
        k = _KEYS[t] = repr(t)
    return k


def _lin(t) -> tuple:
    if t[0] == "c":
        return t[2], {}
    if t[0] == "lin":
        return t[2], dict(t[3])
    return 0, {t: 1}


def _mk_lin(w: int, c: int, d: dict) -> tuple:
    m = _mask(w)
    items = sorted(((x, k & m) for x, k in d.items() if k & m), key=lambda e: _key(e[0]))
    c &= m
    if not items:
        return ("c", w, c)
    if c == 0 and len(items) == 1 and items[0][1] == 1:
        return items[0][0]
    return ("lin", w, c, tuple(items))


def _add(w, a, b):
    ca, da = _lin(a)
    cb, db = _lin(b)
    for x, k in db.items():
        da[x] = da.get(x, 0) + k
    return _mk_lin(w, ca + cb, da)


def _scale(w, a, k):
    ca, da = _lin(a)
    return _mk_lin(w, ca * k, {x: v * k for x, v in da.items()})


def _sub(w, a, b):
    return _add(w, a, _scale(w, b, -1))


def _mul(w, a, b):
    if a[0] == "c":
        return _scale(w, b, a[2])
    if b[0] == "c":
        return _scale(w, a, b[2])
    a, b = sorted((a, b), key=_key)
    return ("mul", w, a, b)


def _low(t, j: int, w: int):
    """``t`` reduced modulo 2**j (the low j bits of a sum depend only on low bits)."""
    if t[0] != "lin":
        return t
    mod = 1 << j
    return _mk_lin(w, t[2] % mod, {x: k % mod for x, k in t[3]})


def _bit(op: str, w: int, args):
    m = _mask(w)
    flat = []
    for a in args:
        flat.extend(a[2] if a[0] == op else (a,))
    if op == "and":
        k = m
        for a in flat:
            if a[0] == "c":
                k &= a[2]
        if k and k != m and k & (k + 1) == 0:          # a low mask: (x + 4) & 3 == x & 3
            flat = [_low(a, k.bit_length(), w) for a in flat]
    k = m if op == "and" else 0
    rest = []
    for a in flat:
        if a[0] == "c":
            k = (k & a[2]) if op == "and" else (k | a[2]) if op == "or" else (k ^ a[2])
        else:
            rest.append(a)
    if op == "xor":
        n = Counter(rest)
        rest = [a for a in n if n[a] % 2]
    else:
        rest = list(dict.fromkeys(rest))
    if (op == "and" and k == 0) or (op == "or" and k == m) or not rest:
        return C(w, k)
    ident = m if op == "and" else 0
    if op == "and" and len(rest) == 1 and rest[0][0] == "b2i":
        return rest[0] if k & 1 else C(w, 0)
    if k == ident and len(rest) == 1:
        return rest[0]
    rest.sort(key=_key)
    return (op, w, tuple(rest) + ((C(w, k),) if k != ident else ()))


def _shift(op: str, w: int, x, s):
    """MIPS shift: the amount is taken modulo the width (``sllv`` reads 5 bits, ``dsllv`` 6)."""
    if s[0] == "c":
        k = s[2] % w
        if k == 0:
            return x
        if op == "shl":
            return _scale(w, x, 1 << k)
        if x[0] == "c":
            return C(w, x[2] >> k if op == "lshr" else _signed(x[2], w) >> k)
        return (op, w, x, C(32, k))
    s = _bit("and", s[1], [s, C(s[1], w - 1)])
    if s[0] == "c":
        return _shift(op, w, x, s)
    return (op, w, x, s)


def _b2i(w, b):
    return C(w, b[2]) if b[0] == "c" else ("b2i", w, b)


def _trunc(w, x):
    if x[1] == w:
        return x
    if x[1] < w:
        raise _Inconsistent(f"truncation of a {x[1]}-bit term to {w} bits")
    op = x[0]
    if op == "c":
        return C(w, x[2])
    if op == "lin":
        acc = {}
        for t, k in x[3]:
            tt = _trunc(w, t)
            acc[tt] = acc.get(tt, 0) + k
        return _mk_lin(w, x[2], acc)
    if op == "mul":
        return _mul(w, _trunc(w, x[2]), _trunc(w, x[3]))
    if op in ("and", "or", "xor"):
        return _bit(op, w, [_trunc(w, a) for a in x[2]])
    if op in ("sext", "zext"):
        inner = x[2]
        if inner[1] == w:
            return inner
        if inner[1] < w:
            return (_sext if op == "sext" else _zext)(w, inner)
        return _trunc(w, inner)
    if op == "trunc":
        return _trunc(w, x[2])
    if op == "b2i":
        return ("b2i", w, x[2])
    return ("trunc", w, x)


def _sext(w, x):
    if x[1] == w:
        return x
    if x[1] > w:
        return _trunc(w, x)
    if x[0] == "c":
        return C(w, _signed(x[2], x[1]))
    if x[0] == "sext":
        return _sext(w, x[2])
    if x[0] == "b2i":
        return ("b2i", w, x[2])
    if x[0] == "zext" and x[2][1] < x[1]:
        return _zext(w, x[2])
    return ("sext", w, x)


def _zext(w, x):
    if x[1] == w:
        return x
    if x[1] > w:
        return _trunc(w, x)
    if x[0] == "c":
        return C(w, x[2])
    if x[0] == "zext":
        return _zext(w, x[2])
    if x[0] == "b2i":
        return ("b2i", w, x[2])
    return ("zext", w, x)


def _fit(t, w):
    """``t`` as a ``w``-bit value: booleans become 0/1, narrower values are sign-extended (a
    32-bit result lives sign-extended in a 64-bit register), wider ones are truncated."""
    if t[0] == "M" or t[0] == "!":
        raise _Inconsistent("a memory state used as a value")
    if t[1] == w:
        return t
    if t[1] == 1:
        return _b2i(w, t)
    return _sext(w, t) if t[1] < w else _trunc(w, t)


def _not(b):
    if b[0] == "c":
        return C(1, 1 - b[2])
    if b[0] == "not":
        return b[2]
    return ("not", 1, b)


def _eq(a, b):
    w = a[1]
    if a == b:
        return TRUE
    d = _sub(w, a, b)
    if d[0] == "c":
        return C(1, int(d[2] == 0))
    c, items = _lin(d)
    rest, k, m = _mk_lin(w, 0, items), -c & _mask(w), _mask(w)
    if rest[0] == "lin" and rest[3][0][1] > m >> 1:        # leading coefficient negative: negate
        rest, k = _scale(w, rest, -1), -k & m
    if rest[0] == "b2i":
        return rest[2] if k == 1 else _not(rest[2]) if k == 0 else FALSE
    return ("eq", 1, rest, C(w, k))


def _slt(a, b):
    w = a[1]
    if a[0] == "c" and b[0] == "c":
        return C(1, int(_signed(a[2], w) < _signed(b[2], w)))
    if a == b:
        return FALSE
    if b[0] == "c":
        return FALSE if b[2] == 1 << (w - 1) else ("slt", 1, a, b)
    if a[0] == "c":                                       # c < x  ==  !(x < c+1)
        return FALSE if a[2] == (1 << (w - 1)) - 1 else _not(("slt", 1, b, C(w, a[2] + 1)))
    return ("slt", 1, a, b)


def _ult(a, b):
    w = a[1]
    if a[0] == "c" and b[0] == "c":
        return C(1, int(a[2] < b[2]))
    if a == b:
        return FALSE
    if b[0] == "c":
        if b[2] == 0:
            return FALSE
        return _eq(a, C(w, 0)) if b[2] == 1 else ("ult", 1, a, b)
    if a[0] == "c":
        return FALSE if a[2] == _mask(w) else _not(_ult(b, C(w, a[2] + 1)))
    return ("ult", 1, a, b)


def _ovf(op, w, a, b) -> bool:
    r = _signed(a, w) + _signed(b, w) if op == "addo" else _signed(a, w) - _signed(b, w)
    return not -(1 << (w - 1)) <= r < 1 << (w - 1)


def _addo(a, b):
    """Signed overflow of a + b (``add``/``addi`` trap)."""
    if a[0] == "c" and b[0] == "c":
        return C(1, int(_ovf("addo", a[1], a[2], b[2])))
    if (a[0] == "c" and a[2] == 0) or (b[0] == "c" and b[2] == 0):
        return FALSE
    a, b = sorted((a, b), key=_key)
    return ("addo", 1, a, b)


def _subo(a, b):
    if a[0] == "c" and b[0] == "c":
        return C(1, int(_ovf("subo", a[1], a[2], b[2])))
    if b[0] == "c" and b[2] == 0:
        return FALSE
    return ("subo", 1, a, b)


def _cdiv(op, w, a, b) -> int:
    """Concrete MIPS division (``b`` nonzero; ``0x80000000 / -1`` gives ``0x80000000``, no trap)."""
    if op in ("udiv", "urem"):
        return (a // b if op == "udiv" else a % b) & _mask(w)
    x, y = _signed(a, w), _signed(b, w)
    q = abs(x) // abs(y)
    q = q if (x < 0) == (y < 0) else -q
    return (q if op == "sdiv" else x - q * y) & _mask(w)


def _divop(op, w, a, b):
    if b[0] == "c" and b[2] != 0:
        if a[0] == "c":
            return C(w, _cdiv(op, w, a[2], b[2]))
        if b[2] == 1:
            return a if op in ("sdiv", "udiv") else C(w, 0)
    return (op, w, a, b)


def _mulh(op, w, a, b):
    if a[0] == "c" and b[0] == "c":
        p = _signed(a[2], w) * _signed(b[2], w) if op == "mulhs" else a[2] * b[2]
        return C(w, p >> w)
    a, b = sorted((a, b), key=_key)
    return (op, w, a, b)


_BUILD = {
    "mul": _mul, "sext": _sext, "zext": _zext, "trunc": _trunc, "b2i": _b2i,
    "eq": lambda w, a, b: _eq(a, b), "slt": lambda w, a, b: _slt(a, b), "ult": lambda w, a, b: _ult(a, b),
    "addo": lambda w, a, b: _addo(a, b), "subo": lambda w, a, b: _subo(a, b),
    **{op: (lambda o: lambda w, a, b: _divop(o, w, a, b))(op) for op in ("sdiv", "udiv", "srem", "urem")},
    **{op: (lambda o: lambda w, a, b: _mulh(o, w, a, b))(op) for op in ("mulhs", "mulhu")},
}


def _rebuild(t, rb):
    op, w = t[0], t[1]
    if op in ("c", "s"):
        return t
    if op == "lin":
        acc = C(w, t[2])
        for x, k in t[3]:
            acc = _add(w, acc, _scale(w, rb(x), k))
        return acc
    if op in ("and", "or", "xor"):
        return _bit(op, w, [rb(a) for a in t[2]])
    if op in ("shl", "lshr", "ashr"):
        return _shift(op, w, rb(t[2]), rb(t[3]))
    if op == "ld":
        return ("ld", w, t[2], rb(t[3]))
    if op == "not":
        return _not(rb(t[2]))
    build = _BUILD.get(op)
    if build is None:
        raise _Inconsistent(f"unknown term operator {op}")
    return build(w, *[rb(a) for a in t[2:]])


def _kids(t) -> tuple:
    op = t[0]
    if op in ("c", "s"):
        return ()
    if op == "lin":
        return tuple(x for x, _ in t[3])
    if op in ("and", "or", "xor"):
        return t[2]
    if op == "ld":
        return (t[3],)
    return t[2:]


def _any(t, pred) -> bool:
    seen, todo = set(), [t]
    while todo:
        x = todo.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if pred(x):
            return True
        todo.extend(_kids(x))
    return False


def _has_sp(t) -> bool:
    return _any(t, lambda x: x == _SP8)


def _has_undef(t) -> bool:
    return _any(t, lambda x: x[0] == "s" and x[2].startswith("undef."))


def _symbols(t, out: dict) -> None:
    todo = [t]
    while todo:
        x = todo.pop()
        if x[0] == "s":
            out[x[2]] = x[1]
        else:
            todo.extend(_kids(x))


def _show(t, depth: int = 8) -> str:
    op = t[0]
    if op == "c":
        return str(t[2]) if t[2] < 10 else hex(t[2])
    if op == "s":
        return "sp" if t == _SP8 else t[2]
    if depth <= 0:
        return "..."
    sh = lambda x: _show(x, depth - 1)
    if t == _SP:
        return "sp"
    if op == "lin":
        parts = [sh(x) if k == 1 else f"{_signed(k, t[1])}*{sh(x)}" for x, k in t[3]]
        return "(" + " + ".join(parts + ([hex(t[2])] if t[2] else [])) + ")"
    if op in ("and", "or", "xor"):
        return "(" + f" {op} ".join(sh(a) for a in t[2]) + ")"
    if op == "ld":
        return f"{t[2]}[{sh(t[3])}]:{t[1]}"
    return f"{op}{t[1]}(" + ", ".join(sh(a) for a in t[2:]) + ")"


# ---- constraint store over path conditions -------------------------------------------------------
def _strip(atom, pol: bool) -> tuple:
    while atom[0] == "not":
        atom, pol = atom[2], not pol
    return atom, pol


def _range(w: int, signed: bool) -> tuple:
    return (-(1 << (w - 1)), (1 << (w - 1)) - 1) if signed else (0, _mask(w))


class _Facts:
    """Equalities, disequalities and signed/unsigned intervals of terms, plus other atoms."""
    __slots__ = ("eqv", "ne", "sb", "ub", "gen")

    def __init__(self):
        self.eqv, self.ne, self.sb, self.ub, self.gen = {}, {}, {}, {}, {}

    def copy(self) -> "_Facts":
        f = _Facts.__new__(_Facts)
        f.eqv, f.sb, f.ub, f.gen = dict(self.eqv), dict(self.sb), dict(self.ub), dict(self.gen)
        f.ne = {k: set(v) for k, v in self.ne.items()}
        return f

    def add(self, atom, pol: bool) -> bool:
        """Record ``atom == pol``; False when that contradicts what is known."""
        atom, pol = _strip(atom, pol)
        op = atom[0]
        if op == "c":
            return bool(atom[2]) == pol
        if op == "eq" and atom[3][0] == "c":
            t, k = atom[2], atom[3][2]
            if pol:
                if self.eqv.get(t, k) != k:
                    return False
                self.eqv[t] = k
            else:
                if self.eqv.get(t) == k:
                    return False
                self.ne.setdefault(t, set()).add(k)
            return self._check(t)
        if op in ("slt", "ult") and atom[3][0] == "c":
            t, signed = atom[2], op == "slt"
            c = _signed(atom[3][2], t[1]) if signed else atom[3][2]
            table = self.sb if signed else self.ub
            lo, hi = table.get(t, _range(t[1], signed))
            table[t] = (lo, min(hi, c - 1)) if pol else (max(lo, c), hi)
            return self._check(t)
        prev = self.gen.get(atom)
        if prev is not None and prev != pol:
            return False
        self.gen[atom] = pol
        return True

    def _check(self, t) -> bool:
        w = t[1]
        m, ne = _mask(w), self.ne.get(t, ())
        if t in self.eqv and self.eqv[t] in ne:
            return False
        for signed, table in ((True, self.sb), (False, self.ub)):
            if t not in table:
                continue
            lo, hi = table[t]
            if lo > hi:
                return False
            if t in self.eqv and not lo <= (_signed(self.eqv[t], w) if signed else self.eqv[t]) <= hi:
                return False
            if ne and hi - lo < 64 and all((x & m) in ne for x in range(lo, hi + 1)):
                return False
        return True

    def decide(self, atom):
        """True/False when the atom follows from what is known, else None."""
        atom, pol = _strip(atom, True)
        r = self._decide(atom)
        return None if r is None else r == pol

    def _decide(self, atom):
        op = atom[0]
        if op == "c":
            return bool(atom[2])
        if op in ("eq", "slt", "ult") and atom[3][0] == "c":
            t, k = atom[2], atom[3][2]
            w = t[1]
            if t in self.eqv:
                v = self.eqv[t]
                return v == k if op == "eq" else _signed(v, w) < _signed(k, w) if op == "slt" else v < k
            if op == "eq":
                if k in self.ne.get(t, ()):
                    return False
                for signed, table in ((True, self.sb), (False, self.ub)):
                    if t in table:
                        lo, hi = table[t]
                        kv = _signed(k, w) if signed else k
                        if not lo <= kv <= hi:
                            return False
                        if lo == hi == kv:
                            return True
                return None
            signed = op == "slt"
            table = self.sb if signed else self.ub
            if t in table:
                lo, hi = table[t]
                kv = _signed(k, w) if signed else k
                if hi < kv:
                    return True
                if lo >= kv:
                    return False
            return None
        return self.gen.get(atom)


class _Rewriter:
    """Rewrites a term under a constraint store: decided atoms become constants, ``x == k``
    substitutes ``k`` for ``x`` and ``x - y == k`` substitutes ``y + k`` for ``x``."""

    def __init__(self, facts: _Facts):
        self.f, self.memo, self.sub, self.active = facts, {}, {}, set()
        for t, k in facts.eqv.items():
            if t[0] == "s":
                self.sub[t] = C(t[1], k)
            elif t[0] == "lin":
                (x, c1), others = t[3][0], t[3][1:]
                if x[0] == "s" and c1 == 1 and x not in self.sub and \
                        not any(_any(y, lambda z, x=x: z == x) for y, _ in others):
                    self.sub[x] = _sub(t[1], C(t[1], k), _mk_lin(t[1], 0, dict(others)))

    def __call__(self, t):
        r = self.memo.get(t)
        if r is not None:
            return r
        if t[0] == "c":
            return t
        if t[0] == "s":
            r, s = t, self.sub.get(t)
            if s is not None and t not in self.active:
                self.active.add(t)
                r = self(s)
                self.active.discard(t)
        else:
            r = _rebuild(t, self)
        r = self._fact(r)
        self.memo[t] = r
        return r

    def _fact(self, r):
        if r[0] == "c":
            return r
        if r[1] == 1 and r[0] in _ATOMS:
            d = self.f.decide(r)
            return r if d is None else C(1, int(d))
        v = self.f.eqv.get(r)
        if v is not None:
            return C(r[1], v)
        if r[0] == "lin":
            w = r[1]
            rest = _mk_lin(w, 0, dict(r[3]))
            v = self.f.eqv.get(rest)
            if v is not None:
                return C(w, v + r[2])
            v = self.f.eqv.get(_scale(w, rest, -1))
            if v is not None:
                return C(w, r[2] - v)
        return r


def _cands(wa: int, wq: int) -> list:
    """Offsets d = a - q at which aligned accesses of widths wa and wq overlap."""
    step = min(wa, wq)
    return [x for x in range(-wa + 1, wq) if x % step == 0]


def _rel(a, wa, q, wq, facts: _Facts) -> tuple:
    """(overlap offset a - q | None for disjoint | _UNK, first undecided atom)."""
    d = _sub(32, a, q)
    if d[0] == "c":
        dv = _signed(d[2], 32)
        return (dv if -wa < dv < wq else None), None
    first = None
    for x in _cands(wa, wq):
        atom = _eq(d, C(32, x))
        r = facts.decide(atom)
        if r:
            return x, None
        if r is None and first is None:
            first = atom
    return (None, None) if first is None else (_UNK, first)


def _canon(stores, facts: _Facts, rw, noalias: bool) -> tuple:
    """A store list as a memory state: wholly overwritten stores dropped; sorted when every pair
    is provably disjoint (so independent stores may come in any order)."""
    out = []
    for a, w, v in stores:
        a, v = rw(a), rw(v)
        keep = []
        for s in out:
            r, _ = _rel(s[0], s[1], a, w, facts)
            if not (isinstance(r, int) and r >= 0 and r + s[1] <= w):
                keep.append(s)
        out = keep + [(a, w, v)]

    def apart(x, y):
        r, _ = _rel(x[0], x[1], y[0], y[1], facts)
        return r is None or (r is _UNK and noalias)
    if all(apart(x, y) for i, x in enumerate(out) for y in out[i + 1:]):
        out.sort(key=lambda s: (_key(s[0]), s[1]))
    return tuple(out)


# ---- lifting: ir.lift plus the statements the validator needs -----------------------------------
class _TVLifter(ir._Lifter):
    def _insn(self, i: int) -> None:
        w = self.words[i]
        op, rs, rt, fn = w >> 26, (w >> 21) & 31, (w >> 16) & 31, w & 63
        imm, pc = w & 0xFFFF, self.vram + 4 * i
        simm = imm - 0x10000 if imm & 0x8000 else imm
        if op == 0 and fn in (32, 34, 44, 46):            # add sub dadd dsub: trap on overflow
            self.emit("ovf", (self.read(rs), self.read(rt)),
                      ("add" if fn in (32, 44) else "sub", 64 if fn >= 44 else 32), pc=pc)
        elif op in (8, 24) and rs != ir.SP:               # addi daddi ($sp arithmetic assumed not to overflow)
            self.emit("ovf", (self.read(rs), self.k(simm)), ("add", 64 if op == 24 else 32), pc=pc)
        elif op in ir._LOADS and not (rs == ir.SP and self.slots is not None):
            width, kind = ir._LOADS[op]
            addr = self.mk("add", (self.read(rs), self.k(simm)))
            self.emit("ldchk", (addr,), (width, kind), pc=pc)
        super()._insn(i)

    def _transfer(self, b, i: int) -> None:
        super()._transfer(b, i)
        s = self.cur.stmts[-1] if self.cur.stmts else None
        if s is not None and s.kind == "ret":
            s.args = tuple(s.args) + tuple(self.read(r) for r in _EXTRA_RET)

    def _call(self, target: int, pc: int, kind: str) -> None:
        super()._call(target, pc, kind)
        if kind == "tail":
            s = self.cur.stmts[-1]
            s.args = tuple(s.args) + tuple(self.read(r) for _, r in _SAVED)


def _lift(words, vram: int) -> ir.Function:
    words = tuple(int(w) & M32 for w in words)
    if not words:
        raise _Unsupported("empty function")
    first = _TVLifter(words, vram).run()
    fn = ir.Function(vram, words, first.blocks, first.values, first.replaced, False)
    slots = first.promotion(fn)
    if slots is None:
        return fn
    second = _TVLifter(words, vram, slots).run()
    return ir.Function(vram, words, second.blocks, second.values, second.replaced, True)


def _is_mmio(a: int) -> bool:
    a &= M32
    return 0xA0000000 <= a < 0xC0000000 or (0x80000000 <= a < 0xA0000000 and (a & 0x1FFFFFFF) >= 0x03F00000)


# ---- path-wise symbolic execution -------------------------------------------------------------
_DIV = {"div": ("sdiv", 32), "rem": ("srem", 32), "divu": ("udiv", 32), "remu": ("urem", 32),
        "ddiv": ("sdiv", 64), "drem": ("srem", 64), "ddivu": ("udiv", 64), "dremu": ("urem", 64)}
_SH = {"shl": "shl", "shr": "lshr", "sar": "ashr", "dshl": "shl", "dshr": "lshr", "dsar": "ashr"}
_END, _MOVED, _TRUNC = "end", "moved", "truncated"


class _Path:
    def __init__(self):
        self.block, self.idx = -1, 0
        self.env, self.memo, self.facts, self.conds = {}, {}, _Facts(), []
        self.events, self.mem, self.visits, self.callmap = [], ("M", 0, ()), {}, {}
        self.ncalls, self.forks, self.divzero = 0, 0, False

    def copy(self) -> "_Path":
        q = _Path.__new__(_Path)
        q.block, q.idx, q.mem, q.ncalls, q.forks, q.divzero = \
            self.block, self.idx, self.mem, self.ncalls, self.forks, self.divzero
        q.env, q.memo, q.facts, q.conds = dict(self.env), dict(self.memo), self.facts.copy(), list(self.conds)
        q.events, q.visits, q.callmap = list(self.events), dict(self.visits), dict(self.callmap)
        return q


class _Rec:
    """One finished path."""
    __slots__ = ("conds", "facts", "outputs", "events", "stores", "base", "fault", "truncated", "divzero")


class _Explorer:
    def __init__(self, fn: ir.Function, side: str, limits: dict, budget: dict):
        self.fn, self.side, self.L, self.budget = fn, side, limits, budget
        self.values, self.blocks = fn.values, fn.blocks
        self.memv = {(v.block, v.attr): v.id for v in fn.values if v.op == "mem" and v.block >= 0}
        self.phis = {b.id: [vid for vid in b.values if fn.values[vid].op == "phi" and vid not in fn.replaced]
                     for b in fn.blocks}
        self.done, self.notes, self.splits = [], set(), Counter()

    # -- driver --
    def explore(self) -> list:
        stack = [_Path()]
        while stack:
            if len(stack) + len(self.done) > self.L["max_paths"]:
                raise _Budget(f"more than {self.L['max_paths']} paths in function {self.side}")
            p = stack.pop()
            try:
                stack.extend(self._run(p))
            except _Fork as e:
                p.forks += 1
                self.splits[e.kind] += 1
                if p.forks > self.L["max_forks"]:
                    raise _Budget(f"more than {self.L['max_forks']} case splits on one path of {self.side}")
                for pol in (False, True):
                    q = p.copy()
                    if q.facts.add(e.atom, pol):
                        q.conds.append((e.atom, pol))
                        stack.append(q)
        return self.done

    def _run(self, p: _Path) -> list:
        if p.block < 0:
            if self._goto(p, 0) is _TRUNC:
                raise _Inconsistent("entry block refused")
        while True:
            self.budget["steps"] += 1
            if self.budget["steps"] > self.L["max_steps"]:
                raise _Budget(f"more than {self.L['max_steps']} statements executed")
            blk = self.blocks[p.block]
            if p.idx >= len(blk.stmts):
                raise _Inconsistent(f"block B{blk.id} has no terminator")
            r = self._stmt(p, blk, blk.stmts[p.idx])
            if r is None:
                p.idx += 1
            elif r is _END:
                return []
            elif r is not _MOVED:
                return r

    def _goto(self, p: _Path, t: int):
        n = p.visits.get(t, 0) + 1
        if n > self.L["unroll"] + 1:
            return _TRUNC
        preds = ([-1] + self.blocks[0].preds) if t == 0 else self.blocks[t].preds
        if p.block not in preds:
            raise _Inconsistent(f"edge B{p.block} -> B{t} is not in the predecessor list")
        k = preds.index(p.block)
        new = {}
        for vid in self.phis[t]:
            try:
                new[vid] = self.val(p, self.values[vid].args[k])
            except _Unsupported as e:               # an unused FP phi must not stop the path
                new[vid] = ("!", str(e))
        p.visits[t] = n
        p.env.update(new)
        if new or n > 1:
            p.memo.clear()
        p.block, p.idx = t, 0
        return _MOVED

    def _finish(self, p: _Path, outputs=None, fault=None, truncated=False):
        r = _Rec()
        r.conds, r.facts, r.outputs, r.events = tuple(p.conds), p.facts, outputs or {}, tuple(p.events)
        r.stores = tuple((a, w, v) for a, w, v, frame in p.mem[2] if not frame)
        r.base, r.fault, r.truncated, r.divzero = p.mem[1], fault, truncated, p.divzero
        self.done.append(r)
        return _END

    def _decide(self, p: _Path, atom, kind: str) -> bool:
        d = p.facts.decide(atom)
        if d is None:
            raise _Fork(atom, kind)
        return d

    # -- statements --
    def _stmt(self, p: _Path, blk, s):
        k = s.kind
        if k == "ldchk":
            width, kind = s.attr
            self._kind_ok(kind, width)
            addr = _fit(self.val(p, s.args[0]), 32)
            self._mmio(addr, s.pc)
            return None if self._aligned(p, addr, width) else self._finish(p, fault=("address-error-load",))
        if k == "store":
            width, kind = s.attr
            self._kind_ok(kind, width)
            addr = _fit(self.val(p, s.args[0]), 32)
            self._mmio(addr, s.pc)
            if not self._aligned(p, addr, width):
                return self._finish(p, fault=("address-error-store",))
            v = _fit(self.val(p, s.args[1]), 8 * width)
            frame = _has_sp(addr)
            if not frame and _has_sp(v):
                raise _Unsupported(f"a stack address is stored to memory at 0x{s.pc:08X}")
            mv = self.memv.get((blk.id, s.pc))
            if mv is None:
                raise _Inconsistent(f"no memory version for the store at 0x{s.pc:08X}")
            p.mem = ("M", p.mem[1], p.mem[2] + ((addr, width, v, frame),))
            p.env[mv] = p.mem
            return None
        if k == "ovf":
            op, w = s.attr
            a, b = (_fit(self.val(p, x), w) for x in s.args)
            atom = _addo(a, b) if op == "add" else _subo(a, b)
            return self._finish(p, fault=("integer-overflow",)) if self._decide(p, atom, "overflow") else None
        if k in ("call", "tail"):
            target, args = self._call_terms(p, s)      # everything that may fork comes before any change
            n, outputs = p.ncalls, None
            if k == "tail":
                regs = dict(zip(_SAVED_NAMES, s.args[7:]))
                fixed = {"v0": S(f"call{n}.v0"), "v1": S(f"call{n}.v1"), "f0": S(f"call{n}.f0")}
                outputs = self._outputs(p, regs, fixed)
            p.events.append((target, args, tuple((a, w, v) for a, w, v, fr in p.mem[2] if not fr), p.mem[1]))
            p.ncalls += 1
            p.mem = ("M", n + 1, tuple(x for x in p.mem[2] if x[3]))   # a callee may change all but our frame
            if k == "tail":
                return self._finish(p, outputs)
            p.callmap[s.pc] = n
            mv = self.memv.get((blk.id, s.pc))
            if mv is None:
                raise _Inconsistent(f"no memory version for the call at 0x{s.pc:08X}")
            p.env[mv] = p.mem
            return None
        if k == "ret":
            regs = dict(zip(("v0", "f0", "v1") + _SAVED_NAMES, s.args))
            return self._finish(p, self._outputs(p, regs))
        if k == "branch":
            cond = self._bool(self.val(p, s.args[0]))
            d = p.facts.decide(cond)
            if d is not None:
                return self._jump(p, s.targets[0] if d else s.targets[1])
            out, cut = [], []
            for pol, t in ((True, s.targets[0]), (False, s.targets[1])):
                q = p.copy()
                if not q.facts.add(cond, pol):
                    continue
                q.conds.append((cond, pol))
                (cut if self._goto(q, t) is _TRUNC else out).append(q)
            for q in cut:                                # recorded only once both sides are settled
                self._finish(q, truncated=True)
            return out
        if k == "goto":
            return self._jump(p, s.targets[0])
        if k == "trap":
            return self._finish(p, fault=("break", s.attr))
        if k == "stop":
            return self._finish(p, fault=("no-return",))
        if k == "ijump":
            raise _Unsupported(f"indirect jump at 0x{s.pc:08X} (jump table or computed goto)")
        if k == "unknown":
            raise _Unsupported(f"{s.attr[1]} at 0x{s.pc:08X}: {s.attr[2]}")
        if k == "setfcsr":
            raise _Unsupported(f"FPU control register write at 0x{s.pc:08X}")
        raise _Inconsistent(f"statement kind {k}")

    def _jump(self, p: _Path, t: int):
        return self._finish(p, truncated=True) if self._goto(p, t) is _TRUNC else _MOVED

    def _kind_ok(self, kind: str, width: int) -> None:
        if kind in ("l", "r"):
            raise _Unsupported("unaligned access (lwl/lwr/swl/swr/ldl/ldr/sdl/sdr)")
        if kind == "f" and width == 8:
            raise _Unsupported("FPU doubleword access")

    def _call_terms(self, p: _Path, s) -> tuple:
        target = _fit(self.val(p, s.args[0]), 32)
        if target == C(32, self.fn.vram):
            target = S("<self>")                        # recursion: position independent
        args = tuple(_fit(self.val(p, a), 32) for a in s.args[1:1 + self.L["call_args"]])
        if _has_sp(target) or any(_has_sp(a) for a in args):
            raise _Unsupported(f"a stack address is passed to the call at 0x{s.pc:08X}")
        return target, args

    def _outputs(self, p: _Path, regs: dict, fixed: dict | None = None) -> dict:
        out = {}
        for name in self.L["outputs"]:
            t = fixed[name] if fixed and name in fixed else self.val(p, regs[name])
            out[name] = _fit(t, 64)
        if self.L["callee_saved"]:
            for name in _SAVED_NAMES:
                out[name] = _fit(self.val(p, regs[name]), 64)
        for name, t in out.items():
            if name != "sp" and _has_sp(t):
                raise _Unsupported(f"a stack address reaches output {name}")
        return out

    def _bool(self, t):
        return t[2] if t[0] == "b2i" else _not(_eq(t, C(t[1], 0)))

    def _mmio(self, addr, pc: int) -> None:
        if _has_sp(addr):
            return
        c = addr[2] if addr[0] in ("c", "lin") else None
        if c is not None and _is_mmio(c):
            raise _Unsupported(f"volatile/MMIO access at 0x{pc:08X} (address {_show(addr)}): never reordered or merged")

    def _aligned(self, p: _Path, addr, width: int) -> bool:
        if width == 1:
            return True
        t = _bit("and", 32, [addr, C(32, width - 1)])
        if t[0] == "c":
            return t[2] == 0
        return self._decide(p, _eq(t, C(32, 0)), "align")

    # -- values --
    def val(self, p: _Path, vid: int):
        vid = self.fn.find(vid)
        t = p.env.get(vid)
        if t is not None:
            if t[0] == "!":
                raise _Unsupported(t[1])
            return t
        t = p.memo.get(vid)
        if t is None:
            t = p.memo[vid] = self._value(p, self.values[vid])
        return t

    def _entry(self, var):
        if var == "mem":
            return ("M", 0, ())
        if var == ir.SP:
            return _SP
        if isinstance(var, int):
            return S(ir.ENTRY.get(var) or f"{ir._varname(var)}.in")
        if isinstance(var, tuple):                       # a promoted stack slot never written here
            off = var[1]
            return S(f"stack{off:+#x}") if off >= 0 else S(f"undef.{self.side}.frame{off:+#x}")
        return S(f"{var}.in")

    def _value(self, p: _Path, v):
        op = v.op
        if op == "const":
            return C(64 if v.attr > M32 else 32, v.attr)
        if op == "in":
            return self._entry(v.attr)
        if op == "undef":
            return S(f"undef.{self.side}.v{v.id}")
        if op in ("phi", "mem"):
            raise _Inconsistent(f"{op} v{v.id} read before its block was executed")
        if op == "result":
            n = p.callmap.get(v.attr[0])
            if n is None:
                raise _Inconsistent(f"call result v{v.id} read before its call")
            return S(f"call{n}.{v.attr[1]}")
        if op == "load":
            return self._load(p, v)
        A = [self.val(p, a) for a in v.args]
        if op in ("add", "sub", "dadd", "dsub"):
            w = 64 if op[0] == "d" else 32
            a, b = _fit(A[0], w), _fit(A[1], w)
            return _add(w, a, b) if op.endswith("add") else _sub(w, a, b)
        if op in ("and", "or", "xor", "nor"):
            w = max(32, A[0][1], A[1][1])
            r = _bit("or" if op == "nor" else op, w, [_fit(A[0], w), _fit(A[1], w)])
            return _bit("xor", w, [r, C(w, _mask(w))]) if op == "nor" else r
        if op in _SH:
            w = 64 if op[0] == "d" else 32
            return _shift(_SH[op], w, _fit(A[0], w), A[1])
        if op in ("lt", "ltu", "eq"):
            w = max(32, A[0][1], A[1][1])
            a, b = _fit(A[0], w), _fit(A[1], w)
            return _b2i(32, (_slt if op == "lt" else _ult if op == "ltu" else _eq)(a, b))
        if op in ("mul", "mulu", "dmul", "dmulu"):
            w = 64 if op[0] == "d" else 32
            return _mul(w, _fit(A[0], w), _fit(A[1], w))
        if op in ("mulhi", "mulhiu", "dmulhi", "dmulhiu"):
            w = 64 if op[0] == "d" else 32
            return _mulh("mulhu" if op.endswith("u") else "mulhs", w, _fit(A[0], w), _fit(A[1], w))
        if op in _DIV:
            kind, w = _DIV[op]
            a, b = _fit(A[0], w), _fit(A[1], w)
            if self._decide(p, _eq(b, C(w, 0)), "div"):  # MIPS leaves HI/LO unpredictable
                p.divzero = True
                tag = hashlib.blake2b(repr((a, b)).encode(), digest_size=4).hexdigest()
                return S(f"undef.{self.side}.{op}{v.id}.{tag}", w)
            return _divop(kind, w, a, b)
        if op == "ext":
            width, kind = v.attr
            x = _trunc(8 * width, _fit(A[0], 32))
            return _sext(32, x) if kind == "s" else _zext(32, x)
        if op in ("mfc1", "mtc1"):
            return A[0]
        raise _Unsupported(f"IR operation '{op}' (floating point or not modelled)")

    def _load(self, p: _Path, v):
        width, kind, mv = v.attr
        self._kind_ok(kind, width)
        addr = _fit(self.val(p, v.args[0]), 32)
        snap = self.val(p, mv)
        if snap[0] != "M":
            raise _Inconsistent(f"load v{v.id} reads a non-memory value")
        raw = self._read(p, addr, width, snap)
        if width == 8:
            return raw
        if width == 4:
            return _zext(64, raw) if kind == "u" else raw
        return _sext(32, raw) if kind == "s" else _zext(32, raw)

    def _relation(self, p: _Path, a, wa: int, q, wq: int, q_frame: bool):
        """Offset of load ``a`` within store ``q`` when they overlap, None when disjoint; forks
        when unknown (or, with ``noalias``, assumes disjoint and records it)."""
        if _has_sp(a) != q_frame:
            return None                                  # the frame is private
        r, atom = _rel(a, wa, q, wq, p.facts)
        if r is not _UNK:
            return r
        if self.L["noalias"] and not q_frame:
            for x in _cands(wa, wq):
                cand = _eq(_sub(32, a, q), C(32, x))
                if p.facts.decide(cand) is None:
                    p.facts.add(cand, False)
                    p.conds.append((cand, False))
            self.notes.add(f"no-alias assumed: {_show(a)} vs {_show(q)}")
            return None
        raise _Fork(atom, "alias")

    def _read(self, p: _Path, a, w: int, snap):
        srcs, left = [None] * w, w
        for q, wq, val, frame in reversed(snap[2]):
            d = self._relation(p, a, w, q, wq, frame)
            if d is None:
                continue
            for i in range(w):
                if srcs[i] is None and 0 <= d + i < wq:
                    srcs[i] = (val, wq, d + i)
                    left -= 1
            if not left:
                break
        frame = _has_sp(a)
        if left == w:
            return self._base(a, w, snap[1], frame)
        first = srcs[0]
        if not left and all(s[0] == first[0] and s[2] == first[2] + i for i, s in enumerate(srcs)):
            val, wq, j = first
            return _trunc(8 * w, _shift("lshr", 8 * wq, val, C(32, 8 * (wq - j - w))))
        acc = C(8 * w, 0)
        for i, s in enumerate(srcs):
            if s is None:
                byte = self._base(_add(32, a, C(32, i)), 1, snap[1], frame)
            else:
                val, wq, j = s
                byte = _trunc(8, _shift("lshr", 8 * wq, val, C(32, 8 * (wq - 1 - j))))
            acc = _bit("or", 8 * w, [acc, _shift("shl", 8 * w, _zext(8 * w, byte), C(32, 8 * (w - 1 - i)))])
        return acc

    def _base(self, a, w: int, base: int, frame: bool):
        """Memory not written on this path: inputs, or what the last call left."""
        if frame:
            off = _sub(32, a, _SP)
            if off[0] == "c":
                o = _signed(off[2], 32)
                if o >= 0:                               # incoming stack arguments: as promoted slots
                    return _trunc(8 * w, S(f"stack{o:+#x}", 32 if w <= 4 else 64))
                return S(f"undef.{self.side}.frame{o:+#x}.{w}", 8 * w)
            return ("ld", 8 * w, "M0", a)                # callees never touch our frame
        return ("ld", 8 * w, f"M{base}", a)


# ---- concrete evaluation (witness search) -----------------------------------------------------
def _hash(*parts) -> int:
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), "big")


class _Concrete:
    def __init__(self, asg: dict, mem: dict, seed: int):
        self.asg, self.mem, self.seed, self.memo = asg, mem, seed, {}

    def byte(self, base: str, addr: int) -> int:
        addr &= M32
        v = self.mem.get((base, addr))
        return _hash(self.seed, base, addr) & 0xFF if v is None else v

    def ev(self, t) -> int:
        r = self.memo.get(t)
        if r is None:
            r = self.memo[t] = self._ev(t)
        return r

    def _ev(self, t) -> int:
        op, w = t[0], t[1]
        m, ev = _mask(w), self.ev
        if op == "c":
            return t[2]
        if op == "s":
            v = self.asg.get(t[2])
            return (_hash(self.seed, "s", t[2]) if v is None else v) & m
        if op == "lin":
            return (t[2] + sum(k * ev(x) for x, k in t[3])) & m
        if op == "mul":
            return (ev(t[2]) * ev(t[3])) & m
        if op in ("and", "or", "xor"):
            r = m if op == "and" else 0
            for a in t[2]:
                r = (r & ev(a)) if op == "and" else (r | ev(a)) if op == "or" else (r ^ ev(a))
            return r
        if op == "shl":
            return (ev(t[2]) << (ev(t[3]) % w)) & m
        if op == "lshr":
            return ev(t[2]) >> (ev(t[3]) % w)
        if op == "ashr":
            return (_signed(ev(t[2]), w) >> (ev(t[3]) % w)) & m
        if op == "sext":
            return _signed(ev(t[2]), t[2][1]) & m
        if op in ("zext", "b2i"):
            return ev(t[2])
        if op == "trunc":
            return ev(t[2]) & m
        if op == "not":
            return 1 - ev(t[2])
        if op in ("eq", "slt", "ult", "addo", "subo"):
            a, b, wa = ev(t[2]), ev(t[3]), t[2][1]
            if op == "eq":
                return int(a == b)
            if op == "slt":
                return int(_signed(a, wa) < _signed(b, wa))
            if op == "ult":
                return int(a < b)
            return int(_ovf(op, wa, a, b))
        if op in ("sdiv", "udiv", "srem", "urem"):
            b = ev(t[3])
            return _cdiv(op, w, ev(t[2]), b) if b else 0
        if op in ("mulhs", "mulhu"):
            a, b = ev(t[2]), ev(t[3])
            return ((_signed(a, w) * _signed(b, w) if op == "mulhs" else a * b) >> w) & m
        if op == "ld":
            a, v = ev(t[3]), 0
            for i in range(w // 8):
                v = (v << 8) | self.byte(t[2], a + i)
            return v
        raise _Inconsistent(f"cannot evaluate term {op}")


def _assign(cx: _Concrete, t, target: int) -> bool:
    """Change an input so that ``t`` evaluates to ``target`` (linear and simple cases only)."""
    w = t[1]
    m = _mask(w)
    target &= m
    op = t[0]
    if op == "s":
        cx.asg[t[2]] = target
        cx.memo.clear()
        return True
    if op == "lin":
        for x, k in sorted(t[3], key=lambda e: e[0][0] != "s"):
            if k & 1:
                rest = (cx.ev(t) - k * cx.ev(x)) & m
                if _assign(cx, x, ((target - rest) * pow(k, -1, 1 << w)) & m):
                    return True
        return False
    if op in ("sext", "zext"):
        return _assign(cx, t[2], target & _mask(t[2][1]))
    if op == "trunc":
        return _assign(cx, t[2], (cx.ev(t[2]) & ~m) | target)
    if op == "and" and len(t[2]) == 2 and t[2][1][0] == "c":
        x, k = t[2][0], t[2][1][2]
        return _assign(cx, x, (cx.ev(x) & ~k) | (target & k))
    if op == "ld":
        a, n = cx.ev(t[3]), w // 8
        for i in range(n):
            cx.mem[(t[2], (a + i) & M32)] = (target >> (8 * (n - 1 - i))) & 0xFF
        cx.memo.clear()
        return True
    return False


def _fix(cx: _Concrete, atom, pol: bool) -> None:
    atom, pol = _strip(atom, pol)
    op = atom[0]
    if op == "eq":
        k = cx.ev(atom[3])
        _assign(cx, atom[2], k if pol else k + 1)
    elif op in ("slt", "ult"):
        a, b = atom[2], atom[3]
        w, bv = a[1], cx.ev(b)
        if op == "slt":
            bs = _signed(bv, w)
            target = (bs - 1 if bs > -(1 << (w - 1)) else None) if pol else bs
        else:
            target = (bv - 1 if bv > 0 else None) if pol else bv
        if (target is None or not _assign(cx, a, target)) and b[0] != "c":
            av = cx.ev(a)
            _assign(cx, b, av + 1 if pol else av)
    elif op in ("addo", "subo"):
        a, b = atom[2], atom[3]
        w = a[1]
        smax, smin = (1 << (w - 1)) - 1, 1 << (w - 1)
        if not pol:
            _assign(cx, a, 0 if op == "addo" else cx.ev(b)) or _assign(cx, b, 0)
            return
        if a[0] == "c":
            a, b = b, a
        if _signed(cx.ev(b), w) == 0:
            _assign(cx, b, 1)
        positive = _signed(cx.ev(b), w) > 0
        _assign(cx, a, (smax if positive else smin) if op == "addo" else (smin if positive else smax))
    cx.memo.clear()


_BOUNDARY = (0, 1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 63, 64, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF,
             0x10000, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF, 0x80100000)


# ---- results ----------------------------------------------------------------------------------
@dataclass
class Result:
    outcome: str                                  # one of OUTCOMES
    assumptions: list = field(default_factory=list)
    counterexample: dict | None = None
    paths: dict = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict:
        return {"outcome": self.outcome, "assumptions": list(self.assumptions),
                "counterexample": self.counterexample, "paths": dict(self.paths), "reason": self.reason,
                "proof": False}


_BASE_ASSUMPTIONS = (
    "bounded translation validation over the n64dk semantic IR: evidence only; never byte identity, "
    "never strict-exact credit, not a universal equivalence proof",
    "integer register inputs are 32-bit values sign-extended into 64-bit registers (o32)",
    "the stack frame is private: it aliases no pointer input or global, callees do not touch it, "
    "and $sp is 8-byte aligned",
    "calls are summarised: equal target, arguments and visible memory give equal results; a call "
    "may change any memory outside the caller's frame",
    "symbolic pointers do not address MMIO or uncached memory (constant KSEG1/MMIO addresses are unsupported)",
    "floating-point state, cop0 and cache effects are outside the model",
)


class _Validator:
    def __init__(self, limits: dict):
        self.L = limits
        self.budget = {"steps": 0}

    def run(self, words_a, vram_a, words_b, vram_b) -> Result:
        paths: dict = {}
        try:
            fa, fb = _lift(words_a, vram_a), _lift(words_b, vram_b)
        except _Unsupported as e:
            return Result("unsupported", reason=str(e))
        except Exception as e:                             # the lifter itself failed
            return Result("inconsistent-model", reason=f"lifting failed: {e!r}")
        try:
            xa, xb = _Explorer(fa, "A", self.L, self.budget), _Explorer(fb, "B", self.L, self.budget)
            for x, name in ((xa, "A"), (xb, "B")):
                try:
                    x.explore()
                except _Unsupported as e:
                    return Result("unsupported", reason=f"function {name}: {e}")
            ra, rb = xa.done, xb.done
            paths = {"a": len(ra), "b": len(rb), "a_truncated": sum(r.truncated for r in ra),
                     "b_truncated": sum(r.truncated for r in rb),
                     "case_splits": dict(xa.splits + xb.splits)}
            for recs, name in ((ra, "A"), (rb, "B")):
                bad = self._selfcheck(recs)
                if bad:
                    return Result("inconsistent-model", paths=paths, reason=f"self-check of function {name}: {bad}")
            if not any(not r.truncated for r in ra) or not any(not r.truncated for r in rb):
                return Result("timeout", paths=paths,
                              reason=f"no path finishes within {self.L['unroll']} loop iterations")
            stats, witness, unresolved = self._compare(ra, rb)
        except _Budget as e:
            return Result("timeout", paths=paths, reason=f"budget exhausted: {e}")
        except _Inconsistent as e:
            return Result("inconsistent-model", paths=paths, reason=str(e))
        except RecursionError:
            return Result("timeout", paths=paths, reason="terms nest deeper than the recursion budget")
        paths.update(stats)
        assumptions = self._assumptions(paths, xa.notes | xb.notes)
        if witness:
            return Result("counterexample", assumptions, witness, paths,
                          f"concrete witness: {', '.join(witness['differs'])} differ")
        if unresolved:
            return Result("timeout", assumptions, None, paths,
                          f"{len(unresolved)} feasible path pairs differ symbolically and no witness was found "
                          f"in {self.L['samples']} samples each (first: {unresolved[0]}); neither proof nor refutation")
        if paths.get("a_truncated") or paths.get("b_truncated"):
            # executions beyond the unroll bound were never compared: agreement on the rest is no
            # equivalence (round 12: 5 oracle-confirmed differences hid there)
            return Result("timeout", assumptions, None, paths,
                          f"the loop bound ({self.L['unroll']} iterations) cut {paths.get('a_truncated', 0)} paths of A "
                          f"and {paths.get('b_truncated', 0)} of B; every compared pair agrees, the rest is unchecked: "
                          f"neither proof nor refutation")
        return Result("equivalent", assumptions, None, paths,
                      "every feasible path pair agrees under the stated assumptions")

    def _assumptions(self, paths: dict, notes: set) -> list:
        out = list(_BASE_ASSUMPTIONS)
        saved = ", callee-saved s0-s7/fp/sp/ra" if self.L["callee_saved"] else ""
        out.append(f"observable behaviour: {'/'.join(self.L['outputs']) or 'no result registers'}{saved}, "
                   f"stores outside the stack frame, calls (target, first {self.L['call_args']} integer "
                   f"argument registers, memory at the call) and faults")
        if self.L["fault_domain"]:
            out.append(f"inputs on which function A faults with {'/'.join(self.L['fault_domain'])} are outside "
                       f"its domain ({paths.get('domain', 0)} path pairs), as in the behaviour oracle")
        if paths.get("a_truncated") or paths.get("b_truncated"):
            out.append(f"loops unrolled to {self.L['unroll']} iterations: executions needing more are not checked "
                       f"({paths['a_truncated']} paths of A, {paths['b_truncated']} of B cut)")
        if paths.get("case_splits", {}).get("alias"):
            out.append(f"pointer alias cases explored by case split ({paths['case_splits']['alias']} splits)")
        if paths.get("undefined_dependent"):
            out.append(f"{paths['undefined_dependent']} path pairs differ only in undefined values (HI/LO after a "
                       f"division by zero, registers clobbered by a call, uninitialised stack): undefined-dependent, "
                       f"not counterexamples")
        out.extend(sorted(notes))
        return out

    # -- pairing --
    def _diffs(self, x: _Rec, y: _Rec, facts: _Facts) -> list:
        rw, noalias, out = _Rewriter(facts), self.L["noalias"], []
        if x.fault != y.fault:
            out.append(("fault", ()))
        if len(x.events) != len(y.events):
            out.append(("events", ()))
        else:
            for i, (ea, eb) in enumerate(zip(x.events, y.events)):
                # each part of a call on its own: an argument register a previous call clobbered
                # (undefined in both) must not hide a defined difference in another argument
                ta = (rw(ea[0]),) + tuple(map(rw, ea[1]))
                tb = (rw(eb[0]),) + tuple(map(rw, eb[1]))
                for j, (u, v) in enumerate(zip(ta, tb)):
                    if u != v:
                        out.append((f"call{i}.{'target' if j == 0 else f'a{j - 1}'}", (u, v)))
                if len(ta) != len(tb):
                    out.append((f"call{i}.arity", ()))
                if _canon(ea[2], facts, rw, noalias) != _canon(eb[2], facts, rw, noalias):
                    out.append((f"call{i}.memory", tuple(s[2] for s in ea[2] + eb[2])))
        if x.fault is None and y.fault is None:
            for name in sorted(x.outputs):
                a, b = rw(x.outputs[name]), rw(y.outputs[name])
                if a != b:
                    out.append((name, (a, b)))
        if _canon(x.stores, facts, rw, noalias) != _canon(y.stores, facts, rw, noalias):
            out.append(("memory", tuple(s[2] for s in x.stores + y.stores)))
        return out

    @staticmethod
    def _undefined(terms) -> bool:
        return bool(terms) and any(_has_undef(t) for t in terms)

    def _selfcheck(self, recs: list) -> str | None:
        done = [r for r in recs if not r.truncated]
        for r in done:
            f = r.facts.copy()
            if not all(f.add(a, p) for a, p in r.conds):
                return "a path condition contradicts itself"
            if self._diffs(r, r, f):
                return "a path differs from itself"
        if len(done) ** 2 <= self.L["max_pairs"]:
            for i, x in enumerate(done):
                for y in done[i + 1:]:
                    f = x.facts.copy()
                    if all(f.add(a, p) for a, p in y.conds):
                        bad = [n for n, terms in self._diffs(x, y, f) if not self._undefined(terms)]
                        if bad:
                            return f"two paths are not separated by their conditions and differ in {bad}"
        return None

    def _compare(self, ra: list, rb: list) -> tuple:
        stats = Counter(pairs=0, infeasible=0, equal=0, undefined_dependent=0, domain=0, unresolved=0)
        unresolved = []
        for x in (r for r in ra if not r.truncated):
            for y in (r for r in rb if not r.truncated):
                stats["pairs"] += 1
                if stats["pairs"] > self.L["max_pairs"]:
                    raise _Budget(f"more than {self.L['max_pairs']} path pairs")
                f = x.facts.copy()
                if not all(f.add(a, p) for a, p in y.conds):
                    stats["infeasible"] += 1
                    continue
                if x.fault and x.fault[0] in self.L["fault_domain"]:
                    stats["domain"] += 1
                    continue
                diffs = self._diffs(x, y, f)
                if not diffs:
                    stats["equal"] += 1
                    continue
                defined = [(n, terms) for n, terms in diffs if not self._undefined(terms)]
                if not defined:
                    stats["undefined_dependent"] += 1
                    continue
                witness = self._search(x, y, defined, stats["pairs"])
                if witness:
                    return dict(stats), witness, unresolved
                stats["unresolved"] += 1
                unresolved.append([n for n, _ in defined])
        return dict(stats), None, unresolved

    # -- witness search --
    def _search(self, x: _Rec, y: _Rec, defined: list, pair_no: int) -> dict | None:
        conds = list(x.conds) + list(y.conds)
        names: dict = {}
        for r in (x, y):
            for t in list(r.outputs.values()) + [a for a, _ in r.conds]:
                _symbols(t, names)
            for target, args, stores, _base in r.events:
                for t in (target,) + args + tuple(s[0] for s in stores) + tuple(s[2] for s in stores):
                    _symbols(t, names)
            for a, _w, v in r.stores:
                _symbols(a, names)
                _symbols(v, names)
        inputs = sorted((n, w) for n, w in names.items() if not n.startswith("undef."))
        rng = random.Random(self.L["seed"] * 1000003 + pair_no)
        for n in range(self.L["samples"]):
            asg = {}
            for name, w in inputs:
                if n < 2:
                    v = n
                elif n < 8 or rng.random() < 0.5:
                    v = rng.choice(_BOUNDARY)
                else:
                    v = rng.getrandbits(w) if rng.random() < 0.6 else rng.randrange(-16, 17)
                asg[name] = v & _mask(w)
            cx = _Concrete(asg, {}, self.L["seed"])
            if not self._satisfy(cx, conds):
                continue
            differs = []
            for name, _terms in defined:
                va, vb = self._concrete(cx, x, name), self._concrete(cx, y, name)
                if not self._same(cx, va, vb):
                    differs.append((name, va, vb))
            if differs:
                return self._witness(cx, inputs, x, y, differs)
        return None

    @staticmethod
    def _satisfy(cx: _Concrete, conds: list) -> bool:
        for _ in range(8):
            bad = [(a, p) for a, p in conds if cx.ev(a) != int(p)]
            if not bad:
                return True
            for a, p in bad:
                if cx.ev(a) != int(p):
                    _fix(cx, a, p)
        return all(cx.ev(a) == int(p) for a, p in conds)

    @staticmethod
    def _cmem(cx: _Concrete, stores, base: str) -> tuple:
        d = {}
        for a, w, v in stores:
            av, vv = cx.ev(a), cx.ev(v)
            for i in range(w):
                d[(av + i) & M32] = (vv >> (8 * (w - 1 - i))) & 0xFF
        return ("mem", d, base)

    def _concrete(self, cx: _Concrete, r: _Rec, name: str):
        if name == "fault":
            return r.fault
        if name == "events":
            return len(r.events)
        if name == "memory":
            return self._cmem(cx, r.stores, f"M{r.base}")
        if name.startswith("call"):
            index, _, part = name[4:].partition(".")
            target, args, stores, base = r.events[int(index)]
            if part == "target":
                return cx.ev(target)
            if part.startswith("a") and part[1:].isdigit():
                return cx.ev(args[int(part[1:])])
            if part == "arity":
                return len(args)
            if part == "memory":
                return self._cmem(cx, stores, f"M{base}")
            return (cx.ev(target), tuple(cx.ev(a) for a in args), self._cmem(cx, stores, f"M{base}"))
        return cx.ev(r.outputs[name])

    def _same(self, cx: _Concrete, a, b) -> bool:
        if isinstance(a, tuple) and isinstance(b, tuple) and a and b and a[0] == b[0] == "mem":
            (_, da, ba), (_, db, bb) = a, b
            return all(da.get(k, cx.byte(ba, k)) == db.get(k, cx.byte(bb, k)) for k in set(da) | set(db))
        if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and not (a and isinstance(a[0], str)):
            return all(self._same(cx, u, v) for u, v in zip(a, b))
        return a == b

    @staticmethod
    def _plain(v):
        if isinstance(v, int):
            return f"0x{v:X}"
        if isinstance(v, tuple) and v and v[0] == "mem":
            return {f"0x{k:08X}": f"0x{b:02X}" for k, b in sorted(v[1].items())}
        if isinstance(v, tuple):
            return [_Validator._plain(x) for x in v]
        return v

    def _witness(self, cx: _Concrete, inputs: list, x: _Rec, y: _Rec, differs: list) -> dict:
        return {
            "inputs": {name: f"0x{cx.asg.get(name, 0):X}" for name, _w in inputs},
            "memory": {f"{base}:0x{a:08X}": f"0x{b:02X}" for (base, a), b in sorted(cx.mem.items())},
            "memory_default": f"bytes not listed are blake2b(seed={self.L['seed']}, base, address) & 0xFF",
            "differs": [n for n, _a, _b in differs],
            "a": {n: self._plain(va) for n, va, _vb in differs},
            "b": {n: self._plain(vb) for n, _va, vb in differs},
            "path_a": [f"{_show(a)} is {p}" for a, p in x.conds],
            "path_b": [f"{_show(a)} is {p}" for a, p in y.conds],
        }


# ---- API --------------------------------------------------------------------------------------
def validate(words_a, vram_a: int, words_b, vram_b: int, limits: dict | None = None) -> Result:
    """Bounded translation validation of function B against function A (the target).

    ``words_*`` are big-endian instruction words (ints), ``vram_*`` their load addresses.
    ``limits`` overrides ``DEFAULT_LIMITS``.  The result is evidence only: ``equivalent`` holds
    under ``Result.assumptions`` and never stands in for byte identity."""
    unknown = set(limits or {}) - set(DEFAULT_LIMITS)
    if unknown:
        raise ValueError(f"unknown limits: {sorted(unknown)}")
    L = {**DEFAULT_LIMITS, **(limits or {})}
    L["outputs"], L["fault_domain"] = tuple(L["outputs"]), tuple(L["fault_domain"])
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10000))
    _KEYS.clear()
    try:
        return _Validator(L).run(words_a, vram_a, words_b, vram_b)
    finally:
        _KEYS.clear()
        sys.setrecursionlimit(old)


def validate_rewrite(c_before: str, c_after: str, compile_fn, limits: dict | None = None) -> Result:
    """Justify one source rewrite (for ``n64dk.match.egraph``): compile both sides with the
    injected ``compile_fn(source) -> (words, vram)`` and validate the machine code.

    ``compile_fn`` must use one toolchain, one set of flags and one placement for both sides, and
    resolve (or leave unresolved, identically) every relocation.  ``equivalent`` justifies the
    rewrite only under the returned assumptions; it never grants match credit."""
    try:
        words_a, vram_a = compile_fn(c_before)
        words_b, vram_b = compile_fn(c_after)
    except Exception as e:                                 # the injected compiler refused a side
        return Result("unsupported", reason=f"compile failed: {e!r}")
    return validate(words_a, vram_a, words_b, vram_b, limits)


def counterexample_record(result: Result) -> str:
    """JSON for a case dossier: the outcome, its assumptions and the concrete witness, if any."""
    return json.dumps({
        "schema": "n64dk.tv/1",
        **result.as_dict(),
        "note": "bounded translation validation: evidence only, never byte identity or strict-exact credit",
    }, indent=2, sort_keys=True)
