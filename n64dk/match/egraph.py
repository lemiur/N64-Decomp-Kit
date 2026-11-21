"""Bounded equality saturation over the C expressions of one function: an opt-in source variant
generator for the variant search (``iterate``), round 12 addendum item C.

What it does.  Every printable statement, condition and ``if``/``else`` assignment pair of the
function becomes a small e-graph: hash-consed e-nodes, a union-find over e-classes and congruence
closure by rebuilding (``EGraph``).  Conditional rewrite rules add equivalent forms; nothing is
ever removed, so the original expression stays in the graph.  Saturation is bounded (e-nodes
per statement, iterations, statements per function) and deterministic (classes in id order, nodes
in insertion order, rules in a fixed order).  Extraction keeps, for every e-class and context
(value, lvalue, branch condition), the few cheapest terms under a MIPS / KMC GCC cost model, and
the function-level step picks a bounded Pareto set of whole-function variants, optionally steered
toward a target's opcode multiset.

Side conditions (``Dialect`` is the overflow model; the default is conservative):

* every value rule needs a side-effect-free, non-volatile e-class (calls, assignments,
  increments and volatile accesses are never rewritten, reordered, duplicated or dropped);
* commutation needs both operands side-effect free; floating point is never reassociated and
  ``a < b`` is never turned into ``!(a >= b)`` for floating operands (NaN);
* reassociation of ``+``/``*`` and ``x * 2^k`` <-> ``x << k`` need an unsigned 32-bit type or a
  provably non-negative operand; a signed type only when ``Dialect.signed_wrap`` (those variants
  carry ``semantic_review``);
* ``x / 2^k`` <-> ``x >> k`` and ``x % 2^k`` <-> ``x & (2^k-1)`` only unsigned or non-negative;
* casts are dropped only when value-preserving with the same promoted type; ``x & 0xFF`` <->
  ``(u8)x`` only when ``x`` promotes to ``int`` (otherwise the expression's type would change);
* pointer forms (``p + i`` <-> ``&p[i]``, ``*(p + i)`` <-> ``p[i]``, ``(*p).f`` <-> ``p->f``)
  need a pointer to a complete element type;
* statement forms (``a = a op b`` <-> ``a op= b``, increment spellings, ``x = c ? a : b`` <->
  ``if/else``) need a pure, non-volatile lvalue and pure operands; an ``if``/``else`` or a
  temporary is only produced for a statement that sits directly in a block;
* a temporary is introduced only for a repeated pure sub-expression of one statement whose other
  parts have no side effects, never across statements, never in an lvalue position, and never
  hoisted out of a conditionally evaluated position when it may trap (memory, division).

Nothing here grants credit: a variant is ordinary source that the normal strict build and
comparison judges.  Not wired into any default: ``iterate`` uses it only for the ``egraph`` family
or when ``N64DK_EGRAPH=1``.
"""
from __future__ import annotations

import copy
import os
import re
import struct
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

from pycparser import c_ast

from . import astforms, astprep
from .astprep import PrepError, apply_edits, expr_text, simple_stmt_text

M32 = 0xFFFFFFFF
INT, UINT = astforms.INT, astforms.UINT
INT_K = ("int", 4, True)

_COMMUTATIVE = ("+", "*", "&", "|", "^", "==", "!=")
_ASSOC = ("+", "*", "&", "|", "^")
_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
_NEGATE = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}
_COMPARE = ("==", "!=", "<", ">", "<=", ">=")
_LOGICAL = ("&&", "||")
_ARITH = ("+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^")
BINARY = _ARITH + _COMPARE + _LOGICAL
_UNARY = {"-": "neg", "+": "pos", "~": "bnot", "!": "not", "*": "deref", "&": "addr"}
_UNARY_BACK = {v: k for k, v in _UNARY.items()}
IMPURE = ("assign", "incdec", "call")
LV_OPS = ("var", "deref", "index", "field", "call", "atom")
WRAPPERS = ("stmt", "ret", "cond", "val", "init", "ifelse", "let")
LEAVES = ("var", "int", "lit", "atom")
_OPCLASS = {"+": "add", "-": "sub", "&": "and", "|": "or", "^": "xor", "<<": "shl", ">>": "sar", "*": "mul",
            "/": "div", "%": "div"}
# opcode classes of the cost model and of a decoded target (names follow ir.py's operations)
VOCABULARY = ("add", "sub", "and", "or", "xor", "nor", "shl", "shr", "sar", "lt", "ltu", "mul", "div", "divu",
              "lui", "li", "branch", "call", "cvt", "fpu",
              "load8s", "load8u", "load16s", "load16u", "load32s", "load32u", "load32f", "load64f", "load64s",
              "store8", "store16", "store32", "store32f", "store64f", "store64")

# Set by a caller that knows the target of a function (the bench): name -> {"target": words,
# "base": words of the current best build}.  Used only to order variants, never as evidence.
TARGETS: dict[str, dict] = {}
STATS: dict[str, float] = {"calls": 0, "units": 0, "nodes": 0, "seconds": 0.0, "candidates": 0}


@dataclass(frozen=True)
class Limits:
    max_nodes: int = 400          # e-nodes per statement e-graph
    max_iterations: int = 5       # saturation iterations per statement
    max_units: int = 64           # statements / conditions per function
    max_total_nodes: int = 12000  # e-nodes per function
    per_class: int = 3            # terms kept per e-class and context during extraction
    extract_rounds: int = 8
    alternatives: int = 4         # alternatives kept per statement
    greedy_steps: int = 8         # target-guided combination steps


@dataclass(frozen=True)
class Dialect:
    """Overflow model.  ``signed_wrap``: signed ``+ - * <<`` may be treated as wrapping (KMC GCC
    emits addu/subu/mult, but C makes signed overflow undefined): off by default."""
    signed_wrap: bool = False


# ============================================================================== analysis data
@dataclass
class Data:
    tnode: object = None
    kind: tuple | None = None
    pure: bool = True             # no side effects (no write, call, increment)
    volatile: bool = False
    memory: bool = False          # reads memory
    trap: bool = False            # may fault (dereference, call, division by a non-constant)
    const: int | None = None
    nonneg: bool = False
    boolean: bool = False         # value is 0 or 1

    def merge(self, other: "Data") -> None:
        if self.tnode is None and other.tnode is not None:
            self.tnode, self.kind = other.tnode, other.kind
        self.pure = self.pure and other.pure
        self.volatile = self.volatile or other.volatile
        self.memory = self.memory or other.memory
        self.trap = self.trap or other.trap
        if self.const is None:
            self.const = other.const
        self.nonneg = self.nonneg or other.nonneg
        self.boolean = self.boolean or other.boolean


@dataclass(frozen=True)
class VarInfo:
    where: str                    # reg, mem (stack), global, larray, garray, func, enum
    tnode: object
    kind: tuple | None
    volatile: bool


def _int_kind(k) -> bool:
    return bool(k) and k[0] == "int"


def _int32(k) -> bool:
    return _int_kind(k) and k[1] == 4 and k[2] is not None


def _promoted(k):
    if _int_kind(k) and k[1] < 4:
        return INT_K
    return k


def _common(ka, kb):
    """Usual arithmetic conversions of two integer kinds of at most 32 bits (None otherwise)."""
    if not (_int_kind(ka) and _int_kind(kb)) or ka[1] > 4 or kb[1] > 4:
        return None
    pa, pb = _promoted(ka), _promoted(kb)
    return ("int", 4, False) if pa[2] is False or pb[2] is False else INT_K


def _convert(value: int, kind) -> int | None:
    """``value`` converted to integer ``kind`` (modular, as KMC GCC does); None for an unknown
    signedness (plain char)."""
    if not _int_kind(kind) or kind[2] is None:
        return None
    bits = 8 * kind[1]
    value &= (1 << bits) - 1
    if kind[2] and value >> (bits - 1):
        value -= 1 << bits
    return value


def _log2(value) -> int | None:
    if value is None or value <= 0 or value & (value - 1):
        return None
    return value.bit_length() - 1


def _lit_attr(value: int, unsigned: bool = False):
    if unsigned:
        value &= M32
    elif not -0x7FFFFFFF <= value <= 0x7FFFFFFF:
        return None
    magnitude = abs(value)
    text = (f"0x{magnitude:X}" if magnitude >= 0x100 else str(magnitude)) + \
        ("U" if unsigned and value <= 0x7FFFFFFF else "")
    return (value, text, "unsigned int" if unsigned else "int")


def _unsigned_literal(attr) -> bool:
    return "u" in attr[1].lower() or attr[0] > 0x7FFFFFFF or "unsigned" in attr[2]


def C(cid):
    return ("c", cid)


def N(op, attr, *kids):
    return ("n", op, attr, kids)


def LIT(value, unsigned=False):
    attr = _lit_attr(value, unsigned)
    return None if attr is None else N("int", attr)


# ============================================================================== e-graph
class EClass:
    __slots__ = ("nodes", "data")

    def __init__(self, nodes, data):
        self.nodes, self.data = nodes, data


class EGraph:
    """E-nodes are ``(op, attr, child class ids)``; ``attr`` is hashable (a name, a literal, a
    cast key, or a serial number that keeps every side effect a distinct node)."""

    def __init__(self, types=None, dialect: Dialect | None = None, variables=None, casts=None):
        self.T = types
        self.dialect = dialect or Dialect()
        self.vars: dict[str, VarInfo] = variables if variables is not None else {}
        self.casts: dict[str, object] = casts if casts is not None else {}
        self.parent: list[int] = []
        self.classes: dict[int, EClass] = {}
        self.memo: dict[tuple, int] = {}
        self.why: dict[tuple, str] = {}          # node -> rule that first created it
        self.size = 0
        self.block_ok = False                    # the statement sits directly in a block

    # -- union-find and hash-consing
    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def canon(self, node):
        op, attr, kids = node
        return (op, attr, tuple(self.find(k) for k in kids))

    def data(self, cid) -> Data:
        return self.classes[self.find(cid)].data

    def nodes(self, cid) -> list:
        return list(self.classes[self.find(cid)].nodes)

    def add(self, op, attr=None, kids=(), data: Data | None = None, why: str | None = None) -> int:
        node = self.canon((op, attr, tuple(kids)))
        hit = self.memo.get(node)
        if hit is not None:
            return self.find(hit)
        cid = len(self.parent)
        self.parent.append(cid)
        self.classes[cid] = EClass([node], data if data is not None else self.derive(node))
        self.memo[node] = cid
        self.size += 1
        if why:
            self.why[node] = why
        return cid

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if b < a:
            a, b = b, a
        self.parent[b] = a
        keep, gone = self.classes[a], self.classes.pop(b)
        keep.nodes.extend(gone.nodes)
        keep.data.merge(gone.data)
        return True

    def rebuild(self) -> None:
        """Congruence closure: re-canonicalise every node; nodes that became equal merge their
        classes; repeat until nothing merges."""
        for _ in range(256):
            memo, why, pending = {}, {}, []
            for cid in sorted(self.classes):
                cls = self.classes[cid]
                unique = {}
                for node in cls.nodes:
                    canon = self.canon(node)
                    if node in self.why:
                        why.setdefault(canon, self.why[node])
                    if canon in unique:
                        continue
                    unique[canon] = None
                    other = memo.get(canon)
                    if other is None:
                        memo[canon] = cid
                    elif other != cid:
                        pending.append((other, cid))
                cls.nodes = list(unique)
            self.why = why
            if not pending:
                self.memo = memo
                return
            for a, b in pending:
                self.union(a, b)
        self.memo = {self.canon(n): cid for cid in sorted(self.classes) for n in self.classes[cid].nodes}

    def build(self, spec, why: str) -> int:
        if spec[0] == "c":
            return self.find(spec[1])
        _, op, attr, kids = spec
        ids = tuple(self.build(k, why) for k in kids)
        return self.add(op, attr, ids, why=why)

    def represents(self, cid: int, term, _seen=None) -> bool:
        """True when ``term`` (``(op, attr, kids)``; serial numbers and literal spellings ignored)
        is one of the expressions of e-class ``cid``."""
        seen = set() if _seen is None else _seen
        key = (self.find(cid), id(term))
        if key in seen:
            return False
        seen.add(key)
        op, attr, kids = term
        for node in self.classes[self.find(cid)].nodes:
            if node[0] != op or len(node[2]) != len(kids) or not _same_attr(op, node[1], attr):
                continue
            if all(self.represents(k, t, seen) for k, t in zip(node[2], kids)):
                seen.discard(key)
                return True
        seen.discard(key)
        return False

    # -- per-class facts
    def derive(self, node, tnode=None) -> Data:
        """Facts of a new node from its children; ``tnode`` (the parser's type of an original
        expression) overrides the inferred type."""
        op, attr, kids = node
        K = [self.data(k) for k in kids]
        d = Data()
        d.pure = all(k.pure for k in K) and op not in IMPURE
        d.volatile = any(k.volatile for k in K)
        d.memory = any(k.memory for k in K) or op in ("deref", "index", "call") or \
            (op == "field" and attr[0] == "->")
        d.trap = any(k.trap for k in K) or op in ("deref", "index", "call") or \
            (op == "field" and attr[0] == "->") or (op in ("/", "%") and not K[1].const)
        if op == "var":
            info = self.vars.get(attr)
            if info is not None:
                d.memory = info.where in ("mem", "global")
                d.volatile = info.volatile
        d.tnode = tnode if tnode is not None and op != "int" else self._type(op, attr, K)
        if self.T is not None and d.tnode is not None:
            d.kind = self.T.kind(d.tnode)
        elif op == "int":
            d.kind = ("int", 4, not _unsigned_literal(attr))
        d.const = self._fold(op, attr, K, d.kind)
        d.boolean = op in _COMPARE or op in _LOGICAL or op == "not" or d.const in (0, 1)
        d.nonneg = (d.const is not None and d.const >= 0) or (_int_kind(d.kind) and d.kind[2] is False) or \
            d.boolean or (op == "&" and _int_kind(d.kind) and any(k.nonneg for k in K)) or \
            (op == ">>" and bool(K) and K[0].nonneg) or \
            (op == "cast" and _int_kind(d.kind) and _int_kind(K[0].kind) and K[0].nonneg
             and d.kind[1] >= K[0].kind[1])
        return d

    def _type(self, op, attr, K):
        if op == "int":
            return UINT if _unsigned_literal(attr) else INT
        T = self.T
        if T is None:
            return None
        t = [k.tnode for k in K]
        if op == "var":
            info = self.vars.get(attr)
            return info.tnode if info is not None else T.var_type(attr)
        if op == "lit":
            if attr[0] in ("float", "double"):
                return astforms.FLOAT if attr[1].lower().endswith("f") else astforms.DOUBLE
            return INT if attr[0] == "char" else None
        if op in ("neg", "pos", "bnot"):
            return T.promote(t[0])
        if op == "not" or op in _COMPARE or op in _LOGICAL:
            return INT
        if op == "deref":
            return T.deref(t[0])
        if op == "addr":
            return c_ast.PtrDecl([], t[0]) if t[0] is not None else None
        if op == "cast":
            typename = self.casts.get(attr)
            return typename.type if typename is not None else None
        if op == "index":
            return T.deref(t[0]) if T.is_pointer(t[0]) else T.deref(t[1])
        if op == "field":
            base = t[0] if attr[0] == "." else T.deref(t[0])
            k = T.kind(base)
            if not k or k[0] != "struct" or k[1].decls is None:
                return None
            return T._field(k[1], attr[1])
        if op in ("<<", ">>"):
            return T.promote(t[0])
        if op in ("+", "-"):
            if T.is_pointer(t[0]):
                if op == "-" and T.is_pointer(t[1]):
                    return INT
                k = T.kind(t[0])
                return c_ast.PtrDecl([], k[1]) if k[0] == "array" else t[0]
            if op == "+" and T.is_pointer(t[1]):
                k = T.kind(t[1])
                return c_ast.PtrDecl([], k[1]) if k[0] == "array" else t[1]
        if op in _ARITH:
            return T.arith(t[0], t[1])
        if op == "?:":
            a, b = t[1], t[2]
            return T.arith(a, b) if T.is_arith(a) and T.is_arith(b) else a
        if op in ("assign", "incdec"):
            return t[0]
        if op == "call":
            k = T.kind(t[0])
            if k and k[0] == "ptr":
                k = T.kind(k[1])
            return k[1].type if k and k[0] == "func" else None
        return None

    def _result(self, value: int, kind) -> int | None:
        """An arithmetic result in ``kind``: unsigned wraps; signed overflow gives None unless the
        dialect wraps."""
        bits = 8 * kind[1]
        if kind[2] is False:
            return value & ((1 << bits) - 1)
        if -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
            return value
        return _convert(value, kind) if self.dialect.signed_wrap else None

    def _fold(self, op, attr, K, kind):
        if op == "int":
            return attr[0]
        if not _int_kind(kind) or kind[1] > 4 or kind[2] is None and op != "cast":
            return None
        vals = [k.const for k in K]
        if not vals or any(v is None for v in vals) or any(not _int_kind(k.kind) or k.kind[1] > 4 for k in K):
            return None
        if op == "not":
            return int(vals[0] == 0)
        if op == "cast":
            return _convert(vals[0], kind)
        if op == "pos":
            return vals[0]
        if op == "neg":
            v = _convert(vals[0], _promoted(K[0].kind))
            return None if v is None else self._result(-v, kind)
        if op == "bnot":
            v = _convert(vals[0], kind)
            return None if v is None else self._result(~v, kind) if kind[2] is not False else (~v) & M32
        if len(vals) != 2:
            return None
        a, b = vals
        if op in _COMPARE:
            common = _common(K[0].kind, K[1].kind)
            if common is None:
                return None
            a, b = _convert(a, common), _convert(b, common)
            return int({"==": a == b, "!=": a != b, "<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op])
        if op == "&&":
            return int(bool(a) and bool(b))
        if op == "||":
            return int(bool(a) or bool(b))
        if op in ("<<", ">>"):
            if not 0 <= b < 32:
                return None
            a = _convert(a, kind)
            if a is None:
                return None
            if op == "<<":
                if kind[2] and a < 0 and not self.dialect.signed_wrap:
                    return None
                return self._result(a << b, kind)
            return a >> b
        a, b = _convert(a, kind), _convert(b, kind)
        if a is None or b is None:
            return None
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == "*":
            r = a * b
        elif op in ("/", "%"):
            if b == 0:
                return None
            q = abs(a) // abs(b)
            q = -q if (a < 0) != (b < 0) else q
            r = q if op == "/" else a - b * q
        elif op == "&":
            r = a & b
        elif op == "|":
            r = a | b
        elif op == "^":
            r = a ^ b
        else:
            return None
        return self._result(r, kind)

    def cast_kind(self, key):
        typename = self.casts.get(key)
        if typename is None or self.T is None:
            return None
        return self.T.kind(typename.type)

    def named_cast(self, size: int) -> str:
        """Key of the cast to the unsigned type of ``size`` bytes (``u8``/``u16`` when declared)."""
        name = {1: "u8", 2: "u16"}[size]
        T = self.T
        if T is not None and name in T.env.typedefs and T.kind(astforms._mk([name])) == ("int", size, False):
            names = [name]
        else:
            names = ["unsigned", "char" if size == 1 else "short"]
        key = " ".join(names)
        if key not in self.casts:
            self.casts[key] = c_ast.Typename(None, [], None, c_ast.TypeDecl(None, [], None,
                                                                            c_ast.IdentifierType(list(names))))
        return key

    def cast_for(self, tnode) -> str | None:
        """Key of a cast to a declared scalar type written as plain type names (``s16``)."""
        if not isinstance(tnode, c_ast.TypeDecl) or not isinstance(tnode.type, c_ast.IdentifierType) \
                or "volatile" in (tnode.quals or []):
            return None
        names = list(tnode.type.names)
        key = " ".join(names)
        if key not in self.casts:
            self.casts[key] = c_ast.Typename(None, [], None, c_ast.TypeDecl(None, [], None,
                                                                            c_ast.IdentifierType(names)))
        return key


def _same_attr(op, a, b) -> bool:
    if op in ("assign", "incdec"):
        return a[0] == b[0]
    if op in ("call", "atom"):
        return True
    if op == "int":
        return a[0] == b[0]
    return a == b


# ============================================================================== rules
def _clean(eg: EGraph, cid) -> bool:
    d = eg.data(cid)
    return d.pure and not d.volatile


def _scalar(eg: EGraph, cid) -> bool:
    k = eg.data(cid).kind
    return bool(k) and k[0] in ("int", "float", "ptr", "array")


def _ptr_elem(eg: EGraph, cid) -> bool:
    """A pointer or array whose element type is complete (indexing is meaningful)."""
    d = eg.data(cid)
    if not d.kind or d.kind[0] not in ("ptr", "array") or eg.T is None:
        return False
    elem = eg.T.kind(eg.T.deref(d.tnode))
    return bool(elem) and elem[0] not in ("void", "func")


def _is_int(eg: EGraph, cid) -> bool:
    return _int_kind(eg.data(cid).kind)


def _r_commute(eg, cid, node):
    op, _, (a, b) = node
    if eg.find(a) != eg.find(b) and _clean(eg, a) and _clean(eg, b):
        yield "commute", N(op, None, C(b), C(a))


def _assoc_ok(eg, op, p, q):
    kp, kq = eg.data(p).kind, eg.data(q).kind
    if not (_int_kind(kp) and _int_kind(kq)) or kp[1] > 4 or kq[1] > 4:
        return None
    if op in ("&", "|", "^"):
        return "assoc"
    common = _common(kp, kq)
    if common is not None and common[2] is False:
        return "assoc"
    return "assoc~wrap" if eg.dialect.signed_wrap else None


def _r_assoc(eg, cid, node):
    op, _, (x, y) = node
    if not _int32(eg.data(cid).kind):
        return
    for inner in eg.nodes(x):
        if inner[0] == op:
            a, b = inner[2]
            tag = _assoc_ok(eg, op, b, y)
            if tag:
                yield tag, N(op, None, C(a), N(op, None, C(b), C(y)))
    for inner in eg.nodes(y):
        if inner[0] == op:
            b, c = inner[2]
            tag = _assoc_ok(eg, op, x, b)
            if tag:
                yield tag, N(op, None, N(op, None, C(x), C(b)), C(c))


def _shift_ok(eg, cid, x, name):
    kc, dx = eg.data(cid).kind, eg.data(x)
    if not (_int32(kc) and _int_kind(dx.kind) and dx.kind[1] <= 4 and _promoted(dx.kind) == kc):
        return None
    if kc[2] is False or dx.nonneg:
        return name
    return name + "~wrap" if eg.dialect.signed_wrap else None


def _r_mulshift(eg, cid, node):
    op, _, (x, k) = node
    c = eg.data(k).const
    if op == "*":
        n = _log2(c)
        if n is not None and 1 <= n <= 30:
            tag = _shift_ok(eg, cid, x, "mul-shift")
            if tag:
                yield tag, N("<<", None, C(x), LIT(n))
    elif op == "<<" and c is not None and 1 <= c <= 30:
        tag = _shift_ok(eg, cid, x, "shift-mul")
        if tag:
            yield tag, N("*", None, C(x), LIT(1 << c))


def _r_divshift(eg, cid, node):
    op, _, (x, k) = node
    kc, dx = eg.data(cid).kind, eg.data(x)
    if not (_int32(kc) and _int_kind(dx.kind) and dx.kind[1] <= 4 and _promoted(dx.kind) == kc):
        return
    if not (kc[2] is False or dx.nonneg):
        return
    unsigned = kc[2] is False
    c = eg.data(k).const
    n = _log2(c)
    if op == "/" and n is not None and n >= 1:
        yield "div-shift", N(">>", None, C(x), LIT(n))
    elif op == "%" and n is not None and n >= 1:
        lit = LIT(c - 1, unsigned)
        if lit:
            yield "mod-mask", N("&", None, C(x), lit)
    elif op == ">>" and c is not None and 1 <= c <= 31:
        lit = LIT(1 << c, unsigned)
        if lit:
            yield "shift-div", N("/", None, C(x), lit)
    elif op == "&" and c is not None and c > 0:
        m = _log2(c + 1)
        lit = LIT(c + 1, unsigned) if m is not None and m >= 1 else None
        if lit:
            yield "mask-mod", N("%", None, C(x), lit)


def _r_flip(eg, cid, node):
    op, _, (a, b) = node
    if _clean(eg, a) and _clean(eg, b):
        yield "compare-swap", N(_FLIP[op], None, C(b), C(a))


def _negatable(eg, op, kids) -> bool:
    if op in ("==", "!="):
        return True
    kinds = [eg.data(k).kind for k in kids]
    return all(k and k[0] in ("int", "ptr", "array") for k in kinds)


def _r_compare(eg, cid, node):
    op, _, kids = node
    if op in _COMPARE and _negatable(eg, op, kids):
        yield "compare-not", N("not", None, N(_NEGATE[op], None, C(kids[0]), C(kids[1])))
    if op in ("<", "<=", ">", ">="):
        yield from _r_flip(eg, cid, node)
        yield from _r_bound(eg, cid, node)


def _r_bound(eg, cid, node):
    op, _, (a, b) = node
    c = eg.data(b).const
    if c is None:
        return
    common = _common(eg.data(a).kind, eg.data(b).kind)
    if common is None:
        return
    unsigned = common[2] is False
    v = _convert(c, common)
    new_op, nv = {"<=": ("<", v + 1), ">": (">=", v + 1), "<": ("<=", v - 1), ">=": (">", v - 1)}[op]
    lo, hi = (0, M32) if unsigned else (-0x7FFFFFFF, 0x7FFFFFFF)
    if lo <= nv <= hi:
        lit = LIT(nv, unsigned)
        if lit:
            yield "compare-bound", N(new_op, None, C(a), lit)


def _r_not(eg, cid, node):
    x = node[2][0]
    if _scalar(eg, x):
        yield "not-eq0", N("==", None, C(x), LIT(0))
    for inner in eg.nodes(x):
        if inner[0] == "not" and _scalar(eg, inner[2][0]):
            yield "notnot-ne0", N("!=", None, C(inner[2][0]), LIT(0))
        if inner[0] in _COMPARE and _negatable(eg, inner[0], inner[2]):
            yield "not-compare", N(_NEGATE[inner[0]], None, C(inner[2][0]), C(inner[2][1]))


def _r_zero(eg, cid, node):
    op, _, (x, z) = node
    if eg.data(z).const != 0 or not _scalar(eg, x):
        return
    if op == "==":
        yield "eq0-not", N("not", None, C(x))
    else:
        yield "ne0-notnot", N("not", None, N("not", None, C(x)))
        if eg.data(x).boolean:
            yield "ne0-drop", C(x)


def _truth_forms(eg, c):
    """Equivalent operands in a truth-value context (a condition, ``&&``/``||``, ``?:``)."""
    if not _clean(eg, c):
        return
    d = eg.data(c)
    if _scalar(eg, c) and not d.boolean:
        yield "truth-ne0", N("!=", None, C(c), LIT(0))
    for inner in eg.nodes(c):
        if inner[0] == "!=" and eg.data(inner[2][1]).const == 0 and _scalar(eg, inner[2][0]):
            yield "truth-drop-ne0", C(inner[2][0])


def _r_truth(eg, cid, node):
    op, attr, kids = node
    for i in {"cond": (0,), "&&": (0, 1), "||": (0, 1), "?:": (0,)}[op]:
        for name, spec in _truth_forms(eg, kids[i]):
            new = [C(k) for k in kids]
            new[i] = spec
            yield name, N(op, attr, *new)


def _preserving(xk, tk) -> bool:
    if xk == tk:
        return True
    if xk[2] is None or tk[2] is None:
        return False
    return tk[1] > xk[1] and (tk[2] is True or xk[2] is False)


def _r_cast(eg, cid, node):
    _, key, (x,) = node
    tk, dx = eg.cast_kind(key), eg.data(x)
    xk = dx.kind
    if not tk or not xk:
        return
    if tk[0] == "int" and xk[0] == "int":
        if xk[1] <= 4 and tk[1] <= 4 and _preserving(xk, tk) and _promoted(xk) == _promoted(tk):
            yield "cast-drop", C(x)
        if tk[2] is False and tk[1] in (1, 2) and xk[1] <= 4 and _promoted(xk) == INT_K:
            yield "cast-mask", N("&", None, C(x), LIT(0xFF if tk[1] == 1 else 0xFFFF))
    elif tk[0] == "float" and xk == tk:
        yield "cast-drop", C(x)
    elif tk[0] == "ptr" and xk[0] == "ptr" and eg.T is not None and \
            eg.T.sig(eg.casts[key].type) == eg.T.sig(dx.tnode) and eg.T.sig(dx.tnode) is not None:
        yield "cast-drop", C(x)


def _r_and(eg, cid, node):
    _, _, (x, m) = node
    c = eg.data(m).const
    xk = eg.data(x).kind
    if c in (0xFF, 0xFFFF) and _int_kind(xk) and xk[1] <= 4 and _promoted(xk) == INT_K:
        yield "mask-cast", N("cast", eg.named_cast(1 if c == 0xFF else 2), C(x))


def _r_plus(eg, cid, node):
    _, _, (p, i) = node
    if _ptr_elem(eg, p) and _is_int(eg, i):
        yield "ptr-index", N("addr", None, N("index", None, C(p), C(i)))


def _r_addr(eg, cid, node):
    for inner in eg.nodes(node[2][0]):
        if inner[0] == "index" and _ptr_elem(eg, inner[2][0]) and _is_int(eg, inner[2][1]):
            yield "index-ptr", N("+", None, C(inner[2][0]), C(inner[2][1]))


def _r_deref(eg, cid, node):
    p = node[2][0]
    for inner in eg.nodes(p):
        if inner[0] == "+" and _ptr_elem(eg, inner[2][0]) and _is_int(eg, inner[2][1]):
            yield "deref-index", N("index", None, C(inner[2][0]), C(inner[2][1]))
    if _ptr_elem(eg, p):
        yield "deref-index0", N("index", None, C(p), LIT(0))


def _r_index(eg, cid, node):
    b, i = node[2]
    if _ptr_elem(eg, b) and _is_int(eg, i):
        if eg.data(i).const == 0:
            yield "index0-deref", N("deref", None, C(b))
        yield "index-deref", N("deref", None, N("+", None, C(b), C(i)))


def _r_field(eg, cid, node):
    (sep, name), (base,) = node[1], node[2]
    if sep == ".":
        for inner in eg.nodes(base):
            if inner[0] == "deref":
                yield "arrow", N("field", ("->", name), C(inner[2][0]))
    else:
        yield "star-dot", N("field", (".", name), N("deref", None, C(base)))


def _ternary_ok(eg, lv, a, b) -> bool:
    if not (_clean(eg, lv) and _clean(eg, a) and _clean(eg, b)):
        return False
    kl, ka, kb = eg.data(lv).kind, eg.data(a).kind, eg.data(b).kind
    if all(_int_kind(k) and k[1] <= 4 for k in (kl, ka, kb)):
        return True                    # every conversion is modular: x = c ? a : b == if/else
    if eg.T is None:
        return False
    sigs = [eg.T.sig(eg.data(c).tnode) for c in (lv, a, b)]
    return sigs[0] is not None and sigs[0] == sigs[1] == sigs[2]


def _r_stmt(eg, cid, node):
    _, attr, (e,) = node
    for inner in eg.nodes(e):
        if inner[0] == "assign":
            (aop, serial), (lv, rv) = inner[1], inner[2]
            if not (_clean(eg, lv) and _clean(eg, rv)):
                continue
            if aop == "=":
                for r in eg.nodes(rv):
                    if r[0] in _ARITH:
                        a, b = r[2]
                        if eg.find(a) == eg.find(lv):
                            yield "op-assign", N("stmt", attr, N("assign", (r[0] + "=", serial), C(lv), C(b)))
                        elif r[0] in _ASSOC and eg.find(b) == eg.find(lv):
                            yield "op-assign", N("stmt", attr, N("assign", (r[0] + "=", serial), C(lv), C(a)))
                    elif r[0] == "?:" and attr is None and eg.block_ok and _ternary_ok(eg, lv, r[2][1], r[2][2]) \
                            and _clean(eg, r[2][0]):
                        c, a, b = r[2]
                        yield "ternary-if", N("ifelse", None, N("cond", None, C(c)),
                                              N("stmt", None, N("assign", ("=", (serial, 1)), C(lv), C(a))),
                                              N("stmt", None, N("assign", ("=", (serial, 2)), C(lv), C(b))))
                    elif r[0] == "cast":
                        tk, lk = eg.cast_kind(r[1]), eg.data(lv).kind
                        x = r[2][0]
                        if tk and tk == lk and tk[0] in ("int", "float") and eg.data(x).kind and \
                                eg.data(x).kind[0] == tk[0]:
                            yield "assign-cast-drop", N("stmt", attr, N("assign", ("=", serial), C(lv), C(x)))
                lk, rk = eg.data(lv).kind, eg.data(rv).kind
                if _int_kind(lk) and lk[1] < 4 and _int_kind(rk) and rk[1] > lk[1] and rk[1] <= 4 and \
                        not any(n[0] == "cast" for n in eg.nodes(rv)):
                    key = eg.cast_for(eg.data(lv).tnode)
                    if key and eg.cast_kind(key) == lk:
                        yield "assign-cast", N("stmt", attr, N("assign", ("=", serial), C(lv), N("cast", key, C(rv))))
            elif aop[:-1] in _ARITH:
                base = aop[:-1]
                yield "expand-assign", N("stmt", attr, N("assign", ("=", serial), C(lv), N(base, None, C(lv), C(rv))))
                if base in ("+", "-") and eg.data(rv).const == 1:
                    for form in (("++", "p++") if base == "+" else ("--", "p--")):
                        yield "incdec", N("stmt", attr, N("incdec", (form, serial), C(lv)))
        elif inner[0] == "incdec":
            form, serial = inner[1]
            lv = inner[2][0]
            if not _clean(eg, lv):
                continue
            sym = form[-1]
            for other in (sym * 2, "p" + sym * 2):
                if other != form:
                    yield "incdec", N("stmt", attr, N("incdec", (other, serial), C(lv)))
            yield "incdec-assign", N("stmt", attr, N("assign", (sym + "=", serial), C(lv), LIT(1)))


def _r_init(eg, cid, node):
    key, (e,) = node[1], node[2]
    dk = eg.cast_kind(key) if key else None
    if not dk:
        return
    if not _clean(eg, e):
        return
    for inner in eg.nodes(e):
        if inner[0] == "cast" and eg.cast_kind(inner[1]) == dk and dk[0] in ("int", "float") and \
                eg.data(inner[2][0]).kind and eg.data(inner[2][0]).kind[0] == dk[0]:
            yield "assign-cast-drop", N("init", key, C(inner[2][0]))
    ek = eg.data(e).kind
    if _int_kind(dk) and dk[1] < 4 and _int_kind(ek) and dk[1] < ek[1] <= 4 and \
            not any(n[0] == "cast" for n in eg.nodes(e)):
        yield "assign-cast", N("init", key, N("cast", key, C(e)))


def _r_ifelse(eg, cid, node):
    cc, s1, s2 = node[2]
    for n1 in eg.nodes(s1):
        if n1[0] != "stmt":
            continue
        for a1 in eg.nodes(n1[2][0]):
            if a1[0] != "assign" or a1[1][0] != "=":
                continue
            for n2 in eg.nodes(s2):
                if n2[0] != "stmt":
                    continue
                for a2 in eg.nodes(n2[2][0]):
                    if a2[0] != "assign" or a2[1][0] != "=":
                        continue
                    (lv, a), (lv2, b) = a1[2], a2[2]
                    if eg.find(lv) != eg.find(lv2) or not _ternary_ok(eg, lv, a, b):
                        continue
                    for cn in eg.nodes(cc):
                        if cn[0] == "cond" and _clean(eg, cn[2][0]):
                            yield "if-ternary", N("stmt", None, N("assign", ("=", ("t", a1[1][1])), C(lv),
                                                                  N("?:", None, C(cn[2][0]), C(a), C(b))))


# op -> [(rule, value rule?)]; a value rule fires only on a pure, non-volatile e-class
RULES: dict[str, list] = {}


def _register(ops, fn, value=True):
    for op in ops:
        RULES.setdefault(op, []).append((fn, value))


_register(_COMMUTATIVE, _r_commute)
_register(_ASSOC, _r_assoc)
_register(("*", "<<"), _r_mulshift)
_register(("/", "%", ">>", "&"), _r_divshift)
_register(_COMPARE, _r_compare)
_register(("not",), _r_not)
_register(("==", "!="), _r_zero)
_register(("&&", "||", "?:"), _r_truth)
_register(("cond",), _r_truth, value=False)
_register(("cast",), _r_cast)
_register(("&",), _r_and)
_register(("+",), _r_plus)
_register(("addr",), _r_addr)
_register(("deref",), _r_deref)
_register(("index",), _r_index)
_register(("field",), _r_field)
_register(("stmt",), _r_stmt, value=False)
_register(("init",), _r_init, value=False)
_register(("ifelse",), _r_ifelse, value=False)


def saturate(eg: EGraph, limits: Limits) -> dict:
    """Bounded, deterministic equality saturation: each iteration collects every match (classes in
    id order, nodes in insertion order, rules in registration order), then applies them until the
    node limit, then rebuilds."""
    stats = {"iterations": 0, "saturated": False, "stopped": ""}
    for iteration in range(limits.max_iterations):
        matches = []
        for cid in sorted(eg.classes):
            cls = eg.classes[cid]
            value_ok = cls.data.pure and not cls.data.volatile
            for node in list(cls.nodes):
                for fn, value in RULES.get(node[0], ()):
                    if value and not value_ok:
                        continue
                    for name, spec in fn(eg, cid, node):
                        if spec is not None and not _has_none(spec):
                            matches.append((cid, name, spec))
            d = cls.data
            if value_ok and d.const is not None and _int_kind(d.kind) and d.kind[1] <= 4 and \
                    not any(n[0] == "int" for n in cls.nodes):
                lit = LIT(d.const, _promoted(d.kind)[2] is False)
                if lit:
                    matches.append((cid, "fold", lit))
        stats["iterations"] = iteration + 1
        if not matches:
            stats["saturated"] = True
            break
        before, merged = eg.size, False
        for cid, name, spec in matches:
            if eg.size >= limits.max_nodes:
                stats["stopped"] = "nodes"
                break
            merged = eg.union(cid, eg.build(spec, name)) or merged
        eg.rebuild()
        if not merged and eg.size == before:
            stats["saturated"] = True
            break
        if stats["stopped"]:
            break
    else:
        stats["stopped"] = stats["stopped"] or "iterations"
    return stats


def _has_none(spec) -> bool:
    if spec is None:
        return True
    if spec[0] == "c":
        return False
    return any(_has_none(k) for k in spec[3])


# ============================================================================== cost model
def _load(kind) -> str:
    if kind and kind[0] == "float":
        return f"load{8 * kind[1]}f"
    if kind and kind[0] == "int":
        return f"load{8 * kind[1]}{'u' if kind[2] is False else 's'}"
    return "load32s"


def _store(kind) -> str:
    if kind and kind[0] == "float":
        return f"store{8 * kind[1]}f"
    if kind and kind[0] == "int":
        return f"store{8 * kind[1]}"
    return "store32"


def _materialize(value: int):
    """Instructions that put a constant in a register: $zero, addiu/ori, lui, lui+ori."""
    u = value & M32
    if u == 0:
        return 0, {}
    s = u - (1 << 32) if u & 0x80000000 else u
    if -0x8000 <= s <= 0x7FFF or u <= 0xFFFF:
        return 1, {"li": 1}
    if u & 0xFFFF == 0:
        return 1, {"lui": 1}
    return 2, {"lui": 1, "or": 1}


def _immediate(op, index, value, mode):
    """Cost of a constant operand of ``op``: free when it fits the instruction's immediate."""
    u = value & M32
    s = u - (1 << 32) if u & 0x80000000 else u
    if op in ("+", "<", ">", "<=", ">=") and -0x8000 <= s <= 0x7FFF:
        return 0, {}
    if op == "-" and index == 1 and -0x7FFF <= s <= 0x8000:
        return 0, {}
    if op in ("&", "|", "^") and u <= 0xFFFF:
        return 0, {}
    if op in ("<<", ">>") and index == 1:
        return 0, {}
    if op in ("==", "!=") and index == 1 and (u == 0 or (mode != "cond" and u <= 0xFFFF)):
        return 0, {}
    if op in ("*", "/", "%") and index == 1 and _log2(u) is not None:
        return 0, {}
    if op == "index" and index == 1:
        return 0, {}
    return _materialize(value)


def _child_modes(op, attr, n, mode):
    if op == "assign":
        return ("lv", "rv")
    if op in ("incdec", "addr"):
        return ("lv",)
    if op == "field" and attr[0] == ".":
        return ("lv",)
    if op in ("&&", "||"):
        return ("cond", "cond")
    if op == "?:":
        return ("cond", "rv", "rv")
    if op == "cond":
        return ("cond",)
    if op == "not" and mode == "cond":
        return ("cond",)
    return ("rv",) * n


class Entry:
    """One extracted term with its estimated cost; ``new`` counts e-nodes a rule created (ties
    go to the form closer to the original)."""
    __slots__ = ("cost", "size", "key", "term", "instr", "mem", "branch", "su", "ops", "rules", "new", "left")

    def __init__(self, term, key, size, instr, mem, branch, su, ops, rules, new, left=0):
        self.term, self.key, self.size, self.new, self.left = term, key, size, new, left
        self.instr, self.mem, self.branch, self.su, self.ops, self.rules = instr, mem, branch, su, ops, rules
        self.cost = instr + 0.25 * mem + 0.5 * branch + 0.02 * su + 0.001 * size


def _rank(entry: Entry):
    # equal cost: fewer constants on the left of a binary operator (``x != 0``, not ``0 != x``: the
    # form C is normally written in), then closer to the original
    return (round(entry.cost, 6), entry.size, entry.left, entry.new, len(entry.rules), entry.key)


def _diverse(entries, n):
    """Up to ``n`` entries in rank order, one per distinct set of rules first."""
    chosen, sets = [], set()
    for e in entries:
        if len(chosen) < n and e.rules not in sets:
            chosen.append(e)
            sets.add(e.rules)
    for e in entries:
        if len(chosen) < n and e not in chosen:
            chosen.append(e)
    return sorted(chosen, key=_rank)


class Extractor:
    """Per e-class and context (``rv`` value, ``lv`` lvalue, ``cond`` branch condition), the
    ``per_class`` cheapest distinct terms, by fixpoint iteration over the reachable classes."""

    def __init__(self, eg: EGraph, limits: Limits):
        self.eg, self.limits = eg, limits

    def _reachable(self, root):
        eg = self.eg
        seen, todo = {}, [(eg.find(root), "rv")]
        while todo:
            item = todo.pop()
            if item in seen:
                continue
            seen[item] = None
            cid, mode = item
            for node in eg.classes[cid].nodes:
                if mode == "lv" and node[0] not in LV_OPS:
                    continue
                for kid, km in zip(node[2], _child_modes(node[0], node[1], len(node[2]), mode)):
                    todo.append((eg.find(kid), km))
        return sorted(seen)

    def run(self, root, root_per: int | None = None) -> dict:
        eg, per = self.eg, self.limits.per_class
        root = eg.find(root)
        order = self._reachable(root)
        table: dict = {}
        for _ in range(self.limits.extract_rounds):
            changed = False
            for cid, mode in order:
                cands = {}
                for node in eg.classes[cid].nodes:
                    if mode == "lv" and node[0] not in LV_OPS:
                        continue
                    for entry in self._node_entries(node, cid, mode, table):
                        old = cands.get(entry.key)
                        if old is None or _rank(entry) < _rank(old):
                            cands[entry.key] = entry
                keep = max(per, root_per or 0) if (cid, mode) == (root, "rv") else per
                best = sorted(cands.values(), key=_rank)[:keep]
                prev = table.get((cid, mode))
                if prev is None or [(e.key, e.cost) for e in prev] != [(e.key, e.cost) for e in best]:
                    if best or prev is not None:
                        table[(cid, mode)] = best
                        changed = True
            if not changed:
                break
        return table

    def _node_entries(self, node, cid, mode, table):
        eg, per = self.eg, self.limits.per_class
        op, attr, kids = node
        options = []
        for kid, km in zip(kids, _child_modes(op, attr, len(kids), mode)):
            kid = eg.find(kid)
            lst = table.get((kid, km))
            if lst and kid == cid and op in ("cast", "pos"):
                lst = [e for e in lst if e.term[0] != op]     # no (T)(T)x after a cast was dropped
            if not lst:
                return []
            kd = eg.data(kid)
            if kd.const is not None and kd.pure and km != "lv":
                options.append([min(lst, key=lambda e: (e.size, e.key))])
            else:
                options.append(lst)
        out = []
        for choice in _combos(options, per):
            out.append(self.combine(node, cid, mode, choice))
        return out

    def entry_of(self, otree, mode="rv") -> Entry:
        """Entry of the original term (``(cid, op, attr, kids)`` from ingestion)."""
        cid, op, attr, kids = otree
        kid_entries = [self.entry_of(k, m) for k, m in zip(kids, _child_modes(op, attr, len(kids), mode))]
        node = (op, attr, tuple(self.eg.find(k[0]) for k in kids))
        return self.combine(node, self.eg.find(cid), mode, kid_entries)

    def combine(self, node, cid, mode, kid_entries) -> Entry:
        eg = self.eg
        op, attr, kids = node
        instr, mem, branch, ops = self._local(node, cid, mode)
        modes = _child_modes(op, attr, len(kids), mode)
        sus, size, rules, new, left = [], 1, set(), 0, 0
        for i, (kid, km, e) in enumerate(zip(kids, modes, kid_entries)):
            new += e.new
            left += e.left
            kd = eg.data(kid)
            if i == 0 and op in BINARY and len(kids) == 2 and kd.const is not None \
                    and eg.data(kids[1]).const is None:
                left += 1
            if kd.const is not None and kd.pure and km != "lv":
                if op in BINARY or op == "index":
                    n, extra = _immediate(op, i, kd.const, mode)
                elif op == "assign" and attr[0] != "=" and i == 1:          # x += 4: addiu
                    n, extra = _immediate(attr[0][:-1], 1, kd.const, mode)
                else:
                    n, extra = _materialize(kd.const)
                instr += n
                for key, value in extra.items():
                    ops[key] = ops.get(key, 0) + value
                sus.append(1 if n else 0)
            else:
                instr += e.instr
                mem += e.mem
                branch += e.branch
                for key, value in e.ops.items():
                    ops[key] = ops.get(key, 0) + value
                sus.append(e.su)
            size += e.size
            rules |= e.rules
        why = eg.why.get(node)
        if why:
            rules.add(why)
            new += 1
        su = self._su(node, sus)
        term = (op, attr, tuple(e.term for e in kid_entries))
        key = f"{op}{attr!r}(" + ",".join(e.key for e in kid_entries) + ")"
        return Entry(term, key, size, instr, mem, branch, su, ops, frozenset(rules), new, left)

    def _su(self, node, sus):
        """Sethi-Ullman register need of the tree (a register-pressure proxy)."""
        op = node[0]
        if not sus:
            if op == "var":
                info = self.eg.vars.get(node[1])
                return 0 if info is not None and info.where == "enum" else 1
            return 1
        if op == "let":
            return max(sus) + 1
        if op in WRAPPERS:
            return max(sus)
        if len(sus) == 2 and op in BINARY:
            a, b = sus
            return a + 1 if a == b and a > 0 else max(a, b, 1)
        return max(max(sus), 1)

    def _lvalue_reg(self, cid) -> bool:
        for node in self.eg.classes[self.eg.find(cid)].nodes:
            if node[0] == "var":
                info = self.eg.vars.get(node[1])
                if info is not None and info.where == "reg":
                    return True
        return False

    def _local(self, node, cid, mode):
        """(instructions, memory accesses, branches, opcode classes) of ``node`` itself."""
        eg = self.eg
        op, attr, kids = node
        d = eg.data(cid)
        K = [eg.data(k) for k in kids]
        ops: dict = {}
        instr = mem = br = 0

        def put(name, n=1):
            ops[name] = ops.get(name, 0) + n
        floating = bool(d.kind) and d.kind[0] == "float"
        if op == "var":
            info = eg.vars.get(attr)
            where = info.where if info is not None else "global"
            if where == "mem":
                if mode != "lv":
                    instr, mem = 1, 1
                    put(_load(d.kind))
            elif where == "global":
                instr = 1
                put("lui")
                if mode != "lv":
                    instr, mem = 2, 1
                    put(_load(d.kind))
            elif where == "larray":
                instr = 1
                put("add")
            elif where in ("garray", "func"):
                instr = 2
                put("lui")
                put("add")
        elif op == "int":
            instr, extra = _materialize(attr[0])
            ops.update(extra)
        elif op in ("lit", "atom"):
            instr = 1 if op == "atom" else 2
        elif op == "neg":
            instr = 1
            put("fpu" if floating else "sub")
        elif op == "bnot":
            instr = 1
            put("nor")
        elif op == "not":
            if mode != "cond":
                instr = 1
                put("ltu")
        elif op in ("deref", "index", "field"):
            if op == "index" and K[1].const is None:
                if not (_int_kind(d.kind) and d.kind[1] == 1):
                    instr += 1
                    put("shl")
                instr += 1
                put("add")
            if mode != "lv" and not (d.kind and d.kind[0] in ("array", "struct", "func")):
                instr += 1
                mem += 1
                put(_load(d.kind))
        elif op == "addr":
            instr = 1
            put("add")
        elif op == "cast":
            tk, xk = eg.cast_kind(attr), K[0].kind
            if tk and xk:
                if tk[0] == "int" and xk[0] == "int" and tk[1] < xk[1]:
                    if tk[2] is False:
                        instr = 1
                        put("and")
                    else:
                        instr = 2
                        put("shl")
                        put("sar")
                elif tk[0] == "float" and xk[0] == "int":
                    instr = 2
                    put("cvt")
                elif tk[0] == "int" and xk[0] == "float":
                    instr = 4
                    put("cvt")
                elif tk[0] == "float" and xk[0] == "float" and tk[1] != xk[1]:
                    instr = 1
                    put("cvt")
        elif op in ("+", "-"):
            if floating:
                instr = 1
                put("fpu")
            else:
                instr = 1
                put("add" if op == "+" else "sub")
                if d.kind and d.kind[0] == "ptr":
                    for k in K:
                        if _int_kind(k.kind) and k.const is None:
                            instr += 1
                            put("shl")
                elif len(K) == 2 and all(k.kind and k.kind[0] in ("ptr", "array") for k in K):
                    instr += 1
                    put("sar")
        elif op in ("&", "|", "^"):
            instr = 1
            put(_OPCLASS[op])
        elif op == "<<":
            instr = 1
            put("shl")
        elif op == ">>":
            instr = 1
            put("shr" if _promoted(d.kind) and _promoted(d.kind)[2] is False else "sar")
        elif op == "*":
            if floating:
                instr = 1
                put("fpu")
            elif any(_log2(k.const & M32 if k.const is not None else None) is not None for k in K):
                instr = 1
                put("shl")
            else:
                instr = 2
                put("mul")
        elif op in ("/", "%"):
            if floating:
                instr = 1
                put("fpu")
            else:
                unsigned = bool(d.kind) and _promoted(d.kind)[2] is False
                c = K[1].const
                if c is not None and _log2(c & M32) is not None:
                    if unsigned:
                        instr = 1
                        put("shr" if op == "/" else "and")
                    else:
                        instr = 3 if op == "/" else 5
                        put("sar")
                        put("shr")
                        put("add")
                else:
                    instr, br = (4, 1) if unsigned else (6, 2)
                    put("divu" if unsigned else "div")
                    put("branch", br)
        elif op in _COMPARE:
            if any(k.kind and k.kind[0] == "float" for k in K):
                instr = 2 if mode == "cond" else 4
                put("fpu")
            else:
                common = _common(K[0].kind, K[1].kind)
                unsigned = (common is not None and common[2] is False) or \
                    any(k.kind and k.kind[0] in ("ptr", "array") for k in K)
                lt = "ltu" if unsigned else "lt"
                zero = K[1].const == 0 or K[0].const == 0
                if mode == "cond":
                    if op not in ("==", "!=") and not (zero and not unsigned):
                        instr = 1
                        put(lt)
                elif op in ("<", ">"):
                    instr = 1
                    put(lt)
                elif op in ("<=", ">="):
                    instr = 2
                    put(lt)
                    put("xor")
                else:
                    instr = 1 if zero else 2
                    put("ltu")
                    if not zero:
                        put("xor")
        elif op in _LOGICAL:
            if mode == "cond":
                instr, br = 1, 1
                put("branch")
            else:
                instr, br = 4, 2
                put("branch", 2)
        elif op == "?:":
            instr, br = 2, 1
            put("branch")
        elif op == "assign":
            lk = K[0].kind
            reg = self._lvalue_reg(kids[0])
            if not reg:
                instr, mem = 1, 1
                put(_store(lk))
            elif _int_kind(lk) and lk[1] < 4 and _int_kind(K[1].kind) and K[1].kind[1] > lk[1]:
                if lk[2] is False:
                    instr = 1
                    put("and")
                else:
                    instr = 2
                    put("shl")
                    put("sar")
            if attr[0] != "=":
                if not reg:
                    instr += 1
                    mem += 1
                    put(_load(lk))
                base = attr[0][:-1]
                instr += 2 if base in ("*", "/", "%") else 1
                put(_OPCLASS.get(base, "add"))
        elif op == "incdec":
            instr = 1
            put("add")
            if not self._lvalue_reg(kids[0]):
                instr, mem = 3, 2
                put(_load(K[0].kind))
                put(_store(K[0].kind))
        elif op == "call":
            instr = 2
            put("call")
            extra = max(0, len(kids) - 5)
            instr += extra
            mem += extra
        elif op == "cond":
            instr, br = 1, 1
            put("branch")
        elif op == "ifelse":
            instr, br = 2, 1
            put("branch")
        elif op == "let":
            instr = 1
        return instr, mem, br, ops


def _combos(options, per):
    """Child choices: the full product for up to two children, else one child varied at a time."""
    if not options:
        return [()]
    if len(options) <= 2:
        out = [()]
        for lst in options:
            out = [prefix + (e,) for prefix in out for e in lst[:per]]
        return out
    base = tuple(lst[0] for lst in options)
    out, seen = [base], {tuple(e.key for e in base)}
    for i, lst in enumerate(options):
        for e in lst[1:per]:
            choice = base[:i] + (e,) + base[i + 1:]
            key = tuple(x.key for x in choice)
            if key not in seen:
                seen.add(key)
                out.append(choice)
    return out


# ============================================================================== target profile
_SPECIAL = {0: "shl", 2: "shr", 3: "sar", 4: "shl", 6: "shr", 7: "sar", 9: "call", 24: "mul", 25: "mul",
            26: "div", 27: "divu", 32: "add", 33: "add", 34: "sub", 35: "sub", 36: "and", 37: "or", 38: "xor",
            39: "nor", 42: "lt", 43: "ltu"}
_IMMEDIATE = {10: "lt", 11: "ltu", 12: "and", 14: "xor", 15: "lui"}
_LOADS = {32: "load8s", 33: "load16s", 35: "load32s", 36: "load8u", 37: "load16u", 39: "load32u",
          49: "load32f", 53: "load64f", 55: "load64s"}
_STORES = {40: "store8", 41: "store16", 43: "store32", 57: "store32f", 61: "store64f", 63: "store64"}
_SAVED = frozenset((*range(16, 24), 30, 31))


def target_profile(target) -> dict[str, int]:
    """Opcode-class multiset of a target: big-endian code bytes, a sequence of instruction words,
    or an ``ir.Fingerprint`` (its operation and memory counters).  Moves, nops, the frame
    adjustment and callee-saved register saves and restores are left out."""
    if hasattr(target, "ops") and hasattr(target, "memory"):
        out: dict[str, int] = {}
        for counter in (target.ops, target.memory):
            for key, value in counter.items():
                if key in VOCABULARY:
                    out[key] = out.get(key, 0) + value
        branches = sum(v for k, v in target.effects.items() if k.startswith("branch(")) if hasattr(target, "effects") else 0
        if branches:
            out["branch"] = out.get("branch", 0) + branches
        return out
    if isinstance(target, (bytes, bytearray)):
        target = struct.unpack(f">{len(target) // 4}I", bytes(target[:len(target) & ~3]))
    out = {}
    for word in target:
        w = int(word) & M32
        if w == 0:
            continue
        op, rs, rt, fn = w >> 26, (w >> 21) & 31, (w >> 16) & 31, w & 63
        name = None
        if op == 0:
            name = _SPECIAL.get(fn)
            if fn in (32, 33, 37) and (rs == 0 or rt == 0):
                name = None                                   # a move
        elif op == 1 or 4 <= op <= 7 or 20 <= op <= 23 or (op == 17 and rs == 8):
            name = "branch"
        elif op == 3:
            name = "call"
        elif op in (8, 9):
            name = None if rt == 29 else ("li" if rs == 0 else "add")
        elif op == 13:
            name = "li" if rs == 0 else "or"
        elif op in _IMMEDIATE:
            name = _IMMEDIATE[op]
        elif op == 17:
            name = "cvt" if fn in (32, 33, 36, 37) or 8 <= fn <= 15 else ("fpu" if rs in (16, 17) else None)
        elif op in _LOADS or op in _STORES:
            frame = rs == 29 and ((op in (35, 43) and rt in _SAVED) or (op in (53, 61) and rt >= 20))
            name = None if frame else (_LOADS.get(op) or _STORES.get(op))
        if name:
            out[name] = out.get(name, 0) + 1
    return out


def distance(ops: dict, target: dict, base: dict | None = None, original: dict | None = None) -> int:
    """L1 distance of an estimate to the target's opcode classes.  With ``base`` (the current
    build) and ``original`` (the estimate of the current source), the estimated change is compared
    with the change the target asks for, so what the model does not see cancels out."""
    keys = sorted(set(ops) | set(target) | set(base or {}) | set(original or {}))
    total = 0
    for k in keys:
        if k not in VOCABULARY:
            continue
        if base is not None:
            want = target.get(k, 0) - base.get(k, 0)
            have = ops.get(k, 0) - (original or {}).get(k, 0)
        else:
            want, have = target.get(k, 0), ops.get(k, 0)
        total += abs(want - have)
    return total


# ============================================================================== function level
_EXPRESSIONS = (c_ast.Assignment, c_ast.UnaryOp, c_ast.BinaryOp, c_ast.FuncCall, c_ast.Cast, c_ast.TernaryOp,
                c_ast.ID, c_ast.Constant, c_ast.ArrayRef, c_ast.StructRef)


@dataclass
class EUnit:
    kind: str            # stmt forstmt ret cond val init ifelse
    root: object         # AST: expression, Return, Decl or If
    start: int
    end: int
    line: int
    hoist: bool          # sits directly in a block: may become several statements
    sn: object = None


@dataclass
class Alt:
    entry: Entry
    edits: list
    decls: list
    label: str
    rules: tuple


@dataclass
class Candidate:
    label: str
    source: str
    rules: tuple
    review: bool
    vector: dict = field(default_factory=dict)


class Function:
    """Ingestion, rendering and per-statement solving for one function."""

    def __init__(self, types, analysis, dialect: Dialect, limits: Limits, words, ctx=None):
        self.T, self.an, self.dialect, self.limits, self.ctx = types, analysis, dialect, limits, ctx
        self.words = set(words)
        self.vars: dict[str, VarInfo] = {}
        self.casts: dict[str, object] = {}
        self.atoms: list = []
        self.atom_index: dict[str, int] = {}
        self.serial = 0

    # -- names and variables
    def fresh(self, base: str) -> str:
        name, k = base, 2
        while name in self.words:
            name, k = f"{base}{k}", k + 1
        self.words.add(name)
        return name

    def var(self, name: str) -> VarInfo:
        info = self.vars.get(name)
        if info is None:
            T, an = self.T, self.an
            t = T.var_type(name)
            k = T.kind(t) if t is not None else None
            if name in an.locals:
                where = "reg" if an.reg(name) else "mem"
                if k and k[0] == "array":
                    where = "larray"
            elif name in T.env.enumerators and name not in T.vars:
                where = "enum"
            else:
                where = "garray" if k and k[0] == "array" else "global"
            if k and k[0] == "func":
                where = "func"
            volatile = T.volatile(t) if t is not None else (an.has_volatile_text or T.env.volatile_fields)
            info = VarInfo(where, t, k, bool(volatile))
            self.vars[name] = info
        return info

    def graph(self, hoist: bool = False) -> EGraph:
        eg = EGraph(self.T, self.dialect, self.vars, self.casts)
        eg.block_ok = hoist
        return eg

    # -- AST -> e-graph (returns (class id, original tree))
    def _serial(self) -> int:
        self.serial += 1
        return self.serial

    def atom(self, eg: EGraph, e):
        eff = self.an.expr(e)
        text = expr_text(e)
        pure = not eff.side and not eff.volatile
        index = self.atom_index.get(text) if pure else None
        if index is None:
            index = len(self.atoms)
            self.atoms.append(e)
            if pure:
                self.atom_index[text] = index
        tnode = self.T.typeof(e)
        d = Data(tnode=tnode, kind=self.T.kind(tnode) if tnode is not None else None, pure=not eff.side,
                 volatile=eff.volatile, memory=bool(eff.memr) or eff.call, trap=bool(eff.memr) or eff.call)
        cid = eg.add("atom", index, (), d)
        return cid, (cid, "atom", index, ())

    def expr(self, eg: EGraph, e):
        if isinstance(e, c_ast.ID):
            self.var(e.name)
            op, attr, kids = "var", e.name, []
        elif isinstance(e, c_ast.Constant):
            value = astforms._int_value(e)
            if value is not None:
                unsigned = "u" in e.value.lower() or value > 0x7FFFFFFF
                op, attr, kids = "int", ((value & M32) if unsigned else value, e.value, e.type), []
            elif e.type == "string":
                return self.atom(eg, e)
            else:
                op, attr, kids = "lit", (e.type, e.value), []
        elif isinstance(e, c_ast.UnaryOp):
            if e.op in _UNARY:
                op, attr, kids = _UNARY[e.op], None, [self.expr(eg, e.expr)]
            elif e.op in ("++", "--", "p++", "p--"):
                op, attr, kids = "incdec", (e.op, self._serial()), [self.expr(eg, e.expr)]
            else:
                return self.atom(eg, e)
        elif isinstance(e, c_ast.BinaryOp) and e.op in BINARY:
            op, attr, kids = e.op, None, [self.expr(eg, e.left), self.expr(eg, e.right)]
        elif isinstance(e, c_ast.Cast):
            key = expr_text(e.to_type)
            self.casts.setdefault(key, e.to_type)
            op, attr, kids = "cast", key, [self.expr(eg, e.expr)]
        elif isinstance(e, c_ast.ArrayRef):
            op, attr, kids = "index", None, [self.expr(eg, e.name), self.expr(eg, e.subscript)]
        elif isinstance(e, c_ast.StructRef) and isinstance(e.field, c_ast.ID):
            op, attr, kids = "field", (e.type, e.field.name), [self.expr(eg, e.name)]
        elif isinstance(e, c_ast.TernaryOp):
            op, attr, kids = "?:", None, [self.expr(eg, e.cond), self.expr(eg, e.iftrue), self.expr(eg, e.iffalse)]
        elif isinstance(e, c_ast.Assignment):
            op, attr, kids = "assign", (e.op, self._serial()), [self.expr(eg, e.lvalue), self.expr(eg, e.rvalue)]
        elif isinstance(e, c_ast.FuncCall):
            args = list(e.args.exprs) if e.args is not None else []
            if any(isinstance(a, c_ast.Typename) for a in args):
                return self.atom(eg, e)
            op, attr, kids = "call", self._serial(), [self.expr(eg, e.name)] + [self.expr(eg, a) for a in args]
        else:
            return self.atom(eg, e)
        ids = tuple(k[0] for k in kids)
        d = eg.derive(eg.canon((op, attr, ids)), self.T.typeof(e))
        eff = self.an.expr(e)
        d.pure = d.pure and not eff.side
        d.volatile = d.volatile or eff.volatile
        d.memory = d.memory or bool(eff.memr)
        cid = eg.add(op, attr, ids, d)
        return cid, (cid, op, attr, tuple(k[1] for k in kids))

    def ingest(self, eg: EGraph, unit: EUnit):
        r = unit.root
        if unit.kind in ("stmt", "forstmt"):
            c, t = self.expr(eg, r)
            attr = "for" if unit.kind == "forstmt" else None
            root = eg.add("stmt", attr, (c,))
            return root, (root, "stmt", attr, (t,))
        if unit.kind == "ret":
            c, t = self.expr(eg, r.expr)
            root = eg.add("ret", None, (c,))
            return root, (root, "ret", None, (t,))
        if unit.kind in ("cond", "val"):
            c, t = self.expr(eg, r)
            root = eg.add(unit.kind, None, (c,))
            return root, (root, unit.kind, None, (t,))
        if unit.kind == "init":
            c, t = self.expr(eg, r.init)
            key = eg.cast_for(r.type)
            root = eg.add("init", key, (c,))
            return root, (root, "init", key, (t,))
        if unit.kind == "ifelse":
            c, ct = self.expr(eg, r.cond)
            cw = eg.add("cond", None, (c,))
            parts, trees = [cw], [(cw, "cond", None, (ct,))]
            for branch in (r.iftrue, r.iffalse):
                a, at = self.expr(eg, _single_assignment(branch))
                s = eg.add("stmt", None, (a,))
                parts.append(s)
                trees.append((s, "stmt", None, (at,)))
            root = eg.add("ifelse", None, tuple(parts))
            return root, (root, "ifelse", None, tuple(trees))
        raise ValueError(unit.kind)

    # -- temporaries
    def add_let(self, eg: EGraph, root: int, otree, unit: EUnit) -> bool:
        """``t = e; S[e := t]`` for the largest pure sub-expression ``e`` repeated in statement S."""
        if unit.kind not in ("stmt", "ret") or not unit.hoist:
            return False
        top = otree[3][0]
        if top[1] in IMPURE:
            if not all(_clean(eg, k[0]) for k in top[3]):
                return False
        elif not _clean(eg, top[0]):
            return False
        counts, risky, sizes = {}, {}, {}

        def walk(t, mode, conditional, depth=0):
            cid, op, attr, kids = t
            c = eg.find(cid)
            size = 1
            for i, (k, km) in enumerate(zip(kids, _child_modes(op, attr, len(kids), mode))):
                cond = conditional or (op in _LOGICAL and i == 1) or (op == "?:" and i > 0)
                size += walk(k, km, cond, depth + 1)
            if mode != "lv" and op not in LEAVES and op not in WRAPPERS and op not in IMPURE and depth > 1:
                d = eg.data(c)
                if d.pure and not d.volatile and d.kind and d.kind[0] in ("int", "float", "ptr") and \
                        self.T.decl_text(d.tnode, "x") is not None:
                    counts[c] = counts.get(c, 0) + 1
                    risky[c] = risky.get(c, False) or (conditional and d.trap)
                    sizes[c] = max(sizes.get(c, 0), size)
            return size
        walk(otree, "rv", False)
        repeated = [c for c in counts if counts[c] >= 2 and not risky[c]]
        if not repeated:
            return False
        target = min(repeated, key=lambda c: (-sizes[c], c))
        name = self.fresh("temp_eg")
        d = eg.data(target)
        self.vars[name] = VarInfo("reg", d.tnode, d.kind, False)

        def spec(t, mode):
            cid, op, attr, kids = t
            if mode != "lv" and eg.find(cid) == target:
                return N("var", name)
            modes = _child_modes(op, attr, len(kids), mode)
            return N(op, attr, *[spec(k, km) for k, km in zip(kids, modes)])
        body = spec(otree, "rv")
        eg.union(root, eg.build(N("let", name, C(target), body), "temp"))
        eg.rebuild()
        return True

    # -- e-graph terms -> C
    def ast(self, term):
        op, attr, kids = term
        if op == "var":
            return c_ast.ID(attr)
        if op == "int":
            node = c_ast.Constant(attr[2], attr[1])
            return c_ast.UnaryOp("-", node) if attr[0] < 0 else node
        if op == "lit":
            return c_ast.Constant(attr[0], attr[1])
        if op == "atom":
            return copy.deepcopy(self.atoms[attr])
        if op in _UNARY_BACK:
            return c_ast.UnaryOp(_UNARY_BACK[op], self.ast(kids[0]))
        if op in BINARY:
            return c_ast.BinaryOp(op, self.ast(kids[0]), self.ast(kids[1]))
        if op == "cast":
            return c_ast.Cast(copy.deepcopy(self.casts[attr]), self.ast(kids[0]))
        if op == "index":
            return c_ast.ArrayRef(self.ast(kids[0]), self.ast(kids[1]))
        if op == "field":
            return c_ast.StructRef(self.ast(kids[0]), attr[0], c_ast.ID(attr[1]))
        if op == "?:":
            return c_ast.TernaryOp(*[self.ast(k) for k in kids])
        if op == "assign":
            return c_ast.Assignment(attr[0], self.ast(kids[0]), self.ast(kids[1]))
        if op == "incdec":
            return c_ast.UnaryOp(attr[0], self.ast(kids[0]))
        if op == "call":
            args = [self.ast(k) for k in kids[1:]]
            return c_ast.FuncCall(self.ast(kids[0]), c_ast.ExprList(args) if args else None)
        raise ValueError(op)

    def text(self, term, unit_kind: str = "stmt", indent: str = "") -> str | None:
        """C text of a unit-level term (a statement, a condition, ``if``/``else`` or a temporary)."""
        op, attr, kids = term
        if op == "stmt":
            e = self.ast(kids[0])
            return expr_text(e) if unit_kind == "forstmt" or attr == "for" else simple_stmt_text(e)
        if op == "ret":
            return simple_stmt_text(c_ast.Return(self.ast(kids[0])))
        if op in ("cond", "val"):
            return expr_text(self.ast(kids[0]))
        if op == "ifelse":
            cond, s1, s2 = kids
            if cond[0] != "cond" or s1[0] != "stmt" or s2[0] != "stmt":
                return None
            inner = indent + "    "
            return (f"if ({expr_text(self.ast(cond[2][0]))}) {{\n{inner}{simple_stmt_text(self.ast(s1[2][0]))}\n"
                    f"{indent}}} else {{\n{inner}{simple_stmt_text(self.ast(s2[2][0]))}\n{indent}}}")
        if op == "let":
            body = self.text(kids[1], unit_kind, indent)
            if body is None:
                return None
            return f"{attr} = {expr_text(self.ast(kids[0]))};\n{indent}{body}"
        return None

    def render(self, unit: EUnit, term):
        """(edits, declarations) replacing ``unit`` with ``term``; None when it cannot be placed."""
        op = term[0]
        indent = self.ctx.indent(unit.sn) if self.ctx is not None and unit.sn is not None else ""
        if op == "let":
            info = self.vars.get(term[1])
            decl = self.T.decl_text(info.tnode, term[1]) if info is not None else None
            if decl is None or not unit.hoist or unit.kind not in ("stmt", "ret"):
                return None
            text = self.text(term, unit.kind, indent)
            return (None if text is None else ([(unit.start, unit.end, text)], [decl]))
        if op == "ifelse" and not (unit.kind == "ifelse" or (unit.kind == "stmt" and unit.hoist)):
            return None
        if unit.kind == "ifelse" and op not in ("ifelse", "stmt"):
            return None
        if op == "init":
            decl = copy.copy(unit.root)
            decl.init = self.ast(term[2][0])
            return [(unit.start, unit.end, simple_stmt_text(decl))], []
        text = self.text(term, unit.kind, indent)
        return None if text is None else ([(unit.start, unit.end, text)], [])

    # -- one statement
    def solve(self, unit: EUnit):
        """(alternatives with the original first, e-node count)."""
        eg = self.graph(unit.hoist)
        root, otree = self.ingest(eg, unit)
        eg.rebuild()
        self.add_let(eg, root, otree, unit)
        saturate(eg, self.limits)
        extractor = Extractor(eg, self.limits)
        table = extractor.run(eg.find(root), root_per=4 * self.limits.alternatives)
        original = extractor.entry_of(otree)
        rendered = self.render(unit, original.term)
        if rendered is None:
            return [], eg.size
        placed, seen = [], {repr(rendered)}
        for entry in table.get((eg.find(root), "rv"), []):
            out = self.render(unit, entry.term)
            if out is None or repr(out) in seen:
                continue
            seen.add(repr(out))
            placed.append((entry, out))
        alts = [Alt(original, [], [], "", ())]
        chosen = _diverse([entry for entry, _out in placed], self.limits.alternatives)
        for entry, out in placed:
            if any(entry is c for c in chosen):
                rules = tuple(sorted(entry.rules))
                label = f"{unit.kind} L{unit.line}: {'+'.join(rules) or 'reprint'}"
                alts.append(Alt(entry, out[0], out[1], label, rules))
        return alts, eg.size


def _single_assignment(stmt):
    if isinstance(stmt, c_ast.Compound) and len(stmt.block_items or []) == 1:
        stmt = stmt.block_items[0]
    if isinstance(stmt, c_ast.Assignment) and stmt.op == "=":
        return stmt
    return None


def units(ctx) -> list[EUnit]:
    """The rewritable places of a function (``astforms.Ctx``): printable expression statements,
    returns, initialised declarations, conditions, for-heads and ``if``/``else`` assignments."""
    prep, out = ctx.prep, []
    for u in ctx.units:
        if not u.printable:
            continue
        sn, root = u.node, u.root
        hoist = sn.parent is not None and sn.parent.kind == "compound" and u.kind == "stmt"
        if u.kind == "stmt":
            if isinstance(root, c_ast.Return):
                if root.expr is None:
                    continue
                kind = "ret"
            elif isinstance(root, c_ast.Decl):
                if root.init is None or isinstance(root.init, c_ast.InitList) or len(sn.group) != 1 or \
                        isinstance(root.type, (c_ast.ArrayDecl, c_ast.FuncDecl)):
                    continue
                kind, hoist = "init", False
            elif isinstance(root, _EXPRESSIONS):
                kind = "stmt"
            else:
                continue
        elif u.kind == "cond":
            kind = "val" if sn.kind == "switch" else "cond"
        elif u.kind == "forpart":
            kind = "cond" if u.index == 1 else "forstmt"
        else:
            continue
        out.append(EUnit(kind, root, u.start, u.end, prep.line(u.start), hoist, sn))
    for sn in ctx.nodes:
        if sn.kind == "if" and sn.has_else and isinstance(sn.ast, c_ast.If) and prep.printable(sn) and \
                _single_assignment(sn.ast.iftrue) is not None and _single_assignment(sn.ast.iffalse) is not None:
            out.append(EUnit("ifelse", sn.ast, sn.start, sn.end, prep.line(sn.start),
                             sn.parent is not None and sn.parent.kind == "compound", sn))
    out.sort(key=lambda u: (u.start, -u.end, u.kind))
    return out


def _overlaps(span, spans) -> bool:
    return any(not (span[1] <= s or e <= span[0]) for s, e in spans)


class _Plan:
    """Whole-function choices (one alternative per statement) and their cost vectors."""

    def __init__(self, solved, target, base, limits):
        self.solved, self.target, self.base, self.limits = solved, target, base, limits
        self.original_ops = self._ops(tuple(0 for _ in solved))

    def normal(self, choice):
        out, spans = list(choice), []
        for i, (unit, _alts) in enumerate(self.solved):
            if out[i]:
                span = (unit.start, unit.end)
                if _overlaps(span, spans):
                    out[i] = 0
                else:
                    spans.append(span)
        return tuple(out)

    def _ops(self, choice):
        total: dict = {}
        for (unit, alts), j in zip(self.solved, choice):
            for k, v in alts[j].entry.ops.items():
                total[k] = total.get(k, 0) + v
        return total

    def vector(self, choice) -> dict:
        entries = [alts[j].entry for (unit, alts), j in zip(self.solved, choice)]
        out = {"instructions": sum(e.instr for e in entries), "registers": max((e.su for e in entries), default=0),
               "memory": sum(e.mem for e in entries), "branches": sum(e.branch for e in entries),
               "cost": round(sum(e.cost for e in entries), 3)}
        if self.target is not None:
            out["target_distance"] = distance(self._ops(choice), self.target, self.base,
                                              self.original_ops if self.base is not None else None)
        return out

    def states(self):
        solved = self.solved
        orig = tuple(0 for _ in solved)
        states = {}
        for i, (_unit, alts) in enumerate(solved):
            for j in range(1, len(alts)):
                states[self.normal(orig[:i] + (j,) + orig[i + 1:])] = None
        best = tuple(min(range(len(alts)), key=lambda j: (alts[j].entry.cost, j)) for _u, alts in solved)
        states[self.normal(best)] = None
        if self.target is not None:
            current = orig
            d = self.vector(current)["target_distance"]
            for _ in range(self.limits.greedy_steps):
                found = None
                for i, (_unit, alts) in enumerate(solved):
                    for j in range(len(alts)):
                        if j == current[i]:
                            continue
                        cand = self.normal(current[:i] + (j,) + current[i + 1:])
                        if cand == current:
                            continue
                        v = self.vector(cand)
                        key = (v["target_distance"], v["cost"], cand)
                        if v["target_distance"] < d and (found is None or key < found[0]):
                            found = (key, cand)
                if found is None:
                    break
                current, d = found[1], found[0][0]
                states[current] = None
        states.pop(orig, None)
        return list(states)


def _dominates(a: dict, b: dict) -> bool:
    keys = [k for k in a if k != "cost"]
    return all(a[k] <= b[k] for k in keys) and any(a[k] < b[k] for k in keys)


def _context(source, name, includes):
    try:
        return astforms.context(source, name, includes)
    except (PrepError, RecursionError):
        return None
    except Exception:                     # pragma: no cover - a parser problem never stops the search
        return None


def candidates(c_source: str, function_name: str, target_words=None, k: int = 8, limits: Limits | None = None, *,
               base_words=None, dialect: Dialect | None = None, includes=None, check: bool = True) -> list[Candidate]:
    """Up to ``k`` distinct whole-source variants of ``function_name``: the Pareto set of the cost
    vectors first (target distance, then estimated cost), then the cheapest of the rest."""
    started = time.perf_counter()
    limits, dialect = limits or Limits(), dialect or Dialect()
    STATS["calls"] += 1
    ctx = _context(c_source, function_name, includes)
    if ctx is None:
        return []
    fn = Function(ctx.types, ctx.an, dialect, limits, ctx.words, ctx)
    solved, total = [], 0
    for unit in units(ctx)[:limits.max_units]:
        if total >= limits.max_total_nodes:
            break
        try:
            alts, size = fn.solve(unit)
        except Exception:                  # an unusual construct: leave the statement alone
            continue
        total += size
        STATS["units"] += 1
        if len(alts) > 1:
            solved.append((unit, alts))
    STATS["nodes"] += total
    out: list[Candidate] = []
    if solved:
        target = target_profile(target_words) if target_words is not None else None
        base = target_profile(base_words) if (base_words is not None and target is not None) else None
        plan = _Plan(solved, target, base, limits)
        states = plan.states()
        vectors = {s: plan.vector(s) for s in states}
        front = [s for s in states if not any(_dominates(vectors[o], vectors[s]) for o in states if o != s)]

        def order(s):
            v = vectors[s]
            return (v.get("target_distance", 0), v["cost"], s)
        ranked = sorted(front, key=order) + sorted([s for s in states if s not in front], key=order)
        seen = {c_source}
        for state in ranked:
            if len(out) >= k:
                break
            edits, decls, labels, rules = [], [], [], set()
            for (unit, alts), j in zip(solved, state):
                if j:
                    alt = alts[j]
                    edits += alt.edits
                    decls += alt.decls
                    labels.append(alt.label)
                    rules |= set(alt.rules)
            if decls:
                body = ctx.prep.body
                indent = ctx.indent(body.children[0]) if body.children else "    "
                edits.append(ctx.decl_insert(body, ("\n" + indent).join(dict.fromkeys(decls))))
            try:
                text = apply_edits(c_source, edits)
            except ValueError:
                continue
            if text in seen:
                continue
            seen.add(text)
            if check and not astforms._check(text, function_name, includes):
                continue
            out.append(Candidate("egraph: " + "; ".join(labels), text, tuple(sorted(rules)),
                                 any("~wrap" in r for r in rules), vectors[state]))
    STATS["candidates"] += len(out)
    STATS["seconds"] += time.perf_counter() - started
    return out


def variants(c_source: str, function_name: str, target_words=None, k: int = 8, limits: Limits | None = None,
             **options) -> list[str]:
    """Distinct whole-source texts (only ``function_name`` differs), best first."""
    return [c.source for c in candidates(c_source, function_name, target_words, k, limits, **options)]


def set_target(name: str, target_words, base_words=None) -> None:
    TARGETS[name] = {"target": target_words, "base": base_words}


def iterate_variants(source: str, name: str) -> list:
    """The hook ``iterate.variants`` calls for the ``egraph`` family (or with ``N64DK_EGRAPH=1``)."""
    from .iterate import Variant
    known = TARGETS.get(name) or {}
    k = int(os.environ.get("N64DK_EGRAPH_K", "8"))
    return [Variant(c.label, "egraph", c.source, c.review)
            for c in candidates(source, name, known.get("target"), k, base_words=known.get("base"))]


# ============================================================================== preprocessor-free entry points
def light_context(source: str, name: str):
    """(types, analysis, words) of ``name`` parsed by pycparser alone, without the preprocessor
    (tests, and sources that need none)."""
    ast, extra = astprep._parse_with_typedef_retry(source, [])
    env = astprep._build_env(ast, extra)
    fns = [e for e in ast.ext if isinstance(e, c_ast.FuncDef) and e.decl.name == name]
    if len(fns) != 1:
        raise PrepError("function-not-found", name)
    prep = SimpleNamespace(env=env, fn=fns[0], source=source, fstart=0, fend=len(source))
    types = astforms.Types(prep)
    return types, astforms.Analysis(prep, types), set(re.findall(r"[A-Za-z_]\w*", source)), fns[0]


def statement_graph(source: str, name: str, which: int = 0, limits: Limits | None = None,
                    dialect: Dialect | None = None, saturated: bool = True):
    """(Function, e-graph, root class, original tree) for the ``which``-th statement of the
    function's outer block (an expression statement, a ``return``, an ``if`` condition or an
    ``if``/``else`` assignment pair)."""
    types, analysis, words, fdef = light_context(source, name)
    limits, dialect = limits or Limits(), dialect or Dialect()
    fn = Function(types, analysis, dialect, limits, words)
    items = [s for s in (fdef.body.block_items or []) if not isinstance(s, c_ast.Decl)]
    stmt = items[which]
    if isinstance(stmt, c_ast.Return):
        unit = EUnit("ret", stmt, 0, 0, 0, True)
    elif isinstance(stmt, c_ast.If) and stmt.iffalse is not None and _single_assignment(stmt.iftrue) is not None \
            and _single_assignment(stmt.iffalse) is not None:
        unit = EUnit("ifelse", stmt, 0, 0, 0, True)
    elif isinstance(stmt, c_ast.If):
        unit = EUnit("cond", stmt.cond, 0, 0, 0, False)
    else:
        unit = EUnit("stmt", stmt, 0, 0, 0, True)
    eg = fn.graph(unit.hoist)
    root, otree = fn.ingest(eg, unit)
    eg.rebuild()
    if saturated:
        fn.add_let(eg, root, otree, unit)
        saturate(eg, limits)
    return fn, eg, root, otree


def plain(otree):
    """``(op, attr, kids)`` of an original tree from ingestion (class ids dropped)."""
    _cid, op, attr, kids = otree
    return (op, attr, tuple(plain(k) for k in kids))


def parse_term(fn: Function, text: str):
    """``(op, attr, kids)`` of a C expression (for ``EGraph.represents``)."""
    names = list(fn.T.env.typedef_names)
    ast, _ = astprep._parse_with_typedef_retry("void __eg_probe(void) { " + text + "; }", names)
    return _term(ast.ext[-1].body.block_items[0])


def _term(e):
    if isinstance(e, c_ast.ID):
        return ("var", e.name, ())
    if isinstance(e, c_ast.Constant):
        value = astforms._int_value(e)
        if value is not None:
            return ("int", (value, e.value, e.type), ())
        return ("lit", (e.type, e.value), ())
    if isinstance(e, c_ast.UnaryOp):
        if e.op in _UNARY:
            return (_UNARY[e.op], None, (_term(e.expr),))
        return ("incdec", (e.op, 0), (_term(e.expr),))
    if isinstance(e, c_ast.BinaryOp):
        return (e.op, None, (_term(e.left), _term(e.right)))
    if isinstance(e, c_ast.Cast):
        return ("cast", expr_text(e.to_type), (_term(e.expr),))
    if isinstance(e, c_ast.ArrayRef):
        return ("index", None, (_term(e.name), _term(e.subscript)))
    if isinstance(e, c_ast.StructRef):
        return ("field", (e.type, e.field.name), (_term(e.name),))
    if isinstance(e, c_ast.TernaryOp):
        return ("?:", None, (_term(e.cond), _term(e.iftrue), _term(e.iffalse)))
    if isinstance(e, c_ast.Assignment):
        return ("assign", (e.op, 0), (_term(e.lvalue), _term(e.rvalue)))
    if isinstance(e, c_ast.FuncCall):
        args = list(e.args.exprs) if e.args is not None else []
        return ("call", 0, (_term(e.name),) + tuple(_term(a) for a in args))
    raise ValueError(type(e).__name__)


def forms(fn: Function, eg: EGraph, root: int, per_class: int = 6) -> list[str]:
    """Texts of the cheapest terms of ``root`` (best first)."""
    limits = Limits(per_class=per_class, extract_rounds=10)
    table = Extractor(eg, limits).run(eg.find(root))
    out = []
    for entry in table.get((eg.find(root), "rv"), []):
        text = fn.text(entry.term)
        if text is not None and text not in out:
            out.append(text)
    return out
