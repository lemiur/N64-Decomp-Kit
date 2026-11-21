"""Which kind of wall a non-matching candidate is up against, in the order it pays to work on them.

A candidate that is not exact differs from retail in one or more of four ways, and they are not equally
worth attacking.  A missing or extra call, or a different number of branches and returns, changes what
the compiler has to place; every register and scheduling difference after that is a consequence, not a
cause.  Spending compiles on register families while the call set still differs is the commonest way a
search wastes its budget.  The routing order is therefore fixed:

    1. call set        the calls made differ (a missing inline body, an extra helper, a duplicated tail)
    2. control flow    branch, return or computed-jump counts differ (a source construct differs)
    3. switch map      same code, but a jump table sends cases to different arms
    4. register/schedule   counts and shape agree; registers, operand order or instruction order differ
    5. referent        the code is identical once addresses are masked: only what is referred to differs

``classify`` names the first class that holds and gives its evidence; ``gate`` uses it to hold back the
causes that belong to a later class.  ``jump_tables`` reads a function's switch tables so a permuted
case-to-arm mapping, which scores as nearly exact, is seen for what it is.  ``scaffolding`` lists the
devices in a source that match bytes without being what a programmer wrote (a ``volatile`` that is not
hardware, a register pinned with ``asm``, an empty arm kept for its side effect on the optimiser), so an
exact match that leans on them is queued for cleanup instead of being mistaken for finished source.
"""
from __future__ import annotations

import re
import struct
from collections import Counter

ORDER = ("call-set", "control-flow", "switch-map", "register-schedule", "referent")
LATER = {"call-set": ("control.", "register.", "schedule.", "frame."),
         "control-flow": ("register.", "schedule.", "frame.")}

_BRANCH_OPS = {4, 5, 6, 7, 20, 21, 22, 23}          # beq bne blez bgtz and their likely forms
_REGIMM_BRANCH = {0, 1, 2, 3, 16, 17, 18, 19}       # bltz bgez (likely, and-link)


def _words(code: bytes) -> tuple[int, ...]:
    count = len(code) // 4
    return struct.unpack(f">{count}I", code[:count * 4])


def shape(code: bytes) -> dict:
    """Counts that do not depend on registers or addresses: calls by target, branches, returns, computed jumps."""
    calls: Counter = Counter()
    branches = returns = computed = indirect_calls = jumps = 0
    for word in _words(code):
        op = word >> 26
        if op == 3:
            calls[(word & 0x03FFFFFF) << 2] += 1
        elif op == 2:
            jumps += 1
        elif op in _BRANCH_OPS or (op == 1 and ((word >> 16) & 31) in _REGIMM_BRANCH) \
                or (op == 17 and ((word >> 21) & 31) == 8):
            branches += 1
        elif op == 0:
            function = word & 0x3F
            if function == 8:
                if (word >> 21) & 31 == 31:
                    returns += 1
                else:
                    computed += 1
            elif function == 9:
                indirect_calls += 1
    return {"calls": calls, "call_count": sum(calls.values()), "indirect_calls": indirect_calls,
            "branches": branches, "returns": returns, "computed_jumps": computed, "jumps": jumps}


def _mask(code: bytes) -> tuple[int, ...]:
    """Instructions with call targets and address halves removed (what a masked diff compares)."""
    out, high = [], set()
    for word in _words(code):
        op, rs, rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
        if op in (2, 3):
            out.append(op << 26)
            high.clear()
        elif op == 15:
            out.append(word & 0xFFFF0000)
            high.add(rt)
        elif rs in high and (op in (9, 13) or op >= 32):
            out.append(word & 0xFFFF0000)
        else:
            out.append(word)
            if op == 0:
                high.discard((word >> 11) & 31)
            elif 8 <= op <= 14 or 32 <= op <= 39:
                high.discard(rt)
    return tuple(out)


def classify(target: bytes, candidate: bytes, *, target_tables=None, candidate_tables=None) -> dict:
    """The first wall class that holds, with what shows it.  ``wall`` is None when the two are identical.

    Call targets are compared as addresses when both sides resolve them (a relocated candidate does);
    a side with unresolved calls (target 0) is compared by count only."""
    if candidate.rstrip(b"\0") == target.rstrip(b"\0"):
        return {"wall": None, "evidence": []}
    retail, built = shape(target), shape(candidate)
    evidence: list[str] = []
    resolved = all(retail["calls"]) and all(built["calls"])       # no call to address 0 on either side
    if retail["call_count"] != built["call_count"] or retail["indirect_calls"] != built["indirect_calls"]:
        evidence.append(f"retail makes {retail['call_count']} direct and {retail['indirect_calls']} indirect "
                        f"call(s), the candidate {built['call_count']} and {built['indirect_calls']}")
    if resolved and retail["calls"] != built["calls"]:
        missing = retail["calls"] - built["calls"]
        extra = built["calls"] - retail["calls"]
        if missing:
            evidence.append("calls only in retail: " + ", ".join(f"0x{t | 0x80000000:08X} x{n}"
                                                                 for t, n in sorted(missing.items())[:6]))
        if extra:
            evidence.append("calls only in the candidate: " + ", ".join(f"0x{t | 0x80000000:08X} x{n}"
                                                                        for t, n in sorted(extra.items())[:6]))
    if evidence:
        return {"wall": "call-set", "evidence": evidence, "retail": retail, "candidate": built}
    for key, name in (("branches", "branch(es)"), ("returns", "return(s)"), ("computed_jumps", "computed jump(s)"),
                      ("jumps", "unconditional jump(s)")):
        if retail[key] != built[key]:
            evidence.append(f"retail has {retail[key]} {name}, the candidate {built[key]}")
    if evidence:
        return {"wall": "control-flow", "evidence": evidence, "retail": retail, "candidate": built}
    if target_tables is not None and candidate_tables is not None:
        for (offset, count, want), (_o, _c, got) in zip(target_tables, candidate_tables):
            if want is not None and got is not None and want != got:
                moved = [f"case {case}: +0x{a:X} in retail, +0x{b:X} in the candidate"
                         for case, (a, b) in enumerate(zip(want, got)) if a != b]
                return {"wall": "switch-map", "retail": retail, "candidate": built,
                        "evidence": [f"jump table used at +0x{offset:X} ({count} cases) maps cases to different "
                                     f"arms"] + moved[:6]}
    if _mask(target) == _mask(candidate) and len(target.rstrip(b"\0")) == len(candidate.rstrip(b"\0")):
        return {"wall": "referent", "retail": retail, "candidate": built,
                "evidence": ["the code is identical once call targets and addresses are masked: a call or a "
                             "global resolves to a different place"]}
    return {"wall": "register-schedule", "retail": retail, "candidate": built,
            "evidence": ["calls, branches and returns agree in number; registers, operands or order differ"]}


def gate(wall: str | None, cause_ids) -> set[str]:
    """Causes that belong to a later class than ``wall`` and should wait until it is closed."""
    prefixes = LATER.get(wall or "", ())
    return {cause for cause in cause_ids if cause.startswith(prefixes)} if prefixes else set()


def jump_tables(code: bytes, vram: int, read) -> list[tuple[int, int, list[int] | None]]:
    """Switch tables a function indexes: (offset of the table load, cases, arm offsets from the function start).

    Recognised as ``sltiu`` (the bound) ... ``lui``/``addu``/``lw off(reg)`` ... ``jr reg``.  ``read(address,
    size)`` returns the bytes there or None; a table that cannot be read, or whose entries do not all point
    inside the function, is reported with None so the caller can say "not verified" instead of "equal"."""
    words = _words(code)
    out, high, bound = [], {}, None
    for index, word in enumerate(words):
        op, rs, rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
        low = word & 0xFFFF
        signed = low - 0x10000 if low & 0x8000 else low
        if op == 11:
            bound = low
        if op == 15:
            high[rt] = low << 16
            continue
        if op == 0 and (word & 0x3F) == 0x21 and (rs in high or rt in high):
            high[(word >> 11) & 31] = high.get(rs, high.get(rt))
            continue
        if op == 9 and rs in high and rt == rs:
            high[rt] = (high[rs] + signed) & 0xFFFFFFFF
            continue
        if op == 35 and rs in high and bound and any(
                (words[k] >> 26) == 0 and (words[k] & 0x3F) == 8 and ((words[k] >> 21) & 31) == rt
                for k in range(index + 1, min(index + 3, len(words)))):
            raw = read((high[rs] + signed) & 0xFFFFFFFF, 4 * bound)
            arms = None
            if raw is not None and len(raw) == 4 * bound:
                arms = [entry - vram for entry in struct.unpack(f">{bound}I", raw)]
                if not all(0 <= arm < len(code) and arm % 4 == 0 for arm in arms):
                    arms = None
            out.append((4 * index, bound, arms))
            bound = None
            continue
        if op in (2, 3):
            high.clear()
    return out


_SCAFFOLD = (
    ("volatile", r"\bvolatile\b", "a volatile qualifier (real only on hardware registers and shared state)"),
    ("pinned-register", r"\bregister\b[^;{}]*\b(?:asm|__asm__)\s*\(\s*\"\$?\w+\"\s*\)",
     "a local pinned to a machine register"),
    ("inline-assembly", r"\b(?:asm|__asm__)\s*\(\s*\"(?!\$?\w+\"\s*\))", "inline assembly"),
    ("empty-arm", r"\b(?:if|else|while|for)\b[^;{}]*\)\s*\{\s*\}|\belse\s*\{\s*\}", "an empty block kept for its effect"),
    ("do-while-0", r"\bdo\b\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*while\s*\(\s*0\s*\)", "a do { } while (0) wrapper"),
    ("offset-cast", r"\*\s*\(\s*[\w\s]+\*\s*\)\s*\(\s*\(\s*(?:char|u8|s8|unsigned char)\s*\*\s*\)[^;]*\+",
     "a field reached by a byte-offset cast instead of a member"),
    ("pointer-as-integer", r"\(\s*(?:s32|u32|int|unsigned(?: int)?)\s*\)\s*&", "an address held in an integer"),
    ("self-assignment", r"(?<![\w.>])(\w+)\s*=\s*\1\s*;", "a value assigned to itself"),
    ("goto", r"\bgoto\s+\w+\s*;", "a goto"),
    ("m2c-placeholder", r"\bM2C_\w+", "a decompiler placeholder"),
)


def scaffolding(source: str) -> list[dict]:
    """Devices in ``source`` that produce the bytes without being plausible original source.

    None of them makes a match less exact; each marks a place where the real cause (a missing inline
    helper, a wrong type, a lost aggregate) is still unknown.  Comments and string literals are ignored."""
    text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    found = []
    for name, pattern, meaning in _SCAFFOLD:
        searched = text if name == "inline-assembly" else re.sub(r'"(?:\\.|[^"\\])*"', '""', text) \
            if name not in ("pinned-register",) else text
        hits = list(re.finditer(pattern, searched, flags=re.S))
        if hits:
            line = searched.count("\n", 0, hits[0].start()) + 1
            found.append({"device": name, "count": len(hits), "first_line": line, "meaning": meaning})
    return found
