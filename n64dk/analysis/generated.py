"""Generated code and data: repeated templates, massive switches, data-driven wrappers, asset tables.

Code produced by a generator (a macro expanded per entry, a script compiler, a table-to-C tool, a
code generator for state machines) leaves families of functions that are the same instruction
sequence with different fields.  This module finds such families among catalogued functions and
states, for each, a **generator hypothesis**: a template (the words with the varying fields
cleared) plus a parameter table (one row per member) and the rule that fills each varying field
from a parameter.  ``instantiate`` regenerates a member's words from the template and its row; the
hypothesis is checked two ways:

* ``template_exact``: every member regenerated from its own row equals its bytes (the fit);
* ``holdout_exact``: for each member, the hypothesis fitted on the *other* members, given only the
  member's independent parameters (read from its own words), regenerates it exactly.  This is the
  test that the relations between fields (an address split into ``%hi``/``%lo``, two offsets a
  constant apart, a field equal to another) are a property of the generator rather than of the
  sample.

Families are keyed by the exact *shape* sequence (every word with its register, shift-amount,
immediate and jump-target fields cleared; opcode and function codes kept).  Trailing zero words
(alignment padding) are not part of the shape; their count is a per-member property.

Parameter rules (``source.kind``):

``param``   the field is parameter *k* (a register, a shift amount, an immediate, a jump target)
``delta``   the field is parameter *k* + a constant (mod 2^bits), the same for every member
``hi``/``lo`` the 16-bit halves of a 32-bit address parameter, a ``lui`` and an instruction using
            its register as a base or source (``hi`` carries for a sign-extending ``%lo``;
            ``hi_ori`` does not); ``lo`` may carry a constant delta (a second field of the same
            symbol)

Labels: ``duplicate`` (no varying field), ``wrapper`` (at most 16 words, exactly one call or tail
jump, no backward branch), ``template`` (the rest).  Also reported: functions with a jump-table
switch whose ``sltiu`` bound is large (``switches``), and fixed-stride tables in data whose columns
are constants, arithmetic progressions or free values (``record_tables``: asset-to-C candidates).

Everything here is a hypothesis about how bytes were produced.  It is evidence for grouping, naming
and source recovery, never credit and never a substitute for rebuilding a member and comparing it.
"""
from __future__ import annotations

import struct
from collections import defaultdict
from dataclasses import dataclass

M32 = 0xFFFFFFFF
MIN_WORDS = 4
MIN_MEMBERS = 3
LO_OPS = {0x08, 0x09, 0x0D, 0x20, 0x21, 0x23, 0x24, 0x25, 0x27, 0x28, 0x29, 0x2B, 0x31, 0x35, 0x37, 0x39, 0x3D,
          0x3F}
R_FIELDS = ((21, 5), (16, 5), (11, 5), (6, 5))
I_FIELDS = ((21, 5), (16, 5), (0, 16))


def fields(word: int) -> tuple[tuple[int, int], ...]:
    """The variable fields of an instruction word, as (shift, bits)."""
    op = word >> 26
    if op == 0:
        return R_FIELDS
    if op == 1:
        return ((21, 5), (0, 16))
    if op in (2, 3):
        return ((0, 26),)
    if 0x10 <= op <= 0x13:
        if (word >> 21) & 31 == 8:                       # bc1t/bc1f: condition bits and offset
            return ((16, 5), (0, 16))
        return ((16, 5), (11, 5), (6, 5))
    return I_FIELDS


def shape(word: int) -> int:
    mask = 0
    for shift, bits in fields(word):
        mask |= ((1 << bits) - 1) << shift
    return word & ~mask & M32


def _words(code: bytes) -> tuple[list[int], int]:
    words = list(struct.unpack(f">{len(code) // 4}I", code[:len(code) & ~3]))
    pad = 0
    while words and words[-1] == 0:
        words.pop()
        pad += 1
    return words, pad


def _sx16(v: int) -> int:
    return v - 0x10000 if v & 0x8000 else v


@dataclass
class Hypothesis:
    template: list                   # words with every varying field cleared
    slots: list                      # {"word", "shift", "bits", "source": {"kind", "param", "delta"}}
    params: list                     # {"name", "kind": "field" | "address", "bits"}

    def as_dict(self) -> dict:
        return {"words": len(self.template), "slots": self.slots, "params": self.params}


def instantiate(h: Hypothesis, params: list[int]) -> list[int]:
    """The words the hypothesis generates from one parameter row."""
    words = list(h.template)
    for s in h.slots:
        src = s["source"]
        p = params[src["param"]]
        kind = src["kind"]
        if kind == "hi":
            v = ((p + 0x8000) >> 16) & 0xFFFF
        elif kind == "hi_ori":
            v = (p >> 16) & 0xFFFF
        else:                                            # param, delta, lo
            v = p + src.get("delta", 0)
        words[s["word"]] |= (v & ((1 << s["bits"]) - 1)) << s["shift"]
    return words


def extract(h: Hypothesis, words: list[int]) -> list[int] | None:
    """A member's independent parameters, read from the fields that define them."""
    values: list[int | None] = [None] * len(h.params)
    halves: dict[int, dict] = defaultdict(dict)
    for s in h.slots:
        src = s["source"]
        v = (words[s["word"]] >> s["shift"]) & ((1 << s["bits"]) - 1)
        k = src["param"]
        if src["kind"] == "param":
            values[k] = v
        elif src["kind"] in ("hi", "hi_ori"):
            halves[k]["hi"], halves[k]["ori"] = v, src["kind"] == "hi_ori"
        elif src["kind"] == "lo" and not src.get("delta") and "lo" not in halves[k]:
            halves[k]["lo"] = v
    for k, half in halves.items():
        if "hi" in half and "lo" in half:
            values[k] = ((half["hi"] << 16) | half["lo"]) if half["ori"] else \
                ((half["hi"] << 16) + _sx16(half["lo"])) & M32
    if any(v is None for v in values):
        return None
    return values


def fit(members: list[list[int]]) -> tuple[Hypothesis, list[list[int]]]:
    """The hypothesis for equal-shape word lists, and each member's parameter row."""
    n = len(members[0])
    template, slots, params, table = [], [], [], [[] for _ in members]
    raw = []                                             # (word, shift, bits, values)
    for i in range(n):
        word = members[0][i]
        cleared = word
        for shift, bits in fields(word):
            mask = (1 << bits) - 1
            vals = [(m[i] >> shift) & mask for m in members]
            if len(set(vals)) > 1:
                cleared &= ~(mask << shift) & M32
                raw.append((i, shift, bits, vals))
        template.append(cleared)
    done: set[int] = set()

    def new_param(kind: str, bits: int, values: list[int]) -> int:
        params.append({"name": f"p{len(params)}", "kind": kind, "bits": bits})
        for row, v in zip(table, values):
            row.append(v)
        return len(params) - 1

    def literal(i: int, shift: int, bits: int) -> int:
        return (template[i] >> shift) & ((1 << bits) - 1)

    for x, (i, shift, bits, vals) in enumerate(raw):
        if x in done:
            continue
        done.add(x)
        word = members[0][i]
        # %hi/%lo pair: a varying lui immediate and the first later use of its register as a base
        if word >> 26 == 0x0F and (shift, bits) == (0, 16) and all(f[0] != 16 for f in _varying(raw, i)):
            reg = literal(i, 16, 5)
            partner = None
            for y in range(x + 1, len(raw)):
                j, s2, b2, v2 = raw[y]
                w2 = members[0][j]
                if y not in done and (s2, b2) == (0, 16) and w2 >> 26 in LO_OPS and j > i \
                        and all(f[0] != 21 for f in _varying(raw, j)) and literal(j, 21, 5) == reg:
                    partner = y
                    break
            if partner is not None:
                j, _s, _b, lo_vals = raw[partner]
                ori = members[0][j] >> 26 == 0x0D
                addresses = [((h << 16) | lo) if ori else ((h << 16) + _sx16(lo)) & M32
                             for h, lo in zip(vals, lo_vals)]
                k = new_param("address", 32, addresses)
                slots.append({"word": i, "shift": 0, "bits": 16,
                              "source": {"kind": "hi_ori" if ori else "hi", "param": k}})
                slots.append({"word": j, "shift": 0, "bits": 16, "source": {"kind": "lo", "param": k, "delta": 0}})
                done.add(partner)
                continue
        source = None
        for k, p in enumerate(params):                  # a constant offset from an earlier parameter
            column = [row[k] for row in table]
            if p["kind"] == "address" and bits == 16:
                deltas = {(v - c) & 0xFFFF for v, c in zip(vals, column)}
                if len(deltas) == 1:
                    source = {"kind": "lo", "param": k, "delta": _sx16(deltas.pop())}
                    break
            elif p["kind"] == "field" and p["bits"] == bits:
                deltas = {(v - c) & ((1 << bits) - 1) for v, c in zip(vals, column)}
                if len(deltas) == 1:
                    source = {"kind": "delta", "param": k, "delta": deltas.pop()}
                    break
        if source is None:
            source = {"kind": "param", "param": new_param("field", bits, vals)}
        slots.append({"word": i, "shift": shift, "bits": bits, "source": source})
    return Hypothesis(template, slots, params), table


def _varying(raw, i: int) -> list[tuple[int, int]]:
    return [(s, b) for j, s, b, _v in raw if j == i]


def holdout(members: list[list[int]]) -> list[bool]:
    """For each member: does the hypothesis fitted on the others regenerate it from its own
    independent parameters?"""
    out = []
    for i, member in enumerate(members):
        others = members[:i] + members[i + 1:]
        h, _table = fit(others)
        row = extract(h, member)
        out.append(row is not None and instantiate(h, row) == member)
    return out


@dataclass
class Cluster:
    kind: str
    members: list                    # (name, vram, size)
    words: int
    hypothesis: Hypothesis
    table: list
    pads: list
    template_exact: int = 0
    holdout_exact: int = 0
    calls: int = 0

    @property
    def bytes(self) -> int:
        return sum(size for _n, _v, size in self.members)

    def as_dict(self, rows: bool = False) -> dict:
        out = {"kind": self.kind, "members": len(self.members), "words": self.words, "bytes": self.bytes,
               "params": len(self.hypothesis.params), "varying_fields": len(self.hypothesis.slots),
               "template_exact": self.template_exact, "holdout_exact": self.holdout_exact,
               "first_members": [f"0x{v:08X}" for _n, v, _s in self.members[:8]]}
        if rows:
            out["hypothesis"] = self.hypothesis.as_dict()
            out["template"] = [f"0x{w:08X}" for w in self.hypothesis.template]
            out["table"] = self.table
            out["pads"] = self.pads
        return out


def _label(words: list[int], hypothesis: Hypothesis) -> str:
    if not hypothesis.slots:
        return "duplicate"
    calls = sum(1 for w in words if w >> 26 in (2, 3))
    backward = any(w >> 26 in (1, 4, 5, 6, 7, 20, 21, 22, 23) and w & 0x8000 for w in words)
    return "wrapper" if len(words) <= 16 and calls == 1 and not backward else "template"


def clusters(functions, min_members: int = MIN_MEMBERS, min_words: int = MIN_WORDS,
             holdout_limit: int = 64) -> list[Cluster]:
    """Families among ``functions`` (``(name, vram, code)``), largest first.  The hold-out check is
    run on families of at most ``holdout_limit`` members (it refits once per member)."""
    groups: dict[tuple, list] = defaultdict(list)
    for name, vram, code in functions:
        words, pad = _words(code)
        if len(words) >= min_words:
            groups[tuple(shape(w) for w in words)].append((name, vram, len(code), words, pad))
    out = []
    for key, group in groups.items():
        if len(group) < min_members:
            continue
        members = [g[3] for g in group]
        h, table = fit(members)
        c = Cluster(_label(members[0], h), [(g[0], g[1], g[2]) for g in group], len(key), h, table,
                    [g[4] for g in group])
        c.template_exact = sum(instantiate(h, row) == m for row, m in zip(table, members))
        if h.slots and len(members) <= holdout_limit:
            c.holdout_exact = sum(holdout(members))
        elif not h.slots:
            c.holdout_exact = len(members)
        else:
            c.holdout_exact = -1                         # not run
        out.append(c)
    out.sort(key=lambda c: (-c.bytes, c.members[0][1]))
    return out


def switches(code: bytes, vram: int, min_cases: int = 64) -> list[dict]:
    """``sltiu`` bounds of at least ``min_cases`` followed within 12 words by an indirect jump
    (``jr`` through a register other than ``$ra``): large compiled switches."""
    words = struct.unpack(f">{len(code) // 4}I", code[:len(code) & ~3])
    out = []
    for i, w in enumerate(words):
        if w >> 26 == 0x0B and (w & 0xFFFF) >= min_cases:
            for k in range(i + 1, min(len(words), i + 13)):
                x = words[k]
                if x >> 26 == 0 and x & 63 == 8 and (x >> 21) & 31 != 31:
                    out.append({"function": f"0x{vram:08X}", "at": f"0x{vram + 4 * i:08X}", "cases": w & 0xFFFF})
                    break
    return out


# ---------------------------------------------------------------------------------------------
# Fixed-stride data tables

def _wclass(w: int, lo: int, hi: int) -> str:
    if w == 0:
        return "z"
    if lo <= w < hi:
        return "p"
    if 0x3C000000 <= (w & 0x7FFFFFFF) < 0x48000000:
        return "f"
    return "i"


def _column_rule(values: list[int]) -> dict:
    if len(set(values)) == 1:
        return {"kind": "const", "value": values[0]}
    d = (values[1] - values[0]) & M32
    if all((values[k + 1] - values[k]) & M32 == d for k in range(len(values) - 1)):
        return {"kind": "ap", "start": values[0], "step": d}
    return {"kind": "free"}


def generate_rows(rules: list[dict], free: list[list[int]], count: int) -> list[int]:
    """The words a table hypothesis generates: ``free`` holds, per record, the free columns' values."""
    out = []
    for r in range(count):
        it = iter(free[r])
        for rule in rules:
            if rule["kind"] == "const":
                out.append(rule["value"])
            elif rule["kind"] == "ap":
                out.append((rule["start"] + r * rule["step"]) & M32)
            else:
                out.append(next(it))
    return out


def record_tables(data: bytes, base: int, start: int = 0, end: int | None = None, image: tuple[int, int] | None = None,
                  max_stride: int = 8, min_records: int = 8) -> list[dict]:
    """Fixed-stride tables of 32-bit words in ``data[start:end]`` (``data`` at vram ``base``).  A
    record is compatible with a run when every column keeps one class (pointer into ``image``,
    float, other integer; zero fits any).  Runs of ``min_records`` or more with a pointer or float
    column are kept, best coverage first, without overlaps.  Each gets a column rule (const,
    arithmetic progression, free) and the hold-out check of predicting its last record from the
    others' rules."""
    end = len(data) if end is None else min(end, len(data))
    start = (start + 3) & ~3
    n = (end - start) // 4
    if n <= 0:
        return []
    words = struct.unpack_from(f">{n}I", data, start)
    lo, hi = image or (base, base + len(data))
    cls = [_wclass(w, lo, hi) for w in words]
    candidates = []
    for stride in range(1, max_stride + 1):
        for phase in range(stride):
            run_start, types, count = phase, [None] * stride, 0

            def close(first: int, count: int, types: list) -> None:
                # trim all-zero records at both ends
                while count and not any(words[first + c] for c in range(stride)):
                    first, count = first + stride, count - 1
                while count and not any(words[first + (count - 1) * stride + c] for c in range(stride)):
                    count -= 1
                if count >= min_records and any(t in ("p", "f") for t in types):
                    candidates.append((count * stride, stride, first, count))

            r = phase
            while r + stride <= n:
                rec = cls[r:r + stride]
                ok = all(c == "z" or t is None or t == c for c, t in zip(rec, types))
                if not ok:
                    close(run_start, count, types)
                    run_start, types, count = r, [None] * stride, 0
                for c_i, c in enumerate(rec):
                    if c != "z" and types[c_i] is None:
                        types[c_i] = c
                count += 1
                r += stride
            close(run_start, count, types)
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    taken: list[tuple[int, int]] = []
    out = []
    for cover, stride, first, count in candidates:
        a, b = first, first + cover
        if any(a < y and x < b for x, y in taken):
            continue
        taken.append((a, b))
        records = [list(words[first + r * stride:first + (r + 1) * stride]) for r in range(count)]
        rules = [_column_rule([rec[c] for rec in records]) for c in range(stride)]
        free = [[rec[c] for c in range(stride) if rules[c]["kind"] == "free"] for rec in records]
        exact = generate_rows(rules, free, count) == [w for rec in records for w in rec]
        head = [_column_rule([rec[c] for rec in records[:-1]]) for c in range(stride)]
        last = records[-1]
        predicted = all(rule["kind"] == "free" or
                        (rule["kind"] == "const" and last[c] == rule["value"]) or
                        (rule["kind"] == "ap" and last[c] == (rule["start"] + (count - 1) * rule["step"]) & M32)
                        for c, rule in enumerate(head))
        out.append({"at": f"0x{(base + start + 4 * first) & M32:08X}", "stride": 4 * stride, "records": count,
                    "bytes": 4 * cover, "columns": [r["kind"] for r in rules],
                    "column_classes": [_majority([cls[first + r * stride + c] for r in range(count)])
                                       for c in range(stride)],
                    "generated_columns": sum(r["kind"] != "free" for r in rules),
                    "template_exact": exact, "holdout_exact": predicted})
    return out


def _majority(classes: list[str]) -> str:
    nonzero = [c for c in classes if c != "z"]
    return max(sorted(set(nonzero)), key=nonzero.count) if nonzero else "z"
