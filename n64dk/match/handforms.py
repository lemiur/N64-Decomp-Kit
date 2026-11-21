"""Rewrites learned from hand matching (Madden Football 64 campaign, ~350 functions by hand).

Each rewrite is a source-to-source form of a draft that keeps what it computes and changes only
how GCC 2.7.2 sees it.  They were mined from (failing C, exact C) pairs of the campaign
(kitwork/kd_pairs.tsv); every form here unlocked functions there.  ``forms_for(causes, ...)``
picks the forms the kit's own diagnosis of a near miss justifies:

* ``global-aggregate`` (schedule, reload/hoist, register): scalar globals at neighbouring
  addresses declared as one array (or a struct when their types differ), a lone global as a
  one-element array.  GCC 2.7.2 marks references into one aggregate MEM_IN_STRUCT: stores and
  loads keep their order, the base address is CSE'd into one register (m15 800D01AC, f06 800A3294,
  m08 800BC684, 800B2968).
* ``volatile-global`` (reload): one global read twice is re-read where the target re-reads it
  (kept as a form; the search already has it as its ``volatile`` family, so it is not selected).
* ``chain-assign`` / ``chain-split`` (schedule, register): ``a = b = x`` stores ``b`` first;
  ``a = x; b = x;`` stores in source order.
* ``split-compound`` / ``fold-compound`` (register, schedule): ``x = (A << 8) | B`` versus
  ``x = A; x <<= 8; x |= B;`` (the length built from two bytes; 800975A0, m38 80027EF0).
* ``tail-duplicate`` / ``tail-merge`` (control, branch form): a statement after an if/else
  written into each branch, or the identical last statement of both branches written once after
  it (cross-jumping; m04 800AA814, m23, m35 800BE784); ``ternary-to-if`` (two ifs merge where a
  ternary does not, m07 800D5E0C).
* ``abs`` (control): ``if (x < 0) x = -x;`` <-> ``x = abs(x);`` style.
* ``narrow-local`` (type width, register): a local retyped to the width the target's own
  extension instructions show (``sll/sra 16`` -> s16, ``andi 0xFFFF`` -> u16, ``andi 0xFF`` -> u8,
  ``sll/sra 24`` -> s8): m25 800E0F30, m22 800AC948, 800C3D68.
* ``top-tested`` (frame size): ``do {} while`` <-> ``while () {}`` (an unexplained +8 frame).

``fixmul_sites(target)`` recognises the fixed-point multiply that the project writes with an
inline-asm macro (``mult; mfhi; mflo; sll 16; srl 16; or``): such a function is routed to the
assisted category, never credited as plain C.

All forms are textual over the function body and its file-scope extern declarations; each is
built and strictly verified like any other candidate, so a form that changes meaning cannot be
credited unless it yields the target's bytes.
"""
from __future__ import annotations

import re
import struct

from . import iterate as _iterate

_SIZES = {"s8": 1, "u8": 1, "char": 1, "s16": 2, "u16": 2, "short": 2, "s32": 4, "u32": 4, "int": 4,
          "f32": 4, "float": 4, "unsigned int": 4, "unsigned short": 2, "unsigned char": 1}
_EXTERN = re.compile(r"(?m)^extern\s+((?:volatile\s+)?(?:unsigned\s+)?(?:s8|u8|s16|u16|s32|u32|f32|int|short|char|"
                     r"float))\s+(D_([0-9A-Fa-f]{8}))\s*;[ \t]*\n")
_LOCAL = re.compile(r"(?m)^([ \t]+)((?:unsigned\s+)?(?:s8|u8|s16|u16|s32|u32|int|short|char))\s+(\w+)\s*;")
_ATOM = r"[A-Za-z_]\w*(?:(?:->|\.)\w+|\[[^\[\]\n]*\])*"
_CONST = r"-?(?:0x[0-9A-Fa-f]+|\d+)"


def _span(source: str, name: str):
    span = _iterate._function_span(source, name)
    if span is None:
        return None
    start, body_start, end = span
    return start, body_start, end, source[body_start:end]


def _with_body(source: str, span, body: str) -> str:
    _start, body_start, end, _old = span
    return source[:body_start] + body + source[end:]


def _uses(text: str, ident: str) -> int:
    return len(re.findall(rf"(?<![\w.>]){re.escape(ident)}\b", text))


# ------------------------------------------------------------------------------ globals
def global_aggregate(source: str, name: str, window: int = 0x40) -> list[tuple[str, str]]:
    span = _span(source, name)
    if span is None:
        return []
    body = span[3]
    decls = [(m, m.group(1).replace("volatile ", ""), m.group(2), int(m.group(3), 16))
             for m in _EXTERN.finditer(source[:span[0]])]
    used = [d for d in decls if _uses(body, d[2])]
    used.sort(key=lambda d: d[3])
    out: list[tuple[str, str]] = []

    def rewrite(group, decl_text, ref):
        text = source
        # declarations first (they precede the function), last to first so offsets hold
        first = group[0]
        for m, *_rest in sorted(group, key=lambda d: d[0].start(), reverse=True):
            repl = decl_text if m is first[0] else ""
            text = text[:m.start()] + repl + text[m.end():]
        sp = _span(text, name)
        new_body = sp[3]
        for _m, typ, ident, addr in group:
            new_body = re.sub(rf"(?<![\w.>]){re.escape(ident)}\b", ref(typ, ident, addr), new_body)
        return _with_body(text, sp, new_body)

    # runs of neighbouring globals
    i = 0
    while i < len(used):
        j = i
        while j + 1 < len(used) and used[j + 1][3] - used[i][3] < window and \
                used[j + 1][3] >= used[j][3] + _SIZES.get(used[j][1], 4):
            j += 1
        group = used[i:j + 1]
        base = group[0]
        if len(group) >= 2:
            same = len({g[1] for g in group}) == 1 and all((g[3] - base[3]) % _SIZES[g[1]] == 0 for g in group)
            if same:
                size = _SIZES[base[1]]
                count = (group[-1][3] - base[3]) // size + 1
                out.append((f"global-aggregate: {base[2]}..{group[-1][2]} as {base[1]} {base[2]}[{count}]",
                            rewrite(group, f"extern {base[1]} {base[2]}[{count}];\n",
                                    lambda t, ident, a: f"{base[2]}[{(a - base[3]) // size}]")))
            aligned = all((g[3] - base[3]) % _SIZES[g[1]] == 0 for g in group)
            if aligned:
                members, at = [], 0
                for _m, typ, ident, addr in group:
                    off = addr - base[3]
                    if off > at:
                        members.append(f"u8 pad_{at:02X}[{off - at}];")
                    members.append(f"{typ} unk_{off:02X};")
                    at = off + _SIZES[typ]
                tag = f"Glob_{base[2][2:]}"
                decl = "typedef struct {\n    " + "\n    ".join(members) + f"\n}} {tag};\nextern {tag} {base[2]};\n"
                out.append((f"global-aggregate: {base[2]}..{group[-1][2]} as struct",
                            rewrite(group, decl, lambda t, ident, a: f"{base[2]}.unk_{a - base[3]:02X}")))
        i = j + 1
    for m, typ, ident, addr in used:
        if _uses(body, ident) >= 2 and not re.search(rf"&\s*{ident}\b", body):
            out.append((f"global-aggregate: {ident} as {typ} {ident}[1]",
                        rewrite([(m, typ, ident, addr)], f"extern {typ} {ident}[1];\n",
                                lambda t, i_, a: f"{i_}[0]")))
    return out


def volatile_global(source: str, name: str, limit: int = 4) -> list[tuple[str, str]]:
    span = _span(source, name)
    if span is None:
        return []
    body, out = span[3], []
    for m in _EXTERN.finditer(source[:span[0]]):
        if "volatile" in m.group(1) or _uses(body, m.group(2)) < 2:
            continue
        text = source[:m.start()] + f"extern volatile {m.group(1)} {m.group(2)};\n" + source[m.end():]
        out.append((f"volatile-global: {m.group(2)}", text))
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------------------ assignments
def chain_forms(source: str, name: str) -> list[tuple[str, str]]:
    span = _span(source, name)
    if span is None:
        return []
    body, out = span[3], []
    # a = E; b = E;  (E constant or plain name)  ->  b = a = E;  and  a = b = E;
    pair = re.compile(rf"(?m)^([ \t]*)({_ATOM}) = ({_CONST}|[A-Za-z_]\w*);[ \t]*\n\1({_ATOM}) = (\3|\2);[ \t]*\n")
    for m in pair.finditer(body):
        ind, a, e, b = m.group(1), m.group(2), m.group(3), m.group(4)
        if a == b or a in e:
            continue
        for first, second in ((b, a), (a, b)):
            new = body[:m.start()] + f"{ind}{first} = {second} = {e};\n" + body[m.end():]
            out.append((f"chain-assign: {first} = {second} = {e}", _with_body(source, span, new)))
    # a = b = E;  ->  b = E; a = b;   and  b = E; a = E;
    chain = re.compile(rf"(?m)^([ \t]*)({_ATOM}) = ({_ATOM}) = ([^;=\n]+);[ \t]*\n")
    for m in chain.finditer(body):
        ind, a, b, e = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        forms = [f"{ind}{b} = {e};\n{ind}{a} = {b};\n"]
        if re.fullmatch(_CONST, e):
            forms.append(f"{ind}{b} = {e};\n{ind}{a} = {e};\n")
        for f in forms:
            out.append((f"chain-split: {a} = {b} = {e}", _with_body(source, span, body[:m.start()] + f + body[m.end():])))
    return out


def _top_split(expr: str) -> tuple[str, str, str] | None:
    """``A op B`` at parenthesis depth 0 with the last lowest-precedence operator: (A, op, B)."""
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for i, ch in enumerate(expr):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(expr) - 1:
                break
        else:
            expr = expr[1:-1].strip()
            continue
        break
    for ops in (("|",), ("^",), ("&",), ("<<", ">>"), ("+", "-")):
        depth, found = 0, None
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif depth == 0 and i > 0:
                for op in ops:
                    if expr.startswith(op, i) and not expr.startswith(op * 2, i) and expr[i - 1] not in "<>|&=!" \
                            and (len(op) == 2 or expr[i + 1:i + 2] not in ("=", op)):
                        if op in "+-" and expr[i - 1] in "*/(":
                            continue
                        found = (i, op)
                        i += len(op) - 1
                        break
            i += 1
        if found:
            i, op = found
            return expr[:i].strip(), op, expr[i + len(op):].strip()
    return None


def split_compound(source: str, name: str) -> list[tuple[str, str]]:
    span = _span(source, name)
    if span is None:
        return []
    body, out = span[3], []
    locals_ = {m.group(3) for m in _LOCAL.finditer(body)}
    head = source[span[0]:span[1]]
    locals_ |= set(re.findall(r"\b(\w+)\s*[,)]", head))
    assign = re.compile(r"(?m)^([ \t]*)(\w+) = ([^;=\n]+);[ \t]*\n")
    for m in assign.finditer(body):
        ind, x, e = m.group(1), m.group(2), m.group(3)
        if x not in locals_ or "(" in e and re.search(r"\w\s*\(", e):
            continue
        steps, rest = [], e
        while True:
            parts = _top_split(rest)
            if parts is None or _uses(parts[2], x):
                break
            rest, op, k = parts
            steps.insert(0, (op, k))
            if _uses(rest, x):
                break
        if not steps or _uses(rest, x) and rest.strip() != x:
            continue
        lines = ([] if rest.strip() == x else [f"{ind}{x} = {rest};\n"]) + [f"{ind}{x} {op}= {k};\n" for op, k in steps]
        if len(lines) < 2 and rest.strip() != x:
            continue
        out.append((f"split-compound: {x} = {e.strip()}", _with_body(source, span, body[:m.start()] + "".join(lines) +
                                                                       body[m.end():])))
    # fold:  x = A; x op= K; ...  ->  x = (A) op K ...
    fold = re.compile(r"(?m)^([ \t]*)(\w+) = ([^;=\n]+);[ \t]*\n((?:\1\2 (?:<<|>>|[-+|&^*])= [^;\n]+;[ \t]*\n)+)")
    for m in fold.finditer(body):
        ind, x, expr = m.group(1), m.group(2), m.group(3).strip()
        if x not in locals_:
            continue
        for step in re.finditer(rf"{re.escape(x)} (<<|>>|[-+|&^*])= ([^;\n]+);", m.group(4)):
            if _uses(step.group(2), x):
                break
            expr = f"({expr}) {step.group(1)} {step.group(2).strip()}"
        else:
            out.append((f"fold-compound: {x}", _with_body(source, span, body[:m.start()] + f"{ind}{x} = {expr};\n" +
                                                            body[m.end():])))
    return out


# ------------------------------------------------------------------------------ control
def _block_end(text: str, opening: int) -> int | None:
    depth = 0
    for i in range(opening, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _last_statement(block: str) -> tuple[int, str] | None:
    inner = block.rstrip()
    m = re.search(r"(?m)^([ \t]*)([^\n{};]+;)\Z", inner)
    if not m or "}" in inner[m.start():] or re.match(r"\s*(?:break|continue|return|goto)\b", m.group(2)):
        return None
    return m.start(), m.group(2).strip()


def tail_forms(source: str, name: str) -> list[tuple[str, str]]:
    span = _span(source, name)
    if span is None:
        return []
    body, out = span[3], []
    for m in re.finditer(r"(?m)^([ \t]*)if \(", body):
        ind = m.group(1)
        open1 = body.find("{", m.end())
        if open1 < 0 or "\n" in body[m.end():open1].strip("\n") and ";" in body[m.end():open1]:
            continue
        close1 = _block_end(body, open1)
        if close1 is None:
            continue
        after = body[close1 + 1:]
        em = re.match(r"\s*else\s*\{", after)
        if not em:
            continue
        open2 = close1 + 1 + em.end() - 1
        close2 = _block_end(body, open2)
        if close2 is None:
            continue
        then, other = body[open1 + 1:close1], body[open2 + 1:close2]
        # merge: identical last statement in both branches -> once after the if/else
        lt, lo = _last_statement(then), _last_statement(other)
        if lt and lo and lt[1] == lo[1]:
            new = (body[:open1 + 1] + then[:lt[0]].rstrip(" \t") + body[close1:open2 + 1] +
                   other[:lo[0]].rstrip(" \t") + body[close2:close2 + 1] + f"\n{ind}{lt[1]}" + body[close2 + 1:])
            out.append((f"tail-merge: {lt[1][:40]}", _with_body(source, span, new)))
        # duplicate: the statement after the if/else into both branches
        nm = re.match(r"[ \t]*\n([ \t]*)([^\n{};]+;)[ \t]*\n", body[close2 + 1:])
        if nm and nm.group(1) == ind and not re.match(r"(?:break|continue|return|goto)\b", nm.group(2)):
            stmt = nm.group(2).strip()
            inner_ind = ind + "    "
            new = (body[:close1].rstrip(" \t") + f"{inner_ind}{stmt}\n{ind}" + body[close1:close2].rstrip(" \t") +
                   f"{inner_ind}{stmt}\n{ind}}}\n" + body[close2 + 1 + nm.end():])
            out.append((f"tail-duplicate: {stmt[:40]}", _with_body(source, span, new)))
    # ternary -> if/else
    for m in re.finditer(rf"(?m)^([ \t]*)({_ATOM}) = ([^;?\n]+) \? ([^;:\n]+) : ([^;\n]+);[ \t]*\n", body):
        ind, x, c, a, b = (g.strip() if i else g for i, g in enumerate(m.groups()))
        new = (f"{ind}if ({c}) {{\n{ind}    {x} = {a};\n{ind}}} else {{\n{ind}    {x} = {b};\n{ind}}}\n")
        out.append((f"ternary-to-if: {x}", _with_body(source, span, body[:m.start()] + new + body[m.end():])))
    # abs
    for m in re.finditer(r"(?m)^([ \t]*)if \((\w+) < 0\) \{?\s*\n?[ \t]*\2 = -\2;[ \t]*\n(?:[ \t]*\}[ \t]*\n)?", body):
        ind, x = m.group(1), m.group(2)
        new_src = _with_body(source, span, body[:m.start()] + f"{ind}{x} = abs({x});\n" + body[m.end():])
        if not re.search(r"\babs\s*\(", source[:span[0]]):
            new_src = "int abs(int);\n" + new_src
        out.append((f"abs: {x}", new_src))
    return out


def top_tested(source: str, name: str) -> list[tuple[str, str]]:
    span = _span(source, name)
    if span is None:
        return []
    body, out = span[3], []
    for m in re.finditer(r"(?m)^([ \t]*)do \{[ \t]*\n", body):
        close = _block_end(body, body.find("{", m.start()))
        if close is None:
            continue
        tail = re.match(r"\s*while \((.*)\);[ \t]*\n", body[close + 1:])
        if not tail:
            continue
        cond = tail.group(1)
        new = body[:m.start()] + f"{m.group(1)}while ({cond}) {{\n" + body[m.end():close + 1] + "\n" + \
            body[close + 1 + tail.end():]
        out.append((f"top-tested: do/while -> while ({cond[:30]})", _with_body(source, span, new)))
    for m in re.finditer(r"(?m)^([ \t]*)while \(([^\n]*)\) \{[ \t]*\n", body):
        cond = m.group(2)
        if cond.strip() in ("1", "TRUE"):
            continue
        close = _block_end(body, body.find("{", m.end() - 3))
        if close is None:
            continue
        new = body[:m.start()] + f"{m.group(1)}do {{\n" + body[m.end():close + 1] + f" while ({cond});" + body[close + 1:]
        out.append((f"top-tested: while -> do/while ({cond[:30]})", _with_body(source, span, new)))
    return out


# ------------------------------------------------------------------------------ types from the target
def _words(code: bytes) -> list[int]:
    return [struct.unpack(">I", code[i:i + 4])[0] for i in range(0, len(code) - len(code) % 4, 4)]


def extension_widths(target: bytes) -> list[str]:
    """Narrow types the target's own extension instructions show, most frequent first."""
    seen: dict[str, int] = {}
    words = _words(target)
    for i, w in enumerate(words):
        op, funct = w >> 26, w & 0x3F
        if op == 0 and funct == 0:                          # sll rd, rt, sa
            sa, rd = (w >> 6) & 0x1F, (w >> 11) & 0x1F
            if sa in (16, 24) and rd:
                for nxt in words[i + 1:i + 3]:
                    if nxt >> 26 == 0 and nxt & 0x3F in (2, 3) and (nxt >> 6) & 0x1F == sa and (nxt >> 16) & 0x1F == rd:
                        t = ("s" if nxt & 0x3F == 3 else "u") + ("16" if sa == 16 else "8")
                        seen[t] = seen.get(t, 0) + 1
                        break
        elif op == 0x0C:                                    # andi
            imm = w & 0xFFFF
            if imm in (0xFF, 0xFFFF):
                t = "u8" if imm == 0xFF else "u16"
                seen[t] = seen.get(t, 0) + 1
    return sorted(seen, key=lambda t: -seen[t])


def narrow_local(source: str, name: str, target: bytes | None, limit: int = 12) -> list[tuple[str, str]]:
    if not target:
        return []
    widths = extension_widths(target)
    span = _span(source, name)
    if span is None or not widths:
        return []
    body, out = span[3], []
    for m in _LOCAL.finditer(body):
        current = m.group(2)
        options = widths if current in ("s32", "u32", "int") else ["s32"] + [w for w in widths if w != current]
        for t in options:
            if t == current:
                continue
            new = body[:m.start(2)] + t + body[m.end(2):]
            out.append((f"narrow-local: {m.group(3)} {current} -> {t}", _with_body(source, span, new)))
            if len(out) >= limit:
                return out
    return out


# ------------------------------------------------------------------------------ assisted class
def fixmul_sites(target: bytes) -> list[int]:
    """Offsets of ``mult/multu; mfhi a; mflo b; sll a,a,16; srl b,b,16; or`` (fixed-point multiply)."""
    words, sites = _words(target), []
    for i, w in enumerate(words):
        if w >> 26 == 0 and w & 0x3F in (0x18, 0x19):
            window = words[i + 1:i + 9]
            hi = [x for x in window if x >> 26 == 0 and x & 0x3F == 0x10]
            lo = [x for x in window if x >> 26 == 0 and x & 0x3F == 0x12]
            sll = [x for x in window if x >> 26 == 0 and x & 0x3F == 0 and (x >> 6) & 0x1F == 16 and x]
            srl = [x for x in window if x >> 26 == 0 and x & 0x3F == 2 and (x >> 6) & 0x1F == 16]
            orr = [x for x in window if x >> 26 == 0 and x & 0x3F == 0x25]
            if hi and lo and sll and srl and orr:
                sites.append(i * 4)
    return sites


# ------------------------------------------------------------------------------ selection
CAUSE_FORMS = {
    "register": ("narrow-local", "split-compound", "chain", "global-aggregate", "tail"),
    "schedule": ("global-aggregate", "chain", "split-compound"),
    "global": ("global-aggregate",),
    "type": ("narrow-local",),
    "control": ("tail", "top-tested"),
    "frame": ("top-tested", "narrow-local"),
    "boundary.shared_tail": ("tail",),
}
_MAKERS = {
    "global-aggregate": lambda s, n, t: global_aggregate(s, n),
    "volatile-global": lambda s, n, t: volatile_global(s, n),
    "chain": lambda s, n, t: chain_forms(s, n),
    "split-compound": lambda s, n, t: split_compound(s, n),
    "tail": lambda s, n, t: tail_forms(s, n),
    "top-tested": lambda s, n, t: top_tested(s, n),
    "narrow-local": lambda s, n, t: narrow_local(s, n, t),
}


# the search's ``hand`` family: every target-free form (volatile globals are the search's own
# ``volatile`` family already, n64dk.match.families)
SEARCH_CAUSES = ("schedule", "register", "control", "frame")


def kinds_for(causes) -> list[str]:
    out: list[str] = []
    for cause in causes:
        key = cause if cause in CAUSE_FORMS else cause.split(".")[0]
        for kind in CAUSE_FORMS.get(key, ()):
            if kind not in out:
                out.append(kind)
    return out


def forms_for(causes, source: str, name: str, target: bytes | None = None,
              limit: int = 40) -> list[tuple[str, str]]:
    """The hand forms the diagnosis justifies, in cause order; each text once."""
    out, seen = [], {source}
    for kind in kinds_for(causes):
        try:
            made = _MAKERS[kind](source, name, target)
        except Exception:                                   # a form that cannot parse this draft
            made = []
        for label, text in made:
            if text not in seen:
                seen.add(text)
                out.append((label, text))
                if len(out) >= limit:
                    return out
    return out
