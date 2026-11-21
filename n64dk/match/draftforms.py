"""Normal forms of decompiler drafts.

A decompiler renders what the compiler emitted, not what the programmer wrote: loops arrive
rotated (``i = 0; do { ...; i += 1; } while (i < n);`` for a ``for`` loop, ``if (c) { do { }
while (c); }`` for a ``while``), narrow values carry explicit masks, comparisons are spelled
``!= 0``.  Each of those is one small edit away from the source form, but a search that changes
one thing at a time sees no gain until several are made together.  ``normalise`` therefore
applies the whole set at once, before any search, using the parser-backed transformations of
``astforms``; ``drafts`` returns the normalised variants to add to a draft portfolio:

``normal``    only steps the engine proves behaviour-preserving;
``normal+``   also the steps that are merely the usual reading of a draft (removing a mask the
              declared type already implies, un-rotating a loop whose first test is not provably
              true); such a source is marked for semantic review.

Nothing is assumed about the compiler: whether a normal form is closer to the original is
decided by building it.
"""
from __future__ import annotations

# label prefixes of astforms variants, most structural first
SAFE = ("control: if+do-while -> while", "control: init+do-while -> for", "control: init+while -> for",
        "expr: x++", "expr: drop != 0")
BOLD = ("control: if+do-while -> while", "control: init+do-while -> for", "control: do-while -> while",
        "control: init+while -> for", "cast: drop mask", "cast: remove", "expr: x++", "expr: drop != 0")
FAMILIES = ("control", "expr", "cast")


def normalise(source: str, name: str, prefixes: tuple[str, ...] = SAFE, allow_review: bool = False,
              limit: int = 24) -> tuple[str, list[str], bool]:
    """Apply every applicable step, most structural first, until none is left.
    Returns (text, applied labels, whether any applied step needs semantic review)."""
    try:
        from . import astforms
    except Exception:                                      # parser not available
        return source, [], False
    text, applied, review = source, [], False
    for _step in range(limit):
        try:
            # one step is chosen out of many offered: validating each offered variant would cost
            # seconds per step on a large function, so only the result is validated, below
            offered = astforms.variants(text, name, FAMILIES, limit=256, check=False)
        except Exception:
            break
        chosen = None
        for prefix in prefixes:
            chosen = next((v for v in offered if v.label.startswith(prefix)
                           and (allow_review or not v.semantic_review)), None)
            if chosen is not None:
                break
        if chosen is None or chosen.source == text:
            break
        text, review = chosen.source, review or chosen.semantic_review
        applied.append(chosen.label)
    if applied:
        try:
            valid = astforms.diagnose(text, name).startswith("ok")
        except Exception:
            valid = False
        if not valid:                                       # never hand on a form that does not parse
            return source, [], False
    return text, applied, review


def drafts(source: str, name: str) -> list[tuple[str, str, bool]]:
    """Normalised variants of a draft: ``[(label, text, needs_review)]`` (only those that differ)."""
    out, seen = [], {source}
    for label, prefixes, allow in (("normal", SAFE, False), ("normal+", BOLD, True)):
        text, applied, review = normalise(source, name, prefixes, allow)
        if applied and text not in seen:
            seen.add(text)
            out.append((label, text, review))
    return out


# ---------------------------------------------------------------------------------------------
# Narrow types.  Without declarations a decompiler types every parameter ``s32`` and writes the
# conversion the compiler emitted as an explicit mask or cast at each use (``arg0 & 0xFF``,
# ``f(i & 0xFFFF)``).  The source had a narrow parameter, or a callee prototype with one, and no
# mask.  Several edits must be made together (type, every use, the callee's declaration), so a
# search that changes one thing at a time never crosses the gap; ``narrow`` makes them at once.

import re

from . import iterate as _iterate

_MASK_TYPE = {"0xFF": "u8", "0xFFFF": "u16"}
_NARROW = ("s8", "u8", "s16", "u16")
_ANY_CAST = r"(?:s8|u8|s16|u16|s32|u32)"


def _wrapped_uses(body: str, name: str) -> tuple[str | None, list[tuple[int, int]]]:
    """Every use of ``name`` with the conversion wrapped around it: (the one type they agree on or
    None, spans to replace by the bare name).  None when any use is unwrapped or they disagree."""
    n = re.escape(name)
    patterns = [
        (rf"\(\({_ANY_CAST}\) {n}\) & (0xFFFF|0xFF)(?![0-9A-Fa-f])", "mask"),
        (rf"\(\({_ANY_CAST}\) \({n} & (0xFFFF|0xFF)\)\)", "mask"),
        (rf"\b{n} & (0xFFFF|0xFF)(?![0-9A-Fa-f])", "mask"),
        (rf"\((s8|u8|s16|u16)\) {n}\b", "cast"),
    ]
    spans, kinds, taken = [], set(), []
    for pattern, kind in patterns:
        for found in re.finditer(pattern, body):
            if any(a < found.end() and found.start() < b for a, b in taken):
                continue
            taken.append(found.span())
            spans.append(found.span())
            kinds.add(_MASK_TYPE[found.group(1)] if kind == "mask" else found.group(1))
    uses = len(re.findall(rf"\b{n}\b", body))
    if not spans or uses != len(spans) or len(kinds) != 1:
        return None, []
    return kinds.pop(), sorted(spans)


def _split_arguments(text: str) -> list[str]:
    out, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += char in "([{"
        depth -= char in ")]}"
        if char == "," and depth == 0:
            out.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    return out + [tail] if tail or out else []


def _calls(body: str, callee: str) -> list[tuple[int, int, list[str]]]:
    """(start of the argument list, its end, arguments) for every call of ``callee``."""
    out = []
    for found in re.finditer(rf"\b{re.escape(callee)}\(", body):
        depth, position = 1, found.end()
        while position < len(body) and depth:
            depth += (body[position] == "(") - (body[position] == ")")
            position += 1
        if depth == 0:
            out.append((found.end(), position - 1, _split_arguments(body[found.end():position - 1])))
    return out


_SIMPLE = r"(?:[A-Za-z_]\w*|\((?:[^()]|\([^()]*\))*\))"


def _narrow_argument(text: str) -> tuple[str, str] | None:
    """``(inner, type)`` when the whole argument is a masked or narrow-cast simple operand."""
    found = re.fullmatch(rf"({_SIMPLE}) & (0xFFFF|0xFF)", text)
    if found:
        return found.group(1), _MASK_TYPE[found.group(2)]
    found = re.fullmatch(rf"\((s8|u8|s16|u16)\) ({_SIMPLE})", text)
    if found:
        return found.group(2), found.group(1)
    return None


def narrow(source: str, name: str) -> tuple[str, list[str]]:
    """Replace explicit conversions by the narrow types that imply them.  Returns (text, applied).

    * a parameter whose every use is masked or cast the same way takes that type;
    * an unprototyped callee whose argument is masked or cast at every call gets a prototype with
      that parameter type (other parameters ``s32``, or ``void *`` for address arguments);
    * a variable that only collects a value from several branches to copy it into another
      (``if (c) v = a; else v = b;  x = v;``) is merged into the variable it is copied to."""
    span = _iterate._function_span(source, name)
    if span is None:
        return source, []
    start, body_start, end = span
    preamble, head, body = source[:start], source[start:body_start], source[body_start:end]
    applied: list[str] = []
    narrowed: dict[str, str] = {}

    # parameters
    listed = re.search(r"\(([^()]*)\)\s*\{?\s*$", head)
    if listed and listed.group(1).strip() not in ("", "void"):
        for declaration in _split_arguments(listed.group(1)):
            found = re.fullmatch(r"(s32|u32) (\w+)", declaration)
            if not found:
                continue
            kind, spans = _wrapped_uses(body, found.group(2))
            if kind is None:
                continue
            for a, b in reversed(spans):
                body = body[:a] + found.group(2) + body[b:]
            head = re.sub(rf"\b{found.group(1)} {found.group(2)}\b", f"{kind} {found.group(2)}", head, count=1)
            narrowed[found.group(2)] = kind
            applied.append(f"narrow: parameter {found.group(2)} -> {kind}")

    # callee prototypes
    pointers = set(re.findall(r"\*\s*(\w+)\s*[;,)=]", head + body))
    for found in list(re.finditer(r"(?m)^([A-Za-z_][\w \*]*?)\b(\w+)\(\);([^\n]*)$", preamble)):
        callee = found.group(2)
        calls = _calls(body, callee)
        if not calls or len({len(arguments) for _a, _b, arguments in calls}) != 1 or not calls[0][2]:
            continue
        count = len(calls[0][2])
        types: list[str | None] = [None] * count
        usable = True
        for index in range(count):
            seen = set()
            for _a, _b, arguments in calls:
                argument = arguments[index]
                hit = _narrow_argument(argument)
                if hit:
                    seen.add(hit[1])
                elif argument in narrowed:
                    seen.add(narrowed[argument])
                elif re.fullmatch(r"-?(?:0x[0-9A-Fa-f]+|\d+)", argument):
                    continue                                  # a literal fits any integer parameter
                else:
                    seen.add("")
            if len(seen) == 1 and "" not in seen:
                types[index] = seen.pop()
            elif seen - {""} and "" in seen or len(seen) > 1:
                usable = usable and not (seen - {""})          # disagreement on a narrowed position
        if not usable or not any(types):
            continue
        for a, b, arguments in reversed(calls):
            rewritten = []
            for index, argument in enumerate(arguments):
                hit = _narrow_argument(argument) if types[index] else None
                rewritten.append(hit[0] if hit else argument)
            body = body[:a] + ", ".join(rewritten) + body[b:]
        sample = calls[0][2]
        parameters = [types[i] or ("void *" if sample[i].startswith("&") or sample[i] in pointers else "s32")
                      for i in range(count)]
        line = f"{found.group(1)}{callee}({', '.join(parameters)});{found.group(3)}"
        preamble = preamble.replace(found.group(0), line, 1)
        applied.append(f"narrow: prototype {callee}({', '.join(parameters)})")

    # a branch-collecting variable copied once into another
    for found in list(re.finditer(r"(?m)^[ \t]*(\w+) = ((?:var|temp|phi)_\w+);\n", body)):
        target, collector = found.group(1), found.group(2)
        if target == collector or re.search(rf"\b{re.escape(collector)}\b", body[found.end():]):
            continue
        if re.search(rf"\b{re.escape(target)}\b", re.sub(rf"(?m)^[ \t]*[\w \*]+\b{re.escape(target)};\n", "",
                                                         body[:found.start()])):
            continue
        writes = re.findall(rf"(?m)^[ \t]*{re.escape(collector)} = ", body[:found.start()])
        if len(writes) < 2 or len(re.findall(rf"\b{re.escape(collector)}\b", body[:found.start()])) != len(writes) + 1:
            continue                                          # read somewhere, or not a collector
        kept = body[:found.start()] + body[found.end():]
        kept = re.sub(rf"(?m)^[ \t]*[\w \*]+\b{re.escape(collector)};\n", "", kept, count=1)
        body = re.sub(rf"\b{re.escape(collector)}\b", target, kept)
        applied.append(f"narrow: merge {collector} into {target}")

    if not applied:
        return source, []
    return preamble + head + body + source[end:], applied


_plain_drafts = drafts


def drafts(source: str, name: str) -> list[tuple[str, str, bool]]:          # noqa: F811
    """Normalised variants of a draft: ``[(label, text, needs_review)]`` (only those that differ).

    ``normal`` and ``normal+`` as above; ``narrow`` (types implied by explicit conversions); and
    ``narrow+normal`` — types first, because un-rotating a loop drops the masks that reveal them."""
    out = _plain_drafts(source, name)
    seen = {source} | {text for _label, text, _review in out}
    narrowed, applied = narrow(source, name)
    if applied:
        if narrowed not in seen:
            seen.add(narrowed)
            out.append(("narrow", narrowed, True))
        text, steps, _review = normalise(narrowed, name, BOLD, True)
        if steps and text not in seen:
            seen.add(text)
            out.append(("narrow+normal", text, True))
    return out


# ---------------------------------------------------------------------------------------------
# Frame fitting.  A decompiler names every value it saw in a register (``temp_v0``, ``temp_t9``),
# and several compilers reserve a stack slot for every declared local whether or not it ends up
# in a register.  The draft's frame is then larger than the original's and every stack-relative
# instruction differs, which makes a draft that is otherwise right look hopeless.  Two edits fix
# the frame, and they are computed, not searched for: remove the temporaries that are only names
# for sub-expressions (``inlined``), and reserve the bytes that are still missing (``pad``).

_DECLARATION = re.compile(r"^[ \t]+(?:(?:volatile|const|register|static|unsigned|signed|struct|union|enum)\s+)*"
                          r"[A-Za-z_]\w*[\w \t\*]*?\b[A-Za-z_]\w*(?:\[[^\]]*\])*(?:\s*=[^;]*)?;[ \t]*(?:/\*.*\*/)?[ \t]*$")


def inlined(source: str, name: str, limit: int = 48, only: str | None = r"temp_\w+") -> tuple[str, list[str]]:
    """Substitute every temporary that only names a sub-expression.  ``only`` restricts the
    variables considered (default: the decompiler's ``temp_*`` names; None: any local the
    engine can substitute).  Returns (text, applied)."""
    wanted = re.compile(rf"inline-temp: (?:{only}) = ") if only else None
    try:
        from . import astforms
    except Exception:
        return source, []
    text, applied = source, []
    for _step in range(limit):
        try:
            offered = astforms.variants(text, name, ("inline-temp",), limit=64, check=False)
        except Exception:
            break
        chosen = next((v for v in offered if not v.semantic_review and v.source != text
                       and (wanted is None or wanted.match(v.label))), None)
        if chosen is None:
            break
        text = chosen.source
        applied.append(chosen.label)
    if applied:
        try:
            valid = astforms.diagnose(text, name).startswith("ok")
        except Exception:
            valid = False
        if not valid:
            return source, []
    return text, applied


def pad(source: str, name: str, size: int, where: str = "top") -> str | None:
    """Reserve ``size`` bytes of frame with an unused local, before (``top``) or after
    (``bottom``) the function's other declarations.  None when the function is not found."""
    span = _iterate._function_span(source, name)
    if span is None or size <= 0:
        return None
    _start, body_start, end = span
    body = source[body_start:end]
    taken = set(re.findall(r"\bpad\d*\b", body))
    label = next(n for n in ["pad"] + [f"pad{i}" for i in range(2, 64)] if n not in taken)
    words = max(1, (size + 3) // 4)
    from .transform_classes import SCAFFOLD_MARK
    line = f"    s32 {label}{'' if words == 1 else f'[{words}]'}; {SCAFFOLD_MARK}\n"
    lines = body.split("\n")
    first = 1 if lines and not lines[0].strip() else 0          # text after the opening brace
    last = first
    while last < len(lines) and (_DECLARATION.match(lines[last]) or
                                 (not lines[last].strip() and last + 1 < len(lines)
                                  and _DECLARATION.match(lines[last + 1]))):
        last += 1
    at = first if where == "top" else last
    offset = body_start + sum(len(text) + 1 for text in lines[:at])
    return source[:offset] + line + source[offset:]


# ---------------------------------------------------------------------------------------------
# Local-free form (IDO).  A draft that copies structure fields into locals and refreshes them
# after every store (``wpos = s->f; used = wpos - s->g; ... s->f += k; wpos = s->f;``) states
# the same values as one that reads the fields where they are used, but it gives the compiler
# other live ranges: IDO 5.3 happened to produce the same code for both, IDO 7.1 did not, and
# the original was written without the locals (Sesame Street, func_80069504: the cached form
# never matched in ~700 attempts; the local-free form matched under IDO 7.1 at once).  The
# form is computed, not searched for:
#
# * ``x = A op k; A = x;`` becomes ``A op= k; x = A;`` (the store folded, the refresh kept);
# * ``x = A op= k;`` is the store plus a refresh ``x = A``;
# * a local whose every assignment gives it the same field expression (no calls, no other
#   local), and which is never read before its first assignment, is replaced at each read by
#   that expression and its assignments are dropped.
#
# Re-reading a field where the draft read a copy is the usual reading of such a draft, not a
# proven equivalence (a store through another pointer could intervene), so the form is marked
# for semantic review; whether it is closer is decided by building it.

_ATOM = re.compile(r"[A-Za-z_]\w*(?:(?:->|\.)\w+|\[[^\[\]]*\])*")
_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
_LOCAL_DECL = re.compile(r"(?m)^[ \t]+(?:(?:volatile|register|unsigned|signed)\s+)*(?:s8|u8|s16|u16|s32|u32|int|"
                         r"short|char|long|f32|f64|float|double)\s+(\w+);[ \t]*\n")


def _field_expression(text: str) -> bool:
    return ("->" in text or "." in text or "[" in text) and not re.search(r"\w\s*\(|\+\+|--|[^=!<>]=(?!=)", text)


def local_free(source: str, name: str, limit: int = 32, refreshed_only: bool = False) -> tuple[str, list[str]]:
    """The draft with locals that only cache field loads replaced by the fields.  Returns
    (text, applied); (source, []) when nothing applies.  ``refreshed_only``: only the locals the
    draft assigns more than once (a copy refreshed after stores), keeping single loads."""
    span = _iterate._function_span(source, name)
    if span is None:
        return source, []
    start, body_start, end = span
    head, body = source[start:body_start], source[body_start:end]
    applied: list[str] = []
    # x = A op k; A = x;   ->   A op= k; x = A;
    fold = re.compile(r"(?m)^([ \t]*)(\w+) = (" + _ATOM.pattern + r") ([-+*/&|^]|<<|>>) ([^;\n]+);\n"
                      r"\1(\3) = \2;\n")

    def folded(found):
        applied.append(f"local-free: fold {found.group(3)} {found.group(4)}= into a store")
        indent, x, field, op, k = found.group(1), found.group(2), found.group(3), found.group(4), found.group(5)
        return f"{indent}{field} {op}= {k};\n{indent}{x} = {field};\n"
    body = fold.sub(folded, body)
    # x = A op= k;   ->   A op= k; x = A;
    chain = re.compile(r"(?m)^([ \t]*)(\w+) = (" + _ATOM.pattern + r") ((?:[-+*/&|^]|<<|>>)?=) ([^;\n]+);\n")
    def chained(found):
        if not _field_expression(found.group(3)):
            return found.group(0)                          # only stores into a field are split
        indent, x, field = found.group(1), found.group(2), found.group(3)
        applied.append(f"local-free: split {x} = {field} {found.group(4)} ...")
        return f"{indent}{field} {found.group(4)} {found.group(5)};\n{indent}{x} = {field};\n"
    body = chain.sub(chained, body)
    locals_ = [m.group(1) for m in _LOCAL_DECL.finditer(body)]
    parameters = set(_IDENT.findall(head))
    for _round in range(limit):
        changed = False
        for var in list(locals_):
            v = re.escape(var)
            own = rf"(?<!->)(?<!\.)\b{v}\b"          # the local, not a field of that name
            if re.search(rf"(?<!&)&(?!&)\s*{v}\b|{own}\s*(?:\+\+|--|[-+*/&|^]=|<<=|>>=)|(?:\+\+|--)\s*{v}\b", body):
                continue
            assignments = list(re.finditer(rf"(?m)^[ \t]*{v} = ([^;\n]+);[ \t]*\n", body))
            if not assignments:
                continue
            values = {a.group(1).strip() for a in assignments}
            if len(values) != 1 or (refreshed_only and len(assignments) < 2):
                continue
            value = values.pop()
            names = set(_IDENT.findall(re.sub(r"(?:->|\.)\w+", "", value)))
            if not _field_expression(value) or names & set(locals_) or var in names:
                continue
            if any(re.search(rf"(?m)^[ \t]*{re.escape(n)} = ", body) for n in names if n in parameters):
                continue                                   # the base pointer itself changes
            declaration = next(m for m in _LOCAL_DECL.finditer(body) if m.group(1) == var)
            rest = body[declaration.end():]
            first_use = re.search(own, rest)
            first_write = re.search(rf"(?m)^[ \t]*{v} = ", rest)
            if first_use is None or first_write is None or first_use.start() != first_write.end() - len(var) - 3:
                continue                                   # read before its first assignment
            for a in reversed(assignments):
                body = body[:a.start()] + body[a.end():]
            body = re.sub(rf"(?m)^[ \t]+(?:(?:volatile|register|unsigned|signed)\s+)*\w+\s+{v};[ \t]*\n", "", body,
                          count=1)
            replacement = value if _ATOM.fullmatch(value) else f"({value})"
            body = re.sub(own, lambda _m: replacement, body)
            locals_.remove(var)
            applied.append(f"local-free: {var} -> {value}")
            changed = True
        if not changed:
            break
    if not any(label.startswith("local-free: ") and " -> " in label for label in applied):
        return source, []
    return source[:body_start] + body + source[end:], applied


def local_free_drafts(source: str, name: str) -> list[tuple[str, str, bool]]:
    """``[(label, text, needs_review)]``: the refreshed copies removed, then every cached field."""
    out, seen = [], {source}
    for label, refreshed in (("local-free", True), ("local-free+", False)):
        text, applied = local_free(source, name, refreshed_only=refreshed)
        if applied and text not in seen:
            seen.add(text)
            out.append((label, text, True))
    return out


_shaped_drafts = drafts


def drafts(source: str, name: str) -> list[tuple[str, str, bool]]:          # noqa: F811
    """As above, plus the ``local-free`` forms (cached field copies read in place)."""
    out = _shaped_drafts(source, name)
    seen = {source} | {text for _label, text, _review in out}
    for label, text, review in local_free_drafts(source, name):
        if text not in seen:
            seen.add(text)
            out.append((label, text, review))
    return out
