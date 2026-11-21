"""Evidence-ordered source iteration in one call: types, signedness and value lifetimes.

Byte-matching work keeps returning to the same two questions:

* which C type does each value have (``s16``/``u16``/``s32``...)?  The type decides ``lh`` vs
  ``lhu``, ``sll/sra`` vs ``andi`` extensions, ``slt`` vs ``sltu`` and the frame layout;
* where does each value live (``$s0`` vs ``$t0``/``$a0``)?  That is decided by whether the
  value is live across a call, i.e. by where it is initialised and last used.

Answering them one edit and one build at a time costs a tool call per guess.  ``descend`` does
the whole search in one call: it enumerates every single-step variant of the declared kinds,
builds them all in parallel, keeps the best, and repeats until nothing improves.  The result
contains the trajectory, the final source and a *sensitivity table* (what every individual
lever did to the score), so the next human or model decision starts from measurements.

Variant families (all edit ordinary C only):

``types``     each scalar local, parameter and the return type tried as every other integer type
``fields``    signedness of each pointer cast / ``M2C_FIELD`` access type flipped (s16* <-> u16*)
``casts``     each ``(u16)``/``(s16)``/``(u8)``/``(s8)`` value cast removed or sign-flipped
``lifetime``  a pure local initialisation moved earlier or later across neighbouring statements
              (before a call it becomes call-crossing and tends to get an ``$s`` register);
              declaration initialisers are split so they can move
``order``     adjacent independent simple assignments swapped; declaration order rotated
``expression`` increment, compound-assignment, and boolean-comparison spellings
``scope``     a bounded inner block around one declaration and its contiguous uses
``split``     split one later assignment into a second carrier (semantic review required)

``lifetime`` and ``order`` moves are restricted to assignments whose right-hand side reads no
memory and calls nothing, and never cross a statement that mentions the same identifiers, a
label or a ``goto``; they preserve behaviour by construction.  Type edits can change
behaviour and remain subject to review like any other candidate.
"""
from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .typed_explore import source_shape

INT_TYPES = ("s32", "u32", "s16", "u16", "s8", "u8")
FAMILIES = ("types", "fields", "casts", "lifetime", "order", "expression", "scope", "split",
            "volatile", "globals", "pointer", "control", "callargs",
            # parser-backed families (n64dk.match.astforms); empty when the source cannot be parsed
            "ast-control", "ast-expr", "ast-temp", "ast-inline-temp", "ast-stmt-order", "ast-decl-order",
            "ast-cast", "ast-block", "ast-sameline",
            # hand-matching rewrites (n64dk.match.handforms)
            "hand")
_EXTENSION_FAMILIES = ("volatile", "globals", "pointer", "control", "callargs")
_AST_PREFIX = "ast-"
_DECL = re.compile(r"^(\s+(?:register\s+)?)(s8|u8|s16|u16|s32|u32)(\s+)(\w+)(\s*;.*|\s*=\s*[^;]+;.*)$")
_INIT = re.compile(r"^(\s+(?:register\s+)?)(\w+)(\s+)(\w+)\s*=\s*([^;]+);\s*$")
_ASSIGN = re.compile(r"^(\s+)(\w+) = ([^;]+);\s*$")
_POINTER_TYPE = re.compile(r"\b([su])(8|16|32)( \*)")
_VALUE_CAST = re.compile(r"\(([su])(8|16)\)\s*")
_POSTINC = re.compile(r"(?m)^(\s*)([A-Za-z_]\w*)\+\+;\s*$")
_PLUS_ONE = re.compile(r"(?m)^(\s*)([A-Za-z_]\w*)\s*\+=\s*1;\s*$")
# ``!x`` <-> ``x == 0`` only for a whole operand: not ``a - b == 0`` and not ``!x->y`` / ``!f(x)``
_NEGATE = re.compile(r"(?<![\w)\]])!\s*([A-Za-z_]\w*)(?![\w(\[.]|\s*->)")
_ZERO_COMPARE = re.compile(r"(?<=[(&|,])\s*([A-Za-z_]\w*)\s*==\s*0(?![\w.x])")
# ``x = x op y`` <-> ``x op= y`` only when y is one operand: ``x = x - a + b`` is not ``x -= a + b``
_OPERAND = r"(?:\([^;()]*\)|[A-Za-z_0-9.]+(?:->\w+|\.\w+|\[\w+\])*)"
_EXPANDED_ASSIGN = re.compile(r"(?m)^(\s*)([A-Za-z_]\w*)\s*=\s*\2\s*([+\-*|&^])\s*(" + _OPERAND + r");\s*$")
_COMPOUND_ASSIGN = re.compile(r"(?m)^(\s*)([A-Za-z_]\w*)\s*([+\-*|&^])=\s*(" + _OPERAND + r");\s*$")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_SAVE = re.compile(r"^sw \$s\d, ")
_WIDTH_OPS = {"lh", "lhu", "lb", "lbu", "slt", "sltu", "slti", "sltiu", "sra", "srl", "andi"}


@dataclass
class Variant:
    label: str
    family: str
    source: str
    semantic_review: bool = False


@dataclass
class Evaluation:
    score: int | None            # None: did not build
    exact: bool = False
    report: dict | None = None
    error: str | None = None
    rank_score: int | None = None
    object_digest: str | None = None
    preprocessed_digest: str | None = None
    outcome: dict | None = None


def _function_span(source: str, name: str) -> tuple[int, int, int] | None:
    """(signature start, body start after ``{``, body end before ``}``)."""
    # Keep offsets and newlines stable while hiding braces/parens in comments, literals and
    # character constants.  Regex-only brace counting truncated bodies at strings such as
    # p("}") and could not recognize split return types or K&R parameter declarations.
    chars = list(source)
    position, state = 0, "code"
    while position < len(chars):
        char = chars[position]
        following = chars[position + 1] if position + 1 < len(chars) else ""
        if state == "code" and char == "/" and following == "/":
            chars[position] = chars[position + 1] = " "; position += 2; state = "line"; continue
        if state == "code" and char == "/" and following == "*":
            chars[position] = chars[position + 1] = " "; position += 2; state = "block"; continue
        if state == "code" and char in {'"', "'"}:
            state = "string" if char == '"' else "char"; chars[position] = " "; position += 1; continue
        if state in {"string", "char"}:
            if char == "\\":
                chars[position] = " ";
                if position + 1 < len(chars) and chars[position + 1] != "\n": chars[position + 1] = " "
                position += 2; continue
            closing = '"' if state == "string" else "'"
            chars[position] = " " if char != "\n" else char
            position += 1
            if char == closing: state = "code"
            continue
        if state == "line":
            if char == "\n": state = "code"
            else: chars[position] = " "
            position += 1; continue
        if state == "block":
            if char == "*" and following == "/":
                chars[position] = chars[position + 1] = " "; position += 2; state = "code"; continue
            if char != "\n": chars[position] = " "
            position += 1; continue
        position += 1
    masked = "".join(chars)
    found = re.search(rf"\b{re.escape(name)}\s*\(", masked)
    if not found:
        return None
    open_paren = masked.find("(", found.start())
    depth, position, close_paren = 1, open_paren + 1, None
    while position < len(masked) and depth:
        depth += (masked[position] == "(") - (masked[position] == ")")
        if depth == 0: close_paren = position
        position += 1
    if close_paren is None:
        return None
    brace = masked.find("{", close_paren + 1, min(len(masked), close_paren + 4096))
    if brace < 0:
        return None
    between = masked[close_paren + 1:brace]
    # ANSI definitions have only whitespace/attributes before the body.  K&R definitions may
    # have declaration statements; reject an ordinary prototype followed by another body.
    if ";" in between:
        params = set(_IDENT.findall(masked[open_paren + 1:close_paren]))
        declarations = [line.strip() for line in between.splitlines() if line.strip()]
        if not params or not declarations or any(not line.endswith(";") for line in declarations):
            return None
        declared = set()
        for line in declarations:
            declared.update(_IDENT.findall(line)[1:])
        if not params <= declared:
            return None
    elif between.strip() and not re.fullmatch(r"(?:__attribute__\s*\(\([^)]*\)\)\s*)*", between.strip()):
        return None
    # Include a return type placed on preceding lines, but never cross another declaration or
    # body boundary.
    boundary = max(masked.rfind(";", 0, found.start()), masked.rfind("}", 0, found.start()),
                   masked.rfind("{", 0, found.start()), masked.rfind("#", 0, found.start()))
    start = boundary + 1
    while start < found.start() and masked[start].isspace():
        start += 1
    depth, position = 1, brace + 1
    while position < len(source) and depth:
        depth += {"{": 1, "}": -1}.get(masked[position], 0)
        position += 1
    return (start, brace + 1, position - 1) if depth == 0 else None


def _pure(expression: str) -> bool:
    """No call, no macro, no memory read: safe to evaluate earlier or later."""
    if re.search(r"[A-Za-z_]\w*\s*\(", re.sub(r"\((?:[su](?:8|16|32)|f32|f64)\)", "", expression)):
        return False
    return not re.search(r"->|\[|(?:^|[=(,+\-*/&|<>!~?:]\s*)\*", expression)


def _local_names(signature: str, declaration_lines: list[str]) -> set[str]:
    """Parameters and locals: the identifiers whose values cannot be changed by a call."""
    names = set(re.findall(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*[,)]", signature))
    for line in declaration_lines:
        found = re.match(r"^\s+(?:[A-Za-z_]\w*[ \*]+)+([A-Za-z_]\w*)\s*(?:\[\w*\])*\s*(?:=[^;]*)?;", line)
        if found and "static" not in line.split() and "extern" not in line.split():
            names.add(found.group(1))
    return names


def _units(lines: list[str]) -> list[tuple[int, int]]:
    """Top-level statements of a body as (first line, last line) index pairs."""
    units, depth, start = [], 0, None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if start is None:
            start = index
        depth += line.count("{") - line.count("}")
        if depth == 0 and stripped.endswith((";", "}", ":")):
            units.append((start, index))
            start = None
    return units


def _blocked(text: str, names: set[str]) -> bool:
    return bool(names & set(_IDENT.findall(text))) or bool(re.search(r"\bgoto\b|^\s*\w+:\s*$", text, re.M))


def variants(source: str, name: str, families=FAMILIES, reach: int = 6) -> list[Variant]:
    span = _function_span(source, name)
    if span is None:
        return []
    start, body_start, body_end = span
    head, signature, body, tail = source[:start], source[start:body_start], source[body_start:body_end], source[body_end:]
    lines = body.split("\n")
    out: list[Variant] = []

    def emit(label, family, new_lines=None, new_signature=None, semantic_review=False):
        text = head + (new_signature or signature) + "\n".join(new_lines if new_lines is not None else lines) + tail
        if text != source:
            out.append(Variant(label, family, text, semantic_review))

    declarations = [(i, _DECL.match(line)) for i, line in enumerate(lines)]
    declarations = [(i, m) for i, m in declarations if m]
    if "types" in families:
        for index, found in declarations:
            for kind in INT_TYPES:
                if kind != found.group(2):
                    changed = list(lines)
                    changed[index] = f"{found.group(1)}{kind}{found.group(3)}{found.group(4)}{found.group(5)}"
                    emit(f"type {found.group(4)}: {found.group(2)} -> {kind}", "types", changed,
                         semantic_review=True)
        for found in re.finditer(r"\b(s8|u8|s16|u16|s32|u32)(\s+)(\w+)(?=\s*[,)])", signature):
            for kind in INT_TYPES:
                if kind != found.group(1):
                    emit(f"param {found.group(3)}: {found.group(1)} -> {kind}", "types",
                         new_signature=signature[:found.start()] + kind + signature[found.start() + len(found.group(1)):],
                         semantic_review=True)
        returned = re.match(r"(\s*(?:static\s+)?)(s8|u8|s16|u16|s32|u32)\b", signature)
        if returned:
            for kind in INT_TYPES:
                if kind != returned.group(2):
                    emit(f"return type: {returned.group(2)} -> {kind}", "types",
                         new_signature=returned.group(1) + kind + signature[returned.end():],
                         semantic_review=True)
    for family, pattern in (("fields", _POINTER_TYPE), ("casts", _VALUE_CAST)):
        if family not in families:
            continue
        for number, found in enumerate(list(pattern.finditer(body))[:64]):
            flipped = "u" if found.group(1) == "s" else "s"
            replacement = (f"{flipped}{found.group(2)}{found.group(3)}" if family == "fields"
                           else f"({flipped}{found.group(2)})")
            line_no = body.count("\n", 0, found.start()) + 1
            changed = body[:found.start()] + replacement + body[found.end():]
            emit(f"{family[:-1]} #{number + 1} line {line_no}: {found.group(0).strip()} -> {replacement}",
                 family, changed.split("\n"), semantic_review=True)
            if family == "casts":
                emit(f"cast #{number + 1} line {line_no}: drop {found.group(0).strip()}", family,
                     (body[:found.start()] + body[found.end():]).split("\n"), semantic_review=True)
    if "lifetime" in families:
        last_declaration = -1                     # C89: declarations lead the block
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            if re.match(r"^\s+(?!return\b|goto\b|break\b|continue\b)(?:[A-Za-z_]\w*[ \*]+)+\w+(\[\w*\])*"
                        r"(\s*=[^;]+)?;\s*$", line):
                last_declaration = index
            else:
                break
        for index, line in enumerate(lines[:last_declaration + 1]):
            found = _INIT.match(line)
            if found and _pure(found.group(5)) and \
                    set(_IDENT.findall(found.group(5))) - set(INT_TYPES) - {"f32", "f64"} <= \
                    _local_names(signature, lines[:index]):
                changed = list(lines)
                changed[index] = f"{found.group(1)}{found.group(2)}{found.group(3)}{found.group(4)};"
                indent = re.match(r"\s*", found.group(1)).group(0)
                changed.insert(last_declaration + 1, f"{indent}{found.group(4)} = {found.group(5)};")
                emit(f"split initialiser of {found.group(4)}", "lifetime", changed)
        units = [u for u in _units(lines) if u[0] > last_declaration]
        local_names = _local_names(signature, lines[:last_declaration + 1])
        for position, (first, last) in enumerate(units):
            found = _ASSIGN.match(lines[first]) if first == last else None
            if not found or not _pure(found.group(3)):
                continue
            names = {found.group(2), *_IDENT.findall(found.group(3))}
            # only values that live in registers or the frame may move: a global read moved across
            # a call, or a global store moved past a return, is a behaviour change
            if not (names - set(INT_TYPES) - {"f32", "f64"}) <= local_names:
                continue
            for direction in (-1, 1):
                crossed = position
                for step in range(1, reach + 1):
                    crossed += direction
                    if not 0 <= crossed < len(units):
                        break
                    other = "\n".join(lines[units[crossed][0]:units[crossed][1] + 1])
                    if _blocked(other, names):
                        break
                    changed = list(lines)
                    moved = changed.pop(first)
                    if direction < 0:
                        changed.insert(units[crossed][0], moved)
                    else:
                        changed.insert(units[crossed][1], moved)
                    has_call = bool(re.search(r"[A-Za-z_]\w*\s*\(", other))
                    emit(f"move `{lines[first].strip()}` {'up' if direction < 0 else 'down'} {step}"
                         f"{' across a call' if has_call else ''}", "lifetime", changed)
    if "order" in families:
        for (i, a), (j, b) in zip(declarations, declarations[1:]):
            # Initializers may call, alias, or depend on the adjacent declaration. Only plain
            # declarations are reorder-safe without a data-flow proof.
            if j == i + 1 and "=" not in a.group(5) and "=" not in b.group(5):
                changed = list(lines)
                changed[i], changed[j] = changed[j], changed[i]
                emit(f"swap declarations {a.group(4)}/{b.group(4)}", "order", changed)
    if "expression" in families:
        forms = ((_POSTINC, lambda m: f"{m[1]}{m[2]} += 1;", "postincrement-to-add"),
                 (_PLUS_ONE, lambda m: f"{m[1]}{m[2]}++;", "add-to-postincrement"),
                 (_EXPANDED_ASSIGN, lambda m: f"{m[1]}{m[2]} {m[3]}= {m[4]};", "expanded-to-compound"),
                 (_COMPOUND_ASSIGN, lambda m: f"{m[1]}{m[2]} = {m[2]} {m[3]} {m[4]};", "compound-to-expanded"))
        for pattern, replacement, label in forms:
            for number, found in enumerate(pattern.finditer(body)):
                changed = body[:found.start()] + replacement(found) + body[found.end():]
                emit(f"{label} #{number + 1}", "expression", changed.split("\n"))
        for pattern, replacement, label in ((_NEGATE, lambda m: f"({m[1]} == 0)", "negation-to-zero"),
                                             (_ZERO_COMPARE, lambda m: f"!{m[1]}", "zero-to-negation")):
            for number, found in enumerate(list(pattern.finditer(body))[:32]):
                changed = body[:found.start()] + replacement(found) + body[found.end():]
                emit(f"{label} #{number + 1}", "expression", changed.split("\n"))
    if "scope" in families:
        # Narrow a plain local's scope only across a straight-line, label-free statement range.
        for index, found in declarations:
            name = found.group(4)
            uses = [i for i in range(index + 1, len(lines)) if re.search(rf"\b{re.escape(name)}\b", lines[i])]
            if not uses or any(re.search(r"\bgoto\b|^\s*\w+:\s*$|[{}]", lines[i])
                               for i in range(index + 1, uses[-1] + 1)):
                continue
            changed = list(lines)
            indent = re.match(r"\s*", changed[index]).group(0)
            changed[index] = indent + "{\n" + changed[index]
            changed[uses[-1]] += "\n" + indent + "}"
            emit(f"block-scope {name}", "scope", changed)
    if "split" in families:
        # Splitting ownership can change promotions/lifetimes; generate it as review-required.
        for index, found in declarations[:16]:
            name, kind = found.group(4), found.group(2)
            assignments = [i for i in range(index + 1, len(lines))
                           if re.match(rf"^\s*{re.escape(name)}\s*=", lines[i])]
            if len(assignments) < 2:
                continue
            at = assignments[-1]
            split_name = name + "_split"
            changed = list(lines)
            changed.insert(index + 1, f"{found.group(1)}{kind} {split_name};")
            at += 1
            changed[at] = re.sub(rf"\b{re.escape(name)}\b", split_name, changed[at], count=1)
            for use in range(at + 1, len(changed)):
                changed[use] = re.sub(rf"\b{re.escape(name)}\b", split_name, changed[use])
            emit(f"split variable {name}", "split", changed, semantic_review=True)
    extension = tuple(f for f in families if f in _EXTENSION_FAMILIES)
    if extension:
        from . import families as extension_families      # imports this module: keep the import late
        out += extension_families.variants(source, name, extension)
    parsed = tuple(f[len(_AST_PREFIX):] for f in families if f.startswith(_AST_PREFIX))
    if parsed:
        try:
            from . import astforms                           # needs pycparser; optional
            for item in astforms.variants(source, name, parsed, limit=96):
                out.append(Variant(item.label, _AST_PREFIX + item.family, item.source, item.semantic_review))
        except Exception:                                    # a parser problem must never stop the search
            pass
    if "hand" in families:
        try:
            from . import handforms
            for label, text in handforms.forms_for(handforms.SEARCH_CAUSES, source, name, None, limit=48):
                out.append(Variant(label, "hand", text, True))
        except Exception:                                    # a form problem must never stop the search
            pass
    if "egraph" in families or os.environ.get("N64DK_EGRAPH") == "1":
        # bounded equality saturation (n64dk.match.egraph): opt-in only, never a default family
        try:
            from . import egraph
            out += egraph.iterate_variants(source, name)
        except Exception:
            pass
    unique, seen = [], {hashlib.sha256(source.encode()).hexdigest()}
    for item in out:
        digest = hashlib.sha256(item.source.encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(item)
    return unique


def evidence(report: dict | None) -> dict:
    """What the residual says about the two questions, from aligned diff entries."""
    entries = (report or {}).get("entries", [])
    retail_saves = sum(1 for e in entries if _SAVE.match(" ".join((e.get("retail") or "").split()))
                       and not (e.get("compiled") or "").strip())
    built_saves = sum(1 for e in entries if _SAVE.match(" ".join((e.get("compiled") or "").split()))
                      and not (e.get("retail") or "").strip())
    def mnemonic(text):
        return (text or "").split()[0] if (text or "").split() else ""
    width = sum(1 for e in entries if {mnemonic(e.get("compiled")), mnemonic(e.get("retail"))} & _WIDTH_OPS)
    out = {"saved_registers_missing": max(0, retail_saves - built_saves),
           "saved_registers_extra": max(0, built_saves - retail_saves), "width_or_sign_sites": width}
    notes = []
    if out["saved_registers_missing"]:
        notes.append(f"retail keeps {out['saved_registers_missing']} more value(s) in $s registers: a value must "
                     "be live across a call (initialise it before the call, or use it after one)")
    if out["saved_registers_extra"]:
        notes.append(f"candidate keeps {out['saved_registers_extra']} extra value(s) in $s registers: a value "
                     "should not be live across a call (initialise it after the call, or recompute it)")
    if width:
        notes.append(f"{width} load/extension/compare site(s) differ: a type or signedness is wrong")
    out["notes"] = notes
    return out


def _family_order(report: dict | None, families) -> list[str]:
    seen = evidence(report)
    from . import families as extension_families
    first = list(extension_families.evidence_families(report))
    if seen["saved_registers_missing"] or seen["saved_registers_extra"]:
        first += ["lifetime"]
    if seen["width_or_sign_sites"]:
        first += ["types", "fields", "casts"]
    return [f for f in first if f in families] + [f for f in families if f not in first]


def _rank(value: Evaluation) -> tuple[int, int]:
    return (value.rank_score if value.rank_score is not None else value.score if value.score is not None else 2**31,
            value.score if value.score is not None else 2**31)


def _residual_vector(value: Evaluation) -> tuple[int, ...]:
    counts = (value.report or {}).get("counts", {})
    keys = ("instruction", "size", "immediate", "branch", "register", "scheduling",
            "hazard-nop", "operand-order")
    return tuple(int(counts.get(key, 0)) for key in keys) + _rank(value)


def _nondominated(items: list[tuple]) -> list[tuple]:
    """Retain residual tradeoffs instead of collapsing the beam to one scalar score."""
    kept = []
    for item in items:
        vector = _residual_vector(item[3])
        dominated = False
        for other in items:
            if other is item:
                continue
            candidate = _residual_vector(other[3])
            if all(a <= b for a, b in zip(candidate, vector)) and candidate != vector:
                dominated = True
                break
        if not dominated:
            kept.append(item)
    return kept


def descend(source: str, name: str, evaluate, families=FAMILIES, rounds: int = 8, jobs: int = 8,
            budget: int = 400, per_round: int = 160, beam: int = 4, patience: int = 2) -> dict:
    """``evaluate(source_text) -> Evaluation``.  Beam descent over single-step variants.

    Each round expands the ``beam`` best sources found so far (neutral ones included, so two
    edits that only help together are reachable), builds every unseen variant in parallel and
    keeps the best.  Stops at exact, after ``patience`` rounds without a new best score, when
    ``budget`` builds are spent, or at once when the start already has the lowest rank there is."""
    base = evaluate(source)
    if base.score is None:
        return {"error": base.error or "baseline does not build", "evaluated": 1}
    # Scores and rank scores are sums of non-negative penalties, so (0, 0) is the lowest rank, and
    # only a strictly lower rank replaces the best: from there no variant can be adopted, and the
    # search would spend its budget to return its start (round 12 base bench: 18 searches, 1,982
    # builds, on one title; the case's outcome is the same without them).
    floor = _rank(base) <= (0, 0)
    best_source, best, best_path, spent, steps, sensitivity, stale = source, base, [], 1, [], [], 0
    tried = {hashlib.sha256(source.encode()).hexdigest()}
    frontier = [(_rank(base), "", source, base, [])]
    object_digests = {base.object_digest} if base.object_digest else set()
    preprocessed_digests = {base.preprocessed_digest} if base.preprocessed_digest else set()
    source_shapes = {source_shape(source)}
    residual_states = {_residual_vector(base)}
    cache_reused = 0
    for round_number in range(1, rounds + 1):
        if best.exact or floor or spent >= budget or stale >= patience or not frontier:
            break
        pool = []
        for _score, path, parent, parent_eval, parent_edits in frontier:
            order = _family_order(parent_eval.report, families)
            children = [v for v in variants(parent, name, families)
                        if hashlib.sha256(v.source.encode()).hexdigest() not in tried]
            children.sort(key=lambda v: order.index(v.family) if v.family in order else len(order))
            for child in children[:per_round]:
                tried.add(hashlib.sha256(child.source.encode()).hexdigest())
                pool.append((path, child, parent_edits))
        pool = pool[:max(0, budget - spent)]
        if not pool:
            break
        with ThreadPoolExecutor(max(1, jobs)) as executor:
            results = list(executor.map(lambda item: evaluate(item[1].source), pool))
        spent += len(pool)
        built = [(r.score, i) for i, r in enumerate(results) if r.score is not None]
        if round_number == 1:
            table = sorted((results[i].score, pool[i][1].label, pool[i][1].family,
                            pool[i][1].semantic_review) for _s, i in built)
            sensitivity = [{"change": label, "family": family, "score": score,
                            "delta": score - base.score, "semantic_review": review}
                           for score, label, family, review in table]
        step = {"round": round_number, "variants": len(pool), "did_not_build": len(pool) - len(built),
                "neutral": sum(1 for score, _i in built if score == best.score),
                "better": sum(1 for score, _i in built if score < best.score)}
        steps.append(step)
        compiled = []
        round_objects = set()
        for score, i in built:
            result, child = results[i], pool[i][1]
            source_shapes.add(source_shape(child.source))
            edits = pool[i][2] + [{"label": child.label, "family": child.family,
                                   "semantic_review": child.semantic_review}]
            compiled.append((_rank(result), (pool[i][0] + " ; " if pool[i][0] else "") + child.label,
                             child.source, result, edits))
            if result.object_digest:
                round_objects.add(result.object_digest)
            if result.preprocessed_digest:
                preprocessed_digests.add(result.preprocessed_digest)
        unique_compiled = {}
        for item in compiled:
            key = item[3].object_digest or hashlib.sha256(item[2].encode()).hexdigest()
            unique_compiled.setdefault(key, item)
        cache_reused += len(compiled) - len(unique_compiled)
        compiled = list(unique_compiled.values())
        candidates = _nondominated(compiled)
        candidates.sort(key=lambda item: item[0])
        frontier = candidates[:beam]
        ranked = sorted(built, key=lambda item: _rank(results[item[1]]))
        new_objects = round_objects - object_digests
        current_states = {_residual_vector(item[3]) for item in frontier}
        if ranked and _rank(results[ranked[0][1]]) < _rank(best):
            score, index = ranked[0]
            best, best_source, stale = results[index], pool[index][1].source, 0
            best_item = next((item for item in frontier if item[2] == best_source), None)
            best_path = best_item[4] if best_item else pool[index][2] + [{
                "label": pool[index][1].label, "family": pool[index][1].family,
                "semantic_review": pool[index][1].semantic_review}]
            step.update({"applied": best_item[1] if best_item else pool[index][1].label,
                         "score": best.score, "rank_score": best.rank_score})
        else:
            new_states = current_states - residual_states
            stale = 0 if new_states or new_objects else stale + 1
        residual_states.update(current_states)
        object_digests.update(round_objects)
        step.update({"unique_objects": len(round_objects), "new_objects": len(new_objects),
                     "nondominated": len(candidates)})
    return {"start_score": base.score, "final_score": best.score, "exact": best.exact, "evaluated": spent,
            "steps": steps, "evidence": evidence(best.report), "sensitivity": sensitivity, "source": best_source,
            "counts": (best.report or {}).get("counts", {}),
            "start_rank_score": base.rank_score, "final_rank_score": best.rank_score,
            "edit_path": best_path, "unique_sources": len(tried),
            "unique_source_shapes": len(source_shapes),
            # Evaluators may supply a real preprocessor-output digest.  Do not present zero
            # as a measurement when the selected compiler pipeline cannot expose one.
            "unique_preprocessed": (len(preprocessed_digests) if preprocessed_digests else None),
            "unique_objects": len(object_digests),
            "object_cache_reused": cache_reused,
            "stopped_reason": ("exact" if best.exact else "floor" if floor else "budget" if spent >= budget else
                               "stagnation" if stale >= patience else "exhausted"),
            "semantic_review_required": any(item["semantic_review"] for item in best_path)}
