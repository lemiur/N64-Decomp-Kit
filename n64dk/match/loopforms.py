"""Loop forms for GCC 2.7.2 register ranking.

GCC 2.7.2 ranks pseudo registers for allocation by ``floor_log2(refs) * refs / live_length``
(ties to the lower pseudo number), and ``refs`` counts each reference once per loop depth.  A
loop the compiler does not see as a loop weights nothing: a loop built with a label and a
backward ``goto`` is such a loop, ``while (1) { ... }`` and ``do { } while`` are not.  Two drafts
with the same statements can therefore come out with the same instructions and a consistent
exchange of registers, and only the loop's form decides which (Sesame Street, strcpy of the KMC
C library: the word loop written with ``goto`` put ``d`` and ``w`` in each other's registers;
written as ``while (1)`` it matched).

``variants`` rewrites in both directions, so a draft written either way is offered the other:

* ``L: { S; if (C) goto L; } return X;`` -> ``while (1) { S; if (!(C)) return X; }`` (and the
  same with the block's declarations moved to the top of the function), ``do { S } while (C);``
  and ``while (1) { S; if (!(C)) break; }``;
* ``while (1) { S; if (!(C)) return X; }`` and ``do { S } while (C);`` -> the goto form.

The statements are not changed, so each form computes what the draft computed; which one is the
original is decided by building it."""
from __future__ import annotations

import re

from . import iterate as _iterate

_DECL_INIT = re.compile(r"^([ \t]*)((?:(?:unsigned|signed|const|volatile)\s+)*[A-Za-z_]\w*(?:\s*\*+\s*|\s+))(\w+) = ([^;]+);")


def _closing(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        depth += text[index] == "{"
        depth -= text[index] == "}"
        if depth == 0:
            return index
    return None


def _indent(text: str, by: str) -> str:
    return "\n".join((by + line) if line.strip() else line for line in text.split("\n"))


def _strip_condition(text: str) -> str:
    return text[2:-1] if text.startswith("!(") and text.endswith(")") and _balanced(text[2:-1]) else f"!({text})"


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        depth += char == "("
        depth -= char == ")"
        if depth < 0:
            return False
    return depth == 0


def _hoist(head_end: int, body: str, inner: str) -> tuple[str, str] | None:
    """Move ``T x = e;`` declarations at the start of ``inner`` to the function's top."""
    lines, moved = inner.split("\n"), []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        found = _DECL_INIT.match(line)
        if not found:
            break
        moved.append(f"    {found.group(2).rstrip()} {found.group(3)};" if not found.group(2).rstrip().endswith("*")
                     else f"    {found.group(2).rstrip()}{found.group(3)};")
        lines[index] = f"{found.group(1)}{found.group(3)} = {found.group(4)};"
    if not moved:
        return None
    return "\n".join(moved) + "\n", "\n".join(lines)


def goto_to_loop(source: str, name: str) -> list[tuple[str, str]]:
    span = _iterate._function_span(source, name)
    if span is None:
        return []
    _start, body_start, end = span
    body = source[body_start:end]
    out = []
    for label in re.finditer(r"(?m)^([ \t]*)(\w+):[ \t]*\n", body):
        lab, indent = label.group(2), label.group(1)
        gotos = list(re.finditer(rf"\bgoto {re.escape(lab)};", body))
        if len(gotos) != 1 or gotos[0].start() < label.end():
            continue
        after = body[label.end():]
        lead = re.match(r"[ \t]*\{", after)
        last = re.compile(rf"(?s)^(.*)\n([ \t]*)if \(([^\n]*)\) goto {re.escape(lab)};[ \t]*\n?([ \t]*)$")
        if lead:
            opening = label.end() + lead.end() - 1
            closing = _closing(body, opening)
            if closing is None:
                continue
            found = last.match(body[opening + 1:closing])
            stop = closing + 1
        else:
            line = re.compile(rf"(?m)^([ \t]*)if \((.*)\) goto {re.escape(lab)};[ \t]*$").search(body, label.end())
            if line is None or line.start() != body.rfind("\n", 0, gotos[0].start()) + 1:
                continue
            between = body[label.end():line.start()].rstrip("\n")
            if between.count("{") != between.count("}"):
                continue
            found = last.match("\n" + _indent(between, "    ") + "\n" + line.group(1) + "    if (" + line.group(2)
                               + ") goto " + lab + ";\n")
            stop = line.end()
        if found is None or not _balanced(found.group(3)):
            continue
        statements, condition = found.group(1).rstrip(), found.group(3)
        tail = body[stop:]
        returning = re.match(r"\s*\n([ \t]*)return\b([^;]*);\s*$", tail)
        prefix = body[:label.start()]
        top, moved = prefix, None
        hoisted = _hoist(0, body, statements)
        if hoisted is not None:
            declarations, moved = hoisted
            # after the function's own declaration block
            decl_end = re.match(r"(\n?(?:[ \t]+[^\n;{}()]*(?:=[^\n;]*)?;[ \t]*\n)*)", prefix)
            top = prefix[:decl_end.end()] + declarations + prefix[decl_end.end():]
        if returning:
            exit_ = f"{indent}    if ({_strip_condition(condition)}) return{returning.group(2)};\n"
            out.append(("loop-form: goto -> while (1) with return",
                        prefix + f"{indent}while (1) {{{statements}\n{exit_}{indent}}}\n"))
            if moved is not None:
                out.append(("loop-form: goto -> while (1) with return, declarations hoisted",
                            top + f"{indent}while (1) {{{moved}\n{exit_}{indent}}}\n"))
        # the condition is read after the block: its declarations must stand outside it
        do_top, do_body = (top, moved) if moved is not None else (prefix, statements)
        out.append(("loop-form: goto -> do-while",
                    do_top + f"{indent}do {{{do_body}\n{indent}}} while ({condition});\n" + tail.lstrip("\n")))
        out.append(("loop-form: goto -> while (1) with break",
                    prefix + f"{indent}while (1) {{{statements}\n{indent}    if ({_strip_condition(condition)}) break;\n"
                             f"{indent}}}\n" + tail.lstrip("\n")))
    return [(label, source[:body_start] + text + source[end:]) for label, text in out]


def loop_to_goto(source: str, name: str) -> list[tuple[str, str]]:
    span = _iterate._function_span(source, name)
    if span is None:
        return []
    _start, body_start, end = span
    body = source[body_start:end]
    taken = set(re.findall(r"(?m)^\s*(\w+):", body))
    label = next(n for n in ["loop"] + [f"loop{i}" for i in range(2, 64)] if n not in taken)
    out = []
    for found in re.finditer(r"(?m)^([ \t]*)while \(1\) \{", body):
        indent = found.group(1)
        closing = _closing(body, found.end() - 1)
        if closing is None:
            continue
        inner = body[found.end():closing]
        exit_ = re.match(r"(?s)^(.*)\n[ \t]*if \(([^\n]*)\) return\b([^;]*);[ \t]*\n[ \t]*$", inner)
        if exit_ is None or not _balanced(exit_.group(2)) or re.search(r"\b(?:break|continue)\b", exit_.group(1)):
            continue
        text = (body[:found.start()] + f"{indent}{label}:\n{indent}{{{exit_.group(1)}\n{indent}    if "
                f"({_strip_condition(exit_.group(2))}) goto {label};\n{indent}}}\n{indent}return{exit_.group(3)};"
                + body[closing + 1:])
        out.append(("loop-form: while (1) -> goto", text))
    for found in re.finditer(r"(?m)^([ \t]*)do \{", body):
        indent = found.group(1)
        closing = _closing(body, found.end() - 1)
        if closing is None:
            continue
        cond = re.match(r"\s*while \((.*)\);", body[closing + 1:])
        inner = body[found.end():closing]
        if cond is None or not _balanced(cond.group(1)) or re.search(r"\b(?:break|continue)\b", inner):
            continue
        text = (body[:found.start()] + f"{indent}{label}:\n{indent}{{{inner.rstrip()}\n{indent}    if ({cond.group(1)}) "
                f"goto {label};\n{indent}}}" + body[closing + 1 + cond.end():])
        out.append(("loop-form: do-while -> goto", text))
    return [(lab, source[:body_start] + text + source[end:]) for lab, text in out]


def variants(source: str, name: str) -> list[tuple[str, str]]:
    """Both directions; only texts that differ from ``source``, each once."""
    out, seen = [], {source}
    for label, text in goto_to_loop(source, name) + loop_to_goto(source, name):
        if text not in seen:
            seen.add(text)
            out.append((label, text))
    return out
