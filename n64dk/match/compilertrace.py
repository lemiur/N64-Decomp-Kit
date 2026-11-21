"""Compiler-internal observability: normalized traces of one build, causal links, probe reduction
(round 12 product item 3).

A residual says *that* two builds differ; the compiler's own dumps say *where* the difference was
made.  GCC 2.x writes one dump per pass with ``-d<letters>`` (``-da``: all of them), annotates its
assembly with the RTL insn and pattern behind each instruction with ``-dp``, and keeps the
preprocessed source with ``-save-temps``.  This module

* decides per recipe whether a white-box trace may be asked for (``trace_mode``): only for the
  GCC-family recipes the kit rebuilds from source (KMC GCC 2.7.2, GCC 2.8.1, EGCS 1.1.2, the
  rebuilt SN 2.7.2 ``cc1``).  The SN ``cc1n64.exe`` builds and every IDO/MIPSpro recipe are traced
  black-box: preprocessed source where the kit's own pipeline writes one, assembler input, object,
  relocations and digests; no undocumented switch is passed to a proprietary binary and no binary
  is modified;
* collects a trace (``collect``): the build is made twice through the kit's build path (or an
  injected one), plain and with the trace switches; the code of the two objects must be identical
  for the trace to describe the plain build (``identical_code``); a compiler that refuses the
  switches is recorded (``refused``) and the trace falls back to black-box;
* parses the artefacts into one schema (``SCHEMA``): preprocessed-source line map, RTL per pass
  (insn uid, kind, source line, registers), register allocation (``gcc27dump``), final assembly
  with the insn behind each instruction, assembler input, object functions and relocations, and
  digests of everything;
* links source lines to RTL insns, pseudos, hard registers, assembly and object offsets where the
  dumps allow (``causal_links``), and says where they do not;
* reduces a source to a minimal self-contained probe that keeps a difference (``reduce``: delta
  debugging over statements, then expression simplification, with an injected predicate).

Diagnostics only: a trace never changes a build, a score or a verdict.
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
from pathlib import Path

SCHEMA = "n64dk.compilertrace/1"

# GCC 2.x/EGCS ``-d`` letters, the suffix of the file each writes and the pass, in pass order
# (GCC 2.7.2 / 2.8.1 / EGCS 1.1 documentation).  The collector reads whatever files appear; a
# suffix not listed is kept as an unknown pass.
PASSES: tuple[tuple[str, str, str], ...] = (
    ("r", "rtl", "RTL generation"), ("j", "jump", "jump optimisation"), ("s", "cse", "common subexpressions"),
    ("F", "addressof", "addressof (EGCS)"), ("G", "gcse", "global CSE (EGCS)"), ("L", "loop", "loop optimisation"),
    ("t", "cse2", "second CSE"), ("b", "bp", "branch probabilities"), ("f", "flow", "life analysis"),
    ("c", "combine", "instruction combination"), ("S", "sched", "first scheduling (sched1)"),
    ("l", "lreg", "local register allocation"), ("g", "greg", "global allocation and reload"),
    ("w", "flow2", "second life analysis (EGCS)"), ("R", "sched2", "second scheduling (sched2)"),
    ("J", "jump2", "second jump optimisation"), ("d", "dbr", "delayed-branch scheduling (reorg)"),
    ("k", "stack", "register stack (x86 only)"))
SUFFIXES = {suffix: (order, letter, title) for order, (letter, suffix, title) in enumerate(PASSES)}
FINAL_PASSES = ("dbr", "jump2", "sched2", "flow2", "greg")      # the latest dump present names hard registers

# lanes whose compiler the kit rebuilds from source or drives as a stock GCC: (style, trace switches)
WHITE_BOX = {"kmc": "driver", "custom:gcc281pm": "driver", "custom:egcs112": "driver", "custom:gcc272snew": "cc1"}
TRACE_FLAGS = {"driver": ("-da", "-dp", "-save-temps"), "cc1": ("-da", "-dp")}
MAX_INSNS = 4000                           # per pass and function kept in a trace
TEXT = 200                                 # characters of an insn's RTL kept


def trace_mode(lane: str) -> str:
    """``driver`` / ``cc1`` (white-box: the trace switches may be passed) or ``black-box``."""
    name = lane.split()[0] if lane else ""
    if name.startswith("kmc"):
        return WHITE_BOX["kmc"]
    return WHITE_BOX.get(name, "black-box")


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


# ---------------------------------------------------------------------------------------------
# parsers

_MARKER = re.compile(r'^#\s*(?:line\s+)?(\d+)\s+"([^"]*)"')


def line_map(preprocessed: str) -> list[tuple[int, str, int]]:
    """``(line of the .i file, source file, source line)`` for every non-marker line, from the
    ``# N "file"`` markers the preprocessor writes."""
    out, current, number = [], "", 0
    for index, text in enumerate(preprocessed.splitlines(), 1):
        found = _MARKER.match(text)
        if found:
            number, current = int(found.group(1)), found.group(2)
            continue
        if current:
            out.append((index, current, number))
        number += 1
    return out


def _forms(text: str):
    """Top-level parenthesised forms (start offset, text) of a dump, comment lines skipped."""
    depth, start, index, string = 0, -1, 0, False
    size = len(text)
    while index < size:
        char = text[index]
        if string:
            if char == "\\":
                index += 1
            elif char == '"':
                string = False
        elif char == '"':
            string = True
        elif char == ";" and depth == 0:
            end = text.find("\n", index)
            index = size if end < 0 else end
            continue
        elif char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")" and depth:
            depth -= 1
            if depth == 0:
                yield start, text[start:index + 1]
        index += 1


_FUNCTION = re.compile(r"^;; Function (\S+)", re.MULTILINE)
_HEAD = re.compile(r"^\((insn|jump_insn|call_insn|code_label|note|barrier)(?:/\w+)*\s+(\d+)\s+(\d+)\s+(\d+)")
_LINE_NOTE = re.compile(r'^\(note(?:/\w+)*\s+\d+\s+\d+\s+\d+\s+\(?"([^"]+)"\)?\s+(\d+)\s*\)')
_NOTE_KIND = re.compile(r"(NOTE_INSN_\w+)")
_REG = re.compile(r"\(reg(?:/[a-z]+)*:([A-Z][A-Z0-9]*)\s+(\d+)")
_CODE = re.compile(r"\)\s+(-?\d+)\s+\{?")


def parse_rtl(text: str) -> dict[str, list[dict]]:
    """``{function: [insn]}`` from one pass dump.  An insn is ``{uid, kind, file, line, regs, code,
    text}``; ``line`` is the line of the last line note before it (None before the first);
    notes and barriers are kept only as line carriers."""
    marks = [(match.start(), match.group(1)) for match in _FUNCTION.finditer(text)]
    out: dict[str, list[dict]] = {}
    for number, (begin, name) in enumerate(marks):
        end = marks[number + 1][0] if number + 1 < len(marks) else len(text)
        insns: list[dict] = []
        source, line = None, None
        for _offset, form in _forms(text[begin:end]):
            head = _HEAD.match(form)
            if head is None:
                continue
            kind, uid = head.group(1), int(head.group(2))
            if kind == "note":
                note = _LINE_NOTE.match(form)
                if note is not None:
                    source, line = note.group(1), int(note.group(2))
                continue
            if kind == "barrier" or len(insns) >= MAX_INSNS:
                continue
            regs = sorted({int(found.group(2)) for found in _REG.finditer(form)})
            code = None
            if kind != "code_label":
                tail = _CODE.findall(form)
                code = int(tail[-1]) if tail else None
            insns.append({"uid": uid, "kind": kind, "file": source, "line": line, "regs": regs, "code": code,
                          "text": " ".join(form.split())[:TEXT]})
        out[name] = insns
    return out


_DP = re.compile(r"#\s*(\d+)\s+([A-Za-z_][\w*]*)(?:/(\d+))?")
_LOC = re.compile(r"^\s*\.loc\s+\d+\s+(\d+)")
_STABN = re.compile(r"^\s*\.stabn\s+68\s*,\s*0\s*,\s*(\d+)")
_ENT = re.compile(r"^\s*\.ent\s+([\w.$]+)")
_END = re.compile(r"^\s*\.end\s+([\w.$]+)")


def parse_asm(text: str) -> dict[str, list[dict]]:
    """``{function: [row]}`` from GCC MIPS assembly (``.ent``/``.end`` bounds).  A row is an
    instruction line: ``{text, mnemonic, uid, pattern, alternative, line}``.  With ``-dp`` the
    comment after an insn's last line names its uid and pattern; lines before it without a comment
    belong to the same insn (a multi-line template).  ``line`` comes from ``.loc``/``.stabn``."""
    out: dict[str, list[dict]] = {}
    current, rows, pending, line = None, [], [], None
    for raw in text.splitlines():
        entered = _ENT.match(raw)
        if entered:
            current, rows, pending = entered.group(1), [], []
            continue
        if current is None:
            continue
        if _END.match(raw):
            for row in pending:
                row["uid"] = None
            out[current] = rows
            current = None
            continue
        for pattern in (_LOC, _STABN):
            found = pattern.match(raw)
            if found:
                line = int(found.group(1))
        body = raw.split("#", 1)[0].strip()
        if not body or body.startswith(".") or body.endswith(":") or not raw[:1].isspace():
            continue
        row = {"text": " ".join(body.split()), "mnemonic": body.split()[0], "uid": None, "pattern": None,
               "alternative": None, "line": line}
        rows.append(row)
        pending.append(row)
        comment = _DP.search(raw) if "#" in raw else None
        if comment is not None:
            for item in pending:
                item["uid"] = int(comment.group(1))
                item["pattern"] = comment.group(2)
                item["alternative"] = int(comment.group(3)) if comment.group(3) else None
            pending = []
    return out


def parse_regalloc(local_dump: str, global_dump: str = "") -> dict:
    """Register allocation through the kit's existing GCC 2.7 parser (``gcc27dump``): pseudo
    priorities and the hard register each pseudo received."""
    from . import gcc27dump
    explained = gcc27dump.explain(local_dump, global_dump)
    assigned = {str(row["number"]): row["hard_register"] for row in explained["pseudos"]
                if row["hard_register"] is not None}
    assigned.update({str(pseudo): hard for pseudo, hard in gcc27dump._global_assignments(global_dump).items()})
    return {"pseudos": explained["pseudos"], "assignments": assigned, "priority_ties": explained["priority_ties"],
            "diagnostics": explained["diagnostics"]}


def object_summary(elf) -> dict:
    """Functions (offset, size, words) and relocations of a relocatable, with digests."""
    from .passprobe import ObjectView
    view = ObjectView(elf)
    functions = {}
    for name, (index, start, end, _sized) in view.ranges.items():
        words = view.words(name) or []
        functions[name] = {"section": elf.sections[index].name, "offset": start, "size": end - start,
                           "words": [f"{word:08X}" for word in words]}
    relocations = []
    for section in elf.sections:
        if section.type not in (1,) or not section.data:
            continue
        for offset, kind, symbol, addend in elf.relocations(section):
            sym = elf.symbols[symbol] if symbol < len(elf.symbols) else None
            name = sym.name if sym is not None and sym.name else (
                elf.sections[sym.section].name if sym is not None and 0 < sym.section < len(elf.sections) else "")
            relocations.append({"section": section.name, "offset": offset, "type": kind, "symbol": name,
                                "addend": addend})
    text = elf.section(".text")
    return {"sha256": sha256(elf.data), "text_sha256": sha256(text.data if text is not None else b""),
            "functions": functions, "relocations": relocations}


# ---------------------------------------------------------------------------------------------
# collection

def _artefacts(folder: Path) -> dict:
    """Classify what a build left in ``folder``: preprocessed source, pass dumps, assembly
    (compiler output and assembler input), objects."""
    found: dict = {"preprocessed": [], "dumps": {}, "unknown_dumps": [], "asm": [], "raw_asm": [], "objects": []}
    if not folder.is_dir():
        return found
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        name = path.name
        suffix = name.rsplit(".", 1)[-1] if "." in name else ""
        if name.endswith(".raw.s"):
            found["raw_asm"].append(path)
        elif suffix == "s":
            found["asm"].append(path)
        elif suffix == "i":
            found["preprocessed"].append(path)
        elif suffix in ("o", "obj"):
            found["objects"].append(path)
        elif suffix in SUFFIXES and name.count(".") >= 2:
            found["dumps"][suffix] = path
        elif re.fullmatch(r"[a-z]+[0-9]?", suffix) and name.count(".") >= 2 and suffix not in ("c", "h", "ld"):
            found["unknown_dumps"].append(path)
    return found


def _text_words(object_path: Path | None) -> list[int] | None:
    if object_path is None or not Path(object_path).is_file():
        return None
    from ..elf import Elf
    elf = Elf(Path(object_path).read_bytes())
    text = elf.section(".text")
    data = text.data if text is not None else b""
    return [struct.unpack_from(">I", data, offset)[0] for offset in range(0, len(data) - len(data) % 4, 4)]


def collect(source, lane: str, flags, cfg, work, *, build=None, line_info: bool = False) -> dict:
    """A trace of ``source`` built under ``lane`` + ``flags``.  ``build(source, spec, cfg, workdir)``
    is the kit's ``pipeline.build`` unless injected.  ``work`` must be private to this call; the
    object cache must not hold the traced build (use a scratch project), or no dump is written.
    ``line_info`` adds ``-g`` for ``.loc`` lines (the code must stay identical: checked)."""
    from ..elf import Elf
    from .pipeline import BuildError, split_flags
    if build is None:
        from .pipeline import build
    source, work = Path(source), Path(work)
    flags = list(flags)
    mode = trace_mode(lane)
    trace: dict = {"schema": SCHEMA, "lane": lane, "flags": flags, "mode": mode, "trace_flags": [],
                   "refused": None, "error": None, "identical_code": None, "source": {"name": source.name,
                   "sha256": sha256(source.read_bytes())}, "limits": []}
    plain_dir, traced_dir = work / "plain", work / "traced"
    try:
        plain = Path(build(source, split_flags(flags, lane), cfg, plain_dir))
    except (BuildError, OSError, RuntimeError) as error:
        trace["error"] = str(error)[-400:]
        return trace
    extra = list(TRACE_FLAGS.get(mode, ())) + (["-g"] if line_info and mode != "black-box" else [])
    traced, folder = plain, plain_dir
    if extra:
        try:
            traced = Path(build(source, split_flags(flags + extra, lane), cfg, traced_dir))
            folder = traced_dir
            trace["trace_flags"] = extra
        except (BuildError, OSError, RuntimeError) as error:
            trace["refused"] = str(error)[-400:]
            trace["mode"] = "black-box"
            trace["limits"].append("the compiler refused the trace switches: black-box trace of the plain build")
    plain_words, traced_words = _text_words(plain), _text_words(traced)
    trace["identical_code"] = plain_words is not None and plain_words == traced_words
    if not trace["identical_code"]:
        trace["limits"].append("the traced object's code differs from the plain build: the dumps describe "
                               "another compilation")
    found = _artefacts(folder)
    if trace["trace_flags"] and not found["dumps"]:
        trace["limits"].append("no pass dump was written (an object-cache hit, or the switches were ignored)")
    if found["preprocessed"]:
        text = found["preprocessed"][0].read_text(errors="replace")
        trace["preprocessed"] = {"name": found["preprocessed"][0].name, "sha256": sha256(text),
                                 "lines": text.count("\n"), "line_map": line_map(text)}
    passes = []
    for suffix, path in sorted(found["dumps"].items(), key=lambda item: SUFFIXES[item[0]][0]):
        order, letter, title = SUFFIXES[suffix]
        text = path.read_text(errors="replace")
        passes.append({"name": suffix, "letter": letter, "title": title, "order": order, "sha256": sha256(text),
                       "functions": parse_rtl(text)})
    trace["passes"] = passes
    trace["unknown_dumps"] = [path.name for path in found["unknown_dumps"]]
    if "lreg" in found["dumps"]:
        trace["regalloc"] = parse_regalloc(found["dumps"]["lreg"].read_text(errors="replace"),
                                           found["dumps"]["greg"].read_text(errors="replace")
                                           if "greg" in found["dumps"] else "")
    compiler_asm = found["raw_asm"][:1] or found["asm"][:1]
    if compiler_asm:
        text = compiler_asm[0].read_text(errors="replace")
        trace["asm"] = {"name": compiler_asm[0].name, "sha256": sha256(text), "functions": parse_asm(text)}
    assembler_input = [path for path in found["asm"] if path not in compiler_asm]
    if assembler_input:
        trace["assembler_input"] = {"name": assembler_input[0].name,
                                    "sha256": sha256(assembler_input[0].read_bytes())}
    trace["object"] = object_summary(Elf(traced.read_bytes()))
    if trace["mode"] == "black-box":
        trace["limits"].append("black-box: no pass dumps; links stop at the assembly and the object")
    return trace


# ---------------------------------------------------------------------------------------------
# causal links

def _first_pseudo(trace: dict) -> int:
    """The lowest pseudo register number: from the allocation dump, else 64 (MIPS hard registers
    are 0-31 integer, 32-63 float; specials follow)."""
    pseudos = [row["number"] for row in (trace.get("regalloc") or {}).get("pseudos", [])]
    return min(pseudos) if pseudos else 64


def causal_links(trace: dict) -> dict:
    """Per function: ``[{line, uid, pseudos, hard, asm, offsets}]`` linking source lines to RTL
    insns, the pseudos they used before allocation, the hard registers after it, the assembly
    instructions they became and (when the assembly maps one-to-one onto the object) the byte
    offsets; plus ``limits`` saying which links could not be made."""
    passes = sorted(trace.get("passes") or [], key=lambda item: item["order"])
    first_pseudo = _first_pseudo(trace)
    assignments = {int(k): v for k, v in ((trace.get("regalloc") or {}).get("assignments") or {}).items()}
    asm = (trace.get("asm") or {}).get("functions") or {}
    objects = ((trace.get("object") or {}).get("functions")) or {}
    names = sorted({name for item in passes for name in item["functions"]} | set(asm))
    out: dict = {"functions": {}, "limits": list(trace.get("limits") or [])}
    if not passes:
        out["limits"].append("no RTL dump: source lines cannot be followed into the compiler")
    for name in names:
        lines: dict[int, tuple] = {}
        pseudos: dict[int, set] = {}
        hard: dict[int, list] = {}
        final = None
        for item in passes:
            insns = item["functions"].get(name) or []
            for insn in insns:
                if insn["uid"] not in lines and insn["line"] is not None:
                    lines[insn["uid"]] = (insn["file"], insn["line"])
                pseudos.setdefault(insn["uid"], set()).update(r for r in insn["regs"] if r >= first_pseudo)
            if item["name"] in FINAL_PASSES and insns:
                final = item
        if final is not None:
            for insn in final["functions"].get(name) or []:
                hard[insn["uid"]] = [r for r in insn["regs"] if r < first_pseudo]
        rows = asm.get(name) or []
        by_uid: dict = {}
        for index, row in enumerate(rows):
            by_uid.setdefault(row["uid"], []).append(index)
        words = len((objects.get(name) or {}).get("words") or [])
        one_to_one = bool(rows) and words == len(rows)
        links, limits = [], []
        uids = sorted(set(lines) | {uid for uid in by_uid if uid is not None},
                      key=lambda uid: ((lines.get(uid) or ("", 1 << 30))[1], uid))
        for uid in uids:
            source = lines.get(uid)
            used = sorted(pseudos.get(uid, set()))
            links.append({"file": source[0] if source else None, "line": source[1] if source else None, "uid": uid,
                          "pseudos": used, "assigned": {str(p): assignments[p] for p in used if p in assignments},
                          "hard": hard.get(uid, []), "asm": [rows[i]["text"] for i in by_uid.get(uid, [])],
                          "offsets": [4 * i for i in by_uid.get(uid, [])] if one_to_one else []})
        if rows and not any(row["uid"] is not None for row in rows):
            limits.append("the assembly has no -dp annotations: instructions are not tied to insns")
        if rows and not one_to_one:
            limits.append(f"{len(rows)} assembly instruction(s) for {words} object word(s): macros or delay-slot "
                          "fills change the count, so no byte offsets are claimed")
        if final is None:
            limits.append("no post-allocation dump: hard registers unknown")
        out["functions"][name] = {"links": links, "limits": limits}
    return out


# ---------------------------------------------------------------------------------------------
# minimal-source reduction

def units(source: str) -> list[str]:
    """Statement-sized pieces of C text whose concatenation is the text: a piece ends after
    ``;``, ``{`` or ``}`` outside strings and comments; a preprocessor line is a piece."""
    out, piece, index, size = [], [], 0, len(source)
    at_line_start = True
    while index < size:
        char = source[index]
        if at_line_start and char == "#":
            end = index
            while True:
                end = source.find("\n", end)
                if end < 0:
                    end = size
                    break
                if source[end - 1] != "\\":
                    end += 1
                    break
                end += 1
            if "".join(piece).strip():
                out.append("".join(piece))
                piece = []
            piece.append(source[index:end])
            out.append("".join(piece))
            piece, index, at_line_start = [], end, True
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            end = size if end < 0 else end + 2
            piece.append(source[index:end])
            index = end
            continue
        if source.startswith("//", index):
            end = source.find("\n", index)
            end = size if end < 0 else end
            piece.append(source[index:end])
            index = end
            continue
        if char in "\"'":
            end = index + 1
            while end < size and source[end] != char:
                end += 2 if source[end] == "\\" else 1
            piece.append(source[index:end + 1])
            index = end + 1
            at_line_start = False
            continue
        piece.append(char)
        index += 1
        if char == "\n":
            at_line_start = True
        elif not char.isspace():
            at_line_start = False
        if char in ";{}":
            following = source.find("\n", index)
            if following >= 0 and not source[index:following].strip():
                piece.append(source[index:following + 1])
                index, at_line_start = following + 1, True
            out.append("".join(piece))
            piece = []
    if piece:
        out.append("".join(piece))
    return out


def _ddmin(count: int, test, budget: list[int]) -> list[int]:
    """Zeller's ddmin over indices ``0..count-1``: a smallest subset (1-minimal when the budget
    allows) for which ``test(subset)`` holds; ``test`` of the full set is assumed true."""
    items, n = list(range(count)), 2
    while len(items) >= 2 and budget[0] > 0:
        chunk = math.ceil(len(items) / n)
        subsets = [items[k:k + chunk] for k in range(0, len(items), chunk)]
        reduced = False
        for subset in subsets:
            if budget[0] <= 0:
                break
            if test(subset):
                items, n, reduced = subset, 2, True
                break
        if not reduced:
            for subset in subsets:
                if budget[0] <= 0:
                    break
                complement = [item for item in items if item not in subset]
                if complement and test(complement):
                    items, n, reduced = complement, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), 2 * n)
    return items


_NUMBER = re.compile(r"(?<![\w.])(0[xX][0-9A-Fa-f]+|\d+)(?![\w.])")
_BINARY = re.compile(r"\b([A-Za-z_]\w*|\d+)\s*(<<|>>|[-+*/%&|^])\s*([A-Za-z_]\w*|\d+)\b")


def _simplifications(piece: str):
    for found in _NUMBER.finditer(piece):
        if found.group(0) in ("0", "1"):
            continue                                  # already simplest: no 0 <-> 1 oscillation
        for value in ("0", "1"):
            yield piece[:found.start()] + value + piece[found.end():]
    for found in _BINARY.finditer(piece):
        for keep in (found.group(1), found.group(3)):
            yield piece[:found.start()] + keep + piece[found.end():]


def reduce(source: str, keeps_difference, *, max_tests: int = 400, name: str = "") -> dict:
    """A minimal self-contained probe of ``source`` for which ``keeps_difference(text)`` still holds
    (the predicate builds the text itself, so a reduction that does not compile is rejected by it).
    Statement pieces are removed by delta debugging, then numbers and binary operations inside the
    remaining pieces are simplified one at a time.  At most ``max_tests`` predicate calls."""
    pieces = units(source)
    cache: dict[str, bool] = {}
    budget = [max_tests]
    calls = [0]

    def holds(text: str) -> bool:
        if text in cache:
            return cache[text]
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        calls[0] += 1
        try:
            cache[text] = bool(keeps_difference(text))
        except Exception:                            # noqa: BLE001 - a crashing predicate rejects the text
            cache[text] = False
        return cache[text]

    if not holds(source):
        return {"source": source, "kept": False, "tests": calls[0], "units_before": len(pieces),
                "units_after": len(pieces), "budget_exhausted": False,
                "reason": "the predicate does not hold for the original source"}
    kept = _ddmin(len(pieces), lambda subset: holds("".join(pieces[i] for i in subset)), budget)
    current = [pieces[i] for i in kept]
    changed = True
    while changed and budget[0] > 0:
        changed = False
        for position, piece in enumerate(current):
            for candidate in _simplifications(piece):
                trial = current[:position] + [candidate] + current[position + 1:]
                if holds("".join(trial)):
                    current, changed = trial, True
                    break
            if changed or budget[0] <= 0:
                break
    body = "".join(current)
    header = (f"/* reduced probe{(' ' + name) if name else ''}: n64dk.match.compilertrace.reduce, "
              f"{len(current)} of {len(pieces)} pieces, original sha256 {sha256(source)[:16]} */\n")
    return {"source": header + body, "body": body, "kept": True, "tests": calls[0], "units_before": len(pieces),
            "units_after": len(current), "budget_exhausted": budget[0] <= 0}


def residual_predicate(compile_words, target_words, component: str, **attribute_options):
    """A ``keeps_difference`` for ``reduce``: the text builds (``compile_words(text)`` returns the
    function's words, or None) and ``passprobe.attribute`` still ranks ``component`` first among the
    components not ruled out.  ``target_words`` come from the same reduction applied to the other
    side, or are fixed."""
    from .passprobe import attribute

    def predicate(text: str) -> bool:
        words = compile_words(text)
        if words is None:
            return False
        target = target_words(text) if callable(target_words) else target_words
        if target is None or list(target) == list(words):
            return False
        live = [row["component"] for row in attribute(target, words, **attribute_options) if not row["ruled_out"]]
        return bool(live) and live[0] == component
    return predicate
