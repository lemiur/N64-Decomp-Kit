"""Two-track source: the matching source and a maintainable view side by side (round 12, item 11).

The **matching source** is what the strict verifier built.  It is the only build input, it is
stored byte for byte, and nothing here ever rewrites it.  The **maintainable view** is derived from
it for people: generated names replaced by the best known names, provisional structure
declarations, comments, normalised statement forms and reconstruction scaffolding removed.  The
two are written side by side (``<stem>.match.c``, ``<stem>.view.c``) with a **transformation
map** (``<stem>.map.json``) that lists every step and what it may have changed:

``presentation-only``    nothing a compiler sees changed (comments, layout)
``exact-preserving``     the compiler sees a change whose output bytes are known to be equal
                         (only after an injected strict build of the view says so)
``behaviour-preserving`` behaviour is the same, the bytes may differ (a consistent rename,
                         an e-graph rewrite under its side conditions, removing frame padding)
``unverified``           may change what is built or what it means (new declarations, removing
                         ``LDSYM`` markers the symbol resolver reads)

The view's class is the weakest class of its steps.  A view is never evidence: it is not
compiled for verification, it never replaces the matching source, and a ``verify`` callback can
only *report* that a strict build of the whole view is byte-identical (``verified: exact``).

Names come from ``n64dk.analysis.naming`` (``NAMES.tsv``): the field a display name came from is
kept, and a generated ``sem_`` name is always shown as generated, never as an original name.
Provisional structures come from ``n64dk.analysis.typeflow.declarations``.  Statement forms come
from the e-graph (``n64dk.match.egraph``), restricted to readability rules."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import astprep
from .transform_classes import SCAFFOLD_MARK

SCHEMA = "n64dk-two-track/1"
CLASSES = ("presentation-only", "exact-preserving", "behaviour-preserving", "unverified")
RANK = {name: rank for rank, name in enumerate(CLASSES)}
NAME_FIELDS = ("original", "sdk", "donor", "semantic", "user")
_LABEL = {"original": "original name from the title's development records",
          "sdk": "library name from an SDK signature",
          "donor": "name proposed by a source donor (another program's name for this code)",
          "semantic": "GENERATED name (sem_): a description of a role, not an original name",
          "user": "name supplied by the operator"}
_IDENT = re.compile(r"[A-Za-z_]\w*\Z")
_CELL = re.compile(r"^(?P<name>\S+) \[(?P<source>[^\]]*)\]$")
_MARK = r"/\*\s*LDSYM\s+[A-Za-z_.$][\w.$]*\s*=\s*(?:0x[0-9A-Fa-f]+|\d+)\s*\*/"
_LDSYM_LINE = re.compile(r"(?m)^[ \t]*" + _MARK + r"[ \t]*\n")         # a marker alone on its line
_LDSYM = re.compile(r"[ \t]*" + _MARK)                                   # a marker after code: line kept
_PAD = re.compile(r"(?m)^[^\n]*" + re.escape(SCAFFOLD_MARK) + r"[^\n]*\n?")
_ASM_NOP = re.compile(r'(?m)^[ \t]*(?:__asm__|asm)\s*(?:__volatile__|volatile)?\s*\(\s*"\s*nop\s*"\s*\)\s*;[ \t]*\n?')
_EXTERN = re.compile(r"^extern\b[^;]*?\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])*\s*;")
# e-graph rules whose result reads more plainly; ``~wrap`` rules are never used for a view
READABLE = frozenset({"op-assign", "incdec", "notnot-ne0", "not-compare", "eq0-not", "cast-drop", "assign-cast-drop",
                      "index0-deref", "truth-drop-ne0", "ne0-drop"})


class TwoTrackError(ValueError):
    pass


@dataclass
class Step:
    kind: str            # egraph, scaffolding, names, structs, comments
    cls: str             # one of CLASSES
    detail: str
    before: str = ""
    after: str = ""
    line: int | None = None

    def as_dict(self) -> dict:
        return {"kind": self.kind, "class": self.cls, "detail": self.detail, "before": self.before,
                "after": self.after, "line": self.line}


@dataclass
class TwoTrack:
    name: str
    matching: str
    view: str
    steps: list[Step] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    verified: str | None = None          # "exact" / "different" from an injected strict build of the view

    @property
    def view_class(self) -> str:
        return max((s.cls for s in self.steps), key=RANK.get, default="presentation-only")

    def build_input(self) -> str:
        """What a build or a verification must use: always the matching source."""
        return self.matching

    def as_map(self) -> dict:
        return {"schema": SCHEMA, "function": self.name,
                "matching_sha256": hashlib.sha256(self.matching.encode()).hexdigest(),
                "view_sha256": hashlib.sha256(self.view.encode()).hexdigest(),
                "class": self.view_class, "verified": self.verified,
                "counts": {c: sum(1 for s in self.steps if s.cls == c) for c in CLASSES},
                "steps": [s.as_dict() for s in self.steps], "notes": list(self.notes),
                "rule": "the matching source is the only build and verification input; the view is never "
                        "compiled for credit and never replaces it"}

    def save(self, directory: str | Path, stem: str | None = None) -> dict:
        """Write ``<stem>.match.c`` (refusing to change an existing, different matching file),
        ``<stem>.view.c`` and ``<stem>.map.json``; returns the three paths."""
        directory = Path(directory)
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem or self.name)
        paths = {"matching": directory / f"{stem}.match.c", "view": directory / f"{stem}.view.c",
                 "map": directory / f"{stem}.map.json"}
        if paths["matching"].exists() and paths["matching"].read_text() != self.matching:
            raise TwoTrackError(f"{paths['matching'].name} exists with other content: the matching source is "
                                "never replaced")
        directory.mkdir(parents=True, exist_ok=True)
        if not paths["matching"].exists():
            paths["matching"].write_text(self.matching)
        paths["view"].write_text(self.view)
        paths["map"].write_text(json.dumps(self.as_map(), indent=1, sort_keys=True) + "\n")
        return {k: str(v) for k, v in paths.items()}


# ------------------------------------------------------------------------------ names
def read_names(text: str) -> dict[str, dict]:
    """``NAMES.tsv`` (``naming.write_tsv``) -> canonical -> {display, from, source}."""
    rows = [line.split("\t") for line in text.splitlines() if line.strip()]
    if not rows:
        return {}
    head = rows[0]
    out = {}
    for row in rows[1:]:
        cells = dict(zip(head, row))
        canonical, shown, origin = cells.get("canonical", ""), cells.get("display", ""), cells.get("display_from", "")
        if not canonical:
            continue
        source = ""
        found = _CELL.match(cells.get(origin, "") or "")
        if found:
            source = found.group("source")
        out[canonical] = {"display": shown, "from": origin, "source": source}
    return out


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def rename(text: str, names: dict[str, dict]) -> tuple[str, list[Step], list[str]]:
    """Replace canonical identifiers (outside comments, strings and directives) with display
    names, consistently in the whole unit.  A semantic name keeps its ``sem_`` prefix and is
    labelled as generated; a name that would collide with an identifier already in the unit is
    skipped."""
    try:
        tokens = astprep.code_tokens(astprep.tokenize(text))
    except astprep.PrepError as error:
        return text, [], [f"names: not renamed ({error})"]
    present = {t.text for t in tokens if t.kind == "ident"}
    plan, notes, labels = {}, [], {}
    for canonical in sorted(names):
        entry = names[canonical]
        new, origin = entry.get("display") or canonical, entry.get("from") or "canonical"
        if origin == "canonical" or new == canonical or canonical not in present:
            continue
        if origin == "semantic" and not new.startswith("sem_"):
            new = "sem_" + new
        if origin not in NAME_FIELDS or not _IDENT.match(new) or new in astprep.KEYWORDS:
            notes.append(f"names: {canonical} not renamed (unusable name {new!r} from {origin})")
            continue
        if new in present or new in plan.values():
            notes.append(f"names: {canonical} not renamed ({new} is already an identifier here)")
            continue
        plan[canonical] = new
        labels[canonical] = _LABEL[origin] + (f" [{entry['source']}]" if entry.get("source") else "")
    if not plan:
        return text, [], notes
    first = {}
    edits = []
    for t in tokens:
        if t.kind == "ident" and t.text in plan:
            edits.append((t.start, t.end, plan[t.text]))
            first.setdefault(t.text, t.start)
    steps = [Step("names", "behaviour-preserving", f"{c} -> {plan[c]}: {labels[c]}", c, plan[c], _line(text, first[c]))
             for c in sorted(plan)]
    return astprep.apply_edits(text, edits), steps, notes


# ------------------------------------------------------------------------------ scaffolding
def remove_scaffolding(text: str) -> tuple[str, list[Step]]:
    steps = []
    for pattern, cls, what in ((_PAD, "behaviour-preserving",
                                "frame padding (reconstruction scaffolding) removed: the frame size changes"),
                               (_ASM_NOP, "behaviour-preserving", "padding nop removed: the code bytes change"),
                               (_LDSYM_LINE, "unverified",
                                "LDSYM marker removed: the build's symbol resolver reads these markers"),
                               (_LDSYM, "unverified",
                                "LDSYM marker removed: the build's symbol resolver reads these markers")):
        while True:
            found = pattern.search(text)
            if not found:
                break
            steps.append(Step("scaffolding", cls, what, found.group().strip(), "", _line(text, found.start())))
            text = text[:found.start()] + text[found.end():]
    return text, steps


# ------------------------------------------------------------------------------ structures
def _blocks(declarations: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """typeflow output -> (struct tag -> block, [(global name, extern line)])."""
    structs, externs, lines, i = {}, [], declarations.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        head = re.match(r"^struct\s+([A-Za-z_]\w*)\s*\{", line)
        if head:
            j = i
            while j < len(lines) and not lines[j].startswith("};"):
                j += 1
            structs[head.group(1)] = "\n".join(lines[i:j + 1])
            i = j + 1
            continue
        found = _EXTERN.match(line)
        if found:
            externs.append((found.group(1), line))
        i += 1
    return structs, externs


def add_structures(text: str, name: str, declarations: str) -> tuple[str, list[Step], list[str]]:
    """Provisional declarations (``typeflow.declarations``) for globals the unit uses but does
    not declare are inserted before the function as code (``unverified``); for globals it already
    declares they are added as comments (``presentation-only``)."""
    try:
        code = astprep.code_tokens(astprep.tokenize(text))
    except astprep.PrepError as error:
        return text, [], [f"structs: not added ({error})"]
    found = astprep.find_function(code, name)
    if found is None:
        return text, [], [f"structs: {name} not found"]
    at = text.rfind("\n", 0, code[found[0]].start) + 1
    used = {t.text for t in code if t.kind == "ident"}
    declared = set(re.findall(r"(?m)^\s*extern\b[^;]*?\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])*\s*;", text))
    tags = set(re.findall(r"\bstruct\s+([A-Za-z_]\w*)\s*\{", text))
    structs, externs = _blocks(declarations)
    code_lines, comment_lines, steps, emitted = [], [], [], set()
    for global_name, line in externs:
        if global_name not in used:
            continue
        if global_name in declared:
            comment_lines.append(f"/* typeflow suggests: {line.replace('*/', '* /')} */")
            steps.append(Step("structs", "presentation-only", f"typeflow declaration of {global_name} shown as a "
                              "comment (the unit already declares it)", "", line))
            continue
        for tag in re.findall(r"\bstruct\s+([A-Za-z_]\w*)", line):
            if tag in structs and tag not in tags and tag not in emitted:
                code_lines.append(structs[tag])
                emitted.add(tag)
        code_lines.append(line)
        steps.append(Step("structs", "unverified", f"provisional declaration of {global_name} inserted (typeflow: "
                          "observed fields only)", "", line))
    block = "\n".join(comment_lines + code_lines)
    if not block:
        return text, [], []
    return text[:at] + block + "\n\n" + text[at:], steps, []


# ------------------------------------------------------------------------------ e-graph statement forms
def egraph_rewriter(text: str, name: str, includes=None) -> list[tuple[int, int, str, tuple]]:
    """Per statement, the smallest e-graph alternative that uses readability rules only
    (``READABLE``; never a ``~wrap`` rule, never a temporary or an if/else), as
    ``(start, end, replacement, rules)`` edits of ``text``.  Needs the host preprocessor."""
    from . import egraph
    ctx = egraph._context(text, name, includes)
    if ctx is None:
        raise TwoTrackError(f"{name}: the e-graph could not parse this unit")
    fn = egraph.Function(ctx.types, ctx.an, egraph.Dialect(), egraph.Limits(), ctx.words, ctx)
    out, spans = [], []
    for unit in egraph.units(ctx):
        if unit.kind == "ifelse":
            continue
        try:
            alts, _size = fn.solve(unit)
        except Exception:                      # an unusual statement stays as it is
            continue
        best = None
        for alt in alts[1:]:
            if alt.decls or not alt.rules or not set(alt.rules) <= READABLE or len(alt.edits) != 1:
                continue
            if best is None or (alt.entry.size, alt.label) < (best.entry.size, best.label):
                best = alt
        if best is None or best.entry.size >= alts[0].entry.size:
            continue
        start, end, replacement = best.edits[0]
        if any(not (end <= s or e <= start) for s, e in spans):
            continue
        spans.append((start, end))
        out.append((start, end, replacement, best.rules))
    return out


def apply_rewrites(text: str, rewrites) -> tuple[str, list[Step]]:
    steps = [Step("egraph", "behaviour-preserving", "statement form: " + "+".join(rules), text[s:e], r, _line(text, s))
             for s, e, r, rules in sorted(rewrites)]
    return astprep.apply_edits(text, [(s, e, r) for s, e, r, _rules in rewrites]), steps


# ------------------------------------------------------------------------------ the view
def header(name: str, steps: list[Step], provenance: dict | None, stem: str) -> str:
    cls = max((s.cls for s in steps), key=RANK.get, default="presentation-only")
    lines = [f"Maintainable view of {name} (n64dk two-track, {SCHEMA}).",
             f"Build and verify {stem}.match.c, never this file: it is not evidence.",
             f"View class: {cls}; {len(steps)} transformation(s), listed in {stem}.map.json."]
    for key in ("case", "status", "recipe", "verdict", "receipt"):
        if provenance and provenance.get(key):
            lines.append(f"{key}: {provenance[key]}")
    for s in steps:
        if s.kind == "names":
            lines.append(f"name {s.detail}")
    body = "\n".join(" * " + line.replace("*/", "* /") for line in lines)
    return "/*\n" + body + "\n */\n"


def build(matching: str, name: str, *, names: dict[str, dict] | None = None, declarations: str | None = None,
          provenance: dict | None = None, rewriter=None, verify=None, scaffolding: bool = True,
          stem: str | None = None) -> TwoTrack:
    """The two tracks of one function's source.  ``rewriter(text, name)`` returns statement
    rewrites (``egraph_rewriter``); ``verify(view_text)`` may build the view strictly and return
    True (byte-identical) or False; neither is required."""
    text, steps, notes = matching, [], []
    if rewriter is not None:
        try:
            text, done = apply_rewrites(text, rewriter(text, name))
            steps += done
        except Exception as error:               # the view is optional: the matching source stands
            notes.append(f"egraph: not applied ({type(error).__name__}: {error})"[:300])
    if scaffolding:
        text, done = remove_scaffolding(text)
        steps += done
    if names:
        text, done, more = rename(text, names)
        steps += done
        notes += more
        name_in_view = next((s.after for s in done if s.before == name), name)
    else:
        name_in_view = name
    if declarations:
        text, done, more = add_structures(text, name_in_view, declarations)
        steps += done
        notes += more
    stem = stem or name
    steps.append(Step("comments", "presentation-only", "provenance header added", "", ""))
    view = header(name, steps, provenance, stem) + text
    track = TwoTrack(name, matching, view, steps, notes)
    if verify is not None:
        result = verify(view)
        track.verified = "exact" if result is True or result == "exact" else ("different" if result in (False, "different")
                                                                               else None)
    return track
