"""splat-style YAML segment configurations <-> the kit's image registry and catalogues.

**YAML.**  PyYAML is not a dependency of the kit (and is not installed in the lab environment), so
``load_yaml`` reads the subset of YAML that splat configurations are written in: block mappings and
sequences (including a sequence at its key's indentation and mappings inside sequence items), flow
sequences and mappings (also spanning lines), plain / single-quoted / double-quoted scalars, ``|``
and ``>`` block scalars, comments and one document.  Scalars resolve as PyYAML (YAML 1.1) resolves
them, the reader splat itself uses: ``0x..`` and decimal integers, ``True``/``yes``/``on`` booleans,
``null``/``~``.  Anchors, aliases, merge keys, tags and multiple documents are refused with the line
number (``YamlSubsetError``): never guessed.  A configuration converted to JSON is read as JSON.

**Import** (``import_config``).  Every top-level segment that has an explicit ``vram``, ROM bytes and
at least one text subsegment (``c``, ``asm``, ``hasm``, ``cpp``, ``textbin``) with a known start
becomes one kit image, identified by ``(vram, rom start)``: overlays that share a VRAM range stay
distinct images.  The image file is the ROM the configuration splits (splat's ``target_path``), so
``vram_delta = vram - rom start`` and one ``copy`` raw segment maps it.  Symbols from
``symbol_addrs`` files are assigned to the image whose range holds them; where several overlays hold
an address, the symbol's ``rom:`` (or ``segment:``) attribute decides, and without one the symbol is
reported ambiguous and left out (fail closed).  A function is a ``type:func`` symbol, or (optionally)
an untyped symbol inside a text subsegment; a function's size is its ``size:`` attribute, else the
distance to the next function or the end of its text subsegment (marked inferred).
Everything imported carries ``PROVENANCE`` ("imported: not proof"); nothing sets a match status.

**Export** (``export_yaml``, ``export_symbol_addrs``).  Kit images (name, ROM start, VRAM, size,
catalogue rows) as splat segments with one ``asm`` subsegment each, and the catalogue as
``symbol_addrs.txt`` with ``rom:`` on every function of an image whose VRAM range overlaps another
image.  Names, addresses and sizes only: never ROM bytes.
"""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import EXPORT_PROVENANCE, PROVENANCE
from .symaddrs import SymbolEntry, SymbolFile, entry as new_entry


# ============================================================================ YAML subset reader

class YamlSubsetError(ValueError):
    pass


_INT_DEC = re.compile(r"^[-+]?(0|[1-9][0-9_]*)$")
_INT_HEX = re.compile(r"^[-+]?0x[0-9a-fA-F_]+$")
_INT_OCT = re.compile(r"^[-+]?0o?[0-7_]+$")
_INT_BIN = re.compile(r"^[-+]?0b[01_]+$")
_FLOAT = re.compile(r"^[-+]?([0-9][0-9_]*)?\.[0-9_]*([eE][-+][0-9]+)?$")
_TRUE = {"true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON", "y", "Y"}
_FALSE = {"false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF", "n", "N"}
_NULL = {"", "~", "null", "Null", "NULL"}


def _resolve(text: str):
    """A plain scalar as PyYAML's (YAML 1.1) resolver reads it."""
    if text in _NULL:
        return None
    if text in _TRUE and text not in ("y", "Y", "n", "N"):     # PyYAML does not resolve y/n
        return True
    if text in _FALSE and text not in ("y", "Y", "n", "N"):
        return False
    try:
        if _INT_HEX.match(text):
            return int(text.replace("_", ""), 16)
        if _INT_BIN.match(text):
            return int(text.replace("_", ""), 0)
        if _INT_DEC.match(text):
            return int(text.replace("_", ""), 10)
        if _INT_OCT.match(text) and len(text.lstrip("+-")) > 1:
            digits = text.replace("_", "").replace("0o", "0")
            sign = -1 if digits.startswith("-") else 1
            return sign * int(digits.lstrip("+-"), 8)
        if _FLOAT.match(text) and text not in (".", "+.", "-."):
            return float(text.replace("_", ""))
    except ValueError:
        pass
    if text in (".inf", ".Inf", ".INF", "+.inf"):
        return float("inf")
    if text in ("-.inf", "-.Inf", "-.INF"):
        return float("-inf")
    if text in (".nan", ".NaN", ".NAN"):
        return float("nan")
    return text


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "/": "/", " ": " ",
            "a": "\a", "b": "\b", "e": "\x1b", "f": "\f", "v": "\v"}


def _double_quoted(body: str, lineno: int) -> str:
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if nxt in _ESCAPES:
            out.append(_ESCAPES[nxt])
            i += 2
        elif nxt in "xuU":
            width = {"x": 2, "u": 4, "U": 8}[nxt]
            digits = body[i + 2:i + 2 + width]
            try:
                out.append(chr(int(digits, 16)))
            except ValueError:
                raise YamlSubsetError(f"line {lineno}: bad escape \\{nxt}{digits}") from None
            i += 2 + width
        else:
            raise YamlSubsetError(f"line {lineno}: unsupported escape \\{nxt}")
    return "".join(out)


def _strip_comment(text: str) -> str:
    """``text`` without a trailing ``# comment`` (a ``#`` at the start or after whitespace,
    outside quotes)."""
    quote = None
    i = 0
    while i < len(text):
        c = text[i]
        if quote:
            if quote == "'" and c == "'" and i + 1 < len(text) and text[i + 1] == "'":
                i += 2
                continue
            if quote == '"' and c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "'\"" and (i == 0 or text[i - 1] in " \t[{,:-"):
            quote = c
        elif c == "#" and (i == 0 or text[i - 1] in " \t"):
            return text[:i].rstrip()
        i += 1
    return text.rstrip()


class _FlowParser:
    def __init__(self, text: str, lineno: int):
        self.s, self.i, self.lineno = text, 0, lineno

    def error(self, what: str):
        raise YamlSubsetError(f"line {self.lineno}: {what} in flow collection {self.s!r}")

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def value(self):
        self.ws()
        if self.i >= len(self.s):
            self.error("unexpected end")
        c = self.s[self.i]
        if c == "[":
            return self.sequence()
        if c == "{":
            return self.mapping()
        if c in "&*!":
            raise YamlSubsetError(f"line {self.lineno}: anchors, aliases and tags are not supported")
        return self.scalar()

    def scalar(self, key: bool = False):
        c = self.s[self.i]
        if c in "'\"":
            return self.quoted()
        start = self.i
        while self.i < len(self.s):
            c = self.s[self.i]
            if c in ",]}":
                break
            if c == ":" and (self.i + 1 >= len(self.s) or self.s[self.i + 1] in " ,]}"):
                break
            self.i += 1
        return _resolve(self.s[start:self.i].strip())

    def quoted(self):
        q = self.s[self.i]
        self.i += 1
        out = []
        while self.i < len(self.s):
            c = self.s[self.i]
            if q == "'" and c == "'":
                if self.i + 1 < len(self.s) and self.s[self.i + 1] == "'":
                    out.append("'")
                    self.i += 2
                    continue
                self.i += 1
                return "".join(out)
            if q == '"' and c == "\\":
                out.append(self.s[self.i:self.i + 2])
                self.i += 2
                continue
            if q == '"' and c == '"':
                self.i += 1
                return _double_quoted("".join(out), self.lineno)
            out.append(c)
            self.i += 1
        self.error("unterminated quoted scalar")

    def sequence(self):
        self.i += 1
        out = []
        while True:
            self.ws()
            if self.i >= len(self.s):
                self.error("unterminated '['")
            if self.s[self.i] == "]":
                self.i += 1
                return out
            item = self.value()
            self.ws()
            if self.i < len(self.s) and self.s[self.i] == ":":         # single-pair mapping in a sequence
                self.i += 1
                item = {item: self.value()}
                self.ws()
            out.append(item)
            if self.i < len(self.s) and self.s[self.i] == ",":
                self.i += 1
            elif self.i < len(self.s) and self.s[self.i] == "]":
                continue
            else:
                self.error("expected ',' or ']'")

    def mapping(self):
        self.i += 1
        out = {}
        while True:
            self.ws()
            if self.i >= len(self.s):
                self.error("unterminated '{'")
            if self.s[self.i] == "}":
                self.i += 1
                return out
            key = self.value()
            self.ws()
            value = None
            if self.i < len(self.s) and self.s[self.i] == ":":
                self.i += 1
                self.ws()
                value = None if self.i < len(self.s) and self.s[self.i] in ",}" else self.value()
            if key in out:
                self.error(f"duplicate key {key!r}")
            out[key] = value
            self.ws()
            if self.i < len(self.s) and self.s[self.i] == ",":
                self.i += 1
            elif self.i < len(self.s) and self.s[self.i] == "}":
                continue
            else:
                self.error("expected ',' or '}'")


def _balance(text: str) -> int:
    depth, quote, i = 0, None, 0
    while i < len(text):
        c = text[i]
        if quote:
            if quote == '"' and c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        i += 1
    return depth


_KEY = re.compile(r"""^(?:"((?:[^"\\]|\\.)*)"|'((?:[^']|'')*)'|([^\s#'"\[\]{},&*!|>%@`][^#]*?))\s*:(?:\s+(.*)|$)""")


class _BlockParser:
    def __init__(self, text: str):
        if "\t" in "".join(line[:len(line) - len(line.lstrip())] for line in text.splitlines()):
            raise YamlSubsetError("tab characters in indentation are not YAML")
        self.raw = text.splitlines()
        self.i = 0
        self._skip_document_start()

    def _skip_document_start(self):
        while self.i < len(self.raw):
            stripped = _strip_comment(self.raw[self.i]).strip()
            if not stripped:
                self.i += 1
            elif stripped.startswith("%"):
                self.i += 1
            elif stripped == "---":
                self.i += 1
                break
            else:
                break

    def peek(self):
        """(indent, content, lineno) of the next meaningful line, or None."""
        while self.i < len(self.raw):
            line = self.raw[self.i]
            content = _strip_comment(line)
            stripped = content.strip()
            if not stripped:
                self.i += 1
                continue
            if stripped == "---" and len(content) - len(content.lstrip()) == 0:
                raise YamlSubsetError(f"line {self.i + 1}: multiple documents are not supported")
            if stripped == "..." and len(content) - len(content.lstrip()) == 0:
                self.i = len(self.raw)
                return None
            return len(content) - len(content.lstrip()), stripped, self.i + 1
        return None

    def document(self):
        head = self.peek()
        if head is None:
            return None
        value = self.block(head[0])
        if self.peek() is not None:
            _indent, content, lineno = self.peek()
            raise YamlSubsetError(f"line {lineno}: unexpected content {content!r}")
        return value

    def block(self, indent: int):
        head = self.peek()
        if head is None:
            return None
        if head[1] == "-" or head[1].startswith("- "):
            return self.sequence(head[0])
        if _KEY.match(head[1]):
            return self.mapping(head[0])
        # a lone scalar (or flow collection) as the whole block
        self.i += 1
        return self.inline_value(head[1], head[2], head[0])

    def flow(self, text: str, lineno: int):
        while _balance(text) > 0:
            if self.i >= len(self.raw):
                raise YamlSubsetError(f"line {lineno}: unterminated flow collection")
            text = text + " " + _strip_comment(self.raw[self.i]).strip()
            self.i += 1
        parser = _FlowParser(text, lineno)
        value = parser.value()
        parser.ws()
        if parser.i != len(parser.s):
            raise YamlSubsetError(f"line {lineno}: trailing text after flow collection: {parser.s[parser.i:]!r}")
        return value

    def inline_value(self, text: str, lineno: int, indent: int):
        """The value written after ``key:`` or ``-`` on the same line (the line already consumed)."""
        if text[0] in "&*!":
            raise YamlSubsetError(f"line {lineno}: anchors, aliases and tags are not supported")
        if text[0] in "[{":
            return self.flow(text, lineno)
        if text[0] in "|>":
            return self.block_scalar(text, lineno, indent)
        if text[0] == '"':
            parser = _FlowParser(text, lineno)
            value = parser.quoted()
            if text[parser.i:].strip():
                raise YamlSubsetError(f"line {lineno}: text after a quoted scalar")
            return value
        if text[0] == "'":
            parser = _FlowParser(text, lineno)
            value = parser.quoted()
            if text[parser.i:].strip():
                raise YamlSubsetError(f"line {lineno}: text after a quoted scalar")
            return value
        # a plain scalar may continue on more-indented lines (folded with spaces)
        parts = [text]
        while True:
            nxt = self.peek()
            if nxt is None or nxt[0] <= indent or nxt[1].startswith("- ") or _KEY.match(nxt[1]):
                break
            parts.append(nxt[1])
            self.i += 1
        return _resolve(" ".join(parts))

    def block_scalar(self, header: str, lineno: int, indent: int) -> str:
        m = re.fullmatch(r"([|>])([-+]?)([1-9]?)([-+]?)\s*", header)
        if not m:
            raise YamlSubsetError(f"line {lineno}: unsupported block scalar header {header!r}")
        style, chomp = m.group(1), (m.group(2) or m.group(4))
        explicit = int(m.group(3)) if m.group(3) else None
        lines, block_indent = [], None
        while self.i < len(self.raw):
            line = self.raw[self.i]
            if not line.strip():
                lines.append("")
                self.i += 1
                continue
            here = len(line) - len(line.lstrip(" "))
            if block_indent is None:
                block_indent = indent + explicit if explicit else here
            if here < block_indent or here <= indent:
                break
            lines.append(line[block_indent:])
            self.i += 1
        while lines and lines[-1] == "" and chomp != "+":
            lines.pop()
        if style == "|":
            body = "\n".join(lines)
        else:
            folded, paragraph = [], []
            for line in lines:
                if line == "":
                    folded.append(" ".join(paragraph))
                    paragraph = []
                else:
                    paragraph.append(line)
            folded.append(" ".join(paragraph))
            body = "\n".join(folded)
        if chomp == "-":
            return body
        return body + "\n" if body or chomp == "+" else body

    def mapping(self, indent: int, first: tuple[str, int] | None = None) -> dict:
        out: dict = {}
        while True:
            if first is not None:
                content, lineno = first
                first = None
            else:
                head = self.peek()
                if head is None or head[0] != indent or head[1] == "-" or head[1].startswith("- "):
                    break
                _ind, content, lineno = head
                self.i += 1
            m = _KEY.match(content)
            if not m:
                raise YamlSubsetError(f"line {lineno}: expected 'key: value', found {content!r}")
            if m.group(1) is not None:
                key = _double_quoted(m.group(1), lineno)
            elif m.group(2) is not None:
                key = m.group(2).replace("''", "'")
            else:
                key = _resolve(m.group(3).strip())
            if key == "<<":
                raise YamlSubsetError(f"line {lineno}: merge keys are not supported")
            if isinstance(key, str) and key.startswith(("&", "*", "!")):
                raise YamlSubsetError(f"line {lineno}: anchors, aliases and tags are not supported")
            rest = (m.group(4) or "").strip()
            if rest:
                value = self.inline_value(rest, lineno, indent)
            else:
                nxt = self.peek()
                if nxt is not None and (nxt[0] > indent or
                                        (nxt[0] == indent and (nxt[1] == "-" or nxt[1].startswith("- ")))):
                    value = self.block(nxt[0])
                else:
                    value = None
            if key in out:
                raise YamlSubsetError(f"line {lineno}: duplicate key {key!r}")
            out[key] = value
        return out

    def sequence(self, indent: int) -> list:
        out = []
        while True:
            head = self.peek()
            if head is None or head[0] != indent or not (head[1] == "-" or head[1].startswith("- ")):
                break
            _ind, content, lineno = head
            self.i += 1
            rest = content[1:].lstrip()
            column = indent + (len(content) - len(rest))
            if not rest:
                nxt = self.peek()
                out.append(self.block(nxt[0]) if nxt is not None and nxt[0] > indent else None)
            elif rest == "-" or rest.startswith("- "):
                raise YamlSubsetError(f"line {lineno}: nested inline sequences ('- - x') are not supported")
            elif rest[0] in "[{'\"|>&*!":
                out.append(self.inline_value(rest, lineno, indent))
            elif _KEY.match(rest):
                out.append(self.mapping(column, (rest, lineno)))
            else:
                out.append(self.inline_value(rest, lineno, indent))
        return out


def load_yaml(text: str):
    """Parse the YAML subset described in the module docstring."""
    return _BlockParser(text).document()


def read_config(source) -> dict:
    """A splat configuration from a path or text: JSON when it is JSON, else the YAML subset."""
    def is_file(name: str) -> bool:
        try:
            return Path(name).is_file()
        except (OSError, ValueError):
            return False

    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and
                                    not source.lstrip().startswith("{") and is_file(source)):
        text = Path(source).read_text()
    else:
        text = str(source)
    stripped = text.lstrip()
    if stripped.startswith("{"):
        data = json.loads(text)
    else:
        data = load_yaml(text)
    if not isinstance(data, dict):
        raise YamlSubsetError("a splat configuration is a mapping at the top level")
    return data


# ============================================================================ YAML subset writer

_PLAIN_OK = re.compile(r"^[A-Za-z0-9_./][A-Za-z0-9_./+\-]*$")
HEX_KEYS = {"start", "vram", "bss_size", "rom", "size", "align", "subalign", "ld_align_segment_start",
            "vram_end", "end"}


def _scalar_text(value, hexadecimal: bool = False) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if hexadecimal:
            return f"-0x{-value:X}" if value < 0 else f"0x{value:X}"
        return str(value)
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    if _PLAIN_OK.match(text) and isinstance(_resolve(text), str):
        return text
    return json.dumps(text, ensure_ascii=False)            # a JSON string is a valid YAML double-quoted scalar


def _flow(value, hex_first: bool = False) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_flow(v, hex_first and i == 0) for i, v in enumerate(value)) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_scalar_text(k)}: {_flow(v, k in HEX_KEYS)}" for k, v in value.items()) + " }"
    return _scalar_text(value, hex_first)


def _simple(value) -> bool:
    return not isinstance(value, (dict, list)) or (
        isinstance(value, list) and all(not isinstance(v, (dict, list)) for v in value))


def dump_yaml(value, indent: int = 0, key: str | None = None) -> str:
    """Block YAML that ``load_yaml`` (and PyYAML) read back to ``value``.  Integers under the
    address-like keys (``HEX_KEYS``) and the first element of flow lists are written in hex."""
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            name = _scalar_text(k)
            if isinstance(v, str) and "\n" in v and not v.endswith("\n\n") and v.strip("\n") == v.rstrip("\n"):
                body = v.rstrip("\n").split("\n")
                header = "|" + ("2" if body[0].startswith(" ") else "") + ("" if v.endswith("\n") else "-")
                lines.append(f"{pad}{name}: {header}")
                lines.extend(f"{pad}  {line}" if line else "" for line in body)
            elif isinstance(v, dict) and v:
                lines.append(f"{pad}{name}:")
                lines.append(dump_yaml(v, indent + 2))
            elif isinstance(v, list) and v and not _simple(v):
                lines.append(f"{pad}{name}:")
                lines.append(dump_yaml(v, indent + 2))
            elif isinstance(v, (list, dict)):
                lines.append(f"{pad}{name}: {_flow(v, True)}")
            else:
                lines.append(f"{pad}{name}: {_scalar_text(v, k in HEX_KEYS)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item:
                body = dump_yaml(item, indent + 2).split("\n")
                lines.append(f"{pad}- {body[0].lstrip()}")
                lines.extend(body[1:])
            elif isinstance(item, list):
                lines.append(f"{pad}- {_flow(item, True)}")
            else:
                lines.append(f"{pad}- {_scalar_text(item)}")
    else:
        lines.append(pad + _scalar_text(value))
    return "\n".join(lines)


# ============================================================================ the segment model

TEXT_TYPES = {"c", "asm", "hasm", "cpp", "textbin"}


@dataclass
class Subsegment:
    start: int | None            # None: "auto" or not given
    type: str | None
    name: str | None
    vram: int | None = None
    args: list = field(default_factory=list)      # list-form positional arguments after the name
    extra: dict = field(default_factory=dict)     # dict-form keys the kit does not read
    children: list["Subsegment"] = field(default_factory=list)


@dataclass
class Segment:
    name: str
    type: str | None
    rom_start: int | None
    rom_end: int | None = None
    vram: int | None = None
    subsegments: list[Subsegment] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    index: int = 0

    @property
    def identity(self) -> tuple[int | None, int | None]:
        return self.vram, self.rom_start

    @property
    def rom_size(self) -> int:
        if self.rom_start is None or self.rom_end is None:
            return 0
        return max(0, self.rom_end - self.rom_start)

    def text_ranges(self) -> list[tuple[int, int, str]]:
        """VRAM ranges of text subsegments with a known start: (start, end, unit name)."""
        if self.vram is None or self.rom_start is None:
            return []
        flat = []

        def walk(items):
            for sub in items:
                flat.append(sub)
                walk(sub.children)
        walk(self.subsegments)
        starts = sorted({s.start for s in flat if s.start is not None} | ({self.rom_end} if self.rom_end else set()))
        out = []
        for sub in flat:
            if sub.start is None or (sub.type or "").lower() not in TEXT_TYPES:
                continue
            later = [s for s in starts if s > sub.start]
            end = later[0] if later else self.rom_end
            if end is None:
                continue
            out.append((self.vram + sub.start - self.rom_start, self.vram + end - self.rom_start, sub.name or ""))
        return sorted(out)


def _subsegment(item, segment_name: str) -> Subsegment:
    if isinstance(item, list):
        start = item[0] if item else None
        return Subsegment(start if isinstance(start, int) else None,
                          str(item[1]) if len(item) > 1 and item[1] is not None else None,
                          str(item[2]) if len(item) > 2 and item[2] is not None else None,
                          None, list(item[3:]))
    if isinstance(item, dict):
        known = {"start", "type", "name", "vram", "subsegments"}
        start = item.get("start")
        return Subsegment(start if isinstance(start, int) else None,
                          str(item["type"]) if item.get("type") is not None else None,
                          str(item["name"]) if item.get("name") is not None else None,
                          item.get("vram") if isinstance(item.get("vram"), int) else None, [],
                          {k: v for k, v in item.items() if k not in known},
                          [_subsegment(x, segment_name) for x in item.get("subsegments") or []])
    raise YamlSubsetError(f"segment {segment_name}: unreadable subsegment {item!r}")


def segments(config: dict) -> tuple[list[Segment], int | None]:
    """The configuration's top-level segments in order, each ``rom_end`` filled from the next
    segment's start, and the ROM end (a final one-element ``[end]`` entry) if given."""
    out: list[Segment] = []
    rom_end = None
    for index, item in enumerate(config.get("segments") or []):
        if isinstance(item, list):
            if len(item) == 1 and isinstance(item[0], int):
                rom_end = item[0]
                continue
            start = item[0] if item and isinstance(item[0], int) else None
            kind = str(item[1]) if len(item) > 1 else None
            name = str(item[2]) if len(item) > 2 and item[2] is not None else (
                f"{kind}_{start:X}" if start is not None else f"segment_{index}")
            out.append(Segment(name, kind, start, None, None, [], {"args": list(item[3:])} if item[3:] else {},
                               index))
        elif isinstance(item, dict):
            known = {"name", "type", "start", "vram", "subsegments"}
            start = item.get("start") if isinstance(item.get("start"), int) else None
            name = str(item.get("name") or (f"{item.get('type')}_{start:X}" if start is not None else f"segment_{index}"))
            out.append(Segment(name, str(item["type"]) if item.get("type") is not None else None, start, None,
                               item.get("vram") if isinstance(item.get("vram"), int) else None,
                               [_subsegment(x, name) for x in item.get("subsegments") or []],
                               {k: v for k, v in item.items() if k not in known}, index))
        else:
            raise YamlSubsetError(f"unreadable segment entry {item!r}")
    starts = [s.rom_start for s in out]
    for k, segment in enumerate(out):
        if segment.rom_start is None:
            continue
        later = [x for x in starts[k + 1:] if x is not None]
        segment.rom_end = later[0] if later else rom_end
    return out, rom_end


# ============================================================================ import into the kit model

@dataclass
class ImportedFunction:
    name: str
    vram: int
    size: int
    size_source: str            # "attribute" | "inferred: next function" | "inferred: end of text unit"
    symbol_type: str            # "func" | "untyped symbol in a text unit"
    line: int | None = None
    provenance: str = PROVENANCE


@dataclass
class ImportedImage:
    name: str
    segment: Segment
    functions: list[ImportedFunction] = field(default_factory=list)
    provenance: str = PROVENANCE

    @property
    def rom_start(self) -> int:
        return self.segment.rom_start

    @property
    def vram(self) -> int:
        return self.segment.vram

    @property
    def rom_size(self) -> int:
        return self.segment.rom_size

    @property
    def vram_delta(self) -> int:
        return self.vram - self.rom_start

    def contains(self, address: int) -> bool:
        return self.vram <= address < self.vram + self.rom_size


@dataclass
class KitImport:
    images: dict[str, ImportedImage]
    skipped_segments: list[dict]
    unassigned: list[dict]
    link_order: dict[str, list[dict]]
    rom_end: int | None
    options: dict
    image_path: str
    provenance: str = PROVENANCE
    warnings: list[str] = field(default_factory=list)

    def overlapping(self) -> list[tuple[str, str]]:
        names = sorted(self.images)
        out = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                x, y = self.images[a], self.images[b]
                if x.vram < y.vram + y.rom_size and y.vram < x.vram + x.rom_size:
                    out.append((a, b))
        return out

    def registry_toml(self, catalogue_dir: str = "imported") -> str:
        lines = [f"# Imported from a splat configuration by n64dk.interop.splat ({self.provenance}).",
                 "# Segment boundaries, names and sizes are the configuration's claims; nothing was checked",
                 "# against bytes.  Each image is the ROM the configuration splits, mapped by one raw segment.", ""]
        for name in sorted(self.images, key=lambda n: (self.images[n].rom_start, self.images[n].vram)):
            image = self.images[name]
            lines += [f"[images.{name}]",
                      f"path = {json.dumps(self.image_path)}",
                      f"vram_delta = 0x{image.vram_delta & 0xFFFFFFFF:08X}",
                      f'catalogue = "{catalogue_dir}/functions_{name}.toml"',
                      "manifests = []",
                      f'raw_segments = [[0x{image.rom_start:X}, 0x{image.rom_start:X}, 0x{image.rom_size:X}, "copy"]]',
                      f"extent = [0x{image.vram:08X}, 0x{image.vram + image.rom_size:08X}]",
                      f"identity = [0x{image.vram:08X}, 0x{image.rom_start:X}]   # (vram, rom): overlays stay distinct",
                      f"provenance = {json.dumps(image.provenance)}", ""]
        return "\n".join(lines)

    def catalogue_toml(self, name: str) -> str:
        image = self.images[name]
        lines = [f"# Function boundaries for {name}: imported from a splat configuration ({image.provenance})",
                 f"# {len(image.functions)} functions; sizes marked 'inferred' were not in the symbol files",
                 "[[section]]", f"name = {json.dumps(name)}", f"rom = 0x{image.rom_start:08X}",
                 f"vram = 0x{image.vram:08X}", f"size = 0x{image.rom_size:X}",
                 f"provenance = {json.dumps(image.provenance)}", "", "functions = ["]
        for f in image.functions:
            lines.append(f'    {{ name = "{f.name}", vram = 0x{f.vram:08X}, size = 0x{f.size:X} }},'
                         f" # {f.provenance}; size {f.size_source}; {f.symbol_type}")
        lines.append("]")
        return "\n".join(lines) + "\n"

    def as_dict(self) -> dict:
        return {"provenance": self.provenance, "image_path": self.image_path,
                "rom_end": None if self.rom_end is None else f"0x{self.rom_end:X}",
                "images": {name: {"identity": [f"0x{i.vram:08X}", f"0x{i.rom_start:X}"],
                                  "rom_size": f"0x{i.rom_size:X}", "functions": len(i.functions),
                                  "inferred_sizes": sum(f.size_source != "attribute" for f in i.functions),
                                  "provenance": i.provenance}
                           for name, i in sorted(self.images.items())},
                "overlapping_images": [list(pair) for pair in self.overlapping()],
                "skipped_segments": self.skipped_segments, "unassigned_symbols": len(self.unassigned),
                "unassigned_by_reason": _count(self.unassigned, "reason"),
                "link_order_units": {k: len(v) for k, v in self.link_order.items()},
                "warnings": self.warnings}


def _count(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for row in rows:
        reason = str(row.get(key, "")).split(":")[0]
        out[reason] = out.get(reason, 0) + 1
    return out


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")


def import_config(config: dict, symbol_files: list[SymbolFile] = (), *, image_path: str | None = None,
                  include_untyped: bool = True) -> KitImport:
    """The configuration as kit images and catalogues (see the module docstring)."""
    options = dict(config.get("options") or {})
    origin = config.get("n64dk_provenance")
    provenance = PROVENANCE + (f"; origin: {origin.get('note')}" if isinstance(origin, dict) and origin.get("note")
                               else "")
    target = image_path or str(options.get("target_path") or "baserom.z64")
    segs, rom_end = segments(config)
    images: dict[str, ImportedImage] = {}
    skipped, warnings = [], []
    seen_identity: dict[tuple, str] = {}
    link_order: dict[str, list[dict]] = {}
    for segment in segs:
        order = [{"rom": None if s.start is None else f"0x{s.start:X}", "type": s.type, "name": s.name}
                 for s in segment.subsegments]
        if order:
            link_order[segment.name] = order
        why = None
        if segment.vram is None:
            why = "no explicit vram" + (f" (follows_vram: {segment.extra['follows_vram']}; not computed)"
                                        if "follows_vram" in segment.extra else "")
        elif segment.rom_start is None:
            why = "no ROM start"
        elif segment.rom_size == 0:
            why = "no ROM bytes (bss only, or the ROM end is unknown)"
        elif not segment.text_ranges():
            why = "no text subsegment with a known start"
        if why:
            skipped.append({"name": segment.name, "type": segment.type, "reason": why})
            continue
        if segment.identity in seen_identity:
            warnings.append(f"segment {segment.name} repeats (vram, rom) of {seen_identity[segment.identity]}: skipped")
            skipped.append({"name": segment.name, "type": segment.type, "reason": "duplicate (vram, rom)"})
            continue
        name = _SAFE_NAME.sub("_", segment.name) or f"segment_{segment.index}"
        if name in images:
            name = f"{name}_{segment.rom_start:X}"
        seen_identity[segment.identity] = name
        images[name] = ImportedImage(name, segment, provenance=provenance)

    unassigned: list[dict] = []
    placed: dict[str, dict[int, ImportedFunction]] = {name: {} for name in images}
    for symbols in symbol_files:
        for e in symbols.entries:
            if e.ignored:
                continue
            kind = (e.type or "").lower()
            if kind and kind != "func":
                continue
            candidates = [i for i in images.values() if i.contains(e.address)]
            if e.rom is not None:
                candidates = [i for i in candidates if i.rom_start <= e.rom < i.rom_start + i.rom_size]
            if e.segment is not None:
                candidates = [i for i in candidates if e.segment in (i.name, i.segment.name)]
            if not candidates:
                if kind == "func":
                    unassigned.append({"name": e.name, "vram": f"0x{e.address:08X}", "line": e.line,
                                       "reason": "outside every imported code segment"})
                continue
            in_text = {i.name: any(lo <= e.address < hi for lo, hi, _n in i.segment.text_ranges())
                       for i in candidates}
            if not kind and not (include_untyped and any(in_text.values())):
                continue                                   # an untyped data symbol: not a catalogue row
            if len(candidates) > 1:
                unassigned.append({"name": e.name, "vram": f"0x{e.address:08X}", "line": e.line,
                                   "reason": "ambiguous: " + ", ".join(sorted(i.name for i in candidates)) +
                                             " map this address and the symbol has no rom:/segment: attribute"})
                continue
            image = candidates[0]
            if not kind and not in_text[image.name]:
                continue
            if e.address in placed[image.name]:
                warnings.append(f"{e.name}: a second function at 0x{e.address:08X} in {image.name}: kept the first")
                continue
            placed[image.name][e.address] = ImportedFunction(
                e.name, e.address, e.size or 0, "attribute" if e.size else "",
                "func" if kind == "func" else "untyped symbol in a text unit", e.line, provenance)
    for name, image in images.items():
        ranges = image.segment.text_ranges()
        rows = sorted(placed[name].values(), key=lambda f: f.vram)
        for k, f in enumerate(rows):
            if f.size_source:
                continue
            unit_end = next((hi for lo, hi, _n in ranges if lo <= f.vram < hi), image.vram + image.rom_size)
            nxt = rows[k + 1].vram if k + 1 < len(rows) else None
            if nxt is not None and nxt <= unit_end:
                f.size, f.size_source = nxt - f.vram, "inferred: next function"
            else:
                f.size, f.size_source = unit_end - f.vram, "inferred: end of text unit"
        image.functions = [f for f in rows if f.size > 0]
    return KitImport(images, skipped, unassigned, link_order, rom_end, options, target, provenance, warnings)


def write_import(kit: KitImport, out_dir, catalogue_dir: str = "imported") -> list[Path]:
    """Write ``n64dk.imported.toml``, one catalogue per image and ``import_report.json`` into a NEW
    (or empty) directory; refuses to overwrite anything."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; imports are written to a new directory")
    (out / catalogue_dir).mkdir(parents=True, exist_ok=True)
    written = [out / "n64dk.imported.toml"]
    written[0].write_text(kit.registry_toml(catalogue_dir))
    for name in sorted(kit.images):
        path = out / catalogue_dir / f"functions_{name}.toml"
        path.write_text(kit.catalogue_toml(name))
        written.append(path)
    report = out / "import_report.json"
    report.write_text(json.dumps(kit.as_dict(), indent=1) + "\n")
    written.append(report)
    return written


# ============================================================================ export from the kit model

@dataclass
class KitImage:
    name: str
    rom_start: int
    vram: int
    size: int
    functions: list[tuple[str, int, int]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)       # carried into the segment (unknown keys preserved)


def _section_of(catalogue: Path | None) -> dict:
    if catalogue is None or not Path(catalogue).is_file():
        return {}
    try:
        data = tomllib.loads(Path(catalogue).read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    sections = data.get("section") if isinstance(data, dict) else None
    return sections[0] if isinstance(sections, list) and sections and isinstance(sections[0], dict) else {}


def kit_images(project, rom_starts: dict[str, int] | None = None) -> tuple[list[KitImage], list[dict]]:
    """Kit images of a project (``n64dk.toml`` ``[images.*]`` and their catalogues) with a known ROM
    start: ``rom_starts`` (explicit), else the first raw segment, else the catalogue section's
    ``rom`` when the image file is a ROM.  Image bytes are never read."""
    from ..match import images as imagemod
    rom_starts = rom_starts or {}
    out, skipped = [], []
    for name, spec in sorted(imagemod.load_registry(project).items()):
        section = _section_of(spec.catalogue)
        rows = spec.rows() if spec.catalogue and Path(spec.catalogue).is_file() else []
        # link space throughout (the catalogue's own addresses are shifted by catalogue_shift)
        vram = section["vram"] - spec.catalogue_shift if isinstance(section.get("vram"), int) else None
        if spec.raw_segments and spec.vram_delta is not None:
            rom, image_off, size, _how = spec.raw_segments[0]
            vram = image_off + spec.vram_delta
        elif vram is None and spec.vram_delta is not None:
            vram = spec.vram_delta
        rom = rom_starts.get(name)
        if rom is None and spec.raw_segments:
            rom = spec.raw_segments[0][0]
        if rom is None and Path(spec.path).suffix.lower() in (".z64", ".n64", ".v64") and \
                isinstance(section.get("rom"), int):
            rom = section["rom"]
        if rom is None or vram is None:
            skipped.append({"image": name, "reason": "no ROM start known (declare raw_segments or pass rom_starts)"
                            if rom is None else "no VRAM known"})
            continue
        size = section.get("size") if isinstance(section.get("size"), int) else None
        if spec.extent:
            size = spec.extent[1] - vram
        if size is None:
            size = max((v + s for _n, v, s in rows), default=vram) - vram
        out.append(KitImage(name, rom, vram, size, list(rows)))
    return out, skipped


def _overlaps(images: list[KitImage]) -> set[str]:
    out = set()
    for i, a in enumerate(images):
        for b in images[i + 1:]:
            if a.vram < b.vram + b.size and b.vram < a.vram + a.size:
                out |= {a.name, b.name}
    return out


def export_config(images: list[KitImage], *, title: str | None = None, options: dict | None = None,
                  rom_end: int | None = None, sha1: str | None = None) -> dict:
    """A splat configuration (as a dict) for the kit images: one code segment each, one ``asm``
    subsegment, ordered by ROM start.  ``options`` are carried as given."""
    config: dict = {}
    if title:
        config["name"] = title
    if sha1:
        config["sha1"] = sha1
    config["options"] = dict(options or {"platform": "n64"})
    config["n64dk_provenance"] = {"note": EXPORT_PROVENANCE,
                                  "overlapping_images": sorted(_overlaps(images))}
    segs: list = []
    ordered = sorted(images, key=lambda i: (i.rom_start, i.vram, i.name))
    ends = [i.rom_start + i.size for i in images]
    end = rom_end if rom_end is not None else (max(ends) if ends else None)
    for k, image in enumerate(ordered):
        seg = {"name": image.name, "type": "code", "start": image.rom_start, "vram": image.vram}
        seg.update({key: v for key, v in image.extra.items() if key not in seg and key != "subsegments"})
        seg["subsegments"] = image.extra.get("subsegments") or [[image.rom_start, "asm", image.name]]
        segs.append(seg)
        # a splat segment runs to the next one's start: bytes no image claims get a bin segment
        # of their own, so every image keeps its size
        image_end = image.rom_start + image.size
        following = ordered[k + 1].rom_start if k + 1 < len(ordered) else end
        if following is not None and following > image_end:
            segs.append([image_end, "bin", f"unclaimed_{image_end:X}"])
    if end is not None:
        segs.append([end])
    config["segments"] = segs
    return config


def export_yaml(images: list[KitImage], **kwargs) -> str:
    header = (f"# {EXPORT_PROVENANCE}\n# Segment starts, VRAM and function boundaries are the kit's; nothing of the "
              "ROM is in this file.\n")
    return header + dump_yaml(export_config(images, **kwargs)) + "\n"


def export_symbol_addrs(images: list[KitImage], existing: SymbolFile | None = None) -> tuple[str, dict]:
    """``symbol_addrs.txt`` for the kit's functions.  With ``existing``, its lines are kept as they
    are (attributes and comments preserved) and only functions it does not name are appended; a
    name it gives a different address is reported, not written."""
    out = existing if existing is not None else SymbolFile()
    if existing is None:
        out.lines.append(f"// {EXPORT_PROVENANCE}")
    known = out.by_name()
    overlapping = _overlaps(images)
    exported: dict[str, str] = {}                  # name -> image it was written for
    added, conflicts, kept, renamed = 0, [], 0, []
    for image in sorted(images, key=lambda i: (i.rom_start, i.vram, i.name)):
        for name, vram, size in sorted(image.functions, key=lambda r: r[1]):
            rom = image.rom_start + (vram - image.vram) if image.name in overlapping else None
            if name in exported and exported[name] != image.name:
                # the same name in two images (address-named functions of overlays sharing a VRAM
                # range): symbol names must be unique, so the later image's copy is qualified
                new = f"{name}_{image.name}"
                renamed.append({"name": name, "image": image.name, "written_as": new})
                name = new
            elif name in known:
                if any(e.address == vram for e in known[name]):
                    kept += 1
                else:
                    conflicts.append({"name": name, "kit": f"0x{vram:08X}",
                                      "existing": [f"0x{e.address:08X}" for e in known[name]]})
                continue
            out.add(new_entry(name, vram, type="func", size=size, rom=rom))
            exported[name] = image.name
            added += 1
    return out.render(), {"added": added, "kept_existing": kept, "conflicts": conflicts, "renamed": renamed,
                          "rom_attribute_on": sorted(overlapping)}
