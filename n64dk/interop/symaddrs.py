"""``symbol_addrs.txt`` read and written without losing anything.

The format (splat and the projects built on it)::

    // a comment line
    name = 0x80001234; // type:func size:0x40 rom:0x1234 some free text

Each symbol line is ``name = address;`` followed by an optional ``//`` comment of whitespace-
separated ``key:value`` attributes.  Any other text in the comment (``// 0x10 (player, true)``),
attributes the kit does not know, blank lines and comment lines are kept: a file that is read and
written back unchanged is byte-identical, and an entry the kit changes keeps its other attributes.

The kit reads: ``type``, ``size``, ``rom`` (the ROM offset that tells overlapping overlays apart),
``segment`` (a segment name, same purpose; newer splat versions) and ``ignore``.  Every other key
(``defined``, ``name_end``, ``force_migration``, ``dont_allow_addend``, ...) is carried, not read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LINE = re.compile(r"^(\s*)([A-Za-z_.$][\w.$]*)(\s*=\s*)(0x[0-9A-Fa-f]+|\d+)(\s*;)(.*)$")
ATTRIBUTE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(\S+)$")


def _number(text: str) -> int | None:
    try:
        return int(text, 0)
    except ValueError:
        return None


@dataclass
class SymbolEntry:
    name: str
    address: int
    attributes: list[tuple[str, str]] = field(default_factory=list)   # in file order
    free_text: list[str] = field(default_factory=list)                 # comment words that are not key:value
    line: int | None = None                                            # 1-based line in the source file
    raw: str | None = None                                             # the source line, for verbatim output
    _parsed: tuple | None = field(default=None, repr=False, compare=False)

    def get(self, key: str) -> str | None:
        for k, v in self.attributes:
            if k == key:
                return v
        return None

    def set(self, key: str, value: str) -> None:
        for i, (k, _v) in enumerate(self.attributes):
            if k == key:
                self.attributes[i] = (k, value)
                return
        self.attributes.append((key, value))

    @property
    def type(self) -> str | None:
        return self.get("type")

    @property
    def size(self) -> int | None:
        value = self.get("size")
        return _number(value) if value is not None else None

    @property
    def rom(self) -> int | None:
        value = self.get("rom")
        return _number(value) if value is not None else None

    @property
    def segment(self) -> str | None:
        return self.get("segment")

    @property
    def ignored(self) -> bool:
        return (self.get("ignore") or "").lower() == "true"

    def _state(self) -> tuple:
        return (self.name, self.address, tuple(self.attributes), tuple(self.free_text))

    def modified(self) -> bool:
        return self.raw is None or self._parsed != self._state()

    def render(self) -> str:
        if not self.modified():
            return self.raw
        comment = " ".join([f"{k}:{v}" for k, v in self.attributes] + self.free_text)
        return f"{self.name} = 0x{self.address:08X};" + (f" // {comment}" if comment else "")


def parse_line(text: str, line: int | None = None) -> SymbolEntry | None:
    m = LINE.match(text)
    if not m:
        return None
    address = _number(m.group(4))
    if address is None:
        return None
    rest = m.group(6)
    attributes, free = [], []
    comment = rest.strip()
    if comment.startswith("//"):
        comment = comment[2:]
    elif comment:
        # text after ';' that is not a comment: keep it verbatim as free text
        free.append(comment)
        comment = ""
    for word in comment.split():
        a = ATTRIBUTE.match(word)
        if a:
            attributes.append((a.group(1), a.group(2)))
        else:
            free.append(word)
    entry = SymbolEntry(m.group(2), address, attributes, free, line, text)
    entry._parsed = entry._state()
    return entry


@dataclass
class SymbolFile:
    """Lines in order: a ``SymbolEntry`` for each symbol line, the text itself for every other line."""
    lines: list = field(default_factory=list)
    trailing_newline: bool = True

    @classmethod
    def parse(cls, text: str) -> "SymbolFile":
        out = cls(trailing_newline=text.endswith("\n") or not text)
        for number, raw in enumerate(text.splitlines(), 1):
            entry = parse_line(raw, number)
            out.lines.append(entry if entry is not None else raw)
        return out

    @classmethod
    def load(cls, path) -> "SymbolFile":
        return cls.parse(Path(path).read_text(errors="replace"))

    @property
    def entries(self) -> list[SymbolEntry]:
        return [line for line in self.lines if isinstance(line, SymbolEntry)]

    def by_name(self) -> dict[str, list[SymbolEntry]]:
        out: dict[str, list[SymbolEntry]] = {}
        for entry in self.entries:
            out.setdefault(entry.name, []).append(entry)
        return out

    def add(self, entry: SymbolEntry) -> None:
        self.lines.append(entry)

    def render(self) -> str:
        text = "\n".join(line.render() if isinstance(line, SymbolEntry) else line for line in self.lines)
        return text + ("\n" if self.trailing_newline and self.lines else "")


def entry(name: str, address: int, **attributes) -> SymbolEntry:
    """A new entry; attributes in the conventional order type, size, rom, then the rest."""
    order = ["type", "size", "rom", "segment"]
    keys = [k for k in order if k in attributes] + [k for k in attributes if k not in order]
    values = []
    for key in keys:
        value = attributes[key]
        if value is None:
            continue
        values.append((key, f"0x{value:X}" if isinstance(value, int) and not isinstance(value, bool)
                       else str(value).lower() if isinstance(value, bool) else str(value)))
    return SymbolEntry(name, address, values)
