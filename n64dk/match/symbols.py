"""Symbol maps for relocating compiled objects at their retail addresses.

Sources: splat ``symbol_addrs.txt`` (``name = 0xADDR; // ...``), function catalogues in TOML
(any table carrying ``name`` plus ``vram``/``address``/``addr``, e.g. ``[[section]]`` with a
``functions`` array), ``/* LDSYM name=0xADDR */`` source markers, and ``__asm__("label")``
aliases on declarations.  Names of the form ``func_XXXXXXXX``/``D_XXXXXXXX`` resolve to
their embedded address when no explicit entry exists.
"""
from __future__ import annotations

import fnmatch
import os
from functools import lru_cache
import re
import tomllib
from pathlib import Path

INFERRED = re.compile(r"(?:D|func|jtbl|sym|data|B|L)_([0-9A-Fa-f]{6,8})")
SPLAT_LINE = re.compile(r"^\s*([A-Za-z_.$][\w.$]*)\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*;(.*)$")
LDSYM = re.compile(r"\bLDSYM\s+([A-Za-z_.$][\w.$]*)\s*=\s*(0x[0-9A-Fa-f]+|\d+)")
ASM_ALIAS = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*(?:\([^;{}]*\)\s*)?__asm__\s*\(\s*\"([^\"]+)\"\s*\)")


class SymbolMap(dict):
    """``dict`` of name -> address that also answers inferable ``func_``/``D_`` names."""

    sizes: dict[str, int]

    def __init__(self, *args, infer_names: bool = True, **kw):
        super().__init__(*args, **kw)
        self.sizes = {}
        self.infer_names = infer_names
        self.conflicts: dict[str, set[int]] = {}      # a name given more than one address

    def assign(self, name: str, address: int) -> None:
        """Set ``name``; a second, different address for it is recorded in ``conflicts``
        (the last one still stands, as before, but it is no longer silent: ``check_names``)."""
        if dict.__contains__(self, name) and dict.__getitem__(self, name) != address:
            self.conflicts.setdefault(name, {dict.__getitem__(self, name)}).add(address)
        self[name] = address

    def __contains__(self, name) -> bool:  # used by n64dk.elf.Elf.relocate
        return dict.__contains__(self, name) or (self.infer_names and bool(INFERRED.fullmatch(str(name))))

    def __missing__(self, name):
        m = INFERRED.fullmatch(str(name))
        if m and self.infer_names:
            return int(m.group(1), 16)
        raise KeyError(name)

    def resolve(self, name: str) -> int | None:
        try:
            return self[name]
        except KeyError:
            return None

    def copy(self) -> "SymbolMap":
        out = SymbolMap(self, infer_names=self.infer_names)
        out.sizes = dict(self.sizes)
        out.conflicts = {k: set(v) for k, v in self.conflicts.items()}
        return out

    def explicit(self) -> "SymbolMap":
        """Copy containing only independently loaded entries.

        Address-bearing names are a useful exploration convention, but their spelling is not
        proof of an address—especially in overlays whose catalogue and link coordinates differ.
        """
        out = SymbolMap(self, infer_names=False)
        out.sizes = dict(self.sizes)
        out.conflicts = {k: set(v) for k, v in self.conflicts.items()}
        return out

    def by_address(self) -> dict[int, str]:
        return {v: k for k, v in self.items()}


def load_splat(text: str, into: SymbolMap) -> None:
    for line in text.splitlines():
        m = SPLAT_LINE.match(line)
        if not m:
            continue
        name, addr = m.group(1), int(m.group(2), 0)
        into.assign(name, addr) if isinstance(into, SymbolMap) else into.__setitem__(name, addr)
        size = re.search(r"\bsize:(0x[0-9A-Fa-f]+|\d+)", m.group(3))
        if size:
            into.sizes[name] = int(size.group(1), 0)


def load_toml(data, into: SymbolMap) -> None:
    if isinstance(data, dict):
        name = data.get("name")
        addr = next((data[k] for k in ("vram", "address", "addr") if isinstance(data.get(k), int)), None)
        if isinstance(name, str) and addr is not None and not isinstance(data.get("functions"), list):
            into.assign(name, addr) if isinstance(into, SymbolMap) else into.__setitem__(name, addr)
            if isinstance(data.get("size"), int):
                into.sizes[name] = data["size"]
        for value in data.values():
            if isinstance(value, (dict, list)):
                load_toml(value, into)
    elif isinstance(data, list):
        for value in data:
            load_toml(value, into)


def load(paths, into: SymbolMap | None = None) -> SymbolMap:
    out = into if into is not None else SymbolMap()
    for path in paths:
        path = Path(path)
        text = path.read_text(errors="replace")
        if path.suffix == ".toml":
            load_toml(tomllib.loads(text), out)
        else:
            load_splat(text, out)
    return out


def from_source(text: str, known: SymbolMap) -> dict[str, int]:
    """Extra symbols from ``LDSYM`` markers and ``__asm__`` aliases resolvable via ``known``."""
    extra = {m.group(1): int(m.group(2), 0) for m in LDSYM.finditer(text)}
    for m in ASM_ALIAS.finditer(text):
        addr = extra.get(m.group(2))
        if addr is None:
            addr = known.resolve(m.group(2))
        if addr is not None:
            extra.setdefault(m.group(2), addr)
    return extra


def discover(project: Path) -> list[Path]:
    """Conventional symbol files inside a project (splat symbol_addrs and function catalogues)."""
    return list(_discover_cached(project.resolve()))


@lru_cache(maxsize=32)
def _discover_cached(project: Path) -> tuple[Path, ...]:
    """Cache immutable discovery results shared by parallel matching workers."""
    found: list[Path] = []
    canonical_build = project / "build" / "us"
    survey_symbols = canonical_build / "survey" / "symbol_addrs.txt"
    if survey_symbols.is_file():
        found.append(survey_symbols)
    audit = canonical_build / "audit"
    if audit.is_dir():
        found.extend(sorted(audit.glob("*_functions/symbol_addrs.txt")))
    for pattern in ("symbol_addrs*.txt", "*function_boundaries*.toml", "*symbols*.toml"):
        for root in (project, project / "config"):
            if root.is_dir():
                found += sorted(p for p in _walk(root, pattern)
                                if not {".toolchain", "build"} & set(p.relative_to(project).parts))
    return tuple(dict.fromkeys(found))


def _walk(root: Path, pattern: str):
    """Files under ``root`` matching ``pattern``, following linked directories (a pinned
    ground-truth snapshot links the project's ignored directories)."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in (
            ".git", ".git-data", ".toolchain", ".venv", ".n64dk", ".tmp",
            "build", "node_modules", "results", "__pycache__",
        )]
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                yield Path(dirpath, name)


# ---------------------------------------------------------------------------------------------
# Real names.  A draft may call library functions by their real names (``sin``, ``cos``,
# ``sqrtf``): at -O0 the operand order around a call depends on the declaration, so the names are
# worth having.  But a name only ever reaches the comparison as an address, and a relocation to
# the wrong one of two library functions of the same shape is invisible in a relocated compare:
# ``sin`` and ``cos`` swapped give the same instructions.  So every real name must resolve to one
# address in the project, and where the kit knows what the code at that address is (an SDK
# signature match: ``n64dk.analysis.naming``'s ``sdk`` field, NAMES.tsv) the two must agree.


def identities_from_names(text: str) -> dict[int, str]:
    """address -> library name from a NAMES.tsv (the ``sdk`` column: ``name [source]``)."""
    rows = [line.split("\t") for line in text.splitlines() if line.strip()]
    if not rows or "sdk" not in rows[0] or "vram" not in rows[0]:
        return {}
    at, column = rows[0].index("vram"), rows[0].index("sdk")
    out = {}
    for row in rows[1:]:
        if len(row) <= max(at, column) or not row[column].strip():
            continue
        name = row[column].split(" [", 1)[0].strip()
        try:
            out[int(row[at], 16)] = name
        except ValueError:
            continue
    return out


def check_names(symbols: SymbolMap, identities: dict[int, str] | None = None) -> list[dict]:
    """Findings about real names: ``ambiguous-name`` (one name, several addresses),
    ``swapped-pair`` (name A at the code identified as B and B at the code identified as A),
    ``identity-mismatch`` (a name at code identified as another library function).  Address
    names (``func_80001234``) carry their own address and are not checked."""
    out = []
    for name, addresses in sorted(getattr(symbols, "conflicts", {}).items()):
        out.append({"kind": "ambiguous-name", "name": name,
                    "addresses": [f"0x{a:08X}" for a in sorted(addresses)],
                    "why": f"{name} is given {len(addresses)} addresses; a name must resolve to one per project"})
    if identities:
        known_names = set(identities.values())
        where = {name: address for name, address in dict.items(symbols) if not INFERRED.fullmatch(name)}
        reported = set()
        for name, address in sorted(where.items()):
            actual = identities.get(address)
            if actual is None or actual == name or name not in known_names:
                continue                     # unknown code, agreeing name, or not a library name
            partner = where.get(actual)
            if partner is not None and identities.get(partner) == name:
                pair = tuple(sorted((name, actual)))
                if pair not in reported:
                    reported.add(pair)
                    out.append({"kind": "swapped-pair", "names": list(pair),
                                "addresses": [f"0x{where[pair[0]]:08X}", f"0x{where[pair[1]]:08X}"],
                                "why": f"{pair[0]} and {pair[1]} are each declared at the other's code "
                                       "(SDK signature identity)"})
                continue
            out.append({"kind": "identity-mismatch", "name": name, "address": f"0x{address:08X}",
                        "identified_as": actual,
                        "why": f"{name} is declared at 0x{address:08X}, whose code is identified as {actual}"})
    return out


class NameError_(ValueError):
    """Real names that do not resolve to one address or contradict the code they name."""


def require_consistent_names(symbols: SymbolMap, identities: dict[int, str] | None = None) -> list[dict]:
    """``check_names``; raises ``NameError_`` on an ambiguous name or a swapped pair (a relocated
    compare cannot see either), returns the remaining findings (identity mismatches) to report."""
    findings = check_names(symbols, identities)
    fatal = [f for f in findings if f["kind"] in ("ambiguous-name", "swapped-pair")]
    if fatal:
        raise NameError_("; ".join(f["why"] + (f" ({', '.join(f['addresses'])})" if "addresses" in f else "")
                                   for f in fatal))
    return findings
