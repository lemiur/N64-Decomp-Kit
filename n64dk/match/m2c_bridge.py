"""First-draft C from retail bytes through m2c (https://github.com/matt-kempster/m2c).

``source-scaffold`` emits an empty body.  Established decompilation workflows instead start
from an m2c draft: it recovers control flow, calls, struct offsets and jump tables, and on
IDO targets a non-trivial share of small functions match from the draft alone.

``emit_asm`` turns one function's retail bytes into symbolized GNU-style assembly
(``glabel``, ``.L`` labels, ``%hi``/``%lo`` pairs, jump tables in ``.rodata``).  It has two
modes for control flow that leaves the function (a branch to a shared epilogue or to a
mis-sized neighbour, or a jump-table entry outside the body):

* ``external="symbol"`` (default) keeps the listing faithful to the retail bytes and names
  the target (``L_XXXXXXXX`` or a known symbol), which is what decomp.me scratches and
  decomp-permuter targets need;
* ``external="stub"`` routes each such edge to a local label after the body that tail-jumps
  to the target.  That is the same control flow, and it is the only form m2c accepts, because
  m2c rejects any branch whose label is not in the function.

``find_jump_tables`` reuses the function discoverer's jump-table recogniser, and ``draft``
runs m2c on the stub listing.  m2c is optional: ``locate_m2c`` reports every place it
looked, and ``draft`` always returns the assembly plus a ``status``.

The draft is a starting point for the normal matching loop, never credited by itself.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import rabbitizer

from ..analysis import functions as fa

MACROS = (".macro glabel label\n    .global \\label\n    \\label:\n.endm\n"
          ".macro jlabel label\n    \\label:\n.endm\n")
_TARGETS = {"kmc": "mips-gcc-c", "gcc": "mips-gcc-c", "sn": "mips-gcc-c", "ido": "mips-ido-c"}
_ADDRESS_LO, _ADDRESS_HI = 0x80000000, 0xC0000000
# m2c only treats a ``jr reg`` as a switch when a ``%lo`` read of a symbol with one of these
# prefixes precedes it (m2c/flow_graph.py); a table named anything else is not found.
JTBL_PREFIXES = ("jtbl", "jpt_", "lbl_", "jumptable_")
ENV_VAR = "N64DK_M2C"
INSTALL_HINT = "pip install 'm2c @ git+https://github.com/matt-kempster/m2c'"
SYMBOL = re.compile(r"^[A-Za-z_.$][A-Za-z0-9_.$]*$")
_ADDU = 0x21

# draft() status values
OK, UNAVAILABLE, INVALID_INPUT, LAUNCH_FAILED, TIMEOUT, FAILED = (
    "ok", "unavailable", "invalid-input", "launch-failed", "timeout", "failed")


def _name(address: int, names: dict[int, str], prefix: str) -> str:
    return names.get(address) or f"{prefix}_{address:08X}"


def _words(code: bytes) -> list[int]:
    return [struct.unpack_from(">I", code, i)[0] for i in range(0, len(code) - len(code) % 4, 4)]


def _pairs(words: list[int], vram: int) -> dict[int, int]:
    """site -> full address for each ``lui`` and its ``%lo`` users (straight-line tracking).

    One ``%hi`` may serve several ``%lo``; the ``lui`` is named after the first one.  An
    ``addu`` of a ``%hi`` register and an index keeps the ``%hi`` for a following load or
    store, which is how both GCC and IDO index jump tables and arrays
    (``lui at,%hi(t); addu at,at,v0; lw v0,%lo(t)(at)``).  ``lui``/``ori`` is a constant
    (``li``), never an address: ``%hi`` of a symbol with bit 15 set would not reassemble to
    the retail ``lui``, so those are left numeric."""
    hi: dict[int, tuple[int, int]] = {}          # register -> (lui site, upper half)
    out: dict[int, int] = {}
    for i, word in enumerate(words):
        site, op, rs, rt = vram + 4 * i, word >> 26, (word >> 21) & 31, (word >> 16) & 31
        if op == 15:
            if rt:
                hi[rt] = (site, (word & 0xFFFF) << 16)
            continue
        ins = rabbitizer.Instruction(word, site)
        carried = None
        if op == 0 and word & 0x3F == _ADDU:
            held = {r for r in (rs, rt) if r and r in hi}
            if len(held) == 1 and rs != rt:
                carried = hi[held.pop()]
        elif (op == 9 or ins.doesLoad() or ins.doesStore()) and rs in hi and rs != 0:
            lui_site, upper = hi[rs]
            lo = word & 0xFFFF
            address = (upper + (lo - 0x10000 if lo & 0x8000 else lo)) & 0xFFFFFFFF
            if _ADDRESS_LO <= address < _ADDRESS_HI:
                out.setdefault(lui_site, address)
                out[site] = address
        dest = ins.getDestinationGpr()
        if dest is not None and not ins.doesStore():
            hi.pop(int(dest.value), None)
        if carried is not None and (word >> 11) & 31:
            hi[(word >> 11) & 31] = carried
    return out


def _tables(jump_tables) -> tuple[list[tuple[int, list[int]]], set[int]]:
    """Split ``[(table_address, targets)]`` into real tables and label-only hints (address 0)."""
    tables: dict[int, list[int]] = {}
    hints: set[int] = set()
    for address, targets in jump_tables or ():
        if address:
            tables.setdefault(address, list(targets))
        else:
            hints.update(targets)
    return sorted(tables.items()), hints


def _table_name(address: int, names: dict[int, str]) -> str:
    known = names.get(address)
    return known if known and known.startswith(JTBL_PREFIXES) else f"jtbl_{address:08X}"


@dataclass
class Layout:
    """Where control flow in one function goes."""
    local: set[int] = field(default_factory=set)       # labelled instructions inside the body
    external: set[int] = field(default_factory=set)    # branch / table targets outside it
    misaligned: set[int] = field(default_factory=set)  # table words that cannot be code


def layout(code: bytes, vram: int, jump_tables=None) -> Layout:
    words = _words(code)
    end = vram + 4 * len(words)
    tables, hints = _tables(jump_tables)
    out = Layout()
    for target in [t for _a, targets in tables for t in targets] + sorted(hints):
        if target % 4:
            out.misaligned.add(target)
        elif vram <= target < end:
            out.local.add(target)
        else:
            out.external.add(target)
    for i, word in enumerate(words):
        ins = rabbitizer.Instruction(word, vram + 4 * i)
        if ins.isBranch():
            target = ins.getBranchVramGeneric() & 0xFFFFFFFF
            if vram <= target < end:
                out.local.add(target)
            elif not ins.doesLink():
                out.external.add(target)
        elif ins.isJumpWithAddress() and not ins.doesLink():
            target = ins.getInstrIndexAsVram()
            if vram < target < end:                     # ``j`` to the entry itself is a tail call
                out.local.add(target)
    return out


def emit_asm(name: str, code: bytes, vram: int, names: dict[int, str] | None = None,
             jump_tables: list[tuple[int, list[int]]] | None = None, macros: bool = True,
             external: str = "symbol", gp: int | None = None) -> str:
    """Symbolized assembly for one function.  ``names`` maps addresses to symbol names;
    ``jump_tables`` is ``[(table_address, [targets...])]`` (see ``find_jump_tables``).
    ``external`` is ``"symbol"`` or ``"stub"`` (see the module docstring).  ``gp`` is the
    program's ``_gp`` when known: GP-relative accesses become ``%gp_rel(sym)($gp)`` so the
    decompiler names the small global (a ``-G`` title)."""
    if external not in ("symbol", "stub"):
        raise ValueError(f"external must be 'symbol' or 'stub', not {external!r}")
    symbols = dict(names or {})
    tables, _hints = _tables(jump_tables)
    for address, _targets in tables:
        symbols[address] = _table_name(address, symbols)
    words = _words(code)
    end = vram + 4 * len(words)
    flow = layout(code, vram, jump_tables)
    pairs = _pairs(words, vram)
    stub = external == "stub"

    def target_text(target: int) -> str:
        if target in flow.local:
            return f".L{target:08X}"
        return f".L{target:08X}" if stub else _name(target, symbols, "L")

    lines = []
    if macros:
        lines.append(MACROS)
    lines += [".set noat", ".set noreorder", ".section .text", f"glabel {name}"]
    for i, word in enumerate(words):
        site = vram + 4 * i
        ins = rabbitizer.Instruction(word, site)
        if site in flow.local:
            lines.append(f".L{site:08X}:")
        override = None
        if ins.isBranch():
            target = ins.getBranchVramGeneric() & 0xFFFFFFFF
            if target in flow.local:
                override = f".L{target:08X}"
            elif ins.doesLink():                         # bal / bgezal out of the body: a call
                override = _name(target, symbols, "func")
            else:
                override = target_text(target)
        elif ins.isJumpWithAddress():
            target = ins.getInstrIndexAsVram()
            override = (f".L{target:08X}" if not ins.doesLink() and target in flow.local
                        else _name(target, symbols, "func"))
        elif site in pairs:
            symbol = _name(pairs[site], symbols, "D")
            override = f"%hi({symbol})" if word >> 26 == 15 else f"%lo({symbol})"
        elif gp is not None and (word >> 21) & 31 == 28 and (word >> 16) & 31 != 28 \
                and (word >> 26 == 9 or ins.doesLoad() or ins.doesStore()):
            field = word & 0xFFFF
            address = (gp + (field - 0x10000 if field & 0x8000 else field)) & 0xFFFFFFFF
            override = f"%gp_rel({_name(address, symbols, 'D')})"
        if not ins.isValid():
            lines.append(f"    /* {site:08X} {word:08X} */ .word 0x{word:08X}")
            continue
        text = ins.disassemble(immOverride=override) if override else ins.disassemble()
        lines.append(f"    /* {site:08X} {word:08X} */ {' '.join(text.split())}")
    if len(code) % 4:
        lines.append(f"    /* {len(code) % 4} trailing byte(s) not a whole instruction; omitted */")
    if stub:
        for target in sorted(flow.external):
            lines += [f".L{target:08X}:",
                      f"    /* not retail code: edge leaving {name} modelled as a tail jump */",
                      f"    j {_name(target, symbols, 'L')}", "    nop"]
    if tables:
        lines.append(".section .rodata")
        for table, targets in tables:
            lines.append(f"glabel {symbols[table]}")
            lines += [f"    .word 0x{t:08X}" if t in flow.misaligned else f"    .word {target_text(t)}"
                      for t in targets]
    return "\n".join(lines) + "\n"


class _TableProbe(fa.Analyser):
    """Only the discoverer's jump-table recogniser (``Analyser.jump_table``), without the
    whole-image indexing its constructor does, so one function in a large ROM is cheap."""

    def __init__(self, image: fa.Image, text_hi: int):   # Analyser.__init__ deliberately not called
        self.img, self.text_hi = image, text_hi
        self.tables: dict[int, fa.JumpTable] = {}
        self.const_jumps: dict[int, int] = {}


class _Layered(fa.Image):
    """The function's own image, with other images of the project behind it for the addresses it does
    not hold: an overlay's jump tables may stand in the resident image (measured on a title: 44
    functions of four overlays, every table at a low resident address)."""

    def __init__(self, data: bytes, vram: int, behind):
        super().__init__(data, vram)
        self._behind = [fa.Image(other, other_vram) for other, other_vram in behind]

    def _layer(self, v: int):
        if self.vram <= v < self.end:
            return None
        return next((layer for layer in self._behind if v in layer), None)

    def __contains__(self, v: int) -> bool:
        return self.vram <= v < self.end or self._layer(v) is not None

    def word(self, v: int) -> int:
        layer = self._layer(v)
        return super().word(v) if layer is None else layer.word(v)


def find_jump_tables(data: bytes, image_vram: int, start: int, size: int,
                     also=()) -> list[tuple[int, list[int]]]:
    """Jump tables dispatched by ``jr reg`` inside ``[start, start+size)`` of a linear image
    whose first byte is at ``image_vram``, as recognised by function discovery.  A table bounded
    by ``sltiu`` keeps every entry (targets outside the body become external edges); an
    unbounded one is read only while entries stay inside the body.  ``also`` lists other images
    ``(data, vram)`` a table may be read from when its address lies outside this one; each is tried by
    itself, in order, for the dispatches still without a table, so an image that merely covers the
    address with something else decides nothing."""
    end = start + size
    found: dict[int, list[int]] = {}
    settled: set[int] = set()
    for behind in [()] + [(other,) for other in also]:
        image = _Layered(data, image_vram, behind) if behind else fa.Image(data, image_vram)
        if size <= 0 or start % 4 or not image_vram <= start < image.end or end > image.end:
            return []
        wide, narrow = _TableProbe(image, image.end), _TableProbe(image, end)
        for pc in range(start, end - 4, 4):
            if pc in settled or not image.ins(pc).isJrNotRa():
                continue
            table = wide.jump_table(start, pc)
            if table is None or not table.bounded:
                table = narrow.jump_table(start, pc)
            if table is not None:
                found.setdefault(table.address, table.targets)
                settled.add(pc)
    return sorted(found.items())


# -- locating m2c ----------------------------------------------------------------------------------
@dataclass
class Discovery:
    command: list[str] | None
    source: str                      # explicit, env, PATH, scripts, module or none
    tried: list[str] = field(default_factory=list)
    error: str | None = None


def _checkout_command(root: Path) -> list[str]:
    boot = f"import sys; sys.path.insert(0, {str(root)!r}); from m2c.main import main; main()"
    return [sys.executable, "-c", boot]


def _command_for(spec: str) -> tuple[list[str] | None, str]:
    """Command for an explicit m2c location: an executable, ``m2c.py``, a source checkout
    (directory containing ``m2c/main.py``) or a command line such as ``python3 /x/m2c.py``."""
    path = Path(spec).expanduser()
    if path.is_dir():
        for root in (path, path.parent) if path.resolve().name == "m2c" else (path,):
            if (root / "m2c" / "main.py").is_file():
                return _checkout_command(root.resolve()), ""
        return None, f"{spec}: directory is not an m2c checkout (no m2c/main.py)"
    if path.is_file():
        if path.suffix == ".py":
            return [sys.executable, str(path.resolve())], ""
        if os.access(path, os.X_OK):
            return [str(path.resolve())], ""
        return None, f"{spec}: file is not executable"
    try:
        argv = shlex.split(spec)
    except ValueError as exc:
        return None, f"{spec!r}: {exc}"
    if argv:
        exe = shutil.which(argv[0])
        if exe:
            return [exe, *argv[1:]], ""
    return None, f"{spec}: not found"


_LOCATED: dict[tuple, "Discovery"] = {}


def locate_m2c(explicit: str | None = None, environ: dict | None = None) -> Discovery:
    """Cached per (explicit, $N64DK_M2C, PATH, interpreter) for the process's own environment:
    every draft asks."""
    if environ is not None:
        return _locate_m2c(explicit, environ)
    key = (explicit, os.environ.get(ENV_VAR), os.environ.get("PATH"), sys.executable)
    found = _LOCATED.get(key)
    if found is None or found.command is None:
        found = _LOCATED[key] = _locate_m2c(explicit, None)
    return found


def _locate_m2c(explicit: str | None = None, environ: dict | None = None) -> Discovery:
    """Find m2c: ``explicit`` path/command, then ``$N64DK_M2C``, then ``m2c`` on PATH, then
    next to the running interpreter, then an importable ``m2c`` package.  An explicit or
    environment setting that does not resolve is an error, never silently replaced."""
    environ = os.environ if environ is None else environ
    for value, source in ((explicit, "explicit"), (environ.get(ENV_VAR), "env")):
        if value:
            command, why = _command_for(value)
            label = f"{source}: {value}" if source == "explicit" else f"${ENV_VAR}={value}"
            return Discovery(command, source, [label], None if command else why)
    tried = ["PATH: m2c"]
    exe = shutil.which("m2c")
    if exe:
        return Discovery([exe], "PATH", tried)
    exe_name = "m2c.exe" if os.name == "nt" else "m2c"
    dirs = []
    for d in (sysconfig.get_path("scripts"), str(Path(sys.executable).parent)):
        if d and d not in dirs:
            dirs.append(d)
    for d in dirs:
        candidate = Path(d) / exe_name
        tried.append(f"scripts: {candidate}")
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return Discovery([str(candidate)], "scripts", tried)
    tried.append("module: m2c.main")
    try:
        spec = importlib.util.find_spec("m2c")
    except (ImportError, ValueError):
        spec = None
    for location in (spec.submodule_search_locations or []) if spec else []:
        if (Path(location) / "main.py").is_file():
            return Discovery([sys.executable, "-c", "from m2c.main import main; main()"], "module", tried)
    return Discovery(None, "none", tried, f"m2c not found ({INSTALL_HINT}, or set ${ENV_VAR} / pass --m2c)")


def find_m2c(explicit: str | None = None) -> list[str] | None:
    return locate_m2c(explicit).command


def m2c_target(compiler: str) -> tuple[str, bool]:
    """(m2c ``-t`` triple, whether ``compiler`` was recognised).  A full triple passes through."""
    if compiler.startswith(("mips-", "mipsel-")):
        return compiler, True
    family = compiler.split("+")[0].rstrip("0123456789.-_").lower()
    return _TARGETS.get(family, "mips-gcc-c"), family in _TARGETS


def _failure_text(stdout: str, stderr: str, returncode: int) -> str:
    """m2c prints DecompFailure/internal errors as a C comment on stdout and usage errors on stderr."""
    match = re.search(r"/\*\s*\n?((?:Decompilation failure|Internal error).*?)\*/", stdout, re.S)
    parts = [match.group(1).strip() if match else "", stderr.strip()]
    text = "\n".join(p for p in parts if p) or stdout.strip()
    return text[-2000:] or f"m2c exit {returncode}"


def draft(name: str, code: bytes, vram: int, compiler: str = "kmc", context: Path | None = None,
          names: dict[int, str] | None = None, jump_tables=None, extra: list[str] | None = None,
          timeout: float = 60, m2c: str | None = None, environ: dict | None = None, gp: int | None = None) -> dict:
    """Run m2c on one function and report what happened.

    Always returns ``asm`` (faithful listing) and ``m2c_input`` (stub listing), plus
    ``status`` (``ok``, ``unavailable``, ``invalid-input``, ``launch-failed``, ``timeout`` or
    ``failed``), ``error``, ``source``, ``command``, ``target``, ``returncode``, ``stderr``,
    ``external`` (out-of-body targets that were stubbed), ``jump_tables``, ``m2c_errors``
    (``M2C_ERROR`` markers in the draft) and ``warnings``.  ``--valid-syntax`` is used so the
    draft is more likely to compile unedited and ``--no-cache`` so no pickle is written next to
    the context file."""
    target, known = m2c_target(compiler)
    flow = layout(code, vram, jump_tables)
    result = {"name": name, "status": None, "error": None, "source": None, "command": None,
              "target": target, "returncode": None, "stderr": "", "m2c_errors": 0,
              "external": [f"0x{t:08X}" for t in sorted(flow.external)],
              "jump_tables": [f"0x{a:08X}" for a, _t in _tables(jump_tables)[0]],
              "warnings": [] if known else [f"unknown compiler {compiler!r}; using m2c target {target}"],
              "asm": "", "m2c_input": ""}
    if flow.misaligned:
        result["warnings"].append("jump-table words that cannot be code addresses: "
                                  + ", ".join(f"0x{t:08X}" for t in sorted(flow.misaligned)))
    if len(code) % 4:
        result["warnings"].append(f"size is not a multiple of 4; last {len(code) % 4} byte(s) ignored")

    def fail(status: str, error: str) -> dict:
        result.update(status=status, error=error)
        return result

    if not SYMBOL.match(name):
        return fail(INVALID_INPUT, f"{name!r} is not a valid assembler symbol")
    if len(code) < 4:
        return fail(INVALID_INPUT, "no instructions (code is shorter than 4 bytes)")
    result["asm"] = emit_asm(name, code, vram, names, jump_tables, gp=gp)
    result["m2c_input"] = emit_asm(name, code, vram, names, jump_tables, macros=False, external="stub", gp=gp)
    if context is not None and not Path(context).is_file():
        return fail(INVALID_INPUT, f"context file not found: {context}")
    found = locate_m2c(m2c, environ)
    if found.command is None:
        result["tried"] = found.tried
        return fail(UNAVAILABLE, found.error or "m2c not found")
    with tempfile.TemporaryDirectory(prefix="n64dk-m2c-") as tmp:
        path = Path(tmp) / "input.s"
        path.write_text(result["m2c_input"])
        argv = [*found.command, "-t", target, "--valid-syntax", "--deterministic-vars", "--no-cache",
                "--globals", "used", "-f", name]
        if context is not None:
            argv += ["--context", str(Path(context).resolve())]
        argv += [*(extra or []), str(path)]
        result["command"] = argv
        try:
            from .. import safety
            from . import m2c_server
            done = None
            if m2c_server.wanted(found.command):
                try:
                    done = m2c_server.run(argv[len(found.command):], tmp, timeout)
                except RuntimeError:
                    done = None                                # the server failed: the command line
            if done is None:
                done = safety.run_tool(argv, kind="light", capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return fail(TIMEOUT, f"m2c timed out after {timeout}s")
        except OSError as exc:
            return fail(LAUNCH_FAILED, f"could not run {argv[0]}: {exc}")
    result.update(returncode=done.returncode, stderr=done.stderr.strip()[-2000:])
    if done.returncode != 0:
        return fail(FAILED, _failure_text(done.stdout, done.stderr, done.returncode))
    if not done.stdout.strip():
        return fail(FAILED, "m2c produced no output" + (f": {done.stderr.strip()[-500:]}" if done.stderr.strip() else ""))
    result.update(status=OK, source=done.stdout, m2c_errors=done.stdout.count("M2C_ERROR("))
    return result
