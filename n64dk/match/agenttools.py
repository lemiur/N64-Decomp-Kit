"""The tool protocol a model in the loop works with (title-independent).

A person or a coding agent matching one function needs a handful of tools, and the Madden campaign
(about 60 agents, ~400 functions the unattended run could not match) showed which: ``dis``,
``draft``, ``try`` (with lane and flag overrides and the sibling assembler lanes), ``exact`` (the
strict verifier, the only acceptance), ``regdump`` (GCC pass dumps), ``neighbours`` (matched
sources near the function, the unit's recipe, struct definitions already worked out) and
``boundary`` (the true start and size when the catalogue is wrong).  This module provides them
from a project's own ``n64dk.toml``; nothing here knows any title.

Configuration read (all optional except the images):

* ``[images.<name>]``: ``path``, ``vram_delta`` (vram - file offset), ``catalogue`` (functions
  toml), ``catalogue_shift`` (catalogue vram - real vram), ``min_vram``, ``manifests`` (strings or
  ``{path, pipeline}`` tables; rows ``OFFSET SIZE SOURCE [FLAGS ...] SHA`` with image-file offsets),
  ``symbols`` (symbol files for this image; default ``[match] symbols``);
* ``[run] pipeline/flags`` (the default recipe), ``[run.images.<name>.recipes]`` (manifest ->
  pipeline), ``[run.images.<name>] pipeline/flags``;
* ``[agent]``: ``regions`` (``[{image, start, end, pipeline, flags}]``: default recipe by address,
  e.g. a library region built with another compiler), ``sibling_lanes`` (``{"kmc+builtin" =
  ["kmc+asn64"]}``: lanes also tried when the recipe is a default), ``lessons`` (the project's
  lessons file).

Addresses are always the REAL (link) address the code and its callers use; a catalogue name is
accepted as a key.  ``exact`` accepts a file only when the kit's ``verify.strict_compare`` says
``exact`` and the source is plain C (no ``M2C_`` scaffolding, no ``asm`` in the file or in a header
it includes from its own folder: that is an *assisted* class, never plain C credit).
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shlex
import struct
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ADDR_NAME = re.compile(r"(?:D|func|jtbl|sym|data|B|L)_([0-9A-Fa-f]{8})")
LOWOPS = {"addiu", "ori", "lw", "sw", "lh", "lhu", "sh", "lb", "lbu", "sb", "lwc1", "swc1", "ldc1", "sdc1",
          "ld", "sd", "lwl", "lwr", "swl", "swr"}
DEFAULT_SIBLINGS = {"kmc+builtin": ["kmc+asn64"]}
TOOLS = ("dis", "draft", "try", "exact", "info", "regdump", "neighbours", "boundary")


class ToolError(ValueError):
    pass


def _num(value) -> int:
    return int(value, 0) if isinstance(value, str) else int(value)


@dataclass
class Image:
    name: str
    path: Path
    delta: int
    catalogue: Path | None = None
    shift: int = 0
    min_vram: int = 0
    manifests: list = field(default_factory=list)       # [(Path, pipeline or None)]
    symbols: list = field(default_factory=list)
    pipeline: str | None = None
    flags: str | None = None


@dataclass
class Function:
    image: str
    name: str
    vram: int
    size: int
    cat_vram: int = 0
    true_range: bool = False                              # size given explicitly (a repaired boundary)

    @property
    def key(self) -> str:
        return f"{self.image}_{self.vram:08X}"


class Workspace:
    """One project's images, catalogues, recipes and declared symbols, for the tools."""

    def __init__(self, project: str | Path, state: str | Path | None = None):
        from .toolchain import load_config
        self.root = Path(project).resolve()
        self.cfg = load_config(self.root)
        toml = self.root / "n64dk.toml"
        self.data = tomllib.loads(toml.read_text()) if toml.is_file() else {}
        self.state = Path(state).resolve() if state else self.root / ".n64dk" / "agent"
        self.agent = dict(self.data.get("agent") or {})
        run = dict(self.data.get("run") or {})
        self.run_pipeline = run.get("pipeline", "kmc+builtin")
        self.run_flags = run.get("flags", "-O2 -G0 -mips3 -mgp32 -mfp32")
        self.images: dict[str, Image] = {}
        run_images = run.get("images") or {}
        default_symbols = [str(p) for p in (self.data.get("match") or {}).get("symbols", [])]
        for name, table in (self.data.get("images") or {}).items():
            if not isinstance(table, dict) or "path" not in table or "vram_delta" not in table:
                continue
            recipes = dict((run_images.get(name) or {}).get("recipes") or {})
            recipes.update(table.get("recipes") or {})
            manifests = []
            for entry in table.get("manifests") or []:
                if isinstance(entry, dict):
                    manifests.append((self.root / entry["path"], entry.get("pipeline") or recipes.get(entry["path"])))
                else:
                    manifests.append((self.root / str(entry), recipes.get(str(entry))))
            self.images[name] = Image(
                name, self.root / table["path"], _num(table["vram_delta"]),
                self.root / table["catalogue"] if table.get("catalogue") else None,
                _num(table.get("catalogue_shift", 0)), _num(table.get("min_vram", 0)), manifests,
                [self.root / p for p in (table.get("symbols") or default_symbols)],
                (run_images.get(name) or {}).get("pipeline"), (run_images.get(name) or {}).get("flags"))
            for alias in table.get("aliases") or []:
                self.images.setdefault(alias, self.images[name])
        self._bytes: dict[str, bytes] = {}
        self._declared: dict[str, object] = {}
        self._manifest: dict[str, list] = {}

    # ------------------------------------------------------------------ images and catalogue
    def image(self, name: str) -> Image:
        if name not in self.images:
            raise ToolError(f"unknown image {name!r}; the project declares: {', '.join(sorted(self.images)) or 'none'}")
        return self.images[name]

    def image_bytes(self, name: str) -> bytes:
        if name not in self._bytes:
            self._bytes[name] = self.image(name).path.read_bytes()
        return self._bytes[name]

    def target(self, fn: Function) -> bytes:
        offset = fn.vram - self.image(fn.image).delta
        return self.image_bytes(fn.image)[offset:offset + fn.size]

    def catalogue(self, name: str) -> list[dict]:
        img = self.image(name)
        if img.catalogue is None or not img.catalogue.is_file():
            return []
        data = tomllib.loads(img.catalogue.read_text())
        rows = []
        for section in data.get("section") or [data]:
            for fn in section.get("functions") or []:
                if fn["vram"] < img.min_vram:
                    continue
                rows.append(dict(name=fn["name"], cat_vram=fn["vram"], vram=fn["vram"] - img.shift, size=fn["size"]))
        return rows

    def lookup(self, image: str, key: str, size: int | None = None) -> Function:
        """KEY: a real address (hex) or a catalogue name.  With ``size`` an address the catalogue
        does not start (a gap, or inside a mis-split row) is accepted as a true function start."""
        rows = self.catalogue(image)
        k = str(key).strip()
        if re.fullmatch(r"(0x)?[0-9A-Fa-f]{1,8}", k):
            v = int(k, 16)
            hit = [r for r in rows if r["vram"] == v]
            if hit:
                r = hit[0]
                return Function(image, r["name"], v, size or r["size"], r["cat_vram"], size is not None)
            if size and v % 4 == 0:
                img = self.image(image)
                if not rows or (min(r["vram"] for r in rows) <= v < max(r["vram"] + r["size"] for r in rows)) \
                        or 0 <= v - img.delta < len(self.image_bytes(image)):
                    return Function(image, f"func_{v:08X}", v, size, v + img.shift, True)
            inside = [r for r in rows if r["vram"] <= v < r["vram"] + r["size"]]
            raise ToolError(f"no {image} function starts at real 0x{v:08X}" +
                            (f" (inside {inside[0]['name']} at real 0x{inside[0]['vram']:08X}; give --size for a "
                             "true start)" if inside else ""))
        hit = [r for r in rows if r["name"] == k]
        if hit:
            r = hit[0]
            return Function(image, r["name"], r["vram"], size or r["size"], r["cat_vram"], size is not None)
        raise ToolError(f"no {image} function named {k}")

    # ------------------------------------------------------------------ manifests and recipes
    @staticmethod
    def _guess_pipeline(source: str, flags: str) -> str:
        if source.endswith(".s"):
            return "as"
        if "-non_shared" in flags or "-Wab," in flags:
            return "ido5.3"
        return "kmc+builtin"

    def manifest_rows(self, image: str) -> list[dict]:
        if image in self._manifest:
            return self._manifest[image]
        img, out = self.image(image), []
        for path, pipe in img.manifests:
            if not path.is_file():
                continue
            for line in path.read_text(errors="replace").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                f = line.split("\t")
                try:
                    start, size = int(f[0], 0), int(f[1], 0)
                except (ValueError, IndexError):
                    continue
                flags = f[3] if len(f) > 4 else ""
                out.append(dict(vram=start + img.delta, size=size, source=f[2], flags=flags,
                                pipe=pipe or self._guess_pipeline(f[2], flags),
                                manifest=str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)))
        self._manifest[image] = out
        return out

    def unit_for(self, image: str, vram: int, size: int) -> dict | None:
        for row in self.manifest_rows(image):
            if row["vram"] < vram + size and vram < row["vram"] + row["size"]:
                return row
        return None

    def default_recipe(self, image: str, vram: int) -> tuple[str, str]:
        for region in self.agent.get("regions") or []:
            if region.get("image", image) == image and _num(region.get("start", 0)) <= vram < \
                    _num(region.get("end", 0xFFFFFFFF)):
                return region.get("pipeline", self.run_pipeline), region.get("flags", self.run_flags)
        img = self.image(image)
        return img.pipeline or self.run_pipeline, img.flags or self.run_flags

    def spec_for(self, fn: Function, flags: str | None = None, pipe: str | None = None):
        from .pipeline import split_flags
        row = self.unit_for(fn.image, fn.vram, fn.size)
        if fn.true_range and row is not None and (row["vram"] != fn.vram or row["size"] != fn.size):
            row = None              # a repaired range: a piece's unit recipe does not apply
        if flags is None and row is not None and row["source"].endswith(".c"):
            flags, base_pipe = row["flags"], row["pipe"]
        else:
            base_pipe, dflags = self.default_recipe(fn.image, fn.vram)
            flags = flags if flags is not None else dflags
        spec = split_flags(shlex.split(flags), base_pipe if not pipe else pipe)
        if pipe:
            spec.pipeline = pipe
        return spec, flags, row

    def lanes(self, fn: Function, flags: str | None = None, pipe: str | None = None) -> list:
        """Recipes to build with: overrides; else the unit's own; else the default plus its
        sibling assembler lanes (``[agent] sibling_lanes``)."""
        spec, fl, row = self.spec_for(fn, flags, pipe)
        out = [(spec, fl)]
        siblings = self.agent.get("sibling_lanes", DEFAULT_SIBLINGS)
        if not pipe and (row is None or flags is not None) and "--assembler=" not in fl:
            for lane in siblings.get(spec.pipeline, []):
                s2, f2, _ = self.spec_for(fn, fl, lane)
                out.append((s2, f2))
        return out

    # ------------------------------------------------------------------ declared symbols
    def _stamp(self, image: str) -> str:
        h = hashlib.sha1()
        img = self.image(image)
        for path, _ in img.manifests:
            if path.is_file():
                h.update(path.read_bytes())
        for path in img.symbols:
            if path.is_file():
                h.update(path.read_bytes())
        return h.hexdigest()[:16]

    def declared(self, image: str):
        """The image's symbol files plus every symbol the project's own manifest units define
        (built once and cached in the state folder)."""
        if image in self._declared:
            return self._declared[image]
        from . import symbols as symmod
        from . import verify as verifymod
        from .pipeline import split_flags
        img = self.image(image)
        syms = symmod.load([p for p in img.symbols if p.is_file()], symmod.SymbolMap(infer_names=False))
        self.state.mkdir(parents=True, exist_ok=True)
        cache = self.state / f"defines_{img.name}.json"
        stamp = self._stamp(image)
        defines = None
        if cache.is_file():
            data = json.loads(cache.read_text())
            if data.get("stamp") == stamp:
                defines = data["defines"]
        if defines is None:
            defines, failed = {}, []
            with tempfile.TemporaryDirectory(prefix="n64dk-agent-decl-") as tmp:
                for i, row in enumerate(self.manifest_rows(image)):
                    source = self.root / row["source"]
                    try:
                        spec = split_flags(shlex.split(row["flags"]), row["pipe"])
                        if row["pipe"] == "as" and not row["flags"]:
                            spec.pipeline = "as"
                        elf = self.build(source, spec, Path(tmp) / f"u{i}")
                    except Exception as error:  # noqa: BLE001 - recorded, not fatal
                        failed.append(f"{row['source']}: {str(error)[-160:]}")
                        continue
                    for name, addr in verifymod.unit_defines(elf, row["vram"], spec.section_bases).items():
                        defines.setdefault(name, addr)
            cache.write_text(json.dumps({"stamp": stamp, "defines": defines, "failed": failed}, indent=0))
        for name, addr in defines.items():
            if name not in syms:
                syms[name] = addr
        self._declared[image] = syms
        return syms

    def names_for(self, image: str) -> dict[int, str]:
        out: dict[int, str] = {}
        for n, a in self.declared(image).items():
            if a not in out or (ADDR_NAME.fullmatch(out[a]) and not ADDR_NAME.fullmatch(n)):
                out[a] = n
        return out

    def function_starts(self, image: str) -> set[int]:
        return {r["vram"] for r in self.catalogue(image)} | {r["vram"] for r in self.manifest_rows(image)}

    # ------------------------------------------------------------------ build and compare
    def build(self, path: Path, spec, workdir: Path):
        from ..elf import Elf
        from .pipeline import build
        return Elf.load(build(Path(path), spec, self.cfg, workdir))

    def strict(self, fn: Function, elf, spec) -> dict:
        """The kit's ``strict_compare`` at the real address with declared information only (see
        the module docstring).  A data section without a declared base is placed where the retail
        text implies and then compared byte for byte there (reported as the base to record)."""
        from . import compare as cmp
        from . import verify as verifymod
        img = self.image(fn.image)
        rom = self.image_bytes(fn.image)
        v2r = lambda v: v - img.delta  # noqa: E731
        syms = self.declared(fn.image).copy()
        by_addr: dict[int, set[str]] = {}
        for n, a in syms.items():
            by_addr.setdefault(a, set()).add(n)
        spelled = {}
        for s in elf.symbols:
            name = s.name
            if not (name and s.section == 0 and s.bind == 1):
                continue
            m = ADDR_NAME.fullmatch(name)
            if m and name not in syms and not by_addr.get(int(m.group(1), 16)):
                spelled[name] = int(m.group(1), 16)
        syms.update(spelled)
        syms.update(spec.symbols)
        bases = dict(spec.section_bases)
        starts = self.function_starts(fn.image)
        v = verifymod.strict_compare(elf, rom, v2r, fn.vram, fn.size, syms, bases, None, starts)
        placed = {}
        sec_assumed = [a for a in v.assumptions if a.kind == "section"]
        if sec_assumed and not v.conflicts:
            sizes = {}
            for a in sec_assumed:
                bases[a.name] = placed[a.name] = a.address
                s = cmp.find_section(elf, a.name)
                if s is not None:
                    sizes[a.name] = s.size
            v = verifymod.strict_compare(elf, rom, v2r, fn.vram, fn.size, syms, bases, sizes, starts)
        return dict(verdict=v.verdict, detail=v.as_dict(), placed=placed, spelled=spelled, bases=bases, syms=syms)

    def explore(self, fn: Function, elf, spec) -> bytes:
        """Exploration compare (retail-inferred addresses allowed): the relocated text only."""
        from . import compare as cmp
        img = self.image(fn.image)
        syms = self.declared(fn.image).copy()
        syms.infer_names = True
        syms.update(spec.symbols)
        res = cmp.compare_object(elf, self.image_bytes(fn.image), lambda v: v - img.delta, fn.vram, fn.size, syms,
                                 spec.section_bases, infer=True, infer_externals=True, with_diff=False,
                                 compare_sections=False)
        merged = syms.copy()
        merged.update(res.inferred_symbols)
        return cmp.relocate(elf, ".text", res.bases, merged)

    def relocated_text(self, elf, vram: int, bases: dict, syms) -> bytes:
        from . import compare as cmp
        have = cmp.complete_bases(elf, ".text", {**bases, ".text": vram}, syms)
        return elf.relocate(cmp.find_section(elf, ".text").name, have, syms)


# ---------------------------------------------------------------------- disassembly

def disasm(code: bytes, vram: int, names: dict[int, str] | None = None) -> list[str]:
    import rabbitizer
    names = names or {}
    out, hi = [], {}
    for i in range(0, len(code) - len(code) % 4, 4):
        w = struct.unpack(">I", code[i:i + 4])[0]
        ins = rabbitizer.Instruction(w, vram + i)
        text = re.sub(r"\s+", " ", ins.disassemble())
        note, op = "", ins.getOpcodeName()
        if ins.isJumpWithAddress():
            t = ins.getInstrIndexAsVram()
            note = names.get(t, f"func_{t:08X}" if ins.doesLink() else "")
        elif op == "lui":
            hi[ins.rt] = ins.getProcessedImmediate() << 16
        elif op in LOWOPS and ins.rs in hi:
            imm = ins.getProcessedImmediate()
            a = (hi[ins.rs] + (imm if op != "ori" else imm & 0xFFFF)) & 0xFFFFFFFF
            nm = names.get(a)
            if nm is None:
                near = [k for k in names if k <= a < k + 0x40 and not names[k].startswith("func_")]
                nm = f"{names[max(near)]}+0x{a - max(near):X}" if near else f"D_{a:08X}"
            note = f"{nm} (0x{a:08X})"
        out.append(f"{i:5X} {vram + i:08X}: {text}" + (f"    ; {note}" if note else ""))
    return out


def masked_hash(code: bytes) -> str:
    """Content hash with relocated fields masked (for twins across images)."""
    import rabbitizer
    words, luis = [], set()
    for i in range(0, len(code) - len(code) % 4, 4):
        w = struct.unpack(">I", code[i:i + 4])[0]
        ins = rabbitizer.Instruction(w, 0x80000000 + i)
        op = ins.getOpcodeName()
        if ins.isJumpWithAddress():
            w &= 0xFC000000
        elif op == "lui":
            w &= 0xFFFF0000
            luis.add(ins.rt)
        elif op in LOWOPS and ins.rs in luis:
            w &= 0xFFFF0000
        words.append(w)
    return hashlib.sha1(struct.pack(f">{len(words)}I", *words)).hexdigest()[:16]


def _ops(code: bytes, vram: int) -> list[str]:
    return [re.sub(r"^\s*\w+ \w+: ", "", x) for x in disasm(code, vram)]


# ---------------------------------------------------------------------- source checks

ASM = re.compile(r"\b(__asm__|asm)\b")
LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def source_problems(path: Path) -> tuple[list[str], list[str]]:
    """(rejections, assisted classes) of a candidate file.  Draft scaffolding is rejected; inline
    assembly in the file is rejected; inline assembly reached through a header of the file's own
    folder (a macro such as a fixed-point multiply) makes the result *assisted*: it may verify,
    but it is never credited as plain C."""
    text = Path(path).read_text(errors="replace")
    rejected, assisted = [], []
    if "M2C_" in text:
        rejected.append("contains M2C_ names (draft scaffolding)")
    if ASM.search(re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)):
        rejected.append("contains asm: not C credit (assembly-assisted)")
    for header in LOCAL_INCLUDE.findall(text):
        candidate = Path(path).parent / header
        if candidate.is_file() and ASM.search(candidate.read_text(errors="replace")):
            assisted.append(f"assisted: {header} holds inline assembly (macro class, never plain C)")
    return rejected, assisted


# ---------------------------------------------------------------------- the tools

def _header(ws: Workspace, fn: Function) -> str:
    row = ws.unit_for(fn.image, fn.vram, fn.size)
    st = f"unit {row['source']} ({Path(row['manifest']).name})" if row else "no unit (not matched)"
    off = fn.vram - ws.image(fn.image).delta
    return (f"== {fn.image} {fn.name} real 0x{fn.vram:08X} size 0x{fn.size:X} ({fn.size}) image offset 0x{off:X}; "
            f"{st}" + (" [true range given]" if fn.true_range else ""))


def _label(spec, flags) -> str:
    return f"{spec.pipeline} {flags}"


def recipe_line(spec, flags: str, placed: dict) -> str:
    """The manifest FLAGS column a verified source is recorded with."""
    words = [w for w in flags.split() if not w.startswith("--section-base=")]
    if spec.pipeline == "kmc+asn64" and "--assembler=asn64" not in words:
        words.insert(0, "--assembler=asn64")
    for sec, base in sorted({**spec.section_bases, **placed}.items()):
        words.insert(0, f"--section-base={sec}:0x{base:08X}")
    return " ".join(words)


def tool_info(ws, fn, files, opts) -> dict:
    lines = [_header(ws, fn)]
    lanes = ws.lanes(fn, opts.get("flags"), opts.get("pipe"))
    for spec, fl in lanes:
        lines.append(f"recipe: {_label(spec, fl)}")
    row = ws.unit_for(fn.image, fn.vram, fn.size)
    if row:
        lines.append(f"source: {ws.root / row['source']}")
    return dict(lines=lines, lanes=[_label(s, f) for s, f in lanes])


def tool_dis(ws, fn, files, opts) -> dict:
    code = ws.target(fn)
    lines = [_header(ws, fn)] + disasm(code, fn.vram, ws.names_for(fn.image))
    lines.append(f"masked hash {masked_hash(code)}")
    return dict(lines=lines)


def tool_draft(ws, fn, files, opts) -> dict:
    from . import draftforms, m2c_bridge
    if not files:
        raise ToolError("draft needs an output file")
    out = Path(files[0])
    img = ws.image(fn.image)
    code = ws.target(fn)
    names = ws.names_for(fn.image)
    name = fn.name if not ADDR_NAME.fullmatch(fn.name) else f"func_{fn.vram:08X}"
    jt = m2c_bridge.find_jump_tables(ws.image_bytes(fn.image), img.delta, fn.vram, fn.size)
    spec, _, _ = ws.spec_for(fn, opts.get("flags"), opts.get("pipe"))
    compiler = "ido" if spec.pipeline.startswith("ido") else "kmc"
    r = m2c_bridge.draft(name, code, fn.vram, compiler, names={k: v for k, v in names.items() if k != fn.vram},
                         jump_tables=jt)
    lines = [_header(ws, fn)]
    if not r.get("source"):
        lines.append(f"draft failed: {r['status']} {(r.get('error') or '')[:300]}")
        return dict(lines=lines, ok=False, files=[])
    from .autodraft import prepare
    headers = ws.agent.get("draft_headers")          # the project's own m2c/types headers, when it has them
    render = (lambda text: "".join(f'#include "{h}"\n' for h in headers) + "\n" + text) if headers else prepare
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(r["source"]))
    written = [str(out)]
    lines.append(f"wrote {out} (m2c, {r['m2c_errors']} M2C_ERROR markers)")
    for i, (label, text, review) in enumerate(draftforms.drafts(r["source"], name)[:4], 1):
        p = out.with_name(out.stem + f".form{i}.c")
        p.write_text(render(text))
        written.append(str(p))
        lines.append(f"wrote {p} ({label}{', review' if review else ''})")
    return dict(lines=lines, ok=True, files=written)


def tool_try(ws, fn, files, opts) -> dict:
    """Build each file under each lane and report MATCH (text equal, addresses inferred) or the
    differing hunks.  Not an acceptance: run ``exact``."""
    show = int(opts.get("show", 14))
    want = ws.target(fn)
    lines, results = [_header(ws, fn)], []
    for path in files:
        name = Path(path).name
        best = None
        for spec, flags in ws.lanes(fn, opts.get("flags"), opts.get("pipe")):
            with tempfile.TemporaryDirectory(prefix="n64dk-try-") as tmp:
                try:
                    elf = ws.build(Path(path), spec, Path(tmp))
                except Exception as error:  # noqa: BLE001
                    lines.append(f"{name} [{spec.pipeline}] BUILD ERROR {str(error)[-900:]}")
                    best = best or dict(file=str(path), status="build-error", hunks=None, lane=_label(spec, flags))
                    break
                try:
                    got = ws.explore(fn, elf, spec)
                except Exception as error:  # noqa: BLE001
                    lines.append(f"{name} [{spec.pipeline}] COMPARE ERROR {str(error)[-300:]}")
                    continue
            if got[:fn.size] == want and not any(got[fn.size:]):
                lines.append(f"{name} MATCH (text, addresses inferred; now run exact) [{_label(spec, flags)}]")
                best = dict(file=str(path), status="match", hunks=0, lane=_label(spec, flags))
                break
            a = _ops(want, fn.vram)
            b = _ops(got.rstrip(b"\0") if len(got) > len(want) else got, fn.vram)
            ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
            lines.append(f"{name} DIFFERS compiled {len(got)} target {len(want)} hunks {len(ops)} (- target, + yours) "
                         f"[{_label(spec, flags)}]")
            for tag, i1, i2, j1, j2 in ops[:show]:
                lines.append(f"  @{4 * i1:X} - {'; '.join(a[i1:i2][:6])}\n        + {'; '.join(b[j1:j2][:6])}")
            cand = dict(file=str(path), status="differs", hunks=len(ops), lane=_label(spec, flags),
                        compiled=len(got), target=len(want))
            if best is None or best.get("hunks") is None or len(ops) < best["hunks"]:
                best = cand
        rejected, assisted = source_problems(Path(path))
        lines += [f"  note: {p}" for p in rejected + assisted]
        if best:
            results.append(best)
    return dict(lines=lines, results=results)


def tool_exact(ws, fn, files, opts) -> dict:
    """The only acceptance: the kit's strict verifier, plain C only."""
    n = int(opts.get("lines", 24))
    want = ws.target(fn)
    lines, results = [_header(ws, fn)], []
    for path in files:
        name = Path(path).name
        best = None
        for spec, flags in ws.lanes(fn, opts.get("flags"), opts.get("pipe")):
            with tempfile.TemporaryDirectory(prefix="n64dk-exact-") as tmp:
                try:
                    elf = ws.build(Path(path), spec, Path(tmp))
                except Exception as error:  # noqa: BLE001
                    lines.append(f"{name} [{spec.pipeline}] BUILD ERROR {str(error)[-900:]}")
                    break
                r = ws.strict(fn, elf, spec)
                r.update(spec=spec, flags=flags)
                if r["verdict"] == "exact":
                    best = r
                    break
                if best is None or (r["detail"]["text_equal"] and not best["detail"]["text_equal"]):
                    best = r
                    try:
                        best["text"] = ws.relocated_text(elf, fn.vram, r["bases"], r["syms"])
                    except Exception:  # noqa: BLE001
                        best["text"] = None
        if best is None:
            results.append(dict(file=str(path), verdict="build-error", accepted=False))
            continue
        rejected, assisted = source_problems(Path(path))
        d = best["detail"]
        ok = best["verdict"] == "exact" and not rejected and not assisted
        word = "EXACT" if ok else ("EXACT-ASSISTED" if best["verdict"] == "exact" and not rejected else "NOT EXACT")
        lines.append(f"{name} {word} (kit strict_compare verdict: {best['verdict']}; text_equal {d['text_equal']}) "
                     f"[{_label(best['spec'], best['flags'])}]")
        recipe = recipe_line(best["spec"], best["flags"], best["placed"])
        if best["verdict"] == "exact":
            lines.append(f"  recipe: {recipe}")
        if best["placed"]:
            lines.append("  data placed at the retail-implied address and compared: " +
                         ", ".join(f"{k} 0x{v:08X}" for k, v in best["placed"].items()))
        if best["spelled"]:
            lines.append(f"  address-named symbols declared from their names: {len(best['spelled'])}")
        lines += [f"  CONFLICT {c}" for c in d["conflicts"]]
        lines += [f"  ASSUMED {a['kind']} {a['name']} = {a['address']} (not declared anywhere: use the project's name "
                  "for that address or an address-spelled name)" for a in d["assumptions"]]
        lines += [f"  SECTION {json.dumps(s)}" for s in d["sections"] if s.get("status") != "equal"]
        lines += [f"  note: {note}" for note in d["notes"][:6]]
        lines += [f"  REJECTED: {p}" for p in rejected] + [f"  {p}" for p in assisted]
        text = best.get("text")
        if not d["text_equal"] and text is not None:
            a = disasm(want, fn.vram)
            b = disasm(text[:max(len(want), len(text.rstrip(b'\0')))], fn.vram)
            shown = 0
            for i in range(max(len(a), len(b))):
                x = a[i].split(": ", 1)[1] if i < len(a) else "-"
                y = b[i].split(": ", 1)[1] if i < len(b) else "-"
                if x != y:
                    lines.append(f"  {4 * i:4X}  rom: {x:<44} built: {y}")
                    shown += 1
                    if shown >= n:
                        break
        results.append(dict(file=str(path), verdict=best["verdict"], text_equal=d["text_equal"], accepted=ok,
                            assisted=bool(assisted and best["verdict"] == "exact" and not rejected),
                            recipe=recipe, pipeline=best["spec"].pipeline, rejected=rejected))
    return dict(lines=lines, results=results)


def tool_regdump(ws, fn, files, opts) -> dict:
    """GCC 2.x pass dumps (``.lreg`` local allocation, ``.greg`` global, ...) of a candidate under
    its recipe, with the allocator priorities ``gcc27dump`` computes."""
    from .. import safety
    from .pipeline import absolute_flags
    if not files:
        raise ToolError("regdump needs a C file")
    letters = opts.get("dumps") or "lg"
    spec, _, _ = ws.spec_for(fn, opts.get("flags"), opts.get("pipe"))
    if not spec.pipeline.startswith("kmc"):
        raise ToolError(f"regdump needs a GCC (kmc) recipe; this one is {spec.pipeline}")
    gcc_dir = ws.cfg.toolchain("kmc")
    if gcc_dir is None:
        raise ToolError("the kmc toolchain is not configured")
    flags = [f for f in absolute_flags(list(spec.flags), ws.root) if not f.startswith("-Wa,")]
    flags += [f"-I{p}" for p in ws.cfg.include]
    lines = [_header(ws, fn)]
    out_dir = ws.state / "regdump"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(files[0]).resolve()
    with tempfile.TemporaryDirectory(prefix="n64dk-regdump-", dir=out_dir) as tmp:
        unit = Path(tmp) / src.name
        unit.write_text(src.read_text(errors="replace"))
        argv = [str(gcc_dir / "gcc"), f"-B{gcc_dir}/", "-x", "c", "-S", *flags, f"-d{letters}", unit.name, "-o",
                unit.with_suffix(".s").name]
        proc = safety.run_tool(argv, kind="compile", cwd=tmp, text=True, timeout=120)   # the kit's scheduler
        if proc.returncode:
            lines.append(f"compile failed: {(proc.stderr or proc.stdout)[-900:]}")
            return dict(lines=lines, dumps={})
        dumps = {p.name: p.read_text(errors="replace") for p in sorted(Path(tmp).glob(unit.name + ".*"))}
    for name, text in dumps.items():
        lines.append(f"==== {name}")
        lines += text.splitlines()
    if any(n.endswith(".lreg") for n in dumps):
        from . import gcc27dump
        local = next(t for n, t in dumps.items() if n.endswith(".lreg"))
        glob = next((t for n, t in dumps.items() if n.endswith(".greg")), "")
        try:
            lines.append("==== priorities (floor_log2(refs) * refs / live_length; ties to the lower pseudo)")
            for p in gcc27dump.parse(local, glob)[:40]:
                lines.append(f"  pseudo {p.number}: refs {p.references} span {p.live_span} priority {p.priority} "
                             f"-> hard {p.hard_register}")
        except Exception as error:  # noqa: BLE001
            lines.append(f"  (priorities unavailable: {error})")
    return dict(lines=lines, dumps=list(dumps))


TYPEDEF = re.compile(r"typedef\s+(?:struct|union)\s*\w*\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*(\w+)\s*;", re.S)


def tool_neighbours(ws, fn, files, opts) -> dict:
    """Matched sources nearest the function in its image (with their recipes), the covering unit,
    struct definitions those sources already worked out, and declarations elsewhere in the
    project of the symbols the target uses."""
    count = int(opts.get("count", 4))
    rows = [r for r in ws.manifest_rows(fn.image) if r["source"].endswith(".c")]
    rows.sort(key=lambda r: abs(r["vram"] - fn.vram))
    lines = [_header(ws, fn)]
    picked = rows[:count]
    lines.append(f"## nearest matched sources ({len(picked)} of {len(rows)} C units in {fn.image})")
    structs: dict[str, str] = {}
    for r in sorted(picked, key=lambda r: r["vram"]):
        lines.append(f"- 0x{r['vram']:08X} size 0x{r['size']:X} {ws.root / r['source']}  [{r['pipe']} {r['flags']}]")
        path = ws.root / r["source"]
        if path.is_file():
            for m in TYPEDEF.finditer(path.read_text(errors="replace")):
                structs.setdefault(m.group(1), m.group(0))
    names = set()
    for line in disasm(ws.target(fn), fn.vram, ws.names_for(fn.image)):
        if ";" in line:
            note = line.split(";", 1)[1].strip()
            names.add(re.split(r"[ +]", note)[0])
    declared = {}
    if names:
        pattern = re.compile(r"^[^\n;{}]*\b(" + "|".join(map(re.escape, sorted(names))) + r")\b[^\n;{}]*;", re.M)
        for image in {fn.image} | set(ws.images):
            for r in ws.manifest_rows(image):
                if len(declared) >= 40:
                    break
                path = ws.root / r["source"]
                if not path.is_file() or not r["source"].endswith(".c"):
                    continue
                for m in pattern.finditer(path.read_text(errors="replace")):
                    if m.group(0).lstrip().startswith(("extern", "typedef")) or "(" in m.group(0):
                        declared.setdefault(m.group(1), (m.group(0).strip(), r["source"]))
    if declared:
        lines.append("## declarations of the target's symbols found in the project's matched sources")
        for name in sorted(declared):
            decl, where = declared[name]
            lines.append(f"- {decl}    // {where}")
    if structs:
        lines.append(f"## struct definitions in those sources ({len(structs)})")
        for name in sorted(structs)[:12]:
            lines.append(structs[name])
    headers = []
    for inc in ws.cfg.include + [ws.root / "include"]:
        if Path(inc).is_dir():
            headers += sorted(str(p) for p in Path(inc).glob("*.h"))[:20]
    if headers:
        lines.append("## project headers: " + ", ".join(dict.fromkeys(headers)))
    return dict(lines=lines, neighbours=[r["source"] for r in picked], structs=sorted(structs),
                declarations=sorted(declared))


JR_RA = 0x03E00008


def _sp_adjust(word: int) -> int | None:
    """N for ``addiu $sp,$sp,N`` (0x27BD....), else None."""
    if word >> 16 == 0x27BD:
        imm = word & 0xFFFF
        return imm - 0x10000 if imm & 0x8000 else imm
    return None


def find_boundary(code: bytes, base: int, vram: int, catalogue: list[dict]) -> dict:
    """The true start and size of the function containing ``vram`` in ``code`` (first byte at
    ``base``), from its own instructions: the start is the nearest stack-allocating prologue at or
    before ``vram`` with no return between; the end is the instruction after the ``jr $ra`` that
    follows the matching stack release (a leaf: the first ``jr $ra``).  Evidence is listed."""
    words = [struct.unpack(">I", code[i:i + 4])[0] for i in range(0, len(code) - len(code) % 4, 4)]
    at = (vram - base) // 4
    if not 0 <= at < len(words):
        raise ToolError("address outside the image")
    evidence = []
    start, frame = at, None
    i = at
    while i >= 0 and at - i < 4096:
        n = _sp_adjust(words[i])
        if n is not None and n < 0:
            start, frame = i, -n
            break
        if words[i] == JR_RA and i < at - 1:
            start = i + 2
            evidence.append(f"a return at 0x{base + 4 * i:08X} before the address with no prologue between: "
                            "a leaf starting after it")
            break
        i -= 1
    if frame:
        evidence.append(f"prologue addiu $sp,$sp,-0x{frame:X} at 0x{base + 4 * start:08X}")
    end = None
    released = frame is None
    for j in range(start, min(len(words) - 1, start + 16384)):
        n = _sp_adjust(words[j])
        if frame and n == frame:
            released = True
        if words[j] == JR_RA and (released or (j + 1 < len(words) and _sp_adjust(words[j + 1]) == frame)):
            end = j + 2
            evidence.append(f"return jr $ra at 0x{base + 4 * j:08X} after the stack release")
            break
    if end is None:
        raise ToolError("no return found")
    true_start, size = base + 4 * start, 4 * (end - start)
    covered = [r for r in catalogue if r["vram"] < true_start + size and true_start < r["vram"] + r["size"]]
    return dict(start=true_start, size=size, frame=frame or 0, evidence=evidence,
                catalogue=[dict(name=r["name"], vram=r["vram"], size=r["size"]) for r in covered])


def tool_boundary(ws, fn, files, opts) -> dict:
    img = ws.image(fn.image)
    data = ws.image_bytes(fn.image)
    lo = max(0, fn.vram - img.delta - 0x4000)
    hi = min(len(data), fn.vram - img.delta + fn.size + 0x10000)
    found = find_boundary(data[lo:hi], img.delta + lo, fn.vram, ws.catalogue(fn.image))
    lines = [_header(ws, fn), f"true function: start 0x{found['start']:08X} size 0x{found['size']:X}"]
    lines += [f"  evidence: {e}" for e in found["evidence"]]
    for r in found["catalogue"]:
        lines.append(f"  catalogue row {r['name']} real 0x{r['vram']:08X} size 0x{r['size']:X}")
    same = found["start"] == fn.vram and found["size"] == fn.size
    lines.append("catalogue boundary agrees" if same else
                 f"catalogue boundary differs: verify the whole function with --size 0x{found['size']:X} at "
                 f"0x{found['start']:08X} (name it func_{found['start']:08X}); never claim a piece on its own")
    from . import verify as verifymod
    outside = verifymod.jump_targets_outside(ws.target(fn), fn.vram)
    if outside:
        lines.append("  jumps leaving the catalogued range: " + ", ".join(f"0x{t:08X}" for _, t in outside[:6]))
    return dict(lines=lines, agrees=same, **{k: found[k] for k in ("start", "size", "frame")})


DISPATCH = {"dis": tool_dis, "draft": tool_draft, "try": tool_try, "exact": tool_exact, "info": tool_info,
            "regdump": tool_regdump, "neighbours": tool_neighbours, "boundary": tool_boundary}


def run_tool(ws: Workspace, tool: str, image: str, key: str, files=(), *, size: int | None = None,
             **opts) -> dict:
    if tool not in DISPATCH:
        raise ToolError(f"unknown tool {tool!r}; tools: {', '.join(TOOLS)}")
    fn = ws.lookup(image, key, size)
    opts = {k: v for k, v in opts.items() if v is not None}
    return DISPATCH[tool](ws, fn, [str(f) for f in files], opts)


# ---------------------------------------------------------------------- sessions (the agent's view)

SESSION = "session.json"


def session_call(session_path: Path, tool: str, files=(), **opts) -> dict:
    """A tool call inside a model-stage work folder: the function, project and budget come from
    the session file; every call is logged (``calls.jsonl``) with its verdict lines, and calls
    past the budget are refused."""
    session = json.loads(Path(session_path).read_text())
    folder = Path(session_path).parent
    log = folder / "calls.jsonl"
    used = len(log.read_text().splitlines()) if log.is_file() else 0
    limit = int(session.get("tool_calls", 0))
    if limit and used >= limit and tool not in ("info",):
        return dict(lines=[f"BUDGET: {used} tool calls used of {limit}; write final.c and LESSON.md now"],
                    refused=True)
    ws = Workspace(session["project"], session.get("state"))
    size = opts.pop("size", None)
    if size is None and session.get("true_range"):
        size = session["size"]
    files = [str((Path.cwd() / f).resolve()) if not Path(f).is_absolute() else str(f) for f in files]
    try:
        result = run_tool(ws, tool, session["image"], f"{session['vram']:08X}" if size else session["key"], files,
                          size=size, **opts)
    except ToolError as error:
        result = dict(lines=[f"ERROR: {error}"], error=str(error))
    entry = {"tool": tool, "files": files, "opts": {k: v for k, v in opts.items() if v is not None}, "size": size,
             "summary": [line for line in result.get("lines", [])[1:]
                         if re.search(r"\b(MATCH|DIFFERS|EXACT|NOT EXACT|BUILD ERROR|ERROR|wrote|true function)", line)][:8]}
    for f in files:
        if Path(f).is_file() and f.endswith(".c"):
            entry.setdefault("sources", {})[Path(f).name] = hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]
    with log.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    return result
