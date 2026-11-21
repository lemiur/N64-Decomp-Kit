"""Proof-carrying receipts: one per settled result, replayable on another machine.

A settled result (a function the strict verifier called exact, a hand-written routine kept as
assembly, a byte-identical function that waits for a symbol) is a claim about a source, a recipe
and an image.  A receipt writes down everything that claim rests on, as digests and declared
values only, so that anyone holding the same image, the same toolchain and the source can check it
without trusting the run that made it:

* the subject: image name, function, address, size, catalogue size;
* the image identity (SHA-256 and the library's ``rom_key``; never a byte of it), the digest of the
  owned code bytes and of every owned data section the comparison read;
* the source digest, and the digest of every header the build reads (the pipeline records no
  preprocessor dependency output, so the closure is found with the pipeline's own include
  resolution: quoted includes beside the including file, then the include roots);
* the recipe: pipeline, compiler flags, declared section bases and symbol pins, normalizer
  arguments; the toolchain identity (``pipeline.toolchain_digest`` and, for catalogued lanes, the
  stage-by-stage component digests of ``components.recipe_identity``); the kit's own wrapper stages
  (the normaliser or converter module a lane runs, a project normaliser hook, the assembler flags);
  the environment variables that can change what a compiler emits (their values hashed);
* the declarations: the digest of the declared symbol map and of each symbol file (advisory), and
  the relocation inputs that bind the verdict (every relocation site of ``.text`` and of the
  compared data sections, and the declared address of every symbol and section base they use);
* the emitted object (the digest of its loaded sections and relocation tables; the whole file's
  digest is advisory: debug records may carry a build directory), the digest of the relocated bytes
  the comparison read, the verifier version, the strict verdict with its assumptions, the verdict
  class, source provenance and ``proof.evaluate``'s proof-grade policy;
* separate claims: byte identity, strict proof, assumptions present, behavioural evidence,
  source provenance, ownership completeness, proof grade.  Each claim is a function of the recorded
  evidence; a receipt whose claims say more than its evidence is rejected without a build.

``identity`` is the SHA-256 of the receipt without ``identity`` and ``issued`` (the timestamp):
the same inputs give the same identity on any machine.  It is a content address, not a
signature: anyone can write a receipt and hash it.  Trust comes from :func:`replay`, which
recomputes every digest from the files on disk, the image supplied locally and the local toolchain,
rebuilds the source with the recorded recipe in a fresh directory, runs ``verify.strict_compare``
again and fails with a precise reason when anything the claim rests on has changed.

Nothing here grants credit or changes a verdict: a receipt records what the strict verifier said,
and replay asks the strict verifier again.  Behavioural evidence (an emulator or oracle run) may be
attached to a receipt; it is reported as such and never yields a strict proof.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from ..elf import GP_RELATIVE, GP_SYMBOL, Elf
from . import compare as cmp
from . import pipeline as pipemod
from . import proof as proofmod
from . import verify as verifymod
from .fingerprint import canonical_digest, sha256_file
from .library import rom_key
from .pipeline import BuildSpec
from .provenance import ProvenanceResult, Verdict as ProvenanceVerdict, classify_source
from .symbols import SymbolMap

SCHEMA = "n64dk-proof-receipt/1"
# exact: C, strict verdict exact.  assumed: byte-identical, rests on retail-derived addresses or
# uncompared data.  assembly: an assembly source that assembles to the bytes.  raw: an assembly
# listing of raw words (it reproduces the bytes and is not source).  behavioural: the bytes differ
# and only behavioural evidence is attached (never matched, never proof).
CLASSES = ("exact", "assumed", "assembly", "raw", "behavioural")
# who earned the result: the run's own C (resolve), assembly, or a donor unit, by donor kind
EARNED_BY = ("generated-c", "assembly", "source-tree-donor", "sdk-donor", "cross-title-donor")
# pin origins that never support a strict proof: the address came from the bytes under comparison,
# or nobody recorded where it came from
DEPENDENT_ORIGINS = ("self-derived", "unrecorded")
# the only origins that support one: the run's own declarations, the declared map, the recipe the
# result is offered with, and the donor stage's independent derivations (transplant.classify_pin,
# corroborated_sections; section-chain: a section whose address its neighbours settle,
# transplant.settle_sections, recomputed from the run's units when its receipt is replayed).
# Anything else (a new label, a typo, an empty string) supports nothing.
INDEPENDENT_ORIGINS = ("run-declaration", "declared", "recipe", "definition", "fixed-address", "transfer",
                       "section-content", "section-chain")
ASSEMBLY_LANE = "gnu-as"                    # n64dk.match.handasm.RECIPE's tool
RAW_MARK = "raw words"                      # n64dk.match.handasm.as_words's header
# environment variables a compiler, preprocessor or emulation layer reads (values are hashed)
BUILD_ENVIRONMENT = ("GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH",
                     "DEPENDENCIES_OUTPUT", "SUNPRO_DEPENDENCIES", "SOURCE_DATE_EPOCH", "LC_ALL", "LANG",
                     "TZ", "QEMU_LD_PREFIX")
# the kit module that implements a lane's own transformation stage
LANE_CODE = {"kmc+builtin": ("normalize",), "kmc+hook": ("normalize",), "kmc+atload": ("normalize",),
             "kmc+atmacro": ("normalize",), "kmc+asn64": ("asn64",)}
_PATH_OPTIONS = ("-I", "-isystem", "-iquote", "-idirafter", "-include", "-imacros")
_MACHINE = re.compile(r"(?:^|[\s=:,'\"(])(?:-[A-Za-z]+)?(?:/(?:home|Users|mnt|tmp|root|var|opt|usr|private|srv|"
                      r"media|run|etc|Volumes)/|[A-Za-z]:[\\/])|^/|\\\\Users\\\\")


class ReceiptError(ValueError):
    pass


def _source_text(path: str | Path) -> bytes:
    """A module's source text with line endings normalised (a checkout's line endings are not code)."""
    return Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").encode("utf-8")


def _verifier_version() -> str:
    digest = hashlib.sha256()
    for module in (verifymod, cmp):
        digest.update(_source_text(module.__file__))
        digest.update(b"\0")
    return digest.hexdigest()


# Any change to the strict verifier or the comparison it relies on invalidates every receipt.
VERIFIER_VERSION = _verifier_version()


@functools.lru_cache(maxsize=None)
def _module_digest(name: str) -> str:
    import importlib
    module = importlib.import_module(f"{__package__}.{name}")
    return hashlib.sha256(_source_text(module.__file__)).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:08X}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------------------------
# Inputs


def rom_identity(data: bytes) -> dict:
    """Identity of the image the comparison reads: digests and size, never its bytes."""
    return {"sha256": _sha(data), "rom_key": rom_key(data), "size": len(data)}


def _portable_path(value: str, project: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(Path(project).resolve()).as_posix()
    except ValueError:
        raise ReceiptError(f"a flag names a path outside the project ({value}): the recipe is not "
                           "portable; give it relative to the project") from None


def portable_flags(flags: list[str], project: Path) -> list[str]:
    """Compiler flags with path-bearing options relative to the project (as ``pipeline.build``
    resolves them); a path outside the project cannot be replayed elsewhere and is refused."""
    out, index = [], 0
    while index < len(flags):
        flag = flags[index]
        if flag in _PATH_OPTIONS and index + 1 < len(flags):
            out += [flag, _portable_path(flags[index + 1], project)]
            index += 2
            continue
        for option in _PATH_OPTIONS:
            if flag.startswith(option) and len(flag) > len(option) and flag != "-I-":
                out.append(option + _portable_path(flag[len(option):], project))
                break
        else:
            out.append(flag)
        index += 1
    return out


def include_roots(project: Path, configured, spec: BuildSpec, source: Path) -> list[Path]:
    """The include search roots ``pipeline.build`` gives the compiler: the configured include
    directories, the ``-I``-style flags (relative to the project) and the source's directory."""
    project = Path(project).resolve()

    def absolute(value) -> Path:
        path = Path(value)
        return (project / path).resolve() if not path.is_absolute() else path.resolve()

    roots = [absolute(value) for value in configured or ()]
    options = ("-I", "-isystem", "-iquote", "-idirafter")
    index = 0
    while index < len(spec.flags):
        flag, value = spec.flags[index], None
        if flag in options and index + 1 < len(spec.flags):
            value = spec.flags[index + 1]
            index += 1
        else:
            for option in options:
                if flag.startswith(option) and len(flag) > len(option) and flag != "-I-":
                    value = flag[len(option):]
                    break
        if value:
            roots.append(absolute(value))
        index += 1
    roots.append(Path(source).parent.resolve())
    return list(dict.fromkeys(root for root in roots if root.is_dir()))


def forced_includes(spec: BuildSpec) -> list[str]:
    out = []
    for index, flag in enumerate(spec.flags):
        if flag == "-include" and index + 1 < len(spec.flags):
            out.append(spec.flags[index + 1])
        elif flag.startswith("-include") and len(flag) > len("-include"):
            out.append(flag[len("-include"):])
    return out


def _portable_name(path: Path, written: str, labels: list[tuple[str, Path | None]]) -> str:
    for label, root in labels:
        if root is None:
            continue
        try:
            return f"{label}/{path.relative_to(Path(root).resolve()).as_posix()}"
        except ValueError:
            continue
    return f"<outside>/{written}"


def header_closure(source: Path, roots: list[Path], forced: list[str],
                   labels: list[tuple[str, Path | None]]) -> dict:
    """``{"headers": {portable name: sha256}, "missing": [...]}`` for every header the source
    reaches.  Resolution is the pipeline's (``pipeline._include_closure_digest``): a quoted name
    beside the including file first, then each root; conditional includes are kept (conservative);
    an include that resolves nowhere, or a computed one, is recorded as missing so that adding the
    file later changes the receipt."""
    headers: dict[str, str] = {}
    missing: set[str] = set()
    visited: set[Path] = set()

    def resolve(name: str, quote: bool, parent: Path | None) -> Path | None:
        candidates = ([parent / name] if quote and parent is not None else []) + [root / name for root in roots]
        return next((item.resolve() for item in candidates if item.is_file()), None)

    def includes(data: bytes):
        macros = {found.group(1).decode(errors="surrogateescape"): (found.group(2), found.group(3))
                  for found in pipemod._STRING_DEFINE_RE.finditer(data)}
        for found in pipemod._INCLUDE_RE.finditer(data):
            yield found.group(1), found.group(2)
        for found in pipemod._COMPUTED_INCLUDE_RE.finditer(data):
            value = macros.get(found.group(1).decode(errors="surrogateescape"))
            if value:
                yield value
            else:
                missing.add("computed:" + found.group(1).decode(errors="replace"))

    def walk(data: bytes, parent: Path) -> None:
        for delimiter, raw in includes(data):
            name = raw.decode(errors="surrogateescape")
            child = resolve(name, delimiter == b'"', parent)
            if child is None:
                missing.add(name)
            else:
                visit(child, name)

    def visit(path: Path, written: str) -> None:
        if path in visited:
            return
        visited.add(path)
        data = path.read_bytes()
        headers[_portable_name(path, written, labels)] = _sha(data)
        walk(data, path.parent)

    source = Path(source)
    walk(source.read_bytes(), source.parent)
    for name in forced:
        child = resolve(name, True, source.parent)
        if child is None:
            missing.add("forced:" + name)
        else:
            visit(child, name)
    return {"headers": dict(sorted(headers.items())), "missing": sorted(missing)}


def default_toolchain_identity(cfg, spec: BuildSpec) -> dict:
    """Identity of the compiler, assembler and converter binaries a pipeline runs."""
    out = {"pipeline": spec.pipeline, "toolchain_digest": None, "recipe": None}
    if spec.pipeline == ASSEMBLY_LANE:
        located = cfg.gnu_as() if cfg is not None else None
        if located and Path(located[0]).is_file():
            out["toolchain_digest"] = sha256_file(Path(located[0]))
        return out
    if cfg is None:
        return out
    digest = pipemod.toolchain_digest(cfg, spec.pipeline)
    out["toolchain_digest"] = None if digest == _sha(b"") else digest      # nothing installed
    from . import components
    recipe = components.RECIPES.get(spec.pipeline)
    if recipe is not None:
        try:
            found = components.recipe_identity(recipe, getattr(cfg, "toolchain_root", None))
        except OSError:
            found = None
        if found is not None:
            out["recipe"] = {"lane": found["lane"], "identity": found["identity"],
                             "stages": [{"stage": stage["stage"], "component": stage["component"],
                                         "files": [list(item) for item in stage["files"]]}
                                        for stage in found["stages"]]}
    return out


def default_wrapper_identity(cfg, spec: BuildSpec) -> dict:
    """The kit's own stages around the tools: the module that rewrites a lane's assembly, a
    project normaliser hook, and the assembler flags the kit chooses."""
    out = {"lane_code": {name: _module_digest(name) for name in LANE_CODE.get(spec.pipeline, ())},
           "normalizer_hook": None, "assembler_flags": None}
    if spec.pipeline == "kmc+hook" and cfg is not None and getattr(cfg, "normalizer", None):
        hook = str(cfg.normalizer)
        found = next((path for path in (Path(hook), Path(cfg.project) / hook) if path.is_file()), None)
        out["normalizer_hook"] = sha256_file(found) if found else _sha(hook.encode())
    if spec.pipeline == "kmc+asn64":
        from . import asn64
        out["assembler_flags"] = list(asn64.GNU_AS_FLAGS)
    elif spec.pipeline.startswith("kmc") or spec.pipeline == "as":
        out["assembler_flags"] = list(pipemod._as_flags(list(spec.flags), spec.pipeline))
    elif spec.pipeline == ASSEMBLY_LANE:
        from . import handasm
        out["assembler_flags"] = list(handasm.ASSEMBLER_FLAGS)
    return out


def environment_identity(environment: Mapping[str, str]) -> dict:
    return {name: _sha(str(environment[name]).encode("utf-8", "surrogateescape"))
            for name in BUILD_ENVIRONMENT if name in environment}


def kit_driver_digest() -> str:
    """The kit's build driver (advisory: a change is reported; the rebuild decides)."""
    return _module_digest("pipeline")


def default_build(cfg, source: Path, spec: BuildSpec, workdir: Path) -> Elf:
    """Build with the kit's pipeline.  ``workdir`` is fresh, and the object cache keys on the
    source directory, so a replay compiles again rather than taking a cached object."""
    if cfg is None:
        raise ReceiptError("no toolchain configuration: pass a project (or a builder)")
    if spec.pipeline == ASSEMBLY_LANE:
        return _assemble(cfg, Path(source), Path(workdir))
    return Elf.load(pipemod.build(Path(source), spec, cfg, Path(workdir)))


def _assemble(cfg, source: Path, workdir: Path) -> Elf:
    from .. import safety
    from . import handasm
    located = cfg.gnu_as()
    if located is None:
        raise ReceiptError("no MIPS GNU assembler for the assembly lane")
    assembler, env = located
    workdir.mkdir(parents=True, exist_ok=True)
    obj = workdir / "unit.o"
    done = safety.run_tool([assembler, *handasm.ASSEMBLER_FLAGS, "-o", str(obj), str(source)], kind="light",
                           capture_output=True, text=True, timeout=60, env={**os.environ, **env})
    if done.returncode != 0 or not obj.is_file():
        raise ReceiptError("assembly failed: " + (done.stderr or "").replace(str(source), source.name)[-300:])
    return Elf.load(obj)


@dataclass
class Context:
    """What a receipt is issued or replayed against: one image of one project.

    ``symbols`` is the declared symbol map (symbol files only); ``base`` the run's own recipe
    declarations (section bases and symbol pins every unit gets); ``catalogue`` vram -> size.
    ``builder``, ``toolchain`` and ``wrapper`` replace the kit's build and identity functions
    (tests; a remote builder)."""
    project: Path
    image: str
    data: bytes
    delta: int                                        # vram - image offset
    symbols: Mapping[str, int] = field(default_factory=dict)
    symbol_files: list = field(default_factory=list)
    cfg: object | None = None
    base: BuildSpec | None = None
    catalogue: dict[int, int] | None = None
    function_starts: set[int] | None = None
    builder: Callable | None = None                   # (source, spec, workdir) -> Elf
    toolchain: Callable | None = None                 # (spec) -> dict
    wrapper: Callable | None = None                   # (spec) -> dict
    environment: Mapping[str, str] | None = None      # default: os.environ
    include: list | None = None                       # default: cfg.include
    toolchain_root: Path | None = None                # default: cfg.toolchain_root (header names)
    _cache: dict = field(default_factory=dict, repr=False)

    def vram_to_rom(self, vram: int) -> int:
        return vram - self.delta

    def build(self, source: Path, spec: BuildSpec, workdir: Path) -> Elf:
        if self.builder is not None:
            return self.builder(source, spec, workdir)
        return default_build(self.cfg, source, spec, workdir)

    def toolchain_identity(self, spec: BuildSpec) -> dict:
        key = ("toolchain", spec.pipeline)
        if key not in self._cache:
            self._cache[key] = (self.toolchain(spec) if self.toolchain is not None
                                else default_toolchain_identity(self.cfg, spec))
        return self._cache[key]

    def wrapper_identity(self, spec: BuildSpec) -> dict:
        return self.wrapper(spec) if self.wrapper is not None else default_wrapper_identity(self.cfg, spec)

    def image_identity(self) -> dict:
        if "image" not in self._cache:
            self._cache["image"] = rom_identity(self.data)
        return self._cache["image"]

    def declarations(self) -> dict:
        if "declarations" not in self._cache:
            files = []
            for path in self.symbol_files:
                path = Path(path)
                if path.is_file():
                    files.append({"name": _portable_name(path.resolve(), path.name,
                                                         [("<project>", Path(self.project))]).replace(
                                      "<outside>/", "<symbols>/"),
                                  "sha256": sha256_file(path)})
            self._cache["declarations"] = {
                "map_sha256": canonical_digest(sorted([name, value] for name, value in dict.items(self.symbols))),
                "files": files}
        return self._cache["declarations"]

    def headers(self, source: Path, spec: BuildSpec) -> dict:
        configured = self.include if self.include is not None else list(getattr(self.cfg, "include", None) or [])
        roots = include_roots(self.project, configured, spec, source)
        tools = self.toolchain_root or getattr(self.cfg, "toolchain_root", None)
        labels = [("<source>", Path(source).parent), ("<project>", Path(self.project)), ("<toolchain>", tools)]
        return header_closure(source, roots, forced_includes(spec), labels)


# ---------------------------------------------------------------------------------------------
# Evidence from an object


def _local_symbols(symbols: Mapping[str, int], spec: BuildSpec) -> SymbolMap:
    local = SymbolMap(dict(symbols), infer_names=False)
    local.update(spec.symbols)
    return local


def _data_sections(elf: Elf) -> list:
    out, seen = [], set()
    for name in cmp.COMPARED:
        section = cmp.find_section(elf, name)
        if section is not None and section.size and section.name not in seen:
            seen.add(section.name)
            out.append(section)
    return out


def relocation_inputs(elf: Elf, vram: int, spec: BuildSpec, symbols: Mapping[str, int]) -> dict:
    """Every relocation site of ``.text`` and of the compared data sections, and the declared value
    of every symbol and section base they use (None: not declared, so taken from the image)."""
    syms = _local_symbols(symbols, spec)
    have = cmp.complete_bases(elf, ".text", {**spec.section_bases, ".text": vram}, syms)
    text = cmp.find_section(elf, ".text")
    sites, names, bases = [], {}, {}
    for section in ([text] if text is not None else []) + _data_sections(elf):
        for item in cmp.relocations(elf, section.name):
            target = item.section if item.section is not None else item.symbol
            sites.append([section.name, item.offset, item.type, target])
            if item.section is None and item.symbol:
                names[item.symbol] = syms.resolve(item.symbol)
            elif item.section is not None:
                bases[item.section] = have.get(item.section)
            if item.type in GP_RELATIVE:
                names[GP_SYMBOL] = syms.resolve(GP_SYMBOL)
    return {"sites": len(sites), "sites_sha256": canonical_digest(sites),
            "symbols": {name: None if value is None else _hex(value) for name, value in sorted(names.items())},
            "section_bases": {name: None if value is None else _hex(value) for name, value in sorted(bases.items())}}


def object_digests(elf: Elf) -> dict:
    """``loaded_sha256`` covers what the image receives (every loaded section's bytes and its
    relocations); ``object_sha256`` the whole file (advisory)."""
    loaded = []
    for name in verifymod.loaded_sections(elf):
        section = elf.section(name)
        relocations = [[item.offset, item.type, item.section if item.section is not None else item.symbol]
                       for item in cmp.relocations(elf, name)]
        loaded.append([name, section.size, _sha(section.data), relocations])
    return {"object_sha256": _sha(elf.data), "loaded_sha256": canonical_digest(loaded)}


def comparison_digest(elf: Elf, vram: int, size: int, spec: BuildSpec, symbols: Mapping[str, int],
                      verdict: verifymod.Verdict) -> str | None:
    """SHA-256 of the relocated bytes the strict comparison read: ``.text`` up to the owned size and
    each data section compared equal, relocated as the verifier relocated them (assumed addresses
    included, so an ``assumed`` receipt records what it compared)."""
    syms = _local_symbols(symbols, spec)
    # as strict_compare: bases from declared information first, then the assumptions
    have = cmp.complete_bases(elf, ".text", {**spec.section_bases, ".text": vram}, syms)
    for item in verdict.assumptions:
        if item.kind == "section":
            have[item.name] = item.address
        elif item.kind == "symbol":
            syms[item.name] = item.address
    text = cmp.find_section(elf, ".text")
    if text is None:
        return None
    digest = hashlib.sha256()
    try:
        digest.update(b".text\0" + elf.relocate(text.name, have, syms)[:size])
        for row in verdict.sections:
            if row.get("status") != "equal":
                continue
            section = cmp.find_section(elf, row["section"])
            digest.update(b"\0" + row["section"].encode() + b"\0" +
                          elf.relocate(section.name, have, syms)[:row["compared"]])
    except (KeyError, ValueError):
        return None
    return digest.hexdigest()


def strict_block(verdict: verifymod.Verdict) -> dict:
    keep = ("section", "size", "status", "base", "compared")
    return {"verdict": verdict.verdict, "text_equal": verdict.text_equal,
            "assumptions": [item.as_dict() for item in verdict.assumptions],
            "conflicts": list(verdict.conflicts),
            "sections": [{key: row[key] for key in keep if key in row} for row in verdict.sections],
            "notes": list(verdict.notes)[:8]}


def _strict(ctx: Context, elf: Elf, spec: BuildSpec, vram: int, size: int, section_sizes) -> verifymod.Verdict:
    local = SymbolMap(dict(ctx.symbols))
    local.update(spec.symbols)
    sizes = {name: int(value) for name, value in (section_sizes or {}).items()} or None
    return verifymod.strict_compare(elf, ctx.data, ctx.vram_to_rom, vram, size, local, dict(spec.section_bases),
                                    sizes, ctx.function_starts)


def owned_bytes(ctx: Context, vram: int, size: int, strict: dict) -> dict:
    offset = ctx.vram_to_rom(vram)
    code = ctx.data[offset:offset + size] if offset >= 0 else b""
    out = {"code_sha256": _sha(code) if len(code) == size else None, "data": {}}
    for row in strict.get("sections") or []:
        if row.get("status") != "equal":
            continue
        at, count = ctx.vram_to_rom(int(row["base"], 16)), int(row["compared"])
        chunk = ctx.data[at:at + count] if at >= 0 else b""
        out["data"][row["section"]] = {"base": row["base"], "size": count, "sha256": _sha(chunk)}
    return out


# ---------------------------------------------------------------------------------------------
# Classes and claims


def verdict_class(source_name: str, source_text: str, strict: dict | None, behavioural) -> str:
    identical = bool(strict) and strict.get("verdict") in ("exact", "assumed") and bool(strict.get("text_equal"))
    if not identical:
        if behavioural is None:
            raise ReceiptError("nothing to receipt: the strict comparison did not find the bytes equal "
                               f"({(strict or {}).get('verdict', 'not run')})")
        return "behavioural"
    if Path(source_name).suffix.lower() in (".s", ".asm"):
        return "raw" if RAW_MARK in source_text else "assembly"
    return "exact" if strict["verdict"] == "exact" else "assumed"


def spec_of(build: dict) -> BuildSpec:
    return BuildSpec(build["pipeline"], list(build["flags"]),
                     {name: int(value, 16) for name, value in (build.get("section_bases") or {}).items()},
                     {name: int(value, 16) for name, value in (build.get("symbols") or {}).items()},
                     list(build.get("normalizer_args") or []))


def derive(cls: str, strict: dict | None, behavioural, provenance: dict, build: dict) -> tuple[dict, dict]:
    """``(claims, proof)`` as the recorded evidence gives them.  Behavioural evidence never makes a
    strict proof; a strict proof needs the strict verdict ``exact`` (declared addresses only, every
    loaded section compared equal)."""
    rows = (strict or {}).get("sections") or []
    byte_identity = bool(strict) and strict.get("verdict") in ("exact", "assumed") and bool(strict.get("text_equal"))
    ownership = byte_identity and all(row.get("status") == "equal" for row in rows)
    independent = pins_independent(build)
    strict_proof = (cls in ("exact", "assembly") and byte_identity and strict.get("verdict") == "exact"
                    and not strict.get("assumptions") and ownership and independent)
    found = ProvenanceResult(ProvenanceVerdict(provenance["verdict"]), tuple(provenance.get("reasons") or ()),
                             tuple(provenance.get("evidence") or ()))
    graded = proofmod.evaluate(spec_of(build), found, strict_proof, ownership)
    proof = {"proof_grade": bool(graded.proof_grade and strict_proof), "reasons": list(graded.reasons)}
    claims = {"byte_identity": byte_identity, "strict_proof": strict_proof,
              "assumptions_present": bool(strict and strict.get("assumptions")),
              "behavioural_evidence": behavioural is not None,
              "source_provenance": provenance["verdict"], "source_provenance_creditable": found.creditable,
              "ownership_complete": ownership, "proof_grade": proof["proof_grade"],
              "code_identity": byte_identity, "unit_identity": ownership, "pins_independent": independent}
    return claims, proof


def pinned(build: dict) -> list[str]:
    """Every address the recipe asserts: symbol pins, and section bases other than ``.text``
    (the strict comparison places ``.text`` at the subject's own address)."""
    return sorted(set(build.get("symbols") or {})
                  | {f"section:{name}" for name in (build.get("section_bases") or {}) if name != ".text"})


def pins_independent(build: dict) -> bool:
    """Whether every asserted address has a recorded origin independent of the bytes compared."""
    origins = build.get("pin_origins") or {}
    return all(_origin_kind(origins.get(name)) in INDEPENDENT_ORIGINS for name in pinned(build))


def _origin_kind(origin) -> str:
    """``transfer: 2 other function(s), e.g. f`` -> ``transfer``; a missing origin is ``unrecorded``."""
    return str("unrecorded" if origin is None else origin).split(":")[0].strip()


def origin_problems(build: dict, ctx: "Context") -> list[str]:
    """Origins that name something the receipt's context can check must hold there:
    ``run-declaration`` (the run's declarations give that value), ``declared`` (the declared
    map does) and ``fixed-address`` (low memory or a hardware register).  The donor stage's
    other origins (``definition``, ``transfer``, ``section-content``, ``section-chain``) are its
    record, checked by the caller against the run (verifyreceipt.verify_donors, which also computes
    the chains of sections again from the run's units)."""
    from .transplant import fixed_address
    origins = build.get("pin_origins") or {}
    base, declared, problems = ctx.base, dict(ctx.symbols), []
    for name in pinned(build):
        kind = _origin_kind(origins.get(name))
        if name.startswith("section:"):
            section = name[len("section:"):]
            value = int(build["section_bases"][section], 16)
            held = None if base is None else base.section_bases.get(section)
            if kind == "run-declaration" and held != value:
                problems.append(f"{name} = {_hex(value)} is said to be a declaration of the run, which does not hold it")
            elif kind == "declared":
                # a base the project states for one donor unit (transplant.stated_section_name)
                suffix = "_" + section.strip(".")
                if not any(key.startswith("n64dk_section_") and key.endswith(suffix) and held_ == value
                           for key, held_ in declared.items()):
                    problems.append(f"{name} = {_hex(value)} is said to be declared; the declared map states no such section base")
            elif kind == "fixed-address":
                problems.append(f"{name} = {_hex(value)}: a section base cannot have origin {kind}")
            continue
        value = int(build["symbols"][name], 16)
        if kind == "run-declaration" and (base is None or base.symbols.get(name) != value):
            problems.append(f"pin {name} = {_hex(value)} is said to be a declaration of the run, which does not hold it")
        elif kind == "declared" and declared.get(name) != value:
            problems.append(f"pin {name} = {_hex(value)} is said to be declared; the declared map does not hold it")
        elif kind == "fixed-address" and not fixed_address(value):
            problems.append(f"pin {name} = {_hex(value)} is said to be a fixed address; it is not one")
    return problems


def receipt_identity(receipt: dict) -> str:
    return canonical_digest({key: value for key, value in receipt.items() if key not in ("identity", "issued")})


def machine_paths(value, found: list | None = None) -> list[str]:
    """Strings in a receipt that name a place on some machine."""
    found = [] if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            machine_paths(key, found)
            machine_paths(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            machine_paths(item, found)
    elif isinstance(value, str) and _MACHINE.search(value):
        found.append(value)
    return found


def _class_problems(receipt: dict) -> list[str]:
    cls, strict = receipt["verdict"]["class"], receipt["verdict"]["strict"]
    kind = receipt["source"]["kind"]
    identical = bool(strict) and strict.get("verdict") in ("exact", "assumed") and bool(strict.get("text_equal"))
    if cls not in CLASSES:
        return [f"unknown verdict class {cls!r}"]
    earned = receipt.get("earned_by")
    if earned is not None:
        if earned not in EARNED_BY:
            return [f"unknown earning class {earned!r}"]
        # generated C is C and the assembly class is assembly; a donor (tree, SDK, cross-title) may ship
        # either, as libultra ships bcmp.s: its verdict class is then held to its kind below
        if (earned == "assembly" and kind != "assembly") or (earned == "generated-c" and kind != "c"):
            return [f"earning class {earned} with a {kind} source"]
    if cls == "behavioural":
        return [] if receipt.get("behavioural") is not None else ["class behavioural without behavioural evidence"]
    if not identical:
        return [f"class {cls} needs a byte-identical strict comparison; the receipt records "
                f"{(strict or {}).get('verdict', 'none')}"]
    if kind == "assembly" and cls not in ("assembly", "raw"):
        return [f"an assembly source cannot be class {cls}"]
    if kind == "c" and cls not in ("exact", "assumed"):
        return [f"a C source cannot be class {cls}"]
    if cls == "exact" and strict["verdict"] != "exact":
        return [f"class exact with strict verdict {strict['verdict']}"]
    if cls == "assumed" and strict["verdict"] != "assumed":
        return [f"class assumed with strict verdict {strict['verdict']}"]
    return []


def _source_class_problems(receipt: dict, data: bytes) -> list[str]:
    """The class follows from the source and the recorded strict comparison, as at issue: a listing
    of raw words is class raw whatever the receipt says (only an assembly class can be a proof)."""
    try:
        wanted = verdict_class(receipt["source"]["name"], data.decode("utf-8", errors="replace"),
                               receipt["verdict"]["strict"], receipt.get("behavioural"))
    except ReceiptError as error:
        return [str(error)]
    found = receipt["verdict"]["class"]
    return [] if wanted == found else [f"the source makes class {wanted}; the receipt says {found}"]


def check_static(receipt) -> list[str]:
    """Problems found without the image, the source or a build: format, identity, verifier
    version, class and claims against the recorded evidence, machine paths."""
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        return [f"not a {SCHEMA} receipt"]
    problems = []
    if receipt.get("identity") != receipt_identity(receipt):
        problems.append("forged or damaged: the receipt's content does not hash to its identity")
    if receipt.get("verifier") != VERIFIER_VERSION:
        problems.append(f"verifier version changed: the receipt was verified by {str(receipt.get('verifier'))[:12]}, "
                        f"this kit's strict verifier is {VERIFIER_VERSION[:12]}; issue the receipt again")
    try:
        problems += _class_problems(receipt)
        claims, proof = derive(receipt["verdict"]["class"], receipt["verdict"]["strict"], receipt.get("behavioural"),
                               receipt["provenance"], receipt["build"])
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return problems + [f"malformed receipt: {type(error).__name__}: {error}"]
    recorded = receipt.get("claims") or {}
    for key in sorted(set(claims) | set(recorded)):
        if recorded.get(key) != claims.get(key):
            problems.append(f"claim {key} is {recorded.get(key)!r}, the recorded evidence gives {claims.get(key)!r}")
    if receipt.get("proof") != proof:
        problems.append("proof grade does not follow from the recorded evidence")
    for value in machine_paths(receipt)[:3]:
        problems.append(f"machine path in the receipt: {value[:80]}")
    return problems


# ---------------------------------------------------------------------------------------------
# Issue


def issue(ctx: Context, *, name: str, vram: int, size: int, source: Path, spec: BuildSpec,
          section_sizes: dict[str, int] | None = None, handwritten_evidence: tuple[str, ...] = (),
          behavioural: dict | None = None, origin: str = "", issued_at: str | None = None,
          pin_origins: Mapping[str, str] | None = None, earned_by: str | None = None,
          in_place: bool = False, source_name: str | None = None) -> dict:
    """Build ``source`` with ``spec`` in a fresh directory, verify it strictly against the image
    and return the receipt.  Raises :class:`ReceiptError` when there is nothing to receipt (the
    bytes differ and no behavioural evidence is attached) or the receipt would not be portable.

    ``pin_origins`` names where each asserted address came from (``pinned``: symbol names and
    ``section:NAME``); an address with none is attributed to the run's declarations, the declared
    map or the recipe, as replay will justify it.  ``in_place``: build ``source`` where it is (a
    private copy of a donor tree whose relative includes must resolve; ``ctx.project`` is that
    copy) and name it ``source_name`` (its path in the tree)."""
    source = Path(source)
    spec = BuildSpec(spec.pipeline, portable_flags(list(spec.flags), ctx.project), dict(spec.section_bases),
                     dict(spec.symbols), list(spec.normalizer_args))
    data = source.read_bytes()
    with tempfile.TemporaryDirectory(prefix="n64dk-receipt-") as tmp:
        if in_place:
            staged = source
        else:
            staged = Path(tmp) / "src" / source.name
            staged.parent.mkdir(parents=True)
            staged.write_bytes(data)
        headers = ctx.headers(staged, spec)
        found = classify_source(staged, tuple(handwritten_evidence))
        elf = ctx.build(staged, spec, Path(tmp) / "build")
    from .transplant import confirm_sections
    verdict, confirmed = confirm_sections(
        lambda bases: _strict(ctx, elf, BuildSpec(spec.pipeline, list(spec.flags), bases, dict(spec.symbols),
                                                   list(spec.normalizer_args)), vram, size, section_sizes),
        elf, dict(spec.section_bases))
    if confirmed:                                           # placed by the unit's own data: recorded as such
        spec.section_bases.update(confirmed)
        pin_origins = {**{f"section:{section}": "section-content" for section in confirmed}, **dict(pin_origins or {})}
    strict = strict_block(verdict)
    text = data.decode("utf-8", errors="replace")
    cls = verdict_class(source.name, text, strict, behavioural)
    build = {"pipeline": spec.pipeline, "flags": list(spec.flags),
             "section_bases": {key: _hex(value) for key, value in sorted(spec.section_bases.items())},
             "symbols": {key: _hex(value) for key, value in sorted(spec.symbols.items())},
             "normalizer_args": list(spec.normalizer_args)}
    build["pin_origins"] = _origins(build, ctx, pin_origins)
    wrong = origin_problems(build, ctx)
    if wrong:
        raise ReceiptError(f"a pin origin does not hold: {wrong[0]}")
    provenance = {"verdict": found.verdict.value, "reasons": list(found.reasons), "evidence": list(found.evidence),
                  "origin": origin}
    claims, proof = derive(cls, strict, behavioural, provenance, build)
    receipt = {
        "schema": SCHEMA,
        "subject": {"image": ctx.image, "name": name, "vram": _hex(vram), "size": int(size),
                    "image_offset": _hex(ctx.vram_to_rom(vram)),
                    "catalogue_size": None if ctx.catalogue is None else ctx.catalogue.get(vram)},
        "image": ctx.image_identity(),
        "owned": owned_bytes(ctx, vram, size, strict),
        "ownership": {"section_sizes": {key: int(value) for key, value in sorted((section_sizes or {}).items())},
                      "function_starts_sha256": None if ctx.function_starts is None
                      else canonical_digest(sorted(ctx.function_starts))},
        "earned_by": earned_by or ("assembly" if source.suffix.lower() in (".s", ".asm") else "generated-c"),
        "source": {"name": source_name or source.name, "sha256": _sha(data),
                   "kind": "assembly" if source.suffix.lower() in (".s", ".asm") else "c"},
        "headers": headers,
        "build": build,
        "toolchain": ctx.toolchain_identity(spec),
        "wrapper": ctx.wrapper_identity(spec),
        "kit_driver_sha256": kit_driver_digest(),
        "environment": environment_identity(os.environ if ctx.environment is None else ctx.environment),
        "declarations": ctx.declarations(),
        "relocations": relocation_inputs(elf, vram, spec, ctx.symbols),
        "link": {"linker_script": None,
                 "note": "no link step: the verifier relocates the object at declared addresses"},
        "object": object_digests(elf),
        "comparison_sha256": comparison_digest(elf, vram, size, spec, ctx.symbols, verdict),
        "verifier": VERIFIER_VERSION,
        "verdict": {"class": cls, "strict": strict},
        "provenance": provenance,
        "proof": proof,
        "behavioural": behavioural,
        "claims": claims,
    }
    paths = machine_paths(receipt)
    if paths:
        raise ReceiptError(f"the receipt would name a machine path: {paths[0][:80]}")
    receipt["identity"] = receipt_identity(receipt)
    receipt["issued"] = {"at": issued_at or _now()}
    return receipt


def _origins(build: dict, ctx: Context, given: Mapping[str, str] | None) -> dict:
    given = dict(given or {})
    base = ctx.base
    out = {}
    for name in pinned(build):
        if name in given:
            out[name] = given[name]
            continue
        if name.startswith("section:"):
            section = name[len("section:"):]
            value = int(build["section_bases"][section], 16)
            out[name] = ("run-declaration" if base is not None and base.section_bases.get(section) == value
                         else "recipe")
            continue
        value = int(build["symbols"][name], 16)
        if base is not None and base.symbols.get(name) == value:
            out[name] = "run-declaration"
        elif dict(ctx.symbols).get(name) == value:
            out[name] = "declared"
        else:
            out[name] = "recipe"
    return out


def receipt_path(root: Path, image: str, name: str, vram: int) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return Path(root) / re.sub(r"[^A-Za-z0-9_.-]", "_", image) / f"{safe}-{vram & 0xFFFFFFFF:08X}.json"


def write(receipt: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------
# Replay


@dataclass
class Replay:
    identity: str | None
    subject: dict
    failures: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)
    rebuilt: bool = False
    elf: object = field(default=None, repr=False, compare=False)   # the rebuilt object (never serialised)

    @property
    def ok(self) -> bool:
        return not self.failures

    def check(self, name: str, problems: list[str]) -> None:
        self.checks[name] = "ok" if not problems else "failed"
        self.failures.extend(problems)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "rebuilt": self.rebuilt, "identity": self.identity, "subject": self.subject,
                "failures": self.failures, "notices": self.notices, "checks": self.checks}


def _recipe_problems(expected: str, build: dict, project: Path) -> list[str]:
    import shlex
    pipeline, _, flags = expected.strip().partition(" ")
    try:
        wanted = pipemod.split_flags(shlex.split(flags), pipeline)
        wanted_flags = portable_flags(wanted.flags, project)
    except (ValueError, pipemod.BuildError):
        return [f"the recipe it is offered with ({expected}) cannot be read"]
    if wanted.pipeline != build["pipeline"] or wanted_flags != list(build["flags"]):
        return [f"recipe differs: offered with {wanted.pipeline} {' '.join(wanted_flags)}, the receipt was built "
                f"with {build['pipeline']} {' '.join(build['flags'])}"]
    # the pins and section bases are part of the recipe the result is built with: a pin the receipt
    # never covered (added, removed or moved in the manifest) is a different build
    problems = []
    for kind, offered, recorded in (("symbol pin", wanted.symbols, build.get("symbols") or {}),
                                    ("section base", wanted.section_bases, build.get("section_bases") or {})):
        recorded = {name: int(value, 16) for name, value in recorded.items()}
        for name in sorted(set(offered) | set(recorded)):
            if kind == "section base" and name == ".text":
                continue
            if name in offered and name not in recorded:
                # a run-wide declaration the build added is recorded with the receipt; only a pin
                # the receipt does not hold is new
                problems.append(f"recipe differs: it pins {kind} {name} = {_hex(offered[name])}, which the receipt "
                                "never covered")
            elif name in offered and offered[name] != recorded[name]:
                problems.append(f"recipe differs: {kind} {name} is {_hex(offered[name])} in the recipe it is offered "
                                f"with, {_hex(recorded[name])} in the receipt")
    return problems


def _subject_problems(receipt: dict, ctx: Context, expect: dict | None) -> list[str]:
    subject, problems = receipt["subject"], []
    vram, size = int(subject["vram"], 16), int(subject["size"])
    if subject.get("image") != ctx.image:
        problems.append(f"receipt is for image {subject.get('image')!r}, replayed against {ctx.image!r}")
    expect = expect or {}
    if "name" in expect and expect["name"] != subject.get("name"):
        problems.append(f"receipt is for {subject.get('name')!r}, offered for {expect['name']!r}")
    if "vram" in expect and expect["vram"] != vram:
        problems.append(f"receipt is for 0x{vram:08X}, offered for 0x{expect['vram']:08X}: a receipt of another "
                        "function")
    if "size" in expect and expect["size"] != size:
        problems.append(f"catalogue boundary changed: the receipt covers 0x{size:X} bytes, the result has "
                        f"0x{expect['size']:X}")
    if ("catalogue_size" in expect and subject.get("catalogue_size") is not None
            and expect["catalogue_size"] != subject.get("catalogue_size")):
        problems.append(f"catalogue boundary changed: the catalogue gave 0x{subject['catalogue_size']:X} bytes at "
                        f"0x{vram:08X}, now 0x{expect['catalogue_size']:X}")
    if "source_sha256" in expect and expect["source_sha256"] != receipt["source"]["sha256"]:
        problems.append("source changed: the result's source is not the one the receipt covers")
    if expect.get("recipe"):
        problems += _recipe_problems(expect["recipe"], receipt["build"], ctx.project)
    return problems


def _catalogue_problems(receipt: dict, ctx: Context, replay: Replay) -> list[str]:
    if ctx.catalogue is None:
        return []
    vram, recorded = int(receipt["subject"]["vram"], 16), receipt["subject"].get("catalogue_size")
    now = ctx.catalogue.get(vram)
    if recorded is None:
        if now is not None:
            replay.notices.append(f"the catalogue now has a function at 0x{vram:08X} (0x{now:X} bytes); the receipt "
                                  "recorded none")
        return []
    if now is None:
        return [f"catalogue boundary changed: 0x{vram:08X} is no longer a catalogue function"]
    if now != recorded:
        return [f"catalogue boundary changed at 0x{vram:08X}: 0x{recorded:X} bytes when issued, now 0x{now:X}"]
    return []


def _owned_problems(receipt: dict, ctx: Context) -> list[str]:
    vram, size = int(receipt["subject"]["vram"], 16), int(receipt["subject"]["size"])
    owned, problems = receipt["owned"], []
    offset = ctx.vram_to_rom(vram)
    code = ctx.data[offset:offset + size] if offset >= 0 else b""
    if len(code) != size or _sha(code) != owned.get("code_sha256"):
        problems.append(f"owned code bytes at 0x{vram:08X} (+0x{size:X}) are not the bytes the receipt covers")
    for section, item in sorted((owned.get("data") or {}).items()):
        at, count = ctx.vram_to_rom(int(item["base"], 16)), int(item["size"])
        chunk = ctx.data[at:at + count] if at >= 0 else b""
        if len(chunk) != count or _sha(chunk) != item["sha256"]:
            problems.append(f"owned {section} bytes at {item['base']} (+0x{count:X}) changed in the image")
    return problems


def _header_problems(recorded: dict, current: dict) -> list[str]:
    problems = []
    before, after = recorded.get("headers") or {}, current.get("headers") or {}
    for name in sorted(set(before) | set(after)):
        if name not in after:
            problems.append(f"header {name} is no longer read (removed or no longer found)")
        elif name not in before:
            problems.append(f"header {name} is now read; the receipt did not depend on it")
        elif before[name] != after[name]:
            problems.append(f"header {name} changed")
    gone, new = set(recorded.get("missing") or ()), set(current.get("missing") or ())
    for name in sorted(gone ^ new):
        problems.append(f"include {name} {'now resolves' if name in gone else 'no longer resolves'}")
    return problems


def _toolchain_problems(recorded: dict, current: dict) -> list[str]:
    if recorded == current:
        return []
    if current.get("toolchain_digest") is None and recorded.get("toolchain_digest") is not None:
        return [f"compiler missing: no {recorded.get('pipeline')} toolchain is installed here"]
    stages = []
    before = {(s["stage"], s["component"]): s.get("files") for s in ((recorded.get("recipe") or {}).get("stages") or [])}
    after = {(s["stage"], s["component"]): s.get("files") for s in ((current.get("recipe") or {}).get("stages") or [])}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            stages.append(f"{key[0]} ({key[1]})")
    detail = f": stages {', '.join(stages)}" if stages else ""
    return [f"compiler changed: the {recorded.get('pipeline')} toolchain is not the one the receipt was built "
            f"with{detail}"]


def _wrapper_problems(recorded: dict, current: dict) -> list[str]:
    problems = []
    before, after = recorded.get("lane_code") or {}, current.get("lane_code") or {}
    for name in sorted(set(before) | set(after)):
        if before.get(name) != after.get(name):
            problems.append(f"wrapper changed: the kit's {name} stage is not the code the receipt was built with")
    if recorded.get("normalizer_hook") != current.get("normalizer_hook"):
        problems.append("wrapper changed: the project's normaliser hook differs")
    if recorded.get("assembler_flags") != current.get("assembler_flags"):
        problems.append(f"wrapper changed: assembler flags {recorded.get('assembler_flags')} -> "
                        f"{current.get('assembler_flags')}")
    return problems


def _environment_problems(recorded: dict, current: dict) -> list[str]:
    problems = []
    for name in sorted(set(recorded) | set(current)):
        if name not in current:
            problems.append(f"build environment: {name} was set when the receipt was issued and is not set now")
        elif name not in recorded:
            problems.append(f"build environment: {name} is set now and was not when the receipt was issued")
        elif recorded[name] != current[name]:
            problems.append(f"build environment: {name} has another value")
    return problems


def _declaration_problems(receipt: dict, ctx: Context) -> list[str]:
    """The run's own declarations (section bases and symbol pins every unit gets) against the
    recipe the receipt was built with."""
    if ctx.base is None:
        return []
    problems, build = [], receipt["build"]
    for kind, now, then in (("section base", ctx.base.section_bases, build.get("section_bases") or {}),
                            ("symbol pin", ctx.base.symbols, build.get("symbols") or {})):
        for name, value in sorted(now.items()):
            if kind == "section base" and name == ".text":
                continue
            if then.get(name) != _hex(value):
                problems.append(f"declaration changed: {kind} {name} is {_hex(value)} now, the receipt was built "
                                f"with {then.get(name)}")
    return problems


_GLABEL = re.compile(r"^\s*(?:glabel|jlabel)\s+(\w+)", re.M)
_LABEL = re.compile(r"^\s*(\w+):", re.M)
_NAME_TOKEN = re.compile(r"(?<![\w.$])[A-Za-z_]\w*")


def stated_pins(text: str, declared: Mapping[str, int]) -> dict[str, int]:
    """The symbol pins an assembly source justifies by itself: every name that states its own
    address (``handasm._STATED``: ``L_8000A174``, ``jtbl_80010CD0``) that ``text`` refers to and
    does not define, at the address it states, as handasm's proof resolves it.  A name
    ``declared`` holds is left to its declaration (membership in the plain mapping: a symbol map
    that answers "present" for every address-stating name would hide the missing pin)."""
    from . import handasm
    plain = dict(declared)
    defined = set(_GLABEL.findall(text)) | set(_LABEL.findall(text))
    out = {}
    for token in sorted(set(_NAME_TOKEN.findall(text)) - defined):
        found = handasm._STATED.match(token)
        if found and token not in plain:
            out[token] = int(found.group(1), 16)
    return out


def _pin_problems(receipt: dict, ctx: Context, data: bytes, expect: dict | None) -> list[str]:
    """Every symbol pin and section base of the receipt's recipe must come from something replay
    holds: the run's own declarations (``ctx.base``), the recipe the result is offered with
    (``--symbol=``, ``--section-base=``), the declared map (same value), or, for an assembly
    source, a name in the source that states its own address (:func:`stated_pins`).  The recipe
    is what the strict verifier relocates against and it overrides the declared map, so a pin
    nothing justifies would let a resealed receipt declare whatever address makes the bytes equal."""
    import shlex
    build, base, offered = receipt["build"], ctx.base, None
    if expect and expect.get("recipe"):
        pipeline, _, flags = expect["recipe"].strip().partition(" ")
        try:
            offered = pipemod.split_flags(shlex.split(flags), pipeline)
        except (ValueError, pipemod.BuildError):
            offered = None                                  # _recipe_problems names it
    declared = dict(ctx.symbols)
    stated = (stated_pins(data.decode("utf-8", errors="replace"), ctx.symbols)
              if receipt["source"].get("kind") == "assembly" else {})
    problems = []
    for name, value in sorted((build.get("symbols") or {}).items()):
        address = int(value, 16)
        if any(found.get(name) == address for found in (base.symbols if base is not None else {},
                                                        offered.symbols if offered is not None else {},
                                                        stated, declared)):
            continue
        if name in declared:
            problems.append(f"symbol pin {name} = {_hex(address)} overrides its declaration ({_hex(declared[name])})")
        else:
            problems.append(f"symbol pin {name} = {_hex(address)} is not justified: not a declaration of the run, "
                            "not in the recipe it is offered with, not an address its source states")
    for name, value in sorted((build.get("section_bases") or {}).items()):
        address = int(value, 16)
        if name == ".text" or any(found.get(name) == address for found in (
                base.section_bases if base is not None else {}, offered.section_bases if offered is not None else {})):
            continue
        if receipt.get("earned_by") == "generated-c" and \
                _origin_kind((build.get("pin_origins") or {}).get(f"section:{name}", "")) == "section-content":
            continue            # placed by the unit's own data: replay confirms it again from the rebuilt object
        problems.append(f"section base {name} = {_hex(address)} is not justified: not a declaration of the run, "
                        "not in the recipe it is offered with")
    return problems


def _relocation_problems(recorded: dict, current: dict) -> list[str]:
    problems = []
    if recorded.get("sites_sha256") != current.get("sites_sha256"):
        problems.append(f"relocation sites differ: the rebuilt object has {current.get('sites')} sites, the receipt "
                        f"{recorded.get('sites')}")
    for kind, key in (("declaration of", "symbols"), ("section base", "section_bases")):
        before, after = recorded.get(key) or {}, current.get(key) or {}
        for name in sorted(set(before) | set(after)):
            then, now = before.get(name), after.get(name)
            if then == now:
                continue
            if now is None and name in after:
                problems.append(f"{kind} {name} is gone (it was {then})")
            elif then is None and name in before:
                problems.append(f"{kind} {name} is now {now}; the receipt took it from the image")
            else:
                problems.append(f"{kind} {name} changed: {then} -> {now}")
    return problems


def replay(receipt: dict, ctx: Context, source: Path, *, expect: dict | None = None,
           rebuild: bool = True, in_place: bool = False) -> Replay:
    """Check ``receipt`` against ``ctx`` and ``source``: every digest recomputed from what is on
    disk, then (``rebuild``) a fresh build with the recorded recipe and a fresh strict comparison.
    ``expect`` names what the receipt is offered as (``image``, ``name``, ``vram``, ``size``,
    ``catalogue_size``, ``recipe``, ``source_sha256``): a receipt of another function, boundary or
    recipe is rejected.  No rebuild is attempted once an input is known to have changed."""
    out = Replay(receipt.get("identity") if isinstance(receipt, dict) else None,
                 (receipt.get("subject") if isinstance(receipt, dict) else None) or {})
    static = check_static(receipt)
    out.check("receipt", static)
    if any(item.startswith(("not a ", "malformed")) for item in static):
        return out
    try:
        vram, size = int(receipt["subject"]["vram"], 16), int(receipt["subject"]["size"])
        spec = spec_of(receipt["build"])
        out.check("subject", _subject_problems(receipt, ctx, expect))
        out.check("image", [] if ctx.image_identity()["sha256"] == receipt["image"]["sha256"] else
                  [f"image identity changed: the receipt is for image {receipt['image']['rom_key']} "
                   f"({receipt['image']['size']} bytes), this image is {ctx.image_identity()['rom_key']} "
                   f"({ctx.image_identity()['size']} bytes)"])
        out.check("catalogue", _catalogue_problems(receipt, ctx, out))
        out.check("owned", _owned_problems(receipt, ctx))
        source = Path(source)
        data = source.read_bytes() if source.is_file() else None
        if data is None:
            out.check("source", ["source missing"])
            return out
        out.check("source", [] if _sha(data) == receipt["source"]["sha256"] else
                  ["source changed: its digest is not the one the receipt covers"])
        out.check("class", _source_class_problems(receipt, data))
        out.check("declarations", _declaration_problems(receipt, ctx))
        out.check("pins", _pin_problems(receipt, ctx, data, expect))
        out.check("origins", origin_problems(receipt["build"], ctx))
        out.check("compiler", _toolchain_problems(receipt["toolchain"], ctx.toolchain_identity(spec)))
        out.check("wrapper", _wrapper_problems(receipt["wrapper"], ctx.wrapper_identity(spec)))
        out.check("environment", _environment_problems(
            receipt["environment"], environment_identity(os.environ if ctx.environment is None else ctx.environment)))
        if receipt.get("kit_driver_sha256") != kit_driver_digest():
            out.notices.append("the kit's build driver changed since the receipt was issued (the rebuild decides)")
        declared = ctx.declarations()
        if declared["map_sha256"] != receipt["declarations"]["map_sha256"]:
            out.notices.append("the declared symbol map differs from the one the receipt was issued with (only the "
                               "declarations its relocations use bind the receipt)")
        starts = None if ctx.function_starts is None else canonical_digest(sorted(ctx.function_starts))
        if starts != receipt["ownership"].get("function_starts_sha256"):
            out.notices.append("the function starts differ (they decide only conflicts of assumed calls)")
        with tempfile.TemporaryDirectory(prefix="n64dk-replay-") as tmp:
            if in_place:                                    # a private copy of a donor tree (see issue)
                staged = source
            else:
                staged = Path(tmp) / "src" / receipt["source"]["name"]
                staged.parent.mkdir(parents=True)
                staged.write_bytes(data)
            out.check("headers", _header_problems(receipt["headers"], ctx.headers(staged, spec)))
            found = classify_source(staged, tuple(receipt["provenance"].get("evidence") or ()))
            out.check("provenance", [] if found.verdict.value == receipt["provenance"]["verdict"] else
                      [f"source provenance is now {found.verdict.value}, the receipt says "
                       f"{receipt['provenance']['verdict']}"])
            if not rebuild:
                out.checks["rebuild"] = "skipped"
                return out
            if out.failures:
                out.checks["rebuild"] = "skipped: an input changed"
                return out
            try:
                elf = ctx.build(staged, spec, Path(tmp) / "build")
            except Exception as error:                      # noqa: BLE001 - any build failure rejects the receipt
                out.check("rebuild", [f"rebuild failed: {type(error).__name__}: {str(error)[-200:]}"])
                return out
        out.rebuilt = True
        out.elf = elf
        out.checks["rebuild"] = "ok"
        rebuilt = object_digests(elf)
        out.check("object", [] if rebuilt["loaded_sha256"] == receipt["object"]["loaded_sha256"] else
                  ["the rebuilt object differs from the emitted object the receipt records (loaded sections or "
                   "relocations)"])
        if rebuilt["object_sha256"] != receipt["object"]["object_sha256"]:
            out.notices.append("the object file differs outside its loaded sections (debug or path records)")
        out.check("relocations", _relocation_problems(receipt["relocations"],
                                                      relocation_inputs(elf, vram, spec, ctx.symbols)))
        verdict = _strict(ctx, elf, spec, vram, size, receipt["ownership"].get("section_sizes"))
        strict = strict_block(verdict)
        recorded = receipt["verdict"]["strict"] or {}
        problems = []
        if receipt.get("earned_by") == "generated-c":
            # a base said to be confirmed by the unit's own data is confirmed again here, from nothing
            # but the rebuilt object and the image: the record alone is not believed
            content = sorted(name[len("section:"):] for name, origin in (receipt["build"].get("pin_origins") or {}).items()
                             if name.startswith("section:") and _origin_kind(origin) == "section-content")
            if content:
                from .transplant import corroborated_sections
                bare = BuildSpec(spec.pipeline, list(spec.flags),
                                 {name: base for name, base in spec.section_bases.items() if name not in content},
                                 dict(spec.symbols), list(spec.normalizer_args))
                again = corroborated_sections(
                    _strict(ctx, elf, bare, vram, size, receipt["ownership"].get("section_sizes")), elf)
                problems += [f"section {name} at {_hex(spec.section_bases[name])} is said to be confirmed by the "
                             f"unit's own data; a fresh comparison does not confirm it"
                             for name in content if again.get(name) != spec.section_bases.get(name)]
        if strict["verdict"] != recorded.get("verdict"):
            problems.append(f"strict verdict on replay is {strict['verdict']}, the receipt records "
                            f"{recorded.get('verdict')}")
        elif (strict["sections"], strict["assumptions"], strict["text_equal"]) != (
                recorded.get("sections"), recorded.get("assumptions"), recorded.get("text_equal")):
            problems.append("strict comparison differs from the receipt's (sections, assumptions or text)")
        if comparison_digest(elf, vram, size, spec, ctx.symbols, verdict) != receipt.get("comparison_sha256"):
            problems.append("the relocated bytes compared on replay are not the ones the receipt records")
        claims, _proof = derive(receipt["verdict"]["class"], strict, receipt.get("behavioural"),
                                receipt["provenance"], receipt["build"])
        if claims != receipt["claims"]:
            changed = sorted(key for key in claims if claims[key] != receipt["claims"].get(key))
            problems.append(f"claims do not hold on replay: {', '.join(changed)}")
        out.check("strict", problems)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        out.check("format", [f"malformed receipt: {type(error).__name__}: {error}"])
    return out


# ---------------------------------------------------------------------------------------------
# Many results


def summarise(results: list[tuple[str, Replay | None, str]], rebuild: bool) -> dict:
    """``results``: ``(label, replay or None, reason when None)``."""
    verified = [label for label, found, _reason in results if found is not None and found.ok]
    rejected = [{"result": label, "failures": found.failures[:6]} for label, found, _reason in results
                if found is not None and not found.ok]
    missing = [{"result": label, "reason": reason} for label, found, reason in results if found is None]
    notices: dict[str, int] = {}
    for _label, found, _reason in results:
        for notice in (found.notices if found is not None else ()):
            notices[notice] = notices.get(notice, 0) + 1
    return {"ok": not rejected and not missing, "results": len(results),
            ("replayed" if rebuild else "verified_without_rebuild"): len(verified),
            "rejected": rejected, "missing": missing, "notices": notices}
