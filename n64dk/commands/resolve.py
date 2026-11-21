"""``resolve-case``, ``resolve-hard`` and ``hard-queue``: the hard-function resolution engine."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..analysis import flow as flowmod
from ..analysis.catalogue import load as load_catalogue
from ..cli import command
from ..match import autodraft, m2c_bridge
from ..match import iterate as itermod
from ..match import resolve as resolvemod
from ..match.engine import match_source
from ..match.pipeline import SMALL_DATA_LANES, small_data_threshold, split_flags
from .match import _common, _setup


def _int(text: str) -> int:
    return int(text, 0)


REDRAFT_DRAFTS = 6             # drafts (decompiler outputs first, then their normal forms) given to the rewrites


def _enabled(args, name: str) -> bool:
    """False when decompile-run switched ``name`` off for an ablation run (``options.disabled``;
    n64dk.commands.run.ABLATION_SWITCHES).  resolve-hard on its own disables nothing."""
    return name not in (getattr(args, "disabled", None) or ())


def _prepare(args):
    """(cfg, target, symbols) for a ROM with a boot segment, or for a linear image when
    ``--vram-delta`` is given (overlays, unpacked programs)."""
    from ..match import symbols as symmod
    from ..match.engine import Target
    from ..match.toolchain import load_config
    from ..rom import RomError
    def extra(syms):
        # additional symbol files (decompile-run: the program's _gp from its own startup code) are
        # added to whatever the project's symbols are, never instead of them
        for path in getattr(args, "extra_symbols", None) or ():
            if Path(path).is_file():
                for name, address in symmod.load([Path(path)]).items():
                    if not dict.__contains__(syms, name):         # the project's own value wins
                        syms[name] = address
        return syms
    try:
        cfg, target, syms, _files = _setup(args)
        return cfg, target, extra(syms)
    except (RomError, SystemExit) as error:
        if args.vram_delta is None:
            raise ValueError(f"{args.rom} is not a ROM with a boot segment ({error}); pass --vram-delta for a "
                             "linear image") from None
    cfg = load_config(args.project, getattr(args, "config", None), getattr(args, "toolchain_dir", None))
    if getattr(args, "include", None):
        cfg.include = list(dict.fromkeys([Path(cfg.project, value).resolve() for value in args.include] + cfg.include))
    data = Path(args.rom).read_bytes()
    files = [Path(value) for value in args.symbols] or cfg.symbols or symmod.discover(cfg.project)
    return cfg, Target(data, args.vram_delta, (0, len(data))), extra(symmod.load(files))


def _state(args, cfg) -> Path:
    return Path(args.state).resolve() if args.state else Path(cfg.project) / ".n64dk" / "resolve"


def _image_label(args) -> str:
    return args.image_name or Path(args.rom).stem


def synthesize_context(target, functions, names: dict[int, str], out_dir: Path, label: str,
                       parts: tuple[str, ...] = ("structs", "globals", "prototypes"),
                       proto_confidence: float = 0.9, headers=()) -> tuple[Path, str, dict]:
    """Recover provisional structs, global declarations and prototypes for a whole image and
    write them as an m2c context.  ``headers`` (recorded declarations, e.g. from the original
    build's debug records) come first, and a recovered prototype never replaces a recorded one.
    Returns (context path, declarations without typedefs for prepending to a draft, summary)."""
    from ..analysis import abi as abimod
    from ..analysis import layouts as layoutmod
    from ..analysis import values as valuemod

    facts = {}
    for _name, vram, size in functions:
        if size >= 8 and 0 <= vram - target.delta <= len(target.rom) - size:
            facts[vram] = valuemod.track(flowmod.analyse_in_image(target.rom, target.delta, vram, size))
    function_names = {vram: name for name, vram, _size in functions}
    layouts = layoutmod.recover(facts, {**names, **function_names})
    if "structs" not in parts:
        layouts.groups, layouts.param_group, layouts.return_group = [], {}, {}
    if "globals" not in parts:
        layouts.globals = []
    prototypes = abimod.solve(facts, function_names) if "prototypes" in parts else {}
    recorded = "".join(Path(header).read_text() for header in headers if Path(header).is_file())
    declared = set(re.findall(r"^[A-Za-z_][^;{}=]*?\b([A-Za-z_]\w*)\([^;{}]*\);$", recorded, re.M))
    if declared:
        prototypes = {vram: proto for vram, proto in prototypes.items()
                      if {**names, **function_names}.get(vram) not in declared}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}.context.h"
    rendered = layoutmod.Declarations(layouts, prototypes, proto_confidence)
    path.write_text(recorded + rendered.full(typedefs=True))
    info = {**layoutmod.summary(layouts), "prototypes": len(rendered.confident),
            "recorded_prototypes": len(declared)}
    return path, recorded + rendered.full(typedefs=False), info, rendered, facts


class _Runner:
    """Build/compare and draft services bound to one image and recipe."""

    def __init__(self, args, cfg, target, syms, work: Path):
        self.args, self.cfg, self.target, self.syms, self.work = args, cfg, target, syms, work
        self.spec = split_flags(shlex.split(args.flags), args.pipeline)
        self.context_path = Path(args.context) if args.context and args.context != "auto" else None
        self.declarations = ""
        self.rendered, self.facts = None, {}
        self.recipe_map = None
        self.unknown: set[int] = set()       # functions in stretches no known compiler explains
        self.translator_notes: dict[int, dict] = {}   # vram -> what the unoptimised-code translator made of it
        # alternative m2c renderings tried by the draft portfolio
        self.draft_options = [tuple(o.split("+")) for o in getattr(args, "draft_options", "").split(",") if o]

    def auto_context(self, functions, names, state: Path, label: str) -> dict | None:
        if self.args.context != "auto":
            return None
        parts = tuple(p for p in self.args.context_parts.split(",") if p)
        self.context_path, self.declarations, info, self.rendered, self.facts = synthesize_context(
            self.target, functions, names, state / "context", label, parts, self.args.proto_confidence,
            getattr(self.args, "context_headers", None) or ())
        return info

    def evaluator(self, vram: int, size: int, filename: str = "candidate.c", spec=None):
        spec = spec or self.spec

        def evaluate(text: str):
            folder = self.work / hashlib.sha256(
                f"{vram}:{spec.pipeline}:{spec.flags}:{text}".encode()).hexdigest()[:20]
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / filename
            path.write_text(text)
            try:
                outcome = match_source(path, spec, self.cfg, self.target, vram, size, self.syms, folder, None,
                                       strict_verification=True)
            except Exception as error:                       # a failed build is data for the case
                return itermod.Evaluation(None, error=str(error)[-400:]), None
            finally:
                # a sweep judges hundreds of thousands of candidates and the work directory is
                # often memory-backed: nothing in the folder is needed once the outcome exists
                shutil.rmtree(folder, ignore_errors=True)
            if outcome.error or outcome.result is None:
                return itermod.Evaluation(None, error=(outcome.error or "no comparison")[-400:]), None
            report = outcome.result.diff
            code = getattr(outcome.result, "compiled_text", b"") or None
            if report is None:
                return itermod.Evaluation(0 if outcome.exploratory_exact else 1, outcome.exact, None,
                                          outcome=outcome.as_dict()), code
            return itermod.Evaluation(report.score, outcome.exact, report.as_dict(limit=4096),
                                      outcome=outcome.as_dict()), code
        return evaluate

    def debug_twin(self, vram: int, size: int, filename: str):
        """The one alternative a large function is offered: its unoptimised GCC recipe with -g (no
        instruction moves; see ``alternatives``).  [] for any other recipe."""
        def only(_family):
            ido = self.spec.pipeline.startswith(("ido", "custom:ido", "custom:mipspro"))
            if ido or "-O0" not in self.spec.flags or "-g" in self.spec.flags:
                return []
            spec = split_flags(list(self.spec.flags) + ["-g"], self.spec.pipeline)
            spec.section_bases.update(self.spec.section_bases)
            spec.symbols.update(self.spec.symbols)
            return [(f"{spec.pipeline} {' '.join(spec.flags)}", spec.pipeline, self.evaluator(vram, size, filename, spec))]
        return only

    def recipes(self, vram: int, size: int, filename: str, limit: int = 12):
        """Alternative build recipes of the detected compiler family, usual retail settings first."""
        from ..match.engine import default_pipelines, sweep_specs

        from ..match import custom_pipeline
        try:                                                  # lanes the project declares ([pipelines.*])
            declared = custom_pipeline.load(self.cfg.project)
        except custom_pipeline.CustomPipelineError:
            declared = {}
        candidates = default_pipelines(self.cfg) + [custom_pipeline.PREFIX + name for name in declared]

        def is_ido(pipeline: str) -> bool:
            if custom_pipeline.is_custom(pipeline):
                return declared[pipeline[len(custom_pipeline.PREFIX):]].family == "ido"
            return pipeline.startswith("ido")

        def usual(pipeline: str, count: int) -> list:
            """The usual settings of one pipeline, most common first."""
            if custom_pipeline.is_custom(pipeline):           # a declared lane states its own usual flags
                lane = declared[pipeline[len(custom_pipeline.PREFIX):]]
                return [split_flags(list(flags), pipeline) for flags in [lane.default_flags, *lane.flag_sets] if flags]
            return sweep_specs([pipeline])[:count]

        if "kmc" in self.cfg.available() and "kmc+atload" not in candidates and self.at_loads(vram, size):
            # the function's own code loads globals through $at, which no installed assembler does
            # from a bare symbol load: offer the assembler variant that does (evidence-gated)
            candidates = ["kmc+atload"] + candidates
        if os.environ.get("N64DK_AT_MACROS") == "1" and "kmc" in self.cfg.available() \
                and "kmc+atmacro" not in candidates and self.at_branches(vram, size):
            # and its comparisons go through $at (a branch macro), which KMC GCC never writes.  Opt-in
            # (N64DK_AT_MACROS=1): on 146 Batman functions it moved drafts ~1% closer and made none
            # exact, so it is not worth its builds by default (round 12, TOOLCHAIN_REPORT.md)
            candidates = ["kmc+atmacro"] + candidates

        small_data = (dict.get(self.syms, "_gp") is not None and "kmc" in self.cfg.available()
                      and self.gp_uses(vram, size))
        if small_data:
            # the program declares _gp and this function addresses through $gp: the small-data
            # lanes (assembler at the compiler's -G), the KMC assembler first
            lanes = ["kmc+asg"] + (["kmc+gas"] if self.cfg.gas29() is not None else [])
            candidates = lanes + [p for p in candidates if p not in lanes]

        def small(spec):
            """A small-data lane at -G0 is the plain lane again: give it the usual -G8."""
            if spec.pipeline in SMALL_DATA_LANES and small_data_threshold(spec.flags) == 0:
                flags, skip = [], False
                for index, flag in enumerate(spec.flags):          # "-G0" or "-G 0"
                    if skip:
                        skip = False
                        continue
                    if flag == "-G":
                        skip = index + 1 < len(spec.flags)
                        continue
                    if not flag.startswith("-G"):
                        flags.append(flag)
                spec.flags = flags + ["-G8"]
            return spec

        def alternatives(family):
            pipelines = [p for p in candidates if is_ido(p) == (family == "ido") and p != self.spec.pipeline]
            if family is None:
                pipelines = [p for p in candidates if p != self.spec.pipeline]
            ruled_out = self.ruled_out(vram, size)
            pipelines = [p for p in pipelines if p not in ruled_out]
            out, seen = [], set()
            if self.recipe_map is not None:
                for row in self.recipe_map.candidates(vram, 4):      # proven nearby, nearest first
                    spec = split_flags(list(row["flags"]), row["pipeline"])
                    key = (spec.pipeline, tuple(spec.flags))
                    if key in seen or key == (self.spec.pipeline, tuple(self.spec.flags)):
                        continue
                    seen.add(key)
                    spec.section_bases.update(self.spec.section_bases)
                    spec.symbols.update(self.spec.symbols)
                    out.append((f"{spec.pipeline} {' '.join(spec.flags)}", spec.pipeline,
                                self.evaluator(vram, size, filename, spec)))
            # The first alternatives must span the compilers, not one compiler's variations: a
            # title built with one compiler usually carries library code built with another, and
            # when the family of a function is uncertain every available compiler is a candidate.
            #   1. the current flags under each pipeline of the same dialect (the other IDO
            #      version, the other KMC lanes);
            #   2. the usual settings of every other pipeline, one each;
            #   3. the other optimisation levels of the current compiler and its dialect siblings
            #      (libraries at -O1 or -O3, a debug file);
            #   4. the remaining usual settings, in turn.
            def dialect(pipeline: str) -> str:
                return pipeline[:3] if pipeline[:3] in ("ido", "kmc") else pipeline
            ordered = []
            if not is_ido(self.spec.pipeline) and "-O0" in self.spec.flags and "-g" not in self.spec.flags:
                # First of all for unoptimised GCC: the same recipe with debug information.  It moves no
                # instruction, but with -g GCC emits no no-op between a label and the loop statement after
                # it, and a title can hold files built both ways.  The drafts-x-recipes stage looks at the
                # first few alternatives only and has no size gate, so the twin has to stand here to be
                # tried on a large function that is a near miss (Wheel of Fortune: four functions, 9 KB).
                ordered.append(split_flags(list(self.spec.flags) + ["-g"], self.spec.pipeline))
            ordered += [split_flags(list(self.spec.flags), pipeline) for pipeline in pipelines
                        if dialect(pipeline) == dialect(self.spec.pipeline)]
            # Lanes the project declares for further compilers come last: calibration (which tries
            # every recipe) decides whether the title was built by one of them; trying each of them
            # on every function of a title built by another compiler only costs builds.
            lanes = [p for p in pipelines if custom_pipeline.is_custom(p)]
            pipelines = [p for p in pipelines if not custom_pipeline.is_custom(p)]
            swept = [usual(pipeline, max(1, limit // max(1, len(pipelines)))) for pipeline in pipelines]
            ordered += [specs[0] for specs in swept if specs]
            # optimisation levels: of the current recipe when its compiler is a candidate for this
            # function, and of each other candidate compiler's usual settings
            current_is_candidate = family is None or is_ido(self.spec.pipeline) == (family == "ido")
            bases = ([self.spec] if current_is_candidate else []) + list(ordered)
            for other in ("-O1", "-O3", "-O2", "unoptimised"):
                for base in bases:
                    level = next((f for f in base.flags if re.fullmatch(r"-O\d?", f)), None)
                    # unoptimised code is requested as -g from IDO and as -O0 from the others
                    wanted = ("-g" if is_ido(base.pipeline) else "-O0") if other == "unoptimised" else other
                    if level is not None and level != wanted:
                        ordered.append(split_flags([wanted if f == level else f for f in base.flags], base.pipeline))
                        if wanted == "-O0":
                            ordered.append(split_flags([wanted if f == level else f for f in base.flags] + ["-g"],
                                                       base.pipeline))
            # Unoptimised GCC with debug information: -g moves no instruction, but a line note then stands
            # between a label and the loop statement after it, so the no-op GCC puts there without -g is not
            # emitted (Wheel of Fortune: files built both ways; measured 2026-11-18).
            for base in bases:
                if not is_ido(base.pipeline) and "-O0" in base.flags and "-g" not in base.flags:
                    ordered.append(split_flags(list(base.flags) + ["-g"], base.pipeline))
            for rank in range(1, max((len(specs) for specs in swept), default=0)):
                ordered += [specs[rank] for specs in swept if rank < len(specs)]
            for lane in lanes:
                for spec in usual(lane, 1):
                    ordered.append(spec)
                    level = next((f for f in spec.flags if re.fullmatch(r"-O\d?", f)), None)
                    wanted = "-g" if is_ido(lane) else "-O0"
                    if level is not None:                     # and the same compiler without optimisation
                        ordered.append(split_flags([wanted if f == level else f for f in spec.flags], lane))
                        if wanted == "-O0":
                            ordered.append(split_flags([wanted if f == level else f for f in spec.flags] + ["-g"], lane))
            for spec in map(small, ordered):
                key = (spec.pipeline, tuple(sorted(spec.flags)))
                if key in seen or (spec.pipeline, tuple(spec.flags)) in seen:
                    continue
                seen.add(key)
                spec.section_bases.update(self.spec.section_bases)
                spec.symbols.update(self.spec.symbols)
                out.append((f"{spec.pipeline} {' '.join(spec.flags)}", spec.pipeline,
                            self.evaluator(vram, size, filename, spec)))
            return out
        return alternatives

    def _draft(self, name, code, vram, compiler, context, declarations, names, tables, evaluate, extra=None):
        """One draft: m2c, self-contained preparation, recovered declarations, mechanical repairs."""
        gp = dict.get(self.syms, "_gp")                     # declared only (never an inferred name)
        result = m2c_bridge.draft(name, code, vram, compiler, context, names, tables, extra, gp=gp)
        if result.get("source") is None:
            return None, f"{result.get('status')}: {result.get('error')}"
        text = autodraft.prepare(result["source"])
        if declarations:
            # drop the draft's own guesses for anything the context already declares
            # drop the draft's own guesses for what the context declares (top level only: a call
            # statement inside the body is not a declaration)
            text = autodraft.PRELUDE + "\n" + declarations + autodraft.drop_redeclared(
                text[len(autodraft.PRELUDE):], declarations)
        for _attempt in range(12):                           # mechanical compile repairs only
            evaluation, _code = evaluate(text)
            if evaluation.score is not None:
                break
            fixed = autodraft.repair(text, evaluation.error or "")
            if fixed is None or fixed == text:
                break                                        # the baseline stage reports the build error
            text = fixed
        return text, None

    def behind(self) -> list:
        """The project's other images as ``(data, vram of the first byte)``: where a jump table of this
        image's code may stand (m2c_bridge.find_jump_tables, ``also``).  Read once; [] without a registry."""
        cached = getattr(self, "_behind", None)
        if cached is None:
            cached = []
            try:
                from ..match import images as imagemod
                for spec in imagemod.load_registry(self.cfg.project).values():
                    if spec.delta is None or not spec.path.is_file():
                        continue
                    data = spec.load()
                    if data != self.target.rom:
                        cached.append((data, spec.delta))
            except Exception:                                # no registry, or an image that cannot be read
                cached = []
            self._behind = cached
        return cached

    def rewriter(self, name: str, vram: int, size: int):
        """``text -> [(label, text)]``: the recipe's rewrites of a source of this function, or None when the
        rewrites are switched off (n64dk.match.resolve.resolve, ``rewrites``)."""
        if not _enabled(self.args, "redraft"):
            return None
        from ..match import redraft
        offset = vram - self.target.delta
        code = self.target.rom[offset:offset + size]

        def read(address: int):
            position = address - self.target.delta
            return int.from_bytes(self.target.rom[position:position + 4], "big") \
                if 0 <= position <= len(self.target.rom) - 4 else None

        def rewrites(text: str):
            return redraft.variants(text, name, code=code, vram=vram, pipeline=self.spec.pipeline,
                                    flags=tuple(self.spec.flags), family=None, read=read)
        return rewrites

    def drafter(self, name: str, vram: int, size: int, evaluate, names: dict[int, str]):
        def draft(family=None):
            offset = vram - self.target.delta
            code = self.target.rom[offset:offset + size]
            tables = m2c_bridge.find_jump_tables(self.target.rom, self.target.delta, vram, size, also=self.behind())
            if family is None and self.args.pipeline.startswith("custom:"):
                from ..match import custom_pipeline
                family = custom_pipeline.family_of(self.args.pipeline, self.cfg.project)
            compiler = "ido" if family == "ido" else ("kmc" if family == "gcc" else self.args.pipeline)
            modes = [("plain", None, "", None)]
            if self.context_path is not None:
                modes.append(("context", self.context_path, self.declarations, None))
            if not self.args.draft_portfolio:
                # single draft: with the context when there is one, falling back to plain
                text, error = self._draft(name, code, vram, compiler, modes[-1][1], modes[-1][2], names, tables,
                                          evaluate)
                if text is None and len(modes) > 1:
                    text, error = self._draft(name, code, vram, compiler, None, "", names, tables, evaluate)
                return text, error
            if self.rendered is not None and vram in self.facts:
                # portfolio: declarations cut down to this function.  Callee prototypes tell the
                # decompiler how many arguments a call takes; the function's own prototype is the
                # least certain part, so it is offered as a separate variant.
                modes = [("plain", None, "", None)]
                folder = self.work / "context"
                folder.mkdir(parents=True, exist_ok=True)
                for label, include_self in (("callees", False), ("callees+self", True)):
                    text = self.rendered.focused(self.facts[vram], vram, include_self, typedefs=True)
                    if text.count("\n") <= 4:
                        continue                                     # nothing recovered for this function
                    path = folder / f"{name}.{label.replace('+', '_')}.h"
                    path.write_text(text)
                    modes.append((label, path, self.rendered.focused(self.facts[vram], vram, include_self,
                                                                     typedefs=False), None))
            from ..match import layoutinfer
            if layoutinfer.enabled():
                # opt-in experiment: competing array / structure-field declarations from this
                # function's own access widths, offsets and strides (at most 4; drafts only)
                modes += layoutinfer.draft_modes(name, code, vram, self.work / "context", self.rendered,
                                                 self.facts)
            modes += [("plain:" + "+".join(flags), None, "", list(flags)) for flags in self.draft_options]
            out, errors = [], []
            for label, context, declarations, extra in modes:
                text, error = self._draft(name, code, vram, compiler, context, declarations, names, tables,
                                          evaluate, extra)
                if text is not None and text not in [t for _l, t in out]:
                    out.append((label, text))
                elif error:
                    errors.append(error)
            from ..match import redraft_gcc_debug

            def read(address: int):
                at = address - self.target.delta
                rom = self.target.rom
                return int.from_bytes(rom[at:at + 4], "big") if 0 <= at <= len(rom) - 4 else None

            def note_translator(found: bool) -> None:
                if not redraft_gcc_debug.is_unoptimised_gcc(code):
                    return
                reason = None if found else redraft_gcc_debug.decline_reason(name, code, vram, read)
                self.translator_notes[vram] = {"declined": reason} if reason else {"translated": True}
            if not out:
                # The unoptimised-GCC translator reads the machine code alone: it needs no draft from the
                # decompiler, only the kit's prelude, so a function the decompiler fails on is still tried.
                alone = (redraft_gcc_debug.variants("", name, code=code, vram=vram, pipeline=self.spec.pipeline,
                                                    flags=tuple(self.spec.flags), family=family, read=read)
                         if _enabled(self.args, "redraft") else [])
                if _enabled(self.args, "redraft"):
                    note_translator(bool(alone))
                if not alone:
                    return None, "; ".join(errors[:2]) or "m2c produced nothing"
                return [(f"translated/{kind}", text) for kind, text in alone]
            # normal forms of each draft (loops un-rotated, masks dropped, ...) join the portfolio
            from ..match import draftforms
            texts = {text for _label, text in out}
            for label, text in (list(out)[:2] if _enabled(self.args, "draft-forms") else []):
                for kind, normal, _review in draftforms.drafts(text, name):
                    if normal not in texts:
                        texts.add(normal)
                        out.append((f"{label}/{kind}", normal))
            # recipe-specific rewrites of the first drafts (n64dk.match.redraft): computed from the
            # target's own machine code, e.g. the frame layout of an unoptimised build.  The normal forms
            # of those drafts are rewritten as well: a rewrite keyed on how a loop or a temporary is written
            # may apply to the normal form only (measured on a title: two functions matched from the form
            # the search reached later and not from the first draft)
            from ..match import redraft
            for label, text in (list(out)[:REDRAFT_DRAFTS] if _enabled(self.args, "redraft") else []):
                for kind, variant in redraft.variants(text, name, code=code, vram=vram, pipeline=self.spec.pipeline,
                                                      flags=tuple(self.spec.flags), family=family, read=read):
                    if variant not in texts:
                        texts.add(variant)
                        out.append((f"{label}/{kind}", variant))
            if _enabled(self.args, "redraft"):
                note_translator(any("gcc-O0-translated" in label for label, _text in out))
            return out
        def assembly():
            """(source, proof) for a hand-written function whose assembly assembles back to its bytes."""
            from ..match import handasm
            offset = vram - self.target.delta
            code = self.target.rom[offset:offset + size]
            return handasm.attempt(name, code, vram, self.cfg, names,
                                   m2c_bridge.find_jump_tables(self.target.rom, self.target.delta, vram, size, also=self.behind()))

        if _enabled(self.args, "handasm"):          # switched off: hand-written code stays unsettled
            draft.assembly = assembly
        return draft

    def strict(self, vram: int, size: int, filename: str, starts: set[int] | None):
        """Non-circular re-verification of an exact candidate with the recipe that produced it."""
        from ..elf import Elf
        from ..match import verify as verifymod
        from ..match.pipeline import build

        def check(text: str, recipe: str) -> dict:
            pipeline, _, flags = (recipe or f"{self.spec.pipeline} {' '.join(self.spec.flags)}").partition(" ")
            spec = split_flags(shlex.split(flags), pipeline)
            spec.section_bases.update(self.spec.section_bases)
            spec.symbols.update(self.spec.symbols)
            folder = self.work / ("strict-" + hashlib.sha256(f"{vram}:{recipe}:{text}".encode()).hexdigest()[:20])
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / filename
            path.write_text(text)
            try:
                elf = Elf.load(build(path, spec, self.cfg, folder))
            finally:
                shutil.rmtree(folder, ignore_errors=True)
            # only independently declared symbols: names in the candidate's own text prove nothing
            symbols = self.syms.copy()
            symbols.update(spec.symbols)
            # A section of the unit (a switch's jump table, its constants) whose address the code gives
            # and whose own bytes are then found there is placed by its contents, as a donor unit's is.
            from ..match import transplant
            verdict, confirmed = transplant.confirm_sections(
                lambda bases: verifymod.strict_compare(elf, self.target.rom, self.target.vram_to_rom, vram, size,
                                                       symbols, bases, None, starts),
                elf, dict(spec.section_bases))
            out = verdict.as_dict()
            if confirmed:
                out["corroborated"] = {name: f"0x{address:08X}" for name, address in sorted(confirmed.items())}
            return out
        return check

    def find_unknown(self, functions) -> dict | None:
        """Stretches of the image no compiler of the marks table explains (``idioms.unknown_stretches``).
        Their functions are still drafted and tried under the recipes; the variant search is not
        run on them (measured on the title that showed the case: 812 such functions, 26 matched,
        none by the search, three quarters of the title's builds)."""
        from ..match import idioms
        self.unknown: set[int] = set()
        if getattr(self, "_idiom_table", None) is None:
            self._idiom_table = idioms.Table.load()
        if not self._idiom_table.classes:
            return None
        marked = []
        for _name, vram, size in functions:
            offset = vram - self.target.delta
            code = self.target.rom[offset:offset + size]
            if size >= 8 and len(code) == size:
                marked.append((vram, idioms.features(code, vram, sequences=False)))
        found = idioms.unknown_stretches(marked, self._idiom_table)
        for row in found:
            self.unknown.update(vram for vram, _marks in marked if row["start"] <= vram <= row["end"])
        if not found:
            return None
        return {"functions": len(self.unknown),
                "stretches": [{"start": f"0x{row['start']:08X}", "end": f"0x{row['end']:08X}",
                               "functions": row["functions"], "marks": row["marks"]} for row in found]}

    UNOPTIMISED = frozenset({"args=homed", "args=homed-fp", "frame=pointer", "nops=many"})
    NEIGHBOURHOOD = 12

    def unoptimised_near(self, vram: int) -> bool:
        """Unoptimised-code marks in the function or in a quarter of its catalogue neighbourhood
        (``NEIGHBOURHOOD`` functions each side): where the wide recipe retry has paid."""
        import bisect
        from ..match import idioms
        if getattr(self, "_unopt", None) is None:
            rows = []
            for _name, start, size in sorted(getattr(self, "functions_for_gate", []), key=lambda r: r[1]):
                offset = start - self.target.delta
                code = self.target.rom[offset:offset + size]
                rows.append((start, bool(idioms.features(code, start, sequences=False) & self.UNOPTIMISED)
                             if len(code) == size and size >= 8 else False))
            self._unopt = rows
            self._unopt_starts = [r[0] for r in rows]
        index = bisect.bisect_left(self._unopt_starts, vram)
        if index < len(self._unopt) and self._unopt[index][0] == vram and self._unopt[index][1]:
            return True
        window = self._unopt[max(0, index - self.NEIGHBOURHOOD):index + self.NEIGHBOURHOOD + 1]
        return bool(window) and 4 * sum(flag for _v, flag in window) >= len(window)

    def gp_uses(self, vram: int, size: int) -> bool:
        """The function's own code addresses memory through ``$gp`` (loads, stores, ``addiu``)."""
        from ..analysis import gp as gpmod
        offset = vram - self.target.delta
        code = self.target.rom[offset:offset + size]
        return gpmod.scan(code, vram, [("", vram, len(code))])["uses"] > 0

    def at_branches(self, vram: int, size: int) -> bool:
        """The function's code compares into ``$at`` for a branch (a branch macro)."""
        from ..match import macros
        offset = vram - self.target.delta
        return macros.signatures(self.target.rom[offset:offset + size], vram)["at-branch"] > 0

    def at_loads(self, vram: int, size: int) -> bool:
        """The function's code loads a global through ``$at`` right after ``lui $at``."""
        if not _enabled(self.args, "at-loads"):
            return False                               # the lane is switched off (ablation)
        from ..match import idioms
        offset = vram - self.target.delta
        return "at=direct-load" in idioms.features(self.target.rom[offset:offset + size], vram, sequences=False)

    def ruled_out(self, vram: int, size: int) -> set[str]:
        """Compiler lanes this function's own machine code rules out: lanes none of whose proven
        functions ever showed an encoding or convention the function has (``n64dk.match.idioms``;
        on 5,429 proven functions of sixteen titles the true lane was never among them)."""
        from ..match import idioms
        if getattr(self, "_idiom_table", None) is None:
            self._idiom_table = idioms.Table.load()
        table = self._idiom_table
        if not table.classes:
            return set()
        offset = vram - self.target.delta
        marks = idioms.features(self.target.rom[offset:offset + size], vram, sequences=False)
        excluded = table.excluded(marks)
        out = set()
        for lane in {name.split()[0] for name in table.classes}:
            classes = [name for name, entry in table.classes.items()
                       if name.split()[0] == lane and entry["functions"] >= 200]
            if classes and all(name in excluded for name in classes):
                out.add(lane)
        return out

    def label(self, spec=None) -> str:
        spec = spec or self.spec
        return f"{spec.pipeline} {' '.join(spec.flags)}"

    def learn_recipes(self, functions, matched_rows) -> dict | None:
        """Partition the image into probable objects and remember which recipe matched where."""
        from ..analysis import units as unitmod
        from ..match import causes as causemod

        def family(flow):
            found = causemod.compiler_family(flow)
            return found["family"] if found["confidence"] >= 0.8 else None
        self.recipe_map = unitmod.RecipeMap(unitmod.units(self.target.rom, self.target.delta, functions,
                                                          family_of=family))
        for vram, size, spec in matched_rows:
            self.recipe_map.add(vram, size, spec.pipeline, spec.flags)
        return self.recipe_map.summary() if matched_rows else None

    def spec_for(self, vram: int):
        """The recipe to start from: one proven inside the same object, else the recipe calibrated
        for the toolchain region the function lies in (``region_map``, regional calibration),
        else the command line's."""
        ranked = self.recipe_map.candidates(vram, 1) if self.recipe_map is not None and self.recipe_map.rows else []
        if ranked and ranked[0]["same_unit"]:
            spec = split_flags(list(ranked[0]["flags"]), ranked[0]["pipeline"])
            spec.section_bases.update(self.spec.section_bases)
            spec.symbols.update(self.spec.symbols)
            return spec
        for region in getattr(self.args, "region_map", None) or ():
            if region["start"] <= vram <= region["end"]:
                spec = split_flags(shlex.split(region["flags"]), region["pipeline"])
                spec.section_bases.update(self.spec.section_bases)
                spec.symbols.update(self.spec.symbols)
                return spec
        return self.spec

    def flow(self, vram: int, size: int):
        return flowmod.analyse_in_image(self.target.rom, self.target.delta, vram, size)


def twin_source(text: str, source_name: str, new_name: str, source_sites, new_sites,
                names: dict[int, str]) -> str | None:
    """Rewrite an exact function's source for a machine-code twin at another address.

    ``*_sites`` are the relocation sites of both bodies (``transplant.mask``): the twin calls and
    references whatever stands at the corresponding sites, so each symbol is renamed to the one at
    the twin's address.  Returns None when the bodies do not correspond site for site."""
    if len(source_sites) != len(new_sites) or any(a[:2] != b[:2] for a, b in zip(source_sites, new_sites)):
        return None
    moved: dict[int, int] = {}
    called: set[int] = set()
    for (_i, kind, old), (_j, _kind, new) in zip(source_sites, new_sites):
        if kind in ("jal", "hi", "lo") and old != new:
            if moved.setdefault(old, new) != new:
                return None                                  # one symbol would need two addresses
            if kind == "jal":
                called.add(old)
    renames = {source_name: new_name}
    for old, new in moved.items():
        if old in names and names[old] != source_name:
            renames[names[old]] = names.get(new) or (f"func_{new:08X}" if old in called else f"D_{new:08X}")
    def swap(found):
        word = found.group(0)
        if word in renames:
            return renames[word]
        numbered = re.fullmatch(r"(func|D|jtbl)_([0-9A-Fa-f]{8})", word)
        if numbered and int(numbered.group(2), 16) in moved:
            return f"{numbered.group(1)}_{moved[int(numbered.group(2), 16)]:08X}"
        return word
    return re.sub(r"[A-Za-z_]\w*", swap, text)


def _twin_pass(store, image: str, todo, target, runner, names, emit, starts) -> list:
    """After a sweep: every unresolved function whose masked machine code equals an exact case's
    gets that case's source, renamed symbol by symbol, built with the same recipe and recorded
    through the normal ladder (so it is strictly verified like any other exact case)."""
    from ..match import transplant
    groups: dict[str, list] = {}
    for name, vram, size in todo:
        offset = vram - target.delta
        masked, sites = transplant.mask(target.rom[offset:offset + size], vram)
        groups.setdefault(f"{transplant.signature(masked)}:{len(masked)}", []).append((name, vram, size, sites))
    won = []
    for members in groups.values():
        if len(members) < 2:
            continue
        cases = {m[1]: store.get(resolvemod.case_id(image, m[1], m[2])) for m in members}
        donors = [m for m in members if cases[m[1]] is not None and cases[m[1]].status == "exact"]
        if not donors:
            continue
        donor = donors[0]
        donor_case = cases[donor[1]]
        text = store.source(donor_case.best_digest)
        if text is None:
            continue
        pipeline, _, flags = donor_case.recipe.partition(" ")
        spec = split_flags(shlex.split(flags), pipeline) if pipeline else runner.spec
        spec.section_bases.update(runner.spec.section_bases)
        spec.symbols.update(runner.spec.symbols)
        for name, vram, size, sites in members:
            case = cases[vram]
            if vram == donor[1] or (case is not None and case.status in ("exact", "blocked-boundary")):
                continue
            candidate = twin_source(text, donor[0], name, donor[3], sites, names)
            if candidate is None:
                continue
            evaluate = runner.evaluator(vram, size, f"{name}.c", spec)
            evaluation, _code = evaluate(candidate)
            if not evaluation.exact:
                continue
            offset = vram - target.delta
            result = resolvemod.resolve(store, image, name, vram, target.rom[offset:offset + size], evaluate,
                                        source=candidate, flow=runner.flow(vram, size), neighbours=starts,
                                        budget=4, jobs=1, force=True, emit=emit, pipeline=spec.pipeline,
                                        recipe_label=runner.label(spec), rom_offset=offset,
                                        strict=runner.strict(vram, size, f"{name}.c", starts))
            if result.status == "exact":
                result.stages.insert(0, {"stage": "twin", "of": donor[0]})
                store.put(result)
                won.append(result)
                if emit:
                    emit({"event": "twin-exact", "case": result.case_id, "name": name, "of": donor[0]})
    return won


_BASIC_TYPEDEF = re.compile(r"(?m)^[ \t]*typedef\s+[^;{}]*?\b(?:s8|u8|s16|u16|s32|u32|s64|u64|f32|f64)\s*;[ \t]*\n?")


def _unit_recipe_pass(store, image: str, todo, target, runner, names, emit, starts, args, settled=None) -> dict:
    """Retry unresolved functions under the recipe proven for their own object.

    Every function of one object was built by one compiler invocation.  The sweep attempts each
    function on its own: its recipe is the image's, or whichever alternative happened to score
    lowest on a draft, and the search then continues under that one.  Once the sweep has matched
    some members of an object (``n64dk.analysis.units``: a run of functions closed by alignment
    padding), their recipe is evidence about the others.  A function that is still unresolved and
    was attempted under a different recipe than its nearest matched neighbour in the same object
    is started again from its drafts under that neighbour's recipe.  The stored case stands when
    the new attempt ends further away."""
    if runner.recipe_map is None:
        return {"retried": 0}
    proven = 0
    for known in store.all():
        if known.image == image and known.status in ("exact", "assumed") and known.recipe:
            pipeline, _, flags = known.recipe.partition(" ")
            runner.recipe_map.add(known.vram, known.size, pipeline, shlex.split(flags))
            proven += 1
    if not proven:
        return {"retried": 0}
    rows = []
    for name, vram, size in todo:
        case = store.get(resolvemod.case_id(image, vram, size))
        if case is None or case.status not in ("open", "near", "does-not-build"):
            continue
        if settled is not None and settled(vram):
            continue                                          # its region's recipe was established and tried
        ranked = runner.recipe_map.candidates(vram, 1)
        if not ranked or not ranked[0]["same_unit"]:
            continue
        label = f"{ranked[0]['pipeline']} {' '.join(ranked[0]['flags'])}"
        if label == case.recipe or any(stage.get("stage") == "unit-recipe" and stage.get("recipe") == label
                                       for stage in case.stages):
            continue                                          # already attempted under this recipe
        rows.append((name, vram, size, ranked[0], label))
    return {**_retry_under_recipes(store, image, rows, target, runner, names, emit, starts, args, "unit-recipe"),
            "proven_members": proven}


def _retry_under_recipes(store, image: str, rows, target, runner, names, emit, starts, args, stage: str) -> dict:
    """Start each ``(name, vram, size, {"pipeline", "flags"}, label)`` again from its drafts under
    that recipe.  The stored case stands when the new attempt ends further away."""
    totals = {"retried": 0, "improved": 0, "exact": 0, "assumed": 0}
    lock = threading.Lock()

    def one(row):
        name, vram, size, recipe, label = row
        before = store.get(resolvemod.case_id(image, vram, size))
        spec = split_flags(list(recipe["flags"]), recipe["pipeline"])
        spec.section_bases.update(runner.spec.section_bases)
        spec.symbols.update(runner.spec.symbols)
        evaluate = runner.evaluator(vram, size, f"{name}.c", spec)
        offset = vram - target.delta
        try:
            case = resolvemod.resolve(store, image, name, vram, target.rom[offset:offset + size], evaluate,
                                      draft=runner.drafter(name, vram, size, evaluate, names),
                                      flow=runner.flow(vram, size), neighbours=starts, budget=args.budget,
                                      search_ceiling=getattr(args, "search_ceiling", resolvemod.SEARCH_CEILING),
                                      jobs=2, emit=emit, pipeline=spec.pipeline, recipe_label=label, rom_offset=offset,
                                      strict=runner.strict(vram, size, f"{name}.c", starts), recipes=None, fresh=True,
                                      frame_fit=_enabled(args, "frame-fit"), search=_enabled(args, "search"),
                                      hand_forms=_enabled(args, "hand-forms"),
                                      rewrites=runner.rewriter(name, vram, size))
        except Exception as error:                           # one bad function must not stop the pass
            if emit:
                emit({"event": "case-error", "name": name, "error": str(error)[-300:]})
            return None
        case.stages.append({"stage": stage, "recipe": label, "kept": case.recipe == label})
        store.put(case)
        rank = {"exact": 0, "assumed": 1, "near": 2}
        better = rank.get(case.status, 3) < rank.get(before.status, 3) or (
            rank.get(case.status, 3) == rank.get(before.status, 3) and case.best_score is not None
            and (before.best_score is None or case.best_score < before.best_score))
        with lock:
            totals["retried"] += 1
            totals["improved"] += bool(better)
            totals["exact"] += case.status == "exact"
            totals["assumed"] += case.status == "assumed"
        if emit:
            emit({"event": stage, "name": name, "recipe": label, "status": case.status,
                  "score": case.best_score, "before": before.status})
        return case

    if not rows:
        return {**totals, "cases": {}}
    with ThreadPoolExecutor(max(1, min(len(rows), (args.jobs or 8)))) as pool:
        cases = {case.case_id: case for case in pool.map(one, rows) if case is not None}
    return {**totals, "cases": cases}


# A function above this size is drafted and built under its recipe, and that is all: no other
# recipes, no variant search.  Measured on sixteen titles, 5% of functions are larger and hold 37%
# of the code; the translators for unoptimised builds match some of them outright (23 of 447 on
# four titles), and searching them costs the most and returns the least.
LARGE_FUNCTION = 0x400
# No function larger than 6,244 bytes has ever been recovered (2026-11-18, 28 titles); catalogue entries of
# 400-860 KB (merged code and data) drove the resolver worker past 3 and 6 GB.  Above this ceiling a function
# is not attempted; it stays assembly with the reason recorded.
IMPLAUSIBLE_FUNCTION_BYTES = 0x20000        # real functions of 17 KB (Wheel of Fortune) and 38 KB (Virtual Chess) exist


def implausible_size() -> int:
    """The size above which a catalogue entry is not attempted (0: no ceiling)."""
    import os
    text = os.environ.get("N64DK_IMPLAUSIBLE_FUNCTION_BYTES", "").strip()
    if not text:
        return IMPLAUSIBLE_FUNCTION_BYTES
    try:
        return int(text, 0)
    except ValueError:
        return IMPLAUSIBLE_FUNCTION_BYTES

REGION_EVIDENCE_LIMIT = 400      # matched functions rebuilt under the other recipes, spread over the image
REGION_RECIPES = 6               # recipes compared (the most common among the matches)


def _region_recipe_pass(store, image: str, todo, target, runner, names, emit, starts, args) -> dict:
    """Retry unresolved functions under the recipe of the region they lie in.

    The sweep records the first recipe that matched each function, which under-reports: a
    function matched under the image's recipe was never tried under another.  Here the matched
    functions' own sources are rebuilt under the other recipes the image has shown (positive
    tests on sources known to compile), the image is segmented into the stretches one recipe
    explains (``n64dk.match.regions``), and every unresolved function attempted under a different
    recipe than its region's is started again under it."""
    from ..match import handasm
    from ..match import regions as regionmod
    matched = [case for case in store.all()
               if case.image == image and case.status in ("exact", "assumed") and case.recipe
               and case.best_digest and case.recipe != handasm.RECIPE]
    tally: dict[str, int] = {}
    for case in matched:
        tally[case.recipe] = tally.get(case.recipe, 0) + 1
    default = runner.label()
    candidates = [label for label, count in sorted(tally.items(), key=lambda item: (-item[1], item[0]))
                  if count >= regionmod.MIN_SUPPORT][:REGION_RECIPES]
    if len(candidates) < 2:
        return {"retried": 0, "recipes": candidates, "reason": "one recipe explains every match"}

    def dialect(label: str) -> str:
        pipeline = label.partition(" ")[0]
        return "ido" if pipeline.startswith("ido") else "gcc" if pipeline[:3] in ("kmc", "sn") else pipeline

    def spec_of(label: str):
        pipeline, _, flags = label.partition(" ")
        spec = split_flags(shlex.split(flags), pipeline)
        spec.section_bases.update(runner.spec.section_bases)
        spec.symbols.update(runner.spec.symbols)
        return spec

    matched.sort(key=lambda case: case.vram)
    step = max(1, -(-len(matched) // REGION_EVIDENCE_LIMIT))
    sampled = matched[::step]
    evidence: dict[int, set[str]] = {case.vram: {case.recipe} for case in matched}
    pairs = [(case, label) for case in sampled for label in candidates
             if label != case.recipe and dialect(label) == dialect(case.recipe)]

    def reproduces(pair) -> bool:
        case, label = pair
        text = store.source(case.best_digest)
        if text is None:
            return False
        try:
            evaluation, _code = runner.evaluator(case.vram, case.size, f"{case.name}.c", spec_of(label))(text)
        except Exception:                                    # noqa: BLE001 - no evidence from this pair
            return False
        return bool(evaluation.exact)

    builds = 0
    if pairs:
        with ThreadPoolExecutor(max(1, min(len(pairs), (args.jobs or 8)))) as pool:
            for (case, label), same in zip(pairs, pool.map(reproduces, pairs)):
                builds += 1
                if same:
                    evidence[case.vram].add(label)
    order = [default] + [label for label in candidates if label != default]
    found = regionmod.Regions.build(evidence, sorted(starts or []), None, order)
    summary = found.summary()
    rows = []
    for name, vram, size in todo:
        case = store.get(resolvemod.case_id(image, vram, size))
        if case is None or case.status not in ("open", "near", "does-not-build"):
            continue
        label = found.recipe_at(vram)
        if label is None or label == case.recipe or any(
                stage.get("stage") in ("region-recipe", "unit-recipe") and stage.get("recipe") == label
                for stage in case.stages):
            continue
        pipeline, _, flags = label.partition(" ")
        rows.append((name, vram, size, {"pipeline": pipeline, "flags": shlex.split(flags)}, label))
    if emit:
        emit({"event": "recipe-regions", "regions": summary["regions"], "evidence_builds": builds,
              "retry": len(rows)})
    return {**_retry_under_recipes(store, image, rows, target, runner, names, emit, starts, args, "region-recipe"),
            "regions": summary["regions"], "evidence_builds": builds, "matched_sampled": len(sampled),
            "_settled": lambda vram: found.recipe_at(vram) is not None}


def prepare_answer(text: str) -> str:
    """A refiner's answer as a buildable unit: the kit's typedef prelude in front, and the
    answer's own redefinitions of those basic types removed (a duplicate typedef is an error
    for these compilers)."""
    if "N64DK_M2C_PRELUDE" in text:
        return text
    return autodraft.PRELUDE + "\n" + _BASIC_TYPEDEF.sub("", text)


def _refine_pass(store, image: str, todo, target, runner, names, emit, starts, settings, args) -> dict:
    """Hand the cases the mechanical ladder left unresolved to the configured refiner
    (``n64dk.match.refine``), nearest first.  An answer that beats the case's best goes through
    the normal ladder, so it is searched, strictly verified and classified like any other
    source.  A case that already holds a ``refine`` stage is not asked about again."""
    from ..match import refine as refinemod
    settings.validate()
    signature = settings.signature()
    rows = []
    wanted = {n for item in (getattr(args, "refine_names", None) or []) for n in item.split(",") if n}
    for name, vram, size in todo:
        if wanted and name not in wanted:
            continue
        case = store.get(resolvemod.case_id(image, vram, size))
        if case is None or case.status not in settings.statuses or (settings.max_size and size > settings.max_size):
            continue
        if settings.max_score and (case.best_score is None or case.best_score > settings.max_score):
            continue
        if any(stage.get("stage") == "refine" and stage.get("signature") == signature
               for stage in case.stages) and not args.force:
            continue
        rows.append((case.best_score if case.best_score is not None else 1 << 30, size, name, vram))
    rows.sort()
    if settings.max_cases:
        rows = rows[:settings.max_cases]
    totals = {"attempted": 0, "improved": 0, "exact": 0, "assumed": 0, "calls": 0, "errors": 0,
              "known_source_answers": 0,
              "cost": {"calls": 0, "cached": 0, "seconds": 0.0, "prompt_bytes": 0, "answer_bytes": 0}}
    lock = threading.Lock()
    known = refinemod.KnownSource.from_directories(settings.known_source) if settings.known_source else None
    answer_cache = Path(store.root) / "refine_cache"
    from .. import mips

    def references(code: bytes, vram: int) -> set[int]:
        try:
            return {ref.address for ref in mips.scan_refs(code, vram)}
        except Exception:
            return set()

    # worked examples: every function of this image already identical from scratch, by recipe;
    # functions the refiner settles join them, so later cases learn from earlier ones
    examples: dict[str, refinemod.Examples] = {}

    def remember(case) -> None:
        if case is None or case.status not in ("exact", "assumed") or not case.best_digest or settings.examples <= 0:
            return
        text = store.source(case.best_digest)
        offset = case.vram - target.delta
        if not text or offset < 0 or offset + case.size > len(target.rom):
            return
        code = target.rom[offset:offset + case.size]
        if case.size > 0x140:
            return
        examples.setdefault(case.recipe or "", refinemod.Examples()).add(
            case.name, case.vram, case.size, references(code, case.vram),
            m2c_bridge.emit_asm(case.name, code, case.vram, names, []), text)

    if settings.examples > 0:
        for stored in store.all():           # not ``known``: that name holds the source guard
            if stored.image == image:
                remember(stored)

    def one(row):
        _score, size, name, vram = row
        case = store.get(resolvemod.case_id(image, vram, size))
        pipeline, _, flags = (case.recipe or "").partition(" ")
        spec = split_flags(shlex.split(flags), pipeline) if pipeline else runner.spec
        spec.section_bases.update(runner.spec.section_bases)
        spec.symbols.update(runner.spec.symbols)
        primary = runner.evaluator(vram, size, f"{name}.c", spec)
        offset = vram - target.delta
        code = target.rom[offset:offset + size]
        flow = runner.flow(vram, size)
        won: dict[str, tuple] = {}                       # answer digest -> (label, pipeline, evaluator)

        def evaluate(text: str):
            """The case's recipe first; an answer that is not exact there is also built under the
            sibling recipes, because the case's recipe is itself only the best guess so far."""
            evaluation, built = primary(text)
            if evaluation.exact or args.no_recipes:
                return evaluation, built
            from ..match import causes as causemod
            family = causemod.compiler_family(flow)
            family = family["family"] if family["confidence"] >= 0.8 else None
            others = list(runner.recipes(vram, size, f"{name}.c")(family))[:resolvemod.CROSS_RECIPES]
            if runner.label(spec) != runner.label(runner.spec):          # the configured default, if not tried yet
                others.insert(0, (runner.label(runner.spec), runner.spec.pipeline,
                                  runner.evaluator(vram, size, f"{name}.c", runner.spec)))
            for label, pipeline, other in others:
                attempt, attempt_built = other(text)
                better = attempt.score is not None and (evaluation.score is None or attempt.score < evaluation.score
                                                        or (attempt.exact and not evaluation.exact))
                if better:
                    evaluation, built = attempt, attempt_built
                    won[hashlib.sha256(text.encode()).hexdigest()] = (label, pipeline, other)
                if evaluation.exact:
                    break
            return evaluation, built

        best_source = store.source(case.best_digest) if case.best_digest else None
        best = primary(best_source)[0] if best_source else None
        failed_draft = None
        if best_source is None:
            # a draft that did not build still shows the recovered structure, and its compiler
            # output says what to fix
            kept = next((stage for stage in reversed(case.stages) if stage.get("source_digest")), None)
            text = store.source(kept["source_digest"]) if kept else None
            if text:
                failed_draft = (refinemod.Examples.body(text), refinemod._strip_paths(kept.get("error") or ""))
        tables = m2c_bridge.find_jump_tables(target.rom, target.delta, vram, size)
        assembly = m2c_bridge.emit_asm(name, code, vram, names, tables)
        declarations = ""
        if runner.rendered is not None and vram in runner.facts:
            declarations = runner.rendered.focused(runner.facts[vram], vram, True, typedefs=False)
        recipe_label = case.recipe or runner.label(spec)
        own = references(code, vram)

        def shown():
            pool = examples.get(recipe_label)
            return pool.pick(vram, own, settings.examples) if pool else []

        result = refinemod.refine(name, settings, evaluate, assembly, recipe_label, best_source,
                                  best, case.causes, prepare=prepare_answer, declarations=declarations, emit=emit,
                                  examples=shown, failed_draft=failed_draft, cache=answer_cache, known=known,
                                  tried=refinemod.tried_summary(case.stages))
        stage = {"stage": "refine", "signature": signature, "cost": result["cost"],
                 "known_source": result["known_source"],
                 "rounds": result["rounds"], "calls": result["calls"],
                 "history": result["history"], "improved": result["improved"], "errors": result["errors"][:2]}
        if result["improved"]:
            before = case.builds
            label, pipeline, chosen = won.get(hashlib.sha256(result["best_source"].encode()).hexdigest(),
                                              (case.recipe or runner.label(spec), spec.pipeline, primary))
            case = resolvemod.resolve(store, image, name, vram, code, chosen, source=result["best_source"],
                                      flow=flow, neighbours=starts, budget=args.budget, jobs=2,
                                      force=True, emit=emit, pipeline=pipeline, recipe_label=label, rom_offset=offset,
                                      strict=runner.strict(vram, size, f"{name}.c", starts),
                                      search_ceiling=getattr(args, "search_ceiling", resolvemod.SEARCH_CEILING),
                                      frame_fit=_enabled(args, "frame-fit"), search=_enabled(args, "search"),
                                      hand_forms=_enabled(args, "hand-forms"))
            case.builds += before
        case.stages.append(stage)
        store.put(case)
        remember(case)
        with lock:
            totals["attempted"] += 1
            totals["improved"] += bool(result["improved"])
            totals["calls"] += result["calls"]
            totals["errors"] += len(result["errors"])
            totals["known_source_answers"] += bool(result["known_source"])
            for key, value in result["cost"].items():
                totals["cost"][key] = round(totals["cost"][key] + value, 2)
            totals["exact"] += case.status == "exact" and bool(result["improved"])
            totals["assumed"] += case.status == "assumed" and bool(result["improved"])
        if emit:
            emit({"event": "refine-case", "name": name, "status": case.status, "score": case.best_score,
                  "improved": result["improved"]})
        return case

    with ThreadPoolExecutor(max(1, settings.jobs)) as pool:
        refined = [c for c in pool.map(one, rows)]
    totals["cases"] = {c.case_id: c for c in refined}
    return totals


def only_names(value) -> set[str]:
    """Function names from ``--only-names``: comma-separated names and/or ``@file`` (one per line)."""
    out: set[str] = set()
    for item in ([value] if isinstance(value, str) else list(value or [])):
        for part in str(item).split(","):
            part = part.strip()
            if part.startswith("@"):
                out.update(line.strip() for line in Path(part[1:]).read_text().split("\n") if line.strip())
            elif part:
                out.add(part)
    return out


WORKER_BATCH = 150


def worker_batch() -> int:
    """Functions one worker process resolves before it exits and a fresh one takes over
    (``N64DK_WORKER_BATCH``; 0 or less means one process per shard, the behaviour before recycling)."""
    text = os.environ.get("N64DK_WORKER_BATCH", "").strip()
    try:
        value = int(text) if text else WORKER_BATCH
    except ValueError:
        value = WORKER_BATCH
    return value if value > 0 else 10 ** 9


WORKER_BATCH_BYTES = 0x10000
LARGE_SERIAL_BYTES = 0x800


def _env_int(name: str, default: int) -> int:
    text = os.environ.get(name, "").strip()
    try:
        return int(text, 0) if text else default
    except ValueError:
        return default


def worker_batch_bytes() -> int:
    """Code bytes one worker process resolves before it is recycled (``N64DK_WORKER_BATCH_BYTES``; 0 or
    less: no byte limit).  Large functions leave the most memory behind, so their batches are shorter."""
    value = _env_int("N64DK_WORKER_BATCH_BYTES", WORKER_BATCH_BYTES)
    return value if value > 0 else 10 ** 12


def large_serial_bytes() -> int:
    """Functions larger than this run one at a time within a worker process (``N64DK_LARGE_SERIAL_BYTES``;
    0 or less: never)."""
    value = _env_int("N64DK_LARGE_SERIAL_BYTES", LARGE_SERIAL_BYTES)
    return value if value > 0 else 10 ** 12


def batches_of(shards, count: int, size_limit: int) -> list:
    """``(worker index, functions)`` batches, interleaved across shards in shard order: each batch ends at
    ``count`` functions or once its code reaches ``size_limit`` bytes (a batch always takes one function)."""
    split = []
    for index, shard in enumerate(shards):
        chunks, current, total = [], [], 0
        for function in shard:
            if current and (len(current) >= count or total + function[2] > size_limit):
                chunks.append(current)
                current, total = [], 0
            current.append(function)
            total += function[2]
        if current:
            chunks.append(current)
        split.append(chunks)
    return [(index, split[index][k]) for k in range(max((len(c) for c in split), default=0))
            for index in range(len(split)) if k < len(split[index])]


def worker_count(requested: int | None, functions: int, memory_per_worker_mb: int = 0,
                 memory_reserve_mb: int = 0) -> int:
    """Worker processes for a sweep.  ``0`` means automatic: one per two hardware threads (a
    worker is one interpreter plus its compiler children), but no more than free memory allows
    and never more than there are functions to share out."""
    if not hasattr(os, "fork"):
        return 1
    if requested is None or requested < 0:
        requested = 1
    if requested == 0:
        requested = max(1, (os.cpu_count() or 4) // 2)
        try:
            with open("/proc/meminfo") as handle:
                available = next(int(line.split()[1]) for line in handle if line.startswith("MemAvailable:")) // 1024
            # Scale defaults from current resources instead of encoding one workstation's peak.
            reserve = memory_reserve_mb or max(256, available // 8)
            per_worker = memory_per_worker_mb or max(256, min(1024, available // max(2, requested)))
            requested = max(1, min(requested, max(1, (available - reserve) // per_worker)))
        except (OSError, StopIteration, ValueError):
            pass
    return max(1, min(requested, functions // 2 or 1))


def _events(path: str | None):
    if not path:
        return None, None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle, lock = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644), threading.Lock()

    def emit(event: dict) -> None:
        payload = (json.dumps({"time": round(time.time(), 3), **event}) + "\n").encode()
        with lock:
            os.write(handle, payload)
    return emit, handle


@command("resolve-case", "Diagnose one unmatched function and climb the resolution ladder (draft, iterate, classify)")
def resolve_case(args):
    cfg, target, syms = _prepare(args)
    catalogue_vram = args.addr
    linked_vram = getattr(args, "_comparison_addr", catalogue_vram)
    coordinate_shift = catalogue_vram - linked_vram
    names = {address: name for name, address in syms.items() if name != "_gp"}
    name = args.name or names.get(linked_vram) or f"func_{catalogue_vram:08X}"
    store = resolvemod.CaseStore(_state(args, cfg))
    emit, handle = _events(args.events)
    try:
        with tempfile.TemporaryDirectory(prefix="n64dk-resolve-") as work:
            runner = _Runner(args, cfg, target, syms, Path(work))
            runner.functions_for_gate = ([(n, v - coordinate_shift, s) for n, v, s in load_catalogue(Path(args.catalogue))]
                                         if args.catalogue else [])
            if args.context == "auto":
                if not args.catalogue:
                    raise SystemExit("--context auto needs --catalogue")
                runner.auto_context([(n, v - coordinate_shift, s) for n, v, s in load_catalogue(Path(args.catalogue))],
                                    names, store.root, _image_label(args))
            offset = linked_vram - target.delta
            code = target.rom[offset:offset + args.size]
            source = Path(args.source).read_text() if args.source else None
            evaluate = runner.evaluator(linked_vram, args.size, Path(args.source).name if args.source else f"{name}.c")
            starts = ({v - coordinate_shift for _n, v, _s in load_catalogue(Path(args.catalogue))}
                      if args.catalogue else None)
            case = resolvemod.resolve(store, _image_label(args), name, linked_vram, code, evaluate, source=source,
                                      draft=None if args.no_draft else runner.drafter(name, linked_vram, args.size,
                                                                                      evaluate, names),
                                      flow=runner.flow(linked_vram, args.size), neighbours=starts,
                                      search_ceiling=getattr(args, "search_ceiling", resolvemod.SEARCH_CEILING),
                                      budget=args.budget, jobs=args.jobs or 8, force=args.force, emit=emit,
                                      pipeline=args.pipeline, recipe_label=runner.label(),
                                      rom_offset=linked_vram - target.delta,
                                      strict=runner.strict(linked_vram, args.size,
                                                           Path(args.source).name if args.source else f"{name}.c",
                                                           starts),
                                      recipes=None if args.no_recipes else runner.recipes(
                                          linked_vram, args.size, Path(args.source).name if args.source else f"{name}.c"))
        result = case.as_dict()
        if getattr(args, "_name_findings", None):
            result["name_findings"] = args._name_findings
        result["coordinates"] = {"image": _image_label(args),
                                 "catalogue_vram": f"0x{catalogue_vram:08X}",
                                 "linked_vram": f"0x{linked_vram:08X}",
                                 "size": args.size}
        if args.out and case.best_digest:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(store.source(case.best_digest) or "")
            result["written"] = args.out
        return result
    finally:
        store.close()
        if handle is not None:
            os.close(handle)


def _case_args(p):
    _common(p, rom_required=True)
    p.add_argument("source", nargs="?", help="existing candidate C file (omit to start from an m2c draft)")
    p.add_argument("--addr", type=_int, required=True)
    p.add_argument("--size", type=_int, required=True)
    p.add_argument("--name")
    _shared(p)
    p.add_argument("--out", help="write the best candidate here")


def _shared(p):
    p.add_argument("--pipeline", default="kmc+builtin")
    p.add_argument("--flags", default="-O2 -G0 -mips3 -mgp32 -mfp32")
    p.add_argument("--context", help="m2c context file, or `auto` to recover structs, globals and prototypes "
                   "from the image (needs --catalogue)")
    p.add_argument("--context-parts", default="structs,globals,prototypes")
    p.add_argument("--proto-confidence", type=float, default=0.9)
    p.add_argument("--catalogue", help="function catalogue: neighbouring starts for boundary evidence")
    p.add_argument("--budget", type=int, default=300, help="maximum builds per function")
    p.add_argument("--search-ceiling", type=int, default=resolvemod.SEARCH_CEILING,
                   help="skip the variant search when the best draft scores above this (0: always search)")
    p.add_argument("--state", help="case store directory (default: <project>/.n64dk/resolve)")
    p.add_argument("--events", help="append JSON-lines progress events to this file")
    p.add_argument("--no-draft", action="store_true", help="never generate an m2c draft")
    p.add_argument("--draft-portfolio", action="store_true",
                   help="build several drafts (plain, with context, alternative m2c renderings) and continue "
                        "from the closest")
    p.add_argument("--draft-options", default="",
                   help="comma-separated extra m2c renderings for the portfolio, e.g. --no-andor,--gotos-only")
    p.add_argument("--no-recipes", action="store_true", help="never try alternative compiler recipes")
    p.add_argument("--force", action="store_true", help="continue past boundary/ABI findings and redo exact cases")


resolve_case.arguments = _case_args


def processing_order(todo: list, order: str | None) -> list:
    """The order functions are attempted in: ``size`` (default, smallest first, which also balances
    worker shards), ``reverse`` or ``shuffle:SEED``.  The set is the same in every order; only the
    determinism checks use the others (every outcome must be the same whatever the order)."""
    if not order or order == "size":
        return todo
    if order == "reverse":
        return todo[::-1]
    if order.startswith("shuffle:"):
        import random
        shuffled = list(todo)
        random.Random(int(order.split(":", 1)[1])).shuffle(shuffled)
        return shuffled
    raise SystemExit(f"--order {order}: expected size, reverse or shuffle:SEED")


@command("resolve-hard", "Run the resolution ladder over every function without matching source and report the queue")
def resolve_hard(args):
    catalogue_functions = load_catalogue(Path(args.catalogue))
    # Give the canonical image resolver one real interval so a composite image obtains its
    # physical offset and catalogue-to-link transformation before the batch is constructed.
    probe_addr = probe_size = None
    if catalogue_functions:
        probe_addr, probe_size = catalogue_functions[0][1], catalogue_functions[0][2]
        args.addr, args.size = probe_addr, probe_size
    cfg, target, syms = _prepare(args)
    if probe_addr is not None:
        linked_probe = getattr(args, "_comparison_addr", probe_addr - args.catalogue_shift)
        coordinate_shift = probe_addr - linked_probe
    else:
        coordinate_shift = args.catalogue_shift
    names = {address: name for name, address in syms.items() if name != "_gp"}
    functions = [(n, v - coordinate_shift, s) for n, v, s in catalogue_functions]
    trimmed = {"functions": 0, "bytes": 0}
    if not getattr(args, "keep_padding", False):
        # A catalogue interval often runs to the next function and so includes the zero words
        # that align the next object.  No C produces them, so a function measured with them
        # can never be exact: the interval that is resolved ends at the code.
        from ..analysis import boundaries as boundmod
        fitted = []
        for name, vram, size in functions:
            offset = vram - target.delta
            padding = 0
            if size >= 12 and 0 <= offset <= len(target.rom) - size:
                try:
                    terminated, padding = boundmod._terminated(flowmod.analyse_in_image(target.rom, target.delta,
                                                                                         vram, size))
                except Exception:
                    terminated, padding = False, 0
                if not terminated or padding >= size:
                    padding = 0
            if padding:
                trimmed["functions"] += 1
                trimmed["bytes"] += padding
            fitted.append((name, vram, size - padding))
        functions = fitted
    starts = {v for _n, v, _s in functions}
    covered, matched_rows = [], []
    for item in args.manifest:
        path, _, pipeline = item.partition("=")
        for line in Path(path).read_text().splitlines():
            parts = line.split("\t")
            if not line.startswith("#") and len(parts) >= 3:
                try:
                    covered.append((int(parts[0], 0), int(parts[1], 0)))
                except ValueError:
                    continue
                if pipeline and len(parts) >= 4 and not parts[2].endswith((".s", ".S")):
                    row = split_flags(shlex.split(parts[3]), pipeline)
                    matched_rows.append((int(parts[0], 0) + target.delta, int(parts[1], 0), row))
    inside = [f for f in functions if f[2] >= args.min_size and 0 <= f[1] - target.delta <= len(target.rom) - f[2]]
    if args.holdout > 1:
        # benchmark mode: every Nth function is treated as unmatched, whatever the manifests say,
        # and its own manifest row is withheld from the recipe map
        todo = sorted(inside, key=lambda f: f[1])[::args.holdout]
        held = {f[1] for f in todo}
        matched_rows = [r for r in matched_rows if not any(r[0] <= v < r[0] + r[1] for v in held)]
    else:
        todo = [f for f in inside if not any(a <= f[1] - target.delta < a + n for a, n in covered)]
    skip = set(getattr(args, "skip_vrams", None) or ())          # e.g. recovered by an earlier stage
    todo = [f for f in todo if f[1] not in skip]
    # every function this pass leaves out is written down with the reason (``scope_<image>.json``
    # beside the case store): a function is never left out silently
    left_out: dict[int, str] = {}

    def keep(rows, kept, reason):
        kept_vrams = {f[1] for f in kept}
        for f in rows:
            if f[1] not in kept_vrams:
                left_out.setdefault(f[1], reason)
        return kept
    chosen_names = only_names(getattr(args, "only_names", None))
    if chosen_names:                                             # a named set: nothing else is attempted
        todo = keep(todo, [f for f in todo if f[0] in chosen_names], "not in the --only-names set")
    todo.sort(key=lambda f: f[2])
    ceiling = implausible_size()
    if ceiling:
        todo = keep(todo, [f for f in todo if f[2] <= ceiling],
                    f"implausibly large for one function (over 0x{ceiling:X} bytes): most likely a boundary error "
                    "that merged code or data; kept as assembly (N64DK_IMPLAUSIBLE_FUNCTION_BYTES changes the ceiling)")
    if args.max_size:
        todo = keep(todo, [f for f in todo if f[2] <= args.max_size], f"larger than the run's --max-size 0x{args.max_size:X}")
    if args.every > 1:
        todo = keep(todo, todo[::args.every], f"sampled out (every {args.every}th function)")
    if args.limit:
        todo = keep(todo, todo[:args.limit], f"beyond the run's --limit {args.limit}")
    todo = processing_order(todo, getattr(args, "order", None))
    store = resolvemod.CaseStore(_state(args, cfg))
    # always observing (predictions and outcomes recorded); changing the search only with
    # --search-economics, the measured gain of which was small (round 10: -2.7% builds)
    from ..match.economics import Governor
    governor = Governor(Path(store.root) / f"economics_{_image_label(args)}.json",
                        active=bool(getattr(args, "search_economics", False)))
    try:
        # (a later pass over a named set, ``write_scope=False``, keeps the sweep's scope file)
        if getattr(args, "write_scope", True):
            scope = Path(store.root) / f"scope_{_image_label(args)}.json"
            scope.write_text(json.dumps({"attempted": len(todo), "left_out": [
                {"vram": f"0x{f[1]:08X}", "name": f[0], "size": f[2], "reason": left_out[f[1]]}
                for f in sorted(inside, key=lambda f: f[1]) if f[1] in left_out]}, indent=0))
    except OSError:
        pass
    # a resumed sweep (decompile-run): the selection above is the original run's, unchanged; the
    # functions it already finished are then not attempted again
    finished = set(getattr(args, "resume_finished", None) or ())
    if finished:
        todo = [f for f in todo if f[1] not in finished]
    emit, handle = _events(args.events)
    image, started = _image_label(args), time.time()
    only = set(getattr(args, "only_statuses", None) or ())
    if only:                                                  # a pass over stored cases of these statuses only
        def wanted(function):
            case = store.get(resolvemod.case_id(image, function[1], function[2]))
            return case is not None and case.status in only
        todo = [f for f in todo if wanted(f)]
    inner = 4 if (args.jobs or 8) >= 8 else 2
    try:
        with tempfile.TemporaryDirectory(prefix="n64dk-resolve-") as work:
            runner = _Runner(args, cfg, target, syms, Path(work))
            runner.functions_for_gate = functions
            context_info = runner.auto_context(functions, names, store.root, image)
            recipe_info = runner.learn_recipes(functions, matched_rows)
            try:
                unknown_info = runner.find_unknown(functions)
            except Exception:                                # noqa: BLE001 - a hint, never a reason to stop
                runner.unknown, unknown_info = set(), None
            if unknown_info and emit:
                emit({"event": "unknown-toolchain", **unknown_info})

            large_turn = threading.Semaphore(1)        # forked workers each get their own, never held at fork

            def one(function, store=store, emit=emit, inner=inner):
                name, vram, size = function
                if size > large_serial_bytes():
                    # one large function at a time per process: a 4-7 KB function alone peaks near 500 MB
                    # (measured 2026-11-18), and two at once on each of two workers took jobs over 3 GB
                    with large_turn:
                        return _one(function, store, emit, inner)
                return _one(function, store, emit, inner)

            def _one(function, store, emit, inner):
                name, vram, size = function
                offset = vram - target.delta
                spec = runner.spec_for(vram)
                evaluate = runner.evaluator(vram, size, f"{name}.c", spec)
                try:
                    case = resolvemod.resolve(store, image, name, vram, target.rom[offset:offset + size], evaluate,
                                              draft=None if args.no_draft else runner.drafter(name, vram, size,
                                                                                              evaluate, names),
                                              flow=runner.flow(vram, size), neighbours=starts, budget=args.budget,
                                              search_ceiling=(1 if ((vram in runner.unknown and
                                                                    getattr(args, "skip_unknown_search", False)) or
                                                                   size > LARGE_FUNCTION) else
                                                              getattr(args, "search_ceiling",
                                                                      resolvemod.SEARCH_CEILING)),
                                              cross_recipes=getattr(args, "cross_recipes", None),
                                              frame_fit=_enabled(args, "frame-fit"), search=_enabled(args, "search"),
                                      hand_forms=_enabled(args, "hand-forms"),
                                              rewrites=runner.rewriter(name, vram, size),
                                              drafts_x_recipes=_enabled(args, "cross-recipes"),
                                              jobs=inner, force=args.force, emit=emit, pipeline=spec.pipeline,
                                              recipe_label=runner.label(spec), rom_offset=offset,
                                              strict=runner.strict(vram, size, f"{name}.c", starts),
                                              recipes=(None if args.no_recipes else
                                                       runner.debug_twin(vram, size, f"{name}.c") if size > LARGE_FUNCTION
                                                       else runner.recipes(vram, size, f"{name}.c")),
                                              governor=governor, fresh=bool(getattr(args, "fresh", False)),
                                              provisional=bool(getattr(args, "provisional", False)),
                                              wide_recipes=((lambda vram=vram: runner.unoptimised_near(vram))
                                                            if getattr(args, "recipe_gate", False) else None),
                                              stratum=(f"{image} | {runner.label(spec)} | "
                                                       f"{'unknown toolchain' if vram in runner.unknown else 'known toolchain'}"))
                    note = runner.translator_notes.pop(vram, None)
                    if note and case is not None and case.status not in ("exact", "assumed"):
                        # what the unoptimised-code translator made of the function: the reason code names it
                        case.stages = [s for s in case.stages if s.get("stage") != "translator"] + [{"stage": "translator", **note}]
                        store.put(case)
                    return case
                except Exception as error:                   # one bad function must not stop the sweep
                    if emit:
                        emit({"event": "case-error", "name": name, "error": str(error)[-300:]})
                    return None

            workers = worker_count(getattr(args, "workers", 1), len(todo),
                                   getattr(args, "worker_memory_mb", 0),
                                   getattr(args, "memory_reserve_mb", 0))
            worker_errors = 0
            if getattr(args, "skip_sweep", False):
                # a later pass that only asks the refiner: the stored cases stand as they are
                cases = [c for c in (store.get(resolvemod.case_id(image, vram, size)) for _n, vram, size in todo)
                         if c is not None]
            elif workers > 1:
                # The search generates its variants in Python, so threads share one interpreter
                # lock.  Each worker process takes every Nth function (the list is sorted by size,
                # so shards are balanced), with its own store connection, event handle and work
                # directory; cases meet in the store, which is opened in WAL mode.
                store.close()
                # threads per worker are oversubscribed: most of their time is spent waiting for a
                # compiler process or for the interpreter lock
                share = max(4, (3 * (args.jobs or 8)) // (2 * workers))
                # Workers are recycled: each process takes one batch of its shard and exits, and a fresh
                # process takes the next.  A worker's memory grows with the work it has done and only an
                # exit returns it (measured 2026-11-18: two workers reached ~1.5 GB each on their largest
                # functions and took six titles over a 3 GB job cap; traced Python objects stayed under
                # 400 MB, malloc arenas and the disassembler were ruled out).  At most ``workers`` run at once.
                batch = worker_batch()
                shards = [todo[index::workers] for index in range(workers)]
                pending = batches_of(shards, batch, worker_batch_bytes())
                running: dict[int, int] = {}
                from .. import safety

                def launch(index, chunk):
                    pid = os.fork()
                    if pid == 0:
                        status = 1
                        try:
                            safety.forked()
                            own_store = resolvemod.CaseStore(_state(args, cfg))
                            own_emit, _own_handle = _events(args.events)
                            runner.work = Path(work) / f"worker{index}"
                            runner.work.mkdir(parents=True, exist_ok=True)
                            own_inner = 2
                            with ThreadPoolExecutor(max(2, share // own_inner)) as pool:
                                list(pool.map(lambda f: one(f, own_store, own_emit, own_inner), chunk))
                            own_store.close()
                            status = 0
                        finally:
                            safety.flush_usage()              # the run's profile adds up its tools and builds
                            os._exit(status)                  # never run the parent's cleanup twice
                    running[pid] = index
                    safety.register_child(pid)                # SIGTERM reaches the worker too

                batches = len(pending)
                try:
                    while pending or running:
                        while pending and len(running) < workers:
                            launch(*pending.pop(0))
                        reaped = False
                        for pid in list(running):
                            done, status = os.waitpid(pid, os.WNOHANG)
                            if done:
                                running.pop(pid)
                                safety.unregister_child(pid)
                                worker_errors += status != 0
                                reaped = True
                        if not reaped:
                            time.sleep(0.2)
                except BaseException:
                    for pid in running:
                        try:
                            os.kill(pid, 15)
                        except ProcessLookupError:
                            pass
                    for pid in running:
                        try:
                            os.waitpid(pid, 0)
                        except ChildProcessError:
                            pass
                    if emit:
                        emit({"event": "workers-cancelled", "count": len(running)})
                    raise
                if emit:
                    emit({"event": "worker-batches", "workers": workers, "batch": batch, "batches": batches})
                store = resolvemod.CaseStore(_state(args, cfg))
                if worker_errors and emit:
                    emit({"event": "worker-failed", "count": worker_errors})
                cases = [c for c in (store.get(resolvemod.case_id(image, vram, size)) for _n, vram, size in todo)
                         if c is not None]
            else:
                with ThreadPoolExecutor(max(1, (args.jobs or 8) // inner)) as pool:
                    cases = [c for c in pool.map(one, todo) if c is not None]
            twins = (_twin_pass(store, image, todo, target, runner, names, emit, starts)
                     if _enabled(args, "twin") else [])
            if twins:
                replaced = {c.case_id for c in twins}
                cases = [c for c in cases if c.case_id not in replaced] + twins
            unit_info = None
            region_enabled = bool(getattr(args, "region_recipes", False))
            unit_enabled = bool(getattr(args, "unit_recipes", False))
            if (region_enabled or unit_enabled) and not only and not args.limit and not args.no_draft:
                # regions first (a recipe that explains a whole stretch), then the nearest proven
                # neighbour inside a closed object for what the regions left
                region_info = None
                if region_enabled:
                    try:
                        region_info = _region_recipe_pass(store, image, todo, target, runner, names, emit, starts, args)
                    except Exception as error:                # noqa: BLE001 - a retry pass must not cost the sweep
                        region_info = {"error": f"{type(error).__name__}: {error}"[-300:]}
                if region_info:
                    retried = region_info.pop("cases", {})
                    cases = [c for c in cases if c.case_id not in retried] + list(retried.values())
                # (measured: where a region's recipe is known, the nearest neighbour's adds nothing)
                settled = region_info.pop("_settled", None) if region_info else None
                if unit_enabled:
                    unit_info = _unit_recipe_pass(store, image, todo, target, runner, names, emit, starts, args, settled)
                    retried = unit_info.pop("cases", {})
                    cases = [c for c in cases if c.case_id not in retried] + list(retried.values())
                    unit_info["regions"] = region_info
                elif region_info is not None:
                    unit_info = {"regions": region_info, "retried": 0,
                                 "reason": "nearest-unit recipe retry disabled"}
            refine_info = None
            from ..match import refine as refinemod
            settings = getattr(args, "refine", None) or refinemod.Settings.from_table(
                None, getattr(args, "refine_command", None))
            if settings is not None:
                for key in ("rounds", "max_cases", "jobs", "max_score"):
                    value = getattr(args, f"refine_{key}", None)
                    if value is not None:
                        setattr(settings, key, value)
                refine_info = _refine_pass(store, image, todo, target, runner, names, emit, starts, settings, args)
                refined = refine_info.pop("cases")
                cases = [c for c in cases if c.case_id not in refined] + list(refined.values())
        elapsed = time.time() - started
        economics = None
        if governor is not None:
            governor.save()
            ranking = governor.ranking()
            economics = {"strata": len(ranking), "plateaued": [row["stratum"] for row in ranking if row["plateaued"]],
                         "searches_skipped": sum(row["skipped"] for row in ranking),
                         "builds_saved_estimate": sum(row["builds_saved_estimate"] for row in ranking),
                         "ranking": ranking[:12]}
        queue = resolvemod.hard_queue(cases)
        exact = [c for c in cases if c.status == "exact"]
        return {"image": image, "functions": len(todo), "resolved_cases": len(cases), "statuses": queue["statuses"],
                "exact": [{"name": c.name, "size": c.size, "source": c.best_digest, "recipe": c.recipe,
                           "strict": (c.strict or {}).get("verdict")} for c in exact],
                "strict": {v: sum(1 for c in exact if (c.strict or {}).get("verdict") == v)
                           for v in sorted({(c.strict or {}).get("verdict") for c in exact}, key=str)},
                "exact_bytes": sum(c.size for c in exact), "builds": sum(c.builds for c in cases),
                "seconds": round(elapsed, 1),
                "exact_per_hour": round(len(exact) * 3600 / elapsed, 1) if elapsed else None,
                "context": context_info, "recipes": recipe_info, "twins": len(twins),
                "workers": workers, "worker_errors": worker_errors, "refine": refine_info, "economics": economics,
                "unit_recipes": unit_info, "unknown_toolchain": unknown_info,
                "padding_trimmed": trimmed,
                "queue": [{k: g[k] for k in ("cause", "meaning", "functions", "bytes", "expected_progress")}
                          | {"best": g["best"][:args.show]} for g in queue["queue"]],
                "state": str(store.root)}
    finally:
        store.close()
        if handle is not None:
            os.close(handle)


def _hard_args(p):
    _common(p, rom_required=True)
    p.add_argument("--catalogue", required=True)
    p.add_argument("--catalogue-shift", type=_int, default=0)
    p.add_argument("--manifest", action="append", default=[], metavar="FILE[=PIPELINE]",
                   help="manifest of already matched code (repeatable); with =PIPELINE its rows also teach the "
                        "per-unit recipe map")
    p.add_argument("--region-recipes", action="store_true",
                   help="experimental: segment the image into recipe regions and retry under each region's recipe "
                        "(measured 3/126; off by default)")
    p.add_argument("--unit-recipes", action="store_true",
                   help="experimental: retry from a nearby unit's recipe (measured 0/165; off by default)")
    p.add_argument("--skip-unknown-search", action="store_true",
                   help="experimental: suppress variant search in regions classified as an unknown toolchain; "
                        "classification remains advisory by default")
    p.add_argument("--search-economics", action="store_true",
                   help="stop the variant search in strata whose searches have returned nothing (n64dk.match.economics)")
    p.add_argument("--only-names", action="append",
                   help="attempt only these functions: comma-separated names or @FILE (one per line); repeatable")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--order", default=None, help="attempt order: size (default), reverse, shuffle:SEED "
                                                 "(determinism checks; outcomes must not depend on it)")
    p.add_argument("--max-size", type=_int, default=0)
    p.add_argument("--min-size", type=_int, default=4)
    p.add_argument("--every", type=int, default=1, help="take every Nth function (sampling for benchmarks)")
    p.add_argument("--cross-recipes", type=int,
                   help="alternative recipes tried on the best drafts when none is exact (default: "
                        f"{resolvemod.CROSS_RECIPES}; a calibration sample uses many more)")
    p.add_argument("--keep-padding", action="store_true",
                   help="resolve catalogue intervals as given, including trailing alignment padding")
    p.add_argument("--refine-command",
                   help="external author for the cases the ladder leaves unresolved: a command that reads "
                        "a prompt on stdin and prints C (see n64dk.match.refine); every answer is built and "
                        "strictly verified")
    p.add_argument("--refine-rounds", type=int, help="rounds of feedback per function (default 3)")
    p.add_argument("--refine-names", action="append", default=[],
                   help="ask only about these functions (comma separated, repeatable)")
    p.add_argument("--refine-max-cases", type=int, help="ask about at most this many cases, nearest first")
    p.add_argument("--refine-max-score", type=int, help="ask only about cases at most this far from the target")
    p.add_argument("--refine-jobs", type=int, help="refiner calls in parallel (default 1)")
    p.add_argument("--workers", type=int, default=1,
                   help="worker processes sharing the sweep (0: automatic, from hardware threads and free "
                        "memory); --jobs is divided among them")
    p.add_argument("--worker-memory-mb", type=int, default=0,
                   help="automatic-worker memory estimate (0: derive from currently available memory)")
    p.add_argument("--memory-reserve-mb", type=int, default=0,
                   help="memory held outside workers (0: one eighth of currently available memory)")
    p.add_argument("--holdout", type=int, default=0,
                   help="benchmark: treat every Nth catalogued function as unmatched even when a manifest "
                        "covers it, withholding its own row from the recipe map")
    p.add_argument("--show", type=int, default=3)
    p.add_argument("--pipeline", default="kmc+builtin")
    p.add_argument("--flags", default="-O2 -G0 -mips3 -mgp32 -mfp32")
    p.add_argument("--context", help="m2c context file, or `auto`")
    p.add_argument("--context-parts", default="structs,globals,prototypes")
    p.add_argument("--proto-confidence", type=float, default=0.9)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--search-ceiling", type=int, default=resolvemod.SEARCH_CEILING,
                   help="skip the variant search when the best draft scores above this (0: always search)")
    p.add_argument("--state")
    p.add_argument("--events")
    p.add_argument("--no-draft", action="store_true")
    p.add_argument("--no-recipes", action="store_true")
    p.add_argument("--draft-portfolio", action="store_true",
                   help="build several drafts (plain, with context, alternative m2c renderings) and continue "
                        "from the closest")
    p.add_argument("--draft-options", default="",
                   help="comma-separated extra m2c renderings for the portfolio, e.g. --no-andor,--gotos-only")
    p.add_argument("--force", action="store_true")


resolve_hard.arguments = _hard_args


@command("hard-queue", "Group stored resolution cases by cause and rank them by expected progress")
def hard_queue(args):
    store = resolvemod.CaseStore(Path(args.state))
    try:
        cases = store.all()
        if args.image_name:
            cases = [c for c in cases if c.image == args.image_name]
        report = resolvemod.hard_queue(cases)
        for group in report["queue"]:
            group["best"] = group["best"][:args.show]
        return report
    finally:
        store.close()


def _queue_args(p):
    p.add_argument("--state", required=True, help="case store directory")
    p.add_argument("--image-name")
    p.add_argument("--show", type=int, default=5)


hard_queue.arguments = _queue_args


@command("resolve-export", "Write exact resolution cases as an integration queue (sources, manifest rows, assumptions)")
def resolve_export(args):
    store = resolvemod.CaseStore(Path(args.state))
    try:
        return resolvemod.export_queue(store, Path(args.out), args.image_name, not args.strict_only)
    finally:
        store.close()


def _export_args(p):
    p.add_argument("--state", required=True, help="case store directory")
    p.add_argument("--out", required=True, help="output directory (never inside a project source tree)")
    p.add_argument("--image-name")
    p.add_argument("--strict-only", action="store_true", help="leave out cases that still rest on assumptions")


resolve_export.arguments = _export_args
