"""Proof receipts on the command line (``n64dk.match.proofreceipt``).

``issue-receipts RUN``     one receipt per settled result of a finished run (strictly exact C, and
                           hand-written assembly; ``--include-assumed`` adds byte-identical results
                           that wait for a symbol), written to ``RUN/receipts`` (opt-in: nothing
                           else writes receipts).
``verify-receipt RECEIPT`` replay one receipt.
``verify-run RUN``         replay the receipt of every settled result of a run.
``verify-export EXPORT``   replay the receipt of every matched ``FUNCTIONS.tsv`` row whose source
                           is in the export.

Replay recomputes every digest from the files on disk, the image supplied locally and the local
toolchain, rebuilds each source with its recorded recipe in a fresh directory and verifies it
strictly again.  ``--static`` checks the digests without building (reported apart, never as
replayed).  Each verify command exits 1 when any result is rejected or has no receipt.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..cli import command, emit
from ..match import proofreceipt as receipts

SETTLED = ("exact", "assembly")
EXPORT_OUTCOMES = {"exact": "exact", "assembly": "assembly", "awaiting-symbols": "assumed"}


def _recipe(spec) -> str:
    return f"{spec.pipeline} {' '.join(spec.flags)}".strip() if spec is not None else ""


class Contexts:
    """One :class:`proofreceipt.Context` per image, from a project and (optionally) the run or the
    export whose declarations are replayed.

    Declarations come from where the results were verified.  For a run: the project's symbol files
    (``[run] symbols``, else ``[match] symbols``, else the conventional files; a later file's value
    wins, as ``symbols.load`` gives it), then the run's recovered ``_gp`` and its declared
    self-describing addresses, which never replace a project value (as the run adds them).  For an
    export: the files under ``EXPORT/symbols``.  ``extra`` files are added last, never replacing."""

    def __init__(self, project, *, run=None, export=None, toolchain=None, extra=(), builder=None):
        from ..match.toolchain import load_config
        self.run = Path(run).resolve() if run else None
        self.export = Path(export).resolve() if export else None
        self.invocation = {}
        if self.run is not None and (self.run / "solver" / "invocation.json").is_file():
            self.invocation = json.loads((self.run / "solver" / "invocation.json").read_text())
        project = project or self.invocation.get("project")
        if not project:
            raise SystemExit("no project: pass --project (a run made by this version records its own)")
        self.project = Path(project).resolve()
        toolchain = toolchain or self.invocation.get("toolchain_dir") or None
        self.cfg = load_config(self.project, None, toolchain)
        self.extra = [Path(path) for path in extra]
        self.builder = builder
        self._made: dict[str, receipts.Context] = {}

    def settings(self, image: str) -> dict:
        from .run import _settings
        try:
            return _settings(self.project, image)
        except (OSError, ValueError):
            return {}

    def base(self, image: str):
        from ..match.pipeline import split_flags
        settings = self.settings(image)
        pipeline = settings.get("pipeline", self.invocation.get("pipeline"))
        flags = settings.get("flags", self.invocation.get("flags"))
        if not pipeline:
            return None
        return split_flags(shlex.split(flags or ""), pipeline)

    def symbol_files(self, image: str) -> tuple[list[Path], list[Path]]:
        """(files read in order, files that only add names not yet declared)"""
        from ..match import symbols as symmod
        if self.export is not None:
            folder = self.export / "symbols"
            files = sorted(folder.glob("symbol_addrs*.txt")) + [folder / "sdk_symbols.txt", folder / "gp_symbols.txt",
                                                                 folder / f"declared_{image}.txt"]
            extras = []
        else:
            files = [self.project / name for name in (self.settings(image).get("symbols") or [])]
            files = files or list(self.cfg.symbols) or list(symmod.discover(self.project))
            extras = ([self.run / "small_data" / "gp_symbols.txt", self.run / image / "declared_symbols.txt"]
                      if self.run is not None else [])
        keep = lambda paths: [Path(path) for path in paths if Path(path).is_file()]    # noqa: E731
        return keep(files), keep([*extras, *self.extra])

    def get(self, image: str) -> receipts.Context:
        if image in self._made:
            return self._made[image]
        from ..match import images as imagemod
        from ..match import symbols as symmod
        spec = imagemod.resolve(self.project, image)
        data, delta = spec.load(), spec.delta
        files, extras = self.symbol_files(image)
        symbols = symmod.load(files)
        for path in extras:
            for name, address in symmod.load([path]).items():
                if not dict.__contains__(symbols, name):
                    symbols[name] = address
        rows = spec.rows()
        catalogue = {vram: size for _name, vram, size in rows} if rows else None
        self._made[image] = receipts.Context(
            project=self.project, image=image, data=data, delta=delta, symbols=symbols, symbol_files=files + extras,
            cfg=self.cfg, base=self.base(image), catalogue=catalogue,
            function_starts=set(catalogue) if catalogue else None, builder=self.builder)
        return self._made[image]


def _spec_for(case, ctx: receipts.Context, text: str | None = None):
    """The recipe a run's strict check built the case with (commands.resolve ``strict``).  An
    assembly source is proven by handasm with the names that state their own address
    (``L_8000A174``, ``jtbl_80010CD0``) resolved to that address; the receipt declares those
    as symbol pins, so the replay relocates exactly as the run's proof did, and a changed pin
    changes the receipt's digest.  Replay accepts such a pin only when the source still states
    it (``receipts.stated_pins``, the same rule)."""
    from ..match.pipeline import split_flags
    recipe = case.recipe or _recipe(ctx.base)
    pipeline, _, flags = recipe.partition(" ")
    spec = split_flags(shlex.split(flags), pipeline)
    if ctx.base is not None:
        spec.section_bases.update(ctx.base.section_bases)
        spec.symbols.update(ctx.base.symbols)
    if text is not None and _suffix(case) == ".s":
        for name, address in receipts.stated_pins(text, ctx.symbols).items():
            spec.symbols.setdefault(name, address)          # a recipe or run pin keeps its value
    return spec


def _suffix(case) -> str:
    from ..match import handasm
    return ".s" if case.status == handasm.STATUS else ".c"


def _run_cases(run: Path) -> tuple[list, dict]:
    """Every settled or byte-identical case of a run (status exact, assembly or assumed), and its source."""
    from ..match import resolve as resolvemod
    store = resolvemod.CaseStore(run / "state", readonly=True)
    try:
        cases = [case for case in store.all() if case.status in (*SETTLED, "assumed")]
        texts = {case.case_id: store.source(case.best_digest) for case in cases}
    finally:
        store.close()
    return sorted(cases, key=lambda case: (case.image, case.vram)), texts


def _orphans(folder: Path, known: set[Path]) -> list[str]:
    """Receipts for which there is no such result: a result that changed or went away."""
    if not folder.is_dir():
        return []
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*.json") if path.resolve() not in known)


def issue_receipts(run, *, contexts: Contexts | None = None, out=None, include_assumed: bool = False,
                   export=None, issued_at: str | None = None, donors: bool = True) -> dict:
    """Write a receipt for every settled result of a finished run into ``RUN/receipts`` (or
    ``out``), and into ``EXPORT/receipts`` when an export is named.  Each result is built again
    (fresh directory) and verified strictly; a result the strict verifier no longer calls
    byte-identical gets no receipt and is listed."""
    from .export import _origin
    run = Path(run).resolve()
    contexts = contexts or Contexts(None, run=run)
    out = Path(out) if out else run / "receipts"
    cases, texts = _run_cases(run)
    cases = [case for case in cases if include_assumed or case.status != "assumed"]
    issued, refused, classes, proofs = 0, [], {}, 0
    for case in cases:
        label = f"{case.image}:{case.name}"
        text = texts.get(case.case_id)
        if text is None:
            refused.append({"result": label, "reason": "source missing from the run's store"})
            continue
        try:
            ctx = contexts.get(case.image)
            with tempfile.TemporaryDirectory(prefix="n64dk-issue-") as tmp:
                source = Path(tmp) / f"{case.name}{_suffix(case)}"
                source.write_text(text)
                receipt = receipts.issue(ctx, name=case.name, vram=case.vram, size=case.size, source=source,
                                         spec=_spec_for(case, ctx, text), origin=_origin(case), issued_at=issued_at)
        except Exception as error:                          # noqa: BLE001 - one result never stops the others
            refused.append({"result": label, "reason": f"{type(error).__name__}: {str(error)[-200:]}"})
            continue
        path = receipts.write(receipt, receipts.receipt_path(out, case.image, case.name, case.vram))
        if export is not None:
            target = receipts.receipt_path(Path(export) / "receipts", case.image, case.name, case.vram)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        issued += 1
        cls = receipt["verdict"]["class"]
        classes[cls] = classes.get(cls, 0) + 1
        proofs += int(receipt["claims"]["strict_proof"])
    summary = {"results": len(cases), "issued": issued, "classes": classes, "strict_proof": proofs,
               "refused": refused[:50], "refused_count": len(refused), "out": str(out)}
    if donors:
        summary["donors"] = issue_donor_receipts(run, contexts=contexts, out=out, include_assumed=include_assumed,
                                                 export=export, issued_at=issued_at)
    return summary


# --------------------------------------------------------------------------------------------- donors
# A donor unit (one translation unit of a source tree, an SDK or another title, transplanted by
# the donor stage) is held to the same proof contract as the run's own C: the same fresh build,
# the same strict comparison, the same receipt.  Its recipe is its manifest row; each address the
# row pins carries the origin the donor stage recorded (transplant.classify_pin); a pin without a
# recorded origin, or one only the unit's own bytes support, never makes a strict proof.

DONOR_HEADER = re.compile(r"^#\s*(\S+) rows proposed by donor transplant from (\S+?);")
_DONOR_NAME = re.compile(r"[A-Za-z0-9_.+-]+")


EXPORT_ROOTS = ("<home>", "<run>", "<project>")      # export.py's path replacements


def filed_path(unit: str) -> Path | None:
    """Where a manifest row's SOURCE is filed in ``src/<donor>/``: the path as written when it is
    relative; a unit kept outside its donor's root is written as an absolute path and filed by its
    last two components (transplant.write_outputs).  None for a path that would leave the tree
    (``..``) or names nothing."""
    path = Path(unit)
    if path.is_absolute() or (path.parts and path.parts[0] in EXPORT_ROOTS):
        # an export writes an absolute path with its root replaced (``<home>/...``): still absolute
        path = Path(*path.parts[1:][-2:]) if len(path.parts) > 1 else Path()
    if not path.parts or path.is_absolute() or ".." in path.parts:
        return None
    return path


def _inside(tree: Path, filed: Path) -> bool:
    """``tree / filed`` is in ``tree`` once links are followed (a link to a file elsewhere is a
    source the tree does not hold)."""
    try:
        return (tree / filed).resolve().is_relative_to(tree.resolve())
    except (OSError, RuntimeError):
        return False


@dataclass
class DonorUnit:
    image: str
    donor: str
    unit: str                 # path inside the donor tree (src/<donor>/...)
    pipeline: str
    flags: str
    rom_start: int
    size: int
    waiting: bool             # identical, awaiting declarations (a proposed_awaiting_* manifest)

    @property
    def name(self) -> str:
        return Path(self.unit).stem

    @property
    def recipe(self) -> str:
        return f"{self.pipeline} {self.flags}".strip()

    @property
    def filed(self) -> Path | None:
        """The source's path inside ``src/<donor>/`` (:func:`filed_path`); None when the donor name
        or the SOURCE would leave the tree the manifest folder holds."""
        if not _DONOR_NAME.fullmatch(self.donor) or self.donor in (".", ".."):
            return None
        return filed_path(self.unit)

    @property
    def earned_by(self) -> str:
        if self.donor.startswith("sdk_"):
            return "sdk-donor"
        if self.donor.startswith("tree_"):
            return "source-tree-donor"
        return "cross-title-donor"


def donor_units(folder: Path, image: str) -> list[DonorUnit]:
    """Every row of the donor manifests in ``folder`` (``RUN/IMAGE/donors`` or ``EXPORT/donors/IMAGE``)."""
    out = []
    for path in sorted(Path(folder).glob("proposed_*.tsv")):
        lines = path.read_text().splitlines()
        found = DONOR_HEADER.match(lines[0]) if lines else None
        if not found:
            continue
        pipeline, donor = found.group(1), found.group(2)
        waiting = path.name.startswith("proposed_awaiting_")
        if waiting and donor.startswith("awaiting_"):
            donor = donor[len("awaiting_"):]
        for line in lines[1:]:
            if not line.strip() or line.startswith("#"):
                continue
            cells = line.split("\t")
            if len(cells) < 3:
                continue
            out.append(DonorUnit(image, donor, cells[2], pipeline, cells[3] if len(cells) > 3 else "",
                                 int(cells[0], 16), int(cells[1], 16), waiting))
    return out


def _donor_origins(run_folder: Path) -> dict[int, dict]:
    """target vram -> the pin origins the donor stage recorded (a run's ``report.json`` rows, or the
    export's ``pin_origins.json``); empty when neither exists."""
    exported = Path(run_folder) / "pin_origins.json"
    if not (Path(run_folder) / "report.json").is_file() and exported.is_file():
        try:
            return {int(vram, 16): (found or None) for vram, found in json.loads(exported.read_text()).items()}
        except (OSError, ValueError):
            return {}
    try:
        rows = json.loads((Path(run_folder) / "report.json").read_text()).get("rows") or []
    except (OSError, ValueError):
        return {}
    out = {}
    for row in rows:
        if row.get("verdict") in ("exact-with-transferred-symbols", "identical-awaiting-symbols"):
            try:
                out[int(str(row["target_vram"]), 16)] = dict(row.get("pin_origin") or {}) or None
            except (KeyError, ValueError):
                continue
    return out


class _DonorTrees:
    """A private copy of each donor tree (sources and the headers they include, as the donor stage
    kept them) and a receipt context built in it: relative includes resolve as in the donor's own
    build, and the copy's object cache starts empty."""

    def __init__(self, contexts: Contexts, root: Path):
        self.contexts, self.root, self.made = contexts, Path(root), {}

    def get(self, image: str, donor: str, tree: Path) -> tuple[receipts.Context, Path]:
        key = (image, donor)
        if key not in self.made:
            copy = self.root / image / donor
            shutil.copytree(tree, copy, symlinks=False)
            base = self.contexts.get(image)
            cfg = dataclasses.replace(base.cfg, project=copy, include=[]) if base.cfg is not None else None
            self.made[key] = (dataclasses.replace(base, project=copy, cfg=cfg, base=None, include=[], _cache={}), copy)
        return self.made[key]


def _donor_pins(spec, origins: dict | None) -> dict:
    origins = origins or {}
    out = {name: origins.get(name, "unrecorded") for name in spec.symbols}
    out.update({f"section:{name}": origins.get(f"section:{name}", "unrecorded")
                for name in spec.section_bases if name != ".text"})
    return out


def issue_donor_receipts(run, *, contexts: Contexts | None = None, out=None, include_assumed: bool = False,
                         export=None, issued_at: str | None = None) -> dict:
    """A receipt for every donor unit of a finished run (``RUN/receipts/donors``), built again in a
    private copy of its donor tree and verified strictly."""
    from ..match.pipeline import split_flags
    run = Path(run).resolve()
    contexts = contexts or Contexts(None, run=run)
    out = Path(out) if out else run / "receipts"
    issued, refused, classes, earned, proofs, units = 0, [], {}, {}, 0, 0
    with tempfile.TemporaryDirectory(prefix="n64dk-donor-issue-") as tmp:
        trees = _DonorTrees(contexts, Path(tmp))
        for folder in sorted(path for path in run.glob("*/donors") if path.is_dir()):
            image = folder.parent.name
            origins = _donor_origins(folder)
            for unit in donor_units(folder, image):
                if unit.waiting and not include_assumed:
                    continue
                units += 1
                label = f"{image}:{unit.donor}:{unit.unit}"
                try:
                    filed = unit.filed
                    if filed is None or not _inside(folder / "src" / unit.donor, filed):
                        raise ValueError(f"SOURCE {unit.unit} is not a file of src/{unit.donor}/")
                    ctx, copy = trees.get(image, unit.donor, folder / "src" / unit.donor)
                    source = copy / filed
                    vram = unit.rom_start + ctx.delta
                    pipeline, _, flags = unit.recipe.partition(" ")
                    spec = split_flags(shlex.split(flags), pipeline)
                    receipt = receipts.issue(ctx, name=unit.name, vram=vram, size=unit.size, source=source, spec=spec,
                                             origin=f"donor {unit.donor}", issued_at=issued_at, in_place=True,
                                             source_name=filed.as_posix(), earned_by=unit.earned_by,
                                             pin_origins=_donor_pins(spec, origins.get(vram)))
                except Exception as error:                  # noqa: BLE001 - one unit never stops the others
                    refused.append({"result": label, "reason": f"{type(error).__name__}: {str(error)[-200:]}"})
                    continue
                path = receipts.write(receipt, receipts.receipt_path(out / "donors", image, unit.name, vram))
                if export is not None:
                    target = receipts.receipt_path(Path(export) / "receipts" / "donors", image, unit.name, vram)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                issued += 1
                cls = receipt["verdict"]["class"]
                classes[cls] = classes.get(cls, 0) + 1
                earned[unit.earned_by] = earned.get(unit.earned_by, 0) + 1
                proofs += int(receipt["claims"]["strict_proof"])
    return {"units": units, "issued": issued, "classes": classes, "earned_by": earned, "strict_proof": proofs,
            "refused": refused[:50], "refused_count": len(refused), "out": str(out / "donors")}


def verify_donors(folders: list[tuple[str, Path]], *, contexts: Contexts, receipts_dir: Path,
                  include_assumed: bool = False, rebuild: bool = True) -> tuple[list, set]:
    """Replay the receipt of every donor unit listed in ``folders`` (``(image, manifest folder)``).
    The recipe a receipt is offered with is the manifest row, never the receipt's own."""
    results, known = [], set()
    with tempfile.TemporaryDirectory(prefix="n64dk-donor-replay-") as tmp:
        trees = _DonorTrees(contexts, Path(tmp))
        for image, folder in folders:
            recorded = _donor_origins(folder)
            replayed = {}                                    # (donor, unit, vram) -> the unit's replay
            for unit in donor_units(folder, image):
                label = f"{image}:{unit.donor}:{unit.unit}"
                base = contexts.get(image)
                vram = unit.rom_start + base.delta
                path = receipts.receipt_path(receipts_dir / "donors", image, unit.name, vram)
                known.add(path.resolve())
                if unit.waiting and not include_assumed:
                    continue
                filed = unit.filed
                if filed is None or not _inside(folder / "src" / unit.donor, filed):
                    results.append((label, None, "donor source outside its donor tree"))
                    continue
                if not (folder / "src" / unit.donor / filed).is_file():
                    results.append((label, None, "donor source missing"))
                    continue
                if not path.is_file():
                    results.append((label, None, "no receipt"))
                    continue
                ctx, copy = trees.get(image, unit.donor, folder / "src" / unit.donor)
                expect = {"image": image, "name": unit.name, "vram": vram, "size": unit.size, "recipe": unit.recipe}
                receipt = _load(path)
                found = receipts.replay(receipt, ctx, copy / filed, expect=expect, rebuild=rebuild, in_place=True)
                if isinstance(receipt, dict) and isinstance(receipt.get("build"), dict):
                    # the origins a receipt asserts are the donor stage's record, not the receipt's word
                    spec = receipts.spec_of(receipt["build"])
                    want = _donor_pins(spec, recorded.get(vram))
                    have = receipt["build"].get("pin_origins") or {}
                    found.check("pin origins", [f"pin origin of {name} is {have.get(name)!r} in the receipt, "
                                                f"{want[name]!r} in the run's record"
                                                for name in sorted(want) if have.get(name) != want[name]])
                    replayed[(unit.donor, unit.unit, vram)] = (found, spec, want)
                results.append((label, found, ""))
            _check_section_chains(folder, image, contexts, trees, replayed, Path(tmp), rebuild)
    return results, known


def _check_section_chains(folder: Path, image: str, contexts: Contexts, trees: "_DonorTrees", replayed: dict,
                          scratch: Path, rebuild: bool) -> None:
    """A section base with the origin ``section-chain`` is the donor stage's conclusion from the
    sections of several units (``transplant.settle_sections``).  It is not believed from the
    record: every unit the run lists for the image (verified or waiting) is built, its ``.data``
    and ``.bss`` are taken at the addresses its manifest row states, the chains are computed
    again by the same rule, and a unit whose section is in none of them fails its replay."""
    from ..match import transplant
    from ..match.pipeline import split_flags
    # by the run's record of each pin's origin (the receipt's own must equal it, checked above)
    resting = {key: [name[len("section:"):] for name, origin in recorded.items()
                     if name.startswith("section:") and str(origin).startswith(transplant.CHAIN_ORIGIN)]
               for key, (_found, _spec, recorded) in replayed.items()}
    resting = {key: names for key, names in resting.items() if names}
    if not resting:
        return
    if not rebuild:
        for key in resting:
            replayed[key][0].checks["section chain"] = "skipped"
        return
    base = contexts.get(image)
    units = []
    for number, unit in enumerate(donor_units(folder, image)):
        vram = unit.rom_start + base.delta
        key = (unit.donor, unit.unit, vram)
        filed = unit.filed
        try:
            if filed is None or not _inside(folder / "src" / unit.donor, filed):
                continue
            ctx, copy = trees.get(image, unit.donor, folder / "src" / unit.donor)
            pipeline, _, flags = unit.recipe.partition(" ")
            spec = split_flags(shlex.split(flags), pipeline)
            elf = replayed[key][0].elf if key in replayed and replayed[key][0].elf is not None else \
                ctx.build(copy / filed, spec, scratch / "chain" / str(number))
            verdict = receipts._strict(ctx, elf, spec, vram, unit.size, None)
        except Exception:                                    # noqa: BLE001 - a unit that does not build is no piece
            continue
        equal = {row["section"] for row in verdict.sections if row.get("status") == "equal"
                 and row.get("compared") == row.get("size") and row.get("base") is not None
                 and int(row["base"], 16) == spec.section_bases.get(row["section"])}
        units.append({"key": key, "text": (vram, unit.size), "elf": elf, "bases": dict(spec.section_bases),
                      "equal": equal})
    chains = transplant.section_chains(transplant.section_pieces(units), base.data, base.vram_to_rom)
    held = {(piece["key"], piece["name"]): piece["address"] for chain in chains for piece in chain}
    for key, names in resting.items():
        found, spec, _recorded = replayed[key]
        found.check("section chain", [
            f"section {name} at {spec.section_bases.get(name, 0):#010x} is said to be settled by sections lying end "
            f"to end; the chains computed again from the run's units do not hold it"
            for name in names if held.get((key, name)) != spec.section_bases.get(name)])


def verify_run(run, *, contexts: Contexts | None = None, receipts_dir=None, include_assumed: bool = False,
               rebuild: bool = True) -> dict:
    """Replay the receipt of every settled result of a run (exact, assembly; ``include_assumed``:
    assumed too).  A required result without a receipt, and a receipt for which the run has no
    such result (an orphan: the result changed or went away), are failures."""
    run = Path(run).resolve()
    contexts = contexts or Contexts(None, run=run)
    folder = Path(receipts_dir) if receipts_dir else run / "receipts"
    cases, texts = _run_cases(run)
    results, known = [], set()
    for case in cases:
        label = f"{case.image}:{case.name}"
        path = receipts.receipt_path(folder, case.image, case.name, case.vram)
        known.add(path.resolve())
        if case.status == "assumed" and not include_assumed:
            continue
        text = texts.get(case.case_id)
        if not path.is_file():
            results.append((label, None, "no receipt"))
            continue
        if text is None:
            results.append((label, None, "source missing from the run's store"))
            continue
        ctx = contexts.get(case.image)
        with tempfile.TemporaryDirectory(prefix="n64dk-verify-") as tmp:
            source = Path(tmp) / f"{case.name}{_suffix(case)}"
            source.write_text(text)
            expect = {"image": case.image, "name": case.name, "vram": case.vram, "size": case.size,
                      "recipe": case.recipe or _recipe(ctx.base)}
            results.append((label, receipts.replay(_load(path), ctx, source, expect=expect, rebuild=rebuild), ""))
    found, donor_known = verify_donors([(path.parent.name, path) for path in sorted(run.glob("*/donors"))
                                        if path.is_dir()], contexts=contexts, receipts_dir=folder,
                                       include_assumed=include_assumed, rebuild=rebuild)
    results += found
    known |= donor_known
    summary = receipts.summarise(results, rebuild)
    summary["donor_units"] = len(found)
    summary["orphans"] = _orphans(folder, known)
    summary["ok"] = summary["ok"] and not summary["orphans"]
    return summary


def _load(path: Path):
    try:
        return receipts.load(path)
    except (OSError, ValueError):
        return None


def _how_recipe(how: str) -> str:
    """The recipe of a ``FUNCTIONS.tsv`` ``how``: ``RECIPE (origin)``; empty when it names none."""
    how = how.strip()
    if how.endswith(")") and " (" in how:
        how = how[:how.index(" (")]
    return "" if how.startswith("(") else how


def verify_export(export, *, contexts: Contexts | None = None, receipts_dir=None, include_assumed: bool = False,
                  rebuild: bool = True, project=None) -> dict:
    """Replay the receipt of every matched ``FUNCTIONS.tsv`` row whose source is in the export
    (``exact`` and ``assembly`` rows; ``include_assumed`` adds ``awaiting-symbols``), and of every
    donor unit of ``EXPORT/donors`` (one receipt per unit: :func:`verify_donors`)."""
    export = Path(export).resolve()
    contexts = contexts or Contexts(project, export=export)
    folder = Path(receipts_dir) if receipts_dir else export / "receipts"
    results, donors, known = [], 0, set()
    with (export / "FUNCTIONS.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        outcome = row.get("outcome") or ""
        if outcome.startswith("donor"):
            donors += 1
            continue
        if outcome not in EXPORT_OUTCOMES:
            continue
        label = f"{row['image']}:{row['function']}"
        vram = int(row["vram"], 16)
        source = export / (row.get("source") or "")
        path = receipts.receipt_path(folder, row["image"], row["function"], vram)
        known.add(path.resolve())
        if EXPORT_OUTCOMES[outcome] == "assumed" and not include_assumed:
            continue
        if not row.get("source") or not source.is_file():
            results.append((label, None, "source missing from the export"))
            continue
        if not path.is_file():
            results.append((label, None, "no receipt"))
            continue
        expect = {"image": row["image"], "name": row["function"], "vram": vram, "size": int(row["code_size"], 16),
                  "catalogue_size": int(row["size"], 16), "recipe": _how_recipe(row.get("how") or "")}
        results.append((label, receipts.replay(_load(path), contexts.get(row["image"]), source, expect=expect,
                                               rebuild=rebuild), ""))
    folders = sorted((path.name, path) for path in (export / "donors").iterdir()
                     if path.is_dir()) if (export / "donors").is_dir() else []
    found, donor_known = verify_donors(folders, contexts=contexts, receipts_dir=folder,
                                       include_assumed=include_assumed, rebuild=rebuild)
    results += found
    known |= donor_known
    summary = receipts.summarise(results, rebuild)
    summary["donor_function_rows"] = donors
    summary["donor_units"] = len(found)
    summary["orphans"] = _orphans(folder, known)
    summary["ok"] = summary["ok"] and not summary["orphans"]
    return summary


def _finish(args, result: dict) -> int:
    emit(result, getattr(args, "format", "text"))
    return 0 if result.get("ok") else 1


@command("issue-receipts", "Write a proof receipt for every settled result of a finished decompile-run "
                           "(opt-in; rebuilds each result once)")
def issue_receipts_command(args):
    contexts = Contexts(args.project, run=args.run, toolchain=args.toolchain_dir, extra=args.symbols)
    return issue_receipts(args.run, contexts=contexts, out=args.out, include_assumed=args.include_assumed,
                          export=args.export)


@command("verify-receipt", "Replay one proof receipt: recompute its digests, rebuild its source and verify it "
                           "strictly")
def verify_receipt_command(args):
    receipt = _load(Path(args.receipt))
    if receipt is None:
        return _finish(args, {"ok": False, "failures": ["not a readable receipt"]})
    contexts = Contexts(args.project, run=args.run, export=args.export, toolchain=args.toolchain_dir,
                        extra=args.symbols)
    source = Path(args.source) if args.source else None
    if source is None and args.run:
        digest = (receipt.get("source") or {}).get("sha256", "")
        source = Path(args.run) / "state" / "sources" / f"{digest}.c"
        if not source.is_file():
            source = None
    if source is None:
        return _finish(args, {"ok": False, "failures": ["no source: pass --source (or --run holding it)"]})
    image = (receipt.get("subject") or {}).get("image")
    result = receipts.replay(receipt, contexts.get(image), source, rebuild=not args.static).as_dict()
    return _finish(args, result)


@command("verify-run", "Replay the proof receipt of every settled result of a decompile-run")
def verify_run_command(args):
    contexts = Contexts(args.project, run=args.run, toolchain=args.toolchain_dir, extra=args.symbols)
    return _finish(args, verify_run(args.run, contexts=contexts, receipts_dir=args.receipts,
                                    include_assumed=args.include_assumed, rebuild=not args.static))


@command("verify-export", "Replay the proof receipt of every matched function of a decompile-export tree")
def verify_export_command(args):
    contexts = Contexts(args.project, export=args.export, toolchain=args.toolchain_dir, extra=args.symbols)
    return _finish(args, verify_export(args.export, contexts=contexts, receipts_dir=args.receipts,
                                       include_assumed=args.include_assumed, rebuild=not args.static))


def _common(p, *, project_required: bool = False):
    p.add_argument("--project", required=project_required,
                   help="the project (ROM, catalogue, include directories); default: the run's own")
    p.add_argument("--toolchain-dir", help="toolchain root (default: the run's, else N64DK_TOOLCHAIN)")
    p.add_argument("--symbols", action="append", default=[], help="additional symbol file (repeatable)")


def _issue_args(p):
    p.add_argument("run", help="the decompile-run output directory")
    _common(p)
    p.add_argument("--out", help="receipt directory (default: RUN/receipts)")
    p.add_argument("--export", help="also copy the receipts into EXPORT/receipts")
    p.add_argument("--include-assumed", action="store_true",
                   help="also receipt byte-identical results that rest on undeclared addresses (class assumed)")


def _receipt_args(p):
    p.add_argument("receipt", help="a receipt file")
    _common(p)
    p.add_argument("--source", help="the source the receipt covers (default: the run's store, with --run)")
    p.add_argument("--run", help="replay with the run's declarations (and its stored source)")
    p.add_argument("--export", help="replay with the export's declarations")
    p.add_argument("--static", action="store_true", help="check every digest without rebuilding")


def _run_args(p):
    p.add_argument("run", help="the decompile-run output directory")
    _common(p)
    p.add_argument("--receipts", help="receipt directory (default: RUN/receipts)")
    p.add_argument("--include-assumed", action="store_true", help="also require receipts for assumed results")
    p.add_argument("--static", action="store_true", help="check every digest without rebuilding")


def _export_args(p):
    p.add_argument("export", help="the decompile-export tree")
    _common(p, project_required=True)
    p.add_argument("--receipts", help="receipt directory (default: EXPORT/receipts)")
    p.add_argument("--include-assumed", action="store_true",
                   help="also require receipts for awaiting-symbols rows")
    p.add_argument("--static", action="store_true", help="check every digest without rebuilding")


issue_receipts_command.arguments = _issue_args
verify_receipt_command.arguments = _receipt_args
verify_run_command.arguments = _run_args
verify_export_command.arguments = _export_args
