"""``decompile-export``: a finished run as a tree that can be committed as a decompilation repository."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from ..cli import command

MATCHED = ("donor", "donor-awaiting-symbols", "donor-assembly",
           "donor-assembly-awaiting-symbols", "exact", "awaiting-symbols", "assembly")
_TEXT = (".c", ".h", ".s", ".tsv", ".txt", ".toml", ".md", ".json", ".inc")


_HOME = re.compile(r"(?:/home/|/Users/|/mnt/[a-z]/Users/|[A-Za-z]:\\\\Users\\\\)[^/\\\\\"'\s]+")


def _scrub(text: str, replacements: list[tuple[str, str]]) -> str:
    """Paths of the machine the run was made on say nothing about the title and name its user."""
    for old, new in replacements:
        text = text.replace(old, new)
    return _HOME.sub("<home>", text)


def _copy(source: Path, target: Path, replacements) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in _TEXT:
        target.write_text(_scrub(source.read_text(errors="replace"), replacements))
    else:
        shutil.copyfile(source, target)


def _unit_class(row: list[str], evidence: dict) -> str:
    """The unit class of one function-table row: its ``vram`` and ``size`` are hexadecimal text."""
    from ..analysis import unitclass
    return unitclass.classify(int(row[2], 16), int(row[4], 16), evidence)


def reproduced_by(outcome: str, disassembled: bool) -> str:
    """What reproduces a function in the export.  Only a matched outcome is C; a function the
    export covers with generated (disassembled) assembly is ``assembly (not decompiled)``, never C."""
    if outcome.startswith("donor-assembly"):
        return "assembly (donor source)"
    if outcome == "assembly":
        return "assembly (hand-written)"
    if outcome in MATCHED:
        return "C"
    return "assembly (not decompiled)" if disassembled else "nothing"


def rules_column(outcome: str, blocked: list[str] | None) -> str:
    """The ``rules`` column of a function's row.  ``blocked`` is the [match] rules that fire in its
    source (None when the source was not checked).  Identical bytes with a blocked rule are
    ``MATCH-PENDING``; hand-written assembly is ``ORIGINAL-ASM``, counted neither as matched nor
    in the total; a donor's source belongs to another project and keeps that project's profile."""
    if outcome == "assembly":
        return "ORIGINAL-ASM"
    if outcome in ("donor", "donor-awaiting-symbols"):
        return "unchecked (donor)"
    if outcome not in ("exact", "awaiting-symbols") or blocked is None:
        return ""
    return "MATCH-PENDING: " + ",".join(blocked) if blocked else "pass"


def _profile_text(name: str, recipes: dict[str, int], summary: dict, relaxed: dict) -> str:
    """The project profile the code rules expect in a project's ``AGENTS.md``, filled with what the
    run established and saying plainly what it did not."""
    seen = [f"  - `{recipe}`: {count:,} functions" for recipe, count in
            sorted(recipes.items(), key=lambda item: -item[1])[:12]] or ["  - none: no function matched from C"]
    changes = [f"- {rule} relaxed: {reason}" for rule, reason in sorted(relaxed.items())] or [
        "- none.  A rule is tightened or relaxed here, with its reason, and in `[code_rules]` of `n64dk.toml`."]
    pending = ", ".join(f"{rule} {count:,}" for rule, count in sorted(summary["pending_by_rule"].items()))
    return "\n".join([
        f"# Project profile: {name}", "",
        "Written by `n64dk decompile-export`.  It is the profile the code rules read (`docs/code-rules.md` in",
        "n64dk): edit it as evidence arrives.  Where it says *not established*, nothing has been proved yet.", "",
        "## Compiler and dialect", "",
        "- Build recipes of the functions matched from C (pipeline and flags, from `FUNCTIONS.tsv`):", *seen,
        "- Dialect: write only what these compilers accept.  IDO is C89: declarations at the top of a block, no",
        "  declarations in a `for` header.  KMC and SN GCC are GCC 2.x: pre-C99, with the GNU extensions of",
        "  that release.  No `stdint.h`, no `stdbool.h`.", "",
        "## Annotation and parking", "",
        "- One function per file, as verified.  No parking mechanism is defined: a function that does not match",
        "  stays as assembly under `asm_not_decompiled/` and is not C.",
        "- A function the evidence shows was written in assembly is `ORIGINAL-ASM` (`asm/`), listed apart and",
        "  counted neither as matched nor in the total.", "",
        "## Naming style", "",
        "- *Not established.*  No evidence of the original's naming has been recorded.  Names are addresses",
        "  (`func_80012345`, `D_80123456`) unless an SDK signature named them.  These are placeholders and open",
        "  naming work (R7), not a failed match.  Never replace one with a confident invented name: use the",
        "  binary's own evidence first (error strings that start with a function name, SDK signatures).",
        "- Placeholder patterns the checker reports: `func_XXXXXXXX`, `D_XXXXXXXX`, `jtbl_XXXXXXXX`, `argN`,",
        "  `temp_*`, `var_*`, `phi_*`, `spXX`, `unkXX`, `M2C_*`, and address-like names.", "",
        "## Layout style", "",
        "- *Not established.*  Record the brace placement, indent, line limit and banner forms here once chosen,",
        "  per module where modules differ.", "",
        "## Evidence sources", "",
        "- SDK signatures (`symbols/sdk_symbols.txt`), the survey's ledger (`config/ledger.tsv`), strings in the",
        "  ROM, donor units (`donors/`), development records where `n64dk dev-records-scan` finds any.", "",
        "## Line information", "",
        "- *Not established.*  If the retail image or its development records carry source line numbers, layout",
        "  changes are not free: rerun the byte gate after any reformatting.", "",
        "## Commands", "",
        "- Per-function check: `n64dk match --project project --addr VRAM --size SIZE --pipeline PIPELINE",
        "  --flags=\"FLAGS\" src/<image>/FUNCTION.c` (values from the function's row in `FUNCTIONS.tsv`).",
        "- Byte gate: `n64dk match-batch` or `n64dk manifest-verify` over `manifests/`.",
        "- Rule checker: `n64dk code-rules src --project . --strict`.", "",
        "## Where this export stands", "",
        f"- Functions matched under the rules: {summary['pass']:,}.  MATCH-PENDING: {summary['match_pending']:,}"
        + (f" ({pending})." if pending else "."),
        "- A function is matched only when its bytes are identical *and* every [match] rule passes (R1, R2, R4,",
        "  R13, R17, R18).  `RULES.tsv` lists every finding; `DECLARATIONS.tsv` lists every `extern` by name.",
        "- Each exported source declares what it needs locally.  That is scaffolding: give every global one",
        "  `extern` in one header and every type one definition, rebuild, and rerun the byte gate.", "",
        "## Rule changes", "", *changes, ""])


def _origin(case) -> str:
    """How a case was matched, from its own record."""
    kinds = [stage.get("stage") for stage in case.stages]
    if "refine" in kinds:
        return "external author, strictly verified"
    if "twin" in kinds:
        return "source of an identical function"
    if any(stage.get("stage") in ("region-recipe", "unit-recipe") and stage.get("kept") for stage in case.stages):
        return "draft, recipe of its region"
    iterate = [stage for stage in case.stages if stage.get("stage") == "iterate"]
    if iterate and iterate[-1].get("final") == 0 and iterate[-1].get("start", 0) > 0:
        return "draft after variant search"
    return "draft"


GP_DEPENDENT = "needs the recovered _gp in symbols/gp_symbols.txt"


def _recovered_gp(project: Path, run: Path, image: str) -> bool:
    """Whether the run's small-data lanes were verified against a ``_gp`` recovered from code
    (``small_data/gp_symbols.txt``) because the project declares none for ``image``."""
    if not (Path(run) / "small_data" / "gp_symbols.txt").is_file():
        return False
    from .run import project_gp
    try:
        return project_gp(Path(project), image) is None
    except Exception:                                    # noqa: BLE001 - unreadable project symbols: the run's file
        return True                                      # is the only _gp known to have been declared


def _how(case, recovered_gp: bool) -> str:
    """The ``how`` of a function exported with its source: recipe and origin.  One built in a
    small-data lane against a recovered ``_gp`` says it verifies again only with that file
    (verification itself is unchanged: strict, relocations included)."""
    from ..match.pipeline import SMALL_DATA_LANES
    origin = _origin(case)
    if recovered_gp and str(case.recipe or "").partition(" ")[0] in SMALL_DATA_LANES:
        origin = f"{origin}; {GP_DEPENDENT}"
    return f"{case.recipe} ({origin})"


def _gp_note(count: int) -> list[str]:
    """COVERAGE.md's note on the functions that depend on the recovered ``_gp`` (none: no note)."""
    if not count:
        return []
    return [f"> **Note.** {count:,} of these functions were built in a small-data lane and verified against the `_gp` "
            "the run recovered from the code that sets `$gp`; the project does not declare it.  They verify again "
            f"only with `symbols/gp_symbols.txt` (their `how` in `FUNCTIONS.tsv` says \"{GP_DEPENDENT}\").", ""]


def _donor_roots(run: Path) -> dict[str, str]:
    """Donor name -> its source root, from the index the run built."""
    try:
        return {name: str(entry.get("root") or "") for name, entry in
                (json.loads((run / "donor_index.json").read_text()).get("donors") or {}).items()}
    except (OSError, ValueError):
        return {}


def rom_coverage(ledger: Path, rom_size: int | None) -> dict | None:
    """What the survey's ledger says about every byte of the ROM: bytes per class, whether its
    rows cover the ROM from the first byte to the last without gap or overlap, and the ranges
    it could not classify (largest first)."""
    import csv
    if not ledger.is_file():
        return None
    rows = []
    with ledger.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                rows.append((int(row["start"], 16), int(row["end"], 16), row.get("class") or "?", row.get("kind") or "",
                             row.get("name") or "", row.get("evidence") or ""))
            except (KeyError, ValueError):
                continue
    rows.sort()
    classes: dict[str, int] = {}
    gaps = overlaps = position = 0
    for start, end, kind, *_rest in rows:
        gaps += max(0, start - position)
        overlaps += max(0, min(end, position) - start)
        classes[kind] = classes.get(kind, 0) + (end - start)
        position = max(position, end)
    size = rom_size if rom_size is not None else position
    gaps += max(0, size - position)
    unknown = sorted(((end - start, start, end, evidence) for start, end, kind, _k, _n, evidence in rows if kind == "unknown"),
                     reverse=True)
    return {"rom_bytes": size, "rows": len(rows), "gap_bytes": gaps, "overlap_bytes": overlaps,
            "bytes_by_class": dict(sorted(classes.items(), key=lambda item: -item[1])),
            "unclassified_ranges": len(unknown),
            "largest_unclassified": [{"start": f"0x{start:X}", "end": f"0x{end:X}", "bytes": length, "evidence": evidence[:160]}
                                     for length, start, end, evidence in unknown[:12]]}


def _disassemble(pending: list, out: Path, project: Path, toolchain, jobs: int = 4) -> dict[tuple[str, int], str]:
    """Assembly source for functions no C reproduces, each proved by assembling it again.

    ``pending`` is ``[(image, function, vram, code, image bytes, delta)]``.  Returns the path of
    the file written for every function whose assembly gives back its bytes."""
    from concurrent.futures import ThreadPoolExecutor
    from ..match import handasm, m2c_bridge
    from ..match import symbols as symmod
    from ..match.toolchain import load_config
    cfg = load_config(project, None, toolchain)
    files = sorted((project / "survey").glob("symbol_addrs*.txt")) + [project / "survey" / "sdk_symbols.txt"]
    try:
        import tomllib
        settings = tomllib.loads((project / "n64dk.toml").read_text()).get("run") or {}
        files += [project / name for name in settings.get("symbols", [])]
    except (OSError, ValueError):
        pass
    try:
        names = {address: name for name, address in symmod.load([f for f in files if f.is_file()]).items()}
    except Exception:                                        # noqa: BLE001 - names are a convenience here
        names = {}

    def one(item):
        image, function, vram, code, data, delta = item
        try:
            tables = m2c_bridge.find_jump_tables(data, delta, vram, len(code))
        except Exception:                                    # noqa: BLE001
            tables = []
        kept = handasm.attempt(function, code, vram, cfg, names, tables)
        if kept is None:
            return (image, vram), None
        path = f"asm_not_decompiled/{image}/{function}.s"
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(kept[0])
        return (image, vram), path

    with ThreadPoolExecutor(max(1, jobs)) as pool:
        return {key: path for key, path in pool.map(one, pending) if path}


def export(project: Path, run: Path, out: Path, title: str | None = None, cover: bool = True,
           toolchain=None) -> dict:
    """Write the export tree and return its counts.  Nothing of the ROM is copied: sources,
    symbol files, recipes and the function table only.

    With ``cover`` every function no C reproduces also gets assembly source that assembles to its
    bytes (``asm_not_decompiled/``), so that each function of the title has a source, and the
    table says which kind: decompiled, hand-written, or only disassembled."""
    from ..match import images as imagemod
    from ..match import resolve as resolvemod
    project, run, out = Path(project).resolve(), Path(run).resolve(), Path(out).resolve()
    replacements = sorted({(str(run), "<run>"), (str(project), "<project>"), (str(Path.home()), "<home>")},
                          key=lambda pair: -len(pair[0]))
    registry = imagemod.load_registry(project)
    store = resolvemod.CaseStore(run / "state", readonly=True)
    try:
        cases = {(case.image, case.vram): case for case in store.all()}
        sources = {key: store.source(case.best_digest) for key, case in cases.items()
                   if case.status in ("exact", "assumed", "assembly") and case.best_digest}
    finally:
        store.close()
    out.mkdir(parents=True, exist_ok=True)
    from ..match import reasons as reasonsmod
    rows, counts, pending = [], {}, []
    checked: dict[str, str] = {}             # exported C of this run, by its path in the tree
    recipes: dict[str, int] = {}
    explained = []                 # per row: (case, recorded reason it was not attempted, dossier)
    gp_dependent = 0               # functions verified against the recovered _gp (symbols/gp_symbols.txt)
    rom_sha1, rom_name, rom_size = None, None, None
    for name in sorted(registry):
        spec = imagemod.with_run_catalogue(imagemod.resolve(project, name), run)
        data, delta = spec.load(), spec.delta
        if name == "main" or rom_sha1 is None:
            rom = Path(spec.path)
            if rom.is_file() and rom.suffix.lower() in (".z64", ".n64", ".v64", ".rom"):
                blob = rom.read_bytes()
                rom_sha1, rom_size = hashlib.sha1(blob).hexdigest(), len(blob)
                rom_name = blob[0x20:0x34].decode("ascii", "replace").strip()
        donated: dict[int, tuple[str, str, str, str]] = {}
        stage = run / name / "stage_donors.json"
        if stage.is_file():
            held = json.loads(stage.read_text())
            for kind, key in (("donor", "verified"), ("donor-awaiting-symbols", "assumed")):
                for row in held.get(key, []):
                    for vram in row.get("function_vrams") or []:
                        unit = str(row.get("unit"))
                        if unit.startswith("/"):              # generated text kept outside the donor's root:
                            unit = "/".join(Path(unit).parts[-2:])   # filed by its last two components
                        source_kind = row.get("source_kind") or (
                            "assembly" if Path(unit).suffix.lower() in (".s", ".asm") else "c")
                        outcome = ("donor-assembly" if kind == "donor" else
                                   "donor-assembly-awaiting-symbols") if source_kind == "assembly" else kind
                        donated.setdefault(int(vram), (outcome, str(row.get("donor")), unit, source_kind))
        scope_file = run / "state" / f"scope_{name}.json"
        left_out = ({int(item["vram"], 16): item["reason"] for item in json.loads(scope_file.read_text())["left_out"]}
                    if scope_file.is_file() else {})
        dossiers = reasonsmod.load_dossiers(run, name)
        why_not = {**reasonsmod.load_not_attempted(run, name), **left_out}
        recovered_gp = _recovered_gp(project, run, name)
        for function, vram, size in spec.rows():
            case = cases.get((name, vram))
            outcome, how, path, score, code_size = "not attempted", left_out.get(vram, ""), "", "", size
            if vram in donated and proved_beside_waiting_donor(donated[vram][0], case,
                                                              sources.get((name, vram)) is not None):
                del donated[vram]
            if vram in donated:
                outcome, donor, unit, _source_kind = donated[vram]
                how = f"donor unit {donor}: {unit}"
                path = f"donors/{name}/src/{donor}/{unit}" if not unit.startswith(("<", "/")) else f"donors/{name}/"
            elif case is not None:
                outcome = {"assumed": "awaiting-symbols"}.get(case.status, case.status)
                how, code_size = case.recipe, case.size       # the catalogue's span may end in alignment padding
                score = "" if case.best_score is None else str(case.best_score)
                text = sources.get((name, vram))
                if text is not None:
                    folder = {"exact": "src", "awaiting-symbols": "src_awaiting_symbols", "assembly": "asm"}[outcome]
                    path = f"{folder}/{name}/{function}.{'s' if outcome == 'assembly' else 'c'}"
                    target = out / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text)
                    if outcome != "assembly":
                        checked[path] = text
                        recipes[case.recipe] = recipes.get(case.recipe, 0) + 1
                    how = _how(case, recovered_gp) if outcome != "assembly" else case.recipe
                    gp_dependent += int(GP_DEPENDENT in how)
                elif outcome in ("exact", "awaiting-symbols", "assembly"):
                    outcome = "matched-source-missing"
            counts[outcome] = counts.get(outcome, 0) + 1
            if outcome not in MATCHED and cover:
                span = data[vram - delta:vram - delta + size]
                if len(span) == size and size:
                    pending.append((name, function, vram, span, data, delta))
            rows.append([name, function, f"0x{vram:08X}", f"0x{vram - delta:X}", f"0x{size:X}", f"0x{code_size:X}",
                         outcome, how, path, score])
            explained.append((case, why_not.get(vram, ""), dossiers.get(vram)))
        # donor units as the donor stage wrote them (sources, the manifests it proposes, symbols)
        donors = run / name / "donors"
        if donors.is_dir():
            for item in sorted(donors.rglob("*")):
                if item.is_file() and item.name != "report.json":
                    _copy(item, out / "donors" / name / item.relative_to(donors), replacements)
            # the origin of every address a donor unit's manifest row pins, as the donor stage recorded it:
            # receipt replay holds each donor receipt's recorded origins to this record
            from .verifyreceipt import _donor_origins
            origins = _donor_origins(donors)
            if origins:
                (out / "donors" / name).mkdir(parents=True, exist_ok=True)
                (out / "donors" / name / "pin_origins.json").write_text(json.dumps(
                    {f"0x{vram:08X}": found for vram, found in sorted(origins.items())}, indent=1, sort_keys=True) + "\n")
        # a run made before sources of awaiting units were kept: fetch them from the donor itself
        missing = {(donor, unit) for kind, donor, unit, _source_kind in donated.values()
                   if not unit.startswith(("<", "/")) and not (out / "donors" / name / "src" / donor / unit).is_file()}
        if missing:
            roots = _donor_roots(run)
            for donor, unit in sorted(missing):
                source = Path(roots.get(donor, "")) / unit
                if roots.get(donor) and source.is_file():
                    _copy(source, out / "donors" / name / "src" / donor / unit, replacements)
        declared = run / name / "declared_symbols.txt"
        if declared.is_file():
            _copy(declared, out / "symbols" / f"declared_{name}.txt", replacements)
    disassembled = _disassemble(pending, out, project, toolchain) if pending else {}
    reproduced = {"C": 0, "assembly (donor source)": 0, "assembly (hand-written)": 0,
                  "assembly (not decompiled)": 0, "nothing": 0}
    from ..analysis import unitclass
    evidence = unitclass.load(project)
    by_class: dict[str, dict[str, int]] = {}
    tallies = []                   # (reason code, size) per row: COVERAGE.md's reason table
    for index, row in enumerate(rows):
        found = None if row[6] in MATCHED else disassembled.get((row[0], int(row[2], 16)))
        kind = reproduced_by(row[6], bool(found))
        if found:
            row[8] = found
        reproduced[kind] += 1
        row.append(kind)
        # separate output categories: boot, RSP and hand-written units never share a C denominator
        unit = _unit_class(row, evidence)
        row.append(unit)
        by_class.setdefault(unit, {}).setdefault(kind, 0)
        by_class[unit][kind] += 1
        # why it is not settled, from the run's own records (empty when settled); explains, never credits
        case, why, dossier = explained[index]
        code, step = reasonsmod.explain(row[6], case, unit_class=unit, not_attempted=why, dossier=dossier)
        row += [code, _scrub(step, replacements)]
        tallies.append((code, int(row[4], 16)))
    # Identical bytes are not yet a match: every [match] code rule has to pass as well.
    from ..match import coderules
    profile = coderules.Profile.load(project)
    report = coderules.check(checked, profile)
    rules = {"pass": 0, "match_pending": 0, "original_asm": 0, "unchecked_donor": 0, "pass_awaiting_symbols": 0,
             "pending_by_rule": {}}
    for row in rows:
        blocked = report.blocked_rules(row[8]) if row[8] in checked else None
        column = rules_column(row[6], blocked)
        row.append(column)
        key = {"pass": "pass", "ORIGINAL-ASM": "original_asm", "unchecked (donor)": "unchecked_donor"}.get(
            column, "match_pending" if column.startswith("MATCH-PENDING") else None)
        if key == "pass" and row[6] != "exact":
            key = "pass_awaiting_symbols"            # no rule fires, but only a strict verdict is a match
        if key:
            rules[key] += 1
        for rule in blocked or ():
            rules["pending_by_rule"][rule] = rules["pending_by_rule"].get(rule, 0) + 1
    rules["open_naming"] = len({finding.path for finding in report.of(coderules.ADVISORY) if finding.rule == "R7"})
    rules["relaxed"] = dict(profile.relax)
    (out / "RULES.tsv").write_text("source\tline\tfunction\trule\ttier\tmessage\n" + "".join(
        f"{f.path}\t{f.line}\t{f.function or ''}\t{f.rule}\t{f.tier}\t{f.message}\n" for f in report.findings))
    (out / "DECLARATIONS.tsv").write_text("name\tkind\tfiles\tconflict\tdeclarations\n" + "".join(
        f"{row['name']}\t{row['kind']}\t{row['files']}\t{'yes' if row['conflict'] else 'no'}\t"
        f"{' | '.join(row['declarations'])}\n" for row in coderules.declarations(checked)))
    header = ("image\tfunction\tvram\trom_offset\tsize\tcode_size\toutcome\thow\tsource\tdistance\treproduced_by"
              "\tunit_class\treason_code\tnext_step\n")
    header = header.rstrip("\n") + "\trules\n"       # appended last: every earlier column keeps its position
    (out / "FUNCTIONS.tsv").write_text(header + "".join("\t".join(row) + "\n" for row in rows))
    (out / "MATCHED.tsv").write_text(header + "".join("\t".join(row) + "\n" for row in rows if row[6] in MATCHED))
    for item in sorted((run / "queue").glob("*")) if (run / "queue").is_dir() else []:
        if item.is_file():
            _copy(item, out / "manifests" / item.name, replacements)
    survey = project / "survey"
    # an image the run resolved against its own catalogue (repaired or corrected): the export carries that one
    run_catalogues = {}
    for name in registry:
        spec = imagemod.resolve(project, name)
        used = imagemod.with_run_catalogue(spec, run).catalogue
        if spec.catalogue and used and Path(used).resolve() != Path(spec.catalogue).resolve():
            run_catalogues[Path(spec.catalogue).resolve()] = Path(used)
    for item in sorted(survey.glob("*")) if survey.is_dir() else []:
        if item.is_file() and item.suffix in (".txt", ".toml", ".json", ".tsv", ".md"):
            folder = "symbols" if item.name.startswith(("symbol_addrs", "sdk_symbols")) else "config"
            _copy(run_catalogues.get(item.resolve(), item), out / folder / item.name, replacements)
    if (project / "n64dk.toml").is_file():
        _copy(project / "n64dk.toml", out / "config" / "n64dk.toml", replacements)
    # the program's _gp the run recovered from the code that sets $gp (small data): every function
    # verified with a GP-relative relocation needs it declared to verify again
    if (run / "small_data" / "gp_symbols.txt").is_file():
        _copy(run / "small_data" / "gp_symbols.txt", out / "symbols" / "gp_symbols.txt", replacements)
    for item in ("report.md", "report.json"):
        if (run / item).is_file():
            _copy(run / item, out / "report" / item, replacements)
    # the unresolved dossier (n64dk.match.dossier): the run's own, else derived from the run's records
    # (read-only: a run made before decompile-run wrote it); UNRESOLVED.md renders it
    from ..match import dossier as dossiermod
    try:
        unresolved = dossiermod.load(run) or dossiermod.collect(run, project=project)
        (out / "report").mkdir(parents=True, exist_ok=True)
        (out / "report" / dossiermod.FILE).write_text(_scrub(json.dumps(unresolved, separators=(",", ":")),
                                                             replacements) + "\n")
        (out / "UNRESOLVED.md").write_text(_scrub(dossiermod.markdown(unresolved), replacements))
        unresolved = {key: unresolved["summary"][key] for key in ("functions", "bytes", "by_retry")}
    except Exception as error:                               # noqa: BLE001 - a report, never the export's failure
        unresolved = {"error": f"{type(error).__name__}: {error}"[-300:]}
    total = len(rows)
    matched = sum(counts.get(key, 0) for key in MATCHED)
    from_c = matched - counts.get("assembly", 0) - counts.get("donor-assembly", 0) - \
        counts.get("donor-assembly-awaiting-symbols", 0)
    name = title or rom_name or project.name
    order = [*MATCHED, *sorted(key for key in counts if key not in MATCHED)]
    counted = total - rules["original_asm"]          # hand-written assembly is in neither count
    by_rule = ", ".join(f"{rule} {count:,}" for rule, count in sorted(rules["pending_by_rule"].items()))
    (out / "AGENTS.md").write_text(_profile_text(name, recipes, rules, profile.relax))
    readme = [f"# {name}", "",
              "Functions of this title that an unattended `n64dk decompile-run` reproduced byte for byte, from the ROM",
              "alone.  Nothing of the ROM is in this tree: sources, symbol files, build recipes and the function table.", "",
              f"- ROM header name: `{rom_name}`", f"- ROM SHA-1: `{rom_sha1}`", f"- ROM size: {rom_size} bytes",
              f"- Functions found: {total:,}",
              f"- **Matched under the code rules: {rules['pass']:,} of {counted:,} ({rules['pass'] / counted:.1%})**: "
              "strictly exact C with every [match] rule passing" if counted else "- Matched under the code rules: 0",
              f"- Bytes identical from C: {from_c:,} ({from_c / total:.1%})" if total else "- Bytes identical from C: 0",
              f"- MATCH-PENDING (bytes identical, a [match] rule fails; not counted): {rules['match_pending']:,}"
              + (f" ({by_rule})" if by_rule else ""),
              f"- Donor C, rules not checked here (the donor project's profile applies): {rules['unchecked_donor']:,}",
              f"- Sources with placeholder names (open naming work, not a failed match): {rules['open_naming']:,}",
              f"- Hand-written code kept as assembly (ORIGINAL-ASM, in neither count): {counts.get('assembly', 0):,}",
              f"- Verified donor assembly (not counted as C): "
              f"{counts.get('donor-assembly', 0) + counts.get('donor-assembly-awaiting-symbols', 0):,}",
              f"- Not decompiled, covered by assembly that assembles to the same bytes: "
              f"{reproduced['assembly (not decompiled)']:,}",
              f"- Functions with no source that reproduces them: {reproduced['nothing']:,}", "",
              "| outcome | functions | meaning |", "|---|---:|---|"]
    meaning = {"donor": "a unit of an already decompiled library or title, rebuilt for this image and compared (strict)",
               "donor-awaiting-symbols": "such a unit, byte-identical, leaning on an address nothing independent confirms",
               "donor-assembly": "assembly from a donor, rebuilt and strictly compared; not counted as C",
               "donor-assembly-awaiting-symbols": "donor assembly with identical bytes but an unconfirmed address",
               "exact": "C source accepted by the strict verifier: bytes and every relocation",
               "awaiting-symbols": "C source that is byte-identical and leans on an undeclared address",
               "assembly": "hand-written code: assembly source that assembles to the same bytes",
               "near": "best draft within ten differences", "open": "best draft further away",
               "does-not-build": "no draft compiles", "blocked-boundary": "the catalogued range is not one whole function",
               "not attempted": "left out by the run's settings (the recipe column says which: size limit, sample, "
                                "name set) or the run was cut short"}
    for key in order:
        if counts.get(key):
            readme.append(f"| {key} | {counts[key]:,} | {meaning.get(key, '')} |")
    readme += ["", "## Layout", "",
               "- `FUNCTIONS.tsv`: every function with its address, size, outcome, recipe and source file.  `size` is",
               "  the catalogue's span (it may end in alignment padding); `code_size` is what the source reproduces.",
               "  `reason_code` and `next_step` say why a function is not settled and what would settle it",
               "  (empty when it is settled; `COVERAGE.md` counts the codes).",
               "  `rules` says whether identical bytes count: `pass`, `MATCH-PENDING: <rules>`, `ORIGINAL-ASM` or",
               "  `unchecked (donor)`.",
               "- `MATCHED.tsv`: the byte-identical ones only (read `rules` before counting one as matched).",
               "- `UNRESOLVED.md`: every function not settled with its reason, the techniques tried on it, the",
               "  eligible next ones at their measured cost, and the evidence that would justify a retry",
               "  (`report/unresolved_dossier.json` holds it in full).",
               "- `RULES.tsv`: every code-rule finding in the exported C, [match] and advisory.",
               "- `DECLARATIONS.tsv`: every `extern` by name, with each way it is declared and in how many files.",
               "- `AGENTS.md`: the project profile the code rules read (dialect, naming, commands, rule changes).",
               "- `src/<image>/<function>.c`: strictly exact functions, one per file, as they were verified.",
               "- `src_awaiting_symbols/`: byte-identical functions that wait for a symbol to be declared",
               "  (`manifests/symbols.assumed.txt` lists the declarations).",
               "- `asm/<image>/<function>.s`: hand-written routines.",
               "- `asm_not_decompiled/<image>/<function>.s`: every function no C reproduces yet, as assembly that",
               "  assembles to its bytes.  It is a disassembly, not a decompilation, and is counted apart.",
               "- `config/ledger.tsv`: every byte of the ROM, from the first to the last, with what it is.",
               "- `donors/<image>/`: library and shared units the donor stage verified: their sources, the manifest rows",
               "  it proposes (`proposed_*.tsv`), the symbols it transferred, and what still awaits symbols.",
               "- `manifests/`: recipe of every exported function (`queue.tsv` exact, `assumed.tsv`, `assembly.tsv`).",
               "- `symbols/gp_symbols.txt` (small-data titles only): the program's `_gp`, recovered from the code that",
               "  sets `$gp`; declare it with the other symbol files when verifying functions built at `-G8`.",
               "- `symbols/`, `config/`: symbol files, the function catalogue, the project configuration, the ROM layout.",
               "- `report/`: the run's own report.", "",
               "Each source is a single function with the declarations the decompiler needed; names are addresses",
               "(`func_80012345`, `D_80123456`) unless a library signature named them.  Sources under `donors/` come",
               "from other projects (the SDK library source, other decompiled titles) and keep those projects' terms.", "",
               "## Matched, and matched under the rules", "",
               "Identical bytes prove a source is faithful; they do not prove it is the source.  A function counts",
               "as matched only when its bytes are identical and no [match] code rule fires in its source: one",
               "declaration per type and global (R1, R2), typed members instead of offset casts (R4), no fake code",
               "(R17: unused locals, `volatile` carriers, empty blocks, `do { } while (0)`, `+ 0`), no assembly",
               "bodies (R18).  Otherwise it is MATCH-PENDING: reported, not counted.", "",
               "A source exported here declares what it needs in its own file, so a function that names a global is",
               "MATCH-PENDING under R2 until the declarations are gathered into shared headers (`DECLARATIONS.tsv`",
               "is the work list; a name declared in two ways is a conflict to settle first).  Rebuild and rerun",
               "the byte gate after each such change: moving a declaration can move data.  `n64dk code-rules src",
               "--project . --strict` repeats the check.", "",
               "## Checking a function", "",
               "```sh", "n64dk decompile-init ROM.z64 --out project",
               "n64dk match --project project --addr VRAM --size SIZE --pipeline PIPELINE --flags=\"FLAGS\" src/main/FUNCTION.c",
               "```", "", "with `VRAM`, `SIZE` (the row's `code_size`), pipeline and flags taken from the function's row in `FUNCTIONS.tsv`.", ""]
    coverage = rom_coverage(project / "survey" / "ledger.tsv", rom_size)
    warning = _denominator_warning(run)
    lines = [f"# Coverage: {name}", "", *([f"> **Warning.** {warning}", ""] if warning else []), "## Functions", "",
             "| reproduced by | functions |", "|---|---:|",
             *[f"| {kind} | {count:,} |" for kind, count in reproduced.items()],
             f"| **all** | **{total:,}** |", "", *_gp_note(gp_dependent),
             "Every function has a source that gives back its bytes when the last row is 0.  Only the first row is",
             "decompilation; the third is disassembly kept so that nothing is left out.", "",
             "## By unit class", "",
             "Boot code, RSP microcode, hand-written routines, SDK library code and game code are counted apart",
             "(`unit_class` in `FUNCTIONS.tsv`); no figure combines them.", "",
             "| unit class | " + " | ".join(reproduced) + " |", "|---" * (len(reproduced) + 1) + "|",
             *[f"| {unit} | " + " | ".join(f"{by_class.get(unit, {}).get(kind, 0):,}" for kind in reproduced) + " |"
               for unit in unitclass.CLASSES if unit in by_class], ""]
    lines += reasonsmod.table(tallies)
    if coverage:
        size = coverage["rom_bytes"] or 1
        lines += ["## Every byte of the ROM", "",
                  f"The survey's ledger (`config/ledger.tsv`) has {coverage['rows']:,} rows.  Bytes no row covers: "
                  f"{coverage['gap_bytes']:,}.  Bytes two rows claim: {coverage['overlap_bytes']:,}.", "",
                  "| class | bytes | share |", "|---|---:|---:|",
                  *[f"| {kind} | {count:,} | {count / size:.1%} |" for kind, count in coverage["bytes_by_class"].items()], ""]
        if coverage["largest_unclassified"]:
            lines += [f"`unknown` is content the survey has a row for and could not identify ({coverage['unclassified_ranges']:,} "
                      "ranges).  The largest, with what was measured about each:", "",
                      "| ROM range | bytes | measured |", "|---|---:|---|",
                      *[f"| {row['start']}..{row['end']} | {row['bytes']:,} | {row['evidence']} |"
                        for row in coverage["largest_unclassified"]], ""]
    (out / "COVERAGE.md").write_text("\n".join(lines))
    if warning:
        readme[2:2] = [f"> **Warning.** {warning}", ""]
    (out / "README.md").write_text("\n".join(readme))
    return {"title": name, "rom_sha1": rom_sha1, "functions": total, "matched": matched, "from_c": from_c,
            "code_rules": rules,
            "reproduced_by": reproduced, "rom": coverage and {k: coverage[k] for k in (
                "rom_bytes", "rows", "gap_bytes", "overlap_bytes", "bytes_by_class", "unclassified_ranges")},
            "counts": {key: counts[key] for key in order if counts.get(key)}, "unit_classes": by_class,
            "reasons": reasonsmod.summary(tallies), "unresolved": unresolved, "out": str(out)}


def _denominator_warning(run: Path) -> str | None:
    """The run's discovery verdict, when its catalogue is not known to be complete: every count in
    the export is over the catalogue, which then is not the title's code."""
    import json as _json
    report = Path(run) / "report.json"
    try:
        found = (_json.loads(report.read_text()).get("discovery") or {}) if report.is_file() else {}
    except ValueError:
        return None
    verdict = found.get("verdict")
    if verdict == "incomplete":
        return found.get("warning") or ("the function catalogue is incomplete: every count and percentage here is "
                                        "over the catalogue, not over the title's code")
    if verdict == "doubtful":
        return ("the function catalogue's completeness is doubtful (weak evidence either way): counts and "
                "percentages here are over the catalogue, which may miss code")
    return None


def proved_beside_waiting_donor(donor_outcome: str, case, has_source: bool) -> bool:
    """Whether a function inside a donor unit is credited to the resolver's own proof instead.

    Only when the donor unit still waits for an address (it recovers nothing yet) and the resolver's case is
    strictly proved (``exact`` or hand-written ``assembly``) with its source stored.  A verified donor unit
    keeps its functions; a case that is itself only byte-identical with assumed addresses changes nothing."""
    return (donor_outcome.endswith("awaiting-symbols") and case is not None
            and getattr(case, "status", None) in ("exact", "assembly") and has_source)


@command("decompile-export", "Write a finished decompile-run as a tree ready to commit: one source per matched "
                             "function, the function table, recipes and symbols (nothing of the ROM)")
def decompile_export(args):
    import os
    return export(Path(args.project), Path(args.run), Path(args.out), args.title, cover=not args.no_cover,
                  toolchain=args.toolchain_dir or os.environ.get("N64DK_TOOLCHAIN"))


def _args(p):
    p.add_argument("--project", required=True, help="the project the run was made on")
    p.add_argument("--run", required=True, help="the run's output directory (holds state/, queue/, report.json)")
    p.add_argument("--out", required=True, help="directory to write (created; existing files are overwritten)")
    p.add_argument("--title", help="title for the README (default: the ROM's header name)")
    p.add_argument("--no-cover", action="store_true",
                   help="do not write assembly for the functions that are not decompiled")
    p.add_argument("--toolchain-dir", help="toolchain root, for the assembler (default: N64DK_TOOLCHAIN)")


decompile_export.arguments = _args
