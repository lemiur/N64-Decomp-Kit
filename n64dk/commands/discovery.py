"""``discovery-audit``: is the function catalogue complete enough to decompile against?

Reads the project (images, catalogues, survey files) and writes nothing unless ``--out`` is
given; with ``--repair`` it writes what it repairs there (never into the project): further
segments placed from the ROM by the call targets no image holds, and catalogues with the
functions the evidence shows are missing."""
from __future__ import annotations

import json
from pathlib import Path

from ..cli import command

PLACE_ROUNDS = 3


def render(result: dict) -> str:
    lines = [f"discovery audit of {result['project']}: {result['verdict']}"]
    for placed in result.get("placed") or []:
        lines.append(f"  placed {placed['name']}: ROM {placed['rom']}..{placed['rom_end']} at {placed['vram']} "
                     f"({placed['landed']} call targets land on its function starts, {placed['functions']} functions)")
    for name, img in result["images"].items():
        if "coverage" not in img:
            lines.append(f"  {name}: {img.get('verdict')}: {'; '.join(img.get('reasons') or [])}")
            continue
        calls = img["direct_calls"]
        lines.append(f"  {name}: {img['verdict']}, {img['functions']} functions, coverage {img['coverage']:.1%} "
                     f"of {img['executable_bytes']} executable bytes; call targets missing {calls['missing']}, "
                     f"interior {calls['interior']}, unmapped {calls['unmapped']}; framed functions outside "
                     f"{img['prologues_outside']['walk_as_functions']}")
        for reason in img.get("reasons") or []:
            lines.append(f"    - {reason}")
        for doubt in img.get("doubts") or []:
            lines.append(f"    ? {doubt}")
        if img.get("repair"):
            rep = img["repair"]
            lines.append(f"    repaired: +{rep['added']} functions, {rep['bytes']} bytes; verdict after: {rep['verdict_after']}")
            for reason in rep.get("reasons_after") or []:
                lines.append(f"    - after repair: {reason}")
    for reason in result.get("rom_reasons") or []:
        lines.append(f"  ROM: {reason}")
    if result.get("verdict_before") and result["verdict_before"] != result["verdict"]:
        lines.append(f"  before repair: {result['verdict_before']}")
    return "\n".join(lines)


def _place(project: Path, found: dict, out: Path, extra: dict) -> list[dict]:
    """Segments placed from the ROM by the call targets no image holds (``coverage.place_unmapped``),
    each written to ``out`` with its catalogue and added to ``extra`` as an ImageSpec."""
    from ..analysis import coverage
    from ..analysis import functions as fnmod
    from ..match.images import ImageSpec
    from ..rom import Rom
    rom_path = project / "baserom.z64"
    targets = found.get("_unmapped") or set()
    if not rom_path.exists() or len(targets) < 3:
        return []
    rom = Rom.load(rom_path)
    taken = coverage.taken_rom(project) + [(spec._rom[0], spec._rom[1]) for spec in extra.values()
                                           if getattr(spec, "_rom", None)]
    placed = []
    for row in coverage.place_unmapped(rom.data, targets, taken, candidates=coverage.unplaced_code(project)):
        name = f"seg_{row['rom']:X}"
        data = rom.data[row["rom"]:row["rom_end"]]
        catalogue = coverage.catalogue_for(rom, data, row["vram"], seeds=row["landed"])
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.bin").write_bytes(data)
        (out / f"functions_{name}.toml").write_text(fnmod.to_toml(catalogue, name))
        (out / f"symbol_addrs_{name}.txt").write_text(fnmod.to_symbol_addrs(catalogue))
        spec = ImageSpec(name, out / f"{name}.bin", row["vram"], out / f"functions_{name}.toml")
        spec._rom = (row["rom"], row["rom_end"])
        extra[name] = spec
        placed.append({"name": name, "rom": f"0x{row['rom']:X}", "rom_end": f"0x{row['rom_end']:X}",
                       "vram": f"0x{row['vram']:08X}", "votes": row["votes"], "landed": len(row["landed"]),
                       "inside": row["inside"], "functions": len(catalogue.functions), "kind": row["kind"],
                       "prologues": row["prologues"], "internal_calls": row["internal_calls"],
                       "bin": str(out / f"{name}.bin"), "catalogue": str(out / f"functions_{name}.toml")})
    return placed


def write_rows(path: Path, rows, like: Path) -> None:
    """A repaired catalogue in the original's format, or a plain TOML table when the original has
    no function to take the format from."""
    from ..analysis import boundaries as boundmod
    try:
        boundmod.write_catalogue(path, rows, like)
    except ValueError:
        lines = ["# Function boundaries repaired by the discovery audit", "functions = ["]
        lines += [f'    {{ name = "{n}", vram = 0x{v:08x}, size = 0x{size:x} }},' for n, v, size in rows]
        path.write_text("\n".join(lines + ["]"]) + "\n")


def run_audit(project: Path, images=None, out: Path | None = None, repair: bool = False) -> dict:
    """Audit; with ``repair`` (needs ``out``), place unmapped code, then add the missing functions
    to every incomplete catalogue, and audit again.  ``extra_images`` in the result names the
    segments placed (name -> bin, vram, catalogue)."""
    from ..analysis import coverage
    from ..match import images as imagemod
    project = Path(project).resolve()
    found = coverage.audit_project(project, images)
    before = coverage_verdict(found)
    extra: dict = {}
    placed: list[dict] = []
    if repair and out is not None:
        for _ in range(PLACE_ROUNDS):
            new = _place(project, found, out, extra)
            if not new:
                break
            placed += new
            found = coverage.audit_project(project, images, extra)
        registry = imagemod.load_registry(project)
        specs = {}
        for name in registry:
            try:
                specs[name] = imagemod.resolve(project, name)
                specs[name].load()
                coverage.rows_of(specs[name])
            except Exception:                                # noqa: BLE001 - not loadable: audited as such
                specs.pop(name, None)
        specs.update(extra)
        bounds = {name: (*coverage.program_extent(project, name, spec, spec.load()),
                         {v for _n, v, _s in coverage.rows_of(spec)}) for name, spec in specs.items()}
        for name, img in found["images"].items():
            if img.get("verdict") != "incomplete" or name not in specs:
                continue
            spec = specs[name]
            rows, added, audits = coverage.repair(
                name, spec.load(), spec.delta, coverage.rows_of(spec),
                others={k: v for k, v in bounds.items() if k != name},
                text_ranges=coverage._text_ranges(project, name), extent=bounds[name][:2])
            out.mkdir(parents=True, exist_ok=True)
            path = None
            if added and spec.catalogue:
                path = out / f"functions_{name}_repaired{spec.catalogue.suffix}"
                if path.exists():
                    path.unlink()
                write_rows(path, [(n, v + spec.catalogue_shift, s) for n, v, s in rows], spec.catalogue)
                if name in extra:                            # a placed segment: its catalogue is ours
                    path.replace(spec.catalogue)
                    path = None
                    extra[name]._rows = None
            (out / f"repair_{name}.json").write_text(json.dumps({"added": added, "audits": audits}, indent=1) + "\n")
            img["repair"] = {"added": sum(1 for a in added if "moved_from" not in a),
                             "moved_starts": sum(1 for a in added if "moved_from" in a),
                             "bytes": sum(a["size"] for a in added),
                             "verdict_after": audits[-1]["verdict"], "coverage_after": audits[-1]["coverage"],
                             "reasons_after": audits[-1]["reasons"], "functions_after": audits[-1]["functions"],
                             "catalogue": str(path) if path else None}
    found.pop("_unmapped", None)
    if placed and found.get("rom"):                         # what got placed is no longer unplaced
        ranges = [(int(p["rom"], 16), int(p["rom_end"], 16)) for p in placed]
        left = [(lo, hi) for lo, hi in coverage.unplaced_code(project)
                if not any(lo < b and a < hi for a, b in ranges)]
        found["rom"]["code_not_placed"] = len(left)
        found["rom"]["code_not_placed_bytes"] = sum(hi - lo for lo, hi in left)
        found["rom_reasons"] = ([f"{found['rom']['code_not_placed_bytes']} bytes of code-like ROM placed in no image"]
                                if left else [])
    found["placed"] = placed
    found["extra_images"] = {name: {"bin": str(spec.path), "vram": f"0x{spec.vram_delta:08X}",
                                    "catalogue": str(spec.catalogue)} for name, spec in extra.items()}
    found["verdict_before"] = before
    found["verdict"] = coverage_verdict(found, after=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "discovery_audit.json").write_text(json.dumps(found, indent=1) + "\n")
    return found


def coverage_verdict(found: dict, after: bool = False) -> str:
    verdicts = [((img.get("repair") or {}).get("verdict_after") if after else None) or img.get("verdict")
                for img in found["images"].values()]
    return ("incomplete" if "incomplete" in verdicts else
            "doubtful" if "doubtful" in verdicts or found.get("rom_reasons") else "complete")


@command("discovery-audit", "Check a project's function catalogues against call, pointer, prologue and "
                            "datamap evidence; incomplete catalogues fail the gate (read-only)")
def discovery_audit(args):
    result = run_audit(Path(args.project).resolve(), args.image or None,
                       Path(args.out).resolve() if args.out else None, args.repair)
    if getattr(args, "overlays", False):
        result["overlays"] = overlay_report(Path(args.project).resolve(), args.image or None)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "overlays.json").write_text(json.dumps(result["overlays"], indent=1) + "\n")
    if getattr(args, "indirect", False):
        result["indirect"] = indirect_report(Path(args.project).resolve(), args.image or None)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "indirect.json").write_text(json.dumps(result["indirect"], indent=1) + "\n")
    if getattr(args, "format", "text") == "text":
        print(render(result))
        for name, rep in (result.get("overlays") or {}).items():
            print(f"overlays {name}: {rep['moved_runs']} run(s) would move; calls landing on a start "
                  f"{rep['calls_landing_now']} now, {rep['calls_landing_inferred']} inferred (report only)")
        for name, rep in (result.get("indirect") or {}).items():
            if "error" in rep:
                print(f"indirect {name}: not analysed: {rep['error']}")
                continue
            status = rep["by_status"]
            print(f"indirect {name}: {rep['jr_sites']} jr + {rep['jalr_sites']} jalr sites: "
                  + ", ".join(f"{k} {v}" for k, v in status.items())
                  + f"; {rep['switch_tables']} switch tables ({rep['switch_table_bytes']} bytes, "
                  f"{rep['tables_inside_catalogued_functions']} inside catalogued functions); "
                  f"{rep['no_return']} non-returning functions, {rep['fake_fallthroughs_into_code']} "
                  f"fake fall-throughs into code (report only)")
        return 0 if result["verdict"] != "incomplete" or args.no_fail else 3
    return result


def _args(p):
    p.add_argument("--project", required=True, help="project directory (read only)")
    p.add_argument("--image", action="append", help="only this image (repeatable)")
    p.add_argument("--out", help="write discovery_audit.json (and what --repair makes) here")
    p.add_argument("--repair", action="store_true",
                   help="place code the title calls but no image holds, and add evidence-backed functions to "
                        "incomplete catalogues; writes under --out, never into the project")
    p.add_argument("--no-fail", action="store_true", help="exit 0 even when a catalogue is incomplete")
    p.add_argument("--overlays", action="store_true",
                   help="also infer per-overlay bases inside each image (a segment holding several separately "
                        "linked overlays); a report only, never applied to a catalogue")
    p.add_argument("--indirect", action="store_true",
                   help="also resolve jr/jalr targets (switch tables, pointer tables, stored pointers) and "
                        "non-returning functions (n64dk.analysis.indirect); a report only, never applied "
                        "to a catalogue")


def overlay_report(project: Path, images=None) -> dict:
    """``n64dk.analysis.overlays.report`` for each image (catalogue starts and prologue evidence)."""
    from ..analysis import coverage, overlays
    from ..match import images as imagemod
    out = {}
    for name in sorted(imagemod.load_registry(project)):
        if images and name not in images:
            continue
        spec = imagemod.resolve(project, name)
        data = spec.load()
        starts = {vram - spec.delta for _n, vram, _s in spec.rows() if 0 <= vram - spec.delta < len(data)}
        out[name] = overlays.report(data, spec.delta, starts | coverage._start_offsets(data))
    return out


def indirect_report(project: Path, images=None) -> dict:
    """``n64dk.analysis.indirect.analyse_image(...).summary()`` for each image (its catalogue)."""
    from ..analysis import coverage, indirect
    from ..match import images as imagemod
    out = {}
    for name in sorted(imagemod.load_registry(project)):
        if images and name not in images:
            continue
        try:
            spec = imagemod.resolve(project, name)
            data = spec.load()
            rows = coverage.rows_of(spec)
        except Exception as error:                        # noqa: BLE001 - reported, not fatal
            out[name] = {"error": str(error)[:200]}
            continue
        out[name] = indirect.analyse_image(data, spec.delta, rows).summary()
    return out


discovery_audit.arguments = _args
