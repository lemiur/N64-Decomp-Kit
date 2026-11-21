"""An ``objdiff.json`` project view for a tree that holds per-unit objects.

objdiff (github.com/encounter/objdiff) compares a *target* object (built from the original
assembly) with a *base* object (built from the decompiled source) unit by unit.  Its project file
lists the units.  Fields written here, and how sure this module is of each:

======================  =============================================================  ==========
field                   meaning                                                        confidence
======================  =============================================================  ==========
``units``               list of units                                                  documented
``units[].name``        unit name shown by objdiff                                     documented
``units[].target_path`` object built from the original (expected) code                 documented
``units[].base_path``   object built from the decompiled source (omitted if missing)   documented
``units[].metadata``    ``complete`` / ``source_path``                                 unverified
``build_base``          objdiff runs the build for base objects                        unverified
``build_target``        objdiff runs the build for target objects                      unverified
======================  =============================================================  ==========

"documented" are the fields of objdiff's published project configuration that this module relies
on; "unverified" fields were written from memory of that schema and were not checked against a
copy of it in this round (no network, no objdiff binary).  ``UNVERIFIED_FIELDS`` names them, and
``project(..., minimal=True)`` leaves them out.  Paths are relative to the project root, POSIX
style.  The kit's ``decompile-export`` tree holds sources, not objects: ``from_export`` says so
instead of inventing object paths.
"""
from __future__ import annotations

import json
from pathlib import Path

UNVERIFIED_FIELDS = ("units[].metadata", "build_base", "build_target")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def units_from_tree(root, target_dir: str, base_dir: str, source_dir: str | None = None,
                    source_suffixes=(".c", ".s")) -> list[dict]:
    """One unit per object under ``root/target_dir`` (``**/*.o``); its base object is the same
    relative path under ``base_dir`` when that file exists."""
    root = Path(root)
    target_root, base_root = root / target_dir, root / base_dir
    units = []
    for obj in sorted(target_root.rglob("*.o")) if target_root.is_dir() else []:
        rel = obj.relative_to(target_root)
        unit = {"name": rel.with_suffix("").as_posix(), "target_path": _rel(obj, root)}
        base = base_root / rel
        if base.is_file():
            unit["base_path"] = _rel(base, root)
        if source_dir is not None:
            for suffix in source_suffixes:
                source = root / source_dir / rel.with_suffix(suffix)
                if source.is_file():
                    unit["source_path"] = _rel(source, root)
                    break
        units.append(unit)
    return units


def project(units: list[dict], *, minimal: bool = False, build_base: bool = False,
            build_target: bool = False) -> dict:
    """The objdiff project document.  Unit dicts carry name, target_path, optional base_path,
    optional source_path and optional complete (bool)."""
    out_units = []
    for unit in units:
        item = {"name": unit["name"], "target_path": unit["target_path"]}
        if unit.get("base_path"):
            item["base_path"] = unit["base_path"]
        if not minimal:
            metadata = {}
            if "complete" in unit:
                metadata["complete"] = bool(unit["complete"])
            if unit.get("source_path"):
                metadata["source_path"] = unit["source_path"]
            if metadata:
                item["metadata"] = metadata
        out_units.append(item)
    doc: dict = {}
    if not minimal:
        doc["build_base"] = build_base
        doc["build_target"] = build_target
    doc["units"] = out_units
    return doc


def write(path, document: dict, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass force=True to replace it")
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def from_export(export_dir, target_dir: str = "build/expected", base_dir: str = "build/base",
                source_dir: str | None = "src") -> dict:
    """objdiff view of an export tree when it has per-unit objects; otherwise the reason why not."""
    root = Path(export_dir)
    units = units_from_tree(root, target_dir, base_dir, source_dir)
    if not units:
        return {"generated": False, "units": 0,
                "reason": f"no per-unit objects under {target_dir}/: the export holds sources and symbol files only; "
                          "build target objects there (and base objects under " + base_dir + "/) first"}
    return {"generated": True, "units": len(units), "with_base": sum("base_path" in u for u in units),
            "document": project(units), "unverified_fields": list(UNVERIFIED_FIELDS)}
