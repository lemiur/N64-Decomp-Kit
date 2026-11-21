"""Reproducible toolchain and supply-chain manifest (round 12, item T).

``build`` describes every external component the kit can drive, without machine paths: identity,
version, origin and source location, licence note, the files a toolchain root holds of it with
their SHA-256 digests and formats, the compatibility layer it runs under, every recipe's identity
(``components.recipe_identity``), the host platform, and, when given, the digests of probe outputs
(assembler macro probes, dialect probes): what the components *do*, not only what they are.

``compare`` takes two manifests (two machines, or one machine on two dates) and lists every
precise difference: a component installed on one side only, a file whose digest differs, a recipe
whose identity differs, a probe whose output digest differs.  Two machines with the same component
digests and different probe digests have a host-side difference (wrapper, emulator, host cpp),
which the probe names.

Supply classes, kept apart (``supply_class``):

``preserved-original``     an original binary run as shipped, natively or under an emulator layer
``static-recompilation``   an original binary translated to native code (IDO static recomp)
``source-rebuild``         a tool rebuilt from source (GNU binutils/GCC builds)
``reconstructed``          a reimplementation meant to behave like an original (object converters,
                           kit-internal assembler macro emulation)
``comparative``            a relative from another platform: never an N64 lane by itself
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from . import components as comp

SCHEMA = 1


def supply_class(row: dict) -> str:
    """The supply class of one ``components.inspect`` row (see the module docstring)."""
    provenance = row.get("provenance", "")
    text = " ".join(str(row.get(k, "")) for k in ("origin", "source_url", "emulator", "notes", "platform")).lower()
    if provenance == "comparative":
        return "comparative"
    if provenance == "reconstructed" or str(row.get("origin", "")).startswith("n64dk/"):
        if "rebuil" in text or "source" in text and "built" in text:
            return "source-rebuild"
        return "reconstructed"
    if "static-recomp" in text or "static recompilation" in text or "(recompiled)" in text:
        return "static-recompilation"
    if provenance.startswith("authentic"):
        return "preserved-original"
    return "reconstructed"


def build(root: Path | None, probes: dict[str, str] | None = None, fetched: dict[str, str] | None = None) -> dict:
    """The manifest for toolchain root ``root`` (None: the catalogue alone).  ``probes`` maps a
    probe name to the digest of its outputs; ``fetched`` maps a component to the image manifest
    digest it was fetched from (``compilers.fetch`` returns it)."""
    rows = comp.inspect(root)
    components = []
    for row in rows:
        components.append({
            "ident": row["ident"], "role": row["role"], "name": row["name"], "version": row["version"],
            "provenance": row["provenance"], "disposition": row["disposition"], "supply_class": supply_class(row),
            "origin": row["origin"], "source_url": row.get("source_url", ""), "licence": row.get("licence", ""),
            "platform": row.get("platform", ""), "emulator": row.get("emulator", ""),
            "installed": row["installed"],
            "files": sorted(({"file": f["file"], "sha256": f.get("sha256"), "bytes": f["bytes"], "format": f["format"]}
                             for f in row["found"]), key=lambda f: f["file"]),
            **({"fetched_manifest": fetched[row["ident"]]} if fetched and row["ident"] in fetched else {})})
    recipes = {}
    for lane, recipe in sorted(comp.RECIPES.items()):
        r = comp.recipe_identity(recipe, root, inspected=rows)
        recipes[lane] = {"identity": r["identity"], "runnable": r["runnable"]}
    manifest = {"schema": SCHEMA,
                "host": {"system": platform.system(), "machine": platform.machine(),
                         "python": ".".join(map(str, sys.version_info[:3]))},
                "components": components, "recipes": recipes, "probes": dict(sorted((probes or {}).items())),
                "host_components": comp.HOST_COMPONENTS}
    manifest["digest"] = digest(manifest)
    return manifest


def digest(manifest: dict) -> str:
    """SHA-256 over the manifest without its own digest and without the host description (two
    hosts with the same components and probe outputs share the digest)."""
    material = {k: v for k, v in manifest.items() if k not in ("digest", "host")}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def compare(a: dict, b: dict) -> list[dict]:
    """Every precise difference between two manifests, most specific first."""
    out = []
    left = {c["ident"]: c for c in a.get("components", [])}
    right = {c["ident"]: c for c in b.get("components", [])}
    for ident in sorted(set(left) | set(right)):
        x, y = left.get(ident), right.get(ident)
        if x is None or y is None:
            out.append({"kind": "component-unknown", "component": ident, "a": x is not None, "b": y is not None})
            continue
        if x["installed"] != y["installed"]:
            out.append({"kind": "component-installed", "component": ident, "a": x["installed"], "b": y["installed"]})
        fx = {f["file"]: f for f in x["files"]}
        fy = {f["file"]: f for f in y["files"]}
        for name in sorted(set(fx) | set(fy)):
            dx, dy = (fx.get(name) or {}).get("sha256"), (fy.get(name) or {}).get("sha256")
            if dx != dy:
                out.append({"kind": "file-digest", "component": ident, "file": name, "a": dx, "b": dy})
    rx, ry = a.get("recipes", {}), b.get("recipes", {})
    for lane in sorted(set(rx) | set(ry)):
        ix, iy = (rx.get(lane) or {}).get("identity"), (ry.get(lane) or {}).get("identity")
        if ix != iy:
            out.append({"kind": "recipe-identity", "recipe": lane, "a": ix, "b": iy})
    px, py = a.get("probes", {}), b.get("probes", {})
    for probe in sorted(set(px) | set(py)):
        if px.get(probe) != py.get(probe):
            out.append({"kind": "probe-output", "probe": probe, "a": px.get(probe), "b": py.get(probe),
                        "note": "same components, different behaviour: a host-side difference (wrapper, "
                                "emulator, host preprocessor)" if not any(o["kind"] == "file-digest" for o in out)
                        else ""})
    if a.get("host") != b.get("host"):
        out.append({"kind": "host", "a": a.get("host"), "b": b.get("host")})
    return out


def probe_digests(models_dir: Path) -> dict[str, str]:
    """Probe output digests from a directory of probe result JSON files (``bench/macros/models``
    style: each file holds ``{"expansions": [...]}``): SHA-256 over the expansions' words."""
    out = {}
    for path in sorted(Path(models_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        expansions = data.get("expansions") if isinstance(data, dict) else None
        if isinstance(expansions, list):
            material = [(e.get("form"), e.get("words") or e.get("text"), e.get("error")) for e in expansions]
            out[path.stem] = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return out
