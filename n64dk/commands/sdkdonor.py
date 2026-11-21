"""``sdk-donor``: build the N64 SDK library from its source reconstruction as donor projects."""
from __future__ import annotations

import os
from pathlib import Path

from ..cli import command


@command("sdk-donor", "Compile a libultra source tree (ultralib layout) per version and flavour and describe each "
                      "as a donor for decompile-run --donors")
def sdk_donor(args):
    from ..match.toolchain import load_config
    from ..sdk import donor
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if not (source / "src").is_dir() or not (source / "include").is_dir():
        raise SystemExit(f"{source} does not look like a libultra source tree (no src/ and include/)")
    out.mkdir(parents=True, exist_ok=True)
    root = args.toolchain_dir or os.environ.get("N64DK_TOOLCHAIN")
    cfg = load_config(out, None, root)
    versions = [v.strip().upper() for v in args.versions.split(",") if v.strip()]
    flavours = [f.strip() for f in args.flavours.split(",") if f.strip()]
    unknown = [v for v in versions if v not in donor.VERSIONS] + [f for f in flavours if f not in donor.FLAVOURS]
    if unknown:
        raise SystemExit(f"unknown version or flavour: {', '.join(unknown)}")
    built, tables = [], []
    for version in versions:
        for flavour in flavours:
            for recipe in (args.recipe or [None]):
                forced = None
                if recipe:
                    parts = recipe.split()
                    forced = (parts[0], parts[1:])
                result = donor.build(source, out, version, flavour, cfg, jobs=args.jobs, recipe=forced)
                if result.get("toml") and not forced:           # a forced recipe is a corpus, not a donor
                    tables.append(result.pop("toml"))
                result.pop("toml", None)
                result["failed_units"] = len(result.get("failed") or [])
                result["failed"] = (result.get("failed") or [])[:5]
                built.append(result)
    donors = out / "donors.toml"
    donors.write_text("# Written by `n64dk sdk-donor`: the SDK library as donors.  Every transplant is rebuilt and "
                      "verified by the run that receives it.\n\n" + "\n".join(tables))
    return {"donors": str(donors), "built": built, "entries": len(tables)}


def _args(p):
    p.add_argument("--recipe", action="append", metavar="'PIPELINE LEVEL'",
                   help="build every C unit with this compiler and level instead of the library's own (repeatable; "
                        "e.g. 'kmc+as -O1'): the same sources under a known recipe, for `recipe-prior --learn-build`")
    p.add_argument("--source", required=True, help="libultra source tree (src/, include/, makefiles/, base/)")
    p.add_argument("--out", required=True, help="directory for the images, manifests and donors.toml")
    p.add_argument("--versions", default="D,E,F,G,H,I,J,K,L", help="comma-separated SDK versions (2.0D .. 2.0L)")
    p.add_argument("--flavours", default="libgultra_rom,libultra_rom,libgultra,libultra",
                   help="comma-separated builds: libgultra[_rom|_d] (GCC), libultra[_rom|_d] (IDO); the plain "
                        "builds (_FINALROM and _DEBUG undefined, NDEBUG) are the ones a title linking rmon "
                        "and the debug-port exception code was built with")
    p.add_argument("--toolchain-dir", help="toolchain root (default: N64DK_TOOLCHAIN)")
    p.add_argument("--jobs", type=int, default=4, help="compilers run at once")


sdk_donor.arguments = _args
