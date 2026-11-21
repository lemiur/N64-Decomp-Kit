"""``n64dk decompile-project``: a finished export (and link-layout output) as a maintainable project tree.

See :mod:`n64dk.match.projectgen`.  The tree holds no byte of the ROM: units nothing reproduces
yet are described by offset, size and SHA-256 and cut from the user's own ROM at build time.
Regenerating into the same directory rewrites only what changed and never overwrites a file
changed by hand.  A generated tree grants no credit; ``make verify`` checks it against a ROM.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ..cli import command


@command("decompile-project", "Turn a decompile-export tree (plus link-layout output) into a maintainable project "
                              "tree with a Makefile, recipes, receipts and per-unit status (no ROM bytes; no credit)")
def decompile_project(args):
    from ..match import projectgen as pg
    try:
        export = pg.resolve_export(args.source)
    except pg.GenerationError as error:
        sys.stderr.write(f"decompile-project refused: {error}\n")
        return 2
    images, roms = {}, []
    if args.project:
        from ..match import images as imagemod
        for name, spec in sorted(imagemod.load_registry(args.project).items()):
            try:
                images[name] = (spec.load(), spec.delta)       # read only: raw units' digests, never copied
            except (OSError, ValueError, LookupError):
                continue
            if Path(spec.path).suffix.lower() in pg.ROM_SUFFIXES:
                roms.append(Path(spec.path))
    lock = None
    if not args.no_toolchain_lock:
        from ..match import toolmanifest
        root = args.toolchain_dir or os.environ.get("N64DK_TOOLCHAIN")
        lock = toolmanifest.build(Path(root) if root else None)
    try:
        report = pg.generate(export, Path(args.out), link_layout=Path(args.link_layout) if args.link_layout else None,
                             images=images, toolchain_lock=lock, force=args.force,
                             protected=[Path(args.project)] if args.project else [], roms=roms)
    except pg.GenerationError as error:
        sys.stderr.write(f"decompile-project refused: {error}\n")
        return 2
    report["written"] = report["written"][:args.limit]
    report["removed"] = report["removed"][:args.limit]
    return report


def _args(p):
    p.add_argument("source", help="a decompile-export tree (or a run directory holding one in export/)")
    p.add_argument("--out", required=True, help="project directory to create, or a generated project to update")
    p.add_argument("--link-layout", metavar="DIR", help="`n64dk link-layout --out` directory (link.ld, objects.tsv)")
    p.add_argument("--project", help="the decompile-init project of the ROM: read only, for the raw units' "
                                     "digests and to keep the output apart from it and its ROM")
    p.add_argument("--toolchain-dir", help="toolchain root for the lock file (default: N64DK_TOOLCHAIN)")
    p.add_argument("--no-toolchain-lock", action="store_true", help="do not hash the toolchain")
    p.add_argument("--force", action="store_true", help="overwrite or remove files changed by hand since generation")
    p.add_argument("--limit", type=int, default=50, help="paths listed per category in the report")


decompile_project.arguments = _args
