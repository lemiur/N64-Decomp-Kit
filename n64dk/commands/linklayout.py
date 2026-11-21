"""``n64dk link-layout``: the build graph and the inferred original link order of a project.

Writes, with ``--out DIR``: ``build_graph.json`` (segments, units, pieces, constraints with
provenance, symbols, holes, hints), ``link_order.json`` (every retained ordering summarised:
representative order, position ranges, ambiguous pairs, per-step confidence, contradictions),
``objects.tsv`` (the object manifest) and ``link.ld`` (a GNU ld script for the representative
order, alternatives and contradictions as comments).  An inference: it grants no match credit.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..analysis import linklayout as ll
from ..cli import command


@command("link-layout", "Infer the build graph and original link order (a hypothesis; grants no match credit)")
def link_layout(args):
    graph = ll.from_project(args.project, images=args.image or None, sdk=not args.no_sdk,
                            source_names=not args.no_source_names, max_align=args.max_align)
    solution = ll.solve(graph, limit=args.limit, budget=args.budget)
    out = ll.summary(graph, solution)
    out["evidence"] = graph.evidence
    if args.out:
        directory = Path(args.out)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "build_graph.json").write_text(json.dumps(graph.as_dict(), indent=1, default=str) + "\n")
        (directory / "link_order.json").write_text(json.dumps(solution, indent=1, default=str) + "\n")
        ll.write_manifest(ll.manifest(graph, solution), directory / "objects.tsv")
        (directory / "link.ld").write_text(ll.linker_script(graph, solution))
        out["out"] = str(directory)
    return out


def _args(p):
    p.add_argument("--project", required=True, help="project directory (n64dk.toml)")
    p.add_argument("--image", action="append", metavar="NAME", help="image to include (repeatable; default: all)")
    p.add_argument("--out", metavar="DIR", help="write build_graph.json, link_order.json, objects.tsv, link.ld")
    p.add_argument("--limit", type=int, default=ll.DEFAULT_LIMIT, help="orderings kept per segment")
    p.add_argument("--budget", type=int, default=ll.DEFAULT_BUDGET, help="search steps per segment enumeration")
    p.add_argument("--max-align", type=int, default=16, help="largest input-section alignment a gap may be fill for")
    p.add_argument("--no-sdk", action="store_true", help="do not use SDK archive membership")
    p.add_argument("--no-source-names", action="store_true", help="do not use assertion-string source names")


link_layout.arguments = _args
