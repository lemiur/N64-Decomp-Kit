"""The model-in-the-loop commands: the tool protocol (``agent-tools``), work orders
(``agent-order``) and the model stage (``agent-stage``)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..cli import command


def _hex(text: str) -> int:
    return int(text, 0) if text.lower().startswith("0x") else int(text, 16)


@command("agent-tools", "Tool protocol for a model or person matching one function: dis, draft, try, exact, info, "
                        "regdump, neighbours, boundary")
def agent_tools(args):
    from ..match import agenttools as tools
    files = list(args.files)
    opts = dict(flags=args.flags, pipe=args.pipe, show=args.show, lines=args.lines, count=args.count)
    if args.tool == "regdump" and files and re.fullmatch(r"[lgLsjSrc]+", files[-1]) and not files[-1].endswith(".c"):
        opts["dumps"] = files.pop()
    try:
        if args.session:
            result = tools.session_call(Path(args.session), args.tool, files, size=args.size, **opts)
        else:
            if not (args.image and args.function):
                raise tools.ToolError("give --session, or --image and --function")
            ws = tools.Workspace(args.project, args.state)
            result = tools.run_tool(ws, args.tool, args.image, args.function, files, size=args.size, **opts)
    except tools.ToolError as error:
        print(f"ERROR: {error}")
        return 2
    if getattr(args, "format", "text") == "json":
        return {k: v for k, v in result.items() if k != "lines"} | {"lines": result.get("lines", [])}
    for line in result.get("lines", []):
        print(line)
    return 0


def _tools_args(p):
    p.add_argument("tool", choices=("dis", "draft", "try", "exact", "info", "regdump", "neighbours", "boundary"))
    p.add_argument("files", nargs="*")
    p.add_argument("--session", help="a model-stage work folder's session.json (function, project, budget)")
    p.add_argument("--project", default=".")
    p.add_argument("--state", help="state folder (default <project>/.n64dk/agent)")
    p.add_argument("--image")
    p.add_argument("--function", help="real address (hex) or catalogue name")
    p.add_argument("--size", type=_hex, help="true size: verify a repaired range starting at the address")
    p.add_argument("--flags", help="compiler flags replacing the recipe's (write --flags='-O2 ...')")
    p.add_argument("--pipe", help="lane, e.g. kmc+builtin, kmc+asn64, ido5.3")
    p.add_argument("--show", type=int, help="try: hunks shown")
    p.add_argument("--lines", type=int, help="exact: differing lines shown")
    p.add_argument("--count", type=int, help="neighbours: sources shown")


agent_tools.arguments = _tools_args


def _functions(ws, args):
    from ..match import agenttools as tools
    out = []
    for item in args.function or []:
        image, _, rest = item.partition(":")
        key, _, size = rest.partition(":")
        out.append(ws.lookup(image, key, _hex(size) if size else None))
    if args.list:
        for line in Path(args.list).read_text().splitlines():
            f = line.split("\t")
            if not line.strip() or line.startswith("#") or len(f) < 2:
                continue
            try:
                out.append(ws.lookup(f[0], f[1], _hex(f[2]) if len(f) > 2 and f[2].strip() else None))
            except (tools.ToolError, ValueError) as error:
                print(f"skipped {f[0]}:{f[1]}: {error}")
    if args.remaining:
        from ..match.inventory import build, rank
        for row in rank(build(ws.root), limit=args.remaining):
            try:
                out.append(ws.lookup(row["image"], row["name"]))
            except tools.ToolError:
                continue
    return out


@command("agent-order", "Write the model stage's work order for one function (no model is called)")
def agent_order(args):
    from ..match import agenttools as tools
    from ..match import agentstage as stage
    ws = tools.Workspace(args.project, args.state)
    fn = ws.lookup(args.image, args.function, args.size)
    folder = Path(args.out).resolve()
    order = stage.build_work_order(ws, fn, folder)
    (folder / "ORDER.md").write_text(order.render())
    print(f"wrote {folder / 'ORDER.md'} (category {order.category}; lanes {len(order.lanes)}; "
          f"draft {order.draft.get('best_file')} {order.draft.get('best_hunks')})")
    return 0


def _order_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--state")
    p.add_argument("--image", required=True)
    p.add_argument("--function", required=True)
    p.add_argument("--size", type=_hex)
    p.add_argument("--out", required=True)


agent_order.arguments = _order_args


@command("agent-stage", "Model stage: an agent loop per unresolved function, accepted only by strict verification")
def agent_stage(args):
    from ..match import agenttools as tools
    from ..match import agentstage as stage
    ws = tools.Workspace(args.project, args.state)
    functions = _functions(ws, args)
    provider = stage.provider_from_config(ws, args.provider, args.transcript)
    budget = stage.Budget(args.attempts, args.seconds, args.tool_calls, args.total_seconds)

    def emit(event, **fields):
        print(json.dumps({"event": event, **fields}), flush=True)

    summary = stage.run_stage(ws, functions, provider, budget, emit)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    return 0


def _stage_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--state")
    p.add_argument("--function", action="append", help="IMAGE:ADDR[:SIZE] (repeatable)")
    p.add_argument("--list", help="TSV: image, address, [true size]")
    p.add_argument("--remaining", type=int, help="take this many remaining functions from the inventory")
    p.add_argument("--provider", choices=("command", "dry-run", "none"), help="default: the configured command")
    p.add_argument("--transcript", help="dry-run: recorded transcript JSON")
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--seconds", type=float, default=1800.0)
    p.add_argument("--tool-calls", type=int, default=80)
    p.add_argument("--total-seconds", type=float, default=0.0)


agent_stage.arguments = _stage_args
