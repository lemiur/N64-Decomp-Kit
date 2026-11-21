"""``migrate PATH [--check]`` and ``release-info``: stored formats, their versions and migration.

``migrate`` takes a ``decompile-run`` directory, a content cache, a store (SQLite file) or a
versioned JSON document and reports, per item, ``current``, ``read-old`` (read as it is),
``migratable`` (a defined step exists), ``migrated``, ``absent``, ``refused`` (with why and what
to do) or ``unknown``.  ``--check`` changes nothing (read-only, sidecar-free SQLite); without it the
defined steps are applied, each atomically (a JSON file keeps a ``.v<N>.bak`` copy).  Exit status 3
when anything is refused or unknown, 2 when ``--check`` finds a pending migration.

``release-info`` prints the version (and whether ``pyproject.toml`` agrees), the format registry
with each current version, and the release checklist (``n64dk.migrate``)."""
from __future__ import annotations

from pathlib import Path

from ..cli import command, emit
from .. import migrate as mig


@command("migrate", "Check (--check) or migrate a run directory, content cache, store or versioned JSON document "
                    "to this kit's formats; refuse with an explanation what cannot be read")
def migrate_command(args):
    report = mig.migrate(Path(args.path), check=args.check)
    code = 3 if not report["ok"] else 2 if args.check and report["pending"] else 0
    if getattr(args, "format", "text") == "text":
        print(f"{report['target']} ({report['kind']}): " + ", ".join(f"{k} {v}" for k, v in report["counts"].items()))
        for item in report["items"]:
            print(f"  {item['file']}: {item['format']} {item['status']}: {item['explanation']}"
                  + (f" [{item['action']}]" if item.get("action") else ""))
        return code
    if code:
        emit(report, "json")
        return code
    return report


def _migrate_args(p):
    p.add_argument("path", help="a run directory, a content cache root, a .sqlite store or a JSON document")
    p.add_argument("--check", action="store_true", help="report only; change nothing")


migrate_command.arguments = _migrate_args


@command("release-info", "Version, format registry with current versions, and the release checklist")
def release_info(_args):
    return {"schema": "n64dk.release-info/1", **mig.version_info(), "formats": mig.registry(),
            "checklist": list(mig.RELEASE_CHECKLIST)}
