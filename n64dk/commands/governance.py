"""``corpus-check``, ``blind-check`` and ``shared-library-pin``: corpus governance and blind isolation
(``n64dk.governance``).

``corpus-check MANIFEST`` validates a corpus manifest (classes, flags no looser than the class,
licences) and, with ``--digests``, compares each entry's recorded tree digest with the tree now.

``blind-check`` writes the isolation receipt of a blind run before it starts (or audits a finished
one from its ``solver/invocation.json``): the forbidden trees come from ``--manifest`` (its
blind-excluded entries for ``--title``) and ``--forbid``; the run's configuration from
``--invocation``, ``--project``, ``--donors``, ``--library`` and the ``N64DK_*`` environment.  It
exits 3 when a forbidden tree is reachable.  ``--shared-library MANIFEST@sha256:<hex>`` (opt-in,
repeatable) declares a shared public library: a content copy whose SHA-256 the pinned manifest lists
is exempt and listed in the receipt; an unpinned, mismatching or invalid manifest is refused (exit 2)
before anything is checked.  Nothing is written but ``--out``.

``shared-library-pin ROOT --name N --provenance URL --out MANIFEST`` writes a manifest pinning every
regular file of a library tree (``.git`` and symlinks skipped) and prints the digest to pin it with."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from ..cli import command, emit
from .. import governance as gov


@command("corpus-check", "Validate a corpus manifest (data classes, licences, permitted uses, blind exclusion)")
def corpus_check(args):
    manifest = gov.Manifest.load(args.manifest)
    result = {"schema": gov.SCHEMA, "entries": len(manifest.entries),
              "classes": sorted({e.data_class for e in manifest.entries}),
              "blind_excluded": [label for label, _root in manifest.forbidden_for(getattr(args, "title", None))]}
    if getattr(args, "digests", False):
        result["digests"] = manifest.verify_digests()
    return result


def _corpus_args(p):
    p.add_argument("manifest", help="a n64dk.corpus/1 JSON manifest")
    p.add_argument("--title", help="list the entries blind-excluded for this title")
    p.add_argument("--digests", action="store_true", help="recompute each entry's tree digest")


corpus_check.arguments = _corpus_args


@command("blind-check", "Verify that no forbidden source tree or donor is reachable from a blind run's "
                        "configuration; write a digest-sealed isolation receipt (read-only)")
def blind_check(args):
    forbidden = [(f"forbid:{Path(p).name}", Path(p)) for p in (args.forbid or [])]
    if args.manifest:
        forbidden += gov.Manifest.load(args.manifest).forbidden_for(args.title)
    if not forbidden:
        raise SystemExit("blind-check needs --forbid PATH or --manifest with blind-excluded entries")
    shared = []
    for spec in getattr(args, "shared_library", None) or []:
        try:
            shared.append(gov.SharedLibrary.load(*gov.parse_pin(spec)))
        except (OSError, ValueError) as exc:          # GovernanceError is a ValueError
            print(f"blind-check: shared library refused: {exc}", file=sys.stderr)
            return 2
    config = {}
    if args.invocation:
        config = json.loads(Path(args.invocation).read_text())
    for key in ("donors", "library"):
        if getattr(args, key, None):
            config[key] = getattr(args, key)
    receipt = gov.check_isolation(forbidden, config=config, project=args.project, content=not args.no_content,
                                  shared=shared)
    if args.out:
        Path(args.out).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    code = 0 if receipt["ok"] else 3
    if getattr(args, "format", "text") == "text":
        exempted = (f", {len(receipt['exemptions'])} exempted by {len(receipt['shared_libraries'])} pinned "
                    f"shared libraries" if "exemptions" in receipt else "")
        print(f"blind isolation: {'ok' if receipt['ok'] else 'VIOLATED'}; {len(receipt['checked'])} locations, "
              f"{receipt['walk']['files_hashed']} files compared, {len(receipt['violations'])} violations"
              f"{exempted}; digest {receipt['digest'][:16]}")
        for row in receipt["violations"][:20]:
            print(f"  {row['key']}: {row['verdict']} ({row['forbidden']})")
        return code
    if code:
        emit(receipt, "json")
        return code
    return receipt


def _blind_args(p):
    p.add_argument("--forbid", action="append", help="a tree the run must not reach (repeatable)")
    p.add_argument("--manifest", help="a corpus manifest: its blind-excluded entries are forbidden")
    p.add_argument("--title", help="only the manifest entries concerning this title")
    p.add_argument("--invocation", help="a run's solver/invocation.json (or any JSON of run options)")
    p.add_argument("--project", help="the project directory (its n64dk.toml is read)")
    p.add_argument("--donors", help="a donors file the run will use")
    p.add_argument("--library", help="a donor library directory the run will use")
    p.add_argument("--no-content", action="store_true", help="skip the byte-for-byte copy check")
    p.add_argument("--shared-library", action="append", metavar="MANIFEST@sha256:HEX",
                   help="a n64dk.shared-library/1 manifest pinned by its SHA-256: copies of the files it "
                        "lists (by digest) are exempt and listed in the receipt (opt-in, repeatable)")
    p.add_argument("--out", help="write the receipt here")


blind_check.arguments = _blind_args


@command("shared-library-pin", "Write a shared-library manifest pinning every file of a public library tree "
                               "by SHA-256, and print the digest to pin it with")
def shared_library_pin(args):
    data = gov.pin_shared_library(args.root, name=args.name, provenance=args.provenance,
                                  revision=args.revision or "", licence=args.licence or "")
    raw = gov.shared_manifest_bytes(data)
    Path(args.out).write_bytes(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    gov.SharedLibrary.load(args.out, digest)          # what was written loads under its own pin
    return {"schema": gov.SHARED_SCHEMA, "name": data["name"], "provenance": data["provenance"],
            "revision": data.get("revision", ""), "files": len(data["files"]), "digest": digest,
            "pin": f"{args.out}@{digest}"}


def _pin_args(p):
    p.add_argument("root", help="the library tree (its own clone, not a copy vendored inside a title)")
    p.add_argument("--name", required=True, help="the library's name")
    p.add_argument("--provenance", required=True, help="the library's public URL")
    p.add_argument("--revision", help="the revision the tree is at (e.g. a commit)")
    p.add_argument("--licence", help="the library's licence")
    p.add_argument("--out", required=True, help="write the manifest here")


shared_library_pin.arguments = _pin_args
