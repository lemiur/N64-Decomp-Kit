"""Plugin and capability boundary (round 12, item P).

A plugin extends one capability of the kit with an external program.  It is described by a
manifest (``plugin.json``) and runs **out of process**, through the kit's safety runner
(``safety.run_tool``: a machine-wide token, its own session, a memory ceiling, the whole process
group killed on timeout).  It exchanges JSON on stdin/stdout: the kit sends ``{"capability",
"request"}``, the plugin answers ``{"result": ...}``.  A plugin never runs inside the kit's process
and never sees anything but its request.

Capabilities (``CAPABILITIES``): codecs, image/overlay discoverers, compiler recipes, assemblers,
linkers, decompilers, semantic-IR adapters, behaviour backends, asset classifiers, debug-record
decoders, refinement providers, exporters.

Safety rules, enforced here, not requested of plugins:

- **Nothing is discovered implicitly.**  A manifest is loaded only from a directory named in an
  explicit allowlist (``N64DK_PLUGIN_DIRS`` or the ``allow`` argument).  A ``plugin.json`` inside a
  project, a ROM folder or a downloaded tree is never loaded because it exists.
- **The manifest must declare** identity, version, capability, supported inputs, determinism,
  resource weight (light/compile/legacy), dependencies, provenance, output schema (required keys),
  cache identity and a safety class; an incomplete manifest is refused with the missing fields.
- **The executable is pinned**: the manifest names it relative to the plugin directory with its
  SHA-256; a different file is refused (no executing whatever is there now).
- **Failure is isolated and recorded**: a timeout, a crash, non-JSON output or output missing the
  declared keys becomes a ``PluginFailure`` record; the caller's run goes on.
- **Plugin output is evidence, never proof**: every result is marked ``proof: false``; strict
  verification of anything a plugin proposes is the kit's own.

Built-in functionality needs no plugin; ``builtin_capabilities()`` lists what the kit does itself
for each capability so a plugin is an addition, not a dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = 1
CAPABILITIES = ("codec", "image-discoverer", "compiler-recipe", "assembler", "linker", "decompiler",
                "ir-adapter", "behaviour-backend", "asset-classifier", "debug-decoder", "refinement-provider",
                "exporter")
WEIGHTS = ("light", "compile", "legacy")
SAFETY = ("pure-function", "reads-inputs-only", "network-free-tool", "untrusted")
REQUIRED = ("id", "version", "capability", "inputs", "deterministic", "weight", "dependencies", "provenance",
            "output_schema", "cache_identity", "safety", "executable", "sha256")


class PluginError(ValueError):
    """A manifest or plugin that the kit refuses to load or run."""


@dataclass(frozen=True)
class Plugin:
    directory: Path
    manifest: dict

    @property
    def ident(self) -> str:
        return f"{self.manifest['id']}@{self.manifest['version']}"

    @property
    def executable(self) -> Path:
        return self.directory / self.manifest["executable"]

    def cache_key(self, request: dict) -> str:
        material = {"plugin": self.ident, "sha256": self.manifest["sha256"],
                    "cache_identity": self.manifest["cache_identity"], "request": request}
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


@dataclass
class PluginFailure:
    plugin: str
    capability: str
    reason: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"plugin": self.plugin, "capability": self.capability, "failure": self.reason,
                "detail": self.detail[-300:]}


@dataclass
class Registry:
    plugins: list[Plugin] = field(default_factory=list)
    refused: list[dict] = field(default_factory=list)

    def for_capability(self, capability: str) -> list[Plugin]:
        return [p for p in self.plugins if p.manifest["capability"] == capability]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate(manifest: dict, directory: Path) -> list[str]:
    """Every reason a manifest is refused (empty: accepted)."""
    problems = [f"missing field {name!r}" for name in REQUIRED if name not in manifest]
    if problems:
        return problems
    if manifest.get("schema", SCHEMA) != SCHEMA:
        problems.append(f"manifest schema {manifest.get('schema')} is not {SCHEMA}")
    if manifest["capability"] not in CAPABILITIES:
        problems.append(f"unknown capability {manifest['capability']!r}")
    if manifest["weight"] not in WEIGHTS:
        problems.append(f"weight must be one of {WEIGHTS}")
    if manifest["safety"] not in SAFETY:
        problems.append(f"safety class must be one of {SAFETY}")
    if not isinstance(manifest["output_schema"], dict) or not isinstance(manifest["output_schema"].get("required", []), list):
        problems.append("output_schema must be an object with a 'required' list")
    executable = (directory / str(manifest["executable"])).resolve()
    if directory.resolve() not in executable.parents:
        problems.append("executable must lie inside the plugin directory")
    elif not executable.is_file():
        problems.append("executable is missing")
    elif _sha256(executable) != manifest["sha256"]:
        problems.append("executable does not have the pinned SHA-256")
    return problems


def allowed_dirs(allow: list[str | Path] | None = None) -> list[Path]:
    """The allowlisted plugin directories: ``allow`` when given, else ``N64DK_PLUGIN_DIRS``
    (``os.pathsep``-separated).  Nothing else is ever searched."""
    if allow is None:
        raw = os.environ.get("N64DK_PLUGIN_DIRS", "")
        allow = [p for p in raw.split(os.pathsep) if p]
    return [Path(p) for p in allow]


def load(allow: list[str | Path] | None = None) -> Registry:
    """Manifests from the allowlisted directories only (each a plugin directory holding
    ``plugin.json``, or a parent of such directories, one level deep)."""
    registry = Registry()
    for root in allowed_dirs(allow):
        candidates = [root] if (root / "plugin.json").is_file() else sorted(p for p in root.glob("*") if (p / "plugin.json").is_file())
        for directory in candidates:
            try:
                manifest = json.loads((directory / "plugin.json").read_text())
            except (OSError, ValueError) as error:
                registry.refused.append({"directory": directory.name, "reasons": [f"unreadable manifest: {error}"]})
                continue
            problems = validate(manifest, directory) if isinstance(manifest, dict) else ["manifest is not an object"]
            if problems:
                registry.refused.append({"directory": directory.name, "reasons": problems})
            else:
                registry.plugins.append(Plugin(directory, manifest))
    return registry


def call(plugin: Plugin, request: dict, timeout: float = 120, runner=None) -> dict | PluginFailure:
    """Run one plugin request out of process.  Returns the plugin's result (marked ``proof:
    false``) or a ``PluginFailure``; never raises for a plugin's own failure."""
    if validate(plugin.manifest, plugin.directory):            # the file changed since loading
        return PluginFailure(plugin.ident, plugin.manifest["capability"], "manifest-invalid",
                             "; ".join(validate(plugin.manifest, plugin.directory)))
    payload = json.dumps({"capability": plugin.manifest["capability"], "request": request})
    if runner is None:
        from . import safety
        runner = safety.run_tool
    import subprocess
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C"}      # nothing else is forwarded
    try:
        done = runner([str(plugin.executable)], kind=plugin.manifest["weight"], timeout=timeout,
                      input=payload, cwd=str(plugin.directory), env=env, text=True)
    except subprocess.TimeoutExpired:
        return PluginFailure(plugin.ident, plugin.manifest["capability"], "timeout", f"after {timeout} s")
    except OSError as error:
        return PluginFailure(plugin.ident, plugin.manifest["capability"], "could-not-start", str(error))
    if done.returncode != 0:
        return PluginFailure(plugin.ident, plugin.manifest["capability"], "exit-status",
                             f"{done.returncode}: {(done.stderr or '')[-200:]}")
    try:
        answer = json.loads(done.stdout)
    except (TypeError, ValueError):
        return PluginFailure(plugin.ident, plugin.manifest["capability"], "output-not-json", str(done.stdout)[:200])
    result = answer.get("result") if isinstance(answer, dict) else None
    missing = [k for k in plugin.manifest["output_schema"].get("required", [])
               if not isinstance(result, dict) or k not in result]
    if missing:
        return PluginFailure(plugin.ident, plugin.manifest["capability"], "output-schema",
                             "missing " + ", ".join(missing))
    return {"plugin": plugin.ident, "cache_key": plugin.cache_key(request), "proof": False, "result": result}


def builtin_capabilities() -> dict[str, list[str]]:
    """What the kit does itself for each capability (no plugin needed)."""
    return {
        "codec": ["n64dk.assets.compress (known codecs)", "n64dk.analysis.loaders_generic.capture (game decoder)"],
        "image-discoverer": ["n64dk.analysis.segments", "n64dk.analysis.overlays"],
        "compiler-recipe": ["n64dk.match.components.RECIPES", "[pipelines.*] project lanes"],
        "assembler": ["kmc/as, gnu/as-2.9, binutils/as, sn/asn64 via components"],
        "linker": ["n64dk.analysis.linklayout (link.ld generation)"],
        "decompiler": ["m2c (n64dk.match.m2c_bridge)"],
        "ir-adapter": ["n64dk.match.ir"],
        "behaviour-backend": ["n64dk.match.oracle (bare R4300 emulator)"],
        "asset-classifier": ["n64dk.assets (gbi, texture, audio)"],
        "debug-decoder": ["n64dk.analysis.devrecords (SN/PsyQ)", "n64dk.analysis.debugscan (ECOFF, STABS, strings)"],
        "refinement-provider": ["n64dk.match.refine (command provider, mock)"],
        "exporter": ["n64dk.commands.export", "n64dk.interop (splat, symbol_addrs, objdiff)"],
    }
