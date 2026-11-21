"""Build pipelines defined in configuration: any compiler, any assembler, any post step.

The built-in pipelines cover KMC GCC 2.7.2, IDO 5.3/7.1 and SN GCC.  Other titles were built
with other toolchains (EGCS 2.91.66 for iQue builds, a modified GCC 2.8.1, other GCC 2.x
releases, IDO C++ front ends), and many need steps after the compiler: ``objcopy -N <file>`` to
drop the file symbol, a script that sets the o32 ABI bit on ``-mips3 -32`` objects, a patch for
64-bit compiles.  Instead of one hard-coded lane per toolchain, a project declares the lane::

    [pipelines.egcs]
    family = "gcc"                       # "gcc" or "ido": decompiler target and diagnosis rules
    default_flags = ["-O2", "-G", "0"]
    timeout = 120
    steps = [
      "{root}/gcc -B{root}/ -c -fno-PIC -mno-abicalls -mcpu=4300 {flags} {includes} -o {object} {source}",
      "{binutils}/mips-linux-gnu-objcopy -N {source_name} {object}",
    ]
    [pipelines.egcs.vars]
    root = "/opt/toolchains/egcs-2.91.66"
    binutils = "/opt/toolchains/binutils-egcs/bin"

and uses it as pipeline ``custom:egcs`` everywhere a pipeline name is accepted.  Placeholders:
``{source}`` (the candidate file, already inside the private build directory), ``{source_name}``,
``{stem}``, ``{object}`` (the object the last step must leave behind), ``{workdir}``,
``{flags}`` (the unit's compiler flags), ``{includes}`` (``-I`` flags, absolute), ``{project}``,
plus every key of ``vars``.  Every step runs with the private build directory as its working
directory, under a timeout, with its output captured; nothing is written to the project.
"""
from __future__ import annotations

import os
import hashlib
import shlex
import signal
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PREFIX = "custom:"



def _tool_digest(path: Path) -> str:
    """A tool file's SHA-256, cached by its size, mtime and inode (see pipeline._file_digest)."""
    from .pipeline import _file_digest
    return _file_digest(Path(path)).hex()

class CustomPipelineError(RuntimeError):
    pass


@dataclass
class Recipe:
    name: str
    steps: list[str]
    family: str = "gcc"
    default_flags: list[str] = field(default_factory=list)
    timeout: float = 120.0
    variables: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    flag_sets: list[list[str]] = field(default_factory=list)    # further usual settings, tried after the default

    def digest_material(self) -> str:
        """Everything that identifies this recipe for an object-cache key (tool files included)."""
        parts = [self.name, self.family, *self.steps, *(f"{k}={v}" for k, v in sorted(self.variables.items())),
                 *(f"{k}={v}" for k, v in sorted(self.env.items()))]
        for value in self.variables.values():
            path = Path(value)
            if path.is_file():
                parts.append(f"{path.resolve()}:{_tool_digest(path)}")
            elif path.is_dir():
                parts.append(f"dir:{path.resolve()}")
        for step in self.steps:
            try:
                token = shlex.split(step)[0]
            except (ValueError, IndexError):
                continue
            path = Path(token)
            if "{" not in token and path.is_absolute() and path.is_file():
                parts.append(f"tool:{path}:{_tool_digest(path)}")
        return "\n".join(parts)


def load(project: str | Path, config: str | Path | None = None) -> dict[str, Recipe]:
    """``[pipelines.*]`` tables of the project's ``n64dk.toml`` (empty when there are none)."""
    path = Path(config) if config else Path(project) / "n64dk.toml"
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text())
    out = {}
    for name, table in (data.get("pipelines") or {}).items():
        steps = table.get("steps")
        if not isinstance(steps, list) or not steps or not all(isinstance(s, str) for s in steps):
            raise CustomPipelineError(f"[pipelines.{name}] needs a non-empty list of step strings")
        family = table.get("family", "gcc")
        if family not in ("gcc", "ido"):
            raise CustomPipelineError(f"[pipelines.{name}] family must be \"gcc\" or \"ido\"")
        out[name] = Recipe(name, list(steps), family, [str(f) for f in table.get("default_flags", [])],
                           float(table.get("timeout", 120)), {k: str(v) for k, v in (table.get("vars") or {}).items()},
                           {k: str(v) for k, v in (table.get("env") or {}).items()},
                           [[str(f) for f in flags] for flags in (table.get("flag_sets") or []) if isinstance(flags, list)])
    return out


def is_custom(pipeline: str) -> bool:
    return pipeline.startswith(PREFIX)


def recipe_for(pipeline: str, project: str | Path, config: str | Path | None = None) -> Recipe:
    name = pipeline[len(PREFIX):]
    recipes = load(project, config)
    if name not in recipes:
        known = ", ".join(sorted(recipes)) or "none defined"
        raise CustomPipelineError(f"no [pipelines.{name}] in the project configuration (known: {known})")
    return recipes[name]


def family_of(pipeline: str, project: str | Path | None = None) -> str:
    """Compiler family of any pipeline name ("ido" or "gcc")."""
    if is_custom(pipeline) and project is not None:
        try:
            return recipe_for(pipeline, project).family
        except CustomPipelineError:
            return "gcc"
    return "ido" if pipeline.startswith("ido") else "gcc"


def build(recipe: Recipe, source: Path, flags: list[str], include_dirs: list[Path], workdir: Path,
          project: Path) -> Path:
    """Run the recipe's steps on ``source`` (which must already be inside ``workdir``) and return
    the object path.  Raises :class:`CustomPipelineError` with the failing step's output."""
    source, workdir = Path(source).resolve(), Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    stem = source.stem if source.suffix else source.name
    obj = workdir / f"{stem}.o"
    values = {"source": str(source), "source_name": source.name, "stem": stem, "object": str(obj),
              "workdir": str(workdir), "project": str(Path(project).resolve()),
              "flags": " ".join(shlex.quote(f) for f in flags),
              "includes": " ".join("-I" + shlex.quote(str(Path(p).resolve())) for p in include_dirs),
              **recipe.variables}
    # TMPDIR is always the private build directory, whatever a lane configures (configured lanes
    # written before round 12 say "/tmp"): the IDO drivers under qemu-irix write their ctm*
    # temporaries there, and about 198,000 were left in a RAM-backed /tmp.  Checked with IDO 6.0.
    environment = {**os.environ, **recipe.env, "TMPDIR": str(workdir)}
    for number, step in enumerate(recipe.steps, 1):
        try:
            argv = shlex.split(step.format(**values))
        except KeyError as error:
            raise CustomPipelineError(f"pipeline {recipe.name} step {number}: unknown placeholder {error}") from None
        if not argv:
            continue
        from .. import safety
        safety.throttle()
        legacy = any(marker in step for marker in ("wibo", "qemu"))
        with safety.tool_tokens(legacy, safety.tool_family(argv)):   # machine-wide, weighted by measured peak
            try:
                family = safety.tool_family(argv)
                launched, rss_report = safety.measured(argv, legacy)
                process = subprocess.Popen(launched, cwd=workdir, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           preexec_fn=safety.child_setup(legacy))
                safety.limit_child(process.pid, legacy)
                safety.register_tool(process.pid)
            except OSError as error:
                raise CustomPipelineError(f"pipeline {recipe.name} step {number}: cannot run {argv[0]}: "
                                          f"{error}") from None
            try:
                output, _ = safety.communicate(process, family, None, recipe.timeout)
            except subprocess.TimeoutExpired:
                raise CustomPipelineError(f"pipeline {recipe.name} step {number} timed out after "
                                          f"{recipe.timeout:g}s: {argv[0]}") from None
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    process.wait(timeout=10)       # the group was killed; streams may be closed
                except Exception:
                    pass
                raise
            finally:
                safety.unregister_tool(process.pid)
                safety.collect_peak(family, rss_report)
        if process.returncode != 0:
            text = output.decode(errors="replace")[-2000:]
            raise CustomPipelineError(f"pipeline {recipe.name} step {number} failed ({argv[0]}, exit "
                                      f"{process.returncode}):\n{text}")
    if not obj.is_file():
        raise CustomPipelineError(f"pipeline {recipe.name} did not produce {obj.name}")
    return obj
