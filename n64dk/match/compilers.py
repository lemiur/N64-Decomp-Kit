"""Catalogue of the compilers known to have built Nintendo 64 retail code.

A title can only be matched with the compiler that built it, and "the compilers that happen to
be installed" is not the same set.  This module is the list the kit measures itself against: for
every compiler the public matching community has needed for the platform it records the build
shape, the directory it is looked for in, whether the kit has a lane for it, and what is known
about where it was used.  ``status`` compares that list with a toolchain root so a missing
compiler is reported before a run instead of being discovered as a title that will not match.

The identifiers are the ones the public decomp.me service uses for its N64 platform, so its
compiler archive can be unpacked straight into a toolchain root.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Compiler:
    ident: str                  # public identifier (decomp.me N64 platform)
    name: str
    family: str                 # "ido", "gcc", "sn", "egcs"
    shape: str                  # how one C file becomes an object
    required: tuple[str, ...]   # files that identify an installed copy
    directories: tuple[str, ...]  # directory names tried inside a toolchain root
    lane: str | None            # built-in pipeline(s), or None: see LANES for configured ones
    used_by: str                # what is publicly documented about its use


CATALOGUE: tuple[Compiler, ...] = (
    Compiler("ido5.3", "SGI IDO 5.3", "ido", "cc driver (static recompilation)", ("cc",), ("ido5.3",), "ido5.3",
             "most first-party and early third-party titles; libultra up to 2.0I"),
    Compiler("ido7.1", "SGI IDO 7.1", "ido", "cc driver (static recompilation)", ("cc",), ("ido7.1",), "ido7.1",
             "later first-party titles; libultra 2.0J onwards built for IDO"),
    Compiler("ido5.3_cfront", "SGI IDO 5.3 C++ (cfront)", "ido", "CC driver under an IRIX user-mode emulator",
             ("usr/lib/CC",), ("ido5.3_cfront",), None, "C++ titles built with IDO 5.3"),
    Compiler("ido4.1", "SGI IDO 4.1", "ido", "cc under an IRIX 4 user-mode emulator, ECOFF converted to ELF",
             ("usr/bin/cc",), ("ido4.1",), None, "very early SDK code"),
    Compiler("ido5.2", "SGI IDO 5.2", "ido", "driver under an IRIX user-mode emulator", ("usr/lib/driver",),
             ("ido5.2",), None, "early titles"),
    Compiler("ido6.0", "SGI IDO 6.0", "ido", "driver under an IRIX user-mode emulator", ("usr/lib/driver",),
             ("ido6.0",), None, "a small number of titles"),
    Compiler("mips_pro_744", "SGI MIPSpro 7.4.4", "ido", "driver under an IRIX user-mode emulator",
             ("usr/lib/driver",), ("mips_pro_744",), None, "rare"),
    Compiler("gcc2.7.2kmc", "KMC GCC 2.7.2", "gcc", "gcc driver with its own GNU as", ("gcc", "cc1", "as"),
             ("kmc-gcc-2.7.2", "gcc2.7.2kmc"), "kmc+as / kmc+asn64", "the Windows SDK's GCC; many third-party titles; "
             "libgultra"),
    Compiler("gcc2.7.2sn0001", "SN Systems GCC 2.7.2 (build 0001)", "sn", "cpp | cc1n64.exe | asn64.exe | psyq-obj-parser",
             ("cc1n64.exe", "asn64.exe"), ("gcc2.7.2sn0001",), None, "early SN Systems kit"),
    Compiler("gcc2.7.2sn0004", "SN Systems GCC 2.7.2 (build 0004)", "sn", "cpp | cc1n64.exe | asn64.exe | psyq-obj-parser",
             ("cc1n64.exe", "asn64.exe"), ("gcc2.7.2sn0004",), None, "SN Systems kit"),
    Compiler("gcc2.7.2sn0006", "SN Systems GCC 2.7.2 (build 0006)", "sn", "cpp | cc1n64.exe | asn64.exe | psyq-obj-parser",
             ("cc1n64.exe", "asn64.exe"), ("gcc2.7.2sn0006",), None, "SN Systems kit"),
    Compiler("gcc2.7.2snew", "SN Systems GCC 2.7.2 (rebuilt, GNU as)", "sn", "cpp | cc1 | modern-asn64 | GNU as",
             ("cc1", "modern-asn64.py"), ("gcc2.7.2snew",), None, "SN titles assembled without asn64.exe"),
    Compiler("gcc2.8.1sn", "SN Systems GCC 2.8.1", "sn", "cpp | cc1n64.exe | asn64.exe | psyq-obj-parser",
             ("cc1n64.exe", "asn64.exe", "psyq-obj-parser"), ("sn-gcc-2.8", "gcc2.8.1sn"), "sn", "later SN Systems kit"),
    Compiler("gcc2.8.1pm", "GCC 2.8.1 (Paper Mario build)", "gcc", "gcc driver with -B", ("gcc", "cc1"),
             ("gcc2.8.1pm",), None, "titles built with a stock 2.8.1 cross compiler"),
    Compiler("egcs_1.1.2-4", "EGCS 1.1.2 (mips-linux)", "egcs", "mips-linux-gcc driver", ("mips-linux-gcc",),
             ("egcs_1.1.2-4",), None, "libultra 2.0L and titles built on Linux hosts"),
    Compiler("egcs_1.1.2-4c", "EGCS 1.1.2 (rebuilt)", "egcs", "gcc driver", ("gcc", "cc1"), ("egcs_1.1.2-4c",), None,
             "as above, a reproducible rebuild"),
    Compiler("gcc4.4.0-mips64-elf", "GCC 4.4.0 mips64-elf", "gcc", "mips64-elf-gcc driver", ("bin/mips64-elf-gcc",),
             ("gcc4.4.0-mips64-elf",), None, "homebrew and late unofficial builds, not retail"),
)


def find(root: Path | None, compiler: Compiler) -> Path | None:
    """The installed directory of ``compiler`` under ``root``, or None."""
    if root is None:
        return None
    for name in compiler.directories:
        path = Path(root) / name
        if path.is_dir() and all((path / required).exists() for required in compiler.required):
            return path
    return None


def status(root: Path | None) -> dict:
    """What a toolchain root covers: per compiler whether it is installed and whether the kit
    can drive it, and the totals a run report should state."""
    import shutil
    host_gcc = shutil.which("gcc") is not None
    rows = []
    for compiler in CATALOGUE:
        path = find(root, compiler)
        lane, why = lane_status(root, compiler)
        if lane and compiler.family == "sn" and not host_gcc:
            # every SN lane preprocesses with the host gcc (``gcc -E``); without it each build fails
            lane, why = None, "the host C preprocessor (gcc) is not installed: install gcc (apt install gcc)"
        rows.append({"id": compiler.ident, "name": compiler.name, "family": compiler.family,
                     "installed": path is not None, "directory": str(path) if path else None,
                     "lane": lane, "no_lane_because": why, "usable": path is not None and lane is not None,
                     "shape": compiler.shape, "used_by": compiler.used_by})
    return {"root": str(root) if root else None, "compilers": rows,
            "known": len(rows), "installed": sum(r["installed"] for r in rows),
            "usable": sum(r["usable"] for r in rows),
            "missing": [r["id"] for r in rows if not r["installed"]],
            "installed_without_lane": {r["id"]: r["no_lane_because"] for r in rows if r["installed"] and not r["lane"]},
            "fetch": "the public decomp.me compiler archive (repository decompme/compilers, platform n64) holds every "
                     "identifier above; unpack each into <toolchain root>/<id>"}


# ---------------------------------------------------------------------------------------------
# Lanes.  The kit has built-in pipelines for four compilers.  Every other catalogued compiler that
# takes ordinary C is driven through a configured pipeline (``n64dk.match.custom_pipeline``): the
# steps below are that configuration, so a project gets a lane for each compiler that is
# installed without anyone writing one.  ``{root}`` is the compiler's directory; ``{wibo}`` runs
# the Windows executables of the SN kits; ``{gnu_as}`` is a modern MIPS GNU assembler.

_SN_PREPROCESS = ("gcc -E -P -undef -D__GNUC__=2 -D__GNUC_MINOR__=7 -Dmips -D__mips__ -D__mips -Dn64 -D__n64__ "
                  "-D__n64 -D_MIPSEB {includes} {source} -o {workdir}/{stem}.i")
_IDO_DRIVER = ("{root}/usr/bin/qemu-irix -silent -L {root} {root}/usr/lib/driver -c -Xcpluscomm -G0 -non_shared "
               "-woff 649,838,712 -32 {flags} {includes} -o {object} {source}")
_SN_FLAGS = ("-O2", "-G0", "-mcpu=vr4300", "-mips3", "-mhard-float", "-meb")
_GCC_FLAGS = ("-O2", "-G0", "-mips3", "-mgp32", "-mfp32")
_IDO_FLAGS = ("-O2", "-mips2")

# identifier -> (lane name, family, default flags, steps, environment, needs)
LANES: dict[str, dict] = {
    # the earliest SN build does not know the VR4300 CPU name
    "gcc2.7.2sn0001": {"lane": "gcc272sn0001", "family": "gcc", "flags": ("-O2", "-G0", "-mips3", "-mhard-float", "-meb"),
                       "needs": ("wibo",), "steps": [
        _SN_PREPROCESS, "{wibo} {root}/cc1n64.exe -quiet {flags} {workdir}/{stem}.i -o {workdir}/{stem}.s",
        "{wibo} {root}/asn64.exe -q {workdir}/{stem}.s -o {workdir}/{stem}.obj",
        "{root}/psyq-obj-parser {workdir}/{stem}.obj -o {object} -b -n -s"]},
    "gcc2.7.2sn0004": {"lane": "gcc272sn0004", "family": "gcc", "flags": _SN_FLAGS, "needs": ("wibo",), "steps": [
        _SN_PREPROCESS, "{wibo} {root}/cc1n64.exe -quiet {flags} {workdir}/{stem}.i -o {workdir}/{stem}.s",
        "{wibo} {root}/asn64.exe -q -G0 {workdir}/{stem}.s -o {workdir}/{stem}.obj",
        "{root}/psyq-obj-parser {workdir}/{stem}.obj -o {object} -b -n"]},
    "gcc2.7.2sn0006": {"lane": "gcc272sn0006", "family": "gcc", "flags": _SN_FLAGS, "needs": ("wibo",), "steps": [
        _SN_PREPROCESS, "{wibo} {root}/cc1n64.exe -quiet {flags} {workdir}/{stem}.i -o {workdir}/{stem}.s",
        "{wibo} {root}/asn64.exe -q -G0 {workdir}/{stem}.s -o {workdir}/{stem}.obj",
        "{root}/psyq-obj-parser {workdir}/{stem}.obj -o {object} -b -n -s"]},
    "gcc2.7.2snew": {"lane": "gcc272snew", "family": "gcc", "flags": ("-O2", "-mips3"), "needs": ("gnu_as",), "steps": [
        "{root}/cpp -lang-c -undef {includes} {source} {workdir}/{stem}.i",
        "{root}/cc1 -mfp32 -mgp32 -G0 -quiet -mcpu=vr4300 -fno-exceptions {flags} {workdir}/{stem}.i -o {workdir}/{stem}.s",
        "python3 {root}/modern-asn64.py {gnu_as} {workdir}/{stem}.s -G0 -EB -mips3 -O1 -mabi=32 -mgp32 "
        "-march=vr4300 -mfp32 -mno-shared -o {object}"]},
    "gcc2.8.1pm": {"lane": "gcc281pm", "family": "gcc", "flags": _GCC_FLAGS, "needs": (), "steps": [
        "{root}/gcc -G0 -c -B {root}/ {flags} {includes} {source} -o {object}"]},
    "egcs_1.1.2-4c": {"lane": "egcs112", "family": "gcc", "flags": ("-O2", "-mips3"), "needs": (),
                      "env": {"COMPILER_PATH": "{root}"}, "steps": [
        "{root}/gcc -c -G 0 -fno-PIC -mgp32 -mfp32 -mcpu=4300 -nostdinc {flags} {includes} {source} -o {object}"]},
    "ido5.2": {"lane": "ido52", "family": "ido", "flags": _IDO_FLAGS, "needs": (), "steps": [
        "{root}/usr/bin/qemu-irix -L {root} {root}/usr/lib/driver -c -Xcpluscomm -G0 -non_shared "
        "-woff 649,838,712 -32 {flags} {includes} -o {object} {source}"]},
    "ido6.0": {"lane": "ido60", "family": "ido", "flags": _IDO_FLAGS, "needs": (), "steps": [_IDO_DRIVER]},
    "mips_pro_744": {"lane": "mipspro744", "family": "ido", "flags": _IDO_FLAGS, "needs": (), "steps": [_IDO_DRIVER]},
}
BUILT_IN = {"ido5.3": "ido5.3", "ido7.1": "ido7.1", "gcc2.7.2kmc": "kmc+as", "gcc2.8.1sn": "sn"}
NO_LANE = {"ido5.3_cfront": "a C++ front end: the kit's drafts are C",
           "ido4.1": "ECOFF objects: needs a conversion step the kit does not drive yet",
           "egcs_1.1.2-4": "the original binaries need a 32-bit loader; the rebuilt egcs_1.1.2-4c is the same "
                           "compiler and is the one driven",
           "gcc4.4.0-mips64-elf": "not a retail compiler: offered only on request"}


def _helper(root: Path | None, name: str) -> str | None:
    """Where a lane's helper tool is: ``wibo`` beside an SN kit or on PATH, a MIPS GNU assembler."""
    import shutil
    if name == "wibo":
        for folder in ("sn-gcc-2.8", "gcc2.8.1sn", "wibo", "."):
            candidate = Path(root) / folder / "wibo" if root else None
            if candidate is not None and candidate.is_file():
                return str(candidate)
        return shutil.which("wibo")
    if name == "gnu_as":
        for prefix in ("mips-binutils/usr/bin", "mips-binutils/bin"):
            candidate = Path(root) / prefix / "mips-linux-gnu-as" if root else None
            if candidate is not None and candidate.is_file():
                return str(candidate)
        return next((found for found in map(shutil.which, ("mips-linux-gnu-as", "mips64-elf-as", "mips-elf-as"))
                     if found), None)
    return None



# Settings real projects build with, beyond each lane's default.  Taken from what is published
# about finished decompilations (their compiler presets and build files): an optimisation level
# or instruction set a title was really built with must be among what calibration tries.
FLAG_SETS: dict[str, tuple[tuple[str, ...], ...]] = {
    "gcc281pm": (("-O2", "-fforce-addr", "-G0", "-mips3", "-mgp32", "-mfp32"),
                 ("-O1", "-G0", "-mips3", "-mgp32", "-mfp32"), ("-O2", "-G0", "-mips2")),
    "gcc272snew": (("-O2", "-mips2"), ("-O1", "-mips3")),
    "egcs112": (("-O2", "-mips2", "-mcpu=4300", "-mno-abicalls"), ("-O2", "-g", "-mips2", "-mcpu=4300"),
                ("-O1", "-mips3")),
    "gcc272sn0001": (("-O1", "-G0", "-mips3", "-mhard-float", "-meb"),),
    "gcc272sn0004": (("-O1", "-G0", "-mcpu=vr4300", "-mips3", "-mhard-float", "-meb"),),
    "gcc272sn0006": (("-O1", "-G0", "-mcpu=vr4300", "-mips3", "-mhard-float", "-meb"),),
    "ido52": (("-O1", "-mips2"), ("-O2", "-mips1"), ("-O2", "-g3", "-mips2")),
    "ido60": (("-O1", "-mips2"), ("-O2", "-mips1"), ("-O2", "-g3", "-mips2")),
    "mipspro744": (("-O1", "-mips2"), ("-O2", "-mips1"), ("-O2", "-g3", "-mips2")),
}

def project_pipelines(root: Path | None) -> dict[str, dict]:
    """``[pipelines.<lane>]`` tables for every installed compiler the kit has no built-in pipeline
    for: lane name -> ``{"family", "default_flags", "steps", "vars", "env", "compiler"}``.  A
    compiler whose helper tool is missing is left out (``lane_status`` says why)."""
    out = {}
    for compiler in CATALOGUE:
        lane = LANES.get(compiler.ident)
        path = find(root, compiler)
        if lane is None or path is None:
            continue
        variables = {"root": str(path)}
        helpers = {need: _helper(root, need) for need in lane["needs"]}
        if any(value is None for value in helpers.values()):
            continue
        variables.update(helpers)
        env = {key: value.format(**variables) for key, value in (lane.get("env") or {}).items()}
        if "gnu_as" in helpers:                               # a bundled assembler brings its own libraries
            library = Path(helpers["gnu_as"]).parent.parent / "lib" / "x86_64-linux-gnu"
            if library.is_dir():
                env["LD_LIBRARY_PATH"] = str(library)
        if any("usr/lib" in step or "usr/bin/qemu" in step for step in lane["steps"]):
            env.setdefault("TMPDIR", "/tmp")
        out[lane["lane"]] = {"family": lane["family"], "default_flags": list(lane["flags"]), "steps": list(lane["steps"]),
                             "flag_sets": [list(flags) for flags in FLAG_SETS.get(lane["lane"], ())],
                             "vars": variables, "env": env, "compiler": compiler.ident}
    return out


def lane_status(root: Path | None, compiler: Compiler) -> tuple[str | None, str | None]:
    """(lane, why not): the pipeline name the kit drives ``compiler`` with, or the reason there is none."""
    if compiler.ident in BUILT_IN:
        return BUILT_IN[compiler.ident], None
    lane = LANES.get(compiler.ident)
    if lane is None:
        return None, NO_LANE.get(compiler.ident, "no lane defined")
    missing = [need for need in lane["needs"] if _helper(root, need) is None]
    if missing:
        return None, "needs " + " and ".join("wibo (runs the SN kit's Windows executables)" if need == "wibo"
                                             else "a MIPS GNU assembler (mips-linux-gnu-as)" for need in missing)
    return "custom:" + lane["lane"], None


def toml_tables(pipelines: dict[str, dict]) -> str:
    """``project_pipelines`` as the text ``decompile-init`` appends to ``n64dk.toml``."""
    import json
    lines = []
    for name, table in pipelines.items():
        lines += [f"[pipelines.{name}]                 # {table['compiler']}", f'family = "{table["family"]}"',
                  "default_flags = " + json.dumps(table["default_flags"]),
                  *(["flag_sets = " + json.dumps(table["flag_sets"])] if table.get("flag_sets") else []), "steps = ["]
        lines += [f"  {json.dumps(step)}," for step in table["steps"]] + ["]"]
        lines += [f"[pipelines.{name}.vars]"] + [f"{key} = {json.dumps(value)}" for key, value in table["vars"].items()]
        if table["env"]:
            lines += [f"[pipelines.{name}.env]"] + [f"{key} = {json.dumps(value)}" for key, value in table["env"].items()]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Fetching.  The public decomp.me project publishes each compiler as an OCI image whose layers
# are tar archives holding ``compilers/<platform>/<id>/``.  Fetching one is: an anonymous pull
# token, the image index, the image manifest, then each layer, checked against the SHA-256 the
# manifest states before anything is unpacked.  Nothing is executed and nothing is installed
# outside the directory given.

REGISTRY = "ghcr.io"
REPOSITORY = "decompme/compilers"
PLATFORM = "n64"
MAX_LAYER_BYTES = 600 * 1024 * 1024


class FetchError(RuntimeError):
    pass


def _get(url: str, headers: dict | None = None, timeout: float = 60):
    import urllib.request
    request = urllib.request.Request(url, headers={"User-Agent": "n64dk", **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)       # noqa: S310 - fixed https registry


def _image(ident: str, host: str | None = None) -> str:
    name = f"{PLATFORM}/{ident.lower().replace('+', 'plus')}"
    return f"{name}/{host}" if host else name


def _safe_members(archive, root: Path):
    """Members that stay inside ``root``: no absolute names, no ``..``, no links leading out."""
    base = root.resolve()
    for member in archive.getmembers():
        target = (base / member.name).resolve()
        if base != target and base not in target.parents:
            raise FetchError(f"archive member escapes the destination: {member.name!r}")
        if member.issym() or member.islnk():
            link = (target.parent / member.linkname).resolve() if member.issym() else (base / member.linkname).resolve()
            if base != link and base not in link.parents:
                raise FetchError(f"archive link escapes the destination: {member.name!r}")
        if member.isdev():
            continue
        yield member


def fetch(ident: str, root: Path, force: bool = False, opener=_get, pin: str | None = None) -> dict:
    """Download one compiler of the catalogue into ``root/<directory>``.  Returns what happened;
    raises ``FetchError`` on a failure that leaves nothing installed.

    ``pin`` (``sha256:...``) is the image manifest digest a reviewer accepted: when given, any other
    manifest is refused before a layer is downloaded, so "latest" cannot drift silently.  The
    result always carries ``manifest_digest``, the value to pin.  Nothing fetched is executed."""
    import hashlib
    import json
    import shutil
    import tarfile
    import tempfile
    compiler = next((c for c in CATALOGUE if c.ident == ident), None)
    if compiler is None:
        raise FetchError(f"{ident!r} is not in the catalogue")
    root = Path(root)
    if find(root, compiler) is not None and not force:
        return {"id": ident, "status": "present", "directory": str(find(root, compiler))}
    destination = root / ident
    last = None
    for host in (None, "linux"):                              # some images are published per host system
        image = _image(ident, host)
        try:
            with opener(f"https://{REGISTRY}/token?scope=repository:{REPOSITORY}/{image}:pull") as reply:
                token = json.load(reply).get("token", "")
            auth = {"Authorization": f"Bearer {token}"}
            base = f"https://{REGISTRY}/v2/{REPOSITORY}/{image}"
            with opener(f"{base}/manifests/latest", {**auth, "Accept": "application/vnd.oci.image.index.v1+json"}) as reply:
                index = json.load(reply)
            entry = next((m for m in index.get("manifests", [])
                          if m.get("mediaType") == "application/vnd.oci.image.manifest.v1+json"), None)
            if entry is None:
                raise FetchError("the image index lists no image manifest")
            if pin and entry.get("digest") != pin:
                raise FetchError(f"{ident}: the registry's manifest {entry.get('digest')} is not the pinned {pin}")
            with opener(f"{base}/manifests/{entry['digest']}", {**auth, "Accept": entry["mediaType"]}) as reply:
                manifest = json.load(reply)
        except FetchError:
            raise
        except Exception as error:                            # not published under this name: try the next
            last = error
            continue
        with tempfile.TemporaryDirectory(prefix="n64dk-compiler-") as tmp:
            unpacked = Path(tmp) / "unpacked"
            unpacked.mkdir()
            size = 0
            for number, layer in enumerate(manifest.get("layers", [])):
                wanted = str(layer.get("digest", ""))
                if not wanted.startswith("sha256:"):
                    raise FetchError(f"layer {number} has no SHA-256 digest")
                path, digest = Path(tmp) / f"layer{number}.tar", hashlib.sha256()
                with opener(f"{base}/blobs/{wanted}", auth, timeout=600) as reply, path.open("wb") as handle:
                    while True:
                        chunk = reply.read(1 << 20)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_LAYER_BYTES:
                            raise FetchError("the image is larger than the size limit")
                        digest.update(chunk)
                        handle.write(chunk)
                if "sha256:" + digest.hexdigest() != wanted:
                    raise FetchError(f"layer {number} does not have the digest its manifest states")
                with tarfile.open(path) as archive:
                    archive.extractall(unpacked, members=list(_safe_members(archive, unpacked)))
            inner = unpacked / "compilers" / PLATFORM / ident
            if not inner.is_dir():
                raise FetchError("the image does not hold compilers/%s/%s" % (PLATFORM, ident))
            root.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(inner), str(destination))
        missing = [name for name in compiler.required if not (destination / name).exists()]
        return {"id": ident, "status": "fetched", "directory": str(destination), "bytes": size, "image": image,
                "manifest_digest": entry.get("digest"), "missing_expected_files": missing}
    raise FetchError(f"{ident}: not found in the registry ({last})")


# ---------------------------------------------------------------------------------------------
# Sibling versions.  A title is not built by one compiler version: its libraries were built when
# and where their vendors built them (Sesame Street: the speech library with IDO 7.1, the game
# with IDO 5.3).  Most small functions come out the same under both versions and say nothing
# about which one built them; the few that differ are near misses under the wrong one (same
# size, a handful of instructions).  So a near miss is tried under the sibling versions of its
# compiler before its source is permuted, and only functions that match under one version and
# not the other count as evidence for a version.

SIBLINGS: tuple[frozenset[str], ...] = (frozenset({"ido5.3", "ido7.1"}), frozenset({"kmc+as", "kmc+asn64"}))


def siblings(pipeline: str) -> tuple[str, ...]:
    """The other versions/lanes of the same compiler (ido5.3 <-> ido7.1, kmc+as <-> kmc+asn64)."""
    for group in SIBLINGS:
        if pipeline in group:
            return tuple(sorted(group - {pipeline}))
    return ()


def _level(flags) -> str | None:
    return next((f for f in flags if f.startswith("-O") or f == "-g"), None)


def sibling_recipe(current: str, label: str, levels=()) -> bool:
    """Whether recipe ``label`` ("pipeline flags") is a sibling of ``current``: a sibling version
    with the same flags, or the same/sibling compiler at an optimisation level of ``levels`` (the
    levels the region already uses) with the flags otherwise unchanged."""
    pipeline, _, flags = current.partition(" ")
    other, _, other_flags = label.partition(" ")
    if label == current or (other != pipeline and other not in siblings(pipeline)):
        return False
    a, b = flags.split(), other_flags.split()
    if a == b:
        return other != pipeline
    rest_a = [f for f in a if f != _level(a)]
    rest_b = [f for f in b if f != _level(b)]
    return rest_a == rest_b and _level(b) in set(levels)


def version_evidence(cases) -> dict:
    """Which version a set of matched functions supports.  ``cases``: ``(recipe label, labels the
    same source is exact under, or None when that was not checked)``.  A function exact under
    several sibling versions supports none of them (``insensitive``); a function checked and exact
    under one version only is ``discriminating``; an unchecked one still votes (``unchecked``).
    Returns ``{"votes": {label: n}, "insensitive": n, "discriminating": n, "unchecked": n}``."""
    votes: dict[str, int] = {}
    insensitive = discriminating = unchecked = 0
    for label, exact_under in cases:
        if exact_under is None:
            unchecked += 1
            votes[label] = votes.get(label, 0) + 1
            continue
        under = set(exact_under or ()) | {label}
        pipeline = label.partition(" ")[0]
        flags = label.partition(" ")[2]
        versions = {u.partition(" ")[0] for u in under if u.partition(" ")[2] == flags}
        if versions & set(siblings(pipeline)):
            insensitive += 1
            continue
        discriminating += 1
        votes[label] = votes.get(label, 0) + 1
    return {"votes": votes, "insensitive": insensitive, "discriminating": discriminating, "unchecked": unchecked}
