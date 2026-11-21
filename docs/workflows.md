# Workflows, as executable examples

Every shell block on this page is a contract. The comment above it (`<!-- example: ... -->`, invisible
when rendered) names the starting state the block assumes (`fixture`), what a real run needs beyond
it (`requires`) and the exit status of its last command (`expect`). `tools/doc_examples.py check`
parses every `n64dk` command here with the kit's real argument parser, so a renamed command or option
fails the test suite (`tests/test_doc_examples.py`). Blocks that need nothing but a scratch directory
are also *run* by the tests, against synthetic fixtures. The complete option list of every command is
generated from the same parsers: [cli-reference.md](cli-reference.md).

The blocks use shell variables; set the ones a block mentions:

| variable | meaning |
|---|---|
| `WORK` | a scratch directory outside the kit, the ROM folder and any source tree |
| `ROM` | a cartridge image (z64, v64 or n64 byte order); read, never written |
| `PROJECT` | a project directory with `n64dk.toml` (from `decompile-init`, or your own) |
| `RUN` | a `decompile-run --out` directory |
| `EXPORT` | a `decompile-export --out` directory |
| `SOURCE` | a C source tree of the title (for example a public decompilation) |
| `TOOLCHAIN` | the toolchain root holding the original compilers (`n64dk compilers`) |
| `RECEIPT` | one proof receipt file (`issue-receipts` writes them) |

## Troubleshooting and doctor

Start here on a new machine. The host doctor needs no project, changes nothing and exits 1 only when
a *required* check fails (dependency, host, disk, total memory, an unsafe `--out`); optional
capabilities and missing compilers are reported, not failed.

<!-- example: fixture=none expect=any id=doctor-host -->
```sh
n64dk --format json doctor --host --out "$WORK/out"
```

Which original compilers a toolchain root holds, and which of them the kit can drive:

<!-- example: fixture=empty-toolchain id=compilers-root -->
```sh
n64dk --format json compilers --toolchain-dir "$TOOLCHAIN"
```

Heavy jobs registered on this machine (a job killed with SIGKILL leaves a `stale` record;
`--prune-stale` removes it), and generated files left inside a project:

<!-- example: fixture=none id=jobs -->
```sh
n64dk --format json jobs
```

<!-- example: fixture=synthetic-project id=workspace-audit -->
```sh
n64dk --format json workspace-audit --project "$PROJECT"
```

The project doctor checks one matching project's toolchains, interpreter and build smokes (it
launches builds, so it is slow):

<!-- example: fixture=project requires=project,slow id=doctor-project -->
```sh
n64dk doctor --project . --profile matching --skip-functional
```

## Bare-ROM run

A title nobody has worked on: `decompile-init` surveys the ROM and writes a project,
`decompile-run` works through every image. Run output belongs under the project's `.n64dk/` or
outside the project.

<!-- example: fixture=rom requires=rom,compiler id=bare-rom-run -->
```sh
n64dk info "$ROM"
n64dk decompile-init "$ROM" --out "$WORK/title"
n64dk decompile-run --project "$WORK/title" --all-images --out "$WORK/title/.n64dk/run" --workers 1
```

## Exhaustive run

`--effort exhaustive` turns every opt-in technique on and the build-saving gates off; bound it by time
or builds. `decompile-everything` does init, an exhaustive run and the export for a list of ROMs.

<!-- example: fixture=rom-project requires=rom,compiler,slow id=exhaustive-run -->
```sh
n64dk decompile-run --project "$PROJECT" --all-images --out "$WORK/run-exhaustive" \
    --effort exhaustive --budget-time 2h --workers 1
n64dk decompile-everything "$ROM" --out "$WORK/everything"
```

## Source-assisted run

A C source tree of the title (or of a library it uses) becomes a donor: `source-donor` compiles its
units, finds them in the target and writes a donors file the run uses.

<!-- example: fixture=source-tree requires=rom,source,compiler,slow id=source-assisted-run -->
```sh
n64dk source-donor --source "$SOURCE" --project "$PROJECT" --out "$WORK/donor" --units split
n64dk decompile-run --project "$PROJECT" --all-images --out "$WORK/run-donor" \
    --donors "$WORK/donor/donors.toml" --workers 1
```

## Custom compiler

`n64dk compilers` lists the catalogued compilers; a compiler without a built-in pipeline gets a
configured lane in `n64dk.toml` (`[pipelines.<name>]` with `family`, `default_flags`, `steps`; see
[toolchains.md](toolchains.md)). A run can be pointed at one pipeline and flag set:

<!-- example: fixture=rom-project requires=rom,compiler id=custom-compiler-run -->
```sh
n64dk compilers --toolchain-dir "$TOOLCHAIN"
n64dk decompile-run --project "$PROJECT" --all-images --out "$WORK/run-ido" \
    --pipeline ido5.3 --flags "-O2 -mips2" --toolchain-dir "$TOOLCHAIN" --workers 1
```

## Small-data (-G) regions

When the program sets `$gp`, `--small-data auto` (the default) recovers `_gp` and offers the
small-data assembler lanes to functions that address through `$gp`; `toolchain-regions` then reports
which recipe each region of the image was built with.

<!-- example: fixture=rom-project requires=rom,compiler id=small-data-run -->
```sh
n64dk decompile-run --project "$PROJECT" --all-images --out "$RUN" --small-data auto --workers 1
n64dk toolchain-regions --project "$PROJECT" --run "$RUN" --out "$WORK/regions.json"
```

## Overlay project

Overlays that share a VRAM range are separate images in `n64dk.toml`, each with its own catalogue;
the kit never picks one for an address that several map:

```toml
[images.ovl_a]
path = "ovl_a.bin"
vram_delta = 0x80100000
catalogue = "ovl_a.toml"

[images.ovl_b]
path = "ovl_b.bin"
vram_delta = 0x80100000
catalogue = "ovl_b.toml"
```

An address both overlays map is refused (exit 3) until an image is named:

<!-- example: fixture=overlay-project expect=3 id=overlay-ambiguous -->
```sh
n64dk image-address --project "$PROJECT" --vram 0x80100010
```

<!-- example: fixture=overlay-project id=overlay-named -->
```sh
n64dk image-address --project "$PROJECT" --image ovl_b --vram 0x80100010
n64dk image-registry --project "$PROJECT"
```

<!-- example: fixture=overlay-project requires=compiler id=overlay-run -->
```sh
n64dk decompile-run --project "$PROJECT" --image ovl_a --image ovl_b --out "$WORK/run-overlays" --workers 1
```

## Resume

A run killed or stopped continues from its case store and `rounds.jsonl`; settled work is not
redone. `decompile-status` is read-only.

<!-- example: fixture=run requires=run,compiler id=resume -->
```sh
n64dk decompile-run --resume "$RUN" --budget-time 30m
n64dk decompile-status "$RUN"
```

## Model refinement (mock)

An external refiner reads a prompt on stdin and prints one fenced C block; whatever it answers is
built, compared and strictly verified by the kit, so a wrong answer costs builds, never credit.
[`examples/refine-mock.sh`](examples/refine-mock.sh) is a mock that always gives the same answer:
it exercises the plumbing without a model.

<!-- example: fixture=rom-project requires=rom,compiler id=refine-mock -->
```sh
n64dk decompile-run --project "$PROJECT" --all-images --out "$WORK/run-refine" --effort quick \
    --limit 20 --refine-command "sh docs/examples/refine-mock.sh" --budget-model-calls 20 --workers 1
```

## Export

`decompile-export` writes the run as a tree to commit: matched C, C awaiting symbols, verified
assembly and generated assembly kept apart, symbol files and the function table. Nothing of the ROM
is copied.

<!-- example: fixture=run requires=run id=export -->
```sh
n64dk decompile-export --project "$PROJECT" --run "$RUN" --out "$WORK/export"
```

## Receipt verification

A proof receipt records how one result was rebuilt and verified (digests of source, recipe, tools
and target). `issue-receipts` writes them; `verify-receipt` replays one, `verify-run` and
`verify-export` replay all; `--static` checks digests without rebuilding. `decompile-receipt
--require` exits 4 when the run's treatment contract is not satisfied.

<!-- example: fixture=export requires=run,export,compiler id=receipts -->
```sh
n64dk issue-receipts "$RUN" --export "$EXPORT"
n64dk verify-run "$RUN" --static
n64dk verify-receipt "$RECEIPT" --run "$RUN"
n64dk verify-export "$EXPORT" --project "$PROJECT"
n64dk decompile-receipt "$RUN" --out "$WORK/receipt" --require
```

## Project rebuild

There is no single command that relinks a whole image. Rebuilding and re-verifying every matched
function of an export is `verify-export` (above, without `--static`); a hand-maintained project's
manifests are rebuilt and classified by `manifest-verify`:

<!-- example: fixture=project requires=project,compiler id=project-rebuild -->
```sh
n64dk manifest-verify --project . config/matching_units.tsv
```

## Dashboard

There is no graphical dashboard. The read-only view of a run is `decompile-status` (rounds table,
remaining queue by primary cause, examples per cause); `--format json` gives the same for scripts.

<!-- example: fixture=run requires=run id=dashboard -->
```sh
n64dk decompile-status "$RUN" --show 10
n64dk --format json decompile-status "$RUN"
```

## The command line itself

<!-- example: fixture=none id=operation-schema -->
```sh
n64dk --format json operation-schema
```
