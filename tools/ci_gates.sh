#!/bin/sh
# tools/ci_gates.sh: the kit's release gates, one step at a time, each under a timeout, at low
# CPU and I/O priority and (where possible) under a hard memory cap with swap forbidden.  Every
# step runs even when an earlier one failed (unless CI_FAIL_FAST=1); the summary lists them all
# and the script exits 1 when any step failed (2 on a usage error), 0 only when all passed.
#
#   sh tools/ci_gates.sh            # from anywhere; it changes to the repository root
#
# Steps, in order (cheap first; the workspace audit last, to catch litter the suite leaves):
#   tracked-files    no SQLite sidecar (*.sqlite-wal, *.sqlite-shm, *-journal) and no
#                    machine-specific path (/home/<user>/, /mnt/<drive>/Users/<user>/) in git
#   compileall       python -m compileall -q n64dk tests
#   python3.11       every .py file compiled by a real 3.11 interpreter (python3.11 on PATH or
#                    CI_PY311); otherwise CI_PY311_CHECK (a static checker script) if set; else skipped
#   operation-schema python -m n64dk --format json operation-schema, parsed and checked
#   unit-suite       python -m unittest discover -s tests -t .
#   self-test        python -m n64dk --format json self-test ("ok" must be true: the command
#                    itself exits 0 either way)
#   workspace-audit  python -m n64dk --format json workspace-audit --project . (no generated file)
#
# Settings (environment):
#   PYTHON                interpreter (default: python3, else python)
#   CI_MEMORY_MAX         cap per step, e.g. 3G, 4096M; "off" for none (default 4G).  Applied with
#                         systemd-run --user --scope -p MemoryMax -p MemorySwapMax=0 when that works,
#                         else as an address-space limit (ulimit -v) of CI_VMEM_FACTOR x the cap
#   CI_VMEM_FACTOR        default 2 (address space is larger than resident memory)
#   CI_MIN_AVAILABLE_MB   a step refuses to start (and fails) below this much available memory
#                         (/proc/meminfo MemAvailable); 0 to disable (default 2048)
#   CI_TIMEOUT_SHORT      seconds for the quick steps (default 600)
#   CI_TIMEOUT_SUITE      seconds for unit-suite and self-test (default 2400; self-test has its
#                         own 1800 s limit inside)
#   CI_NICE               nice level (default 15)
#   CI_LOCK               optional lock file taken (flock) for the whole run, so gates never run
#                         beside another heavy job using the same lock.  Do not set it when this
#                         script is itself started by a wrapper that already holds that lock.
#   CI_LOCK_WAIT_S        longest wait for CI_LOCK (default 86400)
#   CI_REQUIRE_PY311=1    a skipped python3.11 step counts as a failure
#   CI_SKIP               space-separated step names to skip (reported as skipped)
#   CI_FAIL_FAST=1        stop after the first failed step
#   CI_LOG_DIR            where step logs go (default: a new temporary directory, removed when
#                         every gate passed, kept and named in the summary otherwise)
#   CI_TAIL_LINES         log lines printed for a failed step (default 40)
#   CI_PATH_PLACEHOLDERS  user names allowed in tracked paths, as an ERE alternation
#                         (default: someone|user|username|you|example|USER)
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
cd "$ROOT" || exit 2
if [ ! -d n64dk ] || [ ! -d tests ]; then
    echo "ci_gates: $ROOT is not the kit's repository root" >&2
    exit 2
fi

have() { command -v "$1" >/dev/null 2>&1; }

if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif have python3; then PY=python3
else PY=python
fi
if ! have "$PY"; then
    echo "ci_gates: no Python interpreter ($PY)" >&2
    exit 2
fi

MEM=${CI_MEMORY_MAX:-4G}
VMEM_FACTOR=${CI_VMEM_FACTOR:-2}
MIN_AVAIL=${CI_MIN_AVAILABLE_MB:-2048}
T_SHORT=${CI_TIMEOUT_SHORT:-600}
T_SUITE=${CI_TIMEOUT_SUITE:-2400}
NICE_LEVEL=${CI_NICE:-15}
TAIL_LINES=${CI_TAIL_LINES:-40}
PLACEHOLDERS=${CI_PATH_PLACEHOLDERS:-someone|user|username|you|example|USER}
LOG_DIR=${CI_LOG_DIR:-}
OWN_LOG_DIR=
if [ -z "$LOG_DIR" ]; then
    LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/n64dk-ci.XXXXXX") || exit 2
    OWN_LOG_DIR=yes
fi
mkdir -p "$LOG_DIR" || exit 2

# --- the memory cap: systemd scope when the user manager answers, else ulimit -v -----------------
to_kib() {            # 4G / 4096M / 4194304K / bytes -> KiB; empty on a bad value
    case $1 in
        *[Gg]) n=${1%?}; unit=1048576 ;;
        *[Mm]) n=${1%?}; unit=1024 ;;
        *[Kk]) n=${1%?}; unit=1 ;;
        *) n=$1; unit=0 ;;
    esac
    case $n in ''|*[!0-9]*) return 0 ;; esac
    if [ "$unit" -eq 0 ]; then echo $((n / 1024)); else echo $((n * unit)); fi
}
CAP_MODE=none
CAP_NOTE="no memory cap (CI_MEMORY_MAX=off)"
if [ "$MEM" != off ]; then
    MEM_KIB=$(to_kib "$MEM")
    if [ -z "$MEM_KIB" ]; then
        echo "ci_gates: CI_MEMORY_MAX=$MEM is not a size like 4G or 4096M" >&2
        exit 2
    fi
    if have systemd-run && systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 \
            true >/dev/null 2>&1; then
        CAP_MODE=systemd
        CAP_NOTE="systemd scope MemoryMax=$MEM MemorySwapMax=0"
    else
        CAP_MODE=ulimit
        VMEM_KIB=$((MEM_KIB * VMEM_FACTOR))
        CAP_NOTE="ulimit -v ${VMEM_KIB} KiB (no systemd user manager; address space, not resident memory)"
        # glibc reserves 64 MiB of address space per malloc arena and thread: keep that bounded
        MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
        export MALLOC_ARENA_MAX
    fi
fi
IONICE=
if have ionice && ionice -c3 true >/dev/null 2>&1; then IONICE=yes; fi
if ! have timeout; then
    echo "ci_gates: 'timeout' (coreutils) is required: no step runs without a time limit" >&2
    exit 2
fi

# --- optional machine-wide lock ---------------------------------------------------------------
if [ -n "${CI_LOCK:-}" ]; then
    if ! have flock; then
        echo "ci_gates: CI_LOCK is set but flock is not available" >&2
        exit 2
    fi
    exec 9>"$CI_LOCK" || exit 2
    echo "ci_gates: waiting for $CI_LOCK"
    if ! flock -w "${CI_LOCK_WAIT_S:-86400}" 9; then
        echo "ci_gates: $CI_LOCK was held for ${CI_LOCK_WAIT_S:-86400} s; refusing" >&2
        exit 1
    fi
fi

available_mb() {
    if [ -r /proc/meminfo ]; then
        awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo
    fi
}

# --- step bookkeeping -------------------------------------------------------------------------
SUMMARY=$LOG_DIR/summary.txt
: > "$SUMMARY"
FAILED=0
STEP_NO=0

record() {            # record STATUS NAME DETAIL
    printf '%-5s %-17s %s\n' "$1" "$2" "$3" >> "$SUMMARY"
    printf 'ci_gates: %-5s %-17s %s\n' "$1" "$2" "$3"
    if [ "$1" = FAIL ]; then
        FAILED=$((FAILED + 1))
    fi
}

skipped_by_user() {
    case " ${CI_SKIP:-} " in *" $1 "*) return 0 ;; esac
    return 1
}

stop_if_fail_fast() {
    if [ "$FAILED" -gt 0 ] && [ "${CI_FAIL_FAST:-0}" = 1 ]; then
        finish
    fi
}

# run_step NAME TIMEOUT_S COMMAND...: one step, capped, timed, niced; its output to a log file.
run_step() {
    name=$1; limit=$2; shift 2
    STEP_NO=$((STEP_NO + 1))
    log=$LOG_DIR/$(printf '%02d' "$STEP_NO")-$name.log
    if skipped_by_user "$name"; then
        record SKIP "$name" "(CI_SKIP)"
        return 0
    fi
    if [ "$MIN_AVAIL" -gt 0 ]; then
        avail=$(available_mb)
        if [ -n "$avail" ] && [ "$avail" -lt "$MIN_AVAIL" ]; then
            record FAIL "$name" "refused: ${avail} MB available, CI_MIN_AVAILABLE_MB=$MIN_AVAIL"
            stop_if_fail_fast
            return 0
        fi
    fi
    if have nice; then set -- nice -n "$NICE_LEVEL" "$@"; fi
    if [ -n "$IONICE" ]; then set -- ionice -c3 "$@"; fi
    set -- timeout -k 30 "$limit" "$@"
    started=$(date +%s)
    echo "ci_gates: running $name (limit ${limit} s): $*" > "$log"
    case $CAP_MODE in
        systemd)
            systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 \
                -p CPUWeight=20 -p IOWeight=10 -- "$@" >> "$log" 2>&1 ;;
        ulimit)
            (ulimit -v "$VMEM_KIB" && exec "$@") >> "$log" 2>&1 ;;
        *)
            "$@" >> "$log" 2>&1 ;;
    esac
    rc=$?
    seconds=$(( $(date +%s) - started ))
    if [ "$rc" -eq 0 ]; then
        record PASS "$name" "${seconds} s"
    else
        case $rc in
            124) why="timed out after ${limit} s" ;;
            137) why="killed (memory cap, or the timeout's KILL)" ;;
            *)   why="exit $rc" ;;
        esac
        if [ "$rc" -eq 77 ]; then
            # a step that cannot run here (no git work tree, no 3.11 interpreter) exits 77
            if [ "${CI_REQUIRE_PY311:-0}" = 1 ] && [ "$name" = python3.11 ]; then
                record FAIL "$name" "skipped but CI_REQUIRE_PY311=1: $(tail -n 1 "$log")"
            else
                record SKIP "$name" "$(tail -n 1 "$log")"
            fi
        else
            record FAIL "$name" "$why, ${seconds} s, log $log"
            echo "---- last $TAIL_LINES lines of $log"
            tail -n "$TAIL_LINES" "$log"
            echo "----"
        fi
    fi
    stop_if_fail_fast
    return 0
}

finish() {
    echo
    echo "== ci_gates summary ($CAP_NOTE)"
    cat "$SUMMARY"
    if [ "$FAILED" -gt 0 ]; then
        echo "== FAILED: $FAILED step(s); logs in $LOG_DIR"
        exit 1
    fi
    if [ -n "$OWN_LOG_DIR" ]; then
        rm -rf "$LOG_DIR"                      # a temporary directory of our own: nothing to keep
        echo "== all gates passed"
    else
        echo "== all gates passed; logs in $LOG_DIR"
    fi
    exit 0
}

# --- the checks that need more than one command, as one small helper program -------------------
HELPER=$LOG_DIR/ci_check.py
cat > "$HELPER" <<'EOF'
"""ci_gates.sh helper: run an n64dk command in JSON mode and check what it printed."""
import json
import subprocess
import sys
from pathlib import Path

REQUIRED_COMMANDS = {"decompile-run", "decompile-status", "self-test", "operation-schema", "workspace-audit"}


def n64dk(*argv):
    proc = subprocess.run([sys.executable, "-m", "n64dk", "--format", "json", *argv],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(proc.stderr[-4000:])
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        sys.exit(f"n64dk {' '.join(argv)} exited {proc.returncode}")
    try:
        return json.loads(proc.stdout)
    except ValueError as error:
        print(proc.stdout[-4000:])
        sys.exit(f"n64dk {' '.join(argv)} did not print JSON: {error}")


def schema():
    found = n64dk("operation-schema")
    problems = []
    if not isinstance(found, dict) or found.get("protocol") != "n64dk.operations.v1":
        sys.exit(f"unexpected protocol: {found.get('protocol') if isinstance(found, dict) else type(found)}")
    if not isinstance(found.get("global_arguments"), list):
        problems.append("global_arguments is not a list")
    commands = found.get("commands")
    if not isinstance(commands, list) or not commands:
        sys.exit("no commands in the schema")
    names = []
    for row in commands:
        name = row.get("name") if isinstance(row, dict) else None
        if not isinstance(name, str) or not name:
            problems.append(f"a command without a name: {row!r}"[:200])
            continue
        names.append(name)
        if not isinstance(row.get("description"), str):
            problems.append(f"{name}: description is not a string")
        if not isinstance(row.get("arguments"), list):
            problems.append(f"{name}: arguments is not a list")
            continue
        for field in row["arguments"]:
            if not isinstance(field, dict) or not isinstance(field.get("dest"), str) \
                    or not isinstance(field.get("flags"), list) or not isinstance(field.get("required"), bool) \
                    or not isinstance(field.get("multiple"), bool):
                problems.append(f"{name}: malformed argument {field!r}"[:200])
    if len(set(names)) != len(names):
        problems.append("duplicate command names")
    missing = REQUIRED_COMMANDS - set(names)
    if missing:
        problems.append(f"missing commands: {sorted(missing)}")
    json.dumps(found)
    for problem in problems:
        print(problem)
    print(f"operation-schema: {len(names)} commands, {len(problems)} problems")
    return 1 if problems else 0


def self_test():
    found = n64dk("self-test")
    print(str(found.get("output", ""))[-6000:])
    print(f"self-test: ok={found.get('ok')} exit_code={found.get('exit_code')}")
    return 0 if found.get("ok") is True and found.get("exit_code") == 0 else 1


def audit():
    found = n64dk("workspace-audit", "--project", ".")
    summary = found.get("summary") or {}
    for item in (found.get("generated") or [])[:50]:
        print(f"{item.get('kind')}\t{'tracked' if item.get('tracked') else 'untracked'}\t{item.get('path')}")
    print(f"workspace-audit: {summary}, truncated={found.get('truncated')}, git={found.get('git')}")
    return 1 if summary.get("files", 1) or found.get("truncated") else 0


def compile_all(roots):
    """Compile (in memory, no .pyc written) every .py file with this interpreter."""
    problems, files = [], 0
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            files += 1
            try:
                compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as error:
                problems.append(f"{path}:{getattr(error, 'lineno', '?')}: {error}")
    for problem in problems:
        print(problem)
    print(f"{sys.version.split()[0]}: {files} files compiled, {len(problems)} problems")
    return 1 if problems or not files else 0


if __name__ == "__main__":
    task = sys.argv[1]
    sys.exit({"schema": schema, "self-test": self_test, "audit": audit}[task]() if task != "compile"
             else compile_all(sys.argv[2:]))
EOF

# --- the tracked-files check, as a script so that it runs under run_step like every other step ----
TRACKED=$LOG_DIR/tracked_files.sh
cat > "$TRACKED" <<'EOF'
# ci_gates.sh helper: no SQLite sidecar and no machine-specific path is tracked in git.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "not a git work tree: nothing to check"
    exit 77
fi
bad=0
sidecars=$(git ls-files | grep -E '(\.sqlite-wal|\.sqlite-shm|-wal|-shm|-journal)$')
if [ -n "$sidecars" ]; then
    echo "SQLite sidecar files are tracked:"
    echo "$sidecars"
    bad=1
fi
names=$(git ls-files | grep -E '^(home|mnt|Users)/|/home/|/mnt/[a-zA-Z]/Users/')
if [ -n "$names" ]; then
    echo "tracked file names with machine-specific paths:"
    echo "$names"
    bad=1
fi
# contents: /home/<name> or /mnt/<drive>/Users/<name>, with placeholder names allowed
found=$(git grep -I -n -o -E '(/home/|/mnt/[a-zA-Z]/Users/)[A-Za-z0-9._-]+' -- . \
        | grep -v -E "(/home/|/mnt/[a-zA-Z]/Users/)($CI_PATH_PLACEHOLDERS)\$")
if [ -n "$found" ]; then
    echo "machine-specific paths in tracked files (placeholder names allowed: $CI_PATH_PLACEHOLDERS):"
    echo "$found"
    bad=1
fi
if [ "$bad" -eq 0 ]; then
    echo "no tracked sidecars, no machine-specific paths in $(git ls-files | wc -l | tr -d ' ') tracked files"
fi
exit "$bad"
EOF
CI_PATH_PLACEHOLDERS=$PLACEHOLDERS
export CI_PATH_PLACEHOLDERS

# --- the 3.11 interpreter: CI_PY311, python3.11 on PATH, or PYTHON itself when it is 3.11 ------
PY311=${CI_PY311:-}
if [ -z "$PY311" ] && have python3.11; then PY311=python3.11; fi
if [ -z "$PY311" ] && "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' >/dev/null 2>&1; then
    PY311=$PY
fi

echo "ci_gates: repository $ROOT, python $("$PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)," \
     "$CAP_NOTE, logs in $LOG_DIR"


run_step tracked-files "$T_SHORT" sh "$TRACKED"

run_step compileall "$T_SHORT" "$PY" -m compileall -q n64dk tests

if [ -n "$PY311" ]; then
    run_step python3.11 "$T_SHORT" "$PY311" "$HELPER" compile n64dk tests
elif [ -n "${CI_PY311_CHECK:-}" ]; then
    run_step python3.11 "$T_SHORT" "$PY" "$CI_PY311_CHECK" n64dk tests
else
    run_step python3.11 "$T_SHORT" sh -c 'echo "no python3.11 interpreter (set CI_PY311 or CI_PY311_CHECK)"; exit 77'
fi

run_step operation-schema "$T_SHORT" "$PY" "$HELPER" schema
run_step unit-suite "$T_SUITE" "$PY" -m unittest discover -s tests -t .
run_step self-test "$T_SUITE" "$PY" "$HELPER" self-test
run_step workspace-audit "$T_SHORT" "$PY" "$HELPER" audit

finish
