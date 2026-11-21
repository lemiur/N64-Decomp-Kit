#!/bin/sh
# Mock refiner for `n64dk decompile-run --refine-command` (see docs/workflows.md).
#
# Contract of a refiner: the prompt arrives on stdin, one fenced C block leaves on stdout.  This
# mock reads and discards the prompt and always answers with the same trivial function, so a run
# exercises the whole refine path (prompt, cache, build, compare, strict verification, budgets)
# without a model and without sending anything anywhere.  Its answers almost never match: that is
# the point (a wrong answer costs builds, never credit).
cat >/dev/null
printf '%s\n' '```c' 'int n64dk_mock_refiner_answer(void) {' '    return 0;' '}' '```'
