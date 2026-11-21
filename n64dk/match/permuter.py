"""Random source-permutation search on top of the matching engine (needs ``pycparser``).

Preparation makes a source parseable and relocatable on its own: ``__asm__("label")``
aliases are removed from declarations and the C names they covered are recorded as symbols at
the label's address (the ``LDSYM`` idea), then the file is run through the host C
preprocessor.  Each iteration applies 1..k random AST mutations to one function, renders C,
builds it with the configured pipeline and scores it with the classified instruction diff
(0 = exact).  Candidates are evaluated ``jobs`` at a time; a candidate replaces the current
base when its score is lower, or equal with probability ``temperature``.

The search follows the approach of decomp-permuter and of the Global Assault project's
tools/decomp_permuter.py (GPL-3.0); only a small, semantics-preserving mutation set is used
here.  Results are candidates for review, not proof of original source.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import symbols as symmod
from .engine import Target, match_source
from .pipeline import BuildSpec
from .toolchain import Config

ASM_LABEL = re.compile(r"\s*__asm__\s*\(\s*\"([^\"]+)\"\s*\)")
DECL_WITH_LABEL = re.compile(
    r"([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*(?:\((?:[^()]|\([^()]*\))*\)\s*)?)__asm__\s*\(\s*\"([^\"]+)\"\s*\)")
COMMUTATIVE = {"+", "*", "&", "|", "^", "==", "!="}
FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}

# pycparser's AST nodes are recursively deep-copied for each candidate.  Large
# decompilation functions can legitimately exceed Python's default depth.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))


def available() -> bool:
    try:
        import pycparser  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class Prepared:
    text: str                       # preprocessed, parseable C
    symbols: dict[str, int]         # C names -> retail addresses
    function: str                   # C name of the function to permute


def prepare(source: Path, symbols: symmod.SymbolMap, includes: list[Path], function: str | None,
            vram: int) -> Prepared:
    raw = Path(source).read_text(errors="replace")
    extra = symmod.from_source(raw, symbols)
    names: dict[str, int] = dict(extra)
    func_name = function
    for m in DECL_WITH_LABEL.finditer(raw):
        cname, label = m.group(1), m.group(3)
        addr = extra.get(label, symbols.resolve(label))
        if addr is not None:
            names[cname] = addr
            if func_name is None and addr == vram:
                func_name = cname
    stripped = ASM_LABEL.sub("", raw)
    defines = ["-D__attribute__(x)=", "-D__extension__=", "-D__inline__=inline", "-D__asm__(x)="]
    try:
        from .. import safety
        proc = safety.run_tool(["cpp", "-P", "-nostdinc", "-undef", *defines, *[f"-I{p}" for p in includes],
                                "-I", str(Path(source).parent), "-"], kind="light", input=stripped,
                               capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("preprocessing timed out after 120s") from error
    if proc.returncode:
        raise RuntimeError("preprocessing failed: " + proc.stderr.strip()[-400:])
    if func_name is None:
        raise RuntimeError("cannot tell which function to permute; pass the function name")
    return Prepared(proc.stdout, names, func_name)


# ------------------------------------------------------------------------- mutations
def _nodes(node, cls):
    out = []

    def walk(n):
        if isinstance(n, cls):
            out.append(n)
        for _, child in n.children():
            walk(child)
    walk(node)
    return out


def _ids(node) -> set[str]:
    from pycparser import c_ast
    return {n.name for n in _nodes(node, c_ast.ID)}


def _copy_equality_sites(fn):
    """Find same-type scalar copies that can safely carry a one-shot loop.

    Immediately after ``left = right``, equal non-floating scalar values compare
    equal.  The loop is therefore semantically inert, but its loop notes can
    influence GCC 2.7's CSE, allocator, and scheduler before it disappears.
    """
    from pycparser import c_ast, c_generator
    generator = c_generator.CGenerator()
    declarations = {}
    parameter_decls = []
    if getattr(fn.decl.type, "args", None) is not None:
        parameter_decls = [item for item in fn.decl.type.args.params
                           if isinstance(item, c_ast.Decl)]
    for declaration in [*parameter_decls, *_nodes(fn.body, c_ast.Decl)]:
        if not declaration.name or "volatile" in (declaration.quals or []):
            continue
        spelling = generator.visit(declaration.type)
        if any(word in spelling for word in ("float", "double", "struct ", "union ", "[", "(")):
            continue
        declarations[declaration.name] = spelling
    sites = []
    for compound in _nodes(fn.body, c_ast.Compound):
        for index, statement in enumerate(compound.block_items or []):
            left = right = None
            if (isinstance(statement, c_ast.Assignment) and statement.op == "="
                    and isinstance(statement.lvalue, c_ast.ID)
                    and isinstance(statement.rvalue, c_ast.ID)):
                left, right = statement.lvalue.name, statement.rvalue.name
            elif (isinstance(statement, c_ast.Decl) and statement.name
                  and isinstance(statement.init, c_ast.ID)):
                left, right = statement.name, statement.init.name
            if (left and left != right and declarations.get(left)
                    and declarations.get(left) == declarations.get(right)):
                sites.append((compound, index, left, right))
    return sites


def _mutations(fn, rng: random.Random) -> str | None:
    from pycparser import c_ast
    choices = []
    binops = _nodes(fn.body, c_ast.BinaryOp)
    if any(b.op in COMMUTATIVE for b in binops):
        choices.append("commute")
    if any(b.op in FLIP for b in binops):
        choices.append("flip")
    compounds = [c for c in _nodes(fn.body, c_ast.Compound) if c.block_items and len(c.block_items) > 1]
    wrap_do_sites = []
    unwrap_do_sites = []
    control_nodes = (c_ast.Break, c_ast.Continue, c_ast.Goto, c_ast.Label,
                     c_ast.Case, c_ast.Default)
    for compound in compounds:
        items = compound.block_items
        for index, item in enumerate(items):
            if (isinstance(item, c_ast.DoWhile)
                    and isinstance(item.cond, c_ast.Constant)
                    and item.cond.value in {"0", "0U", "0u"}
                    and isinstance(item.stmt, c_ast.Compound)):
                unwrap_do_sites.append((compound, index))
            for length in range(1, min(4, len(items) - index) + 1):
                selected = items[index:index + length]
                if (any(isinstance(statement, c_ast.Decl) for statement in selected)
                        or any(_nodes(statement, control_nodes) for statement in selected)):
                    continue
                wrap_do_sites.append((compound, index, length))
    copy_equality_sites = _copy_equality_sites(fn)
    if copy_equality_sites:
        choices.append("copy-equality-loop")
    if compounds:
        choices += ["swap-decl", "swap-stmt"]
    if wrap_do_sites:
        choices.append("wrap-do")
    if unwrap_do_sites:
        choices.append("unwrap-do")
    split_initializers = []
    merge_initializers = []
    merge_copy_chains = []
    split_copy_chains = []
    for compound in compounds:
        items = compound.block_items
        for index, item in enumerate(items):
            if (isinstance(item, c_ast.Decl) and item.name and item.init is not None
                    and not item.storage and "const" not in (item.quals or [])
                    and not isinstance(item.type, c_ast.ArrayDecl)
                    and not _nodes(item.init, c_ast.FuncCall)):
                split_initializers.append((compound, index))
            if index + 1 < len(items):
                assignment = items[index + 1]
                if (isinstance(item, c_ast.Decl) and item.name and item.init is None
                        and not item.storage and isinstance(assignment, c_ast.Assignment)
                        and assignment.op == "=" and isinstance(assignment.lvalue, c_ast.ID)
                        and assignment.lvalue.name == item.name):
                    merge_initializers.append((compound, index))
                first = item
                second = assignment
                if (isinstance(first, c_ast.Assignment) and first.op == "="
                        and isinstance(first.lvalue, c_ast.ID)
                        and isinstance(second, c_ast.Assignment) and second.op == "="
                        and isinstance(second.lvalue, c_ast.ID)
                        and isinstance(second.rvalue, c_ast.ID)
                        and second.rvalue.name == first.lvalue.name
                        and second.lvalue.name != first.lvalue.name):
                    merge_copy_chains.append((compound, index))
            if (isinstance(item, c_ast.Assignment) and item.op == "="
                    and isinstance(item.lvalue, c_ast.ID)
                    and isinstance(item.rvalue, c_ast.Assignment)
                    and item.rvalue.op == "="
                    and isinstance(item.rvalue.lvalue, c_ast.ID)
                    and item.lvalue.name != item.rvalue.lvalue.name):
                split_copy_chains.append((compound, index))
    if split_initializers:
        choices.append("split-init")
    if merge_initializers:
        choices.append("merge-init")
    if merge_copy_chains:
        choices.append("merge-copy-chain")
    if split_copy_chains:
        choices.append("split-copy-chain")
    assignments = _nodes(fn.body, c_ast.Assignment)
    expand_compounds = [a for a in assignments
                        if a.op in {"+=", "-=", "*=", "&=", "|=", "^=", "<<=", ">>="}
                        and isinstance(a.lvalue, c_ast.ID)]
    make_compounds = [a for a in assignments if a.op == "="
                      and isinstance(a.lvalue, c_ast.ID)
                      and isinstance(a.rvalue, c_ast.BinaryOp)
                      and a.rvalue.op in {"+", "-", "*", "&", "|", "^", "<<", ">>"}
                      and isinstance(a.rvalue.left, c_ast.ID)
                      and a.rvalue.left.name == a.lvalue.name]
    if expand_compounds:
        choices.append("expand-compound")
    if make_compounds:
        choices.append("make-compound")
    ifs = [i for i in _nodes(fn.body, c_ast.If) if i.iffalse is not None]
    if ifs:
        choices.append("invert-if")
    if not choices:
        return None
    kind = rng.choice(choices)
    if kind == "commute":
        b = rng.choice([b for b in binops if b.op in COMMUTATIVE])
        b.left, b.right = b.right, b.left
    elif kind == "flip":
        b = rng.choice([b for b in binops if b.op in FLIP])
        b.left, b.right, b.op = b.right, b.left, FLIP[b.op]
    elif kind == "invert-if":
        i = rng.choice(ifs)
        # Keep repeated population mutations from building an ever-deeper
        # ``!(!(...))`` tree.  Inverting an already-negated condition can
        # remove the outer negation while swapping the arms, which is the
        # same semantics-preserving transformation in canonical form.
        if isinstance(i.cond, c_ast.UnaryOp) and i.cond.op == "!":
            i.cond = i.cond.expr
        else:
            i.cond = c_ast.UnaryOp("!", i.cond)
        i.iftrue, i.iffalse = i.iffalse, i.iftrue
    elif kind == "split-init":
        compound, index = rng.choice(split_initializers)
        declaration = compound.block_items[index]
        initializer = declaration.init
        declaration.init = None
        compound.block_items.insert(index + 1,
                                    c_ast.Assignment("=", c_ast.ID(declaration.name), initializer))
    elif kind == "merge-init":
        compound, index = rng.choice(merge_initializers)
        declaration = compound.block_items[index]
        assignment = compound.block_items[index + 1]
        declaration.init = assignment.rvalue
        del compound.block_items[index + 1]
    elif kind == "merge-copy-chain":
        compound, index = rng.choice(merge_copy_chains)
        first = compound.block_items[index]
        second = compound.block_items[index + 1]
        compound.block_items[index] = c_ast.Assignment(
            "=", second.lvalue,
            c_ast.Assignment("=", first.lvalue, first.rvalue))
        del compound.block_items[index + 1]
    elif kind == "split-copy-chain":
        compound, index = rng.choice(split_copy_chains)
        outer = compound.block_items[index]
        inner = outer.rvalue
        compound.block_items[index:index + 1] = [
            c_ast.Assignment("=", inner.lvalue, inner.rvalue),
            c_ast.Assignment("=", outer.lvalue, c_ast.ID(inner.lvalue.name)),
        ]
    elif kind == "expand-compound":
        assignment = rng.choice(expand_compounds)
        op = assignment.op[:-1]
        assignment.op = "="
        assignment.rvalue = c_ast.BinaryOp(op, c_ast.ID(assignment.lvalue.name),
                                           assignment.rvalue)
    elif kind == "make-compound":
        assignment = rng.choice(make_compounds)
        assignment.op = assignment.rvalue.op + "="
        assignment.rvalue = assignment.rvalue.right
    elif kind == "copy-equality-loop":
        compound, index, left, right = rng.choice(copy_equality_sites)
        loop = c_ast.While(
            c_ast.Constant("int", "1"),
            c_ast.Compound([
                c_ast.If(c_ast.BinaryOp("==", c_ast.ID(left), c_ast.ID(right)),
                         c_ast.Break(), None),
            ]),
        )
        compound.block_items.insert(index + 1, loop)
    elif kind == "wrap-do":
        compound, index, length = rng.choice(wrap_do_sites)
        selected = compound.block_items[index:index + length]
        compound.block_items[index:index + length] = [
            c_ast.DoWhile(c_ast.Constant("int", "0"), c_ast.Compound(selected))
        ]
    elif kind == "unwrap-do":
        compound, index = rng.choice(unwrap_do_sites)
        wrapper = compound.block_items[index]
        compound.block_items[index:index + 1] = list(wrapper.stmt.block_items or [])
    else:
        c = rng.choice(compounds)
        pairs = []
        for k in range(len(c.block_items) - 1):
            a, b = c.block_items[k], c.block_items[k + 1]
            if kind == "swap-decl" and isinstance(a, c_ast.Decl) and isinstance(b, c_ast.Decl) \
                    and a.init is None and b.init is None:
                pairs.append(k)
            elif kind == "swap-stmt" and isinstance(a, c_ast.Assignment) and isinstance(b, c_ast.Assignment) \
                    and isinstance(a.lvalue, c_ast.ID) and isinstance(b.lvalue, c_ast.ID) \
                    and a.lvalue.name != b.lvalue.name and not _nodes(a, c_ast.FuncCall) \
                    and not _nodes(b, c_ast.FuncCall) and a.lvalue.name not in _ids(b.rvalue) \
                    and b.lvalue.name not in _ids(a.rvalue):
                pairs.append(k)
        if not pairs:
            return None
        k = rng.choice(pairs)
        c.block_items[k], c.block_items[k + 1] = c.block_items[k + 1], c.block_items[k]
    return kind


@dataclass
class PermuteResult:
    base_score: int
    best_score: int
    best_source: str
    iterations: int
    exact: bool
    history: list[tuple[int, int]] = field(default_factory=list)  # (iteration, score)
    # the best few results that differ in what they compile to (score, output digest, source): two sources
    # that compile to the same code are one state, however different they read
    frontier: list[tuple[int, str, str]] = field(default_factory=list)


def generate_pool(source: Path, symbols: symmod.SymbolMap, includes: list[Path],
                  function: str | None, vram: int, count: int, seed: int | None,
                  max_mutations: int, parent_score: int, output: Path) -> int:
    """Generate unique, uncompiled candidates for an external batch ranker."""
    from pycparser import c_ast, c_generator, c_parser
    prep = prepare(source, symbols, includes, function, vram)
    base = c_parser.CParser().parse(prep.text, filename=str(source))
    gen = c_generator.CGenerator()
    rng = random.Random(seed)

    def find(tree):
        for ext in tree.ext:
            if isinstance(ext, c_ast.FuncDef) and ext.decl.name == prep.function:
                return ext
        raise RuntimeError(f"function {prep.function} not found")

    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    base_digest = hashlib.sha1(gen.visit(base).encode()).hexdigest()
    population = [(base, [], base_digest, 0)]
    attempts = 0
    window_attempts = 0
    window_unique = 0
    with output.open("w") as stream:
        while len(seen) < count and attempts < count * 20:
            attempts += 1
            window_attempts += 1
            parent_tree, parent_mutations, parent_id, parent_depth = rng.choice(population)
            tree = copy.deepcopy(parent_tree)
            fn = find(tree)
            mutations = list(parent_mutations)
            for _ in range(rng.randint(1, max_mutations)):
                mutation = _mutations(fn, rng)
                if mutation is not None:
                    mutations.append(mutation)
            text = gen.visit(tree)
            digest = hashlib.sha1(text.encode()).hexdigest()
            if digest in seen:
                if window_attempts >= 1000 and window_unique * 10 < window_attempts:
                    break
                continue
            seen.add(digest)
            window_unique += 1
            member = (tree, mutations[-64:], digest, parent_depth + 1)
            if len(population) < 64:
                population.append(member)
            else:
                population[rng.randrange(1, len(population))] = member
            stream.write(json.dumps({
                "id": digest, "function": prep.function, "source": text,
                "mutations": mutations, "parent_score": parent_score,
                "parent_id": parent_id, "generation_depth": parent_depth + 1,
                "compiled": False, "seed": seed, "iteration": attempts,
                "symbols": prep.symbols,
            }, sort_keys=True) + "\n")
            if window_attempts >= 1000:
                if window_unique * 10 < window_attempts:
                    break
                window_attempts = 0
                window_unique = 0
    return len(seen)


def permute(source: Path, spec: BuildSpec, cfg: Config, target: Target, vram: int, size: int | None,
            symbols: symmod.SymbolMap, includes: list[Path], function: str | None = None,
            iterations: int = 200, jobs: int = 4, seed: int | None = None, max_mutations: int = 3,
            temperature: float = 0.1, out_dir: Path | None = None,
            replay_log: Path | None = None, islands: int = 1, keep: int = 4) -> PermuteResult:
    from pycparser import c_ast, c_generator, c_parser
    prep = prepare(source, symbols, includes, function, vram)
    syms = symbols.copy()
    syms.update(prep.symbols)
    ast = c_parser.CParser().parse(prep.text, filename=str(source))
    gen = c_generator.CGenerator()
    rng = random.Random(seed)
    seen: dict[str, int] = {}
    outputs: dict[str, str] = {}                # source digest -> digest of what it compiled to
    frontier: dict[str, tuple[int, str]] = {}   # output digest -> (score, source)

    def find(tree):
        for ext in tree.ext:
            if isinstance(ext, c_ast.FuncDef) and ext.decl.name == prep.function:
                return ext
        raise RuntimeError(f"function {prep.function} not found")

    def note(text: str, value: int) -> None:
        """Keep ``text`` if it is one of the ``keep`` best states that compile to different code."""
        if value >= 1 << 30:
            return
        state = outputs.get(hashlib.sha1(text.encode()).hexdigest()) or "source:" + hashlib.sha1(text.encode()).hexdigest()
        held = frontier.get(state)
        if held is None or value < held[0]:
            frontier[state] = (value, text)
        if len(frontier) > max(1, keep):
            del frontier[max(frontier, key=lambda key: frontier[key][0])]

    with tempfile.TemporaryDirectory(prefix="n64dk-permute-") as tmp:
        work = Path(tmp)

        def score(text: str) -> int:
            key = hashlib.sha1(text.encode()).hexdigest()
            if key in seen:
                return seen[key]
            path = work / f"cand_{key[:16]}.c"
            path.write_text(text)
            o = match_source(path, spec, cfg, target, vram, size, syms, work, includes=[],
                             infer_externals=True)
            if o.exact:
                val = 0
            elif o.result is None:
                val = 1 << 30
            else:
                val = max(1, o.result.diff.score if o.result.diff else 1)
            seen[key] = val
            if o.digest:
                outputs[key] = o.digest
            path.unlink(missing_ok=True)
            return val

        base_text = gen.visit(ast)
        base_score = best_score = current_score = score(base_text)
        best_text, current = base_text, ast
        history = [(0, base_score)]
        note(base_text, base_score)
        # islands: independent walkers from the same start, so one early lucky step does not decide the whole search
        walkers = [[ast, base_score] for _ in range(max(1, islands))]
        turn = 0
        if replay_log is not None:
            replay_log.parent.mkdir(parents=True, exist_ok=True)
            replay_log.write_text(json.dumps({
                "id": hashlib.sha1(base_text.encode()).hexdigest(), "function": prep.function,
                "source": base_text, "mutations": [], "parent_score": base_score,
                "score": base_score, "compiled": True, "exact": base_score == 0,
                "iteration": 0, "seed": seed,
            }, sort_keys=True) + "\n")
        def checkpoint() -> None:
            if out_dir is None:
                return
            out_dir.mkdir(parents=True, exist_ok=True)
            header = "".join(f"/* LDSYM {k}=0x{v:08X} */\n"
                             for k, v in sorted(prep.symbols.items()))
            (out_dir / "best.c").write_text(header + best_text)
            for rank, (state, (value, text)) in enumerate(sorted(frontier.items(), key=lambda item: item[1][0]), 1):
                (out_dir / f"frontier_{rank}.c").write_text(f"/* score {value}; output {state[:16]} */\n" + header + text)
            (out_dir / "progress.tsv").write_text(
                "iteration\tscore\n" + "".join(f"{iteration}\t{value}\n"
                                                  for iteration, value in history))
        checkpoint()
        it = 0
        with ThreadPoolExecutor(jobs) as pool:
            while it < iterations and best_score > 0:
                cands = []
                walker = walkers[turn % len(walkers)]
                turn += 1
                current, current_score = walker
                for _ in range(min(jobs, iterations - it)):
                    tree = copy.deepcopy(current)
                    fn = find(tree)
                    mutations = []
                    for _ in range(rng.randint(1, max_mutations)):
                        mutation = _mutations(fn, rng)
                        if mutation is not None:
                            mutations.append(mutation)
                    cands.append((tree, gen.visit(tree), mutations))
                it += len(cands)
                unique = list(dict.fromkeys(text for _, text, _ in cands))
                by_text = dict(zip(unique, pool.map(score, unique)))
                scores = [by_text[text] for _, text, _ in cands]
                if replay_log is not None:
                    with replay_log.open("a") as stream:
                        for (_tree, text, mutations), sc in zip(cands, scores):
                            stream.write(json.dumps({
                                "id": hashlib.sha1(text.encode()).hexdigest(),
                                "function": prep.function, "source": text,
                                "mutations": mutations, "parent_score": current_score,
                                "score": sc, "compiled": True, "exact": sc == 0,
                                "iteration": it, "seed": seed,
                            }, sort_keys=True) + "\n")
                for (tree, text, _mutations_used), sc in zip(cands, scores):
                    note(text, sc)
                    if sc < current_score or (sc == current_score and rng.random() < temperature):
                        current, current_score = tree, sc
                    if sc < best_score:
                        best_score, best_text = sc, text
                        history.append((it, sc))
                        checkpoint()
                walker[0], walker[1] = current, current_score
    if out_dir is not None:
        checkpoint()
    ranked = sorted(((value, state, text) for state, (value, text) in frontier.items()), key=lambda item: item[0])
    return PermuteResult(base_score, best_score, best_text, it, best_score == 0, history, ranked)
