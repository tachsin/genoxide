"""Where each library's methods are set up in its adapter: the code the project site's benchmark
page shows beside a method's runs (run.py writes it into the run details, runs/<scenario>.json).

METHOD_CODE maps a library to its adapter's file (relative to adapters/) and, per solver (the
"solver" its runs print), a list of (scenarios, blocks): the first entry whose scenarios match a
scenario gives the blocks of code that build and run the solver there, the setup first. Scenarios
are "*" (every one), "real" (Rastrigin, Rosenbrock, Ackley), a problem ("onemax") or a problem
and its mode ("onemax-matched"), several separated by spaces.

A block is found by symbol, not by line number, so the map survives changes to the adapters (and
the adapters don't change for it: their hashes decide which ones need a new check):
- "start": the text a line starts with, after its indentation. It must match one line of the
  file, or the first after the line that "after" starts, when given.
- The block ends where the start line's indentation ends: the lines after it that are blank or
  indented more, and then a closing line at the same indentation (`}`, `)`, `]`, `end`). "end"
  instead: the text its last line starts with, from the start line on.
- The comments (and attributes or decorators) directly above the start line, at its indentation,
  are part of it: they cite the library's docs for the settings.
- "file": another file of the adapter than the library's.
"""

from pathlib import Path

ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"
# the most lines of a block the details keep; the link to the code covers all of it
MAX_LINES = 80
REAL = {"rastrigin", "rosenbrock", "ackley"}
LANGUAGES = {".py": "python", ".rs": "rust", ".java": "java", ".jl": "julia", ".cpp": "cpp"}
COMMENTS = ("#", "//", "/*", "*", "@")
CLOSERS = ("}", ")", "]")

RADIATE_BINARY = {"start": "let build = |budget: Arc<Budget>| {", "after": "for solver in IDIOMATIC_BINARY {"}
RADIATE_PERMUTATION = {"start": "let build = |budget: Arc<Budget>| {", "after": "for solver in IDIOMATIC_PERMUTATION {"}
RADIATE_REAL = {"start": "let build = |budget: Arc<Budget>| {", "after": "for solver in IDIOMATIC_REAL {"}
MOORS = [{"start": "macro_rules! ga {"}, {"start": "macro_rules! run {"}]
GENOXIDE_PYTHON_SOLVE = {"start": "def solve("}
PYMOO_SOLVE = {"start": "def solve("}
PYCMA = [{"start": "def solve("}, {"start": "def solvers("}]
NEVERGRAD_ONEMAX = {"start": 'if problem == "onemax":', "after": "def __init__(self, problem, size):"}
NEVERGRAD_NQUEENS = {"start": 'elif problem == "nqueens":'}
NEVERGRAD_REAL = {"start": "else:", "after": 'elif problem == "nqueens":'}
NEVERGRAD_RUN = {"start": "def run("}
OPENGA_REAL = [{"start": "void solve_real("}, {"start": "RunResult run_ga("}]
JENETICS_GA = 'solvers.add(new Solver("ga", (budget, seed) -> evolve('
JENETICS_EVOLVE = {"start": "static <G extends Gene<?, G>, C extends Comparable<? super C>> long evolve(",
                   "after": "return evolve(engine, budget, 0, seed);"}
JMETAL_GA = 'solvers.add(new Solver("ga", (budget, seed) -> runComponent(budget, termination ->'
JMETAL_COMPONENT = {"start": "static <S extends Solution<?>> long runComponent("}
EVOLUTIONARY_RESTARTS = {"start": "function run_restarting(start, budget::Budget, seed)"}
EVOLUTIONARY_ONEMAX = {"start": 'tolerance = mode == "matched" ? typemax(Int) : 10', "end": "end"}
EVOLUTIONARY_REAL = [{"start": "solve(method; successive_f_tol = 10) = (budget, seed) -> run_restarting(budget, seed) do rng"},
                     EVOLUTIONARY_RESTARTS]
METAHEURISTICS = [{"start": "function algorithm_kwargs(budget, seed)"},
                  {"start": "function run_restarting(start, budget::Budget, seed)"}]
METAHEURISTICS_REAL = {"start": "solve(make) = (budget, seed) -> optimize(", "after": "function real_solvers(problem, size)"}

METHOD_CODE = {
    "genoxide": {
        "file": "genoxide/src/main.rs",
        "methods": {
            "ga": [
                ("onemax-matched", [{"start": 'if args.mode == "matched" {'}]),
                ("onemax", [{"start": "let build = |seed| {",
                             "after": 'return solve(args, seed, "ga", target, build, onemax, anywhere);'}]),
                ("nqueens", [{"start": "let build = |seed| {",
                              "after": 'solve(args, seed, "local_search", 0.0, build, nqueens, anywhere)?;'}]),
                ("rastrigin ackley", [{"start": "let ga = |seed| {"}]),
            ],
            "local_search": [("nqueens", [{"start": "let build = |seed| {", "after": "fn run_nqueens("}])],
            "cma_es": [("real", [{"start": "let cmaes = |seed| {"}])],
            "de": [("real", [{"start": "let de = |seed|"}])],
            "es": [("rosenbrock", [{"start": "let es = |seed| {"}])],
        },
    },
    "genetic_algorithm": {
        "file": "genetic_algorithm/src/main.rs",
        "methods": {
            "evolve": [
                ("onemax-idiomatic", [{"start": "|budget| {", "after": "fn onemax("}]),
                ("real", [{"start": "let solve = |budget: &Budget| {", "after": "fn real_evolve("}]),
            ],
            "hill_climb": [
                ("nqueens", [{"start": "|budget| {", "after": "fn nqueens("}, {"start": "fn nqueens_genotype("}]),
                ("rosenbrock", [{"start": "let solve = |budget: &Budget| {", "after": "fn real_hill_climb("}]),
            ],
        },
    },
    "moors": {
        "file": "moors/src/main.rs",
        "methods": {
            "ga": [
                ("onemax-idiomatic", [{"start": '("onemax", _) => run!('}, *MOORS]),
                ("nqueens", [{"start": '("nqueens", _) => run!('}, *MOORS]),
                ("real", [{"start": "(name, _) => {"}, *MOORS]),
            ],
        },
    },
    "radiate": {
        "file": "radiate/src/main.rs",
        "methods": {
            "ga": [
                ("onemax-matched", [{"start": 'if args.mode == "matched" {'}]),
                ("onemax", [RADIATE_BINARY]),
                ("nqueens", [RADIATE_PERMUTATION]),
                ("real", [RADIATE_REAL]),
            ],
            "ga_uniform": [("onemax", [RADIATE_BINARY])],
            "ga_multipoint": [("onemax", [RADIATE_BINARY])],
            "ga_pmx": [("nqueens", [RADIATE_PERMUTATION])],
            "ga_blend": [("real", [RADIATE_REAL])],
            "ga_intermediate": [("real", [RADIATE_REAL])],
        },
    },
    "genoxide_python": {
        "file": "genoxide_python/bench.py",
        "methods": {
            "ga": [
                ("onemax-matched", [{"start": 'return [("ga", lambda seed: gx.Ga(', "after": 'if mode == "matched":'},
                                    GENOXIDE_PYTHON_SOLVE]),
                ("onemax-idiomatic", [{"start": 'return [("ga", lambda seed: gx.Ga(', "after": "# idiomatic: the GA"},
                                      GENOXIDE_PYTHON_SOLVE]),
                ("nqueens", [{"start": '("ga", lambda seed: gx.Ga('}, GENOXIDE_PYTHON_SOLVE]),
                ("rastrigin ackley", [{"start": 'ga = ("ga", lambda seed: gx.Ga('}, GENOXIDE_PYTHON_SOLVE]),
            ],
            "local_search": [("nqueens", [{"start": '("local_search", lambda seed: gx.LocalSearch('},
                                          GENOXIDE_PYTHON_SOLVE])],
            "cma_es": [("real", [{"start": "def real_solvers(", "end": "function, True)"}, GENOXIDE_PYTHON_SOLVE])],
            "de": [("real", [{"start": "def real_solvers(", "end": 'de = ("de"'}, GENOXIDE_PYTHON_SOLVE])],
        },
    },
    "deap": {
        "file": "deap/bench.py",
        "methods": {
            "ga": [
                ("onemax", [{"start": "def solve_onemax("}, {"start": "def ea_simple("}]),
                ("nqueens", [{"start": "def solve_nqueens("}, {"start": "def ea_simple("}]),
            ],
            "cma_es": [("real", [{"start": "def solve_bipop_cmaes(", "end": "strategy = cma.Strategy("},
                                 {"start": "def bounded_evaluate("}])],
            "de": [
                ("rastrigin ackley", [{"start": "def solve_de("}, {"start": "def mut_de("},
                                      {"start": "def cx_exponential("}]),
                ("rosenbrock", [{"start": "def solve_de("}, {"start": "def bounded_evaluate("}]),
            ],
        },
    },
    "pygad": {
        "file": "pygad/bench.py",
        "methods": {
            "ga": [
                ("onemax", [{"start": 'if problem == "onemax":'}, {"start": "def run_single("}]),
                ("nqueens", [{"start": 'if problem == "nqueens":'}, {"start": "def run_single("}]),
                ("real", [{"start": "function, low, high = REAL_PROBLEMS[problem]",
                           "end": "return config, function, False"}, {"start": "def run_single("}]),
            ],
        },
    },
    "pymoo": {
        "file": "pymoo/bench.py",
        "methods": {
            "ga": [
                ("onemax-idiomatic", [{"start": "def onemax_solvers("}, PYMOO_SOLVE]),
                ("nqueens", [{"start": "def ga():"}, {"start": "def flowshop_convergence("}, PYMOO_SOLVE]),
            ],
            "brkga": [("nqueens", [{"start": "def brkga():"},
                                   {"start": "class PermutationDuplicateElimination(",
                                    "end": "return np.argsort(X, axis=1)"}, PYMOO_SOLVE])],
            "cma_es": [("real", [{"start": "def cma_es():"}, {"start": "def cma_es_seed("}, PYMOO_SOLVE])],
            "nelder_mead": [("rosenbrock", [{"start": 'if problem_name == "rosenbrock":'}, PYMOO_SOLVE])],
            "de": [("rastrigin ackley", [{"start": "def de():"}, PYMOO_SOLVE])],
            "es": [("rastrigin ackley", [{"start": "def es():"}, PYMOO_SOLVE])],
        },
    },
    "pycma": {
        "file": "pycma/bench.py",
        "methods": {
            "ipop_cma_es": [("*", PYCMA)],
            "bipop_cma_es": [("*", PYCMA)],
            "cma_es": [("*", PYCMA)],
            "lq_cma_es": [("*", PYCMA)],
        },
    },
    "nevergrad": {
        "file": "nevergrad/bench.py",
        "methods": {
            "ngiohtuned": [
                ("onemax", [{"start": '("ngiohtuned", ng.optimizers.NgIohTuned),', "after": "def solvers("},
                            NEVERGRAD_ONEMAX, NEVERGRAD_RUN]),
                ("nqueens", [{"start": '("ngiohtuned", ng.optimizers.NgIohTuned),', "after": 'if problem == "nqueens":'},
                             NEVERGRAD_NQUEENS, NEVERGRAD_RUN]),
                ("real", [{"start": '("ngiohtuned", ng.optimizers.NgIohTuned),',
                           "after": '("genetic_de", ng.optimizers.GeneticDE),'}, NEVERGRAD_REAL, NEVERGRAD_RUN]),
            ],
            "discrete_one_plus_one": [("*", [{"start": '("discrete_one_plus_one"'}, NEVERGRAD_ONEMAX, NEVERGRAD_RUN])],
            "portfolio_discrete_one_plus_one": [
                ("*", [{"start": '("portfolio_discrete_one_plus_one"'}, NEVERGRAD_ONEMAX, NEVERGRAD_RUN])],
            "rotated_two_points_de": [
                ("*", [{"start": '("rotated_two_points_de"'}, NEVERGRAD_NQUEENS, NEVERGRAD_RUN])],
            "genetic_de": [("*", [{"start": '("genetic_de", ng.optimizers.GeneticDE),'}, NEVERGRAD_NQUEENS,
                                  NEVERGRAD_RUN])],
            "one_plus_one": [("*", [{"start": '("one_plus_one"'}, NEVERGRAD_REAL, NEVERGRAD_RUN])],
            "cma_es": [("*", [{"start": '("cma_es"'}, NEVERGRAD_REAL, NEVERGRAD_RUN])],
        },
    },
    "scipy": {
        "file": "scipy/bench.py",
        "methods": {
            "de": [("*", [{"start": "def solve_de("}])],
            "dual_annealing": [("*", [{"start": "def solve_dual_annealing("}])],
            "direct": [("*", [{"start": "def solve_direct("}])],
            "lbfgsb": [("*", [{"start": '("lbfgsb"'}, {"start": "def solve_minimize("}])],
            "nelder_mead": [("*", [{"start": '("nelder_mead"'}, {"start": "def solve_minimize("}])],
        },
    },
    "pygmo": {
        "file": "pygmo/bench.py",
        "methods": {
            "ga": [("*", [{"start": '("ga", udp, ToBudget('}, {"start": "class ToBudget:"}])],
            "ihs": [("*", [{"start": '("ihs", udp, ToBudget('}, {"start": "class ToBudget:"}])],
            "gaco": [("*", [{"start": '("gaco", udp, Restarts('}, {"start": "class Restarts:"},
                            {"start": "def with_bfe("}])],
            "sade": [("*", [{"start": '("sade", udp, Restarts('}, {"start": "class Restarts:"}])],
            "cma_es": [("*", [{"start": "population = cmaes_population(problem, size)", "end": "population))"},
                              {"start": "def cmaes_population("}, {"start": "class Restarts:"}])],
            "xnes": [("*", [{"start": 'solvers.append(("xnes", udp, Restarts('}, {"start": "def cmaes_population("},
                            {"start": "class Restarts:"}])],
            "simulated_annealing": [("*", [{"start": 'solvers.append(("simulated_annealing", udp, Reanneal('},
                                           {"start": "class Reanneal:"}])],
        },
    },
    "openga": {
        "file": "openga/bench.cpp",
        "methods": {
            "ga": [
                ("onemax-idiomatic", [{"start": "void solve_onemax("}, {"start": "RunResult run_ga("}]),
                ("nqueens", [{"start": "void solve_nqueens("}, {"start": "RunResult run_ga("}]),
                ("real", [{"start": 'if (args.problem != "rosenbrock") {'}, *OPENGA_REAL]),
            ],
            "ga_assist": [("*", [{"start": "const RealSettings assist{"}, *OPENGA_REAL])],
        },
    },
    "jenetics": {
        "file": "jenetics/Bench.java",
        "methods": {
            "ga": [
                ("onemax-idiomatic", [{"start": JENETICS_GA, "after": 'case "onemax" -> {'}, JENETICS_EVOLVE]),
                ("nqueens", [{"start": JENETICS_GA, "after": 'case "nqueens" -> {'}, JENETICS_EVOLVE]),
                ("real", [{"start": JENETICS_GA, "after": 'case "rastrigin", "rosenbrock", "ackley" -> {'},
                          JENETICS_EVOLVE]),
            ],
        },
    },
    "jmetal": {
        "file": "jmetal/Bench.java",
        "methods": {
            "ga": [
                ("onemax-idiomatic", [{"start": JMETAL_GA, "after": 'case "onemax" -> {'}, JMETAL_COMPONENT]),
                ("nqueens", [{"start": JMETAL_GA, "after": 'case "nqueens" -> {'}, JMETAL_COMPONENT]),
                ("real", [{"start": JMETAL_GA, "after": 'case "rastrigin", "rosenbrock", "ackley" -> {'},
                          JMETAL_COMPONENT]),
            ],
            "es": [("onemax-idiomatic", [{"start": 'solvers.add(new Solver("es", (budget, seed) -> {'}])],
            "de": [("real", [{"start": 'solvers.add(new Solver("de", (budget, seed) -> {'}])],
            "cma_es": [("real", [{"start": 'solvers.add(new Solver("cma_es", (budget, seed) ->'},
                                 {"start": "static long cmaes("}])],
        },
    },
    "evolutionary_jl": {
        "file": "evolutionary_jl/bench.jl",
        "methods": {
            "ga": [
                ("onemax-matched", [{"start": 'if mode == "matched"'}, EVOLUTIONARY_ONEMAX, EVOLUTIONARY_RESTARTS]),
                ("onemax", [{"start": "method = () -> GA(selection = uniformranking(5)"}, EVOLUTIONARY_ONEMAX,
                            EVOLUTIONARY_RESTARTS]),
                ("nqueens", [{"start": "ga = (budget, seed) -> run_restarting(budget, seed) do rng"},
                             EVOLUTIONARY_RESTARTS]),
                ("rosenbrock", [{"start": 'third = ("ga", solve('}, *EVOLUTIONARY_REAL]),
            ],
            "es": [
                ("nqueens", [{"start": "es = (budget, seed) -> run_restarting(budget, seed) do rng"},
                             EVOLUTIONARY_RESTARTS]),
                ("rastrigin ackley", [{"start": 'third = ("es", solve(() -> ES('}, *EVOLUTIONARY_REAL]),
            ],
            "cma_es": [
                ("rosenbrock", [{"start": "cma_es = solve(() -> CMAES())"}, *EVOLUTIONARY_REAL]),
                ("rastrigin ackley", [{"start": "cma_es = solve(() -> CMAES(lambda = 100))"}, *EVOLUTIONARY_REAL]),
            ],
            "de": [
                ("rosenbrock", [{"start": "de = solve(() -> DE(populationSize = 100))"}, *EVOLUTIONARY_REAL]),
                ("rastrigin ackley", [{"start": "de = solve(() -> DE(populationSize = 100, F = 0.9); "
                                                "successive_f_tol = 25)"}, *EVOLUTIONARY_REAL]),
            ],
        },
    },
    "metaheuristics_jl": {
        "file": "metaheuristics_jl/bench.jl",
        "methods": {
            "ga": [
                ("onemax-idiomatic", [{"start": "run = function (budget, seed)"}, *METAHEURISTICS]),
                ("nqueens", [{"start": "ga = function (budget, seed)"}, *METAHEURISTICS]),
            ],
            "brkga": [("nqueens", [{"start": "brkga = function (budget, seed)"}, *METAHEURISTICS])],
            "eca": [("real", [{"start": '("eca", solve(kwargs -> ECA(; kwargs...)), identity),'},
                              METAHEURISTICS_REAL, METAHEURISTICS[0]])],
            "de": [("real", [{"start": '("de", solve(kwargs -> DE(; kwargs...)), identity),'},
                             METAHEURISTICS_REAL, METAHEURISTICS[0]])],
            "pso": [("real", [{"start": '("pso", solve(kwargs -> PSO(; kwargs...)), identity),'},
                              METAHEURISTICS_REAL, METAHEURISTICS[0]])],
        },
    },
}


def matches(key, problem, mode):
    """Whether a METHOD_CODE entry's scenarios include a scenario."""
    for token in key.split():
        if token == "*" or token == problem or token == f"{problem}-{mode}":
            return True
        if token == "real" and problem in REAL:
            return True
    return False


def indentation(line):
    return len(line) - len(line.lstrip())


def closes(text):
    return text.startswith(CLOSERS) or text == "end" or text.startswith(("end ", "end)", "end,", "end;"))


def locate(lines, block):
    """The 1-based first and last line of a block in a file's lines, or a ValueError saying why
    not."""
    begin = 0
    if block.get("after"):
        anchors = [i for i, line in enumerate(lines) if line.lstrip().startswith(block["after"])]
        if not anchors:
            raise ValueError(f"no line starts with {block['after']!r}")
        begin = anchors[0] + 1
    starts = [i for i in range(begin, len(lines)) if lines[i].lstrip().startswith(block["start"])]
    if not starts:
        raise ValueError(f"no line starts with {block['start']!r}")
    if len(starts) > 1 and not block.get("after"):
        raise ValueError(f"{len(starts)} lines start with {block['start']!r}: give `after`")
    start = starts[0]
    indent = indentation(lines[start])
    if block.get("end"):
        ends = [i for i in range(start, len(lines)) if lines[i].lstrip().startswith(block["end"])]
        if not ends:
            raise ValueError(f"no line after {block['start']!r} starts with {block['end']!r}")
        last = ends[0]
    else:
        last = start
        for i in range(start + 1, len(lines)):
            text = lines[i].strip()
            if not text:
                continue
            if indentation(lines[i]) > indent:
                last = i
                continue
            if indentation(lines[i]) == indent and closes(text):
                last = i
            break
    first = start
    while (first > 0 and lines[first - 1].strip() and indentation(lines[first - 1]) == indent
           and lines[first - 1].lstrip().startswith(COMMENTS)):
        first -= 1
    return first + 1, last + 1


def method_code(library, solver, problem, mode):
    """The blocks of code of a library's solver in a scenario: each its path in the repository,
    first and last line, language and excerpt (at most MAX_LINES lines, "lines" its full length).
    An empty list if METHOD_CODE doesn't know it; a block it can't find is printed and left out."""
    entry = METHOD_CODE.get(library)
    if not entry:
        return []
    blocks = next((blocks for key, blocks in entry["methods"].get(solver, []) if matches(key, problem, mode)), [])
    code = []
    for block in blocks:
        relative = block.get("file", entry["file"])
        path = ADAPTERS_DIR / relative
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            first, last = locate(lines, block)
        except (OSError, ValueError) as error:
            print(f"run details: {library} {solver} ({problem}-{mode}): {relative}: {error}", flush=True)
            continue
        excerpt = lines[first - 1:min(last, first - 1 + MAX_LINES)]
        code.append({
            "path": f"benchmarks/adapters/{relative}",
            "start": first,
            "end": last,
            "language": LANGUAGES.get(path.suffix, "text"),
            "excerpt": "\n".join(excerpt),
            **({"lines": last - first + 1} if last - first + 1 > len(excerpt) else {}),
        })
    return code
