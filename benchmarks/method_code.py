"""Where each library's methods are set up in its adapter: the code the project site's benchmark
page shows beside a method's runs (run.py writes it into the run details, runs/<scenario>.json).

METHOD_CODE maps a library to its adapter's file (relative to adapters/) and, per solver (the
"solver" its runs print: "ga", "de" or "cma_es", one per problem of the matched suite), a list of
(scenarios, blocks): the first entry whose scenarios match a scenario gives the blocks of code that
build and run the solver there, the setup first. Scenarios are "*" (every one), "real" (Rastrigin,
Rosenbrock), a problem ("onemax") or a problem and its mode ("onemax-matched"), several separated
by spaces.

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
REAL = {"rastrigin", "rosenbrock"}
LANGUAGES = {".py": "python", ".rs": "rust", ".java": "java", ".jl": "julia", ".cpp": "cpp"}
COMMENTS = ("#", "//", "/*", "*", "@")
CLOSERS = ("}", ")", "]")

EVOLUTIONARY_OPTIONS = {"start": "options(budget::Budget, rng) = Evolutionary.Options("}
EVOLUTIONARY_RUN = {"start": "function run_once(start, budget::Budget, seed, ended_by)"}

METHOD_CODE = {
    "genoxide": {
        "file": "genoxide/src/main.rs",
        "methods": {
            "ga": [("onemax", [{"start": "fn run_onemax("}])],
            "de": [("rastrigin", [{"start": "fn run_rastrigin("}])],
            "cma_es": [("rosenbrock", [{"start": "fn run_rosenbrock("}])],
        },
    },
    "genoxide_python": {
        "file": "genoxide_python/bench.py",
        "methods": {
            "ga": [("onemax", [{"start": "def onemax_ga("}, {"start": "def solve("}])],
            "de": [("rastrigin", [{"start": "def rastrigin_de("}, {"start": "def solve("}])],
            "cma_es": [("rosenbrock", [{"start": "def rosenbrock_cmaes("}, {"start": "def solve("}])],
        },
    },
    "radiate": {
        "file": "radiate/src/main.rs",
        "methods": {
            "ga": [("onemax", [{"start": "fn run_onemax("}])],
        },
    },
    "deap": {
        "file": "deap/bench.py",
        "methods": {
            "ga": [("onemax", [{"start": "def solve_onemax("}, {"start": "def ea_simple("}])],
            "cma_es": [("rosenbrock", [{"start": "def solve_cma_es("}, {"start": "def bounded_evaluate("}])],
        },
    },
    "pygad": {
        "file": "pygad/bench.py",
        "methods": {
            "ga": [("onemax", [{"start": "def onemax_ga("}, {"start": "def run_ga("}])],
        },
    },
    "pymoo": {
        "file": "pymoo/bench.py",
        "methods": {
            "de": [("rastrigin", [{"start": "def de(size, seed):"}, {"start": "def solve("}])],
            "cma_es": [("rosenbrock", [{"start": "def cma_es(size, seed):"}, {"start": "def solve("}])],
        },
    },
    "pycma": {
        "file": "pycma/bench.py",
        "methods": {
            "cma_es": [("rosenbrock", [{"start": "def solve_cma_es("}])],
        },
    },
    "scipy": {
        "file": "scipy/bench.py",
        "methods": {
            "de": [("rastrigin", [{"start": "def solve_de("}])],
        },
    },
    "pygmo": {
        "file": "pygmo/bench.py",
        "methods": {
            "de": [("rastrigin", [{"start": "def run_de("}])],
            "cma_es": [("rosenbrock", [{"start": "def run_cma_es("}, {"start": "def budget_generations("}])],
        },
    },
    "jmetal": {
        "file": "jmetal/Bench.java",
        "methods": {
            "de": [("rastrigin", [{"start": "static long de(Budget budget, RealProblem problem) {"},
                                  {"start": "static final class CountingEvaluator extends"}])],
            "cma_es": [("rosenbrock", [{"start": "static long cmaes("}, {"start": "static void seedCmaes("}])],
        },
    },
    "evolutionary_jl": {
        "file": "evolutionary_jl/bench.jl",
        "methods": {
            "ga": [("onemax", [{"start": "function onemax_ga(size)"}, EVOLUTIONARY_OPTIONS, EVOLUTIONARY_RUN])],
            "cma_es": [("rosenbrock", [{"start": "function rosenbrock_cma_es(size)"}, EVOLUTIONARY_OPTIONS,
                                       EVOLUTIONARY_RUN])],
        },
    },
    "metaheuristics_jl": {
        "file": "metaheuristics_jl/bench.jl",
        "methods": {
            "de": [("rastrigin", [{"start": "function rastrigin_de(size)"}, {"start": "function algorithm_kwargs("}])],
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
