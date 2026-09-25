"""`run.py check`: tests the adapters against the rules marked [checked] in
docs/benchmarks/rules.md, without measuring anything that counts.

For each library and scenario:
- values (rule 1.2): the adapter's `values` command evaluates fixed points, including the optimum,
  and must agree with problems.py;
- runs (rules 1.3, 2.1, 2.3): two short runs; each reported solution must evaluate to the reported
  best (or front), stay within its problem's domain, and each run must end only at the target, its
  budget (plus at most one generation) or its time cap;
- threads (rule 4.3): the adapter's CPU time must stay within 10% of its wall time;
- repeat (rule 5.2): the same seed, twice, must give the same evaluations and results.

A library passes when every scenario it runs passes. The result is saved with a hash of the
adapter's files, and a timed run refuses a library whose adapter changed since it last passed.
"""

import hashlib
import json
import math
import resource
import subprocess
import time

import problems
import run

CHECK_EVALUATIONS = 20_000
CHECK_SECONDS = 10.0
REPEAT_EVALUATIONS = 2_000
# CPU time over wall time above this means other threads did work (rule 4.3); shorter runs are
# dominated by startup and aren't judged
THREADS = 1.10
THREADS_MIN_SECONDS = 1.0
# a run that took this share of its cap was stopped by it
CAPPED = 0.98

CHECK_FILE = run.ROOT / "results" / "check.json"


def execute(command, stdin=None):
    """(stdout, stderr, return code, wall seconds, CPU seconds of the process and its children)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    completed = subprocess.run(command, input=stdin, capture_output=True, text=True, cwd=run.ROOT)
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return completed.stdout, completed.stderr, completed.returncode, wall, cpu


def runs_of(stdout):
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def adapter_hash(name):
    """A hash of the adapter's source files, without its build outputs."""
    folder = run.ROOT / "adapters" / name
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        parts = set(path.relative_to(folder).parts)
        if path.is_file() and not parts & {"target", "__pycache__", "build", ".gradle"}:
            digest.update(str(path.relative_to(folder)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------------
# The checks, each returning a list of failures
# ------------------------------------------------------------------------------------------------


def check_values(adapter, problem, size):
    points = problems.check_points(problem, size)
    stdin = "".join(json.dumps(point) + "\n" for point in points)
    stdout, stderr, code, _, _ = execute(adapter["command"] + ["values", problem, str(size)], stdin)
    if code != 0:
        return [f"values: the adapter failed: {stderr.strip()[-300:]}"]
    values = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if len(values) != len(points):
        return [f"values: {len(points)} points sent, {len(values)} values received"]
    failures = []
    for point, got in zip(points, values):
        expected = problems.value(problem, size, point)
        if not problems.close(got, expected):
            failures.append(f"values: {got} instead of {expected} at {short(point)}")
    return failures[:3]


def in_domain(problem, size, solution):
    if problem == "onemax":
        return len(solution) == size and all(bit in (0, 1, True, False) for bit in solution)
    if problem == "nqueens":
        return sorted(solution) == list(range(size))
    if problem in problems.REAL_BOUNDS:
        lower, upper = problems.REAL_BOUNDS[problem]
        return len(solution) == size and all(lower - 1e-12 <= v <= upper + 1e-12 for v in solution)
    n = problems.FRONT_VARIABLES[problem](size)
    return len(solution) == n and all(-1e-12 <= v <= 1 + 1e-12 for v in solution)


def check_run(r, problem, size, budget, cap):
    """The failures of one run."""
    where = f"{r.get('solver')} seed {r.get('seed')}"
    common = ["library", "solver", "problem", "size", "mode", "seed", "time_s", "generations", "evaluations"]
    front = problem in problems.FRONT_VARIABLES
    fields = common + (["front", "solutions"] if front else ["best", "target", "success", "solution"])
    missing = [field for field in fields if field not in r]
    if missing:
        return [f"{where}: missing {', '.join(missing)}"]
    failures = []
    evaluations, generations = r["evaluations"], max(r["generations"], 1)
    per_generation = math.ceil(evaluations / generations)
    if evaluations > budget + per_generation:
        failures.append(f"{where}: {evaluations} evaluations, over the budget of {budget} by more than "
                        f"a generation ({per_generation})")
    capped = r["time_s"] >= CAPPED * cap
    if front:
        if len(r["front"]) != len(r["solutions"]) or not r["front"]:
            failures.append(f"{where}: {len(r['front'])} points in the front, {len(r['solutions'])} solutions")
        for solution, point in zip(r["solutions"], r["front"]):
            if not in_domain(problem, size, solution):
                failures.append(f"{where}: a solution outside [0, 1]: {short(solution)}")
                break
            expected = problems.value(problem, size, solution)
            if not problems.close(point, expected, 1e-6):
                failures.append(f"{where}: front point {short(point)}, but its solution evaluates to {short(expected)}")
                break
        if evaluations < budget and not capped:
            failures.append(f"{where}: ended after {evaluations} of {budget} evaluations in {r['time_s']:.1f} s")
        return failures
    solution = r["solution"]
    if not in_domain(problem, size, solution):
        return failures + [f"{where}: the solution isn't valid: {short(solution)}"]
    expected = problems.value(problem, size, solution)
    if not problems.close(r["best"], expected, 1e-6):
        failures.append(f"{where}: best {r['best']}, but its solution evaluates to {expected}")
    reached = problems.reached(problem, size, expected)
    if bool(r["success"]) != reached:
        failures.append(f"{where}: success {r['success']}, but the solution's value {expected} says {reached}")
    if not reached and evaluations < budget and not capped:
        failures.append(f"{where}: ended after {evaluations} of {budget} evaluations in {r['time_s']:.1f} s "
                        "without reaching the target (rule 2.2: it must keep going)")
    return failures


def check_scenario(name, adapter, problem, size, mode, budget):
    """(ran, failures, notes) of one library in one scenario."""
    budget = min(budget, CHECK_EVALUATIONS)
    command = adapter["command"] + [problem, str(size), mode, "0", "1", str(budget), str(CHECK_SECONDS)]
    stdout, stderr, code, wall, cpu = execute(command)
    if code != 0:
        return True, [f"the adapter failed: {stderr.strip()[-300:]}"], []
    runs = runs_of(stdout)
    if not runs:
        return False, [], []
    failures = []
    for r in runs:
        failures += check_run(r, problem, size, budget, CHECK_SECONDS)
    notes = []
    if wall >= THREADS_MIN_SECONDS:
        ratio = cpu / wall
        notes.append(f"CPU/wall {ratio:.2f}")
        if ratio > THREADS:
            failures.append(f"threads: CPU time {cpu:.1f} s over {wall:.1f} s of wall time ({ratio:.2f}): "
                            "it used more than one thread")

    # the same seed twice, with a budget of evaluations only
    repeat = adapter["command"] + [problem, str(size), mode, "0", "0", str(REPEAT_EVALUATIONS), "600"]
    results = []
    for _ in range(2):
        stdout, stderr, code, _, _ = execute(repeat)
        if code != 0:
            return True, failures + [f"repeat: the adapter failed: {stderr.strip()[-300:]}"], notes
        results.append({r["solver"]: repeatable(r) for r in runs_of(stdout)})
    if results[0] != results[1]:
        differ = sorted(s for s in results[0] if results[0][s] != results[1].get(s))
        if adapter.get("unseeded"):
            notes.append(f"not repeatable ({', '.join(differ)}): {adapter['unseeded']}")
        else:
            failures.append(f"repeat: seed 0 gives different results for {', '.join(differ)}")

    failures += check_values(adapter, problem, size)
    return True, failures, notes


def repeatable(r):
    """What must be the same for the same seed."""
    if "front" in r:
        return (r["evaluations"], json.dumps(r["front"]))
    return (r["evaluations"], r["best"])


def short(value, limit=80):
    text = json.dumps(value)
    return text if len(text) <= limit else text[:limit] + "..."


# ------------------------------------------------------------------------------------------------


def check(libraries, scenarios):
    """Checks the libraries, prints and saves the outcome; returns whether they all passed."""
    saved = json.loads(CHECK_FILE.read_text(encoding="utf-8")) if CHECK_FILE.exists() else {}
    all_passed = True
    for name in libraries:
        adapter = run.ADAPTERS[name]
        if adapter.get("build"):
            print(f"building {name} adapter ...", flush=True)
            subprocess.run(adapter["build"], check=True)
        library_failures = 0
        for problem, size, mode, budget in scenarios:
            scenario = run.scenario_name(problem, size, mode)
            ran, failures, notes = check_scenario(name, adapter, problem, size, mode, budget)
            if not ran:
                print(f"  {name} {scenario}: doesn't run", flush=True)
                continue
            status = "FAIL" if failures else "pass"
            print(f"  {name} {scenario}: {status}{'  (' + ', '.join(notes) + ')' if notes else ''}", flush=True)
            for failure in failures:
                print(f"      {failure}", flush=True)
            library_failures += len(failures)
        passed = library_failures == 0
        all_passed &= passed
        print(f"{name}: {'PASSED' if passed else f'FAILED ({library_failures} failures)'}", flush=True)
        # only a check of every scenario counts for a timed run
        if len(scenarios) == len(run.SCENARIOS):
            saved[name] = {"hash": adapter_hash(name), "passed": passed,
                           "date": time.strftime("%Y-%m-%d %H:%M")}
    CHECK_FILE.parent.mkdir(exist_ok=True)
    CHECK_FILE.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return all_passed


def unchecked(libraries):
    """The libraries whose current adapter hasn't passed a check of every scenario."""
    saved = json.loads(CHECK_FILE.read_text(encoding="utf-8")) if CHECK_FILE.exists() else {}
    return [name for name in libraries
            if not saved.get(name, {}).get("passed") or saved[name]["hash"] != adapter_hash(name)]
