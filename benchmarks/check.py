"""`run.py check`: tests the adapters against the rules marked [checked] in
docs/benchmarks/rules.md, without measuring anything that counts.

For each library and scenario:
- values (rule 1.2): the adapter's `values` command evaluates fixed points, including the optimum,
  and must agree with problems.py;
- runs (rules 1.3, 2.1, 2.3, 2.4, 3.3): two short runs; each reported solution must evaluate to the
  reported best, stay within its problem's domain, and each run must end only at the
  target, its budget (plus at most one generation) or its time cap, with no evaluation outside the
  bounds and a first hit of the target consistent with the run. `run.py` applies the same checks
  (check_run) to every timed run;
- threads (rule 4.3): the adapter's CPU time, over all its check runs, must stay within 10% of their
  wall time;
- repeat (rule 5.2): the same seed, twice, must give the same evaluations and results, and seed 1
  must give the same alone as after seed 0 in the same process.

The (library, scenario) pairs are checked in parallel, `--jobs` at a time, each in a process of its
own; the output is grouped per library. A library passes when every scenario it runs passes. The
result is saved with a hash of the adapter's files, the reference problems, the pinned Python
libraries and, for genoxide, its sources; a timed run refuses a library whose hash changed since it
last passed.
"""

import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import resource
import subprocess
import tempfile
import time

import problems
import run

CHECK_EVALUATIONS = 20_000
CHECK_SECONDS = 10.0
REPEAT_EVALUATIONS = 2_000
# CPU time over wall time above this means other threads did work (rule 4.3); judged over all of a
# library's short runs together, so a fast adapter is judged too
THREADS = 1.10
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


BUILD_OUTPUTS = {"target", "__pycache__", "build", ".gradle"}
# compiled extension modules, e.g. from `maturin develop` into python/genoxide
BUILT_SUFFIXES = {".so", ".pyd", ".dll", ".dylib", ".pyc"}


def fingerprint_files(name):
    """The files a library's check depends on: its adapter, the reference problems, the pinned
    Python libraries and, for genoxide, the sources it's built from."""
    repository = run.ROOT.parent
    trees = [run.ROOT / "adapters" / name]
    files = [run.ROOT / "problems.py", run.ROOT / "requirements.txt"]
    if name in ("genoxide", "genoxide_python"):
        # the Python package builds the crate
        trees.append(repository / "src")
        files += [repository / "Cargo.toml", repository / "Cargo.lock"]
    if name == "genoxide_python":
        trees += [repository / "python" / "src", repository / "python" / "genoxide"]
        files += [repository / "python" / "Cargo.toml", repository / "python" / "pyproject.toml"]
    for tree in trees:
        for path in tree.rglob("*"):
            if (path.is_file() and not set(path.relative_to(tree).parts) & BUILD_OUTPUTS
                    and path.suffix not in BUILT_SUFFIXES):
                files.append(path)
    return sorted({path for path in files if path.is_file()})


def adapter_hash(name):
    """A hash of the files of fingerprint_files, without build outputs."""
    digest = hashlib.sha256()
    for path in fingerprint_files(name):
        digest.update(path.relative_to(run.ROOT.parent).as_posix().encode())
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
    lower, upper = problems.REAL_BOUNDS[problem]
    return len(solution) == size and all(lower - 1e-12 <= v <= upper + 1e-12 for v in solution)


def check_run(r, problem, size, budget, cap):
    """The failures of one run, in `run.py check` and in every timed run."""
    where = f"{r.get('solver')} seed {r.get('seed')}"
    fields = ["library", "solver", "problem", "size", "mode", "seed", "time_s", "generations", "evaluations",
              "last_generation", "best", "target", "success", "solution", "first_hit"]
    missing = [field for field in fields if field not in r]
    if missing:
        return [f"{where}: missing {', '.join(missing)}"]
    if (r["problem"], r["size"]) != (problem, size):
        return [f"{where}: a run of {r['problem']} {r['size']} in the scenario of {problem} {size}"]
    failures = []
    evaluations, generations = r["evaluations"], max(r["generations"], 1)
    # a generation's size: the average, or the last one's, which can be larger (a restart's initial
    # population, IPOP's growing populations, a library that evaluates only changed children), as
    # the adapter counted it; no generation has more evaluations than the run
    last = r["last_generation"]
    if isinstance(last, bool) or not isinstance(last, int) or not 0 <= last <= evaluations:
        failures.append(f"{where}: last_generation {last}, in a run of {evaluations} evaluations")
        last = 0
    per_generation = max(math.ceil(evaluations / generations), last)
    if evaluations > budget + per_generation:
        failures.append(f"{where}: {evaluations} evaluations, over the budget of {budget} by more than "
                        f"a generation ({per_generation})")
    capped = r["time_s"] >= CAPPED * cap
    # rule 2.4: every evaluated solution inside the bounds
    if problem in problems.REAL_BOUNDS:
        if "outside" not in r:
            failures.append(f"{where}: missing outside (rule 2.4)")
        elif r["outside"] != 0:
            failures.append(f"{where}: {r['outside']} evaluated solutions outside the bounds (rule 2.4)")
    solution = r["solution"]
    if not in_domain(problem, size, solution):
        return failures + [f"{where}: the solution isn't valid: {short(solution)}"]
    expected = problems.value(problem, size, solution)
    if not problems.close(r["best"], expected, 1e-6):
        failures.append(f"{where}: best {r['best']}, but its solution evaluates to {expected}")
    reached = problems.reached(problem, size, expected)
    if bool(r["success"]) != reached:
        failures.append(f"{where}: success {r['success']}, but the solution's value {expected} says {reached}")
    goal = problems.target(problem, size)
    if not (r["target"] == goal or (r["target"] is not None and goal is not None
                                    and problems.close(r["target"], goal))):
        failures.append(f"{where}: target {r['target']}, but the scenario's is {goal}")
    # a run the library itself ended with an error (a crash, rule 8.4) says so in "ended_by": it
    # ends there, not reached, since the matched methods have no restarts
    if not reached and evaluations < budget and not capped and not r.get("ended_by"):
        failures.append(f"{where}: ended after {evaluations} of {budget} evaluations in {r['time_s']:.1f} s "
                        + ("without reaching the target" if goal is not None else "of its fixed budget")
                        + " (rule 2.2: it must keep going)")
    return failures + check_first_hit(r, where, reached)


def check_first_hit(r, where, reached):
    """Rule 3.3: `first_hit` is {"evaluations": E, "time_s": T} of the first evaluation that reached the
    target, within the run, or null when the target was never reached."""
    hit = r["first_hit"]
    if hit is None:
        return [f"{where}: first_hit null, but the solution reaches the target"] if reached else []
    if not reached:
        return [f"{where}: first_hit {short(hit)}, but the solution doesn't reach the target"]
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    if (not isinstance(hit, dict) or set(hit) != {"evaluations", "time_s"} or not number(hit["evaluations"])
            or not float(hit["evaluations"]).is_integer() or not number(hit["time_s"])):
        return [f"{where}: first_hit {short(hit)} isn't {{\"evaluations\": integer, \"time_s\": number}}"]
    failures = []
    if not 1 <= hit["evaluations"] <= r["evaluations"]:
        failures.append(f"{where}: first hit at evaluation {hit['evaluations']}, of {r['evaluations']}")
    if not 0 <= hit["time_s"] <= r["time_s"]:
        failures.append(f"{where}: first hit at {hit['time_s']} s, in a run of {r['time_s']} s")
    return failures


def check_scenario(name, problem, size, mode, budget, cap):
    """(ran, failures, notes, wall seconds, CPU seconds) of one library in one scenario, the seconds
    of its short runs. It runs in a worker process of its own: RUSAGE_CHILDREN then counts only its
    adapter processes."""
    adapter = run.ADAPTERS[name]
    budget = min(budget, CHECK_EVALUATIONS)
    cap = min(cap, CHECK_SECONDS)
    command = adapter["command"] + [problem, str(size), mode, "0", "1", str(budget), str(float(cap))]
    stdout, stderr, code, wall, cpu = execute(command)
    if code != 0:
        return True, [f"the adapter failed: {stderr.strip()[-300:]}"], [], 0.0, 0.0
    runs = runs_of(stdout)
    if not runs:
        return False, [], [], 0.0, 0.0
    failures = []
    for r in runs:
        failures += check_run(r, problem, size, budget, cap)
    notes = [f"CPU/wall {cpu / wall:.2f}"] if wall > 0 else []

    # with a budget of evaluations only: seed 0 alone, seeds 0 and 1 in one process, seed 1 alone.
    # Seed 0 must repeat, and seed 1 must give the same after seed 0 as alone: no state may leak
    # from one run into the next.
    results = {}
    for first, last in (("0", "0"), ("0", "1"), ("1", "1")):
        command = adapter["command"] + [problem, str(size), mode, first, last, str(REPEAT_EVALUATIONS), "600"]
        stdout, stderr, code, _, _ = execute(command)
        if code != 0:
            return True, failures + [f"repeat: the adapter failed: {stderr.strip()[-300:]}"], notes, wall, cpu
        results[first, last] = {(r["solver"], r["seed"]): repeatable(r) for r in runs_of(stdout)}
    for (a, b, seed, what) in ((("0", "0"), ("0", "1"), 0, "seed 0 gives different results"),
                               (("0", "1"), ("1", "1"), 1, "seed 1 gives different results after seed 0 "
                                                           "than alone")):
        alone = {solver: value for (solver, s), value in results[a].items() if s == seed}
        other = {solver: value for (solver, s), value in results[b].items() if s == seed}
        differ = sorted(solver for solver in alone.keys() | other.keys() if alone.get(solver) != other.get(solver))
        if not differ:
            continue
        if adapter.get("unseeded"):
            notes.append(f"not repeatable ({', '.join(differ)}): {adapter['unseeded']}")
        else:
            failures.append(f"repeat: {what} for {', '.join(differ)}")

    failures += check_values(adapter, problem, size)
    return True, failures, notes, wall, cpu


def repeatable(r):
    """What must be the same for the same seed."""
    hit = r.get("first_hit")
    return (r["evaluations"], r["best"], hit.get("evaluations") if isinstance(hit, dict) else None)


def short(value, limit=80):
    text = json.dumps(value)
    return text if len(text) <= limit else text[:limit] + "..."


# ------------------------------------------------------------------------------------------------


def check(libraries, scenarios, jobs=None):
    """Checks the libraries, `jobs` scenarios at a time (default: one per core), prints the outcome
    grouped per library and saves it; returns whether they all passed."""
    for name in libraries:
        adapter = run.ADAPTERS[name]
        if adapter.get("build"):
            print(f"building {name} adapter ...", flush=True)
            subprocess.run(adapter["build"], check=True)
    jobs = jobs or os.cpu_count()
    print(f"checking {len(libraries)} libraries in {len(scenarios)} scenarios, {jobs} at a time ...", flush=True)
    all_passed = True
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        # each library's scenarios, in the order of the libraries, so the first ones finish first
        futures = {name: [(scenario, pool.submit(check_scenario, name, *scenario))
                          for scenario in scenarios]
                   for name in libraries}
        for name in libraries:
            all_passed &= report(name, futures[name], len(scenarios) == len(run.SCENARIOS))
    return all_passed


def report(name, futures, complete):
    """Prints one library's checks, once they're all done, and saves the outcome if `complete`, a
    check of every scenario; returns whether it passed."""
    library_failures = 0
    wall = cpu = 0.0
    for (problem, size, mode, *_), future in futures:
        scenario = run.scenario_name(problem, size, mode)
        try:
            ran, failures, notes, scenario_wall, scenario_cpu = future.result()
        except Exception as error:  # the check itself failed: a failure of the library's check
            ran, failures, notes, scenario_wall, scenario_cpu = True, [f"the check failed: {error!r}"], [], 0.0, 0.0
        if not ran:
            print(f"  {name} {scenario}: doesn't run", flush=True)
            continue
        wall += scenario_wall
        cpu += scenario_cpu
        status = "FAIL" if failures else "pass"
        print(f"  {name} {scenario}: {status}{'  (' + ', '.join(notes) + ')' if notes else ''}", flush=True)
        for failure in failures:
            print(f"      {failure}", flush=True)
        library_failures += len(failures)
    if wall > 0:
        ratio = cpu / wall
        verdict = "FAIL: it used more than one thread" if ratio > THREADS else "pass"
        print(f"  {name} threads: CPU {cpu:.1f} s over {wall:.1f} s of wall time ({ratio:.2f}): {verdict}",
              flush=True)
        library_failures += ratio > THREADS
    passed = library_failures == 0
    print(f"{name}: {'PASSED' if passed else f'FAILED ({library_failures} failures)'}", flush=True)
    # only a check of every scenario counts for a timed run
    if complete:
        save(name, {"hash": adapter_hash(name), "passed": passed, "date": time.strftime("%Y-%m-%d %H:%M")})
    return passed


def save(name, entry):
    """Records one library's check. The file is locked while it's re-read and replaced, so checks of
    other libraries saved at the same time, from this process or another, aren't overwritten, and a
    reader never sees it half written."""
    CHECK_FILE.parent.mkdir(exist_ok=True)
    with open(CHECK_FILE.with_suffix(".lock"), "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = json.loads(CHECK_FILE.read_text(encoding="utf-8")) if CHECK_FILE.exists() else {}
        saved[name] = entry
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CHECK_FILE.parent, delete=False) as file:
            file.write(json.dumps(saved, indent=2))
        os.replace(file.name, CHECK_FILE)


def unchecked(libraries):
    """The libraries whose current adapter hasn't passed a check of every scenario."""
    saved = json.loads(CHECK_FILE.read_text(encoding="utf-8")) if CHECK_FILE.exists() else {}
    return [name for name in libraries
            if not saved.get(name, {}).get("passed") or saved[name]["hash"] != adapter_hash(name)]
