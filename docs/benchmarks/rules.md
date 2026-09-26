# Benchmark rules

Every library is measured under these rules. An adapter that breaks one isn't benchmarked. `run.py check` tests each rule marked **[checked]** before any timed run; the others are reviewed in the adapter's code. Every timed run is checked again with the same run checks. A run that fails is invalid: it's left out of every table and chart, and the results list it with the reason.

## 1. The problems

1.1. The fitness functions are defined once, in Python, in `benchmarks/problems.py`. That definition is the reference.

1.2. Each adapter implements them in its library's language, as its users would. **[checked]** The adapter evaluates fixed points, including the optimum, and the values must match the reference to 1e-9 relative.

1.3. Each run prints the best solution it found, not only its value. **[checked]** `run.py` evaluates that solution with the reference; the run is invalid if the two values differ. It also recomputes whether the solution reaches the target. A multi-objective run prints the solutions of its front, and `run.py` computes their objectives itself.

1.4. Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift towards 0. The optimum is at `s_i = 0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1)`, for i from 0, where `upper` is the box's upper bound: 5.12 for Rastrigin, 32.768 for Ackley. Every adapter computes it in this order, in double precision. Rosenbrock isn't shifted: its optimum, all ones, is already away from the origin.

## 2. The budget

2.1. A run ends at the first of:
- the target, reached by the best solution found;
- the scenario's evaluation budget;
- 60 seconds.

Nothing else ends a run.

2.2. **Methods that stop by themselves keep going.**
- **A limit that's only a budget,** such as a maximum number of generations or iterations, is lifted.
- **A criterion that detects convergence,** such as a tolerance or a number of generations without improvement, ends that attempt, and the method starts again:
  - with the library's restart mechanism for the method, if it has one (IPOP for CMA-ES, for example);
  - otherwise from a new random start. The adapter keeps the best solution and counts every evaluation.

Seeds: attempt 0 uses the run's seed. Restart r, from 1, uses `(seed + 1) * 1_000_000 + r`, so no two runs share a seed. A library's own restart mechanism keeps its own seeding, but any seed the adapter passes follows this formula.

A user with time left would do the same. A converged method with its criterion turned off would spend the rest of the budget where it's stuck.

Which criteria count:
- **They count** when they're part of the method's own settings, or when the docs' example for the problem type sets them.
- **They count** when they stay in effect after the budget is set the documented way.
- **They don't count** when they're a fallback that the documented way of setting a budget replaces. pymoo's default termination, for example, is replaced by `termination=("n_evals", N)`.
- **Matched scenarios have none:** the matched configuration defines no convergence criterion.

2.3. A library that checks the stop only between generations may go past the target or the budget by at most one generation. **[checked]** Evaluations beyond the budget plus one generation make the run invalid. Every run reports `last_generation`, the evaluations its adapter counted since the start of its last generation (a restart's initial population is a generation), and a generation is the larger of that and the run's average.

2.4. **Only inside the bounds.** Every solution a method evaluates must lie inside the problem's box. The library's own bound handling is used: clipping, repair, a transformation or a bounded operator. The page says which. **[checked]** Each run of a continuous or multi-objective problem reports `outside`, the number of evaluated solutions outside the bounds, from the adapter's own counter. It must be 0.

## 3. Counting evaluations

3.1. An evaluation is one call of the fitness function on one solution.

3.2. The adapter counts them itself, with a counter around the fitness function, not with the library's own count. Every call counts: copies, restarts, local search and final polishing. A library that doesn't evaluate a copy again saves that evaluation. That's a real saving, and it's reported.

3.3. **The first hit.** Every single-objective run prints `"first_hit": {"evaluations": E, "time_s": T}`. E is the number of the first evaluation whose true value reaches the target, counting that evaluation. T is the run's clock at that evaluation. A run that never reaches the target prints `"first_hit": null`. The adapter's counter records it at each evaluation, whatever the library's stop granularity. The run still stops as rules 2.1 and 2.3 say. **[checked]** `first_hit` is present when the solution reaches the target and null otherwise; E is at most the run's evaluations, and T at most its time.

3.4. **Batch evaluation.** A Python library with a documented batch or vectorized evaluation interface evaluates a generation at once with numpy, as its users would: pycma's `fmin2(parallel_objective=...)`, PyGAD's `fitness_batch_size`, pygmo's `batch_fitness` with a `bfe` where the algorithm accepts one. An interface that changes the algorithm isn't used: SciPy's `vectorized=True` forces deferred updating. Each row of a batch counts as one evaluation.

## 4. Time

4.1. The clock starts before the initial population is created and stops right after the run ends. It includes every evaluation. It stops before any output: formatting, extracting the front, recomputing objectives. Adapter bookkeeping other than counting evaluations and recording the first hit, such as storing every evaluated solution, isn't done while the clock runs.

4.2. Excluded:
- interpreter and JVM startup;
- imports;
- parsing the command line;
- for Java and Julia, JIT compilation. Each solver makes one untimed, unprinted warm-up run before its timed runs: seed 999999, a budget of 50,000 evaluations and the scenario's time cap. Every JIT-compiled adapter uses the same warm-up.

4.3. **One thread.** Every library runs single-threaded: numpy's BLAS with 1 thread, set by assigning the environment variables (not `setdefault`); Julia with `-t 1`; the JVM with the serial garbage collector. **[checked]** The CPU time of the adapter process must stay within 10% of its wall time; more means it used other threads.

4.4. One run at a time, on the pinned P-cores, with nothing else running (see the [README](../../benchmarks/README.md#methodology)).

4.5. **Builds.** Compiled adapters build with optimizations, for the default target, as their users would: no `-march=native` or `target-cpu=native`.

## 5. Seeds and repeated runs

5.1. Each solver runs 10 seeds, 0 to 9, passed to the library's own random number generator.

5.2. **[checked]** The same seed gives the same evaluations, the same best value and the same first hit, where the library supports seeding. A run doesn't depend on the runs before it in the same process: seed 1 gives the same alone as after seed 0. A library that can't be seeded says so on its page.

5.3. A solver whose first 3 seeds all run the full 60 seconds without reaching the target runs no more seeds. Its result shows 3 runs.

## 6. Which methods run

6.1. **Matched mode:** the same algorithm, operators, rates and population size in every library, as the [README](../../benchmarks/README.md#scenarios) lists them.
- **Only the library's own components.** The adapter uses the library's own operators, selection and replacement, bugs included (rule 8.4). It doesn't write a component the library lacks, or a wrapper that changes what one does. It sets only settings the library exposes: rates, population size, η, divisions.
- **A missing component** means the library doesn't run the scenario. Its adapter prints nothing, and its page says what's missing.
- **Smaller differences** are allowed only where the library can't express the exact setting, such as a per-gene rate with the same mean. The library's page lists them.
- **Matched OneMax** is generational without elitism, as DEAP's `eaSimple`. A library that can only run it elitist, as (μ+λ), or with another replacement doesn't run it.
- **Matched multi-objective** runs only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA. NSGA-III runs on the ZDT problems too, in every library that has it. Duplicate elimination is off. SMS-EMOA is steady-state: one child per step; a library whose SMS-EMOA can't do that doesn't run it. A library's other multi-objective algorithms aren't run; its page lists them.

6.2. **Idiomatic mode:** the methods a library's own documentation recommends for each problem type:
- binary;
- permutation;
- continuous and multimodal;
- continuous and unimodal;
- multi-objective.

For each method, the adapter cites where the library recommends it: a page of the docs, an example or the README. The settings are the documented defaults, or the ones documented for that problem type.

**The docs decide, not our tests.** Where a library documents several methods or settings, the choice follows the docs, in this order:
1. a preference they state;
2. their example for that problem type;
3. otherwise, the default;
4. with no preference, no example and no default, the first they list.

The separate test runs are shown on the page, but they never pick a method or a setting.

6.3. **Not allowed in idiomatic mode:**
- tuning settings to the benchmark problems;
- a method the library doesn't present for the problem type;
- a method its own docs say doesn't suit it, such as CMA-ES without restarts on a multimodal function when the library offers restarts.

6.4. At most 3 methods per library and problem type: the library's first recommendations.

6.5. Every method left out that a reader might expect gets a reason on the library's page.

6.6. **The authors' own library.** A library whose authors run the benchmark uses its methods with their defaults, and literature values where there's no default. Its page cites the source of each setting. Settings from its examples or guides aren't used: some were written on these problems. Its methods are chosen as rule 6.2 says. This applies to genoxide, in Rust and in Python.

## 7. Multi-objective runs

7.1. They have no target, and each run uses its whole evaluation budget.

7.2. The front reported is the non-dominated part of the algorithm's final population, of the size the scenario sets. For SPEA2, that's its archive of that size. An unbounded archive of every solution ever evaluated isn't comparable: it would always look better.

7.3. `run.py` computes the objectives of the front's solutions with the reference, keeps the non-dominated ones, and computes their hypervolume with the same exact code for every library. The objectives the adapter printed are only checked against the reference, never used. The reference point is 1.1 times the nadir of the optimal front: (1.1, 1.1) for ZDT, (1.1, 1.1, 1.1) for DTLZ2, (0.55, 0.55, 0.55) for DTLZ1.

## 8. Reporting

8.1. For every solver and scenario the results show:
- the number of runs, and how many reached the target, as "k of n";
- how many runs the time cap stopped before the target and the budget ("capped"): those are limited by speed, not by the search;
- the expected running time (ERT) to the target, in evaluations and in seconds. It's the sum over all runs of the first hit's evaluations (time) in a run that reached the target and of all its evaluations (time) in a run that didn't, divided by the number of runs that reached it. With no run reaching it, the result is "not reached". The charts sort by it;
- for all runs, the best value at the end: median, best and worst.

A run that ends without reaching the target is reported as "not reached", with the value it got to. It's not a crash: it shows how close the method came within the budget.

8.2. A scenario a library can't run is shown as such, with the reason.

8.3. Library versions are pinned and recorded with each result.

8.4. Library bugs aren't worked around, except where the library's page says so and shows both results. A crash isn't convergence: catching one and restarting is a workaround, labelled so. The [notes](notes.md) list every bug found.

## 9. Open documentation

9.1. Every library has a page in [libraries/](libraries/). For each problem type, it gives:
- the methods chosen, with where the library recommends them, and the settings, with their source;
- the alternatives considered: other methods, settings or functions of the library, and why each was left out;
- the separate test runs that checked the adapter before the benchmark: success rate, evaluations and the best value reached;
- anything the library can't do here, and bugs found.

9.2. Nothing about how a library is run is left to its code alone. The adapter points to its page, and the page points to the lines of the adapter.

9.3. **Better ways are welcome.** If you know a better way to solve one of these problems with one of these libraries, open a [benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml): another method, a setting its docs recommend, or a function we missed. Say which library, problem and method, and where the library documents it. It's tested under these rules. If it does better, it replaces the current one, and the page records the change. That applies to genoxide too.
