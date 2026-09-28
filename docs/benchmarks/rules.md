# Benchmark rules

Every library is measured under these rules. An adapter that breaks one isn't benchmarked. `run.py check` tests each rule marked **[checked]** before any timed run; the others are reviewed in the adapter's code. Every timed run is checked again with the same run checks. A run that fails is invalid: it's left out of every table and chart, and the results list it with the reason.

The suite is matched: three problems, one method each, written down in [section 6](#6-the-methods). Every library runs a problem only with its own implementation of that problem's method, set to the definition. So the results compare implementations of the same algorithm, not each library's pick of a method.

## 1. The problems

1.1. The fitness functions are defined once, in Python, in `benchmarks/problems.py`. That definition is the reference.

1.2. Each adapter implements them in its library's language, as its users would. **[checked]** The adapter evaluates fixed points, including the optimum, and the values must match the reference to 1e-9 relative.

1.3. Each run prints the best solution it found, not only its value. **[checked]** `run.py` evaluates that solution with the reference; the run is invalid if the two values differ. It also recomputes whether the solution reaches the target.

1.4. Rastrigin is shifted, so an optimum at the origin can't favour operators that drift towards 0. The optimum is at `s_i = 0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1)`, for i from 0, where `upper` is the box's upper bound, 5.12. Every adapter computes it in this order, in double precision. Rosenbrock isn't shifted: its optimum, all ones, is already away from the origin.

| Scenario | Problem | Genome | Method | Target | Budget | Time cap |
|---|---|---|---|---|---|---|
| `onemax-1000-matched` | OneMax, number of ones (maximized) | 1000 bits | the GA of [6.2](#6-the-methods) | all ones | 2,000,000 | 60 s |
| `rastrigin-30-matched` | Rastrigin, shifted (minimized) | 30 reals in [−5.12, 5.12] | DE/rand/1/bin of [6.3](#6-the-methods) | none: a fixed budget | 300,000 | 60 s |
| `rosenbrock-10-matched` | Rosenbrock (minimized) | 10 reals in [−5, 10] | CMA-ES of [6.4](#6-the-methods) | ≤ 0.01 | 500,000 | 60 s |

## 2. The budget

2.1. A run ends at the first of:
- the target, reached by the best solution found, in a scenario with one;
- the scenario's evaluation budget;
- the scenario's time cap: 60 seconds.

Nothing else ends a run, except an error of the library itself (rule 8.4) or a convergence criterion it can't turn off (rule 2.2).

2.2. **No convergence criterion.** The three methods have none: a run goes on to the target, the budget or the time cap, even where it has converged.
- **A limit that's only a budget,** such as a maximum number of generations or iterations, is lifted.
- **A library's own convergence criterion** is turned off with its documented settings. One that can't be turned off is set to its tightest value, and the library's page says so. If it still fires, the run ends there, as not reached, and its output names the criterion in `ended_by`; the page and the [notes](notes.md) list it as a difference. The adapter doesn't start the method again: a restart isn't part of the method. Such criteria fire only where the method can no longer change its search (every value of a DE population equal; a CMA-ES step too small to move the mean), where a run without them would stay until its budget. A library that restarts by itself on such a criterion is left out of the problem (rule 6.1).
- **A stalled attempt.** An attempt that makes no new fitness evaluation for 10 consecutive generations has stalled: the adapter ends it and starts a new attempt from a new random start, with the next restart seed, keeping the best solution, the budget and the time cap. Only the GA can stall, when every child is an unchanged copy that isn't evaluated again; DE and CMA-ES evaluate every trial and every sample.
- **No restarts.** No method restarts on its own: no IPOP or BIPOP for CMA-ES, no restarts for DE.

Seeds: attempt 0 uses the run's seed. Restart r, from 1, uses `(seed + 1) * 1_000_000 + r`, so no two runs share a seed.

2.3. A library that checks the stop only between generations may go past the target or the budget by at most one generation. **[checked]** Evaluations beyond the budget plus one generation make the run invalid. Every run reports `last_generation`, the evaluations its adapter counted since the start of its last generation (a restart's initial population is a generation), and a generation is the larger of that and the run's average.

2.4. **Only inside the bounds.** Every solution a method evaluates must lie inside the problem's box. The library's own bound handling is used: clipping, repair, a transformation or resampling. The method's definition names a reference; another bound handling is a difference the library's page lists. **[checked]** Each run of a continuous problem reports `outside`, the number of evaluated solutions outside the bounds, from the adapter's own counter. It must be 0.

## 3. Counting evaluations

3.1. An evaluation is one call of the fitness function on one solution.

3.2. The adapter counts them itself, with a counter around the fitness function, not with the library's own count. Every call counts, restarts included. A library that doesn't evaluate a copy again saves that evaluation. That's a real saving, and it's reported.

3.3. **The first hit.** Every run of a scenario with a target prints `"first_hit": {"evaluations": E, "time_s": T}`; a run of Rastrigin, which has none, prints `"target": null`, `"success": false` and `"first_hit": null`. E is the number of the first evaluation whose true value reaches the target, counting that evaluation. T is the run's clock at that evaluation. A run that never reaches the target prints `"first_hit": null`. The adapter's counter records it at each evaluation, whatever the library's stop granularity. The run still stops as rules 2.1 and 2.3 say. A first hit later than the scenario's time cap counts as not reached. **[checked]** `first_hit` is present when the solution reaches the target and null otherwise; E is at most the run's evaluations, and T at most its time.

3.4. **Batch evaluation.** A Python library with a documented batch or vectorized evaluation interface evaluates a generation at once with numpy, as its users would, if the interface doesn't change the algorithm. Each row of a batch counts as one evaluation.

## 4. Time

4.1. The clock starts before the initial population is created and stops right after the run ends. It includes every evaluation. It stops before any output, such as formatting. Adapter bookkeeping other than counting evaluations and recording the first hit, such as storing every evaluated solution, isn't done while the clock runs.

4.2. Excluded:
- interpreter and JVM startup;
- imports;
- parsing the command line;
- for Java and Julia, JIT compilation. Each method makes one untimed, unprinted warm-up run before its timed runs: seed 999999, a budget of 50,000 evaluations and the scenario's time cap. Every JIT-compiled adapter uses the same warm-up.

4.3. **One thread.** Every library runs single-threaded: numpy's BLAS with 1 thread, set by assigning the environment variables (not `setdefault`); Julia with `-t 1`; the JVM with the serial garbage collector. **[checked]** The CPU time of the adapter process must stay within 10% of its wall time; more means it used other threads.

4.4. One run at a time, on the pinned P-cores, with nothing else running (see the [README](../../benchmarks/README.md#methodology)).

4.5. **Builds.** Compiled adapters build with optimizations, for the default target, as their users would: no `-march=native` or `target-cpu=native`.

## 5. Seeds and repeated runs

5.1. Each method runs 10 seeds, 0 to 9, passed to the library's own random number generator.

5.2. **[checked]** The same seed gives the same evaluations, the same best value and the same first hit, where the library supports seeding. A run doesn't depend on the runs before it in the same process: seed 1 gives the same alone as after seed 0. A library that can't be seeded says so on its page.

5.3. A method whose first 3 seeds all run to the time cap without reaching the target (or, without a target, the budget) runs no more seeds. Its result shows 3 runs.

## 6. The methods

6.1. **One method per problem, the library's own implementation.**
- **Only the library's own implementation of the method,** with its own operators, selection and replacement, bugs included (rule 8.4). The adapter doesn't write a component the library lacks, or a wrapper that changes what one does. It sets only what the library exposes, the way its users would: documented options, such as the population size, the rates and the step size.
- **Set to the definition below.** Each setting the adapter passes, and each default it relies on, is on the library's page, mapped item by item to the definition.
- **Differences that don't change the algorithm are allowed** and listed on the library's page and in the [notes](notes.md): the bound handling (it acts only on genes that leave the box), how a uniform initial population or mean is drawn, the exact formula of a CMA-ES learning rate among Hansen's published defaults, a rate the library can only express another way with the same mean (a per-gene mutation rate for a per-child one, a probability per child for one per pair), an evaluation the library spends on something the definition doesn't (it's counted), the order in which individuals are visited, a detail of how DE draws its indices that lets a few percent of trials use the target itself.
- **Differences that change the algorithm leave the library out of that problem:** other variation operators, other parameter values (a dithered, jittered or adapted F, another population size), another selection or replacement (a steady-state DE instead of a generational one, an elitist GA), restarts, negative (active) CMA-ES weights that can't be turned off, a convergence criterion that ends every run early. The adapter prints nothing for the problem, and the library's page says why.
- **A library that has none of the three methods** isn't in the suite.

6.2. **OneMax 1000: the GA.** DEAP's `eaSimple` with its OneMax example's settings is the reference.
- **Population:** 300 individuals of 1000 bits, each bit uniform at random.
- **Selection:** 300 tournaments of 3, with replacement: 3 individuals drawn uniformly, the best wins.
- **Variation** (`varAnd`): the selected are copied. Each consecutive pair (0, 1), (2, 3), ... is crossed with probability 0.5 by two-point crossover: two cut points drawn uniformly, the segment between them swapped. Then each child is mutated with probability 0.2 by bit flip, each bit flipping with probability 1/1000.
- **Replacement:** generational: the children replace the whole population. No elitism.
- **Evaluation:** a child neither crossed nor mutated keeps its parent's fitness and isn't evaluated again. A library that evaluates it again spends those evaluations; allowed, listed.
- **No convergence criterion;** a stalled attempt starts again (rule 2.2).

6.3. **Rastrigin 30: DE/rand/1/bin** (Storn and Price, 1997), **with a fixed budget and no target.** With these settings DE/rand/1/bin doesn't reach an error of 0.01 on the shifted Rastrigin function in 30 dimensions in any library: CR 0.9 changes almost every gene of a trial at once, which suits rotated functions, and the population settles in a local minimum. A target would only show crosses. So every run uses 300,000 evaluations, and the problem is measured by:
- **the time for the budget,** the median over the runs that used the whole budget: the cost of the implementation, for the same number of evaluations;
- **the error at the end,** the best value found minus the optimum's 0, median, best and worst: every library runs the same algorithm, so the errors should agree within the spread of the seeds. A library far from the others points to a difference from the definition, or a bug.

A run that the library ends before the budget (rule 2.2, `ended_by`) reports its evaluations, time and error, and is marked as ended early: its error counts, its time doesn't count towards the time for the budget. So does a run the time cap stops.

The method:
- **Population:** NP = 100 individuals, uniform at random in the box.
- **Mutation:** for each target x_i, three indices r1, r2, r3 drawn uniformly, distinct and different from i; the mutant is v = x_r1 + F · (x_r2 − x_r3), with F = 0.5 for every trial: no dither, no jitter, no adaptation.
- **Crossover:** binomial, CR = 0.9. An index j_rand is drawn uniformly; gene j of the trial u is v_j if a uniform number is below CR or j = j_rand, else x_ij.
- **Bounds:** every evaluated trial lies in the box (rule 2.4). Reference: a trial gene outside the box is drawn again uniformly in the box, as SciPy and pagmo do.
- **Selection:** u replaces x_i when f(u) ≤ f(x_i), one to one.
- **Generational:** all trials of a generation are built from the population of the previous generation; the replacements take effect for the next generation.
- **No archive, no adaptation, no restarts, no convergence criterion.**
- **Stop:** the fixed budget of 300,000 evaluations, or the time cap; no target.

6.4. **Rosenbrock 10: CMA-ES** with Hansen's defaults. pycma, Hansen's own implementation, is the reference.
- **Samples:** λ = 4 + ⌊3 ln n⌋ = 10 per generation, from m + σ · N(0, C).
- **Recombination:** the best μ = ⌊λ / 2⌋ = 5, with weights w_i ∝ ln((λ + 1) / 2) − ln i, i = 1, ..., μ, positive and summing to 1 (μ_eff ≈ 3.17). No negative weights: no active CMA.
- **Adaptation:** cumulative step-size adaptation, rank-one and rank-μ updates of C, with the h_σ stall of the rank-one update, and the library's default learning rates and damping. Hansen's 2016 tutorial gives c_σ = (μ_eff + 2) / (n + μ_eff + 5), d_σ = 1 + 2 max(0, √((μ_eff − 1) / (n + 1)) − 1) + c_σ, c_c = (4 + μ_eff / n) / (n + 4 + 2 μ_eff / n), c_1 = 2 / ((n + 1.3)² + μ_eff), c_μ = min(1 − c_1, 2 (μ_eff − 2 + 1 / μ_eff) / ((n + 2)² + μ_eff)); a library whose defaults compute one of them with another of Hansen's published formulas runs with its own, and its page lists the values for n = 10.
- **Start:** the mean drawn uniformly at random in the box; σ₀ = 0.3 × (upper − lower) = 4.5; C₀ = I.
- **Bounds:** every evaluated solution lies in the box (rule 2.4). Reference: pycma's default with bounds, `BoundTransform`.
- **No restarts** (no IPOP, no BIPOP) **and no convergence criterion:** a run that converges samples on around its point until the budget or the time cap.

6.5. **Every difference is written down.** Each library's page gives, per problem, its configuration against the definition, every difference and why it doesn't change the algorithm, or why the library doesn't run the problem. The [notes](notes.md) index them.

6.6. **The authors' own library.** genoxide, in Rust and in Python, runs the same definitions, set the same way, with its builder's documented settings; its page lists its differences like any other library's.

## 7. Multi-objective runs

Removed for now. The suite has single-objective problems only; multi-objective ones come back after these, with their rules. The number is kept so the other rules keep theirs.

## 8. Reporting

8.1. For every library and scenario the results show:
- the number of runs, and how many reached the target, as "k of n";
- how many runs the time cap stopped before the target and the budget ("capped"), and the median share of the budget they used: those are limited by speed, not by the search;
- in a scenario with a target, the expected running time (ERT) to the target, in evaluations and in seconds. It's the sum over all runs of the first hit's evaluations (time) in a run that reached the target and of all its evaluations (time) in a run that didn't, divided by the number of runs that reached it. It needs at least 3 runs that reached the target. With fewer, the result shows how many reached it, such as "2/10 reached". The charts sort by it;
- for all runs, the distance of the best value to the optimum at the end: median, best and worst. Without a target (Rastrigin), it's the error at the end, and the median time of the runs that used the whole budget replaces the ERT; runs the library ended early are counted and marked.

A run that ends without reaching the target is reported as "not reached", with the value it got to. It's not a crash: it shows how close the method came within the budget.

The charts show the capped runs apart from the runs that ended at the target or the budget, with the share of the budget they used. A problem the published run has no runs of is shown as awaiting the next run.

8.2. A problem a library can't run is shown as such, with the reason.

8.3. Library versions are pinned and recorded with each result. The runs of the published results are committed beside them, in `results.json.xz`, so their tables and charts can be checked and drawn again.

8.4. Library bugs aren't worked around, except where the library's page says so and shows both results. A crash isn't convergence: with no restarts in the methods, a run the library ends with an error ends there, as not reached, and its output says so in `ended_by`. The [notes](notes.md) list every bug found.

8.5. Removed: the suite has no overall score. Each problem is shown on its own, a bar per library.

## 9. Open documentation

9.1. Every library has a page in [libraries/](libraries/). For each problem, it gives:
- the library's implementation of the method, and its configuration mapped to the definition, with the source of each setting;
- every difference from the definition, and why it doesn't change the algorithm;
- the separate test runs that checked the adapter before the benchmark: success rate, evaluations and the best value reached;
- the problems it can't run, and bugs found.

9.2. Nothing about how a library is run is left to its code alone. The adapter points to its page, and the page points to the lines of the adapter.

9.3. **Corrections are welcome.** If a library can be set closer to a definition than its page says, or an adapter doesn't do what the definition says, open a [benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml): say which library, problem and setting, and where the library documents it. It's tested under these rules, and the page records the change. That applies to genoxide too.

## 10. Instruction counts: genoxide's versions

CPU instructions are counted for genoxide only, to compare its versions on the same runs. The other libraries aren't counted: their times and evaluations are in the charts.

10.1. **The same runs.** In every scenario, genoxide's method makes one run with seed 0, the first seed of every timed run, to its target or its evaluation budget. There's no time cap: nothing else ends the run. The adapter is the same source for every version (`benchmarks/adapters/genoxide`), built against the version: a release from crates.io, or the repository's genoxide for an unreleased one. A version whose API the adapter doesn't compile against isn't measured.

10.2. **Counted by Callgrind.** Each method runs alone in its process, under Valgrind's Callgrind. A process that runs no method, the adapter's startup, is counted too and subtracted. What's left is the run: building the algorithm, every evaluation, and printing the run's result line. The count is exact: it doesn't depend on the machine's load, and a seed gives the same run on every platform. It changes with genoxide, the adapter, the Rust compiler and Valgrind, which the history records with each version.

10.3. **Checked.** Each run is made once without Callgrind too, and must be the same under it: the same evaluations and the same result. It's checked like a timed run (rules 1.3, 2.3, 2.4 and 3.3); a run that fails is recorded with its failures.

10.4. **Reported.** `docs/benchmarks/genoxide-versions.json` keeps a row per version: the date it was measured and released, the machine, the compiler and Valgrind, and per scenario and method the instructions, the evaluations and whether the run reached the target (its best value). The chart `genoxide_versions.svg` and its numbers in `charts.json` show each method's instructions across the versions, hollow where the run didn't reach the target. Fewer instructions to the same target mean less work: a cheaper evaluation, fewer evaluations, or both, which the evaluations tell apart.
