# Benchmark rules

Every library is measured under these rules. An adapter that breaks one isn't benchmarked: `run.py check` tests each rule marked **[checked]** before any timed run, and the others are reviewed in the adapter's code.

## 1. The problems

1.1. The fitness functions are defined once, in `run.py`, in Python. That definition is the reference.

1.2. Each adapter implements them in its library's language, as its users would. **[checked]** The adapter evaluates fixed points, including the optimum, and the values must match the reference to 1e-9 relative.

1.3. Each run prints the best solution it found, not only its value. **[checked]** `run.py` evaluates that solution with the reference, and the run is invalid if the two values differ. A multi-objective run prints the solutions of its front, and `run.py` computes their objectives itself.

1.4. The continuous problems are shifted, so an optimum at the origin can't favour operators that drift towards 0.

## 2. The budget

2.1. A run ends at the first of:
- the target, reached by the best solution found;
- the scenario's evaluation budget;
- 60 seconds.

Nothing else ends a run.

2.2. **Methods that stop by themselves keep going.** Some methods end on their own criteria.
- **A limit that's only a budget,** such as a maximum number of generations or iterations, is lifted.
- **A criterion that detects convergence,** such as a tolerance or a number of generations without improvement, ends that attempt, and the method starts again:
  - with the library's restart mechanism for the method, if it has one (IPOP for CMA-ES, for example);
  - otherwise from a new random start, with a seed derived from the run's seed, such as `seed * 1000 + restart`. The adapter keeps the best solution and counts every evaluation.

A user with time left would do the same. A converged method with its criterion turned off would only spend the rest of the budget where it's stuck.

2.3. A library that checks the stop only between generations may go past the target or the budget by at most one generation. **[checked]** Evaluations beyond the budget plus one generation make the run invalid.

2.4. **Only inside the bounds.** A problem is defined on its box of bounds, and every solution a method evaluates must lie inside it. The library's own bound handling is used: clipping, repair, a transformation or a bounded operator. The page says which. **[checked]** Each run of a continuous or multi-objective problem reports `outside`, the number of evaluated solutions outside the bounds, counted by the adapter's own counter, and it must be 0.

## 3. Counting evaluations

3.1. An evaluation is one call of the fitness function on one solution.

3.2. The adapter counts them itself, with a counter around the fitness function, and doesn't use the library's own count. Every call counts: copies, restarts, local search and final polishing. A library that doesn't evaluate a copy again saves that evaluation. That's a real saving and it's reported.

## 4. Time

4.1. The clock starts before the initial population is created and stops when the run ends. It includes every evaluation.

4.2. Excluded:
- interpreter and JVM startup;
- imports;
- parsing the command line;
- for Java and Julia, one untimed warm-up run with another seed and a small budget, so JIT compilation isn't counted. The same warm-up is used for every JIT-compiled library.

4.3. **One thread.** Every library runs single-threaded: numpy's BLAS with 1 thread, Julia with `-t 1`, the JVM with the serial garbage collector. **[checked]** The CPU time of the adapter process must stay within 10% of its wall time; more means it used other threads.

4.4. One run at a time, on the pinned P-cores, with nothing else running (see the [README](../../benchmarks/README.md#methodology)).

## 5. Seeds and repeated runs

5.1. Each solver runs 10 seeds, 0 to 9, passed to the library's own random number generator.

5.2. **[checked]** The same seed gives the same evaluations and the same best value, where the library supports seeding. A library that can't be seeded is listed in the notes.

5.3. A solver whose first 3 seeds all run for the full 60 seconds without reaching the target runs no more seeds. Its result shows 3 runs.

## 6. Which methods run

6.1. **Matched mode:** the same algorithm, operators, rates and population size in every library, as the scenario lists them. A library that can't match the algorithm itself doesn't run the scenario. Smaller differences are listed in the notes. The matched multi-objective scenarios run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA. A library's other multi-objective algorithms aren't run, and its page lists them.

6.2. **Idiomatic mode:** the methods a library's own documentation recommends for the problem type:
- binary;
- permutation;
- continuous and multimodal;
- continuous and unimodal;
- multi-objective.

For each method, the adapter cites where the library recommends it: a page of the docs, an example or the README. The settings are the documented defaults, or the ones documented for that problem type.

**The docs decide, not our tests.** Where a library documents several methods or settings, the choice follows what the docs say, in this order:
1. a preference they state;
2. their example for that problem type;
3. otherwise, the default.

The separate test runs are shown on the page, but they never pick a method or a setting.

6.3. **Not allowed in idiomatic mode:**
- tuning settings to the benchmark problems;
- a method the library doesn't present for the problem type;
- a method its own docs say doesn't suit it, such as CMA-ES without restarts on a multimodal function when the library offers restarts.

6.4. At most 3 methods per library and problem type, the library's first recommendations.

6.5. Every method left out that a reader might expect gets a reason in the notes.

## 7. Multi-objective runs

7.1. They have no target, and each run uses its whole evaluation budget.

7.2. The front reported is the non-dominated part of the algorithm's final population, of the size the scenario sets. For SPEA2, that's its archive of that size. An unbounded archive of every solution ever evaluated isn't comparable: it would always look better.

7.3. `run.py` computes the objectives of the front's solutions and their hypervolume with the same exact code for every library.

## 8. Reporting

8.1. For every solver and scenario the results show:
- the number of runs;
- the share that reached the target;
- for those runs, the median time and evaluations to the target;
- for all runs, the best value at the end: median, best and worst.

A run that ends without reaching the target is reported as "not reached", with the value it got to. It's not a crash. It shows how close the method came within the budget.

8.2. A scenario a library can't run is shown as such, with the reason.

8.3. Library versions are pinned and recorded with each result.

8.4. Library bugs aren't worked around, except where the notes say so and show both results.

## 9. Open documentation

9.1. Every library has a page in [libraries/](libraries/). For each problem type, it gives:
- the methods chosen, with where the library recommends them, and the settings, with where they come from;
- the alternatives considered: other methods, settings or functions of the library, and why each was left out;
- the separate test runs that checked the adapter before the benchmark: success rate, evaluations and the best value reached;
- anything the library can't do here, and bugs found.

9.2. Nothing about how a library is run is left to its code alone. The adapter points to its page, and the page points to the lines of the adapter.

9.3. **Better ways are welcome.** If you know a better way to solve one of these problems with one of these libraries, open a [benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml). Examples: another method, a setting its docs recommend, or a function we missed. Say which library, problem and method, and where the library documents it. It's tested under these rules, and if it does better, it replaces the current one and the page records the change. That applies to genoxide too.
