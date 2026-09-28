# genoxide for Python (Rust via Python)

genoxide's Python package, from this repository's [python/](../../../python/) folder: genoxide's algorithms, in Rust, calling fitness functions in Python and numpy. Its docs are [python/README.md](../../../python/README.md) and the docstrings of [python/genoxide/\_\_init\_\_.py](../../../python/genoxide/__init__.py).

The package runs the three problems of the matched suite with the same Rust implementations as the Rust library ([genoxide.md](genoxide.md)): `gx.Ga` on OneMax 1000, `gx.De` on Rastrigin 30 and `gx.Cmaes` on Rosenbrock 10, set to the definitions of [rule 6](../rules.md#6-the-methods). A seed gives the same run as in Rust; what differs is the cost of calling Python fitness functions. [Rule 6.6](../rules.md#6-the-methods) applies, as for the Rust library.

Adapter: [benchmarks/adapters/genoxide_python/](../../../benchmarks/adapters/genoxide_python/).
Found a setting that brings the package closer to a definition, or a difference this page misses? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs the package

- **Fitness functions:** numpy with `batch=True`, a generation per call, for every problem ([bench.py, lines 63-136](../../../benchmarks/adapters/genoxide_python/bench.py#L63-L136)): python/README.md's way to a fast fitness function ("Vectorized numpy [...] pays its cost per call once per generation, not once per genome", and its Rastrigin example), and what [rule 3.4](../rules.md#3-counting-evaluations) asks of a Python library with a documented batch interface, as the PyGAD and pymoo adapters do. OneMax is README's `bits.sum()` over the rows, `x.sum(axis=1)`. A batch doesn't change the algorithm ("The same seed repeats a run exactly: one genome at a time, in batches or in parallel"). `bench.py --self-check` compares every function with problems.py.
- **Evaluations:** each function counts the rows of its batch, and records the first hit: the first row, in the batch's order, that reaches the target, numbered from the start of the run (the package asks for the same genomes in the same order either way). Any difference from `result.evaluations` is printed to stderr ([`solve`, lines 222-287](../../../benchmarks/adapters/genoxide_python/bench.py#L222-L287)).
- **Stop:** `run(..., target=..., evaluations=..., time=...)`, after every generation: the time cap is the package's `time` stop, the time left of the run. No method has a convergence criterion or restarts.
- **Stalls (rule 2.2):** a GA child identical to a parent isn't evaluated, so a GA can run generations with nothing to evaluate, and a batch function isn't called for them. The GA's `on_generation` callback, the package's only hook after every generation, counts them: after 10 in a row it returns False, which ends the attempt, and the adapter starts it again with the seeds of rule 2.2 and prints `restarts`. DE and CMA-ES evaluate every trial and sample, and run without a callback. The callback costs the GA about 17% of its time on OneMax: the package builds the `Progress` it gets, with the population as a 300 × 1000 array, after every generation, though the adapter reads none of it.
- **Time:** from before the algorithm object is created to the return of `run`, which creates the initial population.
- **One thread:** `parallel` off; numpy's BLAS set to 1 thread before import.
- **Seeds:** the algorithm's `seed`; a seed repeats a run exactly.
- **Separate tests:** 2026-09-28, the package 0.9.1 built from this repository, seeds 0 to 2, the scenario's budget and cap. The runs were the Rust adapter's, evaluation for evaluation; the counts equalled `result.evaluations`, and `outside` was 0. 2026-09-29, the package 0.9.2 from this repository, after OneMax moved to a batch function and the GA's time cap to `time`: seeds 0 to 2 of every problem gave the same generations, evaluations, best values and first hits as before, and `run.py check` passed.

## OneMax 1000: the GA

**Configuration** ([`onemax_ga`, lines 146-161](../../../benchmarks/adapters/genoxide_python/bench.py#L146-L161)), against [rule 6.2](../rules.md#6-the-methods): `gx.Ga(gx.Binary(1000), population_size=300, select=gx.Tournament(3), crossover=gx.PointCrossover(2), crossover_rate=0.5, mutation=gx.BitFlip(rate=1/1000), mutation_rate=0.2, scheme=gx.Generational(elitism=0))`, the Rust library's settings ([genoxide.md](genoxide.md#onemax-1000-the-ga)).

**Differences:** as in Rust, a child identical to its parent isn't evaluated again.

**Separate tests:** 3 of 3 reached the target, first hits at 52,465 to 58,170 evaluations (median 52,659). With a batch function, about 0.04 s each; with the function per genome the adapter used before 2026-09-29, and the time cap in the callback, about 0.08 s, for the same runs.

## Rastrigin 30: DE/rand/1/bin

**Configuration** ([`rastrigin_de`, lines 164-178](../../../benchmarks/adapters/genoxide_python/bench.py#L164-L178)), against [rule 6.3](../rules.md#6-the-methods): `gx.De(gx.Real((-5.12, 5.12), length=30), population_size=100, strategy="rand1", control={"f": 0.5, "cr": 0.9}, restarts="never")`, the Rust library's `De` ([genoxide.md](genoxide.md#rastrigin-30-derand1bin)).

**Difference:** as in Rust, a trial gene outside the box is set halfway between the target's gene and the bound, not drawn again uniformly.

**No target:** every run uses the fixed budget of 300,000 evaluations (rule 6.3); the adapter passes no target to `run` and prints `"target": null`.

**Separate tests:** seeds 0 and 1 took 0.19 and 0.18 s for the 300,000 evaluations and ended at errors of 154.7 and 123.1, the Rust adapter's runs.

## Rosenbrock 10: CMA-ES

**Configuration** ([`rosenbrock_cmaes`, lines 181-188](../../../benchmarks/adapters/genoxide_python/bench.py#L181-L188)), against [rule 6.4](../rules.md#6-the-methods): `gx.Cmaes(gx.Real((-5.0, 10.0), length=10), restarts="never")` with its defaults: 10 samples, the best 5 with positive log weights, Hansen's 2016 learning rates, a uniform mean and a step of 0.3 of the range ([genoxide.md](genoxide.md#rosenbrock-10-cma-es)).

**Differences:** as in Rust: a sample outside the box is drawn again, up to 100 times, then clipped; the search works in coordinates scaled to 0..=1, the same scaling for every gene here.

**Separate tests:** 3 of 3 reached the target, first hits at 5,837 to 7,451 evaluations (median 5,844), in about 8 ms each.

## Can't run

Nothing: the package has all three methods.

## Bugs found

None.
