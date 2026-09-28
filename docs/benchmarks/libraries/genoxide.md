# genoxide (Rust)

genoxide is this repository's library: genetic algorithms, evolution strategies, CMA-ES, differential evolution, particle swarms, local search and multi-objective algorithms in Rust. Its docs are the [README](../../../README.md), the guide [AGENTS.md](../../../AGENTS.md), the [examples](../../../examples/) and the API docs ([docs.rs/genoxide](https://docs.rs/genoxide), from the rustdoc in [src/](../../../src/)).

genoxide runs the three problems of the matched suite, each with its own implementation of the problem's method ([rule 6](../rules.md#6-the-methods)): `Ga` on OneMax 1000, `De` on Rastrigin 30 and `Cmaes` on Rosenbrock 10. Its authors run the benchmark, so [rule 6.6](../rules.md#6-the-methods) applies: the same definitions as every other library, set with the builders' documented settings, and its differences listed like any other library's.

Adapter: [benchmarks/adapters/genoxide/](../../../benchmarks/adapters/genoxide/).
Found a setting that brings genoxide closer to a definition, or a difference this page misses? [Open a benchmark issue](https://github.com/tachsin/genoxide/issues/new?template=benchmark.yml).

## How the adapter runs genoxide

- **Fitness functions:** in Rust, a closure per genome, as AGENTS.md's templates write them ([main.rs, lines 52-90](../../../benchmarks/adapters/genoxide/src/main.rs#L52-L90)), with the shift of rule 1.4 computed once, before any run.
- **Evaluations:** a counter around the fitness function counts every call and records the first hit ([`solve`, lines 196-318](../../../benchmarks/adapters/genoxide/src/main.rs#L196-L318)). Any difference from `Outcome::evaluations()` is printed to stderr.
- **Stop:** `Stop::target(..).or(Stop::evaluations(..))`, after every generation, and an abort flag for the time cap. No method has a convergence criterion or restarts.
- **Stalls (rule 2.2):** a GA child identical to a parent inherits its fitness without an evaluation (AGENTS.md, [Fitness functions](../../../AGENTS.md#fitness-functions)), so a converged GA can run generations with nothing to evaluate. After 10 in a row, a `Stop::custom` condition ends the attempt ([`stalled`, lines 180-194](../../../benchmarks/adapters/genoxide/src/main.rs#L180-L194)), and the adapter starts it again with the seeds of rule 2.2 and prints `restarts`. DE and CMA-ES evaluate every trial and sample, so they never stall.
- **Time:** from before the algorithm is built (it creates the initial population) to the end of the run.
- **One thread:** built without the default `parallel` feature, so rayon isn't compiled in. With it, the runs are the same and their instructions within 0.5%: the engines evaluate one genome after the other unless `.parallel(true)`, which the adapter doesn't call.
- **Adapter overhead,** measured with Callgrind on seed 0 against the same runs without the counter, the checks and the callbacks: 0.4% of the instructions on OneMax, 3% on Rastrigin (mostly the bounds check of rule 2.4, over 30 genes per evaluation) and 1% on Rosenbrock. The rest is genoxide and the fitness function: about half of Rastrigin is the platform's `cos`.
- **Seeds:** `.seed(...)`; a seed repeats a run exactly.
- **Instruction counts (rule 10):** with `GENOXIDE_BENCH_SOLVER` set, which only `run.py versions` does, the adapter runs only the method it names, or none ([`selected`, lines 169-175](../../../benchmarks/adapters/genoxide/src/main.rs#L169-L175)). The timed runs don't set it.
- **Separate tests:** 2026-09-28, genoxide 0.9.1 from this repository, seeds 0 to 4, the scenario's budget and cap. The counts equalled `Outcome::evaluations()`, and `outside` was 0, in every run. 2026-09-29, genoxide 0.9.2 from this repository, after the bounds check changed: seeds 0 to 2 of every problem gave the same evaluations, best values and first hits as before, and `run.py check` passed.

## OneMax 1000: the GA

**Configuration** ([`run_onemax`, lines 333-352](../../../benchmarks/adapters/genoxide/src/main.rs#L333-L352)), against [rule 6.2](../rules.md#6-the-methods):

| Definition | genoxide |
|---|---|
| 300 individuals, uniform random bits | `.population_size(300)` on `Binary::new(1000)` |
| tournament of 3, with replacement | `Tournament::new(3)` |
| two-point crossover of each consecutive pair, probability 0.5 | `PointCrossover::two_point()`, `.crossover_rate(0.5)` |
| each child mutated with probability 0.2, each bit flipping at 1/1000 | `.mutation_rate(0.2)`, `BitFlip::per_gene(1.0 / 1000.0)` |
| generational, no elitism | `.scheme(Scheme::Generational { elitism: 0 })` |

**Differences:** a child identical to its parent isn't evaluated again, also one that a crossover or a mutation turned back into it; DEAP evaluates every crossed or mutated child. That saves evaluations, and it's counted as it happens (rule 3.2). With these settings the population converges to near-copies, so the saving is large: about 45% of the children DEAP evaluates are copies of a parent, and without them DEAP would need about 56,000 evaluations, like genoxide. The search is the same: with seeds 0 to 2, genoxide reached the target after 517 to 610 generations, DEAP after 566 to 587.

**Separate tests:** 5 of 5 reached the target, first hits at 52,465 to 58,170 evaluations (median 53,597), in about 14 ms each.

## Rastrigin 30: DE/rand/1/bin

**Configuration** ([`run_rastrigin`, lines 356-376](../../../benchmarks/adapters/genoxide/src/main.rs#L356-L376)), against [rule 6.3](../rules.md#6-the-methods):

| Definition | genoxide |
|---|---|
| NP = 100, uniform in the box | `.population_size(100)` on `Real::uniform(30, -5.12..=5.12)` |
| v = x_r1 + F (x_r2 − x_r3), r1, r2, r3 distinct and ≠ i | `.strategy(de::Strategy::Rand1)` |
| F = 0.5 and CR = 0.9, fixed | `.control(de::Control::Fixed { f: 0.5, cr: 0.9 })` (the default, set explicitly) |
| binomial crossover with a forced gene | `De`'s crossover: each gene from the mutant with probability CR, one gene drawn uniformly always |
| u replaces x_i when f(u) ≤ f(x_i) | `De` (its rustdoc: "The trial replaces `x` if it's not worse") |
| generational | `De` builds the trials of a generation from the population, then replaces |
| no restarts | `.restarts(de::Restarts::Never)` (the builder's default is `OnStagnation`, genoxide's own addition) |

**Difference:** the bounds. A trial gene outside the box is set halfway between the target's gene and the bound ("bounce-back", `De`'s rustdoc), not drawn again uniformly. It acts only on genes that leave the box.

**No target:** every run uses the fixed budget of 300,000 evaluations (rule 6.3): `solve` gets no target, stops only at the budget or the time cap, and prints `"target": null`.

**Separate tests:** seeds 0 and 1 took 0.14 and 0.15 s for the 300,000 evaluations, and ended at errors of 154.7 and 123.1. With the earlier budget of 2,000,000, five runs ended at 3.98 to 9.95 (median 5.97): with CR = 0.9, DE/rand/1/bin changes almost every gene of a trial at once, which suits rotated functions, not a separable one like Rastrigin, and the population settles in a local minimum.

## Rosenbrock 10: CMA-ES

**Configuration** ([`run_rosenbrock`, lines 378-402](../../../benchmarks/adapters/genoxide/src/main.rs#L378-L402)), against [rule 6.4](../rules.md#6-the-methods): `Cmaes::builder(Real::uniform(10, -5.0..=10.0))` with its defaults and `.restarts(cmaes::Restarts::Never)` (the default, set explicitly).

| Definition | genoxide (`Cmaes`, [src/algorithm/cmaes.rs](../../../src/algorithm/cmaes.rs)) |
|---|---|
| λ = 4 + ⌊3 ln n⌋ = 10 | `Cmaes::default_population_size(10)` = 10 |
| μ = 5, weights ∝ ln((λ + 1) / 2) − ln i, positive | the same; no negative weights |
| Hansen's 2016 learning rates | the tutorial's formulas: c_σ = 0.2844, d_σ = 1.2844, c_c = 0.2950, c_1 = 0.01528, c_μ = 0.02015 for n = 10 (μ_eff = 3.167) |
| mean uniform in the box, σ₀ = 0.3 × range, C₀ = I | a uniform mean, `initial_step` 0.3 of each range (the default) |
| no restarts, no convergence criterion | `Restarts::Never`: "a converged run goes on sampling around the same point" |

**Differences:**
- **Bounds:** a sample outside the box is drawn again, up to 100 times, and then clipped; the distribution learns from the samples as evaluated. pycma's `BoundTransform` evaluates a transformed point and learns from the untransformed one. It acts only on samples that leave the box.
- **Coordinates:** `Cmaes` searches in coordinates scaled to 0..=1 per gene. Every gene of Rosenbrock has the same range, so this is a uniform scaling, to which CMA-ES is invariant.
- **Eigendecomposition:** every max(1, ⌊1 / (10 n (c_1 + c_μ))⌋) generations, as Hansen's tutorial suggests: every generation here.

**Separate tests:** 5 of 5 reached the target, first hits at 4,877 to 7,451 evaluations (median 5,837), in about 3 ms each.

## Can't run

Nothing: genoxide has all three methods.

## Bugs found

None.
