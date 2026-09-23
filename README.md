# genoxide

**Evolutionary computation for Rust (and Python): fast, correct, reproducible.**

Genetic algorithms, evolution strategies, multi-objective optimization, swarm and local search, all in one library.

> **🚧 Pre-alpha.** There is nothing to install yet. We're designing the API in the open.
> See the [roadmap](ROADMAP.md) and share your ideas in the issues.

## Why genoxide?

Rust has over 200 evolutionary computation crates, but most are small or abandoned. None of them covers what DEAP or pymoo offer in Python. Python has the breadth, but every operator runs in the interpreter. genoxide aims to combine both:

- **Complete:** one library, from a simple GA to NSGA-III, CMA-ES and island models
- **Fast:** native Rust, parallel evaluation, no allocations in the hot loop
- **Correct:** configurations are validated when you build them, and operators are checked against the representation at compile time
- **Reproducible:** the same seed gives the same result, on any number of threads
- **Batteries included:** statistics, hall of fame, checkpointing, cancellation, benchmarks
- **Also for Python:** `pip install genoxide`, planned

## A first look

This is the planned API, and it will change:

```rust
use genoxide::prelude::*;

let result = Ga::builder()
    .genome(Binary::new(100))
    .fitness(|genes: &Bits| genes.count_ones() as f64)
    .maximize()
    .population(200)
    .select(Tournament::new(3))
    .crossover(Uniform::new(0.5))
    .mutate(BitFlip::per_gene(0.01))
    .stop_when(Target(100.0).or(Generations(1_000)))
    .seed(42)
    .run()?;

println!("best: {} after {} generations", result.best_fitness(), result.generations());
```

## Status

| Milestone | Scope | Status |
|---|---|---|
| 0.1 Foundations | Core engine, representations, classic operators, statistics | 🔜 next |
| 0.2 Real-valued & permutations | SBX, polynomial, PMX, OX, 2-opt, local search, memetic | planned |
| 0.3 Evolution strategies & swarm | CMA-ES, DE (JADE, SHADE), PSO, (μ,λ) and (μ+λ)-ES | planned |
| 0.4 Multi-objective | NSGA-II/III, SPEA2, MOEA/D, SMS-EMOA, hypervolume | planned |
| 0.5 Scale | Island model, checkpointing, batch/GPU evaluation | planned |
| 0.6 Python | PyO3 bindings with numpy support | planned |
| 0.7 GP & neuroevolution | Typed tree GP, NEAT | planned |
| 0.8 Frontier | Quality-diversity (MAP-Elites), LLM-guided operators | planned |
| 1.0 | Stable API and published benchmark report | planned |

The details are in [ROADMAP.md](ROADMAP.md).

## Benchmarks

Every release will be benchmarked against DEAP, pymoo, PyGAD and the main Rust libraries. The benchmarks use the same problems, the same fitness functions and the same evaluation budgets, and the results will be published here.

## Contributing

genoxide is at the design stage, which is the best time to shape it. Open an issue for ideas, use cases or API feedback.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in genoxide by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
