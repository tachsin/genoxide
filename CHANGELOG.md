# Changelog

All notable changes to genoxide are documented in this file, generated from the pull request titles by [release-plz](https://release-plz.dev/).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and genoxide adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0](https://github.com/tachsin/genoxide/compare/v0.5.2...v0.6.0) - 2026-09-25

### <!-- 0 -->Added

- [**breaking**] eliminate duplicate children in NSGA-II, NSGA-III, SPEA2 and SMS-EMOA ([#109](https://github.com/tachsin/genoxide/pull/109))
- [**breaking**] restart differential evolution on stagnation, and default to SHADE with a small population ([#113](https://github.com/tachsin/genoxide/pull/113))
- add a Python package with numpy genomes, batch and parallel fitness functions ([#105](https://github.com/tachsin/genoxide/pull/105))

### <!-- 1 -->Fixed

- [**breaking**] shuffle the picks of stochastic universal sampling ([#103](https://github.com/tachsin/genoxide/pull/103))

### <!-- 4 -->Documentation

- benchmark every minor release ([#110](https://github.com/tachsin/genoxide/pull/110))
- benchmark 0.6 and its Python package, and mark 0.6 as released ([#115](https://github.com/tachsin/genoxide/pull/115))

## [0.5.2](https://github.com/tachsin/genoxide/compare/v0.5.1...v0.5.2) - 2026-09-25

### <!-- 4 -->Documentation

- benchmark 16 libraries on 14 scenarios, with the methodology and vertical charts ([#106](https://github.com/tachsin/genoxide/pull/106))

## [0.5.1](https://github.com/tachsin/genoxide/compare/v0.5.0...v0.5.1) - 2026-09-24

### <!-- 1 -->Fixed

- handle infinite scores and huge bounds in selection and SBX, and other edge cases ([#98](https://github.com/tachsin/genoxide/pull/98))
- handle infinite objective values in SMS-EMOA, SPEA2 and hypervolume contributions, and duplicate MOEA/D weights ([#99](https://github.com/tachsin/genoxide/pull/99))
- make DE trials change a gene that can change, and bound lambda ([#101](https://github.com/tachsin/genoxide/pull/101))
- clearer errors from the genoxide program and checkpoints ([#102](https://github.com/tachsin/genoxide/pull/102))
- show observers the individuals migrants replace, and fix engine edge cases ([#100](https://github.com/tachsin/genoxide/pull/100))

## [0.5.0](https://github.com/tachsin/genoxide/compare/v0.4.0...v0.5.0) - 2026-09-24

### <!-- 0 -->Added

- add the island model with ring, fully connected and random migration ([#82](https://github.com/tachsin/genoxide/pull/82))
- add batch evaluation of a whole generation in one call ([#84](https://github.com/tachsin/genoxide/pull/84))
- add progress reporting and a tracing feature ([#86](https://github.com/tachsin/genoxide/pull/86))
- add checkpoints to resume a run exactly, behind a serde feature ([#88](https://github.com/tachsin/genoxide/pull/88))
- add asynchronous evaluation with a steady-state GA ([#90](https://github.com/tachsin/genoxide/pull/90))
- add the genoxide program for runs described in TOML or JSON files ([#92](https://github.com/tachsin/genoxide/pull/92))

### <!-- 4 -->Documentation

- add a GPU example of batch evaluation with wgpu ([#93](https://github.com/tachsin/genoxide/pull/93))

## [0.4.0](https://github.com/tachsin/genoxide/compare/v0.3.0...v0.4.0) - 2026-09-24

### <!-- 0 -->Added

- add multi-objective scores, constrained dominance and non-dominated sorting ([#58](https://github.com/tachsin/genoxide/pull/58))
- add the multi-objective engine and NSGA-II ([#60](https://github.com/tachsin/genoxide/pull/60))
- add multi-objective indicators: hypervolume, IGD, IGD+, GD and spread ([#62](https://github.com/tachsin/genoxide/pull/62))
- add a Pareto archive of non-dominated solutions ([#64](https://github.com/tachsin/genoxide/pull/64))
- add the ZDT and DTLZ test problems and Das-Dennis reference points ([#66](https://github.com/tachsin/genoxide/pull/66))
- add NSGA-III with reference directions ([#68](https://github.com/tachsin/genoxide/pull/68))
- add SPEA2, the strength Pareto evolutionary algorithm 2 ([#72](https://github.com/tachsin/genoxide/pull/72))
- add MOEA/D with Tchebycheff and PBI decomposition ([#74](https://github.com/tachsin/genoxide/pull/74))
- add SMS-EMOA and exclusive hypervolume contributions ([#76](https://github.com/tachsin/genoxide/pull/76))

### <!-- 1 -->Fixed

- [**breaking**] mutate each gene independently at the per-gene rate ([#70](https://github.com/tachsin/genoxide/pull/70))

### <!-- 4 -->Documentation

- benchmark multi-objective algorithms against pymoo and DEAP ([#77](https://github.com/tachsin/genoxide/pull/77))

## [0.3.0](https://github.com/tachsin/genoxide/compare/v0.2.0...v0.3.0) - 2026-09-24

### <!-- 0 -->Added

- add differential evolution with rand/1, best/1 and current-to-pbest/1 ([#41](https://github.com/tachsin/genoxide/pull/41))
- add adaptive differential evolution: JADE, SHADE and L-SHADE ([#43](https://github.com/tachsin/genoxide/pull/43))
- add particle swarm optimization with global and ring topologies ([#45](https://github.com/tachsin/genoxide/pull/45))
- add CMA-ES with IPOP and BIPOP restarts ([#47](https://github.com/tachsin/genoxide/pull/47))
- add sep-CMA-ES, a diagonal covariance mode for high dimensions ([#49](https://github.com/tachsin/genoxide/pull/49))
- add a (μ/ρ +, λ) evolution strategy with self-adaptation ([#51](https://github.com/tachsin/genoxide/pull/51))

### <!-- 4 -->Documentation

- benchmark genoxide's DE and CMA-ES, and refresh the charts ([#53](https://github.com/tachsin/genoxide/pull/53))

## [0.2.0](https://github.com/tachsin/genoxide/compare/v0.1.0...v0.2.0) - 2026-09-24

### <!-- 0 -->Added

- add Gaussian and polynomial mutation for real genomes ([#21](https://github.com/tachsin/genoxide/pull/21))
- add SBX, blend and arithmetic crossover for real genomes ([#23](https://github.com/tachsin/genoxide/pull/23))
- add crossovers and more mutations for permutation genomes ([#25](https://github.com/tachsin/genoxide/pull/25))
- add local search: hill climbing and simulated annealing ([#27](https://github.com/tachsin/genoxide/pull/27))
- add tabu search and iterated local search ([#29](https://github.com/tachsin/genoxide/pull/29))
- add memetic local search on the best parents of the GA ([#31](https://github.com/tachsin/genoxide/pull/31))
- add constraint handling with Deb's feasibility rules and penalty functions ([#33](https://github.com/tachsin/genoxide/pull/33))
- add self-adaptive Gaussian mutation with a step size per genome ([#37](https://github.com/tachsin/genoxide/pull/37))

### <!-- 2 -->Performance

- [**breaking**] choose genes by skipping ahead instead of one random number per gene ([#19](https://github.com/tachsin/genoxide/pull/19))

### <!-- 4 -->Documentation

- publish benchmark charts from Linux, with instruction counts per evaluation ([#35](https://github.com/tachsin/genoxide/pull/35))

## [0.1.0](https://github.com/tachsin/genoxide/releases/tag/v0.1.0) - 2026-09-23

### <!-- 0 -->Added

- add the core types: errors, a portable seedable rng with streams, and fitness values ([#3](https://github.com/tachsin/genoxide/pull/3))
- add genomes with a bit-packed binary representation, individuals and populations ([#5](https://github.com/tachsin/genoxide/pull/5))
- add selection, crossover and bit-flip mutation operators ([#7](https://github.com/tachsin/genoxide/pull/7))
- add the engine: ask/tell genetic algorithm, stop conditions, parallel evaluation and observers ([#9](https://github.com/tachsin/genoxide/pull/9))
- add integer, real and permutation genomes with uniform and swap mutation ([#11](https://github.com/tachsin/genoxide/pull/11))

### <!-- 2 -->Performance

- add criterion and instruction count benchmarks, and genoxide to the benchmark suite ([#16](https://github.com/tachsin/genoxide/pull/16))

### <!-- 4 -->Documentation

- add examples, the real API in the README, and AGENTS.md ([#13](https://github.com/tachsin/genoxide/pull/13))

### <!-- 5 -->Project setup

- Initial project: README, roadmap, licenses and crate skeleton
- Show the Rust advantages and the benchmark methodology in README and roadmap
- Add the benchmark suite and turn the roadmap into checklists
- Set up versioning, automated releases and CI
