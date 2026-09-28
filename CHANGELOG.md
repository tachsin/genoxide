# Changelog

All notable changes to genoxide are documented in this file, generated from the pull request titles by [release-plz](https://release-plz.dev/).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and genoxide adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0](https://github.com/tachsin/genoxide/compare/v0.7.1...v0.8.0) - 2026-09-28

### <!-- 0 -->Added

- [**breaking**] add the test problem library: 16 classic functions and 13 classic multi-objective problems ([#177](https://github.com/tachsin/genoxide/pull/177))
- add the engineering design problems and CEC 2006's g01-g06 ([#191](https://github.com/tachsin/genoxide/pull/191))
- record the examples' output and a trace of each run, with the population in python's progress ([#194](https://github.com/tachsin/genoxide/pull/194))
- *(site)* draw the berlin52 tour on a map of Berlin's districts ([#242](https://github.com/tachsin/genoxide/pull/242))
- add genoxide's logo, wordmark and banner, and use them in the READMEs, on docs.rs and on the docs site ([#247](https://github.com/tachsin/genoxide/pull/247))
- change a GA's rates and operators between generations ([#248](https://github.com/tachsin/genoxide/pull/248))
- re-evaluate a GA's population when the fitness function changes ([#249](https://github.com/tachsin/genoxide/pull/249))
- add examples for the 20 problems of batches 1-3 that had none, with a grid plot on the project pages ([#252](https://github.com/tachsin/genoxide/pull/252))
- an example of its own for each of the 25 problems of batches 1-3 that had none ([#253](https://github.com/tachsin/genoxide/pull/253))
- an overall benchmark score, and the interactive results linked from the README ([#255](https://github.com/tachsin/genoxide/pull/255))
- [**breaking**] genoxide::math, the same to the bit on every platform, and the test problems use it ([#263](https://github.com/tachsin/genoxide/pull/263))

### <!-- 1 -->Fixed

- *(python)* reject NoCrossover with a mutation rate of 0, as documented ([#231](https://github.com/tachsin/genoxide/pull/231))
- *(python)* reject maximizing a test problem, read a batch's objectives from a tuple of columns, and correct the indicator docs ([#233](https://github.com/tachsin/genoxide/pull/233))
- skip points with NaN values in IGD+, check MOEA/D's size, and match multi-objective sizes and docs to their promises ([#234](https://github.com/tachsin/genoxide/pull/234))
- [**breaking**] truncation selects from exactly its fraction, and the engineering optima are feasible and the best known ([#235](https://github.com/tachsin/genoxide/pull/235))
- [**breaking**] count BIPOP's first run as a small one, find a relative fitness program on Unix, and name the setting in common mistakes ([#238](https://github.com/tachsin/genoxide/pull/238))
- *(site)* say which run the player plays, keep the player's focus, show the published benchmark charts, and don't publish a page without its README ([#239](https://github.com/tachsin/genoxide/pull/239))
- name the operator in a CLI error from its table, and check the test problems against their sources ([#241](https://github.com/tachsin/genoxide/pull/241))
- [**breaking**] an example for every problem, and SHADE's restarts, gx.De's options, CarSideImpact's best known value and four problems' ideal and nadir points ([#261](https://github.com/tachsin/genoxide/pull/261))

### <!-- 4 -->Documentation

- explain each example's problem, representation, algorithm and output ([#193](https://github.com/tachsin/genoxide/pull/193))
- check the examples' sources against the originals, count Kursawe's four pieces, and show the output on the docs site ([#198](https://github.com/tachsin/genoxide/pull/198))
- make 0.8 the test problem release in the roadmap, and describe only what exists ([#230](https://github.com/tachsin/genoxide/pull/230))
- describe multi-objective stagnation as it works, qualify reproducibility across platforms, and correct the example READMEs and the problems plan ([#236](https://github.com/tachsin/genoxide/pull/236))
- link every example to its interactive page on tachsin.gr ([#250](https://github.com/tachsin/genoxide/pull/250))

## [0.7.1](https://github.com/tachsin/genoxide/compare/v0.7.0...v0.7.1) - 2026-09-26

### <!-- 2 -->Performance

- cut the per-trial overhead of differential evolution (0.7.1) ([#182](https://github.com/tachsin/genoxide/pull/182))

## [0.7.0](https://github.com/tachsin/genoxide/compare/v0.6.0...v0.7.0) - 2026-09-26

### <!-- 1 -->Fixed

- [**breaking**] restart differential evolution at the next ask, and don't restart constrained populations too early ([#117](https://github.com/tachsin/genoxide/pull/117))
- [**breaking**] confirm duplicate children by equality, portable on 32 and 64 bits ([#118](https://github.com/tachsin/genoxide/pull/118))
- stop runs that can never end, bound huge sizes, and correct the docs ([#121](https://github.com/tachsin/genoxide/pull/121))
- [**breaking**] don't propose steady-state twins, and limit the initial PSO velocities ([#122](https://github.com/tachsin/genoxide/pull/122))
- clearer errors and stricter settings in the Python package ([#119](https://github.com/tachsin/genoxide/pull/119))
- correct the benchmark harness and docs, and use numpy fitness in pymoo and PyGAD ([#123](https://github.com/tachsin/genoxide/pull/123))
- report stalled runs in Python, and bound genome lengths and workers ([#126](https://github.com/tachsin/genoxide/pull/126))
- [**breaking**] use SHADE's published defaults for differential evolution ([#137](https://github.com/tachsin/genoxide/pull/137))

### <!-- 4 -->Documentation

- document the Python API's parameters, ranges and errors ([#164](https://github.com/tachsin/genoxide/pull/164))
- build the Python API reference with pdoc for GitHub Pages ([#162](https://github.com/tachsin/genoxide/pull/162))
- make 0.7 the correctness release, and move GP and neuroevolution to 0.8 ([#127](https://github.com/tachsin/genoxide/pull/127))
- keep the README to what helps choose a library ([#138](https://github.com/tachsin/genoxide/pull/138))
- shorten the README and move the feature list to docs/features.md ([#156](https://github.com/tachsin/genoxide/pull/156))
- tighten AGENTS.md and ROADMAP.md ([#157](https://github.com/tachsin/genoxide/pull/157))
- add the PyPI badge ([#160](https://github.com/tachsin/genoxide/pull/160))
- match docs/cli.md to the genoxide program ([#161](https://github.com/tachsin/genoxide/pull/161))
- fix the api docs and add errors, panics and examples sections ([#163](https://github.com/tachsin/genoxide/pull/163))
- add a docs site with the examples in Rust and Python tabs ([#165](https://github.com/tachsin/genoxide/pull/165))
- examples in a folder each, in rust and python, with problems from the literature ([#166](https://github.com/tachsin/genoxide/pull/166))
- show the benchmark charts of the 0.7 run, with a summary chart on the readme ([#175](https://github.com/tachsin/genoxide/pull/175))
- mark 0.7 as released ([#176](https://github.com/tachsin/genoxide/pull/176))

## [0.6.0](https://github.com/tachsin/genoxide/compare/v0.5.2...v0.6.0) - 2026-09-25

### <!-- 0 -->Added

- [**breaking**] eliminate duplicate children in NSGA-II, NSGA-III, SPEA2 and SMS-EMOA ([#109](https://github.com/tachsin/genoxide/pull/109))
- [**breaking**] restart differential evolution on stagnation, and default to SHADE with a small population ([#113](https://github.com/tachsin/genoxide/pull/113))
- add a Python package with numpy genomes, batch and parallel fitness functions ([#105](https://github.com/tachsin/genoxide/pull/105))
- add NSGA-III, SPEA2, MOEA/D, SMS-EMOA and a progress callback to the Python package ([#112](https://github.com/tachsin/genoxide/pull/112))

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
