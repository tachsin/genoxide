# Changelog

All notable changes to genoxide are documented in this file, generated from the pull request titles by [release-plz](https://release-plz.dev/).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and genoxide adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
