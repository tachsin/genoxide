# Changelog

All notable changes to genoxide are documented in this file, generated from the pull request titles by [release-plz](https://release-plz.dev/).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and genoxide adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.1](https://github.com/tachsin/genoxide/compare/v0.11.0...v0.11.1) - 2026-10-01

### <!-- 0 -->Added

- *(nelder_mead)* add Nelder-Mead, convergence as a stop reason, and random restarts ([#371](https://github.com/tachsin/genoxide/pull/371))
- *(cmaes)* opt-in Restarts::Stop ends a run when it converges ([#388](https://github.com/tachsin/genoxide/pull/388))
- *(line_search)* the Moré-Thuente line search, resumable, reproducing the paper's tables ([#397](https://github.com/tachsin/genoxide/pull/397))
- *(gradient)* gradients through the engine, finite differences and the test problems' analytic gradients ([#398](https://github.com/tachsin/genoxide/pull/398))
- *(linalg)* in-crate linear algebra with the same bits on every platform and thread count ([#399](https://github.com/tachsin/genoxide/pull/399))
- *(lbfgsb)* add L-BFGS-B, completing batch A2 ([#400](https://github.com/tachsin/genoxide/pull/400))
- *(first_order)* gradient descent, momentum, Nesterov, Adam and AdamW ([#403](https://github.com/tachsin/genoxide/pull/403))
- conceptual marine design, the tenth problem of batch 9, with its example ([#401](https://github.com/tachsin/genoxide/pull/401))
- *(mma)* the method of moving asymptotes and GCMMA, for many variables and few constraints ([#404](https://github.com/tachsin/genoxide/pull/404))
- *(continuation)* stages of one problem with the optimizer's state kept, completing batch A3 ([#408](https://github.com/tachsin/genoxide/pull/408))

### <!-- 4 -->Documentation

- *(plan)* batch A3, first-order methods for millions of variables ([#396](https://github.com/tachsin/genoxide/pull/396))
- describe genoxide as it is now, with a method table and an llms.txt for AI assistants ([#407](https://github.com/tachsin/genoxide/pull/407))

## [0.11.0](https://github.com/tachsin/genoxide/compare/v0.10.0...v0.11.0) - 2026-10-01

### <!-- 0 -->Added

- *(python)* evolution strategies, islands and checkpoints ([#342](https://github.com/tachsin/genoxide/pull/342))
- *(gp)* tree genetic programming, strongly typed, with subtree crossover and mutation ([#343](https://github.com/tachsin/genoxide/pull/343))
- *(gp)* point, hoist, shrink and constant mutation, bloat control and Boolean problems ([#345](https://github.com/tachsin/genoxide/pull/345))
- *(nn)* neural networks and pole-balancing tasks, with CMA-ES examples ([#348](https://github.com/tachsin/genoxide/pull/348))
- *(gp)* symbolic regression, with linear scaling, and Koza's regression problems ([#359](https://github.com/tachsin/genoxide/pull/359))
- *(gp)* the Nguyen regression problems, their pages, and accuracy against size ([#360](https://github.com/tachsin/genoxide/pull/360))
- *(gp)* normal constants and the inverse, and G4's outcome ([#361](https://github.com/tachsin/genoxide/pull/361))
- *(neat)* add NEAT, networks whose structure evolves with their weights, and XOR by NEAT ([#363](https://github.com/tachsin/genoxide/pull/363))
- nine multi-objective engineering design problems, each with its own example ([#357](https://github.com/tachsin/genoxide/pull/357))
- *(neat)* add recurrent evaluation, and NEAT on the pole-balancing tasks ([#366](https://github.com/tachsin/genoxide/pull/366))
- fifteen classic test functions, from Beale to Powell, each with its own example ([#367](https://github.com/tachsin/genoxide/pull/367))
- *(open_es)* add OpenAI's evolution strategy, and two spirals by it ([#368](https://github.com/tachsin/genoxide/pull/368))
- *(python)* OpenEs, neural networks and pole-balancing tasks ([#385](https://github.com/tachsin/genoxide/pull/385))
- *(python)* NEAT and its networks ([#387](https://github.com/tachsin/genoxide/pull/387))
- *(python)* genetic programming with built-in primitives ([#389](https://github.com/tachsin/genoxide/pull/389))
- *(python)* genetic programming with your own primitives ([#390](https://github.com/tachsin/genoxide/pull/390))
- *(examples)* the pole-balancing pages animate the cart and its poles ([#391](https://github.com/tachsin/genoxide/pull/391))

### <!-- 1 -->Fixed

- a checkpoint of a De with fewer than 4 individuals, or of Islands with no island, is an error ([#346](https://github.com/tachsin/genoxide/pull/346))
- *(cli)* a fitness program that writes extra lines or stops answering ends the run with an error ([#347](https://github.com/tachsin/genoxide/pull/347))
- lengths above 2^24 are errors for Real and Integer, and CMA-ES restarts stay within 2^24 ([#344](https://github.com/tachsin/genoxide/pull/344))
- *(python)* a parallel run stops on Ctrl+C after the calls under way, not after the generation ([#350](https://github.com/tachsin/genoxide/pull/350))
- the problems' genome and objectives pass type checking, and small gaps in the CLI and Python checks ([#353](https://github.com/tachsin/genoxide/pull/353))
- *(examples)* progress plots that end where the progress does ([#372](https://github.com/tachsin/genoxide/pull/372))
- *(examples)* front pages that end where the front stops visibly moving ([#384](https://github.com/tachsin/genoxide/pull/384))

### <!-- 2 -->Performance

- the NSGA-III association skips the exact distance to reference directions a lower bound rules out ([#354](https://github.com/tachsin/genoxide/pull/354))
- permutation operators, SPEA2, NSGA-III and MOEA/D reuse their scratch space instead of allocating it ([#355](https://github.com/tachsin/genoxide/pull/355))
- [**breaking**] both normals of each polar-method draw, lazy CMA-ES eigendecomposition, and integer powers in the CTP constraints ([#356](https://github.com/tachsin/genoxide/pull/356))
- the last front of SMS-EMOA is sorted once, not after every removal ([#352](https://github.com/tachsin/genoxide/pull/352))

### <!-- 4 -->Documentation

- a plan for genetic programming and neuroevolution ([#340](https://github.com/tachsin/genoxide/pull/340))
- *(contributing)* no AI attribution in commits, PRs, comments or files ([#370](https://github.com/tachsin/genoxide/pull/370))

## [0.10.0](https://github.com/tachsin/genoxide/compare/v0.9.3...v0.10.0) - 2026-09-29

### <!-- 0 -->Added

- *(python)* [**breaking**] the progress object makes the population's arrays when first read ([#324](https://github.com/tachsin/genoxide/pull/324))
- *(examples)* the N-Queens example on 8×8 to 128×128 boards, one family with tabs ([#321](https://github.com/tachsin/genoxide/pull/321))

### <!-- 2 -->Performance

- [**breaking**] move to Rust 1.88, with small performance gains across the library ([#315](https://github.com/tachsin/genoxide/pull/315))

### <!-- 4 -->Documentation

- *(examples)* remove the function suite, since each function has its own page ([#317](https://github.com/tachsin/genoxide/pull/317))
- the problem plan has one problem per example, and no comparison examples ([#322](https://github.com/tachsin/genoxide/pull/322))
- a plan for general optimization methods ([#316](https://github.com/tachsin/genoxide/pull/316))
- the roadmap puts genetic programming and neuroevolution in 0.11, and the other methods from 0.12 ([#325](https://github.com/tachsin/genoxide/pull/325))

## [0.9.3](https://github.com/tachsin/genoxide/compare/v0.9.2...v0.9.3) - 2026-09-29

### <!-- 0 -->Added

- batch 6 of the test problems, CEC 2006's g19-g24 and eight classic functions, each with its example ([#307](https://github.com/tachsin/genoxide/pull/307))
- batches 7 and 8 of the test problems, CTP, C-DTLZ, scaled and inverted DTLZ, and MW, each with its example ([#313](https://github.com/tachsin/genoxide/pull/313))

### <!-- 1 -->Fixed

- *(examples)* every example reaches its optimum ([#312](https://github.com/tachsin/genoxide/pull/312))

### <!-- 2 -->Performance

- recover the instructions 0.9.2 added to the GA, DE, CMA-ES and PSO ([#302](https://github.com/tachsin/genoxide/pull/302))
- a GA keeps its best without a full sort, and breeds into its unused genomes ([#306](https://github.com/tachsin/genoxide/pull/306))

### <!-- 4 -->Documentation

- room in the examples' order for batches 7 and 8 ([#309](https://github.com/tachsin/genoxide/pull/309))

## [0.9.2](https://github.com/tachsin/genoxide/compare/v0.9.1...v0.9.2) - 2026-09-28

### <!-- 0 -->Added

- *(python)* parallel_breeding for De ([#296](https://github.com/tachsin/genoxide/pull/296))
- *(site)* example families as tabs, and a sidebar of every example ([#279](https://github.com/tachsin/genoxide/pull/279))
- *(benchmarks)* single-objective scenarios only, for now (the benchmark harness's commands changed; the library's API didn't) ([#283](https://github.com/tachsin/genoxide/pull/283))
- *(benchmarks)* a matched suite of three problems, one method each (a new harness command set, as above) ([#284](https://github.com/tachsin/genoxide/pull/284))
- breed a GA's offspring in parallel with GaBuilder::parallel_breeding ([#289](https://github.com/tachsin/genoxide/pull/289))
- parameter control with Engine::control, and re-evaluation in the single-objective algorithms ([#292](https://github.com/tachsin/genoxide/pull/292))
- isolated islands that never migrate, with Topology::Isolated ([#293](https://github.com/tachsin/genoxide/pull/293))
- *(python)* parameter control and re-evaluation with run(control=...) ([#294](https://github.com/tachsin/genoxide/pull/294))
- build DE trials and ES offspring in parallel with parallel_breeding ([#295](https://github.com/tachsin/genoxide/pull/295))
- keep what a fitness function computed alongside the fitness ([#301](https://github.com/tachsin/genoxide/pull/301))

### <!-- 2 -->Performance

- push each child straight into the offspring when breeding serially ([#290](https://github.com/tachsin/genoxide/pull/290))
- *(python)* twice as fast on bit genomes, one GIL attach per generation ([#291](https://github.com/tachsin/genoxide/pull/291))
- take an ES's step-size logarithms once per generation ([#298](https://github.com/tachsin/genoxide/pull/298))

### <!-- 4 -->Documentation

- the roadmap as released: this release's control and parallel breeding, genetic programming in 0.10 ([#299](https://github.com/tachsin/genoxide/pull/299))

## [0.9.1](https://github.com/tachsin/genoxide/compare/v0.9.0...v0.9.1) - 2026-09-28

### <!-- 0 -->Added

- batch 5 of the test problems, CEC 2006's g07-g18, each with its example ([#277](https://github.com/tachsin/genoxide/pull/277))

## [0.9.0](https://github.com/tachsin/genoxide/compare/v0.8.0...v0.9.0) - 2026-09-28

### <!-- 0 -->Added

- *(benchmarks)* instruction counts for genoxide only, compared across its versions ([#267](https://github.com/tachsin/genoxide/pull/267))
- *(python)* gx.Cmaes chooses the covariance, full or diagonal ([#268](https://github.com/tachsin/genoxide/pull/268))
- batch 4 of the test problems, DTLZ5-7, the binary ZDT5 and WFG1-9, each with its example ([#270](https://github.com/tachsin/genoxide/pull/270))

### <!-- 1 -->Fixed

- *(site)* draw WFG2's true front in its six pieces ([#272](https://github.com/tachsin/genoxide/pull/272))
- [**breaking**] a multi-objective front has each genome once, in Rust, Python and the CLI ([#273](https://github.com/tachsin/genoxide/pull/273))
- *(benchmarks)* commit the published run, publish and rerun it with one command, and move radiate to 1.3.2 ([#275](https://github.com/tachsin/genoxide/pull/275))
- [**breaking**] polynomial mutation reaches the bounds, without cancellation near them ([#276](https://github.com/tachsin/genoxide/pull/276))

### <!-- 4 -->Documentation

- install with cargo add, so the README never names an old version ([#265](https://github.com/tachsin/genoxide/pull/265))

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
