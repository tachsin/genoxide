//! Benchmark adapter for moors (https://crates.io/crates/moors), the Rust core of moo-rs.
//! What it runs, where each setting comes from, and the separate tests: the page
//! docs/benchmarks/libraries/moors.md. The citations below point to the moo-rs repository at the
//! commit of moors 0.2.11 (488063e, https://github.com/andresliszt/moo-rs/tree/488063e) and to the
//! crate's source (src/..., as in ~/.cargo/registry/src/*/moors-0.2.11).
//!
//! Usage:
//!   ga_bench_moors <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//!   ga_bench_moors values <problem> <size>   (one JSON solution per line on stdin)
//! Prints one JSON line per solver per seed, see ../../README.md for the fields.
//!
//! How moors runs, and what that means for the numbers:
//! - Every algorithm is a GeneticAlgorithm loop with a fixed number of iterations
//!   (src/algorithms/ga.rs). Each iteration evaluates the current population and its offspring
//!   together (`evaluator.evaluate` on the concatenation of both, ga.rs `next`), so the survivors
//!   are evaluated again every generation: a generation of population P and O offspring costs
//!   P + O evaluations. The fitness function counts every row it is given, so `evaluations`
//!   includes these re-evaluations.
//! - The run ends at the iteration count, or early when the duplicate removal leaves no offspring
//!   in 200 tries (EmptyMatingResult, ga.rs `run`, "Terminating the algorithm early"). The adapter
//!   sets the iteration count above any budget and stops the run from an AdaptiveController
//!   (src/algorithms/helpers/controller.rs: "may ... request an early stop"), called after every
//!   generation: at the evaluation budget, at the time limit and (single-objective) at the target.
//!   The budget is checked after each generation, so the last one can go over it by less than one
//!   generation. A single-objective run that ends early by itself is restarted with the seed
//!   seed * 1000 + restart, keeping the best (rule 2.2).
//! - moors has no rayon or BLAS dependency (faer without its rayon feature, ndarray without
//!   threading), so the runs are single-threaded.

use moors::{
    AdaptiveController, AlgorithmBuilder, AlgorithmContext, BitFlipMutation,
    CloseDuplicatesCleaner, ControlSignal, ExactDuplicatesCleaner, GaussianMutation,
    MutationOperator, NoConstraints, Nsga2Builder, Nsga3Builder, OrderCrossover,
    PermutationSampling, Population, RandomSamplingBinary, RandomSamplingFloat, SelectionOperator,
    SimulatedBinaryCrossover, SinglePointBinaryCrossover, Spea2Builder, SurvivalOperator,
    SwapMutation, TwoPointBinaryCrossover,
    genetic::{D01, D12, IndividualSOO},
    impl_constraints_fn,
    random::RandomGenerator,
    selection::{DuelResult, soo::RankSelection},
    survival::moo::{DanAndDenisReferencePoints, StructuredReferencePoints},
    survival::soo::FitnessSurvival,
};
use ndarray::{Array1, Array2, ArrayViewMut1, Dimension, Ix1, Ix2};
use std::cell::{Cell, RefCell};
use std::f64::consts::{E, PI};
use std::io::{BufRead, Write};
use std::rc::Rc;
use std::time::Instant;

// ---------------------------------------------------------------------------------------------
// Fitness functions, identical to benchmarks/problems.py, in the problems' own direction (OneMax
// maximized, the others minimized); moors minimizes, so OneMax is given to it negated
// ---------------------------------------------------------------------------------------------

fn onemax(x: &[f64]) -> f64 {
    x.iter().filter(|&&bit| bit > 0.5).count() as f64
}

fn onemax_minimized(x: &[f64]) -> f64 {
    // the moors fitness function docs (docs/user_guide/fitness_and_constraints/rust/fitness.md)
    // and its knapsack example (docs/getting_started/rust/knapsack.md) minimize the negative
    -onemax(x)
}

/// Number of diagonal conflicts, O(n): for each diagonal, its queens minus one
fn nqueens(x: &[f64]) -> f64 {
    let size = x.len();
    let mut left_diagonal = vec![0usize; 2 * size - 1];
    let mut right_diagonal = vec![0usize; 2 * size - 1];
    for (i, &gene) in x.iter().enumerate() {
        let gene = gene as usize;
        left_diagonal[i + gene] += 1;
        right_diagonal[size - 1 - i + gene] += 1;
    }
    left_diagonal
        .iter()
        .chain(&right_diagonal)
        .map(|&count| count.saturating_sub(1))
        .sum::<usize>() as f64
}

// Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
// towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
fn shift(i: usize) -> f64 {
    2.0 * ((37 * i + 11) % 101) as f64 / 101.0 - 1.0
}

fn rastrigin(x: &[f64]) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .enumerate()
            .map(|(i, v)| {
                let v = v - shift(i);
                v * v - 10.0 * (2.0 * PI * v).cos()
            })
            .sum::<f64>()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let shifted = || x.iter().enumerate().map(|(i, v)| v - shift(i));
    let squares = shifted().map(|v| v * v).sum::<f64>() / n;
    let cosines = shifted().map(|v| (2.0 * PI * v).cos()).sum::<f64>() / n;
    -20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + 20.0 + E
}

fn zdt_g(x: &[f64]) -> f64 {
    1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64
}

fn zdt1(x: &[f64], f: &mut [f64]) {
    let g = zdt_g(x);
    f[0] = x[0];
    f[1] = g * (1.0 - (x[0] / g).sqrt());
}

fn zdt2(x: &[f64], f: &mut [f64]) {
    let g = zdt_g(x);
    f[0] = x[0];
    f[1] = g * (1.0 - (x[0] / g).powi(2));
}

fn zdt3(x: &[f64], f: &mut [f64]) {
    let g = zdt_g(x);
    f[0] = x[0];
    f[1] = g * (1.0 - (x[0] / g).sqrt() - x[0] / g * (10.0 * PI * x[0]).sin());
}

/// DTLZ2 with M = f.len() objectives; the last n - M + 1 variables are the distance variables
fn dtlz2(x: &[f64], f: &mut [f64]) {
    let m = f.len();
    let g: f64 = x[m - 1..].iter().map(|v| (v - 0.5).powi(2)).sum();
    for (i, value) in f.iter_mut().enumerate() {
        let mut v = 1.0 + g;
        for xj in &x[..m - 1 - i] {
            v *= (xj * PI / 2.0).cos();
        }
        if i > 0 {
            v *= (x[m - 1 - i] * PI / 2.0).sin();
        }
        *value = v;
    }
}

/// DTLZ1 with M = f.len() objectives; the last n - M + 1 variables are the distance variables
fn dtlz1(x: &[f64], f: &mut [f64]) {
    let m = f.len();
    let tail = &x[m - 1..];
    let g = 100.0
        * (tail.len() as f64
            + tail
                .iter()
                .map(|v| (v - 0.5).powi(2) - (20.0 * PI * (v - 0.5)).cos())
                .sum::<f64>());
    for (i, value) in f.iter_mut().enumerate() {
        let mut v = 0.5 * (1.0 + g);
        for xj in &x[..m - 1 - i] {
            v *= xj;
        }
        if i > 0 {
            v *= 1.0 - x[m - 1 - i];
        }
        *value = v;
    }
}

type Objectives = fn(&[f64], &mut [f64]);

/// A multi-objective problem of the given size: (function, variables, objectives), as
/// problems.py FRONT_VARIABLES and FRONT_OBJECTIVES
fn front_problem(problem: &str, size: usize) -> (Objectives, usize, usize) {
    match problem {
        "zdt1" => (zdt1, size, 2),
        "zdt2" => (zdt2, size, 2),
        "zdt3" => (zdt3, size, 2),
        // size: the number of objectives, with 10 distance variables (DTLZ2) or 5 (DTLZ1)
        "dtlz2" => (dtlz2, size + 9, size),
        "dtlz1" => (dtlz1, size + 4, size),
        _ => unreachable!(),
    }
}

// the variable bounds: moors clamps the offspring to the bounds of the constraints function
// (docs/user_guide/fitness_and_constraints/rust/lower_upper_bounds.md; src/operators/evolve.rs
// `mating_batch`); the bound constraints are then always satisfied
impl_constraints_fn!(UnitBounds, lower_bound = 0.0, upper_bound = 1.0);
impl_constraints_fn!(RastriginBounds, lower_bound = -5.12, upper_bound = 5.12);
impl_constraints_fn!(RosenbrockBounds, lower_bound = -5.0, upper_bound = 10.0);
impl_constraints_fn!(AckleyBounds, lower_bound = -32.768, upper_bound = 32.768);

// ---------------------------------------------------------------------------------------------
// Operators moors doesn't have, through its documented extension points (the MutationOperator,
// SelectionOperator and SurvivalOperator traits: docs/user_guide/operators/rust/mutation.md and
// docs/user_guide/algorithms/custom/rust-custom.md). Only the matched scenarios use them.
// ---------------------------------------------------------------------------------------------

/// Deb's bounded polynomial mutation (the formula of pymoo's PM): moors has none
#[derive(Debug, Clone)]
struct PolynomialMutation {
    gene_mutation_rate: f64,
    eta: f64,
    lower: f64,
    upper: f64,
}

impl MutationOperator for PolynomialMutation {
    fn mutate<'a>(&self, mut individual: ArrayViewMut1<'a, f64>, rng: &mut impl RandomGenerator) {
        let range = self.upper - self.lower;
        let power = 1.0 / (self.eta + 1.0);
        for gene in individual.iter_mut() {
            if rng.gen_proability() >= self.gene_mutation_rate {
                continue;
            }
            let y = *gene;
            let delta1 = (y - self.lower) / range;
            let delta2 = (self.upper - y) / range;
            let r = rng.gen_proability();
            let delta_q = if r < 0.5 {
                let value = 2.0 * r + (1.0 - 2.0 * r) * (1.0 - delta1).powf(self.eta + 1.0);
                value.powf(power) - 1.0
            } else {
                let value = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - delta2).powf(self.eta + 1.0);
                1.0 - value.powf(power)
            };
            *gene = (y + delta_q * range).clamp(self.lower, self.upper);
        }
    }
}

/// DEAP's selTournament: each parent is the fittest of `size` individuals drawn with replacement.
/// moors' own tournaments are binary: SelectionOperator::operate pairs the participants
/// (src/operators/selection/mod.rs), so this operator replaces `operate`.
#[derive(Debug, Clone)]
struct TournamentSelection {
    size: usize,
}

impl SelectionOperator for TournamentSelection {
    type FDim = Ix1;

    // only used by the default `operate`, which this operator replaces
    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &IndividualSOO<'a, ConstrDim>,
        p2: &IndividualSOO<'a, ConstrDim>,
        _rng: &mut impl RandomGenerator,
    ) -> DuelResult
    where
        ConstrDim: D01,
    {
        match p1.fitness[()].partial_cmp(&p2.fitness[()]) {
            Some(std::cmp::Ordering::Less) => DuelResult::LeftWins,
            Some(std::cmp::Ordering::Greater) => DuelResult::RightWins,
            _ => DuelResult::Tie,
        }
    }

    fn operate<ConstrDim>(
        &self,
        population: &Population<Ix1, ConstrDim>,
        n_crossovers: usize,
        rng: &mut impl RandomGenerator,
    ) -> (Population<Ix1, ConstrDim>, Population<Ix1, ConstrDim>)
    where
        ConstrDim: D12,
        <ConstrDim as Dimension>::Smaller: D01,
        <Ix1 as Dimension>::Smaller: D01,
    {
        let n = population.len();
        let mut winner = || {
            (0..self.size)
                .map(|_| rng.gen_range_usize(0, n))
                .min_by(|&a, &b| population.fitness[a].total_cmp(&population.fitness[b]))
                .expect("a tournament of at least one")
        };
        // the parents of pair i are selected parents 2i and 2i + 1, as DEAP's varAnd pairs them
        let (mut first, mut second) = (Vec::new(), Vec::new());
        for _ in 0..n_crossovers {
            first.push(winner());
            second.push(winner());
        }
        (population.selected(&first), population.selected(&second))
    }
}

/// Generational replacement (DEAP eaSimple: the offspring replace the population). moors
/// evaluates [population; offspring] together (src/algorithms/ga.rs `next`), so the offspring are
/// the last rows.
#[derive(Debug, Clone)]
struct GenerationalSurvival;

impl SurvivalOperator for GenerationalSurvival {
    type FDim = Ix1;

    fn operate<ConstrDim>(
        &mut self,
        population: Population<Ix1, ConstrDim>,
        num_survive: usize,
        _rng: &mut impl RandomGenerator,
    ) -> Population<Ix1, ConstrDim>
    where
        ConstrDim: D12,
    {
        let rows = population.len();
        let offspring: Vec<usize> = (rows.saturating_sub(num_survive)..rows).collect();
        population.selected(&offspring)
    }
}

// ---------------------------------------------------------------------------------------------
// Budget: counts evaluations in the fitness function, stops the run from a controller
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct Counters {
    evaluations: Rc<Cell<usize>>,
    generations: Rc<Cell<usize>>,
    // whether the controller ended the run (budget, time or target), as opposed to moors itself
    stopped: Rc<Cell<bool>>,
    // evaluated solutions with a gene outside the problem's bounds (rule 2.4), as moors proposed
    // them to the fitness function
    outside: Rc<Cell<usize>>,
}

/// Whether a solution has a gene outside [lower, upper]
fn is_outside(x: &[f64], (lower, upper): (f64, f64)) -> bool {
    x.iter().any(|&v| !(lower..=upper).contains(&v))
}

/// The best solution evaluated in a single-objective run: its minimized fitness and genes
#[derive(Debug)]
struct Best {
    fitness: f64,
    genes: Vec<f64>,
}

type Tracker = Rc<RefCell<Best>>;

/// Called by moors after every generation: stops at the evaluations, the time or the target
#[derive(Debug)]
struct Budget {
    counters: Counters,
    max_evaluations: usize,
    max_seconds: f64,
    start: Instant,
    // single-objective: the best solution so far, and the target (minimized) that ends the run
    best: Option<(Tracker, f64)>,
}

impl Budget {
    fn signal(&self) -> ControlSignal {
        let counters = &self.counters;
        counters.generations.set(counters.generations.get() + 1);
        let reached = self
            .best
            .as_ref()
            .is_some_and(|(best, target)| best.borrow().fitness <= *target);
        let stop = reached
            || counters.evaluations.get() >= self.max_evaluations
            || self.start.elapsed().as_secs_f64() >= self.max_seconds;
        counters.stopped.set(stop);
        ControlSignal {
            stop,
            ..Default::default()
        }
    }
}

impl<C: D12> AdaptiveController<Ix1, C> for Budget {
    fn observe(&mut self, _: usize, _: &Population<Ix1, C>, _: &AlgorithmContext) -> ControlSignal {
        self.signal()
    }
}

impl AdaptiveController<Ix2, Ix2> for Budget {
    fn observe(
        &mut self,
        _: usize,
        _: &Population<Ix2, Ix2>,
        _: &AlgorithmContext,
    ) -> ControlSignal {
        self.signal()
    }
}

/// The fitness function moors calls with a population: one value per row (the signature of
/// docs/user_guide/fitness_and_constraints/rust/fitness.md), counting every row, the rows outside
/// the bounds (continuous problems) and keeping the best row
fn soo_fitness(
    counters: &Counters,
    best: &Tracker,
    f: Single,
    bounds: Option<(f64, f64)>,
) -> impl Fn(&Array2<f64>) -> Array1<f64> + 'static {
    let evaluations = counters.evaluations.clone();
    let outside = counters.outside.clone();
    let best = best.clone();
    move |genes: &Array2<f64>| {
        evaluations.set(evaluations.get() + genes.nrows());
        let mut best = best.borrow_mut();
        genes
            .rows()
            .into_iter()
            .map(|row| {
                let owned;
                let x = match row.as_slice() {
                    Some(x) => x,
                    None => {
                        owned = row.to_vec();
                        &owned
                    }
                };
                if bounds.is_some_and(|bounds| is_outside(x, bounds)) {
                    outside.set(outside.get() + 1);
                }
                let value = f(x);
                if value < best.fitness {
                    best.fitness = value;
                    best.genes = row.to_vec();
                }
                value
            })
            .collect()
    }
}

/// The multi-objective fitness function: one row of objectives per row of genes, counting rows
/// and the rows outside [0, 1]
fn moo_fitness(
    counters: &Counters,
    objectives: usize,
    f: Objectives,
) -> impl Fn(&Array2<f64>) -> Array2<f64> + 'static {
    let evaluations = counters.evaluations.clone();
    let outside = counters.outside.clone();
    move |genes: &Array2<f64>| {
        evaluations.set(evaluations.get() + genes.nrows());
        let mut values = Array2::<f64>::zeros((genes.nrows(), objectives));
        for (row, mut out) in genes.rows().into_iter().zip(values.rows_mut()) {
            let out = out.as_slice_mut().expect("standard layout");
            let owned;
            let x = match row.as_slice() {
                Some(x) => x,
                None => {
                    owned = row.to_vec();
                    &owned
                }
            };
            if is_outside(x, (0.0, 1.0)) {
                outside.set(outside.get() + 1);
            }
            f(x, out);
        }
        values
    }
}

// ---------------------------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------------------------

struct Args {
    problem: String,
    size: usize,
    mode: String,
    seed_from: u64,
    seed_to: u64,
    max_evaluations: usize,
    max_seconds: f64,
}

/// Where the JSON lines go. moors prints warnings with println! (e.g. when the duplicate
/// removal leaves too few offspring), so on unix stdout (fd 1) is pointed at stderr and the
/// results are written to a duplicate of the original stdout.
fn results_output() -> Box<dyn Write> {
    #[cfg(unix)]
    {
        use std::os::fd::FromRawFd;
        // SAFETY: plain file descriptor duplication, before anything is printed
        unsafe {
            let fd = libc::dup(1);
            if fd >= 0 && libc::dup2(2, 1) >= 0 {
                return Box::new(std::fs::File::from_raw_fd(fd));
            }
        }
    }
    Box::new(std::io::stdout())
}

fn emit(output: &mut dyn Write, line: String) {
    output.write_all(line.as_bytes()).expect("write result");
    output.write_all(b"\n").expect("write result");
    output.flush().expect("flush result");
}

fn json_reals(values: &[f64]) -> String {
    let values: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
    format!("[{}]", values.join(","))
}

/// A solution as the problem's genome: bits as 0/1, the permutation as integers, reals in full
fn json_solution(problem: &str, genes: &[f64]) -> String {
    match problem {
        "onemax" => {
            let bits: Vec<&str> = genes
                .iter()
                .map(|&bit| if bit > 0.5 { "1" } else { "0" })
                .collect();
            format!("[{}]", bits.join(","))
        }
        "nqueens" => {
            let order: Vec<String> = genes.iter().map(|&g| (g as usize).to_string()).collect();
            format!("[{}]", order.join(","))
        }
        _ => json_reals(genes),
    }
}

/// A value in the problem's direction: integers for OneMax and N-Queens
fn json_value(problem: &str, value: f64) -> String {
    match problem {
        "onemax" | "nqueens" => format!("{}", value as i64),
        _ => format!("{value:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Single-objective runs
// ---------------------------------------------------------------------------------------------

type Single = fn(&[f64]) -> f64;

struct SooProblem {
    // moors minimizes this
    minimized: Single,
    // the value in the problem's own direction
    value: Single,
    // in the problem's own direction
    target: f64,
    maximized: bool,
    // the box of a continuous problem (problems.py REAL_BOUNDS)
    bounds: Option<(f64, f64)>,
}

impl SooProblem {
    fn of(problem: &str, size: usize) -> Self {
        let (minimized, value, target, maximized): (Single, Single, f64, bool) = match problem {
            "onemax" => (onemax_minimized, onemax, size as f64, true),
            "nqueens" => (nqueens, nqueens, 0.0, false),
            "rastrigin" => (rastrigin, rastrigin, 0.01, false),
            "rosenbrock" => (rosenbrock, rosenbrock, 0.01, false),
            "ackley" => (ackley, ackley, 0.01, false),
            _ => unreachable!(),
        };
        let bounds = match problem {
            "rastrigin" => Some((-5.12, 5.12)),
            "rosenbrock" => Some((-5.0, 10.0)),
            "ackley" => Some((-32.768, 32.768)),
            _ => None,
        };
        SooProblem {
            minimized,
            value,
            target,
            maximized,
            bounds,
        }
    }

    // problems.reached
    fn reached(&self, value: f64) -> bool {
        if self.maximized {
            value >= self.target
        } else {
            value <= self.target
        }
    }

    // the target in moors' direction
    fn minimized_target(&self) -> f64 {
        if self.maximized {
            -self.target
        } else {
            self.target
        }
    }
}

fn run_single(output: &mut dyn Write, args: &Args, seed: u64) {
    let size = args.size;
    let problem = SooProblem::of(&args.problem, size);
    let counters = Counters::default();
    let best: Tracker = Rc::new(RefCell::new(Best {
        fitness: f64::INFINITY,
        genes: Vec::new(),
    }));
    let start = Instant::now();
    // rule 2.2: moors' iteration count is only a budget, so it's set above any budget; its one
    // convergence criterion, no new offspring (EmptyMatingResult), ends an attempt, and moors has no
    // restart mechanism, so the GA starts again from a new random population with the seed
    // seed * 1000 + restart, keeping the best
    let mut restart = 0u64;
    loop {
        let run_seed = if restart == 0 {
            seed
        } else {
            seed * 1000 + restart
        };
        let budget = Budget {
            counters: counters.clone(),
            max_evaluations: args.max_evaluations,
            max_seconds: args.max_seconds,
            start,
            best: Some((best.clone(), problem.minimized_target())),
        };
        let fitness = soo_fitness(&counters, &best, problem.minimized, problem.bounds);
        // builds and runs one moors GA; every generation evaluates at least one row, so the
        // budget ends the run before this iteration count
        macro_rules! run {
            ($builder:expr) => {{
                let mut algorithm = $builder
                    .fitness_fn(fitness)
                    .num_vars(size)
                    .controller(budget)
                    .num_iterations(args.max_evaluations.max(1))
                    .seed(run_seed)
                    .verbose(false)
                    .build()
                    .expect("valid moors configuration");
                algorithm.run().expect("moors run");
            }};
        }
        // moors' single-objective GA: AlgorithmBuilder with RankSelection (binary tournament on
        // the fitness rank) and FitnessSurvival (the best of parents and offspring survive). These
        // two are what pymoors' GeneticAlgorithmSOO fixes (pymoors/src/algorithms/soo/mod.rs
        // lines 82-88) and what moors' single-objective test uses (moors/tests/test_ga_soo.rs
        // lines 23-28); the survival and selection of a custom algorithm are the user's choice
        // (docs/user_guide/algorithms/custom/rust-custom.md).
        macro_rules! ga {
            () => {
                AlgorithmBuilder::default()
                    .selector(RankSelection)
                    .survivor(FitnessSurvival)
            };
        }
        match (args.problem.as_str(), args.mode.as_str()) {
            // matched: DEAP eaSimple. Population 300, 300 offspring, tournament of 3, two-point
            // crossover with probability 0.5, bit flip with probability 1 / size per gene on 20%
            // of the children, generational replacement. The tournament and the replacement are
            // the adapter's (above): moors' tournaments are binary and its single-objective
            // survival keeps the best. The one difference left: moors evaluates the 300 parents
            // again every generation (600 evaluations per generation).
            ("onemax", "matched") => run!(
                AlgorithmBuilder::default()
                    .selector(TournamentSelection { size: 3 })
                    .survivor(GenerationalSurvival)
                    .sampler(RandomSamplingBinary::new())
                    .crossover(TwoPointBinaryCrossover)
                    .mutation(BitFlipMutation::new(1.0 / size as f64))
                    .constraints_fn(NoConstraints)
                    .population_size(300)
                    .num_offsprings(300)
                    .crossover_rate(0.5)
                    .mutation_rate(0.2)
            ),
            // binary: the example of the moors README (Quickstart) and docs/getting_started/
            // rust/quick_start.md: RandomSamplingBinary, SinglePointBinaryCrossover at 0.9,
            // BitFlipMutation on 10% of the children, ExactDuplicatesCleaner, population 100
            // with 32 offspring. Its per-gene flip rate of 0.5 is sized for its 5-variable
            // knapsack (2.5 flips per mutated child) and moors doesn't say how to scale it; here it
            // is 1 / size, one flip per mutated child (the page explains it).
            ("onemax", _) => run!(
                ga!()
                    .sampler(RandomSamplingBinary::new())
                    .crossover(SinglePointBinaryCrossover::new())
                    .mutation(BitFlipMutation::new(1.0 / size as f64))
                    .duplicates_cleaner(ExactDuplicatesCleaner::new())
                    .constraints_fn(NoConstraints)
                    .population_size(100)
                    .num_offsprings(32)
                    .crossover_rate(0.9)
                    .mutation_rate(0.1)
            ),
            // permutation: moors has no permutation example, no stated preference and no default
            // among its permutation operators (docs/user_guide/operators/rust/sampling.md,
            // crossover.md, mutation.md): PermutationSampling, OrderCrossover (its one permutation
            // crossover), SwapMutation (the mutation its docs describe as exploring "neighboring
            // permutations"); the rates are AlgorithmBuilder's
            // defaults (crossover 0.9, mutation 0.2: src/algorithms/builder.rs lines 86-89), the
            // population and offspring 200 of the docs' real-valued example
            // (docs/getting_started/rust/real_valued.md), and exact duplicate removal as in the
            // README's discrete example
            ("nqueens", _) => run!(
                ga!()
                    .sampler(PermutationSampling::new())
                    .crossover(OrderCrossover::new())
                    .mutation(SwapMutation::new())
                    .duplicates_cleaner(ExactDuplicatesCleaner::new())
                    .constraints_fn(NoConstraints)
                    .population_size(200)
                    .num_offsprings(200)
                    .crossover_rate(0.9)
                    .mutation_rate(0.2)
            ),
            // continuous: the docs' example for real-valued problems
            // (docs/getting_started/rust/real_valued.md; rule 6.2: no stated preference, so the
            // example for the problem type): RandomSamplingFloat, SimulatedBinaryCrossover with eta
            // 15 at 0.9, GaussianMutation of 10% of the genes with sigma 0.01 on 20% of the
            // children, CloseDuplicatesCleaner(1e-16), population and offspring 200. Bounds
            // (rule 2.4): SBX and the Gaussian mutation ignore them, and moors clamps every
            // offspring to the lower_bound and upper_bound of the constraints function before it
            // is evaluated (docs/user_guide/fitness_and_constraints/rust/lower_upper_bounds.md,
            // src/operators/evolve.rs mating_batch).
            (name, _) => {
                macro_rules! real {
                    ($bounds:expr, $low:expr, $high:expr) => {
                        run!(
                            ga!()
                                .sampler(RandomSamplingFloat::new($low, $high))
                                .crossover(SimulatedBinaryCrossover::new(15.0))
                                .mutation(GaussianMutation::new(0.1, 0.01))
                                .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-16))
                                .constraints_fn($bounds)
                                .population_size(200)
                                .num_offsprings(200)
                                .crossover_rate(0.9)
                                .mutation_rate(0.2)
                        )
                    };
                }
                match name {
                    "rastrigin" => real!(RastriginBounds, -5.12, 5.12),
                    "rosenbrock" => real!(RosenbrockBounds, -5.0, 10.0),
                    _ => real!(AckleyBounds, -32.768, 32.768),
                }
            }
        }
        // ended by the controller (target, budget or time), or by moors itself (EmptyMatingResult)
        if counters.stopped.get()
            || counters.evaluations.get() >= args.max_evaluations
            || start.elapsed().as_secs_f64() >= args.max_seconds
        {
            break;
        }
        restart += 1;
        eprintln!(
            "moors ga on {} seed {seed}: ended by itself after {} evaluations, restart {restart}",
            args.problem,
            counters.evaluations.get()
        );
    }
    let time_s = start.elapsed().as_secs_f64();

    let best = best.borrow();
    let value = (problem.value)(&best.genes);
    emit(
        output,
        format!(
            "{{\"library\":\"moors\",\"solver\":\"ga\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{},\"evaluations\":{},\"restarts\":{restart}{},\"best\":{},\"target\":{},\"success\":{},\"solution\":{}}}",
            args.problem,
            args.size,
            args.mode,
            counters.generations.get(),
            counters.evaluations.get(),
            // rule 2.4, continuous problems
            match problem.bounds {
                Some(_) => format!(",\"outside\":{}", counters.outside.get()),
                None => String::new(),
            },
            json_value(&args.problem, value),
            json_value(&args.problem, problem.target),
            problem.reached(value),
            json_solution(&args.problem, &best.genes),
        ),
    );
}

// ---------------------------------------------------------------------------------------------
// Multi-objective runs
// ---------------------------------------------------------------------------------------------

/// The indices of the non-dominated rows of a fitness matrix (minimized)
fn non_dominated(fitness: &Array2<f64>) -> Vec<usize> {
    let rows: Vec<Vec<f64>> = fitness.rows().into_iter().map(|row| row.to_vec()).collect();
    let dominates = |a: &[f64], b: &[f64]| {
        a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
    };
    (0..rows.len())
        .filter(|&i| !rows.iter().any(|b| dominates(b, &rows[i])))
        .collect()
}

fn print_front(
    output: &mut dyn Write,
    args: &Args,
    seed: u64,
    solver: &str,
    counters: &Counters,
    population: &Population<Ix2, Ix2>,
    time_s: f64,
) {
    let (function, _, objectives) = front_problem(&args.problem, args.size);
    // the non-dominated part of the final population; its objectives computed again from the
    // solutions, after the clock and not counted
    let (mut front, mut solutions) = (Vec::new(), Vec::new());
    for i in non_dominated(&population.fitness) {
        let genes = population.genes.row(i).to_vec();
        let mut values = vec![0.0; objectives];
        function(&genes, &mut values);
        front.push(json_reals(&values));
        solutions.push(json_reals(&genes));
    }
    emit(
        output,
        format!(
            "{{\"library\":\"moors\",\"solver\":\"{solver}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{},\"evaluations\":{},\"outside\":{},\"front\":[{}],\"solutions\":[{}]}}",
            args.problem,
            args.size,
            args.mode,
            counters.generations.get(),
            counters.evaluations.get(),
            counters.outside.get(),
            front.join(","),
            solutions.join(","),
        ),
    );
}

/// The matched multi-objective scenarios (README "Scenarios"; rule 6.1: only NSGA-II, NSGA-III,
/// SPEA2, MOEA/D and SMS-EMOA). moors has NSGA-II, NSGA-III and SPEA2 of these; it has no MOEA/D
/// and no SMS-EMOA. Its other algorithms (AGE-MOEA, IBEA, REVEA, R-NSGA-II) aren't run.
/// Differences from the other libraries:
/// - moors' SBX (src/operators/crossover/sbx.rs) crosses every variable (no per-variable
///   probability 0.5) and never exchanges the two children (prob_exchange 0: the first child gets
///   the smaller value of every variable).
/// - polynomial mutation is the adapter's own (above), applied to every child (mutation_rate 1)
///   with probability 1 / n per variable.
/// - no duplicate elimination (NoDuplicatesCleaner, moors' default).
/// - the survivors are evaluated again every generation: P + P evaluations per generation.
///
/// Bounds (rule 2.4): SBX ignores them; moors clamps every offspring to the lower_bound and
/// upper_bound of the constraints function (UnitBounds) before it is evaluated
/// (docs/user_guide/fitness_and_constraints/rust/lower_upper_bounds.md, src/operators/evolve.rs
/// mating_batch), and the polynomial mutation is bounded.
fn run_front(output: &mut dyn Write, args: &Args, seed: u64) {
    let (function, variables, objectives) = front_problem(&args.problem, args.size);
    // (population, reference directions): 100 (99 divisions) with 2 objectives, 92 and 91
    // (12 divisions) with 3
    let (population, directions) = if objectives == 2 {
        (100, 100)
    } else {
        (92, 91)
    };
    let mutation = PolynomialMutation {
        gene_mutation_rate: 1.0 / variables as f64,
        eta: 20.0,
        lower: 0.0,
        upper: 1.0,
    };
    // Das-Dennis directions (the NSGA-III example, docs/user_guide/algorithms/rust/nsga3.md)
    let reference_points = DanAndDenisReferencePoints::new(directions, objectives).generate();
    assert_eq!(reference_points.nrows(), directions);

    // the settings every solver shares; the budget ends the run before the iteration count
    macro_rules! with_common {
        ($builder:expr, $counters:expr, $budget:expr, $eta:expr, $crossover_rate:expr) => {
            $builder
                .sampler(RandomSamplingFloat::new(0.0, 1.0))
                .crossover(SimulatedBinaryCrossover::new($eta))
                .mutation(mutation.clone())
                .fitness_fn(moo_fitness(&$counters, objectives, function))
                .constraints_fn(UnitBounds)
                .num_vars(variables)
                .population_size(population)
                .num_offsprings(population)
                .crossover_rate($crossover_rate)
                .mutation_rate(1.0)
                .controller($budget)
                .num_iterations(args.max_evaluations.max(1))
                .seed(seed)
                .verbose(false)
        };
    }
    // builds and runs one algorithm: its final population
    macro_rules! run {
        ($builder:expr) => {{
            let mut algorithm = $builder.build().expect("valid moors configuration");
            algorithm.run().expect("moors run");
            algorithm.population.expect("population after run")
        }};
    }

    for solver in ["nsga2", "nsga3", "spea2"] {
        let counters = Counters::default();
        let start = Instant::now();
        let budget = Budget {
            counters: counters.clone(),
            max_evaluations: args.max_evaluations,
            max_seconds: args.max_seconds,
            start,
            best: None,
        };
        let result = match solver {
            // population 100 (92), SBX eta 15 at 0.9, PM eta 20 at 1 / n
            "nsga2" => run!(with_common!(
                Nsga2Builder::default(),
                counters,
                budget,
                15.0,
                0.9
            )),
            // SPEA2's population is its archive (docs/user_guide/algorithms/spea2.md,
            // "Implementation in moo-rs"): 100 (92) individuals. Its mating selection and its
            // archive truncation have bugs, not worked around (the page, "Bugs found")
            "spea2" => run!(with_common!(
                Spea2Builder::default(),
                counters,
                budget,
                15.0,
                0.9
            )),
            // 91 (100) reference directions, population 92 (100), SBX eta 30 at 1
            _ => run!(
                with_common!(Nsga3Builder::default(), counters, budget, 30.0, 1.0)
                    .reference_points(reference_points.clone())
                    .are_aspirational(false)
            ),
        };
        let time_s = start.elapsed().as_secs_f64();
        // the multi-objective algorithms have no convergence criterion and moors' iteration
        // count is above the budget, so the controller ends every run (rules 2.2 and 7.1)
        assert!(
            counters.stopped.get(),
            "moors {solver} ended before its budget"
        );
        print_front(output, args, seed, solver, &counters, &result, time_s);
    }
}

// ---------------------------------------------------------------------------------------------
// `values <problem> <size>`: the value (or objectives) of each solution read from stdin
// ---------------------------------------------------------------------------------------------

fn values(problem: &str, size: usize) {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let x: Vec<f64> = line
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .map(|v| v.trim())
            .filter(|v| !v.is_empty())
            .map(|v| match v {
                "true" => 1.0,
                "false" => 0.0,
                v => v.parse().expect("a number"),
            })
            .collect();
        let text = match problem {
            "onemax" | "nqueens" | "rastrigin" | "rosenbrock" | "ackley" => {
                json_value(problem, (SooProblem::of(problem, size).value)(&x))
            }
            _ => {
                let (function, _, objectives) = front_problem(problem, size);
                let mut out = vec![0.0; objectives];
                function(&x, &mut out);
                json_reals(&out)
            }
        };
        writeln!(stdout, "{text}").expect("write value");
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() == 3 && raw[0] == "values" {
        values(&raw[1], raw[2].parse().expect("size"));
        return;
    }
    if raw.len() != 7 {
        eprintln!(
            "usage: ga_bench_moors <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>\n       ga_bench_moors values <problem> <size>"
        );
        std::process::exit(2);
    }
    let args = Args {
        problem: raw[0].clone(),
        size: raw[1].parse().expect("size"),
        mode: raw[2].clone(),
        seed_from: raw[3].parse().expect("seed_from"),
        seed_to: raw[4].parse().expect("seed_to"),
        max_evaluations: raw[5].parse().expect("max_evaluations"),
        max_seconds: raw[6].parse().expect("max_seconds"),
    };
    let mut output = results_output();
    for seed in args.seed_from..=args.seed_to {
        match args.problem.as_str() {
            "onemax" | "nqueens" | "rastrigin" | "rosenbrock" | "ackley" => {
                run_single(&mut *output, &args, seed)
            }
            "zdt1" | "zdt2" | "zdt3" | "dtlz2" | "dtlz1" => run_front(&mut *output, &args, seed),
            other => {
                eprintln!("unknown problem {other}");
                std::process::exit(2);
            }
        }
    }
}

// the fitness functions against values computed in Python with problems.py
#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    }

    #[test]
    fn single_objective() {
        assert_eq!(onemax(&[1.0, 0.0, 1.0, 1.0]), 3.0);
        assert_eq!(onemax_minimized(&[1.0, 0.0, 1.0, 1.0]), -3.0);
        assert_eq!(nqueens(&[3.0, 1.0, 6.0, 2.0, 5.0, 7.0, 4.0, 0.0]), 0.0);
        assert_eq!(nqueens(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 7.0);
        assert_eq!(nqueens(&[5.0, 2.0, 7.0, 0.0, 1.0, 3.0, 6.0, 4.0]), 5.0);
        let x = [-3.656, 3.474, 2.638, -2.449, -0.046];
        assert!(close(rosenbrock(&x), 31225.614287700908));
        // the shifted optima
        let s: Vec<f64> = (0..5).map(shift).collect();
        assert!(rastrigin(&s).abs() < 1e-12);
        assert_eq!(rosenbrock(&[1.0; 5]), 0.0);
        assert!(ackley(&s).abs() < 1e-12);
    }

    #[test]
    fn multi_objective() {
        let x = [
            0.449, 0.652, 0.789, 0.094, 0.028, 0.836, 0.433, 0.762, 0.002, 0.445, 0.722, 0.229,
            0.945, 0.901, 0.031, 0.025, 0.541, 0.939, 0.381, 0.217, 0.422, 0.029, 0.222, 0.438,
            0.496, 0.233, 0.231, 0.219, 0.46, 0.29,
        ];
        let check = |f: Objectives, x: &[f64], m: usize, expected: &[f64]| {
            let mut out = vec![0.0; m];
            f(x, &mut out);
            assert!(
                out.iter().zip(expected).all(|(a, b)| close(*a, *b)),
                "{out:?} != {expected:?}"
            );
        };
        check(zdt1, &x, 2, &[0.449, 3.270875429024877]);
        check(zdt2, &x, 2, &[0.449, 4.685221019573797]);
        check(zdt3, &x, 2, &[0.449, 2.8220969834206637]);
        check(
            dtlz2,
            &x[..12],
            3,
            &[0.8038438063617359, 1.3210517735402911, 1.3165521791315096],
        );
        check(
            dtlz1,
            &x[..7],
            3,
            &[76.18466998901395, 40.66298336836941, 143.392109131221],
        );
        let mut optimum = [0.0; 30];
        optimum[0] = 0.25;
        check(zdt1, &optimum, 2, &[0.25, 0.5]);
        let mut optimum = [0.5; 12];
        optimum[0] = 0.3;
        optimum[1] = 0.6;
        check(
            dtlz2,
            &optimum,
            3,
            &[0.5237204946142994, 0.7208394201673423, 0.45399049973954675],
        );
        check(dtlz1, &optimum[..7], 3, &[0.09, 0.06, 0.35]);
    }

    #[test]
    fn outside_the_bounds() {
        assert!(!is_outside(&[0.0, 0.5, 1.0], (0.0, 1.0)));
        assert!(is_outside(&[0.0, 1.0 + 1e-12], (0.0, 1.0)));
        assert!(is_outside(&[-5.13], (-5.12, 5.12)));
    }

    #[test]
    fn non_dominated_rows() {
        let fitness = ndarray::array![[1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]];
        assert_eq!(non_dominated(&fitness), vec![0, 1, 3]);
    }
}
