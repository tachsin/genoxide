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
//!   generation. A single-objective run that ends early by itself is restarted, attempt 0 with
//!   the seed itself and restart r with (seed + 1) * 1_000_000 + r, keeping the best (rule 2.2).
//! - The fitness function records the first evaluation whose value reaches the target
//!   (`first_hit`), which also ends the run after its generation.
//! - moors has no rayon or BLAS dependency (faer without its rayon feature, ndarray without
//!   threading), so the runs are single-threaded.
//! - Not run (decision 1 of the fairness review: matched scenarios use only the library's own
//!   components): matched OneMax (moors' tournaments are binary and its single-objective survival
//!   keeps the best of parents and offspring; it has no tournament of 3 and no generational
//!   replacement) and the multi-objective scenarios (moors has no polynomial mutation). The
//!   adapter prints nothing for them; the `values` command still evaluates them.

use moors::{
    AdaptiveController, AlgorithmBuilder, AlgorithmContext, BitFlipMutation,
    CloseDuplicatesCleaner, ControlSignal, ExactDuplicatesCleaner, GaussianMutation,
    NoConstraints, OrderCrossover, PermutationSampling, Population, RandomSamplingBinary,
    RandomSamplingFloat, SimulatedBinaryCrossover, SinglePointBinaryCrossover, SwapMutation,
    genetic::D12, impl_constraints_fn, selection::soo::RankSelection,
    survival::soo::FitnessSurvival,
};
use ndarray::{Array1, Array2, Ix1};
use std::cell::Cell;
use std::f64::consts::{E, PI};
use std::io::{BufRead, Write};
use std::rc::Rc;
use std::sync::LazyLock;
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

// Rastrigin and Ackley are shifted (rule 1.4): gene i is measured from
// s_i = 0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1), upper the box's upper bound, in this
// order. Computed once, before any run, for up to MAX_GENES genes.
const MAX_GENES: usize = 1024;
fn shifts(upper: f64) -> Vec<f64> {
    (0..MAX_GENES)
        .map(|i| 0.8 * upper * (2.0 * ((37 * i + 11) % 101) as f64 / 101.0 - 1.0))
        .collect()
}
static RASTRIGIN_SHIFT: LazyLock<Vec<f64>> = LazyLock::new(|| shifts(5.12));
static ACKLEY_SHIFT: LazyLock<Vec<f64>> = LazyLock::new(|| shifts(32.768));

fn rastrigin(x: &[f64]) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .zip(RASTRIGIN_SHIFT.iter())
            .map(|(v, s)| {
                let v = v - s;
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
    let shifted = || x.iter().zip(ACKLEY_SHIFT.iter()).map(|(v, s)| v - s);
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
impl_constraints_fn!(RastriginBounds, lower_bound = -5.12, upper_bound = 5.12);
impl_constraints_fn!(RosenbrockBounds, lower_bound = -5.0, upper_bound = 10.0);
impl_constraints_fn!(AckleyBounds, lower_bound = -32.768, upper_bound = 32.768);

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
    // (evaluations, seconds) at the first evaluation whose value reaches the target
    first_hit: Rc<Cell<Option<(usize, f64)>>>,
    // the rows of the latest call of the fitness function (rule 2.3): moors evaluates the initial
    // population in one call and each generation's population and offspring in one call
    // (initialization.rs, ga.rs `next`)
    last_generation: Rc<Cell<usize>>,
}

/// Whether a solution has a gene outside [lower, upper]
fn is_outside(x: &[f64], (lower, upper): (f64, f64)) -> bool {
    x.iter().any(|&v| !(lower..=upper).contains(&v))
}

/// Called by moors after every generation: stops at the target, the evaluations or the time
#[derive(Debug)]
struct Budget {
    counters: Counters,
    max_evaluations: usize,
    max_seconds: f64,
    start: Instant,
}

impl Budget {
    fn signal(&self) -> ControlSignal {
        let counters = &self.counters;
        counters.generations.set(counters.generations.get() + 1);
        let stop = counters.first_hit.get().is_some()
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

/// The fitness function moors calls with a population: one value per row (the signature of
/// docs/user_guide/fitness_and_constraints/rust/fitness.md), counting every row and the rows
/// outside the bounds (continuous problems), and recording the first row whose value reaches
/// `target` (minimized)
fn soo_fitness(
    counters: &Counters,
    start: Instant,
    f: Single,
    target: f64,
    bounds: Option<(f64, f64)>,
) -> impl Fn(&Array2<f64>) -> Array1<f64> + 'static {
    let evaluations = counters.evaluations.clone();
    let outside = counters.outside.clone();
    let first_hit = counters.first_hit.clone();
    let last_generation = counters.last_generation.clone();
    move |genes: &Array2<f64>| {
        last_generation.set(genes.nrows());
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
                evaluations.set(evaluations.get() + 1);
                if value <= target && first_hit.get().is_none() {
                    first_hit.set(Some((evaluations.get(), start.elapsed().as_secs_f64())));
                }
                value
            })
            .collect()
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

/// The seed of an attempt (rule 2.2): the run's seed first, then (seed + 1) * 1_000_000 + restart
fn attempt_seed(seed: u64, restart: u64) -> u64 {
    if restart == 0 {
        seed
    } else {
        (seed + 1) * 1_000_000 + restart
    }
}

fn run_single(output: &mut dyn Write, args: &Args, seed: u64) {
    // matched OneMax isn't run (see the top of the file): print nothing
    if args.mode == "matched" {
        return;
    }
    let size = args.size;
    let problem = SooProblem::of(&args.problem, size);
    let counters = Counters::default();
    // the best of all attempts: its minimized fitness and genes, from moors' final populations
    let mut best: Option<(f64, Vec<f64>)> = None;
    let start = Instant::now();
    // rule 2.2: moors' iteration count is only a budget, so it's set above any budget; its one
    // convergence criterion, no new offspring (EmptyMatingResult), ends an attempt, and moors has no
    // restart mechanism, so the GA starts again from a new random population with a new seed,
    // keeping the best
    let mut restart = 0u64;
    loop {
        let run_seed = attempt_seed(seed, restart);
        let budget = Budget {
            counters: counters.clone(),
            max_evaluations: args.max_evaluations,
            max_seconds: args.max_seconds,
            start,
        };
        let fitness = soo_fitness(
            &counters,
            start,
            problem.minimized,
            problem.minimized_target(),
            problem.bounds,
        );
        // builds and runs one moors GA; every generation evaluates at least one row, so the
        // budget ends the run before this iteration count. The best of its final population
        // (FitnessSurvival keeps the best of parents and offspring, so it's the attempt's best)
        // is kept if it's better than the other attempts'.
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
                let population = algorithm.population.expect("population after run");
                let (row, &fitness) = population
                    .fitness
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.total_cmp(b.1))
                    .expect("a population");
                if best.as_ref().is_none_or(|(best, _)| fitness < *best) {
                    best = Some((fitness, population.genes.row(row).to_vec()));
                }
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
    }
    let time_s = start.elapsed().as_secs_f64();

    let (_, genes) = best.expect("an attempt evaluates its population");
    let value = (problem.value)(&genes);
    let first_hit = match counters.first_hit.get() {
        Some((evaluations, time_s)) => {
            format!("{{\"evaluations\":{evaluations},\"time_s\":{time_s:.6}}}")
        }
        None => "null".to_string(),
    };
    emit(
        output,
        format!(
            "{{\"library\":\"moors\",\"solver\":\"ga\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{},\"evaluations\":{},\"last_generation\":{},\"restarts\":{restart}{},\"best\":{},\"target\":{},\"success\":{},\"first_hit\":{first_hit},\"solution\":{}}}",
            args.problem,
            args.size,
            args.mode,
            counters.generations.get(),
            counters.evaluations.get(),
            counters.last_generation.get(),
            // rule 2.4, continuous problems
            match problem.bounds {
                Some(_) => format!(",\"outside\":{}", counters.outside.get()),
                None => String::new(),
            },
            json_value(&args.problem, value),
            json_value(&args.problem, problem.target),
            problem.reached(value),
            json_solution(&args.problem, &genes),
        ),
    );
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
    // the shifts are computed before any run
    LazyLock::force(&RASTRIGIN_SHIFT);
    LazyLock::force(&ACKLEY_SHIFT);
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
            // matched multi-objective: not run (see the top of the file), print nothing
            "zdt1" | "zdt2" | "zdt3" | "dtlz2" | "dtlz1" => {}
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
        assert!(rastrigin(&RASTRIGIN_SHIFT[..5]).abs() < 1e-12);
        assert_eq!(rosenbrock(&[1.0; 5]), 0.0);
        assert!(ackley(&ACKLEY_SHIFT[..5]).abs() < 1e-12);
        assert_eq!(RASTRIGIN_SHIFT[0], -3.20380198019802);
        assert_eq!(ACKLEY_SHIFT[4], 3.893227722772275);
        let x: Vec<f64> = (0..10).map(|i| 0.5 * (i % 7) as f64 - 1.5).collect();
        assert!(close(rastrigin(&x), 145.90969988928046));
        assert!(close(ackley(&x), 20.92235706225884));
        assert_eq!(attempt_seed(4, 0), 4);
        assert_eq!(attempt_seed(0, 3), 1_000_003);
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
}
