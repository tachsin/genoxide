//! Benchmark adapter for moors (https://crates.io/crates/moors), the Rust core of moo-rs.
//!
//! Usage: ga_bench_moors <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//! Prints one JSON line per solver per seed, see ../../README.md for the fields.
//!
//! How moors runs, and what that means for the numbers:
//! - Every algorithm is a GeneticAlgorithm loop with a fixed number of iterations. Each iteration
//!   evaluates the current population and its offspring together (`evaluator.evaluate` on the
//!   concatenation of both), so the survivors are evaluated again every generation: a generation
//!   of population P and P offspring costs 2P evaluations. The fitness function counts every row
//!   it is given, so `evaluations` includes these re-evaluations.
//! - There is no stop condition other than the iteration count, except an AdaptiveController
//!   that is called after every generation and can stop the run: it stops at the evaluation
//!   budget, at the time limit and (single-objective) at the target. Like pymoo's termination and
//!   genoxide's Stop::evaluations, the budget is checked after each generation, so the last
//!   generation can go over it by less than one generation.
//! - moors has no rayon or BLAS dependency (faer without its rayon feature, ndarray without
//!   threading), so the runs are single-threaded.

use moors::{
    AdaptiveController, AgeMoeaBuilder, AlgorithmBuilder, AlgorithmContext, BitFlipMutation,
    CloseDuplicatesCleaner, ControlSignal, ExactDuplicatesCleaner, GaussianMutation, IbeaBuilder,
    MutationOperator, NoConstraints, Nsga2Builder, Nsga3Builder, OrderCrossover,
    PermutationSampling, Population, RandomSamplingBinary, RandomSamplingFloat, ReveaBuilder,
    SimulatedBinaryCrossover, SinglePointBinaryCrossover, Spea2Builder, SwapMutation,
    TwoPointBinaryCrossover, impl_constraints_fn,
    random::RandomGenerator,
    selection::soo::RankSelection,
    survival::moo::{DanAndDenisReferencePoints, StructuredReferencePoints},
    survival::soo::FitnessSurvival,
};
use ndarray::{Array1, Array2, ArrayViewMut1, Ix1, Ix2};
use std::cell::Cell;
use std::f64::consts::{E, PI};
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::time::Instant;

// ---------------------------------------------------------------------------------------------
// Fitness functions, identical to the ones in the other adapters (moors minimizes)
// ---------------------------------------------------------------------------------------------

fn onemax(x: &[f64]) -> f64 {
    // maximize the number of ones: minimize its negative
    -(x.iter().filter(|&&bit| bit > 0.5).count() as f64)
}

/// Number of diagonal conflicts, O(n) (as NQueens.fitness in adapters/pymoo/bench.py)
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

// the variable bounds: moors clamps the offspring to the bounds of the constraints function
// (the documented way, docs/user_guide/fitness_and_constraints); the bound constraints are then
// always satisfied
impl_constraints_fn!(UnitBounds, lower_bound = 0.0, upper_bound = 1.0);
impl_constraints_fn!(RastriginBounds, lower_bound = -5.12, upper_bound = 5.12);
impl_constraints_fn!(RosenbrockBounds, lower_bound = -5.0, upper_bound = 10.0);
impl_constraints_fn!(AckleyBounds, lower_bound = -32.768, upper_bound = 32.768);

// ---------------------------------------------------------------------------------------------
// Polynomial mutation: moors has none, so this implements Deb's bounded polynomial mutation (the
// formula of pymoo's PM) through moors' MutationOperator trait, the documented extension point
// for operators (docs/user_guide/operators/rust/mutation.md)
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// Budget: counts evaluations in the fitness function, stops the run from a controller
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct Counters {
    evaluations: Rc<Cell<usize>>,
    generations: Rc<Cell<usize>>,
}

/// Called by moors after every generation: stops at the evaluations, the time or the target
#[derive(Debug)]
struct Budget {
    counters: Counters,
    max_evaluations: usize,
    max_seconds: f64,
    start: Instant,
    // single-objective: stop when the best fitness (minimized) is at most this
    target: Option<f64>,
}

impl Budget {
    fn exhausted(&self) -> bool {
        self.counters
            .generations
            .set(self.counters.generations.get() + 1);
        self.counters.evaluations.get() >= self.max_evaluations
            || self.start.elapsed().as_secs_f64() >= self.max_seconds
    }
}

impl AdaptiveController<Ix1, Ix2> for Budget {
    fn observe(
        &mut self,
        _: usize,
        population: &Population<Ix1, Ix2>,
        _: &AlgorithmContext,
    ) -> ControlSignal {
        let best = population
            .fitness
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let reached = self.target.is_some_and(|target| best <= target);
        ControlSignal {
            stop: self.exhausted() || reached,
            ..Default::default()
        }
    }
}

impl AdaptiveController<Ix2, Ix2> for Budget {
    fn observe(
        &mut self,
        _: usize,
        _: &Population<Ix2, Ix2>,
        _: &AlgorithmContext,
    ) -> ControlSignal {
        ControlSignal {
            stop: self.exhausted(),
            ..Default::default()
        }
    }
}

fn soo_fitness(
    counters: &Counters,
    f: fn(&[f64]) -> f64,
) -> impl Fn(&Array2<f64>) -> Array1<f64> + 'static {
    let evaluations = counters.evaluations.clone();
    move |genes: &Array2<f64>| {
        evaluations.set(evaluations.get() + genes.nrows());
        genes
            .rows()
            .into_iter()
            .map(|row| match row.as_slice() {
                Some(x) => f(x),
                None => f(&row.to_vec()),
            })
            .collect()
    }
}

fn moo_fitness(
    counters: &Counters,
    objectives: usize,
    f: fn(&[f64], &mut [f64]),
) -> impl Fn(&Array2<f64>) -> Array2<f64> + 'static {
    let evaluations = counters.evaluations.clone();
    move |genes: &Array2<f64>| {
        evaluations.set(evaluations.get() + genes.nrows());
        let mut values = Array2::<f64>::zeros((genes.nrows(), objectives));
        for (row, mut out) in genes.rows().into_iter().zip(values.rows_mut()) {
            let out = out.as_slice_mut().expect("standard layout");
            match row.as_slice() {
                Some(x) => f(x, out),
                None => f(&row.to_vec(), out),
            }
        }
        values
    }
}

// ---------------------------------------------------------------------------------------------
// Runs
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

/// Builds and runs a moors algorithm builder under the budget, timing both (sampling the
/// initial population happens in run()). Returns (final population, seconds).
macro_rules! run_timed {
    ($args:expr, $seed:expr, $counters:expr, $target:expr, $iterations:expr, $builder:expr) => {{
        let start = Instant::now();
        let budget = Budget {
            counters: $counters.clone(),
            max_evaluations: $args.max_evaluations,
            max_seconds: $args.max_seconds,
            start,
            target: $target,
        };
        let mut algorithm = $builder
            .controller(budget)
            .num_iterations($iterations)
            .seed($seed)
            .build()
            .expect("valid moors configuration");
        algorithm.run().expect("moors run");
        let time_s = start.elapsed().as_secs_f64();
        (algorithm.population.expect("population after run"), time_s)
    }};
}

// the iterations of a moors run: the budget ends it through the controller, so this is only an
// upper bound (every generation evaluates at least one individual)
fn iterations(args: &Args) -> usize {
    args.max_evaluations.max(1)
}

struct SooProblem {
    fitness: fn(&[f64]) -> f64,
    // the target in moors' direction (minimized)
    target: f64,
    // from the minimized fitness to the problem's own direction
    to_best: fn(f64) -> f64,
}

fn print_soo(
    output: &mut dyn Write,
    args: &Args,
    seed: u64,
    problem: &SooProblem,
    counters: &Counters,
    fitness: &Array1<f64>,
    time_s: f64,
) {
    let minimum = fitness.iter().copied().fold(f64::INFINITY, f64::min);
    let best = (problem.to_best)(minimum);
    let target = (problem.to_best)(problem.target);
    emit(
        output,
        format!(
            "{{\"library\":\"moors\",\"solver\":\"ga\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{},\"evaluations\":{},\"best\":{best:?},\"target\":{target:?},\"success\":{}}}",
            args.problem,
            args.size,
            args.mode,
            counters.generations.get(),
            counters.evaluations.get(),
            minimum <= problem.target,
        ),
    );
}

fn run_single(output: &mut dyn Write, args: &Args, seed: u64) {
    let size = args.size;
    let problem = match args.problem.as_str() {
        "onemax" => SooProblem {
            fitness: onemax,
            target: -(size as f64),
            to_best: |f| -f,
        },
        "nqueens" => SooProblem {
            fitness: nqueens,
            target: 0.0,
            to_best: |f| f,
        },
        "rastrigin" => SooProblem {
            fitness: rastrigin,
            target: 0.01,
            to_best: |f| f,
        },
        "rosenbrock" => SooProblem {
            fitness: rosenbrock,
            target: 0.01,
            to_best: |f| f,
        },
        "ackley" => SooProblem {
            fitness: ackley,
            target: 0.01,
            to_best: |f| f,
        },
        _ => unreachable!(),
    };
    let counters = Counters::default();
    let fitness = soo_fitness(&counters, problem.fitness);
    let target = Some(problem.target);
    // every single-objective run is moors' GA: AlgorithmBuilder with RankSelection (binary
    // tournament on the fitness rank) and FitnessSurvival (the best of parents + offspring), as
    // in moors' tests/test_ga_soo.rs, the library's single-objective example
    macro_rules! ga {
        () => {
            AlgorithmBuilder::default()
                .selector(RankSelection)
                .survivor(FitnessSurvival)
                .num_vars(size)
                .verbose(false)
        };
    }
    let (population, time_s) = match (args.problem.as_str(), args.mode.as_str()) {
        // as DEAP eaSimple as moors allows: population 300, 300 offspring, two-point crossover
        // with probability 0.5, bit-flip with probability 1 / size per gene on 20% of the
        // children. Differences: moors' tournaments are binary (not 3), and its survival is
        // elitist (the best 300 of parents + offspring, no generational replacement); every
        // generation evaluates the 300 parents again (600 evaluations per generation).
        ("onemax", "matched") => run_timed!(
            args,
            seed,
            counters,
            target,
            iterations(args),
            ga!()
                .sampler(RandomSamplingBinary::new())
                .crossover(TwoPointBinaryCrossover)
                .mutation(BitFlipMutation::new(1.0 / size as f64))
                .fitness_fn(fitness)
                .constraints_fn(NoConstraints)
                .population_size(300)
                .num_offsprings(300)
                .crossover_rate(0.5)
                .mutation_rate(0.2)
        ),
        // the binary example of the moors README (quick start): single-point crossover at 0.9,
        // bit-flip on 10% of the children, exact duplicate removal, population 100 with 32
        // offspring per generation. The README's per-gene bit-flip rate of 0.5 is for its
        // 5-variable knapsack; here it is the usual 1 / size.
        ("onemax", _) => run_timed!(
            args,
            seed,
            counters,
            target,
            iterations(args),
            ga!()
                .sampler(RandomSamplingBinary::new())
                .crossover(SinglePointBinaryCrossover::new())
                .mutation(BitFlipMutation::new(1.0 / size as f64))
                .duplicates_cleaner(ExactDuplicatesCleaner::new())
                .fitness_fn(fitness)
                .constraints_fn(NoConstraints)
                .population_size(100)
                .num_offsprings(32)
                .crossover_rate(0.9)
                .mutation_rate(0.1)
        ),
        // moors has no permutation example; its permutation operators (docs/user_guide/
        // operators: PermutationSampling, OrderCrossover, SwapMutation) with the sizes and rates
        // of its real-valued example (docs/getting_started/rust/real_valued.md: population and
        // offspring 200, crossover 0.9, mutation 0.2) and exact duplicate removal
        ("nqueens", _) => run_timed!(
            args,
            seed,
            counters,
            target,
            iterations(args),
            ga!()
                .sampler(PermutationSampling::new())
                .crossover(OrderCrossover::new())
                .mutation(SwapMutation::new())
                .duplicates_cleaner(ExactDuplicatesCleaner::new())
                .fitness_fn(fitness)
                .constraints_fn(NoConstraints)
                .population_size(200)
                .num_offsprings(200)
                .crossover_rate(0.9)
                .mutation_rate(0.2)
        ),
        // the real-valued example of the moors docs (docs/getting_started/rust/real_valued.md):
        // SBX with eta 15 at 0.9, Gaussian mutation of 10% of the genes with sigma 0.01 on 20% of
        // the children, close-duplicate removal (1e-16), population and offspring 200
        (name, _) => {
            macro_rules! real {
                ($bounds:expr, $low:expr, $high:expr) => {
                    run_timed!(
                        args,
                        seed,
                        counters,
                        target,
                        iterations(args),
                        ga!()
                            .sampler(RandomSamplingFloat::new($low, $high))
                            .crossover(SimulatedBinaryCrossover::new(15.0))
                            .mutation(GaussianMutation::new(0.1, 0.01))
                            .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-16))
                            .fitness_fn(fitness)
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
    };
    print_soo(
        output,
        args,
        seed,
        &problem,
        &counters,
        &population.fitness,
        time_s,
    );
}

/// The non-dominated rows of a fitness matrix (minimized)
fn non_dominated(fitness: &Array2<f64>) -> Vec<Vec<f64>> {
    let rows: Vec<Vec<f64>> = fitness.rows().into_iter().map(|row| row.to_vec()).collect();
    let dominates = |a: &[f64], b: &[f64]| {
        a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
    };
    rows.iter()
        .filter(|a| !rows.iter().any(|b| dominates(b, a)))
        .cloned()
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn print_front(
    output: &mut dyn Write,
    args: &Args,
    seed: u64,
    solver: &str,
    counters: &Counters,
    fitness: &Array2<f64>,
    time_s: f64,
    error: Option<&str>,
) {
    // a failed run: its error message, as a JSON string
    let error = error
        .map(|message| {
            let escaped: String = message
                .chars()
                .map(|c| match c {
                    '"' | '\\' => format!("\\{c}"),
                    c if c.is_control() => " ".to_string(),
                    c => c.to_string(),
                })
                .collect();
            format!(",\"error\":\"moors panicked: {escaped}\"")
        })
        .unwrap_or_default();
    let front: Vec<String> = non_dominated(fitness)
        .iter()
        .map(|values| {
            let values: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
            format!("[{}]", values.join(","))
        })
        .collect();
    emit(
        output,
        format!(
            "{{\"library\":\"moors\",\"solver\":\"{solver}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{},\"evaluations\":{},\"front\":[{}]{error}}}",
            args.problem,
            args.size,
            args.mode,
            counters.generations.get(),
            counters.evaluations.get(),
            front.join(","),
        ),
    );
}

type Objectives = fn(&[f64], &mut [f64]);

/// The matched multi-objective settings. moors has NSGA-II, NSGA-III and SPEA2 (no MOEA/D, no
/// SMS-EMOA); AGE-MOEA, IBEA and REVEA are extra solvers with the NSGA-II operators.
/// Differences from the other libraries:
/// - moors' SBX crosses every variable (no per-variable probability 0.5, no exchange of the two
///   children) and ignores the bounds; the children are clamped to [0, 1] afterwards.
/// - polynomial mutation is the adapter's own (above), applied to every child (mutation_rate 1)
///   with probability 1 / n per variable.
/// - no duplicate elimination (NoDuplicatesCleaner, moors' default).
/// - the survivors are evaluated again every generation: P + P evaluations per generation.
fn run_front(output: &mut dyn Write, args: &Args, seed: u64) {
    // (fitness, variables, objectives, population, reference directions)
    let (function, variables, objectives, population, directions): (
        Objectives,
        usize,
        usize,
        usize,
        usize,
    ) = match args.problem.as_str() {
        "zdt1" => (zdt1, args.size, 2, 100, 100),
        "zdt2" => (zdt2, args.size, 2, 100, 100),
        "zdt3" => (zdt3, args.size, 2, 100, 100),
        // size: the number of objectives, with 10 distance variables (DTLZ2) or 5 (DTLZ1)
        "dtlz2" => (dtlz2, args.size + 9, args.size, 92, 91),
        "dtlz1" => (dtlz1, args.size + 4, args.size, 92, 91),
        _ => unreachable!(),
    };
    let mutation = PolynomialMutation {
        gene_mutation_rate: 1.0 / variables as f64,
        eta: 20.0,
        lower: 0.0,
        upper: 1.0,
    };
    // Das-Dennis: 91 directions (12 divisions) for 3 objectives, 100 (99 divisions) for 2
    let reference_points = DanAndDenisReferencePoints::new(directions, objectives).generate();
    assert_eq!(reference_points.nrows(), directions);

    // the settings every solver shares
    macro_rules! with_common {
        ($builder:expr, $counters:expr, $eta:expr, $crossover_rate:expr) => {
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
                .verbose(false)
        };
    }

    for solver in ["nsga2", "nsga3", "spea2", "age_moea", "ibea", "revea"] {
        let counters = Counters::default();
        let start = Instant::now();
        // moors can panic inside a run (AGE-MOEA asserts a strictly positive central point of
        // the first front, which fails when that front is degenerate, e.g. on ZDT2): such a run
        // is reported with an empty front (hypervolume 0) and its error
        let outcome = catch_unwind(AssertUnwindSafe(|| match solver {
            // population 100 (92), SBX eta 15 at 0.9, PM eta 20 at 1 / n
            "nsga2" => run_timed!(
                args,
                seed,
                counters,
                None,
                iterations(args),
                with_common!(Nsga2Builder::default(), counters, 15.0, 0.9)
            ),
            "spea2" => run_timed!(
                args,
                seed,
                counters,
                None,
                iterations(args),
                with_common!(Spea2Builder::default(), counters, 15.0, 0.9)
            ),
            // 91 (100) reference directions, population 92 (100), SBX eta 30 at 1
            "nsga3" => run_timed!(
                args,
                seed,
                counters,
                None,
                iterations(args),
                with_common!(Nsga3Builder::default(), counters, 30.0, 1.0)
                    .reference_points(reference_points.clone())
                    .are_aspirational(false)
            ),
            // extra solvers, with the NSGA-II settings
            "age_moea" => run_timed!(
                args,
                seed,
                counters,
                None,
                iterations(args),
                with_common!(AgeMoeaBuilder::default(), counters, 15.0, 0.9)
            ),
            // kappa 0.05 and the hypervolume reference point (4, ..., 4) of the moors docs
            // (docs/user_guide/algorithms/rust/ibea.md); moors normalizes the objectives to
            // [0, 1] before the indicator
            "ibea" => run_timed!(
                args,
                seed,
                counters,
                None,
                iterations(args),
                with_common!(IbeaBuilder::default(), counters, 15.0, 0.9)
                    .reference(Array1::from_elem(objectives, 4.0))
                    .kappa(0.05)
            ),
            // alpha 2.5 and reference-vector adaptation frequency 0.2 as in
            // docs/user_guide/algorithms/rust/revea.md, the Das-Dennis directions of NSGA-III.
            // REVEA's angle penalty depends on the planned iteration count, so it gets the
            // number of generations the budget allows at the full population.
            "revea" => {
                let planned = args
                    .max_evaluations
                    .saturating_sub(population)
                    .div_ceil(2 * population)
                    .max(1);
                run_timed!(
                    args,
                    seed,
                    counters,
                    None,
                    planned,
                    with_common!(ReveaBuilder::default(), counters, 15.0, 0.9)
                        .reference_points(reference_points.clone())
                        .alpha(2.5)
                        .frequency(0.2)
                )
            }
            _ => unreachable!(),
        }));
        match outcome {
            Ok((result, time_s)) => print_front(
                output,
                args,
                seed,
                solver,
                &counters,
                &result.fitness,
                time_s,
                None,
            ),
            Err(panic) => {
                let message = panic
                    .downcast_ref::<&str>()
                    .map(|m| m.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown".to_string());
                eprintln!(
                    "moors {solver} panicked on {} seed {seed}: {message}",
                    args.problem
                );
                print_front(
                    output,
                    args,
                    seed,
                    solver,
                    &counters,
                    &Array2::zeros((0, objectives)),
                    start.elapsed().as_secs_f64(),
                    Some(&message),
                );
            }
        }
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() != 7 {
        eprintln!(
            "usage: ga_bench_moors <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>"
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

// the fitness functions against values computed in Python with the formulas of the other adapters
#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    }

    #[test]
    fn single_objective() {
        assert_eq!(onemax(&[1.0, 0.0, 1.0, 1.0]), -3.0);
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
}
