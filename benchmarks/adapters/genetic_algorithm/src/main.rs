//! Benchmark adapter for the genetic_algorithm crate (https://crates.io/crates/genetic_algorithm).
//!
//! Usage: ga_bench_genetic_algorithm <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//!        ga_bench_genetic_algorithm values <problem> <size>   (JSON solutions on stdin, one value per line)
//! Prints one JSON line per solver per seed, see ../../README.md for the fields.
//!
//! The methods, their settings and where the library recommends them are on the library's page,
//! docs/benchmarks/libraries/genetic_algorithm.md. The citations in the comments below are to the
//! files of the crate 0.27.3 (README.md, AGENTS.md, AGENTS_TEMPLATES.md, examples/).
//!
//! How the runs follow the benchmark rules (docs/benchmarks/rules.md):
//! - Evaluations are counted in the fitness functions, every call (rule 3). The fitness also
//!   records the first evaluation whose value reaches the target (`first_hit`).
//! - A run ends at the target (the strategy's `with_target_fitness_score`), or when the budget or
//!   the time is used up (the abort flag, which Evolve and HillClimb check once per generation).
//! - Rule 2.2: `with_max_stale_generations` detects convergence (generations without
//!   improvement), so it ends an attempt and the method starts again. The library's restart
//!   mechanism is `call_repeatedly(n)` (AGENTS.md, "Choosing a call variant"), but with
//!   `with_rng_seed_from_u64` every repeat gets the same seed and repeats the same run
//!   (strategy/evolve/builder.rs and strategy/hill_climb/builder.rs: every repeat is
//!   `self.clone().try_into()`, whose rng is `SmallRng::seed_from_u64(seed)`). So `restarts`
//!   below does what `call_repeatedly` does, one run after the other until a run is conclusive
//!   (target or abort), keeping the best by the library's fitness score, from a new random start:
//!   attempt 0 with `seed`, restart r with `(seed + 1) * 1_000_000 + r`. No run uses a limit
//!   that's only a budget (`with_max_generations`).
//! - Rule 2.4: the RangeGenotype keeps every gene inside its allele range itself (random values
//!   drawn from the range, mutations clamped to it: genotype/range.rs). The fitness counts the
//!   evaluated solutions outside the bounds, as the library proposed them, and each continuous
//!   run prints the count as `outside`.
//! - The clock starts before the strategy is built and stops when it returns; the reported
//!   values are computed from the best genes after it.
//! - Single-threaded: no `with_par_fitness`, no `call_par_*`. The only other thread is the timer,
//!   which sleeps until the time cap.
//! - Seeded with `with_rng_seed_from_u64`.
//! - Matched OneMax isn't run: the library has no generational replacement (see the page).

use genetic_algorithm::strategy::evolve::prelude::*;
use genetic_algorithm::strategy::hill_climb::prelude::*;
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{Duration, Instant};

/// Shared evaluation budget: counts fitness evaluations, records the first one that reaches the
/// target, and sets the abort flag when the budget (evaluations or seconds) is used up.
#[derive(Clone, Debug)]
struct Budget {
    evaluations: Arc<AtomicUsize>,
    // evaluated solutions outside the problem's bounds (rule 2.4)
    outside: Arc<AtomicUsize>,
    // (evaluations, seconds) at the first evaluation that reaches the target
    first_hit: Arc<OnceLock<(usize, f64)>>,
    max_evaluations: usize,
    abort_flag: Arc<AtomicBool>,
    // the start of the clock
    start: Instant,
    // the evaluations at the end of the last generation, and that generation's (rule 2.3)
    generation_end: Arc<AtomicUsize>,
    last_generation: Arc<AtomicUsize>,
}
impl Budget {
    fn new(max_evaluations: usize, abort_flag: Arc<AtomicBool>) -> Self {
        Self {
            evaluations: Arc::new(AtomicUsize::new(0)),
            outside: Arc::new(AtomicUsize::new(0)),
            first_hit: Arc::new(OnceLock::new()),
            max_evaluations,
            abort_flag,
            start: Instant::now(),
            generation_end: Arc::new(AtomicUsize::new(0)),
            last_generation: Arc::new(AtomicUsize::new(0)),
        }
    }
    /// Marks the end of a generation, from the reporter
    fn end_generation(&self) {
        let evaluations = self.evaluations();
        let end = self.generation_end.swap(evaluations, Ordering::Relaxed);
        self.last_generation.store(evaluations - end, Ordering::Relaxed);
    }
    /// The evaluations since the start of the last generation (rule 2.3): of a generation cut
    /// short, or of the last one that ended
    fn last_generation(&self) -> usize {
        match self.evaluations() - self.generation_end.load(Ordering::Relaxed) {
            0 => self.last_generation.load(Ordering::Relaxed),
            partial => partial,
        }
    }
    /// Counts one evaluation, whose value reaches the target or not
    fn count(&self, reached: bool) {
        let evaluations = self.evaluations.fetch_add(1, Ordering::Relaxed) + 1;
        if reached && self.first_hit.get().is_none() {
            let _ = self
                .first_hit
                .set((evaluations, self.start.elapsed().as_secs_f64()));
        }
        if evaluations >= self.max_evaluations {
            self.abort_flag.store(true, Ordering::Relaxed);
        }
    }
    fn evaluations(&self) -> usize {
        self.evaluations.load(Ordering::Relaxed)
    }
    fn aborted(&self) -> bool {
        self.abort_flag.load(Ordering::Relaxed)
    }
}

/// Sets the abort flag after max_seconds, stopped by dropping the returned sender
fn start_timer(
    abort_flag: Arc<AtomicBool>,
    max_seconds: f64,
) -> (mpsc::Sender<()>, std::thread::JoinHandle<()>) {
    let (sender, receiver) = mpsc::channel::<()>();
    let handle = std::thread::spawn(move || {
        if let Err(mpsc::RecvTimeoutError::Timeout) =
            receiver.recv_timeout(Duration::from_secs_f64(max_seconds))
        {
            abort_flag.store(true, Ordering::Relaxed);
        }
    });
    (sender, handle)
}

/// Marks the generations for `Budget::last_generation` (rule 2.3): the strategies call `on_start`
/// after evaluating the initial population (or chromosome) and `on_generation_complete` after
/// every generation (strategy/evolve.rs and strategy/hill_climb.rs, `call`)
#[derive(Clone)]
struct Generations<G: Genotype> {
    budget: Budget,
    genotype: std::marker::PhantomData<G>,
}
impl<G: Genotype> Generations<G> {
    fn new(budget: &Budget) -> Self {
        Self {
            budget: budget.clone(),
            genotype: std::marker::PhantomData,
        }
    }
}
impl<G: Genotype> StrategyReporter for Generations<G> {
    type Genotype = G;
    fn on_start<S: StrategyState<G>, C: StrategyConfig>(&mut self, _: &G, _: &S, _: &C) {
        self.budget.end_generation();
    }
    fn on_generation_complete<S: StrategyState<G>, C: StrategyConfig>(
        &mut self,
        _: &G,
        _: &S,
        _: &C,
    ) {
        self.budget.end_generation();
    }
}

// ---------------------------------------------------------------------------------------------
// Fitness functions, identical to benchmarks/problems.py
// ---------------------------------------------------------------------------------------------

fn onemax_value(bits: &[bool]) -> usize {
    bits.iter().filter(|&&bit| bit).count()
}

/// Diagonal conflicts of queens at (i, order[i]): for each diagonal, its queens minus one, O(n)
fn nqueens_value(order: &[usize]) -> usize {
    let size = order.len();
    let mut left_diagonal = vec![0usize; 2 * size - 1];
    let mut right_diagonal = vec![0usize; 2 * size - 1];
    for (i, &column) in order.iter().enumerate() {
        left_diagonal[i + column] += 1;
        right_diagonal[size - 1 - i + column] += 1;
    }
    left_diagonal
        .iter()
        .chain(right_diagonal.iter())
        .map(|&count| count.saturating_sub(1))
        .sum()
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

fn rastrigin_value(x: &[f64]) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .zip(RASTRIGIN_SHIFT.iter())
            .map(|(x, s)| {
                let x = x - s;
                x * x - 10.0 * (2.0 * std::f64::consts::PI * x).cos()
            })
            .sum::<f64>()
}

fn rosenbrock_value(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|pair| 100.0 * (pair[1] - pair[0] * pair[0]).powi(2) + (1.0 - pair[0]).powi(2))
        .sum()
}

fn ackley_value(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let shifted = || x.iter().zip(ACKLEY_SHIFT.iter()).map(|(x, s)| x - s);
    let squares = shifted().map(|x| x * x).sum::<f64>() / n;
    let cosines = shifted()
        .map(|x| (2.0 * std::f64::consts::PI * x).cos())
        .sum::<f64>()
        / n;
    -20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + 20.0 + std::f64::consts::E
}

#[derive(Clone, Debug)]
struct OneMax(Budget);
impl Fitness for OneMax {
    type Genotype = BinaryGenotype;
    fn calculate_for_chromosome(
        &mut self,
        chromosome: &FitnessChromosome<Self>,
        _genotype: &FitnessGenotype<Self>,
    ) -> Option<FitnessValue> {
        let value = onemax_value(&chromosome.genes);
        self.0.count(value >= chromosome.genes.len());
        Some(value as FitnessValue)
    }
}

#[derive(Clone, Debug)]
struct NQueens(Budget);
impl Fitness for NQueens {
    // AGENTS.md "Which Genotype?": UniqueGenotype for a permutation (N-Queens, TSP); the
    // examples use u8 alleles for the columns
    type Genotype = UniqueGenotype<u8>;
    fn calculate_for_chromosome(
        &mut self,
        chromosome: &FitnessChromosome<Self>,
        _genotype: &FitnessGenotype<Self>,
    ) -> Option<FitnessValue> {
        let order: Vec<usize> = chromosome.genes.iter().map(|&gene| gene as usize).collect();
        let value = nqueens_value(&order);
        self.0.count(value == 0);
        Some(value as FitnessValue)
    }
}

// FitnessValue is isize, so a real value is scaled by a precision (AGENTS.md "Critical:
// FitnessValue is isize"); 1e-5 as in AGENTS.md and AGENTS_TEMPLATES.md "Continuous Optimization".
// The value is rounded up, not truncated as `fitness_value` does, so the library's target of
// 0.01 / 1e-5 = 1000 units means a value of at most 0.01: truncation would count 0.010009 as
// reached. Only the target test depends on it.
const PRECISION: f64 = 1e-5;
const REAL_TARGET: f64 = 0.01;
fn scaled(value: f64) -> FitnessValue {
    (value / PRECISION).ceil() as FitnessValue
}

#[derive(Clone, Debug)]
struct Real {
    budget: Budget,
    function: fn(&[f64]) -> f64,
    low: f64,
    high: f64,
}
impl Fitness for Real {
    type Genotype = RangeGenotype<f64>;
    fn calculate_for_chromosome(
        &mut self,
        chromosome: &FitnessChromosome<Self>,
        _genotype: &FitnessGenotype<Self>,
    ) -> Option<FitnessValue> {
        if chromosome.genes.iter().any(|&x| x < self.low || x > self.high) {
            self.budget.outside.fetch_add(1, Ordering::Relaxed);
        }
        let value = (self.function)(&chromosome.genes);
        self.budget.count(value <= REAL_TARGET);
        Some(scaled(value))
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

/// The outcome of one run: the best solution found and its value, computed in f64 by the
/// fitness function above (not the library's scaled score)
struct Outcome {
    solver: &'static str,
    generations: usize,
    best: String,
    target: String,
    success: bool,
    solution: String,
    // a continuous problem: the run reports `outside` (rule 2.4)
    bounded: bool,
}

fn print_result(args: &Args, seed: u64, outcome: &Outcome, time_s: f64, budget: &Budget) {
    let outside = if outcome.bounded {
        format!(",\"outside\":{}", budget.outside.load(Ordering::Relaxed))
    } else {
        String::new()
    };
    let first_hit = match budget.first_hit.get() {
        Some((evaluations, time_s)) => {
            format!("{{\"evaluations\":{evaluations},\"time_s\":{time_s:.6}}}")
        }
        None => "null".to_string(),
    };
    println!(
        "{{\"library\":\"genetic_algorithm\",\"solver\":\"{}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{},\"time_s\":{:.6},\"generations\":{},\"evaluations\":{},\"last_generation\":{},\"best\":{},\"target\":{},\"success\":{},\"first_hit\":{},\"solution\":{}{}}}",
        outcome.solver,
        args.problem,
        args.size,
        args.mode,
        seed,
        time_s,
        outcome.generations,
        budget.evaluations(),
        budget.last_generation(),
        outcome.best,
        outcome.target,
        outcome.success,
        first_hit,
        outcome.solution,
        outside,
    );
}

/// Times one run, from before the initial population to the end of `solve`; `outcome` builds the
/// reported values from its result after the clock
fn run<R>(
    args: &Args,
    seed: u64,
    solve: impl FnOnce(&Budget) -> R,
    outcome: impl FnOnce(R) -> Outcome,
) {
    let abort_flag = Arc::new(AtomicBool::new(false));
    let (timer, timer_handle) = start_timer(abort_flag.clone(), args.max_seconds);
    // the clock starts here (Budget::new)
    let budget = Budget::new(args.max_evaluations, abort_flag);
    let result = solve(&budget);
    let time_s = budget.start.elapsed().as_secs_f64();
    drop(timer);
    timer_handle.join().unwrap();
    let outcome = outcome(result);
    print_result(args, seed, &outcome, time_s, &budget);
}

/// The seed of an attempt (rule 2.2): the run's seed first, then (seed + 1) * 1_000_000 + restart
fn attempt_seed(seed: u64, restart: u64) -> u64 {
    if restart == 0 {
        seed
    } else {
        (seed + 1) * 1_000_000 + restart
    }
}

/// `call_repeatedly` with a seed per run (see the top of the file): `once(seed)` runs the
/// strategy and returns (its best genes, their fitness score, whether it is conclusive, its
/// generations). Runs until one is conclusive (the target reached or the run aborted), and
/// returns the best genes by the library's (minimized) score and the generations of all runs.
fn restarts<T>(
    seed: u64,
    mut once: impl FnMut(u64) -> (Option<Vec<T>>, Option<FitnessValue>, bool, usize),
) -> (Vec<T>, usize) {
    let mut best: Option<(Vec<T>, FitnessValue)> = None;
    let mut generations = 0;
    for restart in 0.. {
        let (genes, score, conclusive, run_generations) = once(attempt_seed(seed, restart));
        generations += run_generations;
        if let (Some(genes), Some(score)) = (genes, score) {
            if best.as_ref().is_none_or(|(_, best_score)| score < *best_score) {
                best = Some((genes, score));
            }
        }
        if conclusive {
            break;
        }
    }
    let (genes, _) = best.expect("a run evaluates at least one chromosome");
    (genes, generations)
}

fn json_list<T: std::fmt::Debug>(values: impl Iterator<Item = T>) -> String {
    let values: Vec<String> = values.map(|v| format!("{v:?}")).collect();
    format!("[{}]", values.join(","))
}

fn onemax(args: &Args, seed: u64) {
    // Matched OneMax (DEAP's eaSimple: tournament selection of parents, generational replacement
    // without elitism) isn't run (decision 1 and 2 of the fairness review; the page explains it):
    // Evolve's selection keeps survivors from parents and offspring together, and at a
    // replacement rate of 1.0 it keeps every offspring and selects nothing. Print nothing.
    if args.mode == "matched" {
        return;
    }
    let target = args.size;
    run(
        args,
        seed,
        |budget| {
            let genotype = BinaryGenotype::builder()
                .with_genes_size(args.size)
                .build()
                .unwrap();
            // idiomatic: AGENTS.md "If unsure, start here", for binary genotypes:
            // SelectTournament(0.5, 0.02, 4), CrossoverUniform(0.7, 0.8), MutateSingleGene(0.2);
            // population 100 as in README.md "Quick Usage", which is this problem (100 genes,
            // count the true values, target 100). The only ending condition is the target
            // (with_max_stale_generations isn't set), so a run ends at the target or through the
            // abort flag.
            let evolve = Evolve::builder()
                .with_genotype(genotype)
                .with_fitness(OneMax(budget.clone()))
                .with_target_fitness_score(target as FitnessValue)
                .with_abort_flag(budget.abort_flag.clone())
                .with_rng_seed_from_u64(seed)
                .with_target_population_size(100)
                .with_select(SelectTournament::new(0.5, 0.02, 4))
                .with_crossover(CrossoverUniform::new(0.7, 0.8))
                .with_mutate(MutateSingleGene::new(0.2))
                .with_reporter(Generations::new(budget))
                .call()
                .unwrap();
            (evolve.best_genes(), evolve.state.current_generation)
        },
        |(genes, generations)| {
            let genes = genes.expect("the initial population is evaluated");
            let best = onemax_value(&genes);
            Outcome {
                solver: "evolve",
                generations,
                best: best.to_string(),
                target: target.to_string(),
                success: best >= target,
                solution: json_list(genes.iter().map(|&bit| bit as u8)),
                bounded: false,
            }
        },
    );
}

fn nqueens_genotype(size: usize) -> UniqueGenotype<u8> {
    assert!(size <= 256, "nqueens size must be <= 256 for u8 genes");
    UniqueGenotype::builder()
        .with_allele_list((0..size).map(|v| v as u8).collect())
        .build()
        .unwrap()
}

fn nqueens(args: &Args, seed: u64) {
    // HillClimb, the strategy README.md recommends for permutation problems ("When to use which
    // strategy?"), as examples/hill_climb_nqueens.rs (the 64-queens board): the Stochastic
    // variant (AGENTS.md "Which HillClimb Variant?": large genome, plateau traversal; "Use
    // Stochastic with call_repeatedly for genomes >20 genes"), max_stale_generations(10000) and
    // with_replace_on_equal_fitness(true) ("crucial for this problem"). It ends after 10000
    // generations without improvement, so it restarts, as call_repeatedly would.
    run(
        args,
        seed,
        |budget| {
            restarts(seed, |run_seed| {
                let hill_climb = HillClimb::builder()
                    .with_genotype(nqueens_genotype(args.size))
                    .with_variant(HillClimbVariant::Stochastic)
                    .with_max_stale_generations(10000)
                    .with_fitness(NQueens(budget.clone()))
                    .with_fitness_ordering(FitnessOrdering::Minimize)
                    .with_target_fitness_score(0)
                    .with_replace_on_equal_fitness(true)
                    .with_abort_flag(budget.abort_flag.clone())
                    .with_rng_seed_from_u64(run_seed)
                    .with_reporter(Generations::new(budget))
                    .call()
                    .unwrap();
                let score = hill_climb.best_fitness_score();
                let conclusive = score == Some(0) || budget.aborted();
                (
                    hill_climb.best_genes(),
                    score,
                    conclusive,
                    hill_climb.state.current_generation,
                )
            })
        },
        |(genes, generations)| {
            let order: Vec<usize> = genes.iter().map(|&gene| gene as usize).collect();
            let best = nqueens_value(&order);
            Outcome {
                solver: "hill_climb",
                generations,
                best: best.to_string(),
                target: "0".to_string(),
                success: best == 0,
                solution: json_list(genes.iter()),
                bounded: false,
            }
        },
    );
}

/// (lower bound, upper bound, function) of a real-valued problem
fn real_problem(problem: &str) -> (f64, f64, fn(&[f64]) -> f64) {
    match problem {
        "rastrigin" => (-5.12, 5.12, rastrigin_value),
        "rosenbrock" => (-5.0, 10.0, rosenbrock_value),
        _ => (-32.768, 32.768, ackley_value),
    }
}

/// The reported values of a continuous run, from its best genes (after the clock)
fn real_outcome(
    solver: &'static str,
    function: fn(&[f64]) -> f64,
) -> impl FnOnce((Vec<f64>, usize)) -> Outcome {
    move |(genes, generations)| {
        let best = function(&genes);
        Outcome {
            solver,
            generations,
            best: format!("{best:?}"),
            target: format!("{REAL_TARGET:?}"),
            success: best <= REAL_TARGET,
            solution: json_list(genes.iter()),
            bounded: true,
        }
    }
}

/// Evolve on Rastrigin, Rosenbrock and Ackley
fn real_evolve(args: &Args, seed: u64) {
    let (low, high, function) = real_problem(&args.problem);
    let width = high - low;
    let solve = |budget: &Budget| {
        restarts(seed, |run_seed| {
            // Rule 6.2's order (a stated preference, then the example for the problem type,
            // then the default); the page explains each step.
            // The example: examples/evolve_range_float.rs, the library's Evolve example for a
            // real function: population 100, SelectTournament(0.5, 0.02, 4),
            // MutateMultiGene(2, 0.2), precision 1e-5.
            // Stated preferences, which come first:
            // - its comments call StepScaled(vec![0.1, 0.01, 0.001, 0.0001]) (on its range
            //   0..=1; the same shares of the range here) the "best approach for this problem,
            //   converges fast, but needs low max_stale_generations to trigger next scale",
            //   and give the low value, .with_max_stale_generations(100), commented out next
            //   to the 100_000 it runs with: so StepScaled with 100;
            // - AGENTS.md "Which Crossover?" recommends CrossoverUniform or
            //   CrossoverSinglePoint for a RangeGenotype, where the example has
            //   CrossoverMultiPoint(0.7, 0.8, 9, false): so CrossoverUniform(0.7, 0.8), the
            //   rates of the example and of AGENTS.md's presets.
            // The step advances after max_stale_generations without improvement (AGENTS.md
            // "Scale advancement"), and the attempt ends in the last one.
            let genotype = RangeGenotype::<f64>::builder()
                .with_genes_size(args.size)
                .with_allele_range(low..=high)
                .with_mutation_type(MutationType::StepScaled(
                    [0.1, 0.01, 0.001, 0.0001]
                        .iter()
                        .map(|share| share * width)
                        .collect(),
                ))
                .build()
                .unwrap();
            let evolve = Evolve::builder()
                .with_genotype(genotype)
                .with_target_population_size(100)
                .with_max_stale_generations(100)
                .with_fitness(Real {
                    budget: budget.clone(),
                    function,
                    low,
                    high,
                })
                .with_fitness_ordering(FitnessOrdering::Minimize)
                .with_target_fitness_score(scaled(REAL_TARGET))
                .with_select(SelectTournament::new(0.5, 0.02, 4))
                .with_crossover(CrossoverUniform::new(0.7, 0.8))
                .with_mutate(MutateMultiGene::new(2, 0.2))
                .with_abort_flag(budget.abort_flag.clone())
                .with_rng_seed_from_u64(run_seed)
                .with_reporter(Generations::new(budget))
                .call()
                .unwrap();
            let score = evolve.best_fitness_score();
            let conclusive =
                score.is_some_and(|score| score <= scaled(REAL_TARGET)) || budget.aborted();
            (
                evolve.best_genes(),
                score,
                conclusive,
                evolve.state.current_generation,
            )
        })
    };
    run(args, seed, solve, real_outcome("evolve", function));
}

/// HillClimb on Rosenbrock (continuous, unimodal)
fn real_hill_climb(args: &Args, seed: u64) {
    let (low, high, function) = real_problem(&args.problem);
    let width = high - low;
    let solve = |budget: &Budget| {
        restarts(seed, |run_seed| {
            // HillClimb for a "Convex search space, few local optima" (README.md "When to
            // use which strategy?"), SteepestAscent for a small genome (AGENTS.md "Which
            // HillClimb Variant?"), as examples/hill_climb_range.rs: StepScaled(vec![0.1,
            // 0.01, 0.001, 0.0001, 0.00001]) on the range 0..=1 (here the same steps as
            // shares of the range) and max_stale_generations(1), which moves to the next
            // step after a generation without improvement and ends the run in the last.
            // AGENTS.md "Exact local optimum needed: SteepestAscent + call_repeatedly(n)".
            let genotype = RangeGenotype::<f64>::builder()
                .with_genes_size(args.size)
                .with_allele_range(low..=high)
                .with_mutation_type(MutationType::StepScaled(
                    [0.1, 0.01, 0.001, 0.0001, 0.00001]
                        .iter()
                        .map(|share| share * width)
                        .collect(),
                ))
                .build()
                .unwrap();
            let hill_climb = HillClimb::builder()
                .with_genotype(genotype)
                .with_variant(HillClimbVariant::SteepestAscent)
                .with_max_stale_generations(1)
                .with_fitness(Real {
                    budget: budget.clone(),
                    function,
                    low,
                    high,
                })
                .with_fitness_ordering(FitnessOrdering::Minimize)
                .with_target_fitness_score(scaled(REAL_TARGET))
                .with_abort_flag(budget.abort_flag.clone())
                .with_rng_seed_from_u64(run_seed)
                .with_reporter(Generations::new(budget))
                .call()
                .unwrap();
            let score = hill_climb.best_fitness_score();
            let conclusive =
                score.is_some_and(|score| score <= scaled(REAL_TARGET)) || budget.aborted();
            (
                hill_climb.best_genes(),
                score,
                conclusive,
                hill_climb.state.current_generation,
            )
        })
    };
    run(args, seed, solve, real_outcome("hill_climb", function));
}

// ---------------------------------------------------------------------------------------------
// values: the adapter's fitness functions on the solutions read from stdin (rule 1.2)
// ---------------------------------------------------------------------------------------------

/// One JSON array of numbers per line
fn parse_solution(line: &str) -> Vec<f64> {
    let inner = line.trim().trim_start_matches('[').trim_end_matches(']');
    inner
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| match item {
            "true" => 1.0,
            "false" => 0.0,
            number => number.parse().expect("a number"),
        })
        .collect()
}

fn values(problem: &str) {
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        let x = parse_solution(&line);
        match problem {
            "onemax" => {
                let bits: Vec<bool> = x.iter().map(|&v| v != 0.0).collect();
                println!("{}", onemax_value(&bits));
            }
            "nqueens" => {
                let order: Vec<usize> = x.iter().map(|&v| v as usize).collect();
                println!("{}", nqueens_value(&order));
            }
            "rastrigin" | "rosenbrock" | "ackley" => {
                println!("{:?}", (real_problem(problem).2)(&x));
            }
            other => {
                eprintln!("genetic_algorithm adapter: no values for {other}");
                std::process::exit(2);
            }
        }
    }
}

fn main() {
    // the shifts are computed before any run
    LazyLock::force(&RASTRIGIN_SHIFT);
    LazyLock::force(&ACKLEY_SHIFT);
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() == 3 && raw[0] == "values" {
        values(&raw[1]);
        return;
    }
    if raw.len() != 7 {
        eprintln!("usage: ga_bench_genetic_algorithm <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>");
        eprintln!("       ga_bench_genetic_algorithm values <problem> <size>");
        std::process::exit(2);
    }
    let args = Args {
        problem: raw[0].clone(),
        size: raw[1].parse().unwrap(),
        mode: raw[2].clone(),
        seed_from: raw[3].parse().unwrap(),
        seed_to: raw[4].parse().unwrap(),
        max_evaluations: raw[5].parse().unwrap(),
        max_seconds: raw[6].parse().unwrap(),
    };
    assert!(args.size <= MAX_GENES, "at most {MAX_GENES} genes");
    for seed in args.seed_from..=args.seed_to {
        match args.problem.as_str() {
            "onemax" => onemax(&args, seed),
            "nqueens" => nqueens(&args, seed),
            "rastrigin" | "ackley" => real_evolve(&args, seed),
            "rosenbrock" => {
                real_evolve(&args, seed);
                real_hill_climb(&args, seed);
            }
            // no multi-objective strategy: print nothing
            "zdt1" | "zdt2" | "zdt3" | "dtlz1" | "dtlz2" => {}
            other => {
                eprintln!("unknown problem {}", other);
                std::process::exit(2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitness_values() {
        // 0 at the shift, and the values of the Python reference at a fixed point
        let x: Vec<f64> = (0..10).map(|i| 0.5 * (i % 7) as f64 - 1.5).collect();
        assert!(rastrigin_value(&RASTRIGIN_SHIFT[..10]).abs() < 1e-12);
        assert!(ackley_value(&ACKLEY_SHIFT[..10]).abs() < 1e-12);
        assert!((rastrigin_value(&x) - 145.90969988928046).abs() < 1e-9);
        assert!((ackley_value(&x) - 20.92235706225884).abs() < 1e-9);
        assert_eq!(RASTRIGIN_SHIFT[0], -3.20380198019802);
        assert_eq!(ACKLEY_SHIFT[4], 3.893227722772275);
        assert_eq!(rosenbrock_value(&[1.0; 10]), 0.0);
        assert_eq!(nqueens_value(&[3, 1, 6, 2, 5, 7, 4, 0]), 0);
        assert_eq!(nqueens_value(&[0, 1, 2, 3, 4, 5, 6, 7]), 7);
        assert_eq!(parse_solution("[1, 0.5, -2e-3]"), vec![1.0, 0.5, -0.002]);
    }

    #[test]
    fn scaled_target() {
        assert!(scaled(0.01) <= scaled(REAL_TARGET));
        assert!(scaled(0.010001) > scaled(REAL_TARGET));
    }

    #[test]
    fn seeds() {
        assert_eq!(attempt_seed(0, 0), 0);
        assert_eq!(attempt_seed(3, 0), 3);
        assert_eq!(attempt_seed(0, 1), 1_000_001);
        assert_eq!(attempt_seed(9, 2), 10_000_002);
    }
}
