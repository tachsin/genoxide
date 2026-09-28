//! Benchmark adapter for genoxide, the library of this repository.
//!
//! Usage:
//!   ga_bench_genoxide <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//!   ga_bench_genoxide values <problem> <size>
//!
//! The first prints one JSON line per seed, with the best solution, see ../../README.md for the
//! fields. The second reads one JSON solution per line from stdin and prints its value, with the
//! fitness functions below.
//!
//! The suite is matched: each problem has one method, defined in docs/benchmarks/rules.md (rule
//! 6), and every library runs it with its own implementation. genoxide runs the three:
//! - OneMax 1000: the GA of DEAP's eaSimple (`Ga`);
//! - Rastrigin 30: DE/rand/1/bin with F 0.5, CR 0.9 and 100 individuals (`De`);
//! - Rosenbrock 10: CMA-ES with its defaults and no restarts (`Cmaes`).
//!
//! How each setting maps to the definition, and the differences: docs/benchmarks/libraries/genoxide.md.
//! Any other problem, size or mode prints nothing.
//!
//! The rules (docs/benchmarks/rules.md), as this adapter follows them:
//! - the fitness functions are those of problems.py, written in Rust as genoxide's users write
//!   them: a closure per genome;
//! - every call of a fitness function is counted by the adapter itself (rule 3), and that count
//!   is the reported "evaluations"; genoxide's own `Outcome::evaluations()` must be the same, and
//!   a difference is printed to stderr. The counter also records the first evaluation that
//!   reaches the target ("first_hit");
//! - a run ends at the target, the evaluation budget or the time cap only (rule 2.1). No method
//!   has a convergence criterion; an attempt with 10 generations in a row without a genome to
//!   evaluate has stalled (rule 2.2), and `solve` starts it again with the seed
//!   `(seed + 1) * 1_000_000 + restart` (only the GA can stall: DE and CMA-ES evaluate every
//!   trial and sample);
//! - every evaluated solution stays inside the bounds, by genoxide's own bound handling, and the
//!   adapter counts any outside them ("outside", rule 2.4);
//! - the clock starts before the algorithm is built, which creates the random initial population,
//!   and stops when the run ends, before any output (rule 4.1);
//! - one thread: genoxide's `parallel` feature is off (Cargo.toml), and the engines evaluate
//!   sequentially (rule 4.3);
//! - each seed goes to the algorithm's `.seed(...)`, so a seed repeats a run exactly (rule 5.2).
//!
//! `run.py versions` counts the CPU instructions of each run with Callgrind (rule 10). It sets
//! `GENOXIDE_BENCH_SOLVER` to one solver's name, and the adapter runs only that solver; a name
//! no solver has runs none, which counts the startup. Without the variable, every solver runs.

use genoxide::prelude::*;
use std::cell::Cell;
use std::f64::consts::PI;
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Fitness functions, identical to problems.py
// ---------------------------------------------------------------------------------------------

fn onemax(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

// Rastrigin is shifted, so an optimum at the origin can't favour operators that drift towards 0:
// gene i is measured from s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), where `upper` is
// the box's upper bound, computed in this order (rule 1.4). Computed once, before any run, for up
// to MAX_GENES genes.
fn shift(i: usize, upper: f64) -> f64 {
    0.8 * upper * (2.0 * ((37 * i + 11) % 101) as f64 / 101.0 - 1.0)
}

const RASTRIGIN_UPPER: f64 = 5.12;
const MAX_GENES: usize = 1024;
static RASTRIGIN_SHIFT: LazyLock<Vec<f64>> =
    LazyLock::new(|| (0..MAX_GENES).map(|i| shift(i, RASTRIGIN_UPPER)).collect());

fn rastrigin(genome: &Reals) -> f64 {
    10.0 * genome.len() as f64
        + genome
            .iter()
            .zip(RASTRIGIN_SHIFT.iter())
            .map(|(x, s)| {
                let x = x - s;
                x * x - 10.0 * (2.0 * PI * x).cos()
            })
            .sum::<f64>()
}

fn rosenbrock(genome: &Reals) -> f64 {
    genome
        .windows(2)
        .map(|pair| 100.0 * (pair[1] - pair[0] * pair[0]).powi(2) + (1.0 - pair[0]).powi(2))
        .sum()
}

// ---------------------------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------------------------

/// A genome as JSON: bits as 0 and 1, a permutation, reals (Rust's shortest exact form)
trait Json {
    fn json(&self) -> String;
}

impl Json for Bits {
    fn json(&self) -> String {
        let bits: Vec<&str> = self.iter().map(|bit| if bit { "1" } else { "0" }).collect();
        format!("[{}]", bits.join(","))
    }
}

impl Json for Reals {
    fn json(&self) -> String {
        numbers(self)
    }
}

fn numbers(values: &[f64]) -> String {
    let values: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
    format!("[{}]", values.join(","))
}

struct Args {
    problem: String,
    size: usize,
    mode: String,
    seed_from: u64,
    seed_to: u64,
    max_evaluations: u64,
    max_seconds: f64,
}

impl Args {
    fn header(
        &self,
        solver: &str,
        seed: u64,
        time_s: f64,
        generations: u64,
        evaluations: u64,
    ) -> String {
        format!(
            "\"library\":\"genoxide\",\"solver\":\"{solver}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{generations},\"evaluations\":{evaluations}",
            self.problem, self.size, self.mode,
        )
    }
}

// the adapter's own count (rule 3) against genoxide's: they must agree
fn compare_counts(args: &Args, solver: &str, seed: u64, counted: u64, reported: u64) {
    if counted != reported {
        eprintln!(
            "evaluations differ: {} {} {solver} seed {seed}: the adapter counted {counted}, genoxide reports {reported}",
            args.problem, args.size,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Single-objective runs
// ---------------------------------------------------------------------------------------------

/// The seed of attempt `restart` of the run with `seed` (rule 2.2): the seed itself, then
/// `(seed + 1) * 1_000_000 + restart`, so no two runs share a seed
fn attempt_seed(seed: u64, restart: u64) -> u64 {
    if restart == 0 {
        seed
    } else {
        (seed + 1) * 1_000_000 + restart
    }
}

/// Whether `solver` runs: every solver, or only the one `GENOXIDE_BENCH_SOLVER` names
fn selected(solver: &str) -> bool {
    static ONLY: LazyLock<Option<String>> =
        LazyLock::new(|| std::env::var("GENOXIDE_BENCH_SOLVER").ok());
    ONLY.as_deref().is_none_or(|only| only == solver)
}

/// The generations in a row without a genome to evaluate after which an attempt has converged
/// (rule 2.2)
const STALL_GENERATIONS: u64 = 10;

/// A stop condition met after `STALL_GENERATIONS` generations in a row without an evaluation
fn stalled() -> Stop {
    // the evaluations after the last generation, and the generations in a row without a new one
    let state = Mutex::new((0u64, 0u64));
    Stop::custom(move |progress| {
        let mut state = state.lock().expect("the stall state");
        let idle = if progress.evaluations() == state.0 {
            state.1 + 1
        } else {
            0
        };
        *state = (progress.evaluations(), idle);
        idle >= STALL_GENERATIONS
    })
}

/// Builds an algorithm with `build(seed)` and runs it with `fitness`, counting every call, every
/// evaluated genome outside the bounds (`inside`) and the first evaluation that reaches the
/// target, if the scenario has one (Rastrigin has none: it runs a fixed budget), and prints the
/// run. The clock starts before the algorithm is built (its random initial
/// population) and stops when the run ends.
///
/// Rule 2.2: a child identical to a parent inherits its fitness (AGENTS.md, Fitness functions), so
/// a converged GA can run generations without a genome to evaluate. After `STALL_GENERATIONS` of
/// them in a row, a `Stop::custom` condition ends the attempt (`StopReason::Custom`), and the
/// method starts again from a new random start with the seed `(seed + 1) * 1_000_000 + restart`,
/// keeping the best solution and counting every evaluation, until the target, the budget or the
/// time cap. The time cap is an abort flag, set after the generation that reaches it. DE and
/// CMA-ES evaluate every trial and sample, so they never stall, and they have no restarts.
fn solve<A, G>(
    args: &Args,
    seed: u64,
    solver: &str,
    target: Option<f64>,
    build: impl Fn(u64) -> Result<A>,
    fitness: fn(&G) -> f64,
    inside: impl Fn(&G) -> bool + Sync,
) -> Result<()>
where
    A: Algorithm<Genome = G>,
    G: Genome + Json,
{
    if !selected(solver) {
        return Ok(());
    }
    let maximize = args.problem == "onemax";
    let better = |a: f64, b: f64| if maximize { a > b } else { a < b };
    let reaches = |value: f64| match target {
        Some(target) if maximize => value >= target,
        Some(target) => value <= target,
        // a fixed budget: nothing ends the run early
        None => false,
    };
    let (calls, outside) = (AtomicU64::new(0), AtomicU64::new(0));
    // the first evaluation whose value reaches the target: its number and the clock then
    let first_hit = OnceLock::<(u64, f64)>::new();
    let cap = Duration::from_secs_f64(args.max_seconds);
    let abort = Arc::new(AtomicBool::new(false));
    // the last generation's evaluations, for the budget check (rule 2.3): the count after each
    // generation (the engine calls on_generation after every generation, each attempt's initial
    // population included, and stops only after one) and the generation's size
    let (evaluated, last_generation) = (Cell::new(0u64), Cell::new(0u64));
    let start = Instant::now();
    let counted = |genome: &G| {
        let evaluation = calls.fetch_add(1, Ordering::Relaxed) + 1;
        if !inside(genome) {
            outside.fetch_add(1, Ordering::Relaxed);
        }
        let value = fitness(genome);
        if reaches(value) && first_hit.get().is_none() {
            let _ = first_hit.set((evaluation, start.elapsed().as_secs_f64()));
        }
        value
    };
    let mut best: Option<Outcome<G>> = None;
    let (mut generations, mut reported, mut restart) = (0, 0, 0);
    loop {
        let used = calls.load(Ordering::Relaxed);
        let budget = Stop::evaluations(args.max_evaluations - used).or(stalled());
        let stop = match target {
            Some(target) => Stop::target(target).or(budget),
            None => budget,
        };
        let outcome = Engine::new(build(attempt_seed(seed, restart))?, &counted)
            .stop_when(stop)
            .abort_flag(Arc::clone(&abort))
            .on_generation(|_| {
                let evaluations = calls.load(Ordering::Relaxed);
                last_generation.set(evaluations - evaluated.replace(evaluations));
                if start.elapsed() >= cap {
                    abort.store(true, Ordering::Relaxed);
                }
            })
            .run()?;
        generations += outcome.generations();
        reported += outcome.evaluations();
        let stalled = outcome.stop_reason() == StopReason::Custom;
        let score = |outcome: &Outcome<G>| outcome.best_fitness().score().unwrap_or(f64::NAN);
        if best
            .as_ref()
            .is_none_or(|best| better(score(&outcome), score(best)))
        {
            best = Some(outcome);
        }
        if !stalled
            || calls.load(Ordering::Relaxed) >= args.max_evaluations
            || start.elapsed() >= cap
        {
            break;
        }
        restart += 1;
    }
    let time_s = start.elapsed().as_secs_f64();
    let best = best.expect("a run");
    let evaluations = calls.load(Ordering::Relaxed);
    compare_counts(args, solver, seed, evaluations, reported);
    let mut extra = format!(",\"last_generation\":{}", last_generation.get());
    if args.problem != "onemax" {
        extra += &format!(",\"outside\":{}", outside.load(Ordering::Relaxed));
    }
    if restart > 0 {
        extra += &format!(",\"restarts\":{restart}");
    }
    extra += &match first_hit.get() {
        Some((evaluations, time_s)) => {
            format!(",\"first_hit\":{{\"evaluations\":{evaluations},\"time_s\":{time_s:.6}}}")
        }
        None => ",\"first_hit\":null".to_string(),
    };
    let value = best.best_fitness().score().unwrap_or(f64::NAN);
    let target = target.map_or("null".to_string(), |target| format!("{target:?}"));
    println!(
        "{{{}{extra},\"best\":{value:?},\"target\":{target},\"success\":{},\"solution\":{}}}",
        args.header(solver, seed, time_s, generations, evaluations),
        reaches(value),
        best.best_genome().json(),
    );
    Ok(())
}

fn anywhere<G>(_: &G) -> bool {
    true
}

/// Whether every gene is in `bounds` (rule 2.4): one pass without a branch per gene, so the check
/// costs little next to the fitness function
fn within(genome: &Reals, bounds: &std::ops::RangeInclusive<f64>) -> bool {
    let (low, high) = (*bounds.start(), *bounds.end());
    genome
        .iter()
        .fold(true, |inside, &x| inside & (low <= x) & (x <= high))
}

// OneMax 1000: the GA of rule 6.2, DEAP's eaSimple (docs/benchmarks/rules.md): population 300,
// tournament 3 (with replacement), two-point crossover of each consecutive pair with probability
// 0.5, each child mutated with probability 0.2 by a bit-flip at 1 / size per gene, generational
// replacement without elitism; genoxide's own components
fn run_onemax(args: &Args, seed: u64) -> Result<()> {
    let size = args.size;
    let build = |seed| {
        Ga::builder(Binary::new(size)?)
            .population_size(300)
            .select(Tournament::new(3)?)
            .crossover(PointCrossover::two_point())
            .crossover_rate(0.5)
            .mutate(BitFlip::per_gene(1.0 / size as f64)?)
            .mutation_rate(0.2)
            .scheme(Scheme::Generational { elitism: 0 })
            .seed(seed)
            .build()
    };
    solve(args, seed, "ga", Some(size as f64), build, onemax, anywhere)
}

const REAL_TARGET: f64 = 0.01;

// Rastrigin 30: DE/rand/1/bin of rule 6.3, with no target: every run uses the fixed budget. `De`
// with the strategy DE/rand/1, a fixed F 0.5 and
// CR 0.9 (genoxide's `Control::default()`, set explicitly), 100 individuals and no restarts. It
// starts from a uniform random population, crosses binomially with a forced gene, replaces a
// target by a trial that isn't worse, generation by generation, and sets a trial gene outside the
// box halfway between the target's gene and the bound (rule 2.4)
fn run_rastrigin(args: &Args, seed: u64) -> Result<()> {
    let bounds = -RASTRIGIN_UPPER..=RASTRIGIN_UPPER;
    let inside = |genome: &Reals| within(genome, &bounds);
    let build = |seed| {
        De::builder(Real::uniform(args.size, bounds.clone())?)
            .population_size(100)
            .strategy(de::Strategy::Rand1)
            .control(de::Control::Fixed { f: 0.5, cr: 0.9 })
            .restarts(de::Restarts::Never)
            .minimize()
            .seed(seed)
            .build()
    };
    solve(args, seed, "de", None, build, rastrigin, inside)
}

// Rosenbrock 10: CMA-ES of rule 6.4, `Cmaes` with its defaults (4 + 3 ln n samples, the best half
// recombined with positive log weights, Hansen's 2016 learning rates, a mean drawn uniformly in
// the box, an initial step of 0.3 of each range) and no restarts (`Restarts::Never`, the default,
// set explicitly). A sample outside the box is drawn again, up to 100 times, and then clipped
// (rule 2.4)
fn run_rosenbrock(args: &Args, seed: u64) -> Result<()> {
    let bounds = -5.0..=10.0;
    let inside = |genome: &Reals| within(genome, &bounds);
    let build = |seed| {
        Cmaes::builder(Real::uniform(args.size, bounds.clone())?)
            .restarts(cmaes::Restarts::Never)
            .minimize()
            .seed(seed)
            .build()
    };
    solve(
        args,
        seed,
        "cma_es",
        Some(REAL_TARGET),
        build,
        rosenbrock,
        inside,
    )
}

// ---------------------------------------------------------------------------------------------
// The values command (rule 1.2)
// ---------------------------------------------------------------------------------------------

// a JSON array of numbers, e.g. [1, 0, 1] or [0.25, -1e-05]
fn parse_numbers(line: &str) -> Vec<f64> {
    let inner = line.trim().trim_start_matches('[').trim_end_matches(']');
    if inner.trim().is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|value| {
            let value = value.trim();
            match value {
                "true" => 1.0,
                "false" => 0.0,
                _ => value.parse().expect("a number"),
            }
        })
        .collect()
}

fn value(problem: &str, size: usize, x: Vec<f64>) -> Result<String> {
    Ok(match (problem, size) {
        ("onemax", _) => format!("{:?}", onemax(&x.iter().map(|&v| v != 0.0).collect())),
        ("rastrigin", _) => format!("{:?}", rastrigin(&Reals::from(x))),
        ("rosenbrock", _) => format!("{:?}", rosenbrock(&Reals::from(x))),
        (other, size) => panic!("unsupported problem {other} {size}"),
    })
}

fn values(problem: &str, size: usize) -> Result<()> {
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        println!("{}", value(problem, size, parse_numbers(&line))?);
    }
    Ok(())
}

fn main() -> Result<()> {
    // the shift is computed before any run
    LazyLock::force(&RASTRIGIN_SHIFT);
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() == 3 && raw[0] == "values" {
        let size = raw[2].parse().expect("size");
        assert!(size <= MAX_GENES, "at most {MAX_GENES} genes");
        return values(&raw[1], size);
    }
    if raw.len() != 7 {
        eprintln!(
            "usage: ga_bench_genoxide <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>\n       ga_bench_genoxide values <problem> <size>"
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
    assert!(args.size <= MAX_GENES, "at most {MAX_GENES} genes");
    // the matched suite's three scenarios; anything else prints nothing
    let run = match (args.problem.as_str(), args.size, args.mode.as_str()) {
        ("onemax", 1000, "matched") => run_onemax,
        ("rastrigin", 30, "matched") => run_rastrigin,
        ("rosenbrock", 10, "matched") => run_rosenbrock,
        _ => return Ok(()),
    };
    for seed in args.seed_from..=args.seed_to {
        run(&args, seed)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_functions() {
        // 0 at the shift, and the values of a Python reference at a fixed point
        let at_shift: Reals = (0..10).map(|i| shift(i, RASTRIGIN_UPPER)).collect();
        let x: Vec<f64> = (0..10).map(|i| 0.5 * (i % 7) as f64 - 1.5).collect();
        assert!(rastrigin(&at_shift).abs() < 1e-12);
        // the precomputed shift is the formula's values, bit for bit
        assert_eq!(RASTRIGIN_SHIFT[0], -3.20380198019802);
        assert!((rastrigin(&Reals::from(x)) - 145.90969988928046).abs() < 1e-9);
    }
}
