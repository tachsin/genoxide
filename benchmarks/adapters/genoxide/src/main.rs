//! Benchmark adapter for genoxide, the library of this repository.
//!
//! Usage:
//!   ga_bench_genoxide <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//!   ga_bench_genoxide values <problem> <size>
//!
//! The first prints one JSON line per solver per seed, with the best solution (or the final front
//! and its solutions), see ../../README.md for the fields. The second reads one JSON solution per
//! line from stdin and prints its value (or its list of objectives), with the fitness functions
//! below.
//!
//! How each problem is solved, where genoxide's docs recommend each method and where each setting
//! comes from, what was left out and why, and the separate test runs:
//! docs/benchmarks/libraries/genoxide.md. The methods are those genoxide's docs (README.md,
//! AGENTS.md, examples/ and the rustdoc in src/) prefer for the problem type (rule 6.2); they run
//! with their builder's defaults, and a setting without a default takes a standard value from the
//! literature, cited next to it.
//!
//! The rules (docs/benchmarks/rules.md), as this adapter follows them:
//! - the fitness functions are those of problems.py, written in Rust as genoxide's users write
//!   them: a closure per genome;
//! - every call of a fitness function is counted by the adapter itself (rule 3), and that count
//!   is the reported "evaluations"; genoxide's own `Outcome::evaluations()` must be the same, and
//!   a difference is printed to stderr. The counter also records the first evaluation that
//!   reaches the target ("first_hit");
//! - a run ends at the target, the evaluation budget or the time cap only (rule 2.1). On
//!   convergence a method starts again (rule 2.2): DE and CMA-ES (IPOP) with their own restarts;
//!   the others have no convergence criterion but genoxide's stall (10,000 generations without a
//!   genome to evaluate), after which `solve` starts them again with the seed
//!   `(seed + 1) * 1_000_000 + restart`;
//! - every evaluated solution stays inside the bounds, by genoxide's own bound handling, and the
//!   adapter counts any outside them ("outside", rule 2.4);
//! - the clock starts before the algorithm is built, which creates the random initial population,
//!   and stops when the run ends, before any output (rule 4.1);
//! - one thread: genoxide's `parallel` feature is off (Cargo.toml), and the engines evaluate
//!   sequentially (rule 4.3);
//! - each seed goes to the algorithm's `.seed(...)`, so a seed repeats a run exactly (rule 5.2).

use genoxide::Objective::Minimize;
use genoxide::multi::{Decomposition, Moead, Nsga3, SmsEmoa, Spea2, das_dennis};
use genoxide::prelude::*;
use std::cell::Cell;
use std::f64::consts::{E, PI};
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Fitness functions, identical to problems.py
// ---------------------------------------------------------------------------------------------

fn onemax(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

/// Diagonal conflicts of queens at (i, order[i]): for each diagonal, its queens minus one
fn nqueens(genome: &Order) -> f64 {
    let size = genome.len();
    let mut left_diagonal = vec![0usize; 2 * size - 1];
    let mut right_diagonal = vec![0usize; 2 * size - 1];
    for (i, &gene) in genome.iter().enumerate() {
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
// towards 0: gene i is measured from s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), where
// `upper` is the box's upper bound, computed in this order (rule 1.4)
fn shift(i: usize, upper: f64) -> f64 {
    0.8 * upper * (2.0 * ((37 * i + 11) % 101) as f64 / 101.0 - 1.0)
}

const RASTRIGIN_UPPER: f64 = 5.12;
const ACKLEY_UPPER: f64 = 32.768;

fn rastrigin(genome: &Reals) -> f64 {
    10.0 * genome.len() as f64
        + genome
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let x = x - shift(i, RASTRIGIN_UPPER);
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

fn ackley(genome: &Reals) -> f64 {
    let n = genome.len() as f64;
    let shifted = || {
        genome
            .iter()
            .enumerate()
            .map(|(i, x)| x - shift(i, ACKLEY_UPPER))
    };
    let squares = shifted().map(|x| x * x).sum::<f64>() / n;
    let cosines = shifted().map(|x| (2.0 * PI * x).cos()).sum::<f64>() / n;
    -20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + 20.0 + E
}

// The multi-objective problems, all objectives minimized, all variables in [0, 1]

fn zdt_g(x: &[f64]) -> f64 {
    1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64
}

fn zdt1(x: &Reals) -> [f64; 2] {
    let g = zdt_g(x);
    [x[0], g * (1.0 - (x[0] / g).sqrt())]
}

fn zdt2(x: &Reals) -> [f64; 2] {
    let g = zdt_g(x);
    [x[0], g * (1.0 - (x[0] / g).powi(2))]
}

fn zdt3(x: &Reals) -> [f64; 2] {
    let g = zdt_g(x);
    [
        x[0],
        g * (1.0 - (x[0] / g).sqrt() - x[0] / g * (10.0 * PI * x[0]).sin()),
    ]
}

// DTLZ1 and DTLZ2 with M objectives: objective m is the scale times head(v) of each of the first
// M - 1 - m variables, times last(v) of the next one for m > 0
fn dtlz<const M: usize>(
    x: &[f64],
    scale: f64,
    head: impl Fn(f64) -> f64,
    last: impl Fn(f64) -> f64,
) -> [f64; M] {
    let mut values = [0.0; M];
    for (m, value) in values.iter_mut().enumerate() {
        let mut f = scale;
        for &v in &x[..M - 1 - m] {
            f *= head(v);
        }
        if m > 0 {
            f *= last(x[M - 1 - m]);
        }
        *value = f;
    }
    values
}

fn dtlz2<const M: usize>(x: &Reals) -> [f64; M] {
    let g = x[M - 1..].iter().map(|v| (v - 0.5).powi(2)).sum::<f64>();
    dtlz::<M>(
        x,
        1.0 + g,
        |v| (v * PI / 2.0).cos(),
        |v| (v * PI / 2.0).sin(),
    )
}

fn dtlz1<const M: usize>(x: &Reals) -> [f64; M] {
    let tail = &x[M - 1..];
    let g = 100.0
        * (tail.len() as f64
            + tail
                .iter()
                .map(|v| (v - 0.5).powi(2) - (20.0 * PI * (v - 0.5)).cos())
                .sum::<f64>());
    dtlz::<M>(x, 0.5 * (1.0 + g), |v| v, |v| 1.0 - v)
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

impl Json for Order {
    fn json(&self) -> String {
        let genes: Vec<String> = self.iter().map(usize::to_string).collect();
        format!("[{}]", genes.join(","))
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

/// Builds an algorithm with `build(seed)` and runs it with `fitness`, counting every call, every
/// evaluated genome outside the bounds (`inside`) and the first evaluation that reaches the
/// target, and prints the run. The clock starts before the algorithm is built (its random initial
/// population) and stops when the run ends.
///
/// Rule 2.2: genoxide's engine ends a run by itself only with `StopReason::Stalled`, when for
/// 10,000 generations in a row every child was a copy of a parent, so nothing new was evaluated
/// (AGENTS.md, Troubleshooting): a convergence criterion. The method then starts again from a new
/// random start with the seed `(seed + 1) * 1_000_000 + restart`, keeping the best solution and
/// counting every evaluation, until the target, the budget or the time cap. The engine stalls
/// only when its stop conditions all need new evaluations (a target or an evaluation limit), so
/// the time cap is an abort flag, set after the generation that reaches it, not `Stop::time`.
/// CMA-ES (IPOP) and DE restart by themselves on their own convergence criteria, inside one run.
fn solve<A, G>(
    args: &Args,
    seed: u64,
    solver: &str,
    target: f64,
    build: impl Fn(u64) -> Result<A>,
    fitness: fn(&G) -> f64,
    inside: impl Fn(&G) -> bool + Sync,
) -> Result<()>
where
    A: Algorithm<Genome = G>,
    G: Genome + Json,
{
    let maximize = args.problem == "onemax";
    let better = |a: f64, b: f64| if maximize { a > b } else { a < b };
    let reaches = |value: f64| {
        if maximize {
            value >= target
        } else {
            value <= target
        }
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
        let outcome = Engine::new(build(attempt_seed(seed, restart))?, &counted)
            .stop_when(Stop::target(target).or(Stop::evaluations(args.max_evaluations - used)))
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
        let stalled = outcome.stop_reason() == StopReason::Stalled;
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
    if args.problem != "onemax" && args.problem != "nqueens" {
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
    println!(
        "{{{}{extra},\"best\":{value:?},\"target\":{target:?},\"success\":{},\"solution\":{}}}",
        args.header(solver, seed, time_s, generations, evaluations),
        reaches(value),
        best.best_genome().json(),
    );
    Ok(())
}

fn anywhere<G>(_: &G) -> bool {
    true
}

// The standard values from the literature for settings without a default (each is cited on
// docs/benchmarks/libraries/genoxide.md):
// - a GA's population of 100, binary tournament selection, SBX with η 20 and polynomial mutation
//   with η 20 at 1 / n: Deb, Pratap, Agarwal and Meyarivan, "A fast and elitist multiobjective
//   genetic algorithm: NSGA-II", IEEE TEC 2002;
// - bit-flip mutation at 1 / n per gene: Mühlenbein, "How genetic algorithms really work:
//   mutation and hillclimbing", PPSN 1992;
// - swap mutation for permutations;
// - a (15/15, 100)-ES: Bäck and Schwefel, "An overview of evolutionary algorithms for parameter
//   optimization", Evolutionary Computation 1993.
const GA_POPULATION: usize = 100;
const TOURNAMENT: usize = 2;
const ETA: f64 = 20.0;

fn run_onemax(args: &Args, seed: u64) -> Result<()> {
    let size = args.size;
    let target = size as f64;
    if args.mode == "matched" {
        // the matched settings (benchmarks/README.md), as DEAP's eaSimple: population 300,
        // tournament 3 (with replacement), two-point crossover of each consecutive pair with
        // probability 0.5, each child mutated with probability 0.2 by a bit-flip at 1 / size per
        // gene, generational replacement without elitism; genoxide's own components
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
        return solve(args, seed, "ga", target, build, onemax, anywhere);
    }
    // idiomatic: the GA, the one method AGENTS.md presents for binary genomes ("Choosing the
    // pieces"), with the defaults (crossover rate 0.9, mutation rate 1, generational with an
    // elitism of 1), `UniformCrossover`, the first crossover the table lists for binary genomes,
    // and the literature's population 100, binary tournament and bit-flip at 1 / n
    let build = |seed| {
        Ga::builder(Binary::new(size)?)
            .population_size(GA_POPULATION)
            .select(Tournament::new(TOURNAMENT)?)
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::per_gene(1.0 / size as f64)?)
            .seed(seed)
            .build()
    };
    solve(args, seed, "ga", target, build, onemax, anywhere)
}

fn run_nqueens(args: &Args, seed: u64) -> Result<()> {
    let size = args.size;
    // local search, which AGENTS.md prefers on permutations ("It often beats a GA on
    // permutations"), with its defaults: 1 neighbor per step and the acceptance NotWorse (hill
    // climbing across plateaus); the neighbor operator, required, is a swap
    let build = |seed| {
        LocalSearch::builder(Permutation::new(size)?)
            .neighbor(SwapMutation::new())
            .minimize()
            .seed(seed)
            .build()
    };
    solve(args, seed, "local_search", 0.0, build, nqueens, anywhere)?;

    // the GA, which AGENTS.md presents for permutations ("Choosing the pieces"), with the
    // defaults, `OrderCrossover`, the first permutation crossover the table lists, a swap
    // mutation, and the literature's population 100 and binary tournament
    let build = |seed| {
        Ga::builder(Permutation::new(size)?)
            .population_size(GA_POPULATION)
            .select(Tournament::new(TOURNAMENT)?)
            .crossover(OrderCrossover)
            .mutate(SwapMutation::new())
            .minimize()
            .seed(seed)
            .build()
    };
    solve(args, seed, "ga", 0.0, build, nqueens, anywhere)
}

const REAL_TARGET: f64 = 0.01;

// the real-valued problems: Rastrigin and Ackley (multimodal), Rosenbrock (unimodal-ish). Every
// solver keeps its genomes inside the bounds itself (rule 2.4): SBX and polynomial mutation are
// bounded; CMA-ES draws a sample outside the bounds again, up to 100 times, and then clips it; DE
// bounces a gene back between the parent's and the bound; the ES reflects its mutations into the
// bounds.
fn run_real(args: &Args, seed: u64) -> Result<()> {
    let (bounds, fitness): (std::ops::RangeInclusive<f64>, fn(&Reals) -> f64) =
        match args.problem.as_str() {
            "rastrigin" => (-RASTRIGIN_UPPER..=RASTRIGIN_UPPER, rastrigin),
            "rosenbrock" => (-5.0..=10.0, rosenbrock),
            "ackley" => (-ACKLEY_UPPER..=ACKLEY_UPPER, ackley),
            other => unreachable!("unknown problem {other}"),
        };
    let n = args.size;
    let real = || Real::uniform(n, bounds.clone());
    let inside = |genome: &Reals| genome.iter().all(|x| bounds.contains(x));

    // CMA-ES, AGENTS.md: "the strongest general choice for continuous problems", with its
    // defaults ("Nothing needs tuning": 4 + 3 ln n samples, an initial step of 0.3 of each range)
    // and IPOP restarts: "For multimodal functions, add restarts" (AGENTS.md), and the rustdoc of
    // Restarts::Ipop says it "suits multimodal functions with a global structure, like
    // Rastrigin". Rosenbrock runs with them too: rule 2.2 restarts a converged method with the
    // library's restarts
    let cmaes = |seed| {
        Cmaes::builder(real()?)
            .restarts(cmaes::Restarts::Ipop)
            .minimize()
            .seed(seed)
            .build()
    };
    // differential evolution, AGENTS.md: "For continuous problems on `Real` genomes, differential
    // evolution often needs far fewer evaluations than a GA", with the builder's defaults:
    // SHADE's published settings, and genoxide's own restarts when the population converges or
    // stalls
    let de = |seed| De::builder(real()?).minimize().seed(seed).build();

    if args.problem == "rosenbrock" {
        solve(args, seed, "cma_es", REAL_TARGET, cmaes, fitness, inside)?;
        solve(args, seed, "de", REAL_TARGET, de, fitness, inside)?;
        // the evolution strategy, AGENTS.md: "For smooth real-valued problems that need precise
        // answers", with the defaults (intermediate recombination of all parents, comma
        // selection, a step size per gene starting at 0.3 of each range) and the literature's
        // (15, 100)
        let es = |seed| {
            Es::builder(real()?)
                .parents(15)
                .offspring(100)
                .minimize()
                .seed(seed)
                .build()
        };
        return solve(args, seed, "es", REAL_TARGET, es, fitness, inside);
    }

    solve(args, seed, "cma_es", REAL_TARGET, cmaes, fitness, inside)?;
    solve(args, seed, "de", REAL_TARGET, de, fitness, inside)?;
    // the GA, which AGENTS.md presents for continuous functions ("Choosing the pieces"), with the
    // defaults (crossover rate 0.9, mutation rate 1, generational with an elitism of 1), and the
    // literature's population 100, binary tournament, SBX with η 20 and polynomial mutation with
    // η 20 at 1 / n
    let ga = |seed| {
        Ga::builder(real()?)
            .population_size(GA_POPULATION)
            .select(Tournament::new(TOURNAMENT)?)
            .crossover(SimulatedBinaryCrossover::new(ETA)?)
            .mutate(PolynomialMutation::per_gene(1.0 / n as f64, ETA)?)
            .minimize()
            .seed(seed)
            .build()
    };
    solve(args, seed, "ga", REAL_TARGET, ga, fitness, inside)
}

// ---------------------------------------------------------------------------------------------
// Multi-objective runs
// ---------------------------------------------------------------------------------------------

// Builds and runs one multi-objective algorithm with `fitness`, counting every call, and prints
// the non-dominated individuals of its final population (rule 7.2) and their solutions. A run
// has a budget and no target; run.py computes the hypervolume the same way for every library.
fn solve_front<A, const M: usize>(
    args: &Args,
    seed: u64,
    solver: &str,
    build: impl FnOnce() -> Result<A>,
    fitness: fn(&Reals) -> [f64; M],
) -> Result<()>
where
    A: genoxide::multi::MultiObjectiveAlgorithm<M, Genome = Reals>,
{
    // every variable is in [0, 1]: SBX and polynomial mutation keep the genes inside their
    // bounds (rule 2.4), and the adapter counts any evaluated genome outside them
    let (calls, outside) = (AtomicU64::new(0), AtomicU64::new(0));
    let counted = |genome: &Reals| {
        calls.fetch_add(1, Ordering::Relaxed);
        if !genome.iter().all(|x| (0.0..=1.0).contains(x)) {
            outside.fetch_add(1, Ordering::Relaxed);
        }
        fitness(genome)
    };
    // the last generation's evaluations (rule 2.3): the count after each generation, and the
    // generation's size
    let (evaluated, last_generation) = (Cell::new(0u64), Cell::new(0u64));
    let start = Instant::now();
    let outcome = MultiEngine::new(build()?, counted)
        .stop_when(
            Stop::evaluations(args.max_evaluations)
                .or(Stop::time(Duration::from_secs_f64(args.max_seconds))),
        )
        .on_generation(|_| {
            let evaluations = calls.load(Ordering::Relaxed);
            last_generation.set(evaluations - evaluated.replace(evaluations));
        })
        .run()?;
    let time_s = start.elapsed().as_secs_f64();
    let evaluations = calls.load(Ordering::Relaxed);
    compare_counts(args, solver, seed, evaluations, outcome.evaluations());
    // none of these algorithms has a convergence criterion (rule 2.2), and with a time limit the
    // engine doesn't stall: every run ends at its budget or the time cap
    // the non-dominated individuals of the final population, each with its solution
    let (front, solutions): (Vec<String>, Vec<String>) = outcome
        .front()
        .iter()
        .filter_map(|individual| {
            let values = individual.fitness()?.values()?;
            Some((numbers(&values), individual.genome().json()))
        })
        .unzip();
    println!(
        "{{{},\"last_generation\":{},\"outside\":{},\"front\":[{}],\"solutions\":[{}]}}",
        args.header(solver, seed, time_s, outcome.generations(), evaluations),
        last_generation.get(),
        outside.load(Ordering::Relaxed),
        front.join(","),
        solutions.join(","),
    );
    Ok(())
}

// The matched settings (benchmarks/README.md), with genoxide's own operators: NSGA-II, SPEA2 and
// SMS-EMOA with 100 individuals (92 with 3 objectives), SBX with η 15 at 0.9 (their default rate)
// and polynomial mutation with η 20 at 1 / n; NSGA-III with Das-Dennis directions (99 divisions
// with 2 objectives, 12 with 3) and SBX with η 30 at 1 (its default rate); MOEA/D with 100 weight
// vectors (91 with 3 objectives), 20 neighbors and parents from the neighborhood with probability
// 0.9 (its defaults), Tchebycheff (PBI with θ 5 with 3 objectives), SBX with η 20 at 1 (its
// default rate). No duplicate elimination (on by default in genoxide, so turned off here), and
// SMS-EMOA is steady-state: one child per generation.
fn run_front_problem<const M: usize>(
    args: &Args,
    seed: u64,
    variables: usize,
    fitness: fn(&Reals) -> [f64; M],
) -> Result<()> {
    let (population, divisions) = if M == 2 { (100, 99) } else { (92, 12) };
    let real = || Real::uniform(variables, 0.0..=1.0);
    let mutation = || PolynomialMutation::per_gene(1.0 / variables as f64, 20.0);
    let objectives = [Minimize; M];

    let nsga2 = || {
        Nsga2::builder(real()?, objectives)
            .population_size(population)
            .crossover(SimulatedBinaryCrossover::new(15.0)?)
            .mutate(mutation()?)
            .eliminate_duplicates(false)
            .seed(seed)
            .build()
    };
    solve_front(args, seed, "nsga2", nsga2, fitness)?;
    let nsga3 = || {
        Nsga3::builder(real()?, objectives, das_dennis::<M>(divisions))
            .population_size(population)
            .crossover(SimulatedBinaryCrossover::new(30.0)?)
            .mutate(mutation()?)
            .eliminate_duplicates(false)
            .seed(seed)
            .build()
    };
    solve_front(args, seed, "nsga3", nsga3, fitness)?;
    let spea2 = || {
        Spea2::builder(real()?, objectives)
            .population_size(population)
            .crossover(SimulatedBinaryCrossover::new(15.0)?)
            .mutate(mutation()?)
            .eliminate_duplicates(false)
            .seed(seed)
            .build()
    };
    solve_front(args, seed, "spea2", spea2, fitness)?;
    let sms_emoa = || {
        SmsEmoa::builder(real()?, objectives)
            .population_size(population)
            .offspring(1)
            .crossover(SimulatedBinaryCrossover::new(15.0)?)
            .mutate(mutation()?)
            .eliminate_duplicates(false)
            .seed(seed)
            .build()
    };
    solve_front(args, seed, "sms_emoa", sms_emoa, fitness)?;
    let moead = || {
        let decomposition = if M == 2 {
            Decomposition::Tchebycheff
        } else {
            Decomposition::Pbi { theta: 5.0 }
        };
        Moead::builder(real()?, objectives, das_dennis::<M>(divisions))
            .decomposition(decomposition)
            .crossover(SimulatedBinaryCrossover::new(20.0)?)
            .mutate(mutation()?)
            .seed(seed)
            .build()
    };
    solve_front(args, seed, "moead", moead, fitness)
}

fn run_front(args: &Args, seed: u64) -> Result<()> {
    // ZDT: `size` variables; DTLZ: `size` objectives, with k = 10 (DTLZ2) and 5 (DTLZ1) distance
    // variables (problems.py)
    match (args.problem.as_str(), args.size) {
        ("zdt1", n) => run_front_problem(args, seed, n, zdt1),
        ("zdt2", n) => run_front_problem(args, seed, n, zdt2),
        ("zdt3", n) => run_front_problem(args, seed, n, zdt3),
        ("dtlz2", 3) => run_front_problem(args, seed, 3 + 9, dtlz2::<3>),
        ("dtlz1", 3) => run_front_problem(args, seed, 3 + 4, dtlz1::<3>),
        (other, size) => unreachable!("unsupported problem {other} {size}"),
    }
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
        ("nqueens", _) => format!(
            "{:?}",
            nqueens(&Order::new(x.iter().map(|&v| v as usize).collect())?)
        ),
        ("rastrigin", _) => format!("{:?}", rastrigin(&Reals::from(x))),
        ("rosenbrock", _) => format!("{:?}", rosenbrock(&Reals::from(x))),
        ("ackley", _) => format!("{:?}", ackley(&Reals::from(x))),
        ("zdt1", _) => numbers(&zdt1(&Reals::from(x))),
        ("zdt2", _) => numbers(&zdt2(&Reals::from(x))),
        ("zdt3", _) => numbers(&zdt3(&Reals::from(x))),
        ("dtlz2", 3) => numbers(&dtlz2::<3>(&Reals::from(x))),
        ("dtlz1", 3) => numbers(&dtlz1::<3>(&Reals::from(x))),
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
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() == 3 && raw[0] == "values" {
        return values(&raw[1], raw[2].parse().expect("size"));
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
    for seed in args.seed_from..=args.seed_to {
        match args.problem.as_str() {
            "onemax" => run_onemax(&args, seed)?,
            "nqueens" => run_nqueens(&args, seed)?,
            "rastrigin" | "rosenbrock" | "ackley" => run_real(&args, seed)?,
            "zdt1" | "zdt2" | "zdt3" | "dtlz1" | "dtlz2" => run_front(&args, seed)?,
            other => {
                eprintln!("unknown problem {other}");
                std::process::exit(2);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use genoxide::multi::MultiFitnessFunction;
    use genoxide::multi::problems::{Dtlz1, Dtlz2, Zdt1, Zdt2, Zdt3};

    #[test]
    fn shifted_functions() {
        // 0 at the shift, and the values of a Python reference at a fixed point
        let at_shift = |upper| -> Reals { (0..10).map(|i| shift(i, upper)).collect() };
        let x: Vec<f64> = (0..10).map(|i| 0.5 * (i % 7) as f64 - 1.5).collect();
        assert!(rastrigin(&at_shift(RASTRIGIN_UPPER)).abs() < 1e-12);
        assert!(ackley(&at_shift(ACKLEY_UPPER)).abs() < 1e-12);
        assert!((rastrigin(&Reals::from(x.clone())) - 145.90969988928046).abs() < 1e-9);
        assert!((ackley(&Reals::from(x.clone())) - 20.92235706225884).abs() < 1e-9);
    }

    // genoxide's own test problems (multi::problems) agree with the adapter's functions
    #[test]
    fn genoxide_test_problems_agree() {
        let close = |a: &[f64], b: &[f64]| {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0))
        };
        for i in 0..50 {
            let point = |n: usize| -> Reals {
                (0..n)
                    .map(|j| ((i * 7919 + j * 104_729) % 1000) as f64 / 999.0)
                    .collect()
            };
            let x = point(30);
            assert!(close(&zdt1(&x), Zdt1::new(30).evaluate(&x).as_ref()));
            assert!(close(&zdt2(&x), Zdt2::new(30).evaluate(&x).as_ref()));
            assert!(close(&zdt3(&x), Zdt3::new(30).evaluate(&x).as_ref()));
            let x = point(12);
            assert!(close(
                &dtlz2::<3>(&x),
                Dtlz2::<3>::default().evaluate(&x).as_ref()
            ));
            let x = point(7);
            assert!(close(
                &dtlz1::<3>(&x),
                Dtlz1::<3>::default().evaluate(&x).as_ref()
            ));
        }
    }
}
