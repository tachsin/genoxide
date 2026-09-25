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
//! How each problem is solved, where genoxide's docs recommend each method and setting, what was
//! left out and why, and the separate test runs: docs/benchmarks/libraries/genoxide.md. The
//! citations next to each solver below point to genoxide's own docs: README.md, AGENTS.md,
//! examples/ and the rustdoc in src/.
//!
//! The rules (docs/benchmarks/rules.md), as this adapter follows them:
//! - the fitness functions are those of problems.py, written in Rust as genoxide's users write
//!   them: a closure per genome;
//! - every call of a fitness function is counted by the adapter itself (rule 3), and that count
//!   is the reported "evaluations"; genoxide's own `Outcome::evaluations()` must be the same, and
//!   a difference is printed to stderr;
//! - a run ends at the target, the evaluation budget or the time cap only (rule 2.1): every
//!   solver here either runs by itself until a stop condition (GA, PSO, local search) or restarts
//!   by itself when it converges (DE's restarts, CMA-ES with IPOP restarts);
//! - the clock covers building the algorithm, which creates the random initial population, and
//!   the whole run (rule 4.1);
//! - one thread: genoxide's `parallel` feature is off (Cargo.toml), and the engines evaluate
//!   sequentially (rule 4.3);
//! - each seed goes to the algorithm's `.seed(...)`, so a seed repeats a run exactly (rule 5.2).

use genoxide::Objective::Minimize;
use genoxide::multi::{Decomposition, Moead, Nsga3, SmsEmoa, Spea2, das_dennis};
use genoxide::prelude::*;
use std::cell::Cell;
use std::f64::consts::{E, PI};
use std::io::BufRead;
use std::sync::atomic::{AtomicU64, Ordering};
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
// towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
fn shift(i: usize) -> f64 {
    2.0 * ((37 * i + 11) % 101) as f64 / 101.0 - 1.0
}

fn rastrigin(genome: &Reals) -> f64 {
    10.0 * genome.len() as f64
        + genome
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let x = x - shift(i);
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
    let shifted = || genome.iter().enumerate().map(|(i, x)| x - shift(i));
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
    // the budget: the target, the evaluations or the time, whichever comes first (rule 2.1)
    fn stop(&self, target: f64) -> Stop {
        Stop::target(target)
            .or(Stop::evaluations(self.max_evaluations))
            .or(Stop::time(Duration::from_secs_f64(self.max_seconds)))
    }

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

/// Builds an algorithm and runs it with `fitness`, counting every call, and prints the run. The
/// clock covers building the algorithm (its random initial population) and the run.
fn solve<A, G>(
    args: &Args,
    seed: u64,
    solver: &str,
    target: f64,
    build: impl FnOnce() -> Result<A>,
    fitness: fn(&G) -> f64,
) -> Result<()>
where
    A: Algorithm<Genome = G>,
    G: Genome + Json,
{
    let calls = AtomicU64::new(0);
    let counted = |genome: &G| {
        calls.fetch_add(1, Ordering::Relaxed);
        fitness(genome)
    };
    // CMA-ES with IPOP restarts doubles its population at each restart, so its last generation
    // can be larger than the average: the run reports it, for the budget check (rule 2.3). The
    // other solvers' generations don't grow.
    let (evaluated, last_generation) = (Cell::new(0u64), Cell::new(0u64));
    let start = Instant::now();
    let mut engine = Engine::new(build()?, counted).stop_when(args.stop(target));
    if solver == "cma_es" {
        engine = engine.on_generation(|snapshot| {
            let evaluations = snapshot.progress().evaluations();
            last_generation.set(evaluations - evaluated.replace(evaluations));
        });
    }
    let outcome = engine.run()?;
    let time_s = start.elapsed().as_secs_f64();
    let evaluations = calls.load(Ordering::Relaxed);
    compare_counts(args, solver, seed, evaluations, outcome.evaluations());
    let last_generation = if solver == "cma_es" {
        format!(",\"last_generation\":{}", last_generation.get())
    } else {
        String::new()
    };
    let best = outcome.best_fitness().score().unwrap_or(f64::NAN);
    let success = if args.problem == "onemax" {
        best >= target
    } else {
        best <= target
    };
    println!(
        "{{{}{last_generation},\"best\":{best:?},\"target\":{target:?},\"success\":{success},\"solution\":{}}}",
        args.header(solver, seed, time_s, outcome.generations(), evaluations),
        outcome.best_genome().json(),
    );
    Ok(())
}

fn run_onemax(args: &Args, seed: u64) -> Result<()> {
    let size = args.size;
    let target = size as f64;
    if args.mode == "matched" {
        // the matched settings (benchmarks/README.md), as DEAP's eaSimple: population 300,
        // tournament 3, two-point crossover with probability 0.5, bit-flip with probability
        // 1 / size per gene on 20% of the children, no elitism
        let build = || {
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
        return solve(args, seed, "ga", target, build, onemax);
    }
    // idiomatic: genoxide's OneMax, the same in examples/one_max.rs (lines 15-21), README.md's
    // "A first look", AGENTS.md's first program and python/README.md: population 100, tournament
    // 3, uniform crossover, bit-flip at 1 / length per gene; the default rates (crossover 0.9,
    // mutation 1) and scheme (generational, elitism 1)
    let build = || {
        Ga::builder(Binary::new(size)?)
            .population_size(100)
            .select(Tournament::new(3)?)
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::per_gene(1.0 / size as f64)?)
            .seed(seed)
            .build()
    };
    solve(args, seed, "ga", target, build, onemax)
}

fn run_nqueens(args: &Args, seed: u64) -> Result<()> {
    let size = args.size;
    // examples/n_queens.rs (lines 29-37), also AGENTS.md's permutation template and its scheme
    // table ("(μ+λ): mutation-only search (with NoCrossover)"): (20 + 20) with tournament 2, no
    // crossover and swap mutation
    let build = || {
        Ga::builder(Permutation::new(size)?)
            .population_size(20)
            .select(Tournament::new(2)?)
            .crossover(NoCrossover)
            .mutate(SwapMutation::new())
            .scheme(Scheme::MuPlusLambda { lambda: 20 })
            .minimize()
            .seed(seed)
            .build()
    };
    solve(args, seed, "ga", 0.0, build, nqueens)?;

    // the N-Queens example of LocalSearch's rustdoc (src/algorithm/local_search.rs, lines
    // 110-115): hill climbing with swap neighbors, the best of 4 per step, and the default
    // acceptance NotWorse, which moves to equal neighbors across plateaus. AGENTS.md: local search
    // "often beats a GA on permutations"
    let build = || {
        LocalSearch::builder(Permutation::new(size)?)
            .neighbor(SwapMutation::new())
            .neighbors(4)
            .minimize()
            .seed(seed)
            .build()
    };
    solve(args, seed, "local_search", 0.0, build, nqueens)?;

    // tabu search, as python/examples/n_queens.py (N-Queens 64): swap neighbors, 32 per step,
    // tenure 20. AGENTS.md: tabu search walks out of local optima; "use several neighbors"
    let build = || {
        LocalSearch::builder(Permutation::new(size)?)
            .neighbor(SwapMutation::new())
            .neighbors(32)
            .acceptance(Acceptance::Tabu { tenure: 20 })
            .minimize()
            .seed(seed)
            .build()
    };
    solve(args, seed, "tabu_search", 0.0, build, nqueens)
}

const REAL_TARGET: f64 = 0.01;

// the real-valued problems: Rastrigin and Ackley (multimodal), Rosenbrock (unimodal-ish)
fn run_real(args: &Args, seed: u64) -> Result<()> {
    let (bounds, fitness): (std::ops::RangeInclusive<f64>, fn(&Reals) -> f64) =
        match args.problem.as_str() {
            "rastrigin" => (-5.12..=5.12, rastrigin),
            "rosenbrock" => (-5.0..=10.0, rosenbrock),
            "ackley" => (-32.768..=32.768, ackley),
            other => unreachable!("unknown problem {other}"),
        };
    let real = || Real::uniform(args.size, bounds.clone());

    // CMA-ES, AGENTS.md: "the strongest general choice for continuous problems", with the
    // defaults ("nothing needs tuning": population 4 + 3 ln n, initial step 0.3 of each range)
    // and IPOP restarts, "for multimodal functions" (AGENTS.md's CMA-ES template, on Rastrigin;
    // python/README.md; Restarts::Ipop in src/algorithm/cmaes.rs, "like Rastrigin"). Rosenbrock
    // runs with them too: without restarts a converged CMA-ES samples around the same point
    // until the budget, and rule 2.2 asks for the library's restarts
    let cmaes = || {
        Cmaes::builder(real()?)
            .restarts(cmaes::Restarts::Ipop)
            .minimize()
            .seed(seed)
            .build()
    };
    // differential evolution with its defaults, AGENTS.md's DE template (on Rastrigin):
    // current-to-pbest/1 with an archive, SHADE's adaptation, the number of genes + 10
    // individuals, and restarts when the population converges or stalls
    let de = || De::builder(real()?).minimize().seed(seed).build();

    if args.problem == "rosenbrock" {
        solve(args, seed, "cma_es", REAL_TARGET, cmaes, fitness)?;
        solve(args, seed, "de", REAL_TARGET, de, fitness)?;
        // particle swarm, AGENTS.md's PSO template (on Rosenbrock, target 0.01): 40 particles
        // ("20 to 50"), the default global topology and constriction coefficients
        let pso = || {
            Pso::builder(real()?)
                .population_size(40)
                .minimize()
                .seed(seed)
                .build()
        };
        return solve(args, seed, "pso", REAL_TARGET, pso, fitness);
    }

    // the GA of examples/rastrigin.rs (lines 27-36): population 100, tournament 3, uniform
    // crossover, polynomial mutation with η 20 at 1 / length per gene ("the usual rate"), elitism
    // 2; single-threaded like every adapter
    let ga = || {
        Ga::builder(real()?)
            .population_size(100)
            .select(Tournament::new(3)?)
            .crossover(UniformCrossover::new())
            .mutate(PolynomialMutation::per_gene(1.0 / args.size as f64, 20.0)?)
            .scheme(Scheme::Generational { elitism: 2 })
            .minimize()
            .seed(seed)
            .build()
    };
    solve(args, seed, "ga", REAL_TARGET, ga, fitness)?;
    solve(args, seed, "de", REAL_TARGET, de, fitness)?;
    solve(args, seed, "cma_es", REAL_TARGET, cmaes, fitness)
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
    let calls = AtomicU64::new(0);
    let counted = |genome: &Reals| {
        calls.fetch_add(1, Ordering::Relaxed);
        fitness(genome)
    };
    let start = Instant::now();
    let outcome = MultiEngine::new(build()?, counted)
        .stop_when(
            Stop::evaluations(args.max_evaluations)
                .or(Stop::time(Duration::from_secs_f64(args.max_seconds))),
        )
        .run()?;
    let time_s = start.elapsed().as_secs_f64();
    let evaluations = calls.load(Ordering::Relaxed);
    compare_counts(args, solver, seed, evaluations, outcome.evaluations());
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
        "{{{},\"front\":[{}],\"solutions\":[{}]}}",
        args.header(solver, seed, time_s, outcome.generations(), evaluations),
        front.join(","),
        solutions.join(","),
    );
    Ok(())
}

// The matched settings (benchmarks/README.md): NSGA-II, SPEA2 and SMS-EMOA with 100 individuals
// (92 with 3 objectives), SBX with η 15 at 0.9 (their default rate) and polynomial mutation with
// η 20 at 1 / n; NSGA-III with Das-Dennis directions (12 divisions with 3 objectives) and SBX
// with η 30 at 1 (its default rate); MOEA/D with 100 weight vectors (91 with 3 objectives), 20
// neighbors and parents from the neighborhood with probability 0.9 (its defaults), Tchebycheff
// (PBI with θ 5 with 3 objectives), SBX with η 20 at 1 (its default rate). NSGA-III runs only
// with 3 objectives: AGENTS.md presents it "for 3 or more objectives".
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
            .seed(seed)
            .build()
    };
    solve_front(args, seed, "nsga2", nsga2, fitness)?;
    if M > 2 {
        let nsga3 = || {
            Nsga3::builder(real()?, objectives, das_dennis::<M>(divisions))
                .population_size(population)
                .crossover(SimulatedBinaryCrossover::new(30.0)?)
                .mutate(mutation()?)
                .seed(seed)
                .build()
        };
        solve_front(args, seed, "nsga3", nsga3, fitness)?;
    }
    let spea2 = || {
        Spea2::builder(real()?, objectives)
            .population_size(population)
            .crossover(SimulatedBinaryCrossover::new(15.0)?)
            .mutate(mutation()?)
            .seed(seed)
            .build()
    };
    solve_front(args, seed, "spea2", spea2, fitness)?;
    let sms_emoa = || {
        SmsEmoa::builder(real()?, objectives)
            .population_size(population)
            .crossover(SimulatedBinaryCrossover::new(15.0)?)
            .mutate(mutation()?)
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
        let s: Vec<f64> = (0..10).map(shift).collect();
        let x: Vec<f64> = (0..10).map(|i| 0.5 * (i % 7) as f64 - 1.5).collect();
        assert!(rastrigin(&Reals::from(s.clone())).abs() < 1e-12);
        assert!(ackley(&Reals::from(s.clone())).abs() < 1e-12);
        assert!((rastrigin(&Reals::from(x.clone())) - 87.78147018265213).abs() < 1e-9);
        assert!((ackley(&Reals::from(x.clone())) - 5.149902035382837).abs() < 1e-9);
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
