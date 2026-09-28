//! Benchmark adapter for radiate (https://crates.io/crates/radiate).
//!
//! Usage:
//!   ga_bench_radiate <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//!   ga_bench_radiate values <problem> <size>
//! The first prints one JSON line per seed (see ../../README.md, "Adding a library"), the second
//! reads one JSON solution per line and prints its value per line.
//!
//! The suite is matched: each problem has one method, defined in docs/benchmarks/rules.md (rule
//! 6), and every library runs it with its own implementation. radiate, a GA library, runs one: the
//! GA of OneMax 1000 (onemax 1000 matched); any other problem, size or mode prints nothing, since
//! radiate has no DE or CMA-ES. How each setting maps to the definition, and the differences:
//! docs/benchmarks/libraries/radiate.md.
//!
//! radiate has one search method, its `GeneticEngine`: a GA whose generation keeps
//! `population_size * (1 - offspring_fraction)` survivors (survivor selector) and breeds
//! `population_size * offspring_fraction` offspring (offspring selector, then crossover and
//! mutation). Only the individuals whose genome changed are evaluated again. By default it
//! replaces every individual older than `max_age` = 20 generations with a random one
//! (radiate-engines-1.3.2/src/builder/mod.rs, `max_age: 20`; steps/filter.rs).
//!
//! Every run ends only at the target, the budget or the time cap (rule 2.1): the adapter's
//! `until` limit is the engine's only stop criterion, checked after every generation, and radiate
//! has no stop criterion of its own (an engine without a limit runs forever,
//! docs/source/engine/index.md, "Common Pitfalls"), neither a budget nor a convergence test.
//! The engine runs the way radiate's guide calls the cheap one (docs/source/engine/runtime.md and
//! generations.md): `iter().until(closure).last()`, the runtime's own `last()`, which is `run()`,
//! checks the closure against a borrowed `GenerationView` of the live engine and builds a
//! `Generation` (a clone of the population and the metrics) only once, at the end; iterating, or
//! the engine's `run(closure)`, would clone them every generation. The rest is radiate's
//! defaults, which already do no extra work here: no diversity or species step, no Pareto front
//! for one objective, no event subscribers, the serial executor. Its metrics step runs every
//! generation and can't be turned off (radiate-engines-1.3.2/src/builder/mod.rs, `build_audit_step`).
//! Rule 2.2: radiate evaluates only the individuals an alterer changed, so an attempt can go on
//! without evaluating anything; after 10 generations in a row without an evaluation, the `until`
//! limit ends the attempt, and the engine starts again from a new random population, seeded
//! `(seed + 1) * 1_000_000 + restart`, keeping the budget and the best.
//! Single-threaded: radiate is built without its `rayon` feature and no executor is set, so the
//! engine uses its default `Executor::Serial` for the fitness, the species and the events.
//! Seeded: every run is inside `random_provider::scoped_seed(seed, ..)`, which reseeds the
//! thread-local generator radiate draws all its random numbers from (`random_provider::seed`
//! only reseeds the global generator new threads start from, so it can't reseed a second run in
//! the same thread; radiate-core-1.3.2/src/domain/random_provider.rs).
//! Fitness: through `raw_fitness_fn`, radiate's documented way to evaluate the genotype without
//! decoding it (docs/source/fitness.md, "Raw Fitness"). The values are computed in f64 like in the
//! other adapters (radiate keeps them as an f32 `Score`, exact for OneMax's counts). The fitness
//! wrapper counts every call and
//! records the first evaluation whose f64 value reaches the target (`first_hit`), which also ends
//! the run after its generation. The run reports radiate's own best solution (`Generation::value`,
//! the best of the run), with its value recomputed in f64 after the clock.

use radiate::prelude::*;
use std::io::BufRead;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// The fitness function, the same as benchmarks/problems.py
// ---------------------------------------------------------------------------------------------

fn onemax(bits: impl Iterator<Item = bool>) -> f64 {
    bits.filter(|&bit| bit).count() as f64
}

// ---------------------------------------------------------------------------------------------
// The budget: counts the evaluations and records the first one that reaches the target, in the
// fitness function
// ---------------------------------------------------------------------------------------------

/// The first evaluation whose value reaches the target
struct Hit {
    evaluations: usize,
    time_s: f64,
    genes: Vec<f64>,
}

struct Budget {
    evaluations: AtomicUsize,
    first_hit: OnceLock<Hit>,
    target: f64,
    max_evaluations: usize,
    start: Instant,
    deadline: Instant,
    // the evaluations at the end of the last generation, and that generation's (rule 2.3)
    generation_end: AtomicUsize,
    last_generation: AtomicUsize,
    // the generations in a row without an evaluation (rule 2.2)
    idle: AtomicUsize,
}

impl Budget {
    fn new(args: &Args, start: Instant, target: f64) -> Arc<Self> {
        Arc::new(Self {
            evaluations: AtomicUsize::new(0),
            first_hit: OnceLock::new(),
            target,
            max_evaluations: args.max_evaluations,
            start,
            deadline: start + Duration::from_secs_f64(args.max_seconds),
            generation_end: AtomicUsize::new(0),
            last_generation: AtomicUsize::new(0),
            idle: AtomicUsize::new(0),
        })
    }

    /// Counts one evaluation of `value`, and records it if it's the first to reach the target:
    /// its count, the clock and (once per run) a copy of its genes, reported if radiate's own best
    /// doesn't reach the target
    fn record(&self, value: f64, genes: impl FnOnce() -> Vec<f64>) -> f64 {
        let evaluations = self.evaluations.fetch_add(1, Ordering::Relaxed) + 1;
        if self.reaches(value) && self.first_hit.get().is_none() {
            let time_s = self.start.elapsed().as_secs_f64();
            let _ = self.first_hit.set(Hit {
                evaluations,
                time_s,
                genes: genes(),
            });
        }
        value
    }

    /// Whether `value` reaches the target (problems.reached): OneMax is maximized
    fn reaches(&self, value: f64) -> bool {
        value >= self.target
    }

    fn evaluations(&self) -> usize {
        self.evaluations.load(Ordering::Relaxed)
    }

    /// Marks the end of a generation: radiate evaluates only the individuals whose genome changed,
    /// so generations differ in size
    fn end_generation(&self) {
        let evaluations = self.evaluations();
        let end = self.generation_end.swap(evaluations, Ordering::Relaxed);
        self.last_generation
            .store(evaluations - end, Ordering::Relaxed);
        if evaluations == end {
            self.idle.fetch_add(1, Ordering::Relaxed);
        } else {
            self.idle.store(0, Ordering::Relaxed);
        }
    }

    /// Whether the attempt has run `STALL_GENERATIONS` generations in a row without an evaluation
    fn stalled(&self) -> bool {
        self.idle.load(Ordering::Relaxed) >= STALL_GENERATIONS
    }

    /// The evaluations since the start of the last generation (rule 2.3): of a generation cut
    /// short, or of the last one that ended
    fn last_generation(&self) -> usize {
        match self.evaluations() - self.generation_end.load(Ordering::Relaxed) {
            0 => self.last_generation.load(Ordering::Relaxed),
            partial => partial,
        }
    }

    // checked by the engine after every generation: the target (an evaluated solution reached
    // it), the budget or the time
    fn done(&self) -> bool {
        self.first_hit.get().is_some()
            || self.evaluations() >= self.max_evaluations
            || Instant::now() >= self.deadline
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

/// The generations in a row without an evaluation after which an attempt has converged (rule 2.2)
const STALL_GENERATIONS: usize = 10;

/// The seed of an attempt (rule 2.2): the run's seed first, then (seed + 1) * 1_000_000 + restart
fn attempt_seed(seed: u64, restart: u64) -> u64 {
    if restart == 0 {
        seed
    } else {
        (seed + 1) * 1_000_000 + restart
    }
}

/// Builds the engine (the random initial population) and runs it until the budget is done or,
/// with `stall`, the attempt has stalled, checked after every generation, where the generation's
/// end is marked too. Returns the last generation.
fn run_engine<C, T>(
    budget: &Arc<Budget>,
    engine: GeneticEngine<C, T>,
    stall: bool,
) -> Generation<C, T>
where
    C: Chromosome + Clone + PartialEq + 'static,
    T: Clone + Send + Sync + 'static,
{
    let stop = Arc::clone(budget);
    stop.idle.store(0, Ordering::Relaxed);
    engine
        .iter()
        .until(move |_: GenerationView<C, T>| {
            stop.end_generation();
            stop.done() || (stall && stop.stalled())
        })
        .last()
        .expect("radiate engine failed")
}

/// Prints a single-objective run: `best` is the value of `solution` (a JSON array)
#[allow(clippy::too_many_arguments)]
fn print_single(
    args: &Args,
    seed: u64,
    solver: &str,
    budget: &Budget,
    generations: usize,
    restarts: u64,
    time_s: f64,
    best: f64,
    solution: String,
) {
    let first_hit = match budget.first_hit.get() {
        Some(hit) => format!(
            "{{\"evaluations\":{},\"time_s\":{:.6}}}",
            hit.evaluations, hit.time_s
        ),
        None => "null".to_string(),
    };
    let restarts = if restarts > 0 {
        format!(",\"restarts\":{restarts}")
    } else {
        String::new()
    };
    println!(
        "{{\"library\":\"radiate\",\"solver\":\"{solver}\",\"problem\":\"{}\",\"size\":{},\"mode\":\"{}\",\"seed\":{seed},\"time_s\":{time_s:.6},\"generations\":{generations},\"evaluations\":{},\"last_generation\":{}{restarts},\"best\":{best:?},\"target\":{:?},\"success\":{},\"first_hit\":{first_hit},\"solution\":{solution}}}",
        args.problem,
        args.size,
        args.mode,
        budget.evaluations(),
        budget.last_generation(),
        budget.target,
        budget.reaches(best),
    );
}

/// Runs `build` (which gets the budget for its fitness function) with the seed, and prints the
/// run. An attempt that stalls starts again with the next seed of `attempt_seed` (rule 2.2).
/// `report` gives the value and the JSON solution of radiate's best of each attempt
/// (`Generation::value`), after the clock; the run's best is the best of them.
fn run_single<C, T>(
    args: &Args,
    seed: u64,
    solver: &str,
    target: f64,
    build: impl Fn(Arc<Budget>) -> GeneticEngine<C, T>,
    report: impl Fn(&T) -> (f64, String),
) where
    C: Chromosome + Clone + PartialEq + 'static,
    T: Clone + Send + Sync + 'static,
{
    // the clock starts before the engine creates the initial population
    let start = Instant::now();
    let budget = Budget::new(args, start, target);
    let (mut bests, mut generations, mut restart) = (Vec::new(), 0, 0);
    loop {
        let generation = random_provider::scoped_seed(attempt_seed(seed, restart), || {
            run_engine(&budget, build(Arc::clone(&budget)), true)
        });
        generations += generation.index();
        bests.push(generation.value().clone());
        if budget.done() {
            break;
        }
        restart += 1;
    }
    let time_s = start.elapsed().as_secs_f64();
    let (mut best, mut solution) = bests
        .iter()
        .map(report)
        .reduce(|a, b| if b.0 > a.0 { b } else { a })
        .expect("an attempt");
    // if radiate's best doesn't reach the target while an evaluated solution did, that solution
    // is the run's best
    if let Some(hit) = budget.first_hit.get() {
        if !budget.reaches(best) {
            best = budget.target;
            solution = json_integers(hit.genes.iter().map(|&v| v as usize));
        }
    }
    print_single(
        args,
        seed,
        solver,
        &budget,
        generations,
        restart,
        time_s,
        best,
        solution,
    );
}

/// A JSON array of integers
fn json_integers(values: impl Iterator<Item = usize>) -> String {
    let values: Vec<String> = values.map(|v| v.to_string()).collect();
    format!("[{}]", values.join(","))
}

// ---------------------------------------------------------------------------------------------
// OneMax 1000: the GA of rule 6.2 (DEAP's eaSimple) with radiate's own components
// ---------------------------------------------------------------------------------------------

fn run_onemax(args: &Args, seed: u64) {
    let size = args.size;
    let fitness = |budget: Arc<Budget>| {
        move |genotype: &Genotype<BitChromosome>| {
            let genes = genotype[0].as_slice();
            budget.record(onemax(genes.iter().map(|gene| *gene.allele())), || {
                genes
                    .iter()
                    .map(|gene| *gene.allele() as u8 as f64)
                    .collect()
            })
        }
    };
    // as DEAP's eaSimple: population 300, tournament of 3 (with replacement, like selTournament),
    // two-point crossover 0.5, bit flip 1/size on 20% of the children, no elitism. All radiate's
    // own selector and alterers. Differences:
    // - offspring_fraction 1.0: every generation is 300 selected and altered copies, no
    //   survivors, as eaSimple. max_age off: radiate by default replaces individuals older than
    //   20 generations with random ones, eaSimple doesn't.
    // - crossover: radiate visits every child and with probability 0.5 crosses it with a random
    //   other child (both change), DEAP crosses the disjoint pairs (0,1), (2,3), ... with
    //   probability 0.5: the same expected 150 crossovers per generation, but a child can be
    //   crossed more than once. radiate's cut points are drawn from 0..size (a cut at 0 is a
    //   no-op), DEAP's from 1..size.
    // - mutation: radiate has no per-individual mutation probability, so BitFlip flips each bit
    //   with probability 0.2 / size: the same expected 0.2 flipped bits per child, spread over
    //   more children (18% instead of 12.6% mutated).
    // - evaluations: both evaluate only the changed children; DEAP also re-evaluates a child it
    //   chose to mutate when no bit happened to flip.
    let build = |budget: Arc<Budget>| {
        GeneticEngine::builder()
            .codec(BitCodec::vector(size))
            .raw_fitness_fn(fitness(budget))
            .population_size(300)
            .offspring_fraction(1.0)
            .max_age(usize::MAX)
            .offspring_selector(TournamentSelector::new(3))
            .alter(alters!(
                MultiPointCrossover::new(0.5, 2),
                BitFlipMutator::new(0.2 / size as f32)
            ))
            .build()
    };
    let report = |bits: &Vec<bool>| {
        (
            onemax(bits.iter().copied()),
            json_integers(bits.iter().map(|&bit| bit as usize)),
        )
    };
    run_single(args, seed, "ga", size as f64, build, report);
}

// ---------------------------------------------------------------------------------------------
// The `values` command: the adapter's own fitness functions at the given solutions (rule 1.2)
// ---------------------------------------------------------------------------------------------

fn parse_numbers(line: &str) -> Vec<f64> {
    line.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
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

fn values(problem: &str, _size: usize) {
    assert_eq!(problem, "onemax", "radiate runs OneMax only");
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        let x = parse_numbers(&line);
        println!("{:?}", onemax(x.iter().map(|&v| v != 0.0)));
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
            "usage: ga_bench_radiate <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>\n       ga_bench_radiate values <problem> <size>"
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
    // the matched suite's OneMax 1000; anything else prints nothing
    if (args.problem.as_str(), args.size, args.mode.as_str()) != ("onemax", 1000, "matched") {
        return;
    }
    for seed in args.seed_from..=args.seed_to {
        run_onemax(&args, seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitness_value() {
        assert_eq!(onemax([true, false, true].into_iter()), 2.0);
    }

    #[test]
    fn parse_solutions() {
        assert_eq!(parse_numbers("[1, 0,true]"), vec![1.0, 0.0, 1.0]);
        assert_eq!(parse_numbers("[-0.5,1e-7]"), vec![-0.5, 1e-7]);
    }
}
