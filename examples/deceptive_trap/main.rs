//! Deceptive trap: maximize 10 blocks of Deb and Goldberg's trap function of 4 bits, which leads
//! each block away from its optimum, with a genetic algorithm whose two-point crossover keeps
//! the blocks together.
//!
//! A block of 4 bits scores 3 − u for u ones below 4, and 4 with all of them: fully deceptive,
//! every schema of order below 4 favoring all zeros. A run from seed 1 prints each generation;
//! then runs from seeds 1 to 20 count how often the GA reaches the optimum, 40. As contrasts: the
//! same GA with uniform crossover, which breaks the blocks apart, and hill climbing, one bit at a
//! time, which climbs to the deceptive attractor. The function is genoxide's
//! `problems::binary::Trap`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example deceptive_trap
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::binary::Trap;

const BLOCKS: usize = 10;
const K: usize = 4;
const BITS: usize = BLOCKS * K;
const SEEDS: u64 = 20;
const GENERATIONS: u64 = 300;
// the evaluations of a hill climb
const CLIMB: u64 = 10_000;

// the genetic algorithm with `crossover`, from `seed`
fn ga<C: Crossover<Binary>>(
    problem: &Trap,
    crossover: C,
    seed: u64,
) -> Result<Ga<Binary, Tournament, C, BitFlip>> {
    Ga::builder(problem.representation())
        .population_size(1000)
        .select(Tournament::new(4)?)
        .crossover(crossover)
        .mutate(BitFlip::per_gene(1.0 / BITS as f64)?)
        .seed(seed)
        .build()
}

// the blocks of `genome` that are all ones
fn blocks_of_ones(genome: &Bits) -> usize {
    (0..BLOCKS)
        .filter(|block| (block * K..(block + 1) * K).all(|i| genome.get(i) == Some(true)))
        .count()
}

// the median of `values`
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

fn main() -> Result<()> {
    let problem = Trap::new(BLOCKS, K);
    let optimum = problem.optimum().expect("known").value();
    println!("Deceptive trap: {BLOCKS} blocks of {K} bits, maximum {optimum}");
    println!("a GA with two-point crossover, from seed 1");
    println!("generation  best  median  blocks of ones in the best");
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(BITS, optimum);
    let outcome = Engine::new(ga(&problem, PointCrossover::two_point(), 1)?, problem)
        .stop_when(Stop::target(optimum).or(Stop::generations(GENERATIONS)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            let scores = snapshot.population().iter();
            let scores = scores.filter_map(|individual| individual.fitness()?.score());
            println!(
                "{:>10}  {:>4}  {:>6}  {:>2}",
                progress.generation(),
                snapshot
                    .best()
                    .fitness()
                    .and_then(Fitness::score)
                    .unwrap_or(0.0),
                median(scores.collect()),
                blocks_of_ones(snapshot.best().genome())
            );
        })
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    println!(
        "{} after {} generations and {} evaluations",
        outcome.best_fitness(),
        outcome.generations(),
        outcome.evaluations()
    );

    // two-point crossover from seeds 1 to 20, and uniform crossover as a contrast
    let mut reached = Vec::new();
    for seed in 1..=SEEDS {
        let outcome = Engine::new(ga(&problem, PointCrossover::two_point(), seed)?, problem)
            .stop_when(Stop::target(optimum).or(Stop::generations(GENERATIONS)))
            .run()?;
        if outcome.stop_reason() == StopReason::Target {
            reached.push(outcome.generations() as f64);
        }
    }
    println!(
        "\nseeds 1 to {SEEDS}, two-point crossover: {} of {SEEDS} reach {optimum}, after a median \
         of {:.1} generations",
        reached.len(),
        median(reached)
    );
    let (mut uniform, mut bests) = (0, Vec::new());
    for seed in 1..=SEEDS {
        let outcome = Engine::new(ga(&problem, UniformCrossover::new(), seed)?, problem)
            .stop_when(Stop::target(optimum).or(Stop::generations(GENERATIONS)))
            .run()?;
        uniform += usize::from(outcome.stop_reason() == StopReason::Target);
        bests.push(blocks_of_ones(outcome.best_genome()) as f64);
    }
    println!(
        "contrast, uniform crossover: {uniform} of {SEEDS} reach it in {GENERATIONS} generations; \
         a median of {:.1} blocks of ones",
        median(bests)
    );

    // hill climbing: one bit flipped at a time, kept if no worse
    let (mut scores, mut blocks) = (Vec::new(), Vec::new());
    for seed in 1..=SEEDS {
        let search = LocalSearch::builder(problem.representation())
            .neighbor(BitFlip::count(1)?)
            .seed(seed)
            .build()?;
        let outcome = Engine::new(search, problem)
            .stop_when(Stop::target(optimum).or(Stop::evaluations(CLIMB)))
            .run()?;
        scores.push(outcome.best_fitness().score().expect("valid"));
        blocks.push(blocks_of_ones(outcome.best_genome()) as f64);
    }
    println!(
        "contrast, hill climbing ({CLIMB} evaluations): a median best of {:.1}, with {:.1} \
         blocks of ones",
        median(scores),
        median(blocks)
    );
    trace.write();
    Ok(())
}
