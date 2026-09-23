//! OneMax: find the bit string with the most ones.
//!
//! The "hello world" of genetic algorithms: a binary genome, tournament selection, uniform
//! crossover and bit-flip mutation, with statistics per generation.
//!
//! ```text
//! cargo run --release --example one_max
//! ```

use genoxide::prelude::*;

const LEN: usize = 500;

fn main() -> Result<()> {
    let ga = Ga::builder(Binary::new(LEN)?)
        .population_size(100)
        .select(Tournament::new(3)?)
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / LEN as f64)?)
        .seed(42)
        .build()?;

    let mut statistics = Statistics::new();
    let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
        .stop_when(Stop::target(LEN as f64).or(Stop::generations(10_000)))
        .observe(&mut statistics)
        .run()?;

    println!("generation  best  mean");
    for record in statistics.records().iter().step_by(25) {
        println!(
            "{:>10}  {:>4}  {:>6.1}",
            record.generation,
            record.best.and_then(Fitness::score).unwrap_or(f64::NAN),
            record.mean.unwrap_or(f64::NAN),
        );
    }
    println!(
        "\n{:?} after {} generations and {} evaluations ({:?}): {}",
        outcome.stop_reason(),
        outcome.generations(),
        outcome.evaluations(),
        outcome.elapsed(),
        outcome.best_fitness(),
    );
    Ok(())
}
