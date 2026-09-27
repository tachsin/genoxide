//! Gear train design (Sandgren, 1990): the numbers of teeth of a compound gear train of four
//! gears, from 12 to 60 each, whose ratio is closest to 1/6.931. An integer problem.
//!
//! The problem is genoxide's `GearTrain`, on integer genes; its score is the squared error of
//! the ratio. A genetic algorithm with uniform crossover and a mutation that redraws each gene
//! with probability 0.25 searches the 49⁴ ≈ 5.8 million designs, until it reaches the minimum,
//! 2.700857e-12, known by evaluating them all.
//!
//! ```text
//! cargo run --release --example gear_train
//! ```

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::engineering::GearTrain;

fn main() -> Result<()> {
    let problem = GearTrain;
    let minimum = problem.optimum().expect("known").value();
    let ga = Ga::builder(problem.representation())
        .population_size(100)
        .select(Tournament::new(2)?)
        .crossover(UniformCrossover::new())
        .mutate(UniformMutation::per_gene(0.25)?)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(ga, problem)
        .stop_when(Stop::target(minimum).or(Stop::generations(2_000)))
        .run()?;

    let teeth = outcome.best_genome();
    let ratio = (teeth[0] * teeth[1]) as f64 / (teeth[2] * teeth[3]) as f64;
    println!(
        "error {:.6e} after {} generations (the minimum: {minimum:.6e})",
        outcome.best_fitness().score().unwrap_or(f64::NAN),
        outcome.generations()
    );
    println!(
        "teeth ({}, {}, {}, {}), ratio {ratio:.8} (the target: {:.8})",
        teeth[0],
        teeth[1],
        teeth[2],
        teeth[3],
        1.0 / 6.931
    );
    Ok(())
}
