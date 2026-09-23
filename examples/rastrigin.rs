//! Rastrigin: minimize a real-valued function with many local minima.
//!
//! Shows a real-valued genome, minimization, parallel evaluation and a time limit. The global
//! minimum is 0, at the origin.
//!
//! Uniform mutation finds the basin of the global minimum, but it's a blunt tool for the last
//! digits: the Gaussian and polynomial mutations planned for 0.2 fine-tune much better.
//!
//! ```text
//! cargo run --release --example rastrigin
//! ```

use genoxide::prelude::*;
use std::f64::consts::TAU;
use std::time::Duration;

const DIMENSIONS: usize = 10;

fn rastrigin(x: &Reals) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (TAU * xi).cos())
            .sum::<f64>()
}

fn main() -> Result<()> {
    let ga = Ga::builder(Real::uniform(DIMENSIONS, -5.12..=5.12)?)
        .population_size(200)
        .select(Tournament::new(4)?)
        .crossover(UniformCrossover::new())
        .mutate(UniformMutation::count(1)?)
        .scheme(Scheme::Generational { elitism: 2 })
        .minimize()
        .seed(3)
        .build()?;

    let outcome = Engine::new(ga, rastrigin)
        // worth it for expensive fitness functions; the results are the same either way
        .parallel(true)
        .stop_when(
            Stop::target(1e-3)
                .or(Stop::stagnation(500))
                .or(Stop::time(Duration::from_secs(30))),
        )
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            if progress.generation() % 200 == 0 {
                let best = progress.best().unwrap_or(Fitness::invalid());
                println!("{:>6}: {best:.6}", progress.generation());
            }
        })
        .run()?;

    println!(
        "\n{:?} after {} generations ({:?}): {:.6}",
        outcome.stop_reason(),
        outcome.generations(),
        outcome.elapsed(),
        outcome.best_fitness()
    );
    println!("at {:.4?}", &outcome.best_genome()[..]);
    Ok(())
}
