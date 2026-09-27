//! OneMax: find the bit string with the most ones.
//!
//! The "hello world" of genetic algorithms: a binary genome, tournament selection, uniform
//! crossover and bit-flip mutation, with the best count printed every 50 generations.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example one_max
//! ```

mod trace;

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

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    println!("generation  best");
    let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
        .stop_when(Stop::target(LEN as f64).or(Stop::generations(10_000)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            if progress.generation() % 50 == 0 {
                let best = progress.best().unwrap_or(Fitness::invalid());
                println!("{:>10}  {best:>4}", progress.generation());
            }
        })
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    println!(
        "\n{} ones after {} generations and {} evaluations (the optimum: {LEN})",
        outcome.best_fitness(),
        outcome.generations(),
        outcome.evaluations()
    );
    trace.write();
    Ok(())
}
