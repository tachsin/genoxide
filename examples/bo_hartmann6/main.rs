//! Bayesian optimization of Hartmann's 6-D function in batches: 4 points a round, chosen one after
//! the other with the Kriging believer and evaluated in parallel, to within 1e-4 of the global
//! minimum. Then, as a contrast, the same search one point a round: fewer evaluations, more rounds.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of both runs for the plot on the
//! example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example bo_hartmann6
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Hartmann6, Problem};

// how close to the global minimum, and the evaluations each run may take at most
const TOLERANCE: f64 = 1e-4;
const BUDGET: u64 = 200;
const SEED: u64 = 3;

fn main() -> Result<()> {
    let optimum = Hartmann6.optimum().expect("known");
    let minimum = optimum.value();
    println!("Hartmann's 6-D function in [0, 1]^6: global minimum {minimum:.6}");
    println!(
        "14 points of a Latin hypercube, then 4 points a round by log-EI and the Kriging believer"
    );
    println!("round  evaluations          best     f - f*");
    let (batched, rounds) = search(4, minimum, true)?;
    let best = batched.best_fitness().score().expect("valid");
    let x = batched.best_genome();
    let distance = x
        .iter()
        .zip(&optimum.solutions()[0][..])
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    println!(
        "4 points a round: within {TOLERANCE:.0e} of the minimum after {} evaluations in {} \
         rounds, {distance:.1e} from its point",
        batched.evaluations(),
        batched.generations()
    );
    assert!(best - minimum <= TOLERANCE);

    // the contrast: one point a round, the same seed
    let (single, single_rounds) = search(1, minimum, false)?;
    let reached = single.best_fitness().score().expect("valid") - minimum <= TOLERANCE;
    println!(
        "1 point a round:  {} after {} evaluations in {} rounds",
        if reached {
            format!("within {TOLERANCE:.0e} of the minimum")
        } else {
            "not within the tolerance".to_string()
        },
        single.evaluations(),
        single.generations()
    );
    trace::write(&rounds, &single_rounds);
    Ok(())
}

// a search in batches of `batch` points evaluated in parallel, to within the tolerance of
// `minimum` or the budget, printing a row per round if `print`; its outcome, and its best value's
// distance above the minimum after each round
fn search(batch: usize, minimum: f64, print: bool) -> Result<(Outcome<Reals>, Vec<f64>)> {
    let bo = Bo::builder(Hartmann6.representation())
        .batch(batch)
        .minimize()
        .seed(SEED)
        .build()?;
    let mut rounds = Vec::new();
    let outcome = Engine::new(bo, Hartmann6)
        .parallel(true)
        .stop_when(Stop::target(minimum + TOLERANCE).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            let best = progress.best().and_then(Fitness::score).expect("valid");
            rounds.push(best - minimum);
            if print {
                println!(
                    "{:>5} {:>12} {:>13.6} {:>10}",
                    progress.generation(),
                    progress.evaluations(),
                    best,
                    format!("{:.1e}", best - minimum)
                );
            }
        })
        .run()?;
    Ok((outcome, rounds))
}
