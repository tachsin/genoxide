//! LeadingOnes: maximize the number of ones before the first zero of a string of 100 bits, with
//! the (1+1) evolutionary algorithm.
//!
//! The (1+1) EA keeps one string, flips each of its bits with probability 1/n and keeps the child
//! if it's no worse: genoxide's `LocalSearch` with `BitFlip::per_gene(1/n)` as its neighbor. A run
//! from seed 1 prints the evaluations at which the leading ones reach 10, 20, … 100; then runs
//! from seeds 1 to 100 give the evaluations to the optimum, which Droste, Jansen and Wegener
//! (2002) proved to be Θ(n²). The function is genoxide's `problems::binary::LeadingOnes`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example leading_ones
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::binary::LeadingOnes;

const BITS: usize = 100;
const SEEDS: u64 = 100;
// the most evaluations of a run
const BUDGET: u64 = 1_000_000;

// the (1+1) evolutionary algorithm from `seed`
fn one_plus_one(problem: &LeadingOnes, seed: u64) -> Result<LocalSearch<Binary, BitFlip>> {
    LocalSearch::builder(problem.representation())
        .neighbor(BitFlip::per_gene(1.0 / BITS as f64)?)
        .seed(seed)
        .build()
}

fn main() -> Result<()> {
    let problem = LeadingOnes::new(BITS);
    let optimum = problem.optimum().expect("known").value();
    println!("LeadingOnes of {BITS} bits, the (1+1) EA from seed 1");
    println!("leading ones  evaluations");
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(BITS, optimum);
    let mut next = 10.0;
    let outcome = Engine::new(one_plus_one(&problem, 1)?, problem)
        .stop_when(Stop::target(optimum).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            let best = progress.best().and_then(Fitness::score).unwrap_or(0.0);
            // the evaluations at which the leading ones first reach each tenth of the string
            while best >= next {
                println!("{next:>12.0}  {:>11}", progress.evaluations());
                next += 10.0;
            }
        })
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    println!(
        "{} leading ones after {} evaluations (the optimum: {BITS})",
        outcome.best_fitness(),
        outcome.evaluations()
    );

    // the evaluations to the optimum from seeds 1 to 100
    let mut evaluations = Vec::new();
    for seed in 1..=SEEDS {
        let outcome = Engine::new(one_plus_one(&problem, seed)?, problem)
            .stop_when(Stop::target(optimum).or(Stop::evaluations(BUDGET)))
            .run()?;
        if outcome.stop_reason() == StopReason::Target {
            evaluations.push(outcome.evaluations());
        }
    }
    evaluations.sort_unstable();
    let count = evaluations.len();
    let mean = evaluations.iter().sum::<u64>() as f64 / count as f64;
    let median = if count % 2 == 1 {
        evaluations[count / 2] as f64
    } else {
        (evaluations[count / 2 - 1] + evaluations[count / 2]) as f64 / 2.0
    };
    println!(
        "\nseeds 1 to {SEEDS}: {count} reach the optimum, after {mean:.0} evaluations on average"
    );
    println!(
        "(fewest {}, median {median:.1}, most {}); n² = {}",
        evaluations[0],
        evaluations[count - 1],
        BITS * BITS
    );
    trace.write();
    Ok(())
}
