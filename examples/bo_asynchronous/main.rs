//! Asynchronous Bayesian optimization for an expensive function whose evaluations take different
//! times: an `AsyncEngine` gives each of 4 workers a new point as soon as it's done, chosen by the
//! model of the results so far with the points still being evaluated added at a lie. It reaches
//! the global minimum of Hartmann's 3-D function to within 1e-4. Then, as a contrast in time,
//! batches of 4 points, each evaluated together, which wait for their slowest evaluation.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of the asynchronous run for the plot on
//! the example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example bo_asynchronous
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Hartmann3, Problem};
use std::time::{Duration, Instant};

// the workers, how close to the global minimum, and the evaluations a run may take at most
const WORKERS: usize = 4;
const TOLERANCE: f64 = 1e-4;
const BUDGET: u64 = 120;
const SEED: u64 = 1;

// Hartmann 3, taking 10 to 50 ms: a time drawn from the point's bits and a seed, so the same point
// always takes as long
fn simulation(x: &Reals) -> f64 {
    let mut hash = 0x9e37_79b9_7f4a_7c15_u64 ^ SEED;
    for gene in x.iter() {
        hash = splitmix64(hash ^ gene.to_bits());
    }
    std::thread::sleep(Duration::from_millis(10 + hash % 41));
    Hartmann3.evaluate(x)
}

// a step of SplitMix64 (Steele, Lea and Flood, 2014): a well-mixed 64-bit hash
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn main() -> Result<()> {
    let minimum = Hartmann3.optimum().expect("known").value();
    println!(
        "Hartmann's 3-D function, evaluations of 10 to 50 ms, {WORKERS} at a time: global \
         minimum {minimum:.6}"
    );
    let bo = || {
        Bo::builder(Hartmann3.representation())
            .batch(WORKERS)
            .minimize()
            .seed(SEED)
            .build()
    };
    let stop = || Stop::target(minimum + TOLERANCE).or(Stop::evaluations(BUDGET));

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(WORKERS);
    let start = Instant::now();
    let asynchronous = AsyncEngine::new(bo()?, trace.timed(simulation))
        .workers(WORKERS)
        .stop_when(stop())
        .observe(&mut trace)
        .run()?;
    let elapsed = start.elapsed();
    report("asynchronous", &asynchronous, elapsed, minimum);
    assert!(asynchronous.best_fitness().score().expect("valid") - minimum <= TOLERANCE);
    trace.write();

    // the contrast: batches of 4, a round waiting for its slowest evaluation
    let start = Instant::now();
    let batches = Engine::new(bo()?, simulation)
        .parallel(true)
        .stop_when(stop())
        .run()?;
    report("in batches", &batches, start.elapsed(), minimum);
    Ok(())
}

fn report(name: &str, outcome: &Outcome<Reals>, elapsed: Duration, minimum: f64) {
    let best = outcome.best_fitness().score().expect("valid");
    println!(
        "{name:>12}: {:.2} s, {} evaluations, {:.1} evaluations/s, best {best:.6}, {:.1e} above \
         the minimum",
        elapsed.as_secs_f64(),
        outcome.evaluations(),
        outcome.evaluations() as f64 / elapsed.as_secs_f64(),
        best - minimum
    );
}
