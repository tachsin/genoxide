//! Easom: minimize Easom's function, a single narrow well in a nearly flat plane, with CMA-ES,
//! from 30 seeds, without restarts and with IPOP restarts.
//!
//! Away from its well, the function is within a hair of 0, and covered with tiny local minima. A
//! run of CMA-ES without restarts that doesn't find the well early converges into one of them;
//! with IPOP restarts, a run that has converged starts again from a random point with twice the
//! population. The table counts the runs that reach the minimum. The function, its bounds and its
//! minimum come from genoxide's `problems::Easom`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example easom
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Easom, Problem};

const SEEDS: u64 = 30;
const BUDGET: u64 = 10_000;
// a run stops once its error to the minimum is at most this
const ERROR: f64 = 1e-8;

fn main() -> Result<()> {
    let problem = Easom;
    let minimum = problem.optimum().expect("known").value();
    let target = minimum + ERROR;
    println!(
        "Easom: minimum {minimum} at (pi, pi), {SEEDS} seeds, {BUDGET} evaluations at most per run"
    );
    println!("runs              at -1  elsewhere  evaluations: median  largest");
    // with GENOXIDE_TRACE=<file>, a trace of the runs without restarts for the plot on the
    // example's page
    let mut trace = trace::Trace::from_env();
    for restarts in [cmaes::Restarts::Never, cmaes::Restarts::Ipop] {
        let (mut reached, mut elsewhere) = (0, 0);
        // the evaluations of the runs that reach the target
        let mut evaluations = Vec::new();
        for seed in 1..=SEEDS {
            let cmaes = Cmaes::builder(problem.representation())
                .restarts(restarts)
                .minimize()
                .seed(seed)
                .build()?;
            let traced = matches!(restarts, cmaes::Restarts::Never);
            let outcome = Engine::new(cmaes, problem)
                .stop_when(Stop::target(target).or(Stop::evaluations(BUDGET)))
                .on_generation(|snapshot| {
                    if traced {
                        trace.record(snapshot);
                    }
                })
                .run()?;
            if outcome.stop_reason() == StopReason::Target {
                reached += 1;
                evaluations.push(outcome.evaluations());
            } else {
                elsewhere += 1;
            }
        }
        let name = match restarts {
            cmaes::Restarts::Never => "CMA-ES",
            _ => "CMA-ES with IPOP",
        };
        let largest = evaluations.iter().max().copied().unwrap_or(0);
        println!(
            "{name:<16}  {reached:>5}  {elsewhere:>9}  {:>19}  {largest:>7}",
            median(&mut evaluations)
        );
    }
    println!("evaluations: of the runs that reach the minimum, to within 1e-8");
    trace.write();
    Ok(())
}

// the median of the evaluations, rounded down; 0 without any
fn median(evaluations: &mut [u64]) -> u64 {
    evaluations.sort_unstable();
    let middle = evaluations.len() / 2;
    match evaluations.len() {
        0 => 0,
        n if n % 2 == 1 => evaluations[middle],
        _ => (evaluations[middle - 1] + evaluations[middle]) / 2,
    }
}
