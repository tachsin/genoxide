//! Hartmann 6-D: minimize Hartmann's function in 6 dimensions with CMA-ES, from 30 seeds,
//! without restarts and with IPOP restarts.
//!
//! The function has two basins of nearly the same depth. A run of CMA-ES without restarts
//! converges into one of them and stays; with IPOP restarts, a run that has converged starts
//! again from a random point with twice the population. The table counts the runs that end in
//! each minimum. The function, its bounds and its best known minimum come from genoxide's
//! `problems::Hartmann6`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example hartmann6
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Hartmann6, Problem};

const SEEDS: u64 = 30;
const BUDGET: u64 = 20_000;
// a run stops once its error to the best known minimum is at most this
const ERROR: f64 = 1e-6;
// the other local minimum, where a third of local searches end
const OTHER_MINIMUM: f64 = -3.203_161_918_396_231;

fn main() -> Result<()> {
    let problem = Hartmann6;
    let minimum = problem.optimum().expect("known").value();
    let target = minimum + ERROR;
    println!(
        "Hartmann 6-D: best known minimum {minimum:.5}, {SEEDS} seeds, {BUDGET} evaluations at \
         most per run"
    );
    let columns = format!("at {minimum:.5}  at {OTHER_MINIMUM:.5}  elsewhere");
    println!("{:<16}  {columns}  evaluations: median  largest", "runs");
    // the seed of the first run without restarts that ends at the other minimum: the trace
    // records the run with IPOP restarts from that seed, or from seed 1 if none does
    let mut trace_seed = None;
    for restarts in [cmaes::Restarts::Never, cmaes::Restarts::Ipop] {
        if matches!(restarts, cmaes::Restarts::Ipop) {
            trace_seed.get_or_insert(1);
        }
        let (mut global, mut other, mut elsewhere) = (0, 0, 0);
        // the evaluations of the runs that reach the target
        let mut evaluations = Vec::new();
        for seed in 1..=SEEDS {
            let cmaes = Cmaes::builder(problem.representation())
                .restarts(restarts)
                .minimize()
                .seed(seed)
                .build()?;
            let mut trace = match trace_seed {
                Some(traced) if traced == seed && matches!(restarts, cmaes::Restarts::Ipop) => {
                    trace::Trace::from_env()
                }
                _ => trace::Trace::none(),
            };
            let outcome = Engine::new(cmaes, problem)
                .stop_when(Stop::target(target).or(Stop::evaluations(BUDGET)))
                .on_generation(|snapshot| trace.record(snapshot))
                .run()?;
            trace.write();
            let value = outcome.best_fitness().score().expect("valid");
            if outcome.stop_reason() == StopReason::Target {
                global += 1;
                evaluations.push(outcome.evaluations());
            } else if (value - OTHER_MINIMUM).abs() <= ERROR {
                other += 1;
                trace_seed.get_or_insert(seed);
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
            "{name:<16}  {global:>11}  {other:>11}  {elsewhere:>9}  {:>19}  {largest:>7}",
            median(&mut evaluations)
        );
    }
    println!("evaluations: of the runs that reach the best known minimum");
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
