//! Bohachevsky 3: minimize Bohachevsky's third function, a bowl with oblique ripples, with CMA-ES
//! without and with IPOP restarts and differential evolution, from 30 seeds each.
//!
//! The table counts the runs that reach the minimum, to within 1e-8, and the evaluations they take.
//! The function, its bounds and its minimum come from genoxide's `problems::Bohachevsky3`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example bohachevsky3
//! ```

mod trace;

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{Bohachevsky3, Problem};

const SEEDS: u64 = 30;
const BUDGET: u64 = 10_000;
// a run stops once its error to the minimum is at most this
const ERROR: f64 = 1e-8;
// the algorithms of the table, in its order
const ALGORITHMS: [Algorithm; 3] = [Algorithm::Cmaes, Algorithm::CmaesIpop, Algorithm::De];
// the algorithm whose runs the trace records
const TRACED: Algorithm = Algorithm::Cmaes;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    Cmaes,
    CmaesIpop,
    De,
}

impl Algorithm {
    fn name(self) -> &'static str {
        match self {
            Algorithm::Cmaes => "CMA-ES",
            Algorithm::CmaesIpop => "CMA-ES with IPOP",
            Algorithm::De => "DE",
        }
    }
}

fn main() -> Result<()> {
    let problem = Bohachevsky3;
    let target = problem.optimum().expect("known").value() + ERROR;
    println!(
        "Bohachevsky 3: minimum 0 at (0, 0), {SEEDS} seeds, {BUDGET} evaluations at most per run"
    );
    println!("runs              at min  elsewhere  evaluations: median  largest");
    // with GENOXIDE_TRACE=<file>, a trace of the runs of CMA-ES for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    for algorithm in ALGORITHMS {
        let (mut reached, mut elsewhere) = (0, 0);
        // the evaluations of the runs that reach the target
        let mut evaluations = Vec::new();
        for seed in 1..=SEEDS {
            let traced = algorithm == TRACED;
            let outcome = run(algorithm, seed, target, |snapshot| {
                if traced {
                    trace.record(snapshot);
                }
            })?;
            if outcome.stop_reason() == StopReason::Target {
                reached += 1;
                evaluations.push(outcome.evaluations());
            } else {
                elsewhere += 1;
            }
        }
        let largest = evaluations.iter().max().copied().unwrap_or(0);
        println!(
            "{:<16}  {reached:>6}  {elsewhere:>9}  {:>19}  {largest:>7}",
            algorithm.name(),
            median(&mut evaluations)
        );
    }
    println!("evaluations: of the runs that reach the minimum, to within 1e-8");
    trace.write();
    Ok(())
}

// a run of `algorithm` from `seed`, until its best is at most `target` or it has used BUDGET
// evaluations, which calls `record` after each generation
fn run(
    algorithm: Algorithm,
    seed: u64,
    target: f64,
    record: impl FnMut(&Snapshot<'_, Reals>),
) -> Result<Outcome<Reals>> {
    let problem = Bohachevsky3;
    let real = problem.representation();
    let stop = Stop::target(target).or(Stop::evaluations(BUDGET));
    match algorithm {
        Algorithm::Cmaes => {
            let cmaes = Cmaes::builder(real).minimize().seed(seed).build()?;
            Engine::new(cmaes, problem)
                .stop_when(stop)
                .on_generation(record)
                .run()
        }
        Algorithm::CmaesIpop => {
            let cmaes = Cmaes::builder(real)
                .restarts(cmaes::Restarts::Ipop)
                .minimize()
                .seed(seed)
                .build()?;
            Engine::new(cmaes, problem)
                .stop_when(stop)
                .on_generation(record)
                .run()
        }
        Algorithm::De => {
            let de = De::builder(real)
                .population_size(20)
                .minimize()
                .seed(seed)
                .build()?;
            Engine::new(de, problem)
                .stop_when(stop)
                .on_generation(record)
                .run()
        }
    }
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
