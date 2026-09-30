//! Kowalik: fit Kowalik and Osborne's rational model to 11 data points, minimizing the sum of
//! squared errors in 4 dimensions, with CMA-ES without and with IPOP restarts, differential
//! evolution and particle swarm optimization, from 30 seeds each.
//!
//! The table counts the runs that reach the best known minimum, to within 1e-8, and the evaluations
//! they take. The function, its data, its bounds and its minimum come from genoxide's
//! `problems::Kowalik`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example kowalik
//! ```

mod trace;

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{Kowalik, Problem};

const SEEDS: u64 = 30;
const BUDGET: u64 = 20_000;
// a run stops once its error to the best known minimum is at most this
const ERROR: f64 = 1e-8;
// the algorithms of the table, in its order
const ALGORITHMS: [Algorithm; 4] = [
    Algorithm::Cmaes,
    Algorithm::CmaesIpop,
    Algorithm::De,
    Algorithm::Pso,
];
// the algorithm whose runs the trace records, and from which seed
const TRACED: Algorithm = Algorithm::CmaesIpop;
const TRACED_SEED: u64 = 6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    Cmaes,
    CmaesIpop,
    De,
    Pso,
}

impl Algorithm {
    fn name(self) -> &'static str {
        match self {
            Algorithm::Cmaes => "CMA-ES",
            Algorithm::CmaesIpop => "CMA-ES with IPOP",
            Algorithm::De => "DE",
            Algorithm::Pso => "PSO",
        }
    }
}

fn main() -> Result<()> {
    let problem = Kowalik;
    let target = problem.optimum().expect("known").value() + ERROR;
    println!(
        "Kowalik: best known minimum 3.07486e-4, {SEEDS} seeds, {BUDGET} \
         evaluations at most per run"
    );
    println!("runs              at min  elsewhere  evaluations: median  largest");
    // with GENOXIDE_TRACE=<file>, a trace of one of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    for algorithm in ALGORITHMS {
        let (mut reached, mut elsewhere) = (0, 0);
        // the evaluations of the runs that reach the target
        let mut evaluations = Vec::new();
        for seed in 1..=SEEDS {
            let traced = algorithm == TRACED && seed == TRACED_SEED;
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
    println!("evaluations: of the runs that reach the best known minimum, to within 1e-8");
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
    let problem = Kowalik;
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
        Algorithm::Pso => {
            let pso = Pso::builder(real)
                .population_size(40)
                .minimize()
                .seed(seed)
                .build()?;
            Engine::new(pso, problem)
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
