//! Langermann: minimize Langermann's function in two dimensions, rings of ripples around five
//! centers, with CMA-ES without and with IPOP restarts, particle swarms with a global and a ring
//! topology, and differential evolution, from 30 seeds each.
//!
//! The deepest minimum is a narrow well next to a wider one nearly as deep. The table counts the
//! runs that reach the best known minimum, to within 1e-8, and the evaluations they take. The
//! function, its bounds and its minimum come from genoxide's `problems::Langermann`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example langermann
//! ```

mod trace;

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{Langermann, Problem};

const SEEDS: u64 = 30;
const BUDGET: u64 = 50_000;
// a run stops once its error to the best known minimum is at most this
const ERROR: f64 = 1e-8;
// the algorithms of the table, in its order
const ALGORITHMS: [Algorithm; 5] = [
    Algorithm::Cmaes,
    Algorithm::CmaesIpop,
    Algorithm::Pso,
    Algorithm::PsoRing,
    Algorithm::De,
];
// the algorithm whose runs the trace records
const TRACED: Algorithm = Algorithm::Cmaes;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    Cmaes,
    CmaesIpop,
    Pso,
    PsoRing,
    De,
}

impl Algorithm {
    fn name(self) -> &'static str {
        match self {
            Algorithm::Cmaes => "CMA-ES",
            Algorithm::CmaesIpop => "CMA-ES with IPOP",
            Algorithm::Pso => "PSO",
            Algorithm::PsoRing => "PSO (ring)",
            Algorithm::De => "DE",
        }
    }
}

fn main() -> Result<()> {
    let problem = Langermann;
    let target = problem.optimum().expect("known").value() + ERROR;
    println!(
        "Langermann: best known minimum -4.15581, {SEEDS} seeds, {BUDGET} \
         evaluations at most per run"
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
    let problem = Langermann;
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
        Algorithm::PsoRing => {
            let pso = Pso::builder(real)
                .population_size(40)
                .topology(pso::Topology::Ring { neighbors: 1 })
                .minimize()
                .seed(seed)
                .build()?;
            Engine::new(pso, problem)
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
