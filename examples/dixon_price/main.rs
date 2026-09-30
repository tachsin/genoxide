//! Dixon-Price: minimize the Dixon-Price function, a chain of curved valleys, in 5 and 10
//! dimensions, with CMA-ES with IPOP restarts, differential evolution and particle swarms with a
//! global and a ring topology, from 30 seeds each.
//!
//! In 3 dimensions or more, the function has a stationary point with the value 2/3 at (1/3, 0, …,
//! 0), where searches stall, and more of them the more dimensions. The table counts the runs that
//! reach the minimum, 0, to within 1e-8, those that end at the stationary point, and the
//! evaluations of the runs that reach the minimum. The function, its bounds and its minima come
//! from genoxide's `problems::DixonPrice`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of one of its runs for the plot on the
//! example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example dixon_price
//! ```

mod trace;

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{DixonPrice, Problem};

const SEEDS: u64 = 30;
// the evaluations of a run, at most, per dimension
const BUDGET: u64 = 20_000;
// a run stops once its error to the minimum is at most this
const ERROR: f64 = 1e-8;
// the value at the stationary point (1/3, 0, …, 0), in 3 dimensions or more
const STATIONARY: f64 = 2.0 / 3.0;
// the rows of the table: an algorithm, and the number of dimensions
const ROWS: [(Algorithm, usize); 5] = [
    (Algorithm::CmaesIpop, 5),
    (Algorithm::CmaesIpop, 10),
    (Algorithm::De, 10),
    (Algorithm::Pso, 10),
    (Algorithm::PsoRing, 10),
];
// the run that the trace records: its algorithm, dimensions and seed
const TRACED: (Algorithm, usize, u64) = (Algorithm::PsoRing, 10, 2);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    CmaesIpop,
    De,
    Pso,
    PsoRing,
}

impl Algorithm {
    fn name(self) -> &'static str {
        match self {
            Algorithm::CmaesIpop => "CMA-ES with IPOP",
            Algorithm::De => "DE",
            Algorithm::Pso => "PSO",
            Algorithm::PsoRing => "PSO (ring)",
        }
    }
}

fn main() -> Result<()> {
    println!(
        "Dixon-Price: minimum 0, stationary point 2/3, {SEEDS} seeds, {BUDGET} evaluations \
         per dimension at most per run"
    );
    println!("runs                      at 0  at 2/3  elsewhere  evaluations: median  largest");
    // with GENOXIDE_TRACE=<file>, a trace of one of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    for (algorithm, dimensions) in ROWS {
        let (mut reached, mut stalled, mut elsewhere) = (0, 0, 0);
        // the evaluations of the runs that reach the target
        let mut evaluations = Vec::new();
        for seed in 1..=SEEDS {
            let traced = (algorithm, dimensions, seed) == TRACED;
            let outcome = run(algorithm, dimensions, seed, |snapshot| {
                if traced {
                    trace.record(snapshot);
                }
            })?;
            let value = outcome.best_fitness().score().expect("valid");
            if outcome.stop_reason() == StopReason::Target {
                reached += 1;
                evaluations.push(outcome.evaluations());
            } else if (value - STATIONARY).abs() <= ERROR {
                stalled += 1;
            } else {
                elsewhere += 1;
            }
        }
        let largest = evaluations.iter().max().copied().unwrap_or(0);
        let name = format!("{}, n = {dimensions}", algorithm.name());
        println!(
            "{name:<24}  {reached:>4}  {stalled:>6}  {elsewhere:>9}  {:>19}  {largest:>7}",
            median(&mut evaluations)
        );
    }
    println!("evaluations: of the runs that reach the minimum, to within 1e-8");
    trace.write();
    Ok(())
}

// a run of `algorithm` in `dimensions` dimensions from `seed`, until it's within ERROR of the
// minimum or it has used BUDGET evaluations per dimension, which calls `record` after each
// generation
fn run(
    algorithm: Algorithm,
    dimensions: usize,
    seed: u64,
    record: impl FnMut(&Snapshot<'_, Reals>),
) -> Result<Outcome<Reals>> {
    let problem = DixonPrice::new(dimensions);
    let real = problem.representation();
    let budget = BUDGET * dimensions as u64;
    let stop = Stop::target(ERROR).or(Stop::evaluations(budget));
    match algorithm {
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
            let de = De::builder(real).minimize().seed(seed).build()?;
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
