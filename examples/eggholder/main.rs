//! Eggholder: minimize the eggholder function, deep local minima all over and the deepest on the
//! edge of the box, from 30 seeds each with CMA-ES with IPOP restarts and with particle swarms of
//! a global and a ring topology.
//!
//! The runs that end close together are grouped, and the table gives each group's best point and
//! value, and how many runs of each algorithm end there. The function, its bounds and its best
//! known minimum come from genoxide's `problems::Eggholder`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example eggholder
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Eggholder, Problem};

const SEEDS: u64 = 30;
const BUDGET: u64 = 20_000;
const PARTICLES: usize = 40;
// a run stops once its error to the best known minimum is at most this
const ERROR: f64 = 1e-6;
const ALGORITHMS: [&str; 3] = ["CMA-ES, IPOP", "PSO, global", "PSO, ring"];

// a group of runs that ended close together: its best point and value, and its runs per
// algorithm
struct Minimum {
    point: [f64; 2],
    value: f64,
    runs: [u64; 3],
}

fn main() -> Result<()> {
    let problem = Eggholder;
    let minimum = problem.optimum().expect("known").value();
    let target = minimum + ERROR;
    let stop = || Stop::target(target).or(Stop::evaluations(BUDGET));
    let mut minima: Vec<Minimum> = Vec::new();
    // per algorithm, the runs that reach the target
    let mut reached = [0u64; 3];
    // with GENOXIDE_TRACE=<file>, a trace of the runs of the swarm with the global topology for
    // the plot on the example's page
    let mut trace = trace::Trace::from_env();
    for a in 0..ALGORITHMS.len() {
        for seed in 1..=SEEDS {
            let outcome = if a == 0 {
                let cmaes = Cmaes::builder(problem.representation())
                    .restarts(cmaes::Restarts::Ipop)
                    .minimize()
                    .seed(seed)
                    .build()?;
                Engine::new(cmaes, problem).stop_when(stop()).run()?
            } else {
                let topology = if a == 1 {
                    pso::Topology::Global
                } else {
                    pso::Topology::Ring { neighbors: 1 }
                };
                let swarm = Pso::builder(problem.representation())
                    .population_size(PARTICLES)
                    .topology(topology)
                    .minimize()
                    .seed(seed)
                    .build()?;
                Engine::new(swarm, problem)
                    .stop_when(stop())
                    .on_generation(|snapshot| {
                        if a == 1 {
                            trace.record(snapshot);
                        }
                    })
                    .run()?
            };
            if outcome.stop_reason() == StopReason::Target {
                reached[a] += 1;
            }
            let end = outcome.best_genome();
            let point = [end[0], end[1]];
            let value = outcome.best_fitness().score().expect("valid");
            // the same minimum: within 2% of the bounds' width, 20.48, in both genes
            let close = |minimum: &&mut Minimum| {
                (0..2).all(|i| (point[i] - minimum.point[i]).abs() <= 20.48)
            };
            match minima.iter_mut().find(close) {
                Some(minimum) => {
                    minimum.runs[a] += 1;
                    if value < minimum.value {
                        (minimum.point, minimum.value) = (point, value);
                    }
                }
                None => {
                    let mut runs = [0; 3];
                    runs[a] = 1;
                    minima.push(Minimum { point, value, runs });
                }
            }
        }
    }
    minima.sort_by(|a, b| a.value.total_cmp(&b.value));

    println!(
        "Eggholder: best known minimum {minimum:.4} at (512, 404.2318), {SEEDS} seeds, {BUDGET} \
         evaluations at most per run"
    );
    let [cmaes, global, ring] = ALGORITHMS;
    println!("runs ending near        best value  {cmaes}  {global}  {ring}");
    for minimum in &minima {
        let [x1, x2] = minimum.point;
        let [a, b, c] = minimum.runs;
        let at = format!("({x1:.1}, {x2:.1})");
        println!(
            "{at:<16}  {:>16.4}  {a:>12}  {b:>11}  {c:>9}",
            minimum.value
        );
    }
    let [a, b, c] = reached;
    let label = "error below 1e-6";
    println!("{label:<34}  {a:>12}  {b:>11}  {c:>9}");
    trace.write();
    Ok(())
}
