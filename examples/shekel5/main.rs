//! Shekel 5: minimize Shekel's function with 5 wells in 4 dimensions with particle swarms, from
//! 30 seeds, with a global and a ring topology.
//!
//! The function has a well at each of 5 points, the deepest at (4, 4, 4, 4). A swarm whose
//! particles all follow the best position found so far (the global topology) can gather in
//! another well before a particle falls into the deepest; a ring topology spreads good positions
//! slowly and keeps exploring longer. The table counts the runs that end in each well. The
//! function, its bounds and its best known minimum come from genoxide's `problems::Shekel5`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example shekel5
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Problem, Shekel5};

// the wells: Shekel's first points aᵢ
const WELLS: [[f64; 4]; 5] = [
    [4.0, 4.0, 4.0, 4.0],
    [1.0, 1.0, 1.0, 1.0],
    [8.0, 8.0, 8.0, 8.0],
    [6.0, 6.0, 6.0, 6.0],
    [3.0, 7.0, 3.0, 7.0],
];
const SEEDS: u64 = 30;
const BUDGET: u64 = 10_000;
const PARTICLES: usize = 40;
// a run stops once its error to the best known minimum is at most this
const ERROR: f64 = 1e-6;

fn main() -> Result<()> {
    let problem = Shekel5;
    let minimum = problem.optimum().expect("known").value();
    let target = minimum + ERROR;
    let topologies = [pso::Topology::Global, pso::Topology::Ring { neighbors: 1 }];
    // per topology: the runs that end in each well, those that reach the target, and their
    // evaluations
    let mut wells = [[0u64; WELLS.len()]; 2];
    let mut reached = [0u64; 2];
    let mut evaluations = [Vec::new(), Vec::new()];
    // with GENOXIDE_TRACE=<file>, a trace of the runs with the global topology for the plot on
    // the example's page
    let mut trace = trace::Trace::from_env();
    for (t, topology) in topologies.into_iter().enumerate() {
        for seed in 1..=SEEDS {
            let swarm = Pso::builder(problem.representation())
                .population_size(PARTICLES)
                .topology(topology)
                .minimize()
                .seed(seed)
                .build()?;
            let outcome = Engine::new(swarm, problem)
                .stop_when(Stop::target(target).or(Stop::evaluations(BUDGET)))
                .on_generation(|snapshot| {
                    if t == 0 {
                        trace.record(snapshot);
                    }
                })
                .run()?;
            wells[t][nearest(outcome.best_genome())] += 1;
            if outcome.stop_reason() == StopReason::Target {
                reached[t] += 1;
                evaluations[t].push(outcome.evaluations());
            }
        }
    }

    println!(
        "Shekel 5: best known minimum {minimum:.5}, {SEEDS} seeds, {BUDGET} evaluations at most \
         per run, {PARTICLES} particles"
    );
    println!("runs ending in the well at  PSO, global  PSO, ring");
    for (i, well) in WELLS.iter().enumerate() {
        if wells[0][i] + wells[1][i] == 0 {
            continue;
        }
        let at = format!("a{} = ({})", i + 1, coordinates(well));
        println!("{at:<26}  {:>11}  {:>9}", wells[0][i], wells[1][i]);
    }
    println!(
        "{:<26}  {:>11}  {:>9}",
        "error below 1e-6", reached[0], reached[1]
    );
    let [global, ring] = evaluations.map(|mut evaluations| median(&mut evaluations));
    println!("{:<26}  {global:>11}  {ring:>9}", "evaluations (median)");
    trace.write();
    Ok(())
}

// the index of the well nearest `x`
fn nearest(x: &Reals) -> usize {
    let distance = |well: &[f64; 4]| -> f64 { (0..4).map(|j| (x[j] - well[j]).powi(2)).sum() };
    (0..WELLS.len())
        .min_by(|&a, &b| distance(&WELLS[a]).total_cmp(&distance(&WELLS[b])))
        .expect("wells")
}

// a well's coordinates, e.g. 3, 7, 3, 7 or 7, 3.6, 7, 3.6
fn coordinates(well: &[f64; 4]) -> String {
    let text: Vec<String> = well.iter().map(|x| x.to_string()).collect();
    text.join(", ")
}

// the median of the evaluations of the runs that reach the target, rounded down; 0 without any
fn median(evaluations: &mut [u64]) -> u64 {
    evaluations.sort_unstable();
    let middle = evaluations.len() / 2;
    match evaluations.len() {
        0 => 0,
        n if n % 2 == 1 => evaluations[middle],
        _ => (evaluations[middle - 1] + evaluations[middle]) / 2,
    }
}
