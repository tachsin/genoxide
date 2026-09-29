//! Schaffer F6: minimize Schaffer's F6, rings of local minima around the global one, from 30
//! seeds each with SHADE, a differential evolution, and, for contrast, with a particle swarm and a
//! genetic algorithm.
//!
//! The function depends only on the distance r from the origin, and its local minima are rings
//! near r = π, 2π, 3π, …; the table counts the runs whose best point ends on each ring. The
//! function, its bounds and its minimum come from genoxide's `problems::SchafferF6`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example schaffer_f6
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Problem, SchafferF6};
use std::f64::consts::PI;

const SEEDS: u64 = 30;
const BUDGET: u64 = 50_000;
// a run stops once its error to the minimum is at most this
const ERROR: f64 = 1e-6;

// the runs that end near a ring: the best value among them, and the runs per algorithm
#[derive(Clone, Copy)]
struct Ring {
    value: f64,
    runs: [u64; 3],
}

fn main() -> Result<()> {
    let problem = SchafferF6;
    let target = problem.optimum().expect("known").value() + ERROR;
    let stop = || Stop::target(target).or(Stop::evaluations(BUDGET));
    // by ring: 0 is the minimum, k the ring near r = kπ
    let mut rings: Vec<Option<Ring>> = Vec::new();
    // per algorithm, the runs that reach the target
    let mut reached = [0u64; 3];
    // with GENOXIDE_TRACE=<file>, a trace of SHADE's runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    for (a, reached) in reached.iter_mut().enumerate() {
        for seed in 1..=SEEDS {
            let outcome = if a == 0 {
                let shade = De::builder(problem.representation())
                    .minimize()
                    .seed(seed)
                    .build()?;
                Engine::new(shade, problem)
                    .stop_when(stop())
                    .on_generation(|snapshot| trace.record(snapshot))
                    .run()?
            } else if a == 2 {
                let ga = Ga::builder(problem.representation())
                    .population_size(100)
                    .select(Tournament::new(3)?)
                    .crossover(SimulatedBinaryCrossover::new(15.0)?)
                    .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
                    .minimize()
                    .seed(seed)
                    .build()?;
                Engine::new(ga, problem).stop_when(stop()).run()?
            } else {
                let swarm = Pso::builder(problem.representation())
                    .population_size(40)
                    .minimize()
                    .seed(seed)
                    .build()?;
                Engine::new(swarm, problem).stop_when(stop()).run()?
            };
            if outcome.stop_reason() == StopReason::Target {
                *reached += 1;
            }
            let end = outcome.best_genome();
            let value = outcome.best_fitness().score().expect("valid");
            let ring = ((end[0] * end[0] + end[1] * end[1]).sqrt() / PI).round() as usize;
            if rings.len() <= ring {
                rings.resize(ring + 1, None);
            }
            let entry = rings[ring].get_or_insert(Ring {
                value,
                runs: [0; 3],
            });
            entry.value = entry.value.min(value);
            entry.runs[a] += 1;
        }
    }

    println!(
        "Schaffer F6: minimum 0 at the origin, {SEEDS} seeds, {BUDGET} evaluations at most per \
         run"
    );
    println!("runs ending near        best value  SHADE  PSO  GA");
    for (k, ring) in rings.iter().enumerate() {
        let Some(ring) = ring else { continue };
        let near = match k {
            0 => "the minimum, r = 0".to_string(),
            _ => format!("the ring at r = {:.2}", k as f64 * PI),
        };
        let [shade, swarm, ga] = ring.runs;
        println!(
            "{near:<22}  {:>10.6}  {shade:>5}  {swarm:>3}  {ga:>2}",
            ring.value
        );
    }
    let [shade, swarm, ga] = reached;
    println!(
        "{:<34}  {shade:>5}  {swarm:>3}  {ga:>2}",
        "error below 1e-6"
    );
    trace.write();
    Ok(())
}
