//! Nguyen-9: find the formula sin(x) + sin(y²) from 100 points of it, by genetic programming.
//!
//! The ninth of Nguyen's twelve symbolic regression problems (Uy et al. 2011): trees of Koza's
//! functions (+, −, ×, protected division, sin, cos, exp and a protected logarithm) and the
//! variables x and y, evolved by a genetic algorithm with subtree crossover and mutation, fitted to the
//! root mean squared error on the points. The run stops at exact recovery: an error at the level of
//! rounding, on the 100 training points and on 500 test points in [−1, 1]², with the expression
//! printed.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example nguyen_9
//! ```

mod trace;

use genoxide::gp::regression::problems::{Nguyen9, RegressionProblem};
use genoxide::gp::{Gp, SubtreeCrossover, SubtreeMutation, Tree};
use genoxide::prelude::*;

// the islands, their trees, and the generations between migrations
const ISLANDS: u64 = 8;
const POPULATION: usize = 500;
const INTERVAL: u64 = 10;

fn main() -> Result<()> {
    // Nguyen's function set and sampling: 100 training points uniform in [-1, 1]^2, from a fixed
    // seed, and 500 test points from another; the RMSE of the tree itself, without linear scaling
    let problem = Nguyen9::new();
    let regression = problem.regression().clone().linear_scaling(false);
    let dataset = problem.dataset();
    let (training, test) = (dataset.training(), dataset.test().expect("a test sample"));
    // exact recovery: an error of at most 1e-10 of the values' standard deviation
    let tolerance = 1e-10 * training.deviation();
    println!("Nguyen-9: sin(x) + sin(y^2) from 100 points in [-1, 1]^2");
    println!(
        "{ISLANDS} islands of {POPULATION} trees, until the RMSE is at most {tolerance:.2e}\n"
    );

    // Koza's limits and initialization: depth 17, ramped half-and-half of depths 2 to 6
    let gp = Gp::builder(problem.primitives().clone()).build()?;
    let set = gp.primitives().clone();
    let islands = (0..ISLANDS)
        .map(|island| {
            let seed = 100 + island;
            // Koza's even division among the depths and methods, without duplicates
            let initial =
                gp.ramped_half_and_half(POPULATION, &mut StreamRng::seed_from_u64(seed))?;
            Ga::builder(gp.clone())
                .population_size(POPULATION)
                .initial_genomes(initial)
                .select(Tournament::new(7)?)
                .crossover(SubtreeCrossover::new())
                .mutate(SubtreeMutation::new())
                .crossover_rate(0.9)
                .mutation_rate(0.1)
                .minimize()
                .seed(seed)
                .build()
        })
        .collect::<Result<Vec<_>>>()?;
    let islands = Islands::builder(islands)
        .interval(INTERVAL)
        .migrants(2)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env("nguyen_9", &problem, &regression, -1.0..=1.0);
    let outcome = Engine::new(islands, regression.clone())
        .stop_when(Stop::target(tolerance).or(Stop::generations(200)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best: &Tree = outcome.best_genome();
    let rmse = |sample| regression.error(best, sample).unwrap_or(f64::NAN);
    println!(
        "{:?} after {} generations and {} evaluations",
        outcome.stop_reason(),
        outcome.generations(),
        outcome.evaluations()
    );
    println!("RMSE on the 100 training points: {:.2e}", rmse(training));
    println!("RMSE on the 500 test points:     {:.2e}", rmse(test));
    println!(
        "\nthe expression, {} nodes of depth {}:\n{}",
        best.len(),
        best.depth(&set),
        best.display(&set)
    );
    trace.write();
    Ok(())
}
