//! Accuracy against size: the trade-off between a formula's error and its size on Nguyen-7,
//! ln(x + 1) + ln(x² + 1), by NSGA-II on trees.
//!
//! Genetic programming rarely recovers Nguyen-7 exactly: the search settles on trees of hundreds
//! of nodes that fit the points to a few decimals. Minimizing the error and the size together, as
//! two objectives, gives instead the whole trade-off at once, the Pareto front: for each size, the
//! smallest error found, from a constant to large and accurate trees, with the small ones readable.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example accuracy_and_size
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::engine::FitnessFunction;
use genoxide::gp::regression::problems::{Nguyen7, RegressionProblem};
use genoxide::gp::{Gp, SubtreeCrossover, SubtreeMutation, Tree};
use genoxide::prelude::*;

const POPULATION: usize = 1000;
const GENERATIONS: u64 = 200;

fn main() -> Result<()> {
    // Nguyen's function set and sampling: 20 training points uniform in [0, 2], from a fixed seed,
    // and 100 test points from another; the RMSE after linear scaling
    let problem = Nguyen7::new();
    let regression = problem.regression().clone();
    let test = problem.dataset().test().expect("a test sample");
    println!("Nguyen-7: ln(x + 1) + ln(x^2 + 1) from 20 points in [0, 2]");
    println!(
        "NSGA-II, {POPULATION} trees for {GENERATIONS} generations, minimizing the RMSE and the size\n"
    );

    // Koza's limits and initialization: depth 17, ramped half-and-half of depths 2 to 6
    let gp = Gp::builder(problem.primitives().clone()).build()?;
    let initial = gp.ramped_half_and_half(POPULATION, &mut StreamRng::seed_from_u64(1))?;
    let nsga2 = Nsga2::builder(gp, [Minimize, Minimize])
        .population_size(POPULATION)
        .initial_genomes(initial)
        .crossover(SubtreeCrossover::new())
        .mutate(SubtreeMutation::new())
        .crossover_rate(0.9)
        .mutation_rate(0.1)
        .seed(1)
        .build()?;
    // the two objectives: the training RMSE, and the number of nodes; a tree whose value isn't
    // finite at a point is invalid
    let objectives = |tree: &Tree| -> Option<[f64; 2]> {
        let error = regression.evaluate(tree)?;
        Some([error, tree.len() as f64])
    };
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga2, objectives)
        .stop_when(Stop::generations(GENERATIONS))
        .parallel(true)
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    // the front, from the smallest tree to the most accurate: one tree per point, since trees that
    // differ only in the order of their arguments have the same error and size
    let mut front: Vec<(&Tree, f64)> = outcome
        .front()
        .iter()
        .filter_map(|individual| Some((individual.genome(), individual.fitness()?.values()?[0])))
        .collect();
    front.sort_by(|(a, error_a), (b, error_b)| {
        a.len().cmp(&b.len()).then(error_a.total_cmp(error_b))
    });
    front.dedup_by(|(a, error_a), (b, error_b)| a.len() == b.len() && error_a == error_b);
    println!(
        "the Pareto front after {} evaluations: {} points",
        outcome.evaluations(),
        front.len()
    );
    println!(
        "{:>5} {:>9} {:>9}  a + b * (expression), up to 25 nodes",
        "size", "RMSE", "test RMSE"
    );
    for (tree, error) in front {
        let test_error = regression.error(tree, test).unwrap_or(f64::NAN);
        let expression = match regression.scaling(tree) {
            Some(scaling) if tree.len() <= 25 => format!(
                "{:.4} {} {:.4} * ({})",
                scaling.intercept,
                if scaling.slope < 0.0 { '-' } else { '+' },
                scaling.slope.abs(),
                tree.display(regression.primitives())
            ),
            _ => String::new(),
        };
        println!(
            "{:>5} {error:>9.2e} {test_error:>9.2e}  {expression}",
            tree.len()
        );
    }
    trace.write();
    Ok(())
}
