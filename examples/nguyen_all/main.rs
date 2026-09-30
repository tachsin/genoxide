//! Nguyen-1 to 12: all twelve of Nguyen's symbolic regression problems, by genetic programming,
//! with and without linear scaling.
//!
//! Each problem is run twice with the same search, the one of the Nguyen-1, 5 and 9 pages: eight
//! islands of 500 trees of Koza's functions, with subtree crossover and mutation, for at most 200
//! generations. Once on the RMSE of the tree itself, and once after linear scaling (a + b × tree,
//! with a and b fitted by least squares). A problem is recovered when the error is at the level of
//! rounding on the training and the test points; the table gives the generations it took, or the
//! test error of the best tree when it wasn't.
//!
//! ```text
//! cargo run --release --example nguyen_all
//! ```

use genoxide::gp::regression::Regression;
use genoxide::gp::regression::problems::{RegressionProblem, all};
use genoxide::gp::{Gp, SubtreeCrossover, SubtreeMutation};
use genoxide::prelude::*;

// the islands, their trees, and the generations between migrations
const ISLANDS: u64 = 8;
const POPULATION: usize = 500;
const INTERVAL: u64 = 10;

fn main() -> Result<()> {
    println!("Nguyen-1 to 12, {ISLANDS} islands of {POPULATION} trees for at most 200 generations");
    println!("recovered: the generation of exact recovery; else the best tree's test RMSE\n");
    println!(
        "{:<10} {:<32} {:>18} {:>18}",
        "problem", "target", "tree itself", "linear scaling"
    );
    let problems = all()
        .into_iter()
        .filter(|problem| problem.name().starts_with("Nguyen"));
    for problem in problems {
        let regression = problem.regression();
        let without = run(problem.as_ref(), regression.clone().linear_scaling(false))?;
        let with = run(problem.as_ref(), regression.clone())?;
        println!(
            "{:<10} {:<32} {without:>18} {with:>18}",
            problem.name(),
            problem.formula()
        );
    }
    Ok(())
}

// a run of the search on the problem, as "recovered at 7" or "test RMSE 1.2e-3"
fn run(problem: &dyn RegressionProblem, regression: Regression) -> Result<String> {
    let dataset = problem.dataset();
    let (training, test) = (dataset.training(), dataset.test().expect("a test sample"));
    // exact recovery: an error of at most 1e-10 of the values' standard deviation
    let tolerance = 1e-10 * training.deviation();
    // Koza's limits and initialization: depth 17, ramped half-and-half of depths 2 to 6
    let gp = Gp::builder(problem.primitives().clone()).build()?;
    let islands = (0..ISLANDS)
        .map(|island| {
            let seed = 100 + island;
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
    // parallel evaluation: the same results as without, sooner
    let outcome = Engine::new(islands, regression.clone())
        .stop_when(Stop::target(tolerance).or(Stop::generations(200)))
        .parallel(true)
        .run()?;
    let best = outcome.best_genome();
    let test_error = regression.error(best, test);
    let recovered = test_error.is_some_and(|error| error <= 1e-10 * test.deviation());
    Ok(match (outcome.stop_reason(), recovered, test_error) {
        (StopReason::Target, true, _) => format!("recovered at {}", outcome.generations()),
        (_, _, Some(error)) => format!("test RMSE {error:.1e}"),
        (_, _, None) => "test not finite".to_string(),
    })
}
