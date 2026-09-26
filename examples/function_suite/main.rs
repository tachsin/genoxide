//! Function suite: CMA-ES, SHADE and PSO on twelve classic test functions in 10 dimensions.
//!
//! Each algorithm has a budget of 10,000 evaluations per dimension and stops early within 1e-8
//! of the known minimum. The table gives the error to the minimum: the best value found minus
//! the minimum. The functions, their bounds and their minima come from genoxide's `problems`.
//!
//! ```text
//! cargo run --release --example function_suite
//! ```

use genoxide::prelude::*;
use genoxide::problems::{self, DynProblem};

const DIMENSIONS: usize = 10;
const BUDGET: u64 = 10_000 * DIMENSIONS as u64;

fn main() -> Result<()> {
    let functions: Vec<Box<dyn DynProblem>> = vec![
        problems::boxed(problems::Sphere::new(DIMENSIONS)),
        problems::boxed(problems::AxisParallelEllipsoid::new(DIMENSIONS)),
        problems::boxed(problems::Schwefel1_2::new(DIMENSIONS)),
        problems::boxed(problems::Zakharov::new(DIMENSIONS)),
        problems::boxed(problems::Rosenbrock::new(DIMENSIONS)),
        problems::boxed(problems::Rastrigin::new(DIMENSIONS)),
        problems::boxed(problems::Ackley::new(DIMENSIONS)),
        problems::boxed(problems::Griewank::new(DIMENSIONS)),
        problems::boxed(problems::Schwefel2_26::new(DIMENSIONS)),
        problems::boxed(problems::Levy::new(DIMENSIONS)),
        problems::boxed(problems::StyblinskiTang::new(DIMENSIONS)),
        problems::boxed(problems::Michalewicz::new(DIMENSIONS)),
    ];

    println!("Error to the minimum in {DIMENSIONS} dimensions, {BUDGET} evaluations at most");
    println!(
        "{:<22}{:>10}{:>10}{:>10}",
        "function", "CMA-ES", "SHADE", "PSO"
    );
    for problem in &functions {
        let minimum = problem.optimum().expect("known").value();
        let stop = || Stop::target(minimum + 1e-8).or(Stop::evaluations(BUDGET));
        let fitness = |x: &Reals| problem.evaluate(x);

        let cmaes = Cmaes::builder(problem.real())
            .restarts(cmaes::Restarts::Ipop)
            .minimize()
            .seed(1)
            .build()?;
        let cmaes = Engine::new(cmaes, fitness).stop_when(stop()).run()?;

        let shade = De::builder(problem.real()).minimize().seed(1).build()?;
        let shade = Engine::new(shade, fitness).stop_when(stop()).run()?;

        let pso = Pso::builder(problem.real())
            .population_size(40)
            .minimize()
            .seed(1)
            .build()?;
        let pso = Engine::new(pso, fitness).stop_when(stop()).run()?;

        let error = |outcome: &Outcome<Reals>| {
            let best = outcome.best_fitness().score().expect("valid");
            // rounding can put a solution a few ulps below the minimum
            scientific((best - minimum).max(0.0))
        };
        println!(
            "{:<22}{:>10}{:>10}{:>10}",
            problem.name(),
            error(&cmaes),
            error(&shade),
            error(&pso)
        );
    }
    Ok(())
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
