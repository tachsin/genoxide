//! Bayesian optimization of Branin's function: 30 evaluations chosen by a Gaussian process and the
//! log expected improvement, then the model's mean minimized by L-BFGS-B and evaluated once.
//!
//! The search comes within 1e-4 of one of the three global minima, and the polish of the model,
//! for one evaluation more, closes most of what is left.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example bayesian_optimization
//! ```

mod trace;

use genoxide::model::gp::GaussianProcess;
use genoxide::prelude::*;
use genoxide::problems::{Branin, Problem};
use std::cell::RefCell;

// the evaluations of the search: the initial design, then the points the model chooses
const EVALUATIONS: u64 = 30;

fn main() -> Result<()> {
    let problem = Branin;
    let optimum = problem.optimum().expect("known");
    let minimum = optimum.value();
    let bo = Bo::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    println!("Branin's function in [-5, 10] x [0, 15]: three global minima of {minimum:.6}");
    println!(
        "{} points of a Latin hypercube, then a point per step by log-EI on a Gaussian process",
        bo.initial_points()
    );
    println!("evaluation         x1         x2            f     f - f*");
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let trace = RefCell::new(trace::Trace::from_env());
    let mut printed = 0;
    let mut engine = Engine::new(bo, problem)
        .stop_when(Stop::evaluations(EVALUATIONS))
        .on_generation(|snapshot| {
            // the points evaluated in this generation
            let population = snapshot.population().as_slice();
            for (index, individual) in population.iter().enumerate().skip(printed) {
                let x = individual.genome();
                let value = individual
                    .fitness()
                    .and_then(Fitness::score)
                    .expect("valid");
                println!(
                    "{:>10} {:>10.6} {:>10.6} {:>12.6} {:>10}",
                    index + 1,
                    x[0],
                    x[1],
                    value,
                    scientific(value - minimum)
                );
            }
            printed = population.len();
        })
        .control(|bo, progress| {
            trace.borrow_mut().record(bo, progress);
            Ok(())
        });
    let outcome = engine.run()?;
    let best = outcome.best_genome().clone();
    let best_value = outcome.best_fitness().score().expect("valid");

    // a Gaussian process of every evaluation, its mean minimized from the best point
    let evaluated = engine.algorithm().population();
    let points: Vec<Reals> = evaluated.iter().map(|x| x.genome().clone()).collect();
    let values: Vec<f64> = evaluated
        .iter()
        .map(|x| x.fitness().and_then(Fitness::score).expect("valid"))
        .collect();
    // the engine borrows the trace in its control
    drop(engine);
    let model = GaussianProcess::builder(problem.representation()).fit(&points, &values)?;
    let mean = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        let mut variance_gradient = [0.0; 2];
        let prediction = model.predict_with_gradient(x, gradient, &mut variance_gradient);
        prediction.mean()
    });
    let lbfgsb = Lbfgsb::builder(problem.representation())
        .initial_genome(best)
        .minimize()
        .seed(1)
        .build()?;
    let polished = Engine::new(lbfgsb, mean)
        .stop_when(Stop::evaluations(1_000))
        .run()?;
    let x = polished.best_genome();
    // one evaluation of the function there
    let value = problem.evaluate(x);
    let nearest = optimum
        .solutions()
        .iter()
        .min_by(|a, b| distance(a, x).total_cmp(&distance(b, x)))
        .expect("three minima");
    println!(
        "the model of the {} evaluations, its mean minimized by L-BFGS-B from the best point:",
        points.len()
    );
    println!(
        "({:.6}, {:.6}): predicted {:.6}, evaluated {value:.6}, {} above the minimum at \
         ({:.6}, {:.6})",
        x[0],
        x[1],
        model.predict(x).mean(),
        scientific(value - minimum),
        nearest[0],
        nearest[1]
    );
    let best_value = best_value.min(value);
    println!(
        "{} evaluations: the best {} above the global minimum",
        points.len() + 1,
        scientific(best_value - minimum)
    );
    assert!(best_value - minimum <= 1e-4);
    trace.into_inner().write(x, value);
    Ok(())
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
