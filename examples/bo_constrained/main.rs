//! Constrained Bayesian optimization of the toy problem of Gramacy et al. (2016): a linear
//! objective on [0, 1]² with two constraints whose values the fitness function gives one by one,
//! each modeled by a Gaussian process. The search maximizes the log expected improvement over the
//! best feasible point plus the logarithm of the probability of feasibility, and reaches the
//! global minimum, on the boundary of a wavy constraint, to within 1e-5. Then, as a contrast, the
//! same search told only the total violation, which it ignores.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example bo_constrained
//! ```

mod trace;

use genoxide::constraint::Constrained;
use genoxide::math::sin;
use genoxide::prelude::*;
use std::cell::RefCell;
use std::f64::consts::PI;

// the global minimum and its point (tests/reference/gramacy_toy.py, with mpmath)
const MINIMUM: f64 = 0.599_788_052_010_067_6;
const MINIMIZER: [f64; 2] = [0.195_122_683_472_071_76, 0.404_665_368_537_995_8];
// how close to the minimum, and the evaluations the search may take at most
const TOLERANCE: f64 = 1e-5;
const BUDGET: u64 = 60;

// x₁ + x₂, and the values of the two constraints g(x) <= 0 into `g`
fn toy(x: &Reals, g: &mut [f64]) -> f64 {
    g[0] = 1.5 - x[0] - 2.0 * x[1] - 0.5 * sin(2.0 * PI * (x[0] * x[0] - 2.0 * x[1]));
    g[1] = x[0] * x[0] + x[1] * x[1] - 1.5;
    x[0] + x[1]
}

fn main() -> Result<()> {
    println!("Gramacy et al.'s toy problem: minimize x1 + x2 on [0, 1]^2 subject to");
    println!(
        "  c1 = 1.5 - x1 - 2 x2 - sin(2 pi (x1^2 - 2 x2)) / 2 <= 0, c2 = x1^2 + x2^2 - 1.5 <= 0"
    );
    println!(
        "global minimum {MINIMUM:.6} at ({:.6}, {:.6}), on the boundary c1 = 0",
        MINIMIZER[0], MINIMIZER[1]
    );
    println!("6 points of a Latin hypercube, then a point per step by log-EI x P(feasible)");
    println!(
        "evaluation        x1        x2         f         c1         c2  best feasible f - f*"
    );
    let bo = Bo::builder(Real::uniform(2, 0.0..=1.0)?)
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let trace = RefCell::new(trace::Trace::from_env());
    let mut printed = 0;
    let mut best = f64::INFINITY;
    let mut engine = Engine::new(bo, Constrained::new(2, toy))
        .stop_when(Stop::target(MINIMUM + TOLERANCE).or(Stop::evaluations(BUDGET)))
        .control(|bo, progress| {
            // the points evaluated in this generation
            for index in printed..bo.population().len() {
                let individual = &bo.population().as_slice()[index];
                let (x, fitness) = (
                    individual.genome(),
                    individual.fitness().expect("evaluated"),
                );
                let g = bo.constraint_values(index);
                if fitness.is_feasible() {
                    best = best.min(fitness.score().expect("valid"));
                }
                let gap = if best.is_finite() {
                    format!("{:.1e}", best - MINIMUM)
                } else {
                    "none yet".to_string()
                };
                println!(
                    "{:>10} {:>9.6} {:>9.6} {:>9.6} {:>10.6} {:>10.6} {gap:>20}",
                    index + 1,
                    x[0],
                    x[1],
                    fitness.score().expect("valid"),
                    g[0],
                    g[1]
                );
            }
            printed = bo.population().len();
            trace.borrow_mut().record(bo, progress);
            Ok(())
        });
    let outcome = engine.run()?;
    drop(engine);
    let x = outcome.best_genome();
    let value = outcome.best_fitness().score().expect("valid");
    assert!(outcome.best_fitness().is_feasible());
    let distance = (x[0] - MINIMIZER[0]).hypot(x[1] - MINIMIZER[1]);
    println!(
        "{} evaluations: the best feasible point ({:.6}, {:.6}), {value:.6}, {:.1e} above the \
         minimum, {distance:.1e} from its point",
        outcome.evaluations(),
        x[0],
        x[1],
        value - MINIMUM
    );
    assert!(value - MINIMUM <= TOLERANCE);
    trace.into_inner().write();

    // the contrast: the same problem as (score, violation), without the constraints' values
    let violation = |x: &Reals| {
        let mut g = [0.0; 2];
        let score = toy(x, &mut g);
        (score, g[0].max(0.0) + g[1].max(0.0))
    };
    let blind = Bo::builder(Real::uniform(2, 0.0..=1.0)?)
        .minimize()
        .seed(1)
        .build()?;
    let mut engine =
        Engine::new(blind, violation).stop_when(Stop::evaluations(outcome.evaluations()));
    let contrast = engine.run()?;
    let feasible = engine
        .algorithm()
        .population()
        .iter()
        .filter(|individual| individual.fitness().is_some_and(Fitness::is_feasible))
        .count();
    let fitness = contrast.best_fitness();
    println!(
        "without the constraints' values: {feasible} feasible points of {}, the best {}",
        contrast.evaluations(),
        if fitness.is_feasible() {
            format!(
                "{:.6}, {:.1e} above the minimum",
                fitness.score().expect("valid"),
                fitness.score().expect("valid") - MINIMUM
            )
        } else {
            format!("infeasible, by {:.1e}", fitness.violation())
        }
    );
    Ok(())
}
