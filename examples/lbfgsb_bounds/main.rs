//! L-BFGS-B with a bound that cuts the valley: minimize Rosenbrock's function in the box
//! [−2, 0.5] × [−1, 3], whose minimum (1, 1) lies outside it. The minimum in the box is on its
//! edge, at (0.5, 0.25), where f = 0.25, and L-BFGS-B lands on it exactly.
//!
//! Then Nelder-Mead in the same box, for contrast: it approaches the bound without reaching it.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example lbfgsb_bounds
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Rosenbrock;

fn main() -> Result<()> {
    let real = Real::new([-2.0..=0.5, -1.0..=3.0])?;
    let start = Reals::from(vec![-1.2, 1.0]);
    let lbfgsb = Lbfgsb::builder(real.clone())
        .initial_genome(start.clone())
        .minimize()
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let mut rows = Vec::new();
    let mut criterion = None;
    let mut engine = Engine::new(lbfgsb, Rosenbrock::new(2))
        .stop_when(Stop::evaluations(1_000))
        .on_generation(|snapshot| trace.record(snapshot))
        .control(|lbfgsb: &mut Lbfgsb, progress| {
            let x = lbfgsb.population()[0].genome();
            let value = lbfgsb.population()[0].fitness().and_then(Fitness::score);
            rows.push(format!(
                "{:>5}  {:>11}  {:>9.6}  {:>9.6}  {:>9.6}  {:>18}",
                progress.generation(),
                progress.evaluations(),
                x[0],
                x[1],
                value.expect("valid"),
                scientific(lbfgsb.projected_gradient())
            ));
            criterion = lbfgsb.converged();
            Ok(())
        });
    let outcome = engine.run()?;
    drop(engine);

    println!(
        "Rosenbrock's function in [-2, 0.5] x [-1, 3], from (-1.2, 1); its minimum (1, 1) is outside"
    );
    println!("L-BFGS-B, the current point after each round");
    println!("round  evaluations         x1         x2          f  projected gradient");
    for row in &rows {
        println!("{row}");
    }
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(criterion, Some(lbfgsb::Criterion::ProjectedGradient));
    let x = outcome.best_genome();
    // the gradient at the end: −400 x₁ (x₂ − x₁²) − 2 (1 − x₁), and 200 (x₂ − x₁²)
    let valley = x[1] - x[0] * x[0];
    let gradient = [-400.0 * x[0] * valley + 2.0 * (x[0] - 1.0), 200.0 * valley];
    println!(
        "L-BFGS-B: converged at ({:?}, {:?}), f = {:?}, after {} evaluations",
        x[0],
        x[1],
        outcome.best_fitness().score().expect("valid"),
        outcome.evaluations()
    );
    println!(
        "the gradient there is ({:?}, {:?}): f falls only beyond the bound x1 = 0.5",
        gradient[0], gradient[1]
    );

    let nelder_mead = NelderMead::builder(real)
        .initial_genome(start)
        .minimize()
        .build()?;
    let contrast = Engine::new(nelder_mead, Rosenbrock::new(2))
        .stop_when(Stop::evaluations(10_000))
        .run()?;
    let y = contrast.best_genome();
    println!(
        "Nelder-Mead, for contrast: ({:?}, {:?}), f = {:?}, after {} evaluations",
        y[0],
        y[1],
        contrast.best_fitness().score().expect("valid"),
        contrast.evaluations()
    );
    trace.write();
    Ok(())
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
