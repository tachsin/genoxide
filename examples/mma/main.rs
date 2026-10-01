//! MMA, the method of moving asymptotes, on a million variables: minimize Σ cⱼ/xⱼ subject to
//! Σ xⱼ ≤ V, a problem whose minimum is known in closed form, xⱼ = V √cⱼ / Σ √cₖ.
//!
//! Each iteration replaces the function and the constraint by convex, separable approximations
//! around the current point, and solves them through their dual, in the constraint's single
//! multiplier. The example prints the error of the best value and the largest error of a variable
//! as the run goes, and at the end the multiplier against its exact value.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example mma
//! ```

mod trace;

use genoxide::constraint::Constrained;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;

// the variables, and the volume: on average 1 per variable
const N: usize = 1_000_000;
const VOLUME: f64 = N as f64;
// a row of the table every this many iterations
const EVERY: u64 = 4;

fn main() -> Result<()> {
    // the costs cycle through 1 to 9
    let c: Vec<f64> = (0..N).map(|j| 1.0 + (j % 9) as f64).collect();
    // the minimum: the Lagrange conditions cⱼ/xⱼ² = λ and the volume give xⱼ = V √cⱼ / Σ √cₖ,
    // the value (Σ √cₖ)² / V and the multiplier λ = (Σ √cₖ)² / V²
    let roots: Vec<f64> = c.iter().map(|c| c.sqrt()).collect();
    let total: f64 = roots.iter().sum();
    let exact: Vec<f64> = roots.iter().map(|root| VOLUME * root / total).collect();
    let minimum: f64 = c.iter().zip(&exact).map(|(c, x)| c / x).sum();
    let multiplier = (total / VOLUME) * (total / VOLUME);

    // the value, its gradient, the constraint Σ xⱼ − V ≤ 0 and its gradient, all ones
    let problem = Constrained::differentiable(
        1,
        |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
            let (mut value, mut sum) = (0.0, 0.0);
            for j in 0..N {
                value += c[j] / x[j];
                gradient[j] = -c[j] / (x[j] * x[j]);
                jacobian[j] = 1.0;
                sum += x[j];
            }
            g[0] = sum - VOLUME;
            value
        },
    );
    // from xⱼ = 0.5, half the volume, in [0.01, 10]; the dual's sums in parallel, with the same
    // results as one after the other
    let mma = Mma::builder(Real::uniform(N, 0.01..=10.0)?)
        .initial_genome(Reals::from(vec![0.5; N]))
        .parallel_sums(true)
        .minimize()
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let mut rows = Vec::new();
    let mut engine = Engine::new(mma, problem)
        .stop_when(Stop::evaluations(200))
        .on_generation(|snapshot| {
            trace.record(snapshot, minimum);
            rows.push(row(snapshot, &exact));
        });
    let outcome = engine.run()?;
    let mma = engine.into_algorithm();

    println!("minimize the sum of c_j / x_j subject to the sum of x_j <= {VOLUME}");
    println!("{N} variables in [0.01, 10], c_j = 1 + (j mod 9), from x_j = 0.5");
    println!("iteration  best value        largest error of a variable");
    let last = rows.len() - 1;
    for (iteration, row) in rows.iter().enumerate() {
        if (iteration as u64).is_multiple_of(EVERY) || iteration == last {
            println!("{row}");
        }
    }
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    let criterion = match mma.converged() {
        Some(genoxide::algorithm::mma::Convergence::Kkt) => "the KKT conditions",
        _ => "the step",
    };
    println!(
        "converged by {criterion} after {} iterations and {} evaluations",
        mma.iterations(),
        outcome.evaluations()
    );
    let best = outcome.best_fitness();
    let value = best.score().expect("valid");
    println!("value {value:.10e}, the minimum {minimum:.10e}");
    let x = outcome.best_genome();
    let sum: f64 = x.iter().sum();
    println!(
        "largest relative error of a variable {}, the volume used {sum:.6} (violation {})",
        scientific(largest_error(x, &exact)),
        scientific(best.violation())
    );
    let found = mma.multipliers()[0];
    println!(
        "multiplier {found:.10}, the exact (sum of the roots of c_j)^2 / V^2 = {multiplier:.10}"
    );
    trace.write();
    Ok(())
}

// the largest relative error of a variable
fn largest_error(x: &[f64], exact: &[f64]) -> f64 {
    x.iter()
        .zip(exact)
        .map(|(x, exact)| ((x - exact) / exact).abs())
        .fold(0.0, f64::max)
}

// a row of the table: the iteration, the best value, and the largest relative error of a
// variable of the current point
fn row(snapshot: &Snapshot<'_, Reals>, exact: &[f64]) -> String {
    let progress = snapshot.progress();
    let value = progress.best().and_then(Fitness::score).expect("valid");
    let current = snapshot.population()[0].genome();
    format!(
        "{:>9}  {value:<16.10e}  {:>27}",
        progress.generation(),
        scientific(largest_error(current, exact))
    )
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
