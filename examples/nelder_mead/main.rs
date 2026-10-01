//! Nelder-Mead: minimize Rosenbrock's function in two dimensions from the classic start
//! (−1.2, 1), with the simplex method, to its minimum at (1, 1).
//!
//! The simplex is a triangle that reflects, expands, contracts and shrinks down the curved valley
//! of the function. The run stops when the triangle has collapsed on the minimum. Then the same
//! run with speculative asks: the same triangles in fewer rounds of evaluations.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example nelder_mead
//! ```

mod trace;

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::Rosenbrock;

// a row of the table every this many rounds
const EVERY: u64 = 20;

fn main() -> Result<()> {
    let real = Real::new([-2.0..=2.0, -1.0..=3.0])?;
    let nelder_mead = |speculative| {
        NelderMead::builder(real.clone())
            .initial_genome(Reals::from(vec![-1.2, 1.0]))
            .speculative(speculative)
            .minimize()
            .build()
    };
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let mut rows = Vec::new();
    let outcome = Engine::new(nelder_mead(false)?, Rosenbrock::new(2))
        .stop_when(Stop::evaluations(10_000))
        .on_generation(|snapshot| {
            trace.record(snapshot);
            rows.push(row(snapshot, &real));
        })
        .run()?;

    println!("Rosenbrock's function in two dimensions, from (-1.2, 1) where f = 24.2");
    println!("Nelder-Mead with Gao and Han's coefficients, a first simplex of 0.1 of each range");
    println!("round  evaluations  best value  simplex size");
    let last = rows.len() - 1;
    for (round, row) in rows.iter().enumerate() {
        if (round as u64).is_multiple_of(EVERY) || round == last {
            println!("{row}");
        }
    }
    let best = outcome.best_genome();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    println!(
        "converged after {} rounds and {} evaluations, at ({:.10}, {:.10}), f = {}",
        outcome.generations(),
        outcome.evaluations(),
        best[0],
        best[1],
        scientific(outcome.best_fitness().score().expect("valid"))
    );

    let speculative = Engine::new(nelder_mead(true)?, Rosenbrock::new(2))
        .stop_when(Stop::evaluations(10_000))
        .run()?;
    let same = if speculative.best_genome() == best {
        "the same end"
    } else {
        "another end"
    };
    println!(
        "speculative asks: {same} after {} rounds and {} evaluations",
        speculative.generations(),
        speculative.evaluations()
    );
    trace.write();
    Ok(())
}

// a row of the table: the round, the evaluations, the best value and the simplex size, the
// largest difference between a vertex and the best one in a gene, as a fraction of its range
fn row(snapshot: &Snapshot<'_, Reals>, real: &Real) -> String {
    let progress = snapshot.progress();
    let vertices = snapshot.population().as_slice();
    let best = vertices[0].genome();
    let mut size = 0.0f64;
    for vertex in &vertices[1..] {
        for ((x, b), range) in vertex.genome().iter().zip(best.iter()).zip(real.bounds()) {
            size = size.max((x - b).abs() / (range.end() - range.start()));
        }
    }
    let value = progress.best().and_then(Fitness::score).expect("valid");
    format!(
        "{:>5}  {:>11}  {:>10}  {:>12}",
        progress.generation(),
        progress.evaluations(),
        scientific(value),
        scientific(size)
    )
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
