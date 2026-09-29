//! N-Queens: place N queens on an N×N board so that no two attack each other.
//!
//! A permutation genome puts one queen in each row and each column (queen `row` is in column
//! `order[row]`), so only the diagonals can conflict. Permutations have no position-wise
//! crossover, so this uses (μ+λ) with swap mutation only.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example n_queens_128
//! ```

mod trace;

use genoxide::prelude::*;

const N: usize = 128;

// the number of pairs of queens on the same diagonal
fn conflicts(order: &Order) -> f64 {
    let mut diagonals = [0usize; 2 * N];
    let mut anti_diagonals = [0usize; 2 * N];
    for (row, &column) in order.iter().enumerate() {
        diagonals[row + N - column] += 1;
        anti_diagonals[row + column] += 1;
    }
    let pairs = |count: &usize| count * count.saturating_sub(1) / 2;
    (diagonals.iter().map(pairs).sum::<usize>() + anti_diagonals.iter().map(pairs).sum::<usize>())
        as f64
}

fn main() -> Result<()> {
    let ga = Ga::builder(Permutation::new(N)?)
        .population_size(20)
        .select(Tournament::new(2)?)
        .crossover(NoCrossover)
        .mutate(SwapMutation::new())
        .scheme(Scheme::MuPlusLambda { lambda: 20 })
        .minimize()
        .seed(1)
        .build()?;

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(ga, conflicts)
        .stop_when(Stop::target(0.0).or(Stop::generations(50_000)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    println!(
        "{} conflicts after {} generations and {} evaluations",
        outcome.best_fitness(),
        outcome.generations(),
        outcome.evaluations()
    );
    println!("columns {:?}", &outcome.best_genome()[..]);
    trace.write();
    Ok(())
}
