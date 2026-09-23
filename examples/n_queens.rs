//! N-Queens: place N queens on an N×N board so that no two attack each other.
//!
//! A permutation genome puts one queen in each row and each column (queen `row` is in column
//! `order[row]`), so only the diagonals can conflict. Permutations have no position-wise
//! crossover, so this uses (μ+λ) with swap mutation only.
//!
//! ```text
//! cargo run --release --example n_queens
//! ```

use genoxide::prelude::*;

const N: usize = 64;

// the number of pairs of queens on the same diagonal
fn conflicts(order: &Order) -> f64 {
    let mut diagonals = vec![0usize; 2 * N];
    let mut anti_diagonals = vec![0usize; 2 * N];
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

    let outcome = Engine::new(ga, conflicts)
        .stop_when(Stop::target(0.0).or(Stop::generations(50_000)))
        .run()?;

    println!(
        "{:?} after {} generations: {} conflicts",
        outcome.stop_reason(),
        outcome.generations(),
        outcome.best_fitness()
    );
    if N <= 16 {
        for &column in outcome.best_genome().iter() {
            let row: String = (0..N)
                .map(|c| if c == column { 'Q' } else { '.' })
                .collect();
            println!("{row}");
        }
    }
    Ok(())
}
