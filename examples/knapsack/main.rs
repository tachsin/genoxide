//! 0/1 knapsack: choose items with the highest total value that fit in the knapsack.
//!
//! Shows a constraint with Deb's feasibility rules: the fitness function returns the value and how
//! far the weight exceeds the capacity, so overweight selections still guide the search towards
//! the feasible ones. The result is checked against the optimum found by dynamic programming.
//!
//! ```text
//! cargo run --release --example knapsack
//! ```

use genoxide::prelude::*;

// (weight, value)
const ITEMS: [(u32, u32); 20] = [
    (23, 92),
    (31, 57),
    (29, 49),
    (44, 68),
    (53, 60),
    (38, 43),
    (63, 67),
    (85, 84),
    (89, 87),
    (82, 72),
    (12, 31),
    (17, 29),
    (41, 52),
    (35, 38),
    (27, 44),
    (58, 61),
    (19, 26),
    (46, 55),
    (71, 70),
    (33, 41),
];
const CAPACITY: u32 = 400;

// the total weight and value of the selected items
fn totals(selection: &Bits) -> (u32, u32) {
    let (mut weight, mut value) = (0, 0);
    for (item, selected) in selection.iter().enumerate() {
        if selected {
            weight += ITEMS[item].0;
            value += ITEMS[item].1;
        }
    }
    (weight, value)
}

// the total value, and how much the weight exceeds the capacity (0 if the items fit)
fn value(selection: &Bits) -> (f64, f64) {
    let (weight, value) = totals(selection);
    (
        f64::from(value),
        constraint::at_most(f64::from(weight), f64::from(CAPACITY)),
    )
}

// the best value that fits, by dynamic programming over the capacities
fn optimum() -> u32 {
    let mut best = [0; CAPACITY as usize + 1];
    for (weight, value) in ITEMS {
        for capacity in (weight as usize..=CAPACITY as usize).rev() {
            best[capacity] = best[capacity].max(best[capacity - weight as usize] + value);
        }
    }
    best[CAPACITY as usize]
}

fn main() -> Result<()> {
    let ga = Ga::builder(Binary::new(ITEMS.len())?)
        .population_size(60)
        .select(Tournament::new(3)?)
        .crossover(PointCrossover::two_point())
        .mutate(BitFlip::per_gene(1.0 / ITEMS.len() as f64)?)
        .seed(7)
        .build()?;

    let outcome = Engine::new(ga, value)
        .stop_when(Stop::stagnation(200).or(Stop::generations(2_000)))
        .run()?;

    let best = outcome.best_genome();
    let items: Vec<usize> = (0..ITEMS.len())
        .filter(|&item| best.get(item) == Some(true))
        .collect();
    let (weight, value) = totals(best);
    println!("items {items:?}");
    println!("value {value}, weight {weight} of {CAPACITY}");
    println!(
        "after {} evaluations; the optimum is {}",
        outcome.evaluations(),
        optimum()
    );
    Ok(())
}
