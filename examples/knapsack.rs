//! 0/1 knapsack: choose items with the highest total value that fit in the knapsack.
//!
//! Shows a constraint with Deb's feasibility rules: the fitness function returns the value and how
//! far the weight exceeds the capacity, so overweight selections still guide the search towards
//! the feasible ones. Also a hall of fame. The result is checked against the optimum found by
//! brute force.
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

// the total value, and how much the weight exceeds the capacity (0 if the items fit)
fn value(selection: &Bits) -> (f64, f64) {
    let (mut weight, mut value) = (0, 0);
    for (item, selected) in selection.iter().enumerate() {
        if selected {
            weight += ITEMS[item].0;
            value += ITEMS[item].1;
        }
    }
    (
        f64::from(value),
        constraint::at_most(f64::from(weight), f64::from(CAPACITY)),
    )
}

// the best value of the selections that fit, over all 2^20 selections
fn optimum() -> f64 {
    (0u32..1 << ITEMS.len())
        .map(|mask| value(&(0..ITEMS.len()).map(|item| mask >> item & 1 == 1).collect()))
        .filter(|&(_, violation)| violation == 0.0)
        .fold(0.0, |best, (value, _)| f64::max(best, value))
}

fn main() -> Result<()> {
    let ga = Ga::builder(Binary::new(ITEMS.len())?)
        .population_size(60)
        .select(Tournament::new(3)?)
        .crossover(PointCrossover::two_point())
        .mutate(BitFlip::per_gene(1.0 / ITEMS.len() as f64)?)
        .seed(7)
        .build()?;

    let mut hall_of_fame = HallOfFame::new(3)?;
    let outcome = Engine::new(ga, value)
        .stop_when(Stop::stagnation(200).or(Stop::generations(2_000)))
        .observe(&mut hall_of_fame)
        .run()?;

    println!("best selections:");
    for individual in hall_of_fame.individuals() {
        let value = individual.fitness().unwrap_or(Fitness::invalid());
        println!("  {} value {value}", individual.genome());
    }
    let optimum = optimum();
    println!(
        "\nfound {} after {} evaluations, the optimum is {optimum}",
        outcome.best_fitness(),
        outcome.evaluations()
    );
    Ok(())
}
