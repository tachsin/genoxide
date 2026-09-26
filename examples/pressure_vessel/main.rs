//! Pressure vessel design (Sandgren, 1990): the cheapest cylindrical vessel with hemispherical
//! heads that holds 1,296,000 cubic inches, a constrained mixed discrete-continuous problem. The
//! best known cost is 6059.714335.
//!
//! The variables are the thickness of the shell and of the heads, multiples of 0.0625 inch, and
//! the inner radius and the length of the shell. Genes 0 and 1 are real numbers from 1 to 99,
//! rounded to the nearest whole number of 0.0625-inch plates. The fitness function returns the cost
//! and the violation of the four constraints, which Deb's feasibility rules compare. SHADE, a
//! differential evolution, searches the genes.
//!
//! ```text
//! cargo run --release --example pressure_vessel
//! ```

use genoxide::prelude::*;
use std::f64::consts::PI;

const BEST_KNOWN: f64 = 6059.714335;

// the design of genes `x`: the thicknesses of the shell and the heads, the radius and the length
fn design(x: &[f64]) -> [f64; 4] {
    [
        (x[0] + 0.5).floor() * 0.0625,
        (x[1] + 0.5).floor() * 0.0625,
        x[2],
        x[3],
    ]
}

// the cost of the material, forming and welding, and the violation of the constraints
fn cost(x: &Reals) -> (f64, f64) {
    let [shell, head, radius, length] = design(x);
    let cost = 0.6224 * shell * radius * length
        + 1.7781 * head * radius * radius
        + 3.1661 * shell * shell * length
        + 19.84 * shell * shell * radius;
    let volume = PI * radius * radius * length + 4.0 / 3.0 * PI * radius * radius * radius;
    // the volume constraint relative to the volume, of the order of the others
    let violation = constraint::at_least(shell, 0.0193 * radius)
        + constraint::at_least(head, 0.00954 * radius)
        + constraint::at_least(volume, 1_296_000.0) / 1_296_000.0
        + constraint::at_most(length, 240.0);
    (cost, violation)
}

fn main() -> Result<()> {
    let real = Real::new([1.0..=99.0, 1.0..=99.0, 10.0..=200.0, 10.0..=200.0])?;
    let de = De::builder(real).minimize().seed(1).build()?;
    let outcome = Engine::new(de, cost)
        .stop_when(Stop::evaluations(50_000))
        .run()?;

    let best = outcome.best_fitness();
    let [shell, head, radius, length] = design(outcome.best_genome());
    println!(
        "cost {:.6} after {} evaluations (the best known: {BEST_KNOWN})",
        best.score().unwrap_or(f64::NAN),
        outcome.evaluations()
    );
    println!("violation {:.6}", best.violation());
    println!("shell {shell:.4}, heads {head:.4}, radius {radius:.6}, length {length:.6}");
    Ok(())
}
