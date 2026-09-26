//! Travelling salesman: the shortest round trip through the 52 locations in Berlin of TSPLIB's
//! berlin52, whose optimal tour has length 7542.
//!
//! A permutation genome is the order of the visits. Local search with inversion neighbors (a
//! random 2-opt move: a reversed segment of the tour) and simulated annealing, which also accepts
//! worse tours, less and less often as the temperature cools.
//!
//! ```text
//! cargo run --release --example tsp_berlin52
//! ```

use genoxide::prelude::*;

// the coordinates of the locations, from berlin52.tsp
const LOCATIONS: [(f64, f64); 52] = [
    (565.0, 575.0),
    (25.0, 185.0),
    (345.0, 750.0),
    (945.0, 685.0),
    (845.0, 655.0),
    (880.0, 660.0),
    (25.0, 230.0),
    (525.0, 1000.0),
    (580.0, 1175.0),
    (650.0, 1130.0),
    (1605.0, 620.0),
    (1220.0, 580.0),
    (1465.0, 200.0),
    (1530.0, 5.0),
    (845.0, 680.0),
    (725.0, 370.0),
    (145.0, 665.0),
    (415.0, 635.0),
    (510.0, 875.0),
    (560.0, 365.0),
    (300.0, 465.0),
    (520.0, 585.0),
    (480.0, 415.0),
    (835.0, 625.0),
    (975.0, 580.0),
    (1215.0, 245.0),
    (1320.0, 315.0),
    (1250.0, 400.0),
    (660.0, 180.0),
    (410.0, 250.0),
    (420.0, 555.0),
    (575.0, 665.0),
    (1150.0, 1160.0),
    (700.0, 580.0),
    (685.0, 595.0),
    (685.0, 610.0),
    (770.0, 610.0),
    (795.0, 645.0),
    (720.0, 635.0),
    (760.0, 650.0),
    (475.0, 960.0),
    (95.0, 260.0),
    (875.0, 920.0),
    (700.0, 500.0),
    (555.0, 815.0),
    (830.0, 485.0),
    (1170.0, 65.0),
    (830.0, 610.0),
    (605.0, 625.0),
    (595.0, 360.0),
    (1340.0, 725.0),
    (1740.0, 245.0),
];
const OPTIMUM: f64 = 7542.0;

// TSPLIB's EUC_2D distance: the Euclidean distance, rounded to the nearest integer
fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (a.0 - b.0, a.1 - b.1);
    ((dx * dx + dy * dy).sqrt() + 0.5).floor()
}

fn main() -> Result<()> {
    let distances: Vec<Vec<f64>> = LOCATIONS
        .iter()
        .map(|&a| LOCATIONS.iter().map(|&b| distance(a, b)).collect())
        .collect();
    let tour_length = |order: &Order| {
        (0..order.len())
            .map(|i| distances[order[i]][order[(i + 1) % order.len()]])
            .sum::<f64>()
    };

    let search = LocalSearch::builder(Permutation::new(LOCATIONS.len())?)
        .neighbor(InversionMutation)
        .acceptance(Acceptance::Annealing {
            initial_temperature: 100.0,
            cooling: 0.99996,
        })
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(search, tour_length)
        .stop_when(Stop::target(OPTIMUM).or(Stop::evaluations(200_000)))
        .run()?;

    println!(
        "tour length {} after {} evaluations (the optimum: {OPTIMUM})",
        outcome.best_fitness(),
        outcome.evaluations()
    );
    // the tour from location 1, numbered from 1 as in TSPLIB
    let order = outcome.best_genome();
    let start = order.iter().position(|&i| i == 0).unwrap_or(0);
    let tour: Vec<usize> = order[start..]
        .iter()
        .chain(&order[..start])
        .map(|location| location + 1)
        .collect();
    println!("tour {tour:?}");
    Ok(())
}
