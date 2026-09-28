//! Hartmann 3-D: find the local minima of Hartmann's function in 3 dimensions by restarting a
//! local search from random points.
//!
//! Each search is a hill climber with Gaussian steps; it ends in the minimum whose basin it
//! started in. The searches that end close together are grouped, and the table gives each group's
//! best point and value. The function, its bounds and its best known minimum come from genoxide's
//! `problems::Hartmann3`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example hartmann3
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Hartmann3, Problem};

const SEARCHES: u64 = 30;
const STEPS: u64 = 1_000;

// a group of searches that ended close together: its best point and value, and its searches
struct Minimum {
    point: [f64; 3],
    value: f64,
    searches: u64,
}

fn main() -> Result<()> {
    let problem = Hartmann3;
    let optimum = problem.optimum().expect("known");
    let mut minima: Vec<Minimum> = Vec::new();
    // with GENOXIDE_TRACE=<file>, a trace of the searches for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    for seed in 1..=SEARCHES {
        let search = LocalSearch::builder(problem.representation())
            .neighbor(GaussianMutation::per_gene(1.0, 0.001)?)
            .neighbors(10)
            .acceptance(Acceptance::Improving)
            .minimize()
            .seed(seed)
            .build()?;
        let outcome = Engine::new(search, problem)
            .stop_when(Stop::generations(STEPS))
            .on_generation(|snapshot| trace.record(snapshot))
            .run()?;
        let end = outcome.best_genome();
        let point = [end[0], end[1], end[2]];
        let value = outcome.best_fitness().score().expect("valid");
        // the same minimum: within 0.02 in every gene (the bounds are [0, 1])
        let close =
            |minimum: &&mut Minimum| (0..3).all(|i| (point[i] - minimum.point[i]).abs() <= 0.02);
        match minima.iter_mut().find(close) {
            Some(minimum) => {
                minimum.searches += 1;
                if value < minimum.value {
                    (minimum.point, minimum.value) = (point, value);
                }
            }
            None => minima.push(Minimum {
                point,
                value,
                searches: 1,
            }),
        }
    }
    // by value, to 6 significant digits, then by the first gene
    let key = |minimum: &Minimum| (significant(minimum.value), minimum.point[0]);
    minima.sort_by(|a, b| {
        let (a, b) = (key(a), key(b));
        a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1))
    });

    println!("{SEARCHES} local searches from random points in [0, 1]^3, {STEPS} steps each");
    println!("minimum reached          searches  best value");
    for minimum in &minima {
        let global = if (minimum.value - optimum.value()).abs() < 1e-3 {
            "  global"
        } else {
            ""
        };
        let [x1, x2, x3] = minimum.point;
        println!(
            "({x1:.3}, {x2:.3}, {x3:.3})  {:>8}  {:>10.5}{global}",
            minimum.searches, minimum.value
        );
    }
    trace.write();
    Ok(())
}

// the value rounded to 6 significant digits
fn significant(value: f64) -> f64 {
    format!("{value:.5e}").parse().expect("a number")
}
