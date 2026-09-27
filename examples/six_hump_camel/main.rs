//! Six-hump camel: find the six minima of the six-hump camel-back function, two of them global, by
//! restarting a local search from random points.
//!
//! Each search is a hill climber with Gaussian steps; it ends in the minimum whose basin it
//! started in. The searches that end close together are grouped, and the table gives each group's
//! best point and value. The function, its bounds and its global minima come from genoxide's
//! `problems::SixHumpCamel`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example six_hump_camel
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Problem, SixHumpCamel};

const SEARCHES: u64 = 30;
const STEPS: u64 = 1_000;

// a group of searches that ended close together: its best point and value, and its searches
struct Minimum {
    point: [f64; 2],
    value: f64,
    searches: u64,
}

fn main() -> Result<()> {
    let problem = SixHumpCamel;
    let optimum = problem.optimum().expect("known");
    let bounds = problem.representation().bounds().to_vec();
    let width = [
        bounds[0].end() - bounds[0].start(),
        bounds[1].end() - bounds[1].start(),
    ];
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
        let point = [end[0], end[1]];
        let value = outcome.best_fitness().score().expect("valid");
        // the same minimum: within 2% of the bounds' width in both genes
        let close = |minimum: &&mut Minimum| {
            (0..2).all(|i| (point[i] - minimum.point[i]).abs() <= 0.02 * width[i])
        };
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
    // by value, to 5 significant digits, then by the first gene
    let key = |minimum: &Minimum| (significant(minimum.value), minimum.point[0]);
    minima.sort_by(|a, b| {
        let (a, b) = (key(a), key(b));
        a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1))
    });

    println!(
        "{SEARCHES} local searches from random points in [{:.0}, {:.0}] x [{:.0}, {:.0}], {STEPS} \
         steps each",
        bounds[0].start(),
        bounds[0].end(),
        bounds[1].start(),
        bounds[1].end()
    );
    println!("minimum reached   searches  best value");
    for minimum in &minima {
        let global = if (minimum.value - optimum.value()).abs() < 1e-3 {
            "  global"
        } else {
            ""
        };
        println!(
            "({:>6.3}, {:>6.3})  {:>8}  {:>10}{global}",
            minimum.point[0],
            minimum.point[1],
            minimum.searches,
            five_digits(minimum.value)
        );
    }
    trace.write();
    Ok(())
}

// the value rounded to 5 significant digits
fn significant(value: f64) -> f64 {
    format!("{value:.4e}").parse().expect("a number")
}

// 5 significant digits, trailing zeros kept, e.g. 0.39789, 3.0000 or 840.00
fn five_digits(value: f64) -> String {
    let rounded = significant(value);
    let magnitude = rounded.abs().log10().floor() as i32;
    let decimals = (4 - magnitude).max(0) as usize;
    format!("{rounded:.decimals$}")
}
