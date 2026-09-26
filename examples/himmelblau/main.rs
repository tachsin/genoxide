//! Himmelblau: find the four global minima of a two-dimensional function by restarting a local
//! search from random points.
//!
//! Each search is a hill climber with Gaussian steps; it ends in the minimum whose basin it
//! started in. The known minima come from genoxide's `problems::Himmelblau`.
//!
//! ```text
//! cargo run --release --example himmelblau
//! ```

use genoxide::prelude::*;
use genoxide::problems::{Himmelblau, Problem};

const SEARCHES: u64 = 20;

fn main() -> Result<()> {
    let problem = Himmelblau;
    let optimum = problem.optimum().expect("known");
    let minima = optimum.solutions();
    // per known minimum: the searches that ended nearest it, and the best and worst values they
    // reached
    let mut found = vec![(0, f64::INFINITY, 0.0f64); minima.len()];
    for seed in 1..=SEARCHES {
        let search = LocalSearch::builder(problem.representation())
            .neighbor(GaussianMutation::per_gene(1.0, 0.001)?)
            .neighbors(10)
            .acceptance(Acceptance::Improving)
            .minimize()
            .seed(seed)
            .build()?;
        let outcome = Engine::new(search, problem)
            .stop_when(Stop::generations(1_000))
            .run()?;
        let end = outcome.best_genome();
        let nearest = (0..minima.len())
            .min_by(|&a, &b| distance(end, &minima[a]).total_cmp(&distance(end, &minima[b])))
            .expect("four minima");
        let value = outcome.best_fitness().score().expect("valid");
        let (searches, best, worst) = &mut found[nearest];
        *searches += 1;
        *best = best.min(value);
        *worst = worst.max(value);
    }

    println!("{SEARCHES} local searches from random points in [-5, 5] x [-5, 5]");
    println!("minimum                  searches  values reached");
    for (minimum, (searches, best, worst)) in minima.iter().zip(found) {
        println!(
            "({:>9.6}, {:>9.6})  {searches:>8}  {} to {}",
            minimum[0],
            minimum[1],
            scientific(best),
            scientific(worst)
        );
    }
    Ok(())
}

fn distance(a: &Reals, b: &Reals) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
