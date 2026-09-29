//! MW8: minimize three objectives over 15 variables, subject to one constraint, whose Pareto front
//! is the unit sphere in four bands, with η = 20 and η = 2.
//!
//! From genoxide's `multi::problems::Mw8`. Runs η = 20 and η = 2, and prints each final front's
//! size, its hypervolume with the objectives divided by the front's nadir point, as a share of that
//! of a sample of the optimal front with as many points as there are reference directions, and the
//! median and largest distance g of its solutions from the front; then the hypervolumes of the
//! whole front and of the sample.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example mw8
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{MultiProblem, Mw8};
use genoxide::prelude::*;

// the reference point of the hypervolume, with the objectives divided by the nadir point
const REFERENCE: [f64; 3] = [1.1, 1.1, 1.1];

fn main() -> Result<()> {
    let problem = Mw8::<3>::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // 91 reference directions: Das and Dennis's points with 12 divisions
    let directions = multi::das_dennis::<3>(12);
    // the sample: as many points of the optimal front, what a front of 91 solutions can be
    let sample = normalized(&problem.optimal_front(directions.len()).expect("known"));
    // NSGA-III with the paper's settings: polynomial mutation with η = 20, 600 generations
    let algorithm = Nsga3::builder(problem.representation(), [Minimize; 3], directions.clone())
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 15.0, 20.0)?)
        .seed(1)
        .build()?;
    run(problem, "η = 20", algorithm, 600, &sample, &mut trace)?;
    // NSGA-III with Deb and Jain's crossover (η = 30) and mutation with η = 2, for longer
    let algorithm = Nsga3::builder(problem.representation(), [Minimize; 3], directions.clone())
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 15.0, 2.0)?)
        .seed(1)
        .build()?;
    run(problem, "η = 2", algorithm, 2000, &sample, &mut trace)?;
    let whole = normalized(&problem.optimal_front(3_000).expect("known"));
    println!(
        "the whole front: hypervolume {:.4}; the sample of {} of its points: {:.4}",
        hypervolume(&whole, &REFERENCE, &[Minimize; 3]),
        sample.len(),
        hypervolume(&sample, &REFERENCE, &[Minimize; 3])
    );
    trace.write(&problem);
    Ok(())
}

// runs `algorithm` for `generations`, and prints its final front's size, its hypervolume, as a
// share of the sample's, and the median and largest distance g of its solutions from the front
fn run<A>(
    problem: Mw8<3>,
    name: &'static str,
    algorithm: A,
    generations: u64,
    sample: &[[f64; 3]],
    trace: &mut trace::Trace,
) -> Result<()>
where
    A: MultiObjectiveAlgorithm<3, Genome = Reals>,
{
    let mut record = trace.front(name);
    let outcome = MultiEngine::new(algorithm, problem)
        .stop_when(Stop::generations(generations))
        .on_generation(|snapshot| record(snapshot))
        .run()?;
    let front = outcome.front_values();
    let volume = hypervolume(&normalized(&front), &REFERENCE, &[Minimize; 3]);
    let percent = 100.0 * volume / hypervolume(sample, &REFERENCE, &[Minimize; 3]);
    // g₂, from f₁² + f₂² + f₃² = g₂²: the median (the middle value, or the upper of the two middle
    // ones) and the largest
    let mut g: Vec<f64> = normalized(&front)
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
        .collect();
    g.sort_by(f64::total_cmp);
    let (median, largest) = (g[g.len() / 2], g[g.len() - 1]);
    println!(
        "{name}, {generations} generations: {} solutions, hypervolume {volume:.4}, \
         {percent:.1}% of the sample's, g {median:.5} (median) to {largest:.5}",
        front.len()
    );
    Ok(())
}

// the objectives divided by the front's nadir point (its ideal point is the origin)
fn normalized(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let problem = Mw8::<3>::default();
    let ideal = problem.ideal_point().expect("known");
    let nadir = problem.nadir_point().expect("known");
    let scale = |p: &[f64; 3]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}
