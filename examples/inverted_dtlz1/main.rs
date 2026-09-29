//! Inverted DTLZ1: minimize three objectives over 7 variables, whose Pareto front is DTLZ1's
//! triangle turned upside down, with usual directions and inverted directions.
//!
//! From genoxide's `multi::problems::InvertedDtlz1`. Runs usual directions and inverted directions,
//! and prints each final front's size, its hypervolume with the objectives divided by the front's
//! nadir point, as a share of that of a sample of the optimal front with as many points as there
//! are reference directions, and the median and largest distance g of its solutions from the front;
//! then the hypervolumes of the whole front and of the sample.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example inverted_dtlz1
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{InvertedDtlz1, MultiProblem};
use genoxide::prelude::*;

// the reference point of the hypervolume, with the objectives divided by the nadir point
const REFERENCE: [f64; 3] = [1.1, 1.1, 1.1];

fn main() -> Result<()> {
    let problem = InvertedDtlz1::<3>::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // 91 reference directions: Das and Dennis's points with 12 divisions
    let directions = multi::das_dennis::<3>(12);
    // the same directions turned upside down, as the front is: (1 − d)/2
    let inverted: Vec<[f64; 3]> = directions
        .iter()
        .map(|d| d.map(|v| (1.0 - v) / 2.0))
        .collect();
    // the sample: as many points of the optimal front, what a front of 91 solutions can be
    let sample = normalized(&problem.optimal_front(directions.len()).expect("known"));
    // NSGA-III with Das and Dennis's 91 directions, Deb and Jain's settings
    let algorithm = Nsga3::builder(problem.representation(), [Minimize; 3], directions.clone())
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 7.0, 20.0)?)
        .seed(1)
        .build()?;
    run(
        problem,
        "usual directions",
        algorithm,
        2000,
        &sample,
        &mut trace,
    )?;
    // NSGA-III with the same directions turned upside down, (1 − d)/2
    let algorithm = Nsga3::builder(problem.representation(), [Minimize; 3], inverted)
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 7.0, 20.0)?)
        .seed(1)
        .build()?;
    run(
        problem,
        "inverted directions",
        algorithm,
        2000,
        &sample,
        &mut trace,
    )?;
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
    problem: InvertedDtlz1<3>,
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
    // g, from f₁ + f₂ + f₃ = 1 + g: the median (the middle value, or the upper of the two middle
    // ones) and the largest
    let mut g: Vec<f64> = normalized(&front)
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) / 2.0 - 1.0)
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
    let problem = InvertedDtlz1::<3>::default();
    let ideal = problem.ideal_point().expect("known");
    let nadir = problem.nadir_point().expect("known");
    let scale = |p: &[f64; 3]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}
