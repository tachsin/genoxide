//! MW14: minimize three objectives over 15 variables, subject to one constraint, whose Pareto front
//! is four disconnected patches, with NSGA-III and SMS-EMOA.
//!
//! From genoxide's `multi::problems::Mw14`. Runs NSGA-III and SMS-EMOA, and prints each final
//! front's size, its hypervolume with the objectives divided by the front's nadir point, as a share
//! of that of a sample of the optimal front with as many points as there are reference directions,
//! and how many of its solutions lie in the gaps between the patches; then the hypervolumes of the
//! whole front and of the sample.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example mw14
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{MultiProblem, Mw14};
use genoxide::prelude::*;

// the reference point of the hypervolume, with the objectives divided by the nadir point
const REFERENCE: [f64; 3] = [1.1, 1.1, 1.1];

// MW14's gaps: each of f₁ and f₂ on the front is in [0, A] or (B, 1.5]
const A: f64 = 0.731_352_297_489_732_5;
const B: f64 = 1.329_633_908_740_225_9;

fn main() -> Result<()> {
    let problem = Mw14::<3>::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // 91 reference directions: Das and Dennis's points with 12 divisions
    let directions = multi::das_dennis::<3>(12);
    // the sample: as many points of the optimal front, what a front of 91 solutions can be
    let sample = normalized(&problem.optimal_front(directions.len()).expect("known"));
    // NSGA-III with the paper's operators: η = 20 for both
    let algorithm = Nsga3::builder(problem.representation(), [Minimize; 3], directions.clone())
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 15.0, 20.0)?)
        .seed(1)
        .build()?;
    run(problem, "NSGA-III", algorithm, 3000, &sample, &mut trace)?;
    // SMS-EMOA, which keeps the solutions that add the most hypervolume
    let algorithm = SmsEmoa::builder(problem.representation(), [Minimize; 3])
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 15.0, 20.0)?)
        .seed(1)
        .build()?;
    run(problem, "SMS-EMOA", algorithm, 2000, &sample, &mut trace)?;
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
// share of the sample's, and how many of its solutions lie in the gaps between the patches
fn run<A>(
    problem: Mw14<3>,
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
    let gaps = front
        .iter()
        .filter(|p| (p[0] > A && p[0] <= B) || (p[1] > A && p[1] <= B))
        .count();
    println!(
        "{name}, {generations} generations: {} solutions, hypervolume {volume:.4}, {percent:.1}% \
         of the sample's, {gaps} in the gaps",
        front.len()
    );
    Ok(())
}

// the objectives divided by the front's nadir point (its ideal point is the origin, but for f₃)
fn normalized(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let problem = Mw14::<3>::default();
    let ideal = problem.ideal_point().expect("known");
    let nadir = problem.nadir_point().expect("known");
    let scale = |p: &[f64; 3]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}
