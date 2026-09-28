//! DTLZ5 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1],
//! whose Pareto front is a curve, a quarter circle in the plane f₁ = f₂.
//!
//! NSGA-III with the settings of the DTLZ2 example, and NSGA-II with the same population and
//! operators, each for 250 generations. Prints, for each, the size of its front, its IGD+ to
//! 1,000 points of the curve, its hypervolume, the largest gap between its solutions along the
//! curve and how far its farthest solution is from the front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of the NSGA-II run for the plot on the
//! example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example dtlz5_3obj
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::math;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Dtlz5, MultiProblem};
use genoxide::prelude::*;

const VARIABLES: usize = 12;

// the reference point of the hypervolume: 1.1 times the nadir point (1/√2, 1/√2, 1)
const REFERENCE: [f64; 3] = [
    1.1 * std::f64::consts::FRAC_1_SQRT_2,
    1.1 * std::f64::consts::FRAC_1_SQRT_2,
    1.1,
];

fn main() -> Result<()> {
    let problem = Dtlz5::<3>::new(VARIABLES);
    let directions = multi::das_dennis::<3>(12);
    let nsga3 = Nsga3::builder(problem.representation(), [Minimize; 3], directions)
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / VARIABLES as f64, 20.0)?)
        .seed(1)
        .build()?;
    run("NSGA-III", nsga3, |_| {})?;

    // with GENOXIDE_TRACE=<file>, a trace of the NSGA-II run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 3])
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / VARIABLES as f64, 20.0)?)
        .seed(1)
        .build()?;
    run("NSGA-II", nsga2, |snapshot| trace.record(snapshot))?;

    // the whole curve's hypervolume, 1.1³/2 − 1.1π/4 + 1/3
    println!("the whole front: hypervolume 0.1349");
    trace.write();
    Ok(())
}

// runs `algorithm` for 250 generations, and reports its front
fn run<A>(name: &str, algorithm: A, record: impl FnMut(&MultiSnapshot<'_, Reals, 3>)) -> Result<()>
where
    A: MultiObjectiveAlgorithm<3, Genome = Reals>,
{
    let problem = Dtlz5::<3>::new(VARIABLES);
    let outcome = MultiEngine::new(algorithm, problem)
        .stop_when(Stop::generations(250))
        .on_generation(record)
        .run()?;
    let front = outcome.front_values();
    // IGD+ to 1,000 points evenly spread along the curve
    let optimal = problem.optimal_front(1000).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 3]);
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 3]);
    // θ₁ in degrees, 0 at (1/√2, 1/√2, 0) and 90 at (0, 0, 1); the largest gap between
    // neighbors, or between an end of the curve and the nearest solution
    let angle = |f: &[f64; 3]| math::atan2(f[2], (f[0] * f[0] + f[1] * f[1]).sqrt()).to_degrees();
    let mut angles: Vec<f64> = front.iter().map(angle).chain([0.0, 90.0]).collect();
    angles.sort_by(f64::total_cmp);
    let gap = angles.windows(2).map(|w| w[1] - w[0]).fold(0.0, f64::max);
    // the objectives lie on a sphere of radius 1 + g: g is 0 on the front
    let farthest = front
        .iter()
        .map(|f| f.iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0)
        .fold(0.0, f64::max);
    println!(
        "{name:<8} {} solutions, IGD+ {distance:.4}, hypervolume {volume:.4}, largest gap \
         {gap:.1}°, largest g {farthest:.4}",
        front.len()
    );
    Ok(())
}
