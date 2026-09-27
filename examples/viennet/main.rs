//! Viennet's three problems: minimize three objectives of two variables, with NSGA-III.
//!
//! VNT1, VNT2 and VNT3 from genoxide's `multi::problems`, each with the 91 reference directions of
//! Das and Dennis's method with 12 divisions, a population of 92 and 30 generations. Prints, per
//! problem, the size of the final front and its normalized hypervolume: the hypervolume with the
//! objectives mapped to [0, 1] by the ideal and nadir points, and the reference point
//! (1.1, 1.1, 1.1). For VNT1, whose front is known, also the IGD+ to 1,035 points of it.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example viennet
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiFitnessFunction;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Viennet1, Viennet2, Viennet3};
use genoxide::prelude::*;

const GENERATIONS: u64 = 30;

fn main() -> Result<()> {
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    println!("problem  solutions  hypervolume  IGD+");
    // VNT1's ideal and nadir points are known: its objectives' minima, and their worst values
    // on the front, at the minima of the other two
    let ideal = Viennet1.ideal_point().expect("known");
    let nadir = Viennet1.nadir_point().expect("known");
    solve(Viennet1, Scale { ideal, nadir }, &mut trace)?;
    // VNT2's and VNT3's aren't in genoxide. VNT2's objectives are convex: the worst values on its
    // front are at the minima of the other two, as for VNT1. VNT3's nadir point is an estimate,
    // from the non-dominated points of a 2,001 × 2,001 grid of the variables.
    let ideal = [3.0, -17.0, -13.0];
    let nadir = [4.2452, -16.4766, -12.0531];
    solve(Viennet2, Scale { ideal, nadir }, &mut trace)?;
    let ideal = [0.0, 15.0, -0.1];
    let nadir = [8.1964, 17.0370, 0.1762];
    solve(Viennet3, Scale { ideal, nadir }, &mut trace)?;
    trace.write();
    Ok(())
}

// runs NSGA-III on the problem, and prints the size of its front, the front's normalized
// hypervolume and, if the optimal front is known, the IGD+ to it
fn solve<P>(problem: P, scale: Scale, trace: &mut trace::Trace) -> Result<()>
where
    P: MultiProblem<3, Representation = Real>
        + MultiFitnessFunction<Reals, 3, Output = [f64; 3]>
        + Copy,
{
    let nsga3 = Nsga3::builder(
        problem.representation(),
        [Minimize; 3],
        multi::das_dennis::<3>(12),
    )
    .population_size(92)
    .crossover(SimulatedBinaryCrossover::new(30.0)?)
    .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
    .seed(1)
    .build()?;
    let outcome = MultiEngine::new(nsga3, problem)
        .stop_when(Stop::generations(GENERATIONS))
        .on_generation(trace.fronts(problem.name(), scale))
        .run()?;

    let front = outcome.front_values();
    let volume = scale.hypervolume(&front);
    // the IGD+ to at least 1,000 points of the optimal front: for VNT1, the images of 1,035
    // points spread evenly over the triangle of its optimal solutions
    let distance = match problem.optimal_front(1000) {
        Some(optimal) => format!("{:.4}", igd_plus(&front, &optimal, &[Minimize; 3])),
        None => "-".to_string(),
    };
    println!(
        "{:<7}  {:>9}  {volume:>11.4}  {distance:>6}",
        problem.name(),
        front.len()
    );
    Ok(())
}

/// The ideal and nadir points of a problem, which map its objectives to [0, 1].
#[derive(Clone, Copy)]
pub struct Scale {
    ideal: [f64; 3],
    nadir: [f64; 3],
}

impl Scale {
    /// The hypervolume of `front` with the objectives mapped to [0, 1], and the reference point
    /// (1.1, 1.1, 1.1).
    pub fn hypervolume(&self, front: &[[f64; 3]]) -> f64 {
        let normalized = front.iter().map(|point| {
            std::array::from_fn(|j| (point[j] - self.ideal[j]) / (self.nadir[j] - self.ideal[j]))
        });
        let normalized: Vec<[f64; 3]> = normalized.collect();
        hypervolume(&normalized, &[1.1; 3], &[Minimize; 3])
    }
}
