//! DTLZ7 with 3 objectives: minimize three conflicting objectives over 22 variables in [0, 1],
//! whose Pareto front is four disconnected regions.
//!
//! NSGA-III with the settings of the DTLZ2 example, and NSGA-II with the same population and
//! operators, each for 250 generations. Prints, for each, how many solutions of its front lie in
//! each region, its IGD+ to 1,024 points of the front and its hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of the NSGA-III run for the plot on the
//! example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example dtlz7_3obj
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Dtlz7, MultiProblem};
use genoxide::prelude::*;

const VARIABLES: usize = 22;

// f₁ and f₂ on the front are in [0, A] or (B, C], from genoxide's docs
const A: f64 = 0.251_411_836_088_917_1;
const B: f64 = 0.631_626_530_700_061_2;
const C: f64 = 0.859_400_856_644_723_9;

// the reference point of the hypervolume: 1.1 times the nadir point (C, C, 6)
const REFERENCE: [f64; 3] = [1.1 * C, 1.1 * C, 6.6];

fn main() -> Result<()> {
    let problem = Dtlz7::<3>::new(VARIABLES);
    // with GENOXIDE_TRACE=<file>, a trace of the NSGA-III run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let directions = multi::das_dennis::<3>(12);
    let nsga3 = Nsga3::builder(problem.representation(), [Minimize; 3], directions)
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / VARIABLES as f64, 20.0)?)
        .seed(1)
        .build()?;
    run("NSGA-III", nsga3, |snapshot| trace.record(snapshot))?;

    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 3])
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / VARIABLES as f64, 20.0)?)
        .seed(1)
        .build()?;
    run("NSGA-II", nsga2, |_| {})?;

    // the whole front's hypervolume, integrated numerically
    println!("the whole front: hypervolume 1.7392");
    trace.write();
    Ok(())
}

// runs `algorithm` for 250 generations, and reports its front
fn run<T>(name: &str, algorithm: T, record: impl FnMut(&MultiSnapshot<'_, Reals, 3>)) -> Result<()>
where
    T: MultiObjectiveAlgorithm<3, Genome = Reals>,
{
    let problem = Dtlz7::<3>::new(VARIABLES);
    let outcome = MultiEngine::new(algorithm, problem)
        .stop_when(Stop::generations(250))
        .on_generation(record)
        .run()?;
    let front = outcome.front_values();
    // the solutions in each region: f₁ and f₂ low or high, split halfway between A and B
    let high = |f: f64| usize::from(f > (A + B) / 2.0);
    let mut regions = [0; 4];
    for f in &front {
        regions[2 * high(f[0]) + high(f[1])] += 1;
    }
    // IGD+ to a grid of 32 × 32 values of f₁ and f₂ over the regions
    let optimal = problem.optimal_front(1000).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 3]);
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 3]);
    let [low_low, low_high, high_low, high_high] = regions;
    println!(
        "{name:<8} {} solutions, {low_low} + {low_high} + {high_low} + {high_high} in the four \
         regions, IGD+ {distance:.4}, hypervolume {volume:.4}",
        front.len()
    );
    Ok(())
}
