//! DTLZ9 with 3 objectives: minimize three objectives over 30 variables subject to 2 constraints,
//! whose front is a curve, with NSGA-II and SMS-EMOA.
//!
//! From genoxide's `multi::problems::Dtlz9`. Runs NSGA-II for the report's 500 generations and
//! SMS-EMOA for 20,000, with a population of 100. Prints each final front's size, how many of its
//! solutions are feasible, their IGD+ to 2,000 points of the optimal front and their hypervolume,
//! with the objectives normalized by the front's ideal and nadir points, as a share of that of a
//! sample of the front with at least as many points as the population; then the hypervolumes of the
//! whole front and of the sample.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example dtlz9_3obj
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiFitnessFunction;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Dtlz9, MultiProblem};
use genoxide::prelude::*;

// the reference point of the hypervolume, with the objectives normalized by the front's ideal and
// nadir points
pub const REFERENCE: [f64; 3] = [1.1, 1.1, 1.1];

fn main() -> Result<()> {
    let problem = Dtlz9::<3>::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // the sample: at least as many points of the optimal front as the population has, about what
    // a front of that many solutions can be
    let sample = normalized(&problem, &problem.optimal_front(100).expect("known"));
    let rate = 1.0 / problem.variables() as f64;
    // NSGA-II for the report's 500 generations
    let algorithm = Nsga2::builder(problem.representation(), [Minimize; 3])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
        .seed(1)
        .build()?;
    run(
        &problem, "NSGA-II", algorithm, problem, 500, &sample, &mut trace,
    )?;
    // SMS-EMOA, which keeps the solutions that add the most hypervolume, for 20,000 generations
    let algorithm = SmsEmoa::builder(problem.representation(), [Minimize; 3])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
        .seed(1)
        .build()?;
    run(
        &problem, "SMS-EMOA", algorithm, problem, 20_000, &sample, &mut trace,
    )?;
    let whole = normalized(&problem, &problem.optimal_front(3_000).expect("known"));
    println!(
        "the whole front: hypervolume {:.4}; a sample of {} of its points: {:.4}",
        hypervolume(&whole, &REFERENCE, &[Minimize; 3]),
        sample.len(),
        hypervolume(&sample, &REFERENCE, &[Minimize; 3])
    );
    trace.write(&problem);
    Ok(())
}

// runs `algorithm` on `fitness` for `generations`, and prints its final front's size, how many of
// its solutions the problem finds feasible, their IGD+ to 2,000 points of the optimal front and
// their hypervolume, with normalized objectives, as a share of the sample's
fn run<A, F>(
    problem: &Dtlz9<3>,
    name: &'static str,
    algorithm: A,
    fitness: F,
    generations: u64,
    sample: &[[f64; 3]],
    trace: &mut trace::Trace,
) -> Result<()>
where
    A: MultiObjectiveAlgorithm<3, Genome = Reals>,
    F: MultiFitnessFunction<Reals, 3> + Sync,
{
    let mut record = trace.front(name, *problem);
    let outcome = MultiEngine::new(algorithm, fitness)
        .stop_when(Stop::generations(generations))
        .on_generation(|snapshot| record(snapshot))
        .run()?;
    let scores: Vec<([f64; 3], f64)> = outcome
        .front()
        .iter()
        .map(|x| problem.evaluate(x.genome()))
        .collect();
    let feasible: Vec<[f64; 3]> = scores.iter().filter(|s| s.1 == 0.0).map(|s| s.0).collect();
    let noun = if scores.len() == 1 {
        "solution"
    } else {
        "solutions"
    };
    print!(
        "{name}, {generations} generations: {} {noun}, ",
        scores.len()
    );
    if feasible.is_empty() {
        let least = scores.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
        println!("none feasible, the least violation {least:.4}");
        return Ok(());
    }
    let found = normalized(problem, &feasible);
    let optimal = normalized(problem, &problem.optimal_front(2_000).expect("known"));
    let distance = igd_plus(&found, &optimal, &[Minimize; 3]);
    let volume = hypervolume(&found, &REFERENCE, &[Minimize; 3]);
    let percent = 100.0 * volume / hypervolume(sample, &REFERENCE, &[Minimize; 3]);
    println!(
        "{} feasible, IGD+ {distance:.4}, hypervolume {volume:.4}, {percent:.1}% of the \
         sample's",
        feasible.len()
    );
    Ok(())
}

// the objectives normalized by the front's ideal and nadir points: the front spans [0, 1] in each
pub fn normalized(problem: &Dtlz9<3>, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let ideal = problem.ideal_point().expect("known");
    let nadir = problem.nadir_point().expect("known");
    let scale = |p: &[f64; 3]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}
