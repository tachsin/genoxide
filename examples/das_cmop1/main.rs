//! DAS-CMOP1: minimize two objectives over 30 variables subject to 11 constraints, whose front is
//! pieces of a concave curve, with MOEA/D-DE and NSGA-II.
//!
//! Fan et al.'s DAS-CMOP1, from genoxide's `multi::problems::DasCmop1`, with the difficulty triplet of
//! the paper's figure 6, (0, 0.5, 0.5). Runs MOEA/D with differential evolution (MOEA/D-DE) and,
//! as a contrast, NSGA-II with the paper's settings, a population of 300 for 1,000 generations
//! each (300,000 evaluations), with constraint dominance. Prints each final front's size, how many
//! of its solutions are feasible, and its IGD+ to 500 points of the optimal front and hypervolume,
//! with the objectives normalized by the front's ideal and nadir points, as a share of that of a
//! sample of the front with as many points as the population; then the hypervolumes of the whole
//! front and of the sample.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example das_cmop1
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{DasCmop1, MultiProblem};
use genoxide::multi::{DifferentialEvolutionCrossover, MultiFitnessFunction};
use genoxide::prelude::*;

// the reference point of the hypervolume, with the objectives normalized by the front's ideal and
// nadir points: 1.1 times the nadir point
const REFERENCE: [f64; 2] = [1.1, 1.1];

// the paper's population and evaluations: 300 for 1,000 generations
const POPULATION: usize = 300;
const GENERATIONS: u64 = 1_000;

fn main() -> Result<()> {
    let problem = DasCmop1::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // the sample: as many points of the optimal front as the population has, about what a front
    // of that many solutions can be
    let sample = normalized(&problem, &problem.optimal_front(POPULATION).expect("known"));
    let rate = 1.0 / problem.variables() as f64;
    // MOEA/D-DE: a subproblem per weight vector, 30 neighbours (the paper's 0.1 N), parents from
    // the neighbourhood with probability 0.2, differential evolution with F = 0.5 and CR = 1
    let moead = Moead::builder(
        problem.representation(),
        [Minimize; 2],
        multi::das_dennis::<2>(POPULATION - 1),
    )
    .neighbors(30)
    .neighbor_mating(0.2)
    .crossover(DifferentialEvolutionCrossover::new(0.5, 1.0)?)
    .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
    .seed(1)
    .build()?;
    run(&problem, "MOEA/D-DE", moead, &sample, &mut trace)?;
    // the contrast: NSGA-II with the paper's settings
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(POPULATION)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
        .seed(1)
        .build()?;
    run(&problem, "NSGA-II", nsga2, &sample, &mut trace)?;
    // the hypervolume of the whole front, from 20,000 of its points
    let whole = normalized(&problem, &problem.optimal_front(20_000).expect("known"));
    println!(
        "the whole front: hypervolume {:.4}; a sample of {} of its points: {:.4}",
        hypervolume(&whole, &REFERENCE, &[Minimize; 2]),
        sample.len(),
        hypervolume(&sample, &REFERENCE, &[Minimize; 2])
    );
    trace.write(&problem);
    Ok(())
}

// runs `algorithm` for the paper's 1,000 generations, and prints its final front's size, how many
// of it are feasible, and its IGD+ to 500 points of the optimal front and hypervolume, with
// normalized objectives, as a share of the sample's
fn run<A>(
    problem: &DasCmop1,
    name: &'static str,
    algorithm: A,
    sample: &[[f64; 2]],
    trace: &mut trace::Trace,
) -> Result<()>
where
    A: MultiObjectiveAlgorithm<2, Genome = Reals>,
{
    let mut record = trace.fronts(name);
    let outcome = MultiEngine::new(algorithm, *problem)
        .stop_when(Stop::generations(GENERATIONS))
        .on_generation(|snapshot| record(snapshot))
        .run()?;
    let scores: Vec<([f64; 2], f64)> = outcome
        .front()
        .iter()
        .map(|x| problem.evaluate(x.genome()))
        .collect();
    let feasible: Vec<[f64; 2]> = scores.iter().filter(|s| s.1 == 0.0).map(|s| s.0).collect();
    let noun = if scores.len() == 1 {
        "solution"
    } else {
        "solutions"
    };
    print!(
        "{name}, {GENERATIONS} generations: {} {noun}, ",
        scores.len()
    );
    if feasible.is_empty() {
        let least = scores.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
        println!("none feasible, the least violation {least:.4}");
        return Ok(());
    }
    let found = normalized(problem, &feasible);
    let optimal = normalized(problem, &problem.optimal_front(500).expect("known"));
    let distance = igd_plus(&found, &optimal, &[Minimize; 2]);
    let volume = hypervolume(&found, &REFERENCE, &[Minimize; 2]);
    let percent = 100.0 * volume / hypervolume(sample, &REFERENCE, &[Minimize; 2]);
    println!(
        "{} feasible, IGD+ {distance:.4}, hypervolume {volume:.4}, {percent:.1}% of the \
         sample's",
        feasible.len()
    );
    Ok(())
}

// the objectives normalized by the front's ideal and nadir points: the front spans [0, 1] in each
pub fn normalized(problem: &DasCmop1, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let ideal = problem.ideal_point().expect("known");
    let nadir = problem.nadir_point().expect("known");
    let scale = |p: &[f64; 2]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}
