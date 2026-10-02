//! DAS-CMOP9: minimize three objectives over 30 variables subject to 7 constraints, whose front is
//! patches of a sphere, with linked variables, with MOEA/D-DE and NSGA-II.
//!
//! From genoxide's `multi::problems::DasCmop9`. Runs MOEA/D with differential evolution
//! (MOEA/D-DE) and, as a contrast, NSGA-II with the paper's settings, a population of 300 for
//! 1,000 generations each, with the difficulty triplet of the paper's figure 6, (0.5, 0.5, 0.5).
//! Prints each final front's size, how many of its solutions are feasible, their IGD+ to 2,000
//! points of the optimal front and their hypervolume, with the objectives normalized by the
//! front's ideal and nadir points, as a share of that of a sample of the front with at least as
//! many points as the population, and the median and largest distance from them to a dense sample
//! of the front. Then the same indicators for the best point of the dense sample for each of
//! MOEA/D's 300 weight vectors, and for 666, and the hypervolumes of the whole front and of the
//! sample.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example das_cmop9
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{DasCmop9, MultiProblem};
use genoxide::multi::{DifferentialEvolutionCrossover, MultiFitnessFunction};
use genoxide::prelude::*;

// the reference point of the hypervolume, with the objectives normalized by the front's ideal and
// nadir points
pub const REFERENCE: [f64; 3] = [1.1, 1.1, 1.1];

// the paper's population and evaluations: 300 for 1,000 generations; MOEA/D's 300 weight vectors
// are Das and Dennis's with 23 divisions
const POPULATION: usize = 300;
const GENERATIONS: u64 = 1_000;
const DIVISIONS: usize = 23;

fn main() -> Result<()> {
    let problem = DasCmop9::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // the sample: at least as many points of the optimal front as the population has, about what
    // a front of that many solutions can be
    let sample = normalized(&problem, &problem.optimal_front(POPULATION).expect("known"));
    // a dense sample of the front, at least 40,000 points
    let dense = problem.optimal_front(40_000).expect("known");
    let rate = 1.0 / problem.variables() as f64;
    // MOEA/D-DE: a subproblem per weight vector, 30 neighbours (the paper's 0.1 N), parents from
    // the neighbourhood with probability 0.2, differential evolution with F = 0.5 and CR = 1
    let moead = Moead::builder(
        problem.representation(),
        [Minimize; 3],
        multi::das_dennis::<3>(DIVISIONS),
    )
    .neighbors(30)
    .neighbor_mating(0.2)
    .crossover(DifferentialEvolutionCrossover::new(0.5, 1.0)?)
    .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
    .seed(1)
    .build()?;
    run(&problem, "MOEA/D-DE", moead, &sample, &dense, &mut trace)?;
    // the contrast: NSGA-II with the paper's settings
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 3])
        .population_size(POPULATION)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(rate, 20.0)?)
        .seed(1)
        .build()?;
    run(&problem, "NSGA-II", nsga2, &sample, &dense, &mut trace)?;
    // what a decomposition into Tchebycheff subproblems can reach: each weight vector's best point
    // of the dense sample, with MOEA/D's 300 vectors and with 666 (35 divisions)
    for divisions in [DIVISIONS, 35] {
        let weights = multi::das_dennis::<3>(divisions);
        let best = per_weight(&problem, &dense, &weights);
        let (distance, volume, percent) = indicators(&problem, &best, &sample);
        println!(
            "the best point of the front for each of {} weight vectors: {} points, IGD+ \
             {distance:.4}, hypervolume {volume:.4}, {percent:.1}% of the sample's",
            weights.len(),
            best.len()
        );
    }
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

// runs `algorithm` for the paper's 1,000 generations, and prints its final front's size, how many
// of its solutions the problem finds feasible, their IGD+ to 2,000 points of the optimal front and
// their hypervolume, with normalized objectives, as a share of the sample's, and the median and
// largest distance from them to the dense sample of the front
fn run<A>(
    problem: &DasCmop9,
    name: &'static str,
    algorithm: A,
    sample: &[[f64; 3]],
    dense: &[[f64; 3]],
    trace: &mut trace::Trace,
) -> Result<()>
where
    A: MultiObjectiveAlgorithm<3, Genome = Reals>,
{
    let mut record = trace.front(name, *problem);
    let outcome = MultiEngine::new(algorithm, *problem)
        .stop_when(Stop::generations(GENERATIONS))
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
        "{name}, {GENERATIONS} generations: {} {noun}, ",
        scores.len()
    );
    if feasible.is_empty() {
        let least = scores.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
        println!("none feasible, the least violation {least:.4}");
        return Ok(());
    }
    let (distance, volume, percent) = indicators(problem, &feasible, sample);
    let (median, farthest) = distances(problem, &feasible, dense);
    println!(
        "{} feasible, IGD+ {distance:.4}, hypervolume {volume:.4}, {percent:.1}% of the \
         sample's; from the front: {median:.4} (median), {farthest:.4} (farthest)",
        feasible.len()
    );
    Ok(())
}

// the IGD+ of `points` to 2,000 points of the optimal front, their hypervolume, and that as a
// percentage of the sample's, with normalized objectives
fn indicators(problem: &DasCmop9, points: &[[f64; 3]], sample: &[[f64; 3]]) -> (f64, f64, f64) {
    let found = normalized(problem, points);
    let optimal = normalized(problem, &problem.optimal_front(2_000).expect("known"));
    let distance = igd_plus(&found, &optimal, &[Minimize; 3]);
    let volume = hypervolume(&found, &REFERENCE, &[Minimize; 3]);
    let percent = 100.0 * volume / hypervolume(sample, &REFERENCE, &[Minimize; 3]);
    (distance, volume, percent)
}

// the median and the largest distance from a point of `points` to the nearest point of `dense`,
// with normalized objectives
fn distances(problem: &DasCmop9, points: &[[f64; 3]], dense: &[[f64; 3]]) -> (f64, f64) {
    let dense = normalized(problem, dense);
    let squared =
        |p: &[f64; 3], q: &[f64; 3]| -> f64 { (0..3).map(|j| (p[j] - q[j]) * (p[j] - q[j])).sum() };
    let mut distances: Vec<f64> = normalized(problem, points)
        .iter()
        .map(|p| {
            dense
                .iter()
                .map(|q| squared(p, q))
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .collect();
    distances.sort_by(f64::total_cmp);
    let n = distances.len();
    let median = if n % 2 == 1 {
        distances[n / 2]
    } else {
        (distances[n / 2 - 1] + distances[n / 2]) / 2.0
    };
    (median, distances[n - 1])
}

// for each weight vector, the point of `dense` that minimizes MOEA/D's Tchebycheff value
// maxⱼ wⱼ |fⱼ − zⱼ| around the ideal point z (of two equal, the one with the least Σ (fⱼ − zⱼ),
// then the first), each point once
fn per_weight(problem: &DasCmop9, dense: &[[f64; 3]], weights: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let ideal = problem.ideal_point().expect("known");
    let mut best: Vec<[f64; 3]> = weights
        .iter()
        .map(|w| {
            let mut chosen = (f64::INFINITY, f64::INFINITY, 0);
            for (i, f) in dense.iter().enumerate() {
                let value = (0..3)
                    .map(|j| w[j] * (f[j] - ideal[j]).abs())
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum = (f[0] - ideal[0]) + (f[1] - ideal[1]) + (f[2] - ideal[2]);
                if value < chosen.0 || (value == chosen.0 && sum < chosen.1) {
                    chosen = (value, sum, i);
                }
            }
            dense[chosen.2]
        })
        .collect();
    best.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    best.dedup();
    best
}

// the objectives normalized by the front's ideal and nadir points: the front spans [0, 1] in each
pub fn normalized(problem: &DasCmop9, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let ideal = problem.ideal_point().expect("known");
    let nadir = problem.nadir_point().expect("known");
    let scale = |p: &[f64; 3]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}
