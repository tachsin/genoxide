//! WFG1: minimize two objectives over 24 variables, with a front of convex and mixed parts behind
//! a flat region and a strong polynomial bias, with NSGA-II.
//!
//! The first problem of Huband, Hingston, Barone and While's WFG toolkit, from genoxide's
//! `multi::problems::Wfg1`, with 2 objectives and the recommended sizes, k = 4 and l = 20. In
//! double precision no genome reaches the front, so the target is the closest front a genome can
//! have: the front moved by the smallest distance x_M that a genome gets. Runs NSGA-II with
//! simulated binary crossover at η = 15, the ZDT examples' setting, for 2,500 generations, and at
//! η = 2 for 20,000, and prints the fronts after 2,500 and after 20,000 generations: the size, the
//! IGD+ to 500 points of the optimal front and to the closest front, and the hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example wfg1
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Wfg1};
use genoxide::multi::{MultiFitnessFunction, MultiSnapshot};
use genoxide::prelude::*;

// the reference point of the hypervolume: 1.1 times the front's worst point, (2, 4)
const REFERENCE: [f64; 2] = [2.2, 4.4];

// the generation of the first report, ten times the NSGA-II paper's budget, and of the last
const FIRST: u64 = 2_500;
const LAST: u64 = 20_000;

fn main() -> Result<()> {
    let problem = Wfg1::<2>::default();

    // the closest a genome gets: the paper's optimal solutions, the distance parameters zᵢ at
    // 0.35 × 2i, still have a distance x_M of about 0.069 in double precision; with the position
    // parameters at their upper bounds, x₁ = 1, f₂ = x_M
    let z: Reals = (1..=24)
        .map(|i| {
            if i <= 4 {
                2.0 * i as f64
            } else {
                0.35 * 2.0 * i as f64
            }
        })
        .collect();
    let distance = problem.evaluate(&z)[1];
    // every objective value of the front moved by x_M: the closest front a genome can have
    let moved = |points: usize| {
        let front = problem.optimal_front(points).expect("known").into_iter();
        front
            .map(|[f1, f2]| [f1 + distance, f2 + distance])
            .collect::<Vec<_>>()
    };
    let closest = moved(500);

    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // the ZDT examples' settings for 2,500 generations, then SBX with η = 2 for 20,000: children
    // further from their parents
    for (name, eta, generations) in [("SBX eta 15", 15.0, FIRST), ("SBX eta 2", 2.0, LAST)] {
        // polynomial mutation at a rate of 1/24, one gene per child on average
        let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
            .population_size(100)
            .crossover(SimulatedBinaryCrossover::new(eta)?)
            .mutate(PolynomialMutation::per_gene(1.0 / 24.0, 20.0)?)
            .seed(1)
            .build()?;
        run(name, nsga2, generations, &closest, &mut trace)?;
    }

    let optimal = problem.optimal_front(500).expect("known");
    println!(
        "the closest front, {distance:.4} away: IGD+ {:.4}, hypervolume {:.4} with 100 points",
        igd_plus(&closest, &optimal, &[Minimize; 2]),
        hypervolume(&moved(100), &REFERENCE, &[Minimize; 2])
    );
    println!("the whole front: hypervolume 6.7857");
    trace.write();
    Ok(())
}

// runs `algorithm` for `generations`, and reports its front after 2,500 and at the end
fn run<A>(
    name: &'static str,
    algorithm: A,
    generations: u64,
    closest: &[[f64; 2]],
    trace: &mut trace::Trace,
) -> Result<()>
where
    A: MultiObjectiveAlgorithm<2, Genome = Reals>,
{
    let mut record = trace.fronts(name);
    let mut first = Vec::new();
    let outcome = MultiEngine::new(algorithm, Wfg1::<2>::default())
        .stop_when(Stop::generations(generations))
        .on_generation(|snapshot| {
            if snapshot.progress().generation() == FIRST {
                first = front_of(snapshot);
            }
            record(snapshot);
        })
        .run()?;
    if generations > FIRST {
        report(name, FIRST, &first, closest);
    }
    report(name, generations, &outcome.front_values(), closest);
    Ok(())
}

// the objective values of the front after a generation
fn front_of(snapshot: &MultiSnapshot<'_, Reals, 2>) -> Vec<[f64; 2]> {
    let front = snapshot.front().iter();
    front.filter_map(|x| x.fitness()?.values()).collect()
}

// the size of a front; its IGD+ to 500 points of the optimal front, evenly spread along it; its
// IGD+ to the closest front, with f₁ and f₂ divided by the front's range, 2 and 4; and its
// hypervolume
fn report(name: &str, generations: u64, front: &[[f64; 2]], closest: &[[f64; 2]]) {
    let optimal = Wfg1::<2>::default().optimal_front(500).expect("known");
    let distance = igd_plus(front, &optimal, &[Minimize; 2]);
    let scaled = |points: &[[f64; 2]]| {
        let points = points.iter().map(|[f1, f2]| [f1 / 2.0, f2 / 4.0]);
        points.collect::<Vec<_>>()
    };
    let to_closest = igd_plus(&scaled(front), &scaled(closest), &[Minimize; 2]);
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    println!(
        "{name:<10} after {generations:>5} generations: {} solutions, IGD+ {distance:.4} \
         ({to_closest:.4} to the closest front), hypervolume {volume:.4}",
        front.len()
    );
}
