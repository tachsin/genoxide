//! WFG1: minimize two objectives over 24 variables, with a front of convex and mixed parts behind
//! a flat region and a strong polynomial bias, with NSGA-II and SMS-EMOA.
//!
//! The first problem of Huband, Hingston, Barone and While's WFG toolkit, from genoxide's
//! `multi::problems::Wfg1`, with 2 objectives and the recommended sizes, k = 4 and l = 20. Runs
//! each algorithm for 2,500 generations, and prints its front after 250 and after 2,500: the
//! size, the IGD+ to 500 points of the optimal front, and the hypervolume. In double precision no
//! genome reaches the front, so it also prints the closest that a genome gets.
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

// the generation of the first report: the NSGA-II paper's budget
const FIRST: u64 = 250;

// the generation of the last report
const LAST: u64 = 2_500;

fn main() -> Result<()> {
    let problem = Wfg1::<2>::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();

    // polynomial mutation at a rate of 1/24, one gene per child on average
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 24.0, 20.0)?)
        .seed(1)
        .build()?;
    run("NSGA-II", nsga2, &mut trace)?;

    let sms_emoa = SmsEmoa::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 24.0, 20.0)?)
        .seed(1)
        .build()?;
    run("SMS-EMOA", sms_emoa, &mut trace)?;

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
    let optimal = problem.optimal_front(500).expect("known");
    println!(
        "the closest front, {distance:.4} away: IGD+ {:.4}, hypervolume {:.4} with 100 points",
        igd_plus(&moved(500), &optimal, &[Minimize; 2]),
        hypervolume(&moved(100), &REFERENCE, &[Minimize; 2])
    );
    println!("the whole front: hypervolume 6.7857");
    trace.write();
    Ok(())
}

// runs `algorithm` for 2,500 generations, and reports its front after 250 and after 2,500
fn run<A>(name: &'static str, algorithm: A, trace: &mut trace::Trace) -> Result<()>
where
    A: MultiObjectiveAlgorithm<2, Genome = Reals>,
{
    let mut record = trace.fronts(name);
    let mut first = Vec::new();
    let outcome = MultiEngine::new(algorithm, Wfg1::<2>::default())
        .stop_when(Stop::generations(LAST))
        .on_generation(|snapshot| {
            if snapshot.progress().generation() == FIRST {
                first = front_of(snapshot);
            }
            record(snapshot);
        })
        .run()?;
    report(name, FIRST, &first);
    report(name, LAST, &outcome.front_values());
    Ok(())
}

// the objective values of the front after a generation
fn front_of(snapshot: &MultiSnapshot<'_, Reals, 2>) -> Vec<[f64; 2]> {
    let front = snapshot.front().iter();
    front.filter_map(|x| x.fitness()?.values()).collect()
}

// the size of a front, its IGD+ to 500 points of the optimal front, evenly spread along it, and
// its hypervolume
fn report(name: &str, generations: u64, front: &[[f64; 2]]) {
    let optimal = Wfg1::<2>::default().optimal_front(500).expect("known");
    let distance = igd_plus(front, &optimal, &[Minimize; 2]);
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    println!(
        "{name:<8} after {generations} generations: {} solutions, IGD+ {distance:.4}, \
         hypervolume {volume:.4}",
        front.len()
    );
}
