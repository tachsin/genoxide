//! WFG3: minimize two objectives over 24 variables, with a linear front and non-separable
//! distance parameters, with NSGA-II and SMS-EMOA.
//!
//! The third problem of Huband, Hingston, Barone and While's WFG toolkit, from genoxide's
//! `multi::problems::Wfg3`, with 2 objectives and the recommended sizes, k = 4 and l = 20. Runs
//! each algorithm for 1,000 generations, and prints its front after 250 and after 1,000: the size,
//! the IGD+ to 500 points of the optimal front, the hypervolume, and the largest gap between
//! neighbors on the front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example wfg3
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Wfg3};
use genoxide::prelude::*;

// the reference point of the hypervolume: 1.1 times the front's worst point, (2, 4)
const REFERENCE: [f64; 2] = [2.2, 4.4];

// the generation of the first report: the NSGA-II paper's budget
const FIRST: u64 = 250;

// the generation of the last report
const LAST: u64 = 1_000;

fn main() -> Result<()> {
    let problem = Wfg3::<2>::default();
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

    // the segment from (0, 4) to (2, 0): 2.2 × 4.4 less the triangle under it, 4
    println!("the whole front: hypervolume 5.6800; 100 evenly spaced points: gaps of 0.0452");
    trace.write();
    Ok(())
}

// runs `algorithm` for 1,000 generations, and reports its front after 250 and after 1,000
fn run<A>(name: &'static str, algorithm: A, trace: &mut trace::Trace) -> Result<()>
where
    A: MultiObjectiveAlgorithm<2, Genome = Reals>,
{
    let mut record = trace.fronts(name);
    let mut first = Vec::new();
    let outcome = MultiEngine::new(algorithm, Wfg3::<2>::default())
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

// the size of a front, its IGD+ to 500 points of the optimal front, evenly spaced, its
// hypervolume, and the largest distance between neighbors on it, sorted by f₁
fn report(name: &str, generations: u64, front: &[[f64; 2]]) {
    let optimal = Wfg3::<2>::default().optimal_front(500).expect("known");
    let distance = igd_plus(front, &optimal, &[Minimize; 2]);
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    let mut sorted = front.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let gap = sorted
        .windows(2)
        .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
        .fold(0.0, f64::max);
    println!(
        "{name:<8} after {generations} generations: {} solutions, IGD+ {distance:.4}, \
         hypervolume {volume:.4}, largest gap {gap:.4}",
        front.len()
    );
}
