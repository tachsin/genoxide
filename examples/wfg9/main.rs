//! WFG9: minimize two objectives whose concave front lies behind multimodal, deceptive and
//! non-separable parameters, with NSGA-II and SMS-EMOA.
//!
//! The Walking Fish Group's ninth problem, from genoxide's `multi::problems::Wfg9`, with 2
//! objectives, 4 position and 20 distance parameters. Runs NSGA-II with simulated binary crossover
//! for 1,000 generations, and SMS-EMOA with blend crossover for 5,000, and prints their fronts
//! after 1,000 and 5,000: the size, the IGD+ to 500 points of the optimal front, the hypervolume,
//! and how far the solutions are from the front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example wfg9
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Wfg9};
use genoxide::prelude::*;

// the reference point of the hypervolume: 1.1 times the front's nadir point (2, 4)
const REFERENCE: [f64; 2] = [2.2, 4.4];

// the generation of NSGA-II's report, and of SMS-EMOA's first
const FIRST: u64 = 1_000;

// the generation of SMS-EMOA's last report
const LAST: u64 = 5_000;

fn main() -> Result<()> {
    // 2 objectives, k = 4 position and l = 20 distance parameters
    let problem = Wfg9::<2>::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // polynomial mutation at a rate of 1/24, one gene per child on average
    let mutation = PolynomialMutation::per_gene(1.0 / 24.0, 20.0)?;

    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(mutation)
        .seed(1)
        .build()?;
    run("NSGA-II", nsga2, &[FIRST], &mut trace)?;

    // blend crossover: each gene of a child drawn from the parents' interval, widened by 0.3 of
    // its length on each side
    let sms_emoa = SmsEmoa::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(BlendCrossover::new(0.3)?)
        .mutate(mutation)
        .seed(1)
        .build()?;
    run("SMS-EMOA", sms_emoa, &[FIRST, LAST], &mut trace)?;

    // the box up to the reference point, less the quarter ellipse under the front, of area 2π
    let whole = REFERENCE[0] * REFERENCE[1] - 2.0 * std::f64::consts::PI;
    println!("the whole front: hypervolume {whole:.4}");
    trace.write();
    Ok(())
}

// runs `algorithm` up to the last of `reports`, and reports its front after each of them
fn run<A>(name: &'static str, algorithm: A, reports: &[u64], trace: &mut trace::Trace) -> Result<()>
where
    A: MultiObjectiveAlgorithm<2, Genome = Reals>,
{
    let mut record = trace.fronts(name);
    let mut fronts = Vec::new();
    let last = *reports.last().expect("a report");
    MultiEngine::new(algorithm, Wfg9::<2>::default())
        .stop_when(Stop::generations(last))
        .on_generation(|snapshot| {
            if reports.contains(&snapshot.progress().generation()) {
                fronts.push(front_of(snapshot));
            }
            record(snapshot);
        })
        .run()?;
    for (generations, front) in reports.iter().zip(&fronts) {
        report(name, *generations, front);
    }
    Ok(())
}

// the objective values of the front after a generation
fn front_of(snapshot: &MultiSnapshot<'_, Reals, 2>) -> Vec<[f64; 2]> {
    let front = snapshot.front().iter();
    front.filter_map(|x| x.fitness()?.values()).collect()
}

// how far a point is from the front: the d with (f₁ − d, f₂ − d) on it, where
// ((f₁ − d) / 2)² + ((f₂ − d) / 4)² = 1, the smaller root of that quadratic. WFG adds the
// distance parameters' value, x_M, to both objectives: d is that value.
fn distance(f: &[f64; 2]) -> f64 {
    let a = 0.25 + 0.0625;
    let b = f[0] / 2.0 + f[1] / 8.0;
    let c = f[0] * f[0] / 4.0 + f[1] * f[1] / 16.0 - 1.0;
    ((b - (b * b - 4.0 * a * c).sqrt()) / (2.0 * a)).max(0.0)
}

// the size of a front, its IGD+ to 500 points of the optimal front, also with the objectives
// scaled to [0, 1] over the front's ranges, 2 and 4, its hypervolume, and the distances of its
// points from the front
fn report(name: &str, generations: u64, front: &[[f64; 2]]) {
    let optimal = Wfg9::<2>::default().optimal_front(500).expect("known");
    let igd = igd_plus(front, &optimal, &[Minimize; 2]);
    let scale = |points: &[[f64; 2]]| -> Vec<[f64; 2]> {
        points.iter().map(|f| [f[0] / 2.0, f[1] / 4.0]).collect()
    };
    let scaled = igd_plus(&scale(front), &scale(&optimal), &[Minimize; 2]);
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    let distances: Vec<f64> = front.iter().map(distance).collect();
    let least = distances.iter().copied().fold(f64::INFINITY, f64::min);
    let most = distances.iter().copied().fold(0.0, f64::max);
    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    println!(
        "{name:<8} after {generations} generations: {} solutions, IGD+ {igd:.4} (scaled \
         {scaled:.4}), hypervolume {volume:.4}",
        front.len()
    );
    println!("         distance from the front {least:.4} to {most:.4}, mean {mean:.4}");
}
