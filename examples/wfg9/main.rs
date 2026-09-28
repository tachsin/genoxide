//! WFG9: minimize two objectives whose concave front lies behind multimodal, deceptive and
//! non-separable parameters, with NSGA-II and MOEA/D.
//!
//! The Walking Fish Group's ninth problem, from genoxide's `multi::problems::Wfg9`, with 2
//! objectives, 4 position and 20 distance parameters. Runs each algorithm for 1,000 generations,
//! and prints its front after 250 and after 1,000: the size, the IGD+ to 500 points of the
//! optimal front, the hypervolume, and how far the solutions are from the front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example wfg9
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Wfg9};
use genoxide::multi::{Decomposition, MultiSnapshot, das_dennis};
use genoxide::prelude::*;

// the reference point of the hypervolume: 1.1 times the front's nadir point (2, 4)
const REFERENCE: [f64; 2] = [2.2, 4.4];

// the generation of the first report: 25,000 evaluations
const FIRST: u64 = 250;

// the generations of each run
const GENERATIONS: u64 = 1_000;

fn main() -> Result<()> {
    // 2 objectives, k = 4 position and l = 20 distance parameters
    let problem = Wfg9::<2>::default();
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

    // a subproblem per weight vector, 101 of them 0.01 apart, each scored by PBI
    let weights = das_dennis::<2>(100);
    let moead = Moead::builder(problem.representation(), [Minimize; 2], weights)
        .decomposition(Decomposition::Pbi { theta: 5.0 })
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 24.0, 20.0)?)
        .seed(1)
        .build()?;
    run("MOEA/D", moead, &mut trace)?;

    // the box up to the reference point, less the quarter ellipse under the front, of area 2π
    let whole = REFERENCE[0] * REFERENCE[1] - 2.0 * std::f64::consts::PI;
    println!("the whole front: hypervolume {whole:.4}");
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
    let outcome = MultiEngine::new(algorithm, Wfg9::<2>::default())
        .stop_when(Stop::generations(GENERATIONS))
        .on_generation(|snapshot| {
            if snapshot.progress().generation() == FIRST {
                first = front_of(snapshot);
            }
            record(snapshot);
        })
        .run()?;
    report(name, FIRST, &first);
    report(name, GENERATIONS, &outcome.front_values());
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

// the size of a front, its IGD+ to 500 points of the optimal front, its hypervolume, and the
// distances of its points from the front
fn report(name: &str, generations: u64, front: &[[f64; 2]]) {
    let optimal = Wfg9::<2>::default().optimal_front(500).expect("known");
    let igd = igd_plus(front, &optimal, &[Minimize; 2]);
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    let distances: Vec<f64> = front.iter().map(distance).collect();
    let least = distances.iter().copied().fold(f64::INFINITY, f64::min);
    let most = distances.iter().copied().fold(0.0, f64::max);
    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    println!(
        "{name:<8} after {generations} generations: {} solutions, IGD+ {igd:.4}, hypervolume \
         {volume:.4}",
        front.len()
    );
    println!("         distance from the front {least:.4} to {most:.4}, mean {mean:.4}");
}
