//! WFG5: minimize two conflicting objectives over 24 variables, with a concave Pareto front and
//! deceptive parameters, with NSGA-II and SMS-EMOA.
//!
//! The Walking Fish Group's fifth problem, from genoxide's `multi::problems::Wfg5`, with the
//! recommended sizes: 4 position and 20 distance parameters. Runs each algorithm for 1,000
//! generations, and prints its front after 250 and after 1,000: the size, the IGD+ to 500 points
//! of the optimal front and the hypervolume; then how far the front is from the optimal one, and
//! where the population's distance parameters are: at the bounds, where their deceptive shift
//! leads, or near 0.35, its optimum.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example wfg5
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Wfg5};
use genoxide::prelude::*;

// the reference point of the hypervolume, 1.1 times the nadir point (2, 4)
const REFERENCE: [f64; 2] = [2.2, 4.4];

// the generation of the first report: the NSGA-II paper's budget
const FIRST: u64 = 250;

// the length of the run
const GENERATIONS: u64 = 1_000;

// the position parameters, k: the distance parameters follow them
const POSITION: usize = 4;

fn main() -> Result<()> {
    let problem = Wfg5::<2>::default();
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

    println!("the whole front: hypervolume 3.3968");
    trace.write();
    Ok(())
}

// what is reported after a generation: the front's objective values, and the population's
// distance parameters
struct Report {
    front: Vec<[f64; 2]>,
    distance_parameters: Vec<f64>,
}

impl Report {
    fn of(snapshot: &MultiSnapshot<'_, Reals, 2>) -> Self {
        let front = snapshot.front().iter();
        let front = front.filter_map(|x| x.fitness()?.values()).collect();
        let population = snapshot.population().iter();
        let distance_parameters = population.flat_map(|x| distance_parameters(x.genome()));
        Self {
            front,
            distance_parameters: distance_parameters.collect(),
        }
    }
}

// runs `algorithm` for 1,000 generations, and reports after 250 and after 1,000
fn run<A>(name: &'static str, algorithm: A, trace: &mut trace::Trace) -> Result<()>
where
    A: MultiObjectiveAlgorithm<2, Genome = Reals>,
{
    let mut record = trace.fronts(name);
    let mut reports = Vec::new();
    MultiEngine::new(algorithm, Wfg5::<2>::default())
        .stop_when(Stop::generations(GENERATIONS))
        .on_generation(|snapshot| {
            let generation = snapshot.progress().generation();
            if generation == FIRST || generation == GENERATIONS {
                reports.push((generation, Report::of(snapshot)));
            }
            record(snapshot);
        })
        .run()?;
    for (generation, report) in reports {
        print(name, generation, &report);
    }
    Ok(())
}

// the distance parameters of a genome, normalized to [0, 1]: y = z / 2i for i from k + 1
fn distance_parameters(z: &Reals) -> impl Iterator<Item = f64> + '_ {
    let genes = z.iter().enumerate().skip(POSITION);
    genes.map(|(i, zi)| zi / (2.0 * (i + 1) as f64))
}

// the distance x_M of a point from the optimal front: its objectives are x_M + 2 sin(x₁π/2) and
// x_M + 4 cos(x₁π/2), so ((f₁ − x_M) / 2)² + ((f₂ − x_M) / 4)² = 1, of which x_M is the smaller
// root
fn distance(f: &[f64; 2]) -> f64 {
    let (a, b, c) = (
        5.0 / 16.0,
        f[0] / 2.0 + f[1] / 8.0,
        f[0] * f[0] / 4.0 + f[1] * f[1] / 16.0 - 1.0,
    );
    (b - (b * b - 4.0 * a * c).max(0.0).sqrt()) / (2.0 * a)
}

// prints the size of a front, its IGD+ to 500 points of the optimal front and its hypervolume;
// then the least and the largest distance of its points from the optimal front, and how many of
// the population's distance parameters are within 0.001 of the bounds 0 and 1, and within 0.001
// of 0.35
fn print(name: &str, generations: u64, report: &Report) {
    let front = &report.front;
    let optimal = Wfg5::<2>::default().optimal_front(500).expect("known");
    let igd = igd_plus(front, &optimal, &[Minimize; 2]);
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    println!(
        "{name} after {generations} generations: {} solutions, IGD+ {igd:.4}, hypervolume \
         {volume:.4}",
        front.len()
    );
    let distances = front.iter().map(distance);
    let least = distances.clone().fold(f64::INFINITY, f64::min);
    let largest = distances.fold(0.0, f64::max);
    let parameters = &report.distance_parameters;
    let bounds = parameters.iter().filter(|&&y| y < 0.001 || y > 0.999);
    let optimum = parameters.iter().filter(|&&y| (y - 0.35).abs() < 0.001);
    println!(
        "  distance {least:.5} to {largest:.5}; genes at the bounds {}, near 0.35 {}, of {}",
        bounds.count(),
        optimum.count(),
        parameters.len()
    );
}
