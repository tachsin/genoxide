//! ZDT4: minimize two conflicting objectives over 10 variables, with the convex Pareto front of
//! ZDT1 behind 21⁹ local fronts, with NSGA-II and SMS-EMOA.
//!
//! Zitzler, Deb and Thiele's fourth problem, from genoxide's `multi::problems::Zdt4`. Runs each
//! algorithm for 500 generations, and prints its front after 250 and after 500: the size, the
//! IGD+ to 500 points of the optimal front, and the hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example zdt4
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Zdt4};
use genoxide::prelude::*;

// the reference point of the hypervolume, beyond the front's worst point (1, 1)
const REFERENCE: [f64; 2] = [1.1, 1.1];

// the generation of the first report: the NSGA-II paper's budget
const HALFWAY: u64 = 250;

fn main() -> Result<()> {
    let problem = Zdt4::new(10);
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();

    // polynomial mutation at a rate of 1/10, one gene per child on average
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 10.0, 20.0)?)
        .seed(1)
        .build()?;
    run("NSGA-II", nsga2, &mut trace)?;

    let sms_emoa = SmsEmoa::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 10.0, 20.0)?)
        .seed(1)
        .build()?;
    run("SMS-EMOA", sms_emoa, &mut trace)?;

    println!("the whole front: hypervolume 0.8767");
    trace.write();
    Ok(())
}

// runs `algorithm` for 500 generations, and reports its front after 250 and after 500
fn run<A>(name: &'static str, algorithm: A, trace: &mut trace::Trace) -> Result<()>
where
    A: MultiObjectiveAlgorithm<2, Genome = Reals>,
{
    let mut record = trace.fronts(name);
    let mut halfway = Vec::new();
    let outcome = MultiEngine::new(algorithm, Zdt4::new(10))
        .stop_when(Stop::generations(2 * HALFWAY))
        .on_generation(|snapshot| {
            if snapshot.progress().generation() == HALFWAY {
                halfway = front_of(snapshot);
            }
            record(snapshot);
        })
        .run()?;
    report(name, HALFWAY, &halfway);
    report(name, 2 * HALFWAY, &outcome.front_values());
    Ok(())
}

// the objective values of the front after a generation
fn front_of(snapshot: &MultiSnapshot<'_, Reals, 2>) -> Vec<[f64; 2]> {
    let front = snapshot.front().iter();
    front.filter_map(|x| x.fitness()?.values()).collect()
}

// the size of a front, its IGD+ to 500 points of the optimal front, evenly spaced in f₁, and its
// hypervolume
fn report(name: &str, generations: u64, front: &[[f64; 2]]) {
    let optimal = Zdt4::new(10).optimal_front(500).expect("known");
    let distance = igd_plus(front, &optimal, &[Minimize; 2]);
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    println!(
        "{name:<8} after {generations} generations: {} solutions, IGD+ {distance:.4}, \
         hypervolume {volume:.4}",
        front.len()
    );
}
