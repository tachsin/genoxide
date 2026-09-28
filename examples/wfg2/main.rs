//! WFG2: minimize two objectives over 24 variables, with a convex front in six disconnected
//! regions and non-separable distance parameters, with NSGA-II and MOEA/D.
//!
//! The second problem of Huband, Hingston, Barone and While's WFG toolkit, from genoxide's
//! `multi::problems::Wfg2`, with 2 objectives and the recommended sizes, k = 4 and l = 20. Runs
//! each algorithm for 1,000 generations, and prints its front after 250 and after 1,000: the size,
//! the solutions on each region of the optimal front, the IGD+ to 500 points of it, and the
//! hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example wfg2
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Wfg2};
use genoxide::multi::{MultiSnapshot, das_dennis};
use genoxide::prelude::*;

// the reference point of the hypervolume: 1.1 times the front's worst point, (2, 4)
const REFERENCE: [f64; 2] = [2.2, 4.4];

// the generation of the first report: the NSGA-II paper's budget
const FIRST: u64 = 250;

// the generation of the last report
const LAST: u64 = 1_000;

// the six regions of the optimal front, as ranges of f₁ = 2 (1 − cos(x₁π/2)), from the ranges of
// x₁ where 1 − x₁ cos²(5πx₁) is below all its values at smaller x₁
const REGIONS: [(f64, f64); 6] = [
    (0.0, 0.0043),
    (0.0414, 0.1074),
    (0.3028, 0.3912),
    (0.7351, 0.8331),
    (1.2904, 1.3894),
    (1.9133, 2.0),
];

fn main() -> Result<()> {
    let problem = Wfg2::<2>::default();
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

    // 101 weight vectors, evenly spread, and the same operators
    let moead = Moead::builder(
        problem.representation(),
        [Minimize; 2],
        das_dennis::<2>(100),
    )
    .crossover(SimulatedBinaryCrossover::new(15.0)?)
    .mutate(PolynomialMutation::per_gene(1.0 / 24.0, 20.0)?)
    .seed(1)
    .build()?;
    run("MOEA/D", moead, &mut trace)?;

    println!("the whole front: hypervolume 6.1511");
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
    let outcome = MultiEngine::new(algorithm, Wfg2::<2>::default())
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

// the size of a front, its solutions on each region (f₁ within 0.01 of the region's range), its
// IGD+ to 500 points of the optimal front, evenly spread along it, and its hypervolume. A front has
// each solution once, though MOEA/D's subproblems can hold copies of one.
fn report(name: &str, generations: u64, front: &[[f64; 2]]) {
    let optimal = Wfg2::<2>::default().optimal_front(500).expect("known");
    let distance = igd_plus(front, &optimal, &[Minimize; 2]);
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    let counts: Vec<String> = REGIONS
        .iter()
        .map(|(low, high)| {
            let on = front
                .iter()
                .filter(|[f1, _]| (low - 0.01..=high + 0.01).contains(f1));
            on.count().to_string()
        })
        .collect();
    println!(
        "{name:<7} after {generations} generations: {} solutions ({}), IGD+ {distance:.4}, \
         hypervolume {volume:.4}",
        front.len(),
        counts.join(", ")
    );
}
