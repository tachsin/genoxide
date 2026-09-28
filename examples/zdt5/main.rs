//! ZDT5: minimize two conflicting objectives over a string of 80 bits, whose Pareto front is 31
//! points behind deceptive fronts, with NSGA-II.
//!
//! Zitzler, Deb and Thiele's fifth problem, from genoxide's `multi::problems::Zdt5`. Runs NSGA-II
//! three times for 250 generations: with uniform crossover and a population of 100, with
//! two-point crossover and a population of 100, and with two-point crossover and a population of
//! 1000. Prints, for each, the g of its front (10 on the optimal front), how many of the 31
//! optimal points it found, its IGD+ to them and its hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example zdt5
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Zdt5};
use genoxide::prelude::*;

// the reference point of the hypervolume: 1.1 times the nadir point (31, 10)
const REFERENCE: [f64; 2] = [34.1, 11.0];

const GENERATIONS: u64 = 250;

fn main() -> Result<()> {
    let problem = Zdt5::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();

    // bit-flip mutation at a rate of 1/80, one bit per child on average; uniform crossover takes
    // each bit from either parent
    let uniform = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / 80.0)?)
        .seed(1)
        .build()?;
    run("uniform, 100", uniform, &mut trace)?;

    // two-point crossover exchanges a segment, and keeps most 5-bit substrings whole
    for size in [100, 1000] {
        let two_point = Nsga2::builder(problem.representation(), [Minimize; 2])
            .population_size(size)
            .crossover(PointCrossover::two_point())
            .mutate(BitFlip::per_gene(1.0 / 80.0)?)
            .seed(1)
            .build()?;
        let name = if size == 100 {
            "two-point, 100"
        } else {
            "two-point, 1000"
        };
        run(name, two_point, &mut trace)?;
    }

    println!("the whole front: g = 10, 31 points, hypervolume 323.15");
    trace.write();
    Ok(())
}

// runs `algorithm` for 250 generations, and reports its front
fn run<A>(name: &'static str, algorithm: A, trace: &mut trace::Trace) -> Result<()>
where
    A: MultiObjectiveAlgorithm<2, Genome = Bits>,
{
    let outcome = MultiEngine::new(algorithm, Zdt5::default())
        .stop_when(Stop::generations(GENERATIONS))
        .on_generation(trace.fronts(name))
        .run()?;
    let front = outcome.front_values();
    // f₁ f₂ = g, a whole number: 10 on the optimal front
    let g: Vec<i64> = front.iter().map(|f| (f[0] * f[1]).round() as i64).collect();
    let low = g.iter().min().expect("a front");
    let high = g.iter().max().expect("a front");
    let range = if low == high {
        format!("= {low}")
    } else {
        format!("{low} to {high}")
    };
    // the distinct values of f₁ where g = 10
    let mut optimal: Vec<i64> = front
        .iter()
        .zip(&g)
        .filter(|&(_, &g)| g == 10)
        .map(|([f1, _], _)| *f1 as i64)
        .collect();
    optimal.sort_unstable();
    optimal.dedup();
    let optimal_front = Zdt5::default().optimal_front(31).expect("known");
    let distance = igd_plus(&front, &optimal_front, &[Minimize; 2]);
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
    println!(
        "{name:<15}  g {range:<8} {:>2} of the 31 optimal points, IGD+ {distance:.4}, \
         hypervolume {volume:.2}",
        optimal.len()
    );
    Ok(())
}
