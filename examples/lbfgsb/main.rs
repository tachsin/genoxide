//! L-BFGS-B: minimize Rosenbrock's function in 100 dimensions from the classic start
//! (−1.2, 1, −1.2, 1, …) to f ≤ 1e-10, once with its analytic gradient and once with forward
//! differences, to compare their cost.
//!
//! Both runs take about the same steps: the gradient is the same to about 7 digits. The analytic
//! gradient costs one evaluation per trial point, forward differences 101.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example lbfgsb
//! ```

mod trace;

use genoxide::gradient::Gradients;
use genoxide::prelude::*;
use genoxide::problems::{Problem, Rosenbrock};

const N: usize = 100;
// a row of the table every this many rounds
const EVERY: u64 = 50;

// a round of a run: its evaluations and best value
#[derive(Clone, Copy)]
struct Round {
    evaluations: u64,
    best: f64,
}

// the rounds of a run to f ≤ 1e-10, its outcome and its iterations
fn run(gradients: Gradients) -> Result<(Vec<Round>, Outcome<Reals>, u64)> {
    let problem = Rosenbrock::new(N);
    let start: Reals = (0..N)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect();
    let lbfgsb = Lbfgsb::builder(problem.representation())
        .initial_genome(start)
        .gradients(gradients)
        .gradient_tolerance(0.0)
        .function_tolerance(0.0)
        .minimize()
        .build()?;
    let mut rounds = Vec::new();
    let mut engine = Engine::new(lbfgsb, problem)
        .stop_when(Stop::target(1e-10).or(Stop::evaluations(200_000)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            rounds.push(Round {
                evaluations: progress.evaluations(),
                best: progress.best().and_then(Fitness::score).expect("valid"),
            });
        });
    let outcome = engine.run()?;
    let iterations = engine.algorithm().iterations();
    drop(engine);
    Ok((rounds, outcome, iterations))
}

fn main() -> Result<()> {
    let start: Reals = (0..N)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect();
    let value = Rosenbrock::new(N).evaluate(&start);
    let (analytic, analytic_outcome, analytic_iterations) = run(Gradients::Auto)?;
    let (forward, forward_outcome, forward_iterations) = run(Gradients::Forward { step: None })?;

    println!(
        "Rosenbrock's function in {N} dimensions, from (-1.2, 1, -1.2, 1, ...) where f = {value:.0}"
    );
    println!(
        "L-BFGS-B with 10 pairs to f <= 1e-10: the analytic gradient, and forward differences"
    );
    println!("       analytic gradient        forward differences");
    println!("round  evaluations  best value  evaluations  best value");
    let rounds = analytic.len().max(forward.len());
    let cell = |rounds: &[Round], round: usize| match rounds.get(round) {
        Some(row) => format!("{:>11}  {:>10}", row.evaluations, scientific(row.best)),
        None => format!("{:>11}  {:>10}", "", ""),
    };
    for round in 0..rounds {
        let last = round + 1 == analytic.len() || round + 1 == forward.len();
        if (round as u64).is_multiple_of(EVERY) || last {
            println!(
                "{round:>5}  {}  {}",
                cell(&analytic, round),
                cell(&forward, round)
            );
        }
    }
    for (name, outcome, iterations) in [
        ("analytic gradient", &analytic_outcome, analytic_iterations),
        ("forward differences", &forward_outcome, forward_iterations),
    ] {
        assert_eq!(outcome.stop_reason(), StopReason::Target);
        println!(
            "{name}: f = {} after {iterations} iterations and {} evaluations",
            scientific(outcome.best_fitness().score().expect("valid")),
            outcome.evaluations()
        );
    }
    println!(
        "forward differences took {:.0} times the evaluations",
        forward_outcome.evaluations() as f64 / analytic_outcome.evaluations() as f64
    );

    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let analytic: Vec<(u64, f64)> = analytic.iter().map(|r| (r.evaluations, r.best)).collect();
    let forward: Vec<(u64, f64)> = forward.iter().map(|r| (r.evaluations, r.best)).collect();
    trace::write_runs(&analytic, &forward);
    Ok(())
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
