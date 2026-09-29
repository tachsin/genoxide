//! XOR neuroevolution: evolve the 9 weights of a 2-2-1 neural network until it computes XOR.
//!
//! XOR isn't linearly separable, so the network needs its hidden layer: two sigmoid units, each
//! with a weight per input and a bias, and a sigmoid output unit with a weight per hidden unit
//! and a bias. The fitness is the sum of the squared errors over the four input pairs, minimized
//! by CMA-ES with BIPOP restarts, which escape the flat regions where the network outputs 0.5 or
//! solves three of the four cases. With the weights in [-10, 10], the smallest known error is
//! 2.162140e-4, with every weight on a bound.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example xor_neuroevolution
//! ```

mod trace;

use genoxide::prelude::*;

// the inputs and the expected output
const CASES: [([f64; 2], f64); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

// the smallest known squared error with the weights in [-10, 10]: every weight on a bound, the
// biases inside
const MINIMUM: f64 = 2.162_140e-4;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// the network's output: w[0..3] and w[3..6] are the hidden units' weights and biases, w[6..9] the
// output unit's
fn output(w: &[f64], [a, b]: [f64; 2]) -> f64 {
    let hidden_1 = sigmoid(w[0] * a + w[1] * b + w[2]);
    let hidden_2 = sigmoid(w[3] * a + w[4] * b + w[5]);
    sigmoid(w[6] * hidden_1 + w[7] * hidden_2 + w[8])
}

fn squared_error(w: &Reals) -> f64 {
    let mut error = 0.0;
    for (input, expected) in CASES {
        let difference = output(w, input) - expected;
        error += difference * difference;
    }
    error
}

fn main() -> Result<()> {
    let cmaes = Cmaes::builder(Real::uniform(9, -10.0..=10.0)?)
        .restarts(cmaes::Restarts::Bipop)
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(cmaes, squared_error)
        .stop_when(Stop::target(MINIMUM + 1e-6).or(Stop::evaluations(200_000)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    println!(
        "squared error {:.9} after {} evaluations (the smallest known: {MINIMUM:.9})",
        outcome.best_fitness(),
        outcome.evaluations()
    );
    for ([a, b], expected) in CASES {
        let value = output(outcome.best_genome(), [a, b]);
        println!("{a:.0} xor {b:.0} = {expected:.0}: {value:.3}");
    }
    trace.write();
    Ok(())
}
