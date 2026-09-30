//! Double pole balancing: evolve the weights of a neural network that balances two poles on a
//! cart for 100,000 steps, by CMA-ES.
//!
//! Wieland's (1991) task, with Florian's (2007) corrected equations and Gomez, Schmidhuber and
//! Miikkulainen's (2008) settings: a 1 kg cart on a 4.8 m track, with two poles side by side, of
//! 1 m and 0.1 kg and of 0.1 m and 0.01 kg, the long one starting at 4° from vertical, and a force
//! of up to 10 N every 0.02 s. The network sees the cart's position and velocity and each pole's
//! angle and angular velocity, and outputs the force: 6 inputs, 6 hidden tanh units and a tanh
//! output, without biases, 42 weights (Igel's 2003 network). The fitness is the number of steps
//! before a pole passes 36° or the cart leaves the track, maximized by CMA-ES until a network
//! balances them for 100,000 steps, over 33 minutes of simulated time.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example double_pole
//! ```

mod trace;

use genoxide::neat::{Neat, Network};
use genoxide::nn::{Activation, Mlp};
use genoxide::prelude::*;
use genoxide::problems::control::{DoublePole, SUCCESS_STEPS};

// 6 inputs, 6 hidden units, 1 output, without biases: Igel's (2003) best network for this task
pub fn network() -> Result<Mlp> {
    let mlp = Mlp::new([6, 6, 1], Activation::Tanh)?;
    Ok(mlp.output_activation(Activation::Tanh).bias(false))
}

// the largest |x|, |θ₁| and |θ₂| (in degrees) over an episode of `steps` steps
fn extent(task: &DoublePole, network: &Mlp, weights: &[f64], steps: u32) -> Result<[f64; 3]> {
    let mut policy = network.with(weights)?;
    let mut task = *task;
    let (mut observation, mut action) = ([0.0; 6], [0.0]);
    let mut largest = [0.0_f64; 3];
    for _ in 0..steps {
        task.observe(&mut observation);
        policy.forward(&observation, &mut action);
        task.step(action[0]);
        let [x, _, theta_1, _, theta_2, _] = task.state();
        let values = [
            x.abs(),
            theta_1.abs().to_degrees(),
            theta_2.abs().to_degrees(),
        ];
        for (largest, value) in largest.iter_mut().zip(values) {
            *largest = largest.max(value);
        }
    }
    Ok(largest)
}

fn main() -> Result<()> {
    let mlp = network()?;
    let task = DoublePole::new();
    // the steps balanced, up to 100,000
    let steps = |weights: &Reals| -> Option<f64> {
        let mut policy = mlp.with(weights).ok()?;
        Some(f64::from(task.run(&mut policy, SUCCESS_STEPS)))
    };
    let cmaes = Cmaes::builder(mlp.representation(-1.0..=1.0)?)
        .restarts(cmaes::Restarts::Ipop)
        .maximize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(&mlp, &task);
    let outcome = Engine::new(cmaes, steps)
        .stop_when(Stop::target(f64::from(SUCCESS_STEPS)).or(Stop::evaluations(100_000)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    println!(
        "balanced for {} steps (the goal: {SUCCESS_STEPS}) after {} evaluations in {} generations",
        outcome.best_fitness(),
        outcome.evaluations(),
        outcome.generations()
    );
    let weights = outcome.best_genome();
    let [position, long, short] = extent(&task, &mlp, weights, SUCCESS_STEPS)?;
    println!(
        "over the 100000 steps: the cart within {position:.4} m, the poles within {long:.4}° and {short:.4}°"
    );
    let rounded: Vec<String> = weights.iter().map(|w| format!("{w:.3}")).collect();
    println!("weights: [{}]", rounded.join(", "));
    trace.write(weights);
    neat(&task)?;
    Ok(())
}

// the same task by NEAT, with the paper's settings: networks that grow from the inputs and a bias
// connected to the output, whose output, in (0, 1), is the force as 2 × output − 1
fn neat(task: &DoublePole) -> Result<()> {
    let steps = |network: &Network| -> Option<f64> {
        let mut evaluator = network.feed_forward().ok()?;
        let mut policy = |observation: &[f64], action: &mut [f64]| {
            let mut output = [0.0];
            evaluator.activate(observation, &mut output);
            action[0] = 2.0 * output[0] - 1.0;
        };
        Some(f64::from(task.run(&mut policy, SUCCESS_STEPS)))
    };
    let neat = Neat::builder(6, 1).seed(1).build()?;
    let outcome = Engine::new(neat, steps)
        .stop_when(Stop::target(f64::from(SUCCESS_STEPS)).or(Stop::evaluations(100_000)))
        .run()?;
    let network = outcome.best_genome();
    println!(
        "\nNEAT: balanced for {} steps after {} evaluations in {} generations",
        outcome.best_fitness(),
        outcome.evaluations(),
        outcome.generations()
    );
    println!(
        "the network: {} hidden nodes, {} enabled connections",
        network.hidden(),
        network.enabled()
    );
    Ok(())
}
