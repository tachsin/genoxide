//! Cart-pole: evolve the weights of a neural network that balances a pole on a cart for 100,000
//! steps, by CMA-ES.
//!
//! The classic control task (Barto, Sutton and Anderson 1983), with Florian's (2007) corrected
//! equations and Gomez, Schmidhuber and Miikkulainen's (2008) settings: a 1 kg cart on a 4.8 m
//! track, a pole of 1 m and 0.1 kg starting at 4° from vertical, and a force of up to 10 N every
//! 0.02 s. The network sees the cart's position and velocity and the pole's angle and angular
//! velocity, and outputs the force: 4 inputs, 8 hidden tanh units and a tanh output, without
//! biases, 40 weights. The fitness is the number of steps before the pole passes 12° or the cart
//! leaves the track, maximized by CMA-ES until a network balances it for 100,000 steps, over 33
//! minutes of simulated time.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cart_pole
//! ```

mod trace;

use genoxide::nn::{Activation, Mlp};
use genoxide::prelude::*;
use genoxide::problems::control::{CartPole, SUCCESS_STEPS};

// 4 inputs, 8 hidden units, 1 output, without biases: Igel's (2003) network for this task
pub fn network() -> Result<Mlp> {
    let mlp = Mlp::new([4, 8, 1], Activation::Tanh)?;
    Ok(mlp.output_activation(Activation::Tanh).bias(false))
}

// the largest |x| and |θ| (in degrees) over an episode of `steps` steps
fn extent(task: &CartPole, network: &Mlp, weights: &[f64], steps: u32) -> Result<(f64, f64)> {
    let mut policy = network.with(weights)?;
    let mut task = *task;
    let (mut observation, mut action) = ([0.0; 4], [0.0]);
    let (mut position, mut angle) = (0.0_f64, 0.0_f64);
    for _ in 0..steps {
        task.observe(&mut observation);
        policy.forward(&observation, &mut action);
        task.step(action[0]);
        let [x, _, theta, _] = task.state();
        position = position.max(x.abs());
        angle = angle.max(theta.abs().to_degrees());
    }
    Ok((position, angle))
}

fn main() -> Result<()> {
    let mlp = network()?;
    let task = CartPole::new();
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
    let (position, angle) = extent(&task, &mlp, weights, SUCCESS_STEPS)?;
    println!("over the 100000 steps: the cart within {position:.4} m, the pole within {angle:.4}°");
    let rounded: Vec<String> = weights.iter().map(|w| format!("{w:.3}")).collect();
    println!("weights: [{}]", rounded.join(", "));
    trace.write(weights);
    Ok(())
}
