//! Double pole balancing without velocities: evolve the weights of a recurrent neural network that
//! balances two poles on a cart seeing only their angles and the cart's position, by CMA-ES.
//!
//! Wieland's (1991) double pole, with Florian's (2007) corrected equations and Gomez, Schmidhuber
//! and Miikkulainen's (2008) settings, made non-Markovian by Gruau, Whitley and Pyeatt (1996): the
//! network doesn't see the velocities, and has to infer them from what it saw before. It is an
//! Elman network: 3 inputs, 3 hidden tanh units that also receive their own previous outputs, and a
//! tanh output, without biases, 21 weights. The fitness is Gruau et al.'s damping fitness over 1000
//! steps, maximized by CMA-ES with its step size bounded below, as Igel (2003) did. The task is
//! solved, by Gruau et al.'s criteria, when the best network of a generation balances the poles
//! for 100,000 steps and, from 625 other starts, for 1000 steps from at least 200.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example double_pole_no_velocities
//! ```

mod trace;

use genoxide::nn::{Activation, Elman};
use genoxide::prelude::*;
use genoxide::problems::control::{DoublePole, GENERALIZATION_THRESHOLD, SUCCESS_STEPS};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// 3 inputs, 3 hidden units with a context, 1 output, without biases
pub fn network() -> Result<Elman> {
    let elman = Elman::new(3, 3, 1, Activation::Tanh)?;
    Ok(elman.output_activation(Activation::Tanh).bias(false))
}

// a network that solved the task: its weights, and the starts of the generalization test it passed
struct Solution {
    weights: Reals,
    generalization: u32,
    evaluations: u64,
    generation: u64,
}

// the largest |x|, |θ₁| and |θ₂| (in degrees) over an episode of `steps` steps
fn extent(task: &DoublePole, network: &Elman, weights: &[f64], steps: u32) -> Result<[f64; 3]> {
    let mut policy = network.with(weights)?;
    let mut task = *task;
    let (mut observation, mut action) = ([0.0; 3], [0.0]);
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
    let elman = network()?;
    let task = DoublePole::without_velocities();
    // Gruau et al.'s damping fitness over 1000 steps
    let damping = |weights: &Reals| -> Option<f64> {
        let mut policy = elman.with(weights).ok()?;
        Some(task.damping_fitness(&mut policy))
    };
    let cmaes = Cmaes::builder(elman.representation(-1.0..=1.0)?)
        .min_step(0.05)
        .restarts(cmaes::Restarts::Ipop)
        .maximize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(&elman, &task);
    let solved = Arc::new(AtomicBool::new(false));
    let stop = Arc::clone(&solved);
    let mut solution = None;
    let outcome = Engine::new(cmaes, damping)
        .stop_when(Stop::custom(move |_| stop.load(Ordering::Relaxed)))
        .stop_when(Stop::evaluations(100_000))
        .on_generation(|snapshot| trace.record(snapshot))
        // the tests of the generation's best network, if it balanced the 1000 steps (then its
        // fitness is at least 0.1)
        .on_generation(|snapshot| {
            let best = snapshot.population().best(Objective::Maximize);
            let score = |best: &&Individual<Reals>| best.fitness().and_then(Fitness::score);
            let Some(best) = best.filter(|best| score(best) >= Some(0.1)) else {
                return;
            };
            let mut policy = elman.with(best.genome()).expect("the network's weights");
            if task.run(&mut policy, SUCCESS_STEPS) < SUCCESS_STEPS {
                return;
            }
            let generalization = task.generalization(&mut policy);
            if generalization >= GENERALIZATION_THRESHOLD {
                let progress = snapshot.progress();
                solution = Some(Solution {
                    weights: best.genome().clone(),
                    generalization,
                    evaluations: progress.evaluations(),
                    generation: progress.generation(),
                });
                solved.store(true, Ordering::Relaxed);
            }
        })
        .run()?;

    let Some(solution) = solution else {
        println!("not solved after {} evaluations", outcome.evaluations());
        return Ok(());
    };
    println!(
        "solved after {} evaluations in {} generations",
        solution.evaluations, solution.generation
    );
    println!(
        "balanced for {SUCCESS_STEPS} steps from the start, and for 1000 steps from {} of the 625 \
         generalization starts (at least {GENERALIZATION_THRESHOLD} needed)",
        solution.generalization
    );
    let weights = &solution.weights;
    let mut policy = elman.with(weights)?;
    println!("damping fitness {:.6}", task.damping_fitness(&mut policy));
    let [position, long, short] = extent(&task, &elman, weights, SUCCESS_STEPS)?;
    println!(
        "over the 100000 steps: the cart within {position:.4} m, the poles within {long:.4}° and \
         {short:.4}°"
    );
    let rounded: Vec<String> = weights.iter().map(|w| format!("{w:.3}")).collect();
    println!("weights: [{}]", rounded.join(", "));
    trace.write(weights);
    Ok(())
}
