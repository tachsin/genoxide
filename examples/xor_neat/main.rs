//! XOR by NEAT: evolve a network's structure and weights until it computes XOR, from networks
//! without hidden nodes.
//!
//! The NEAT paper's first experiment (Stanley and Miikkulainen 2002): XOR isn't linearly
//! separable, so a network needs at least one hidden node, which NEAT has to discover. The
//! initial networks connect the two inputs and the bias straight to the output; mutations add
//! nodes and connections, speciation protects the new structure while its weights are tuned. The
//! fitness is the paper's, (4 − Σ|error|)², maximized, with its fitness sharing, and the run stops
//! at the paper's success criterion: every output on the right side of 0.5.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example xor_neat
//! ```

mod trace;

use genoxide::neat::{Neat, Network, NodeKind, Sharing};
use genoxide::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// the inputs and the expected output
const CASES: [([f64; 2], f64); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

// the network's outputs for the four cases
fn outputs(network: &Network) -> [f64; 4] {
    let mut evaluator = network.feed_forward().expect("feed-forward networks");
    let mut values = [0.0; 4];
    for (value, (input, _)) in values.iter_mut().zip(&CASES) {
        let mut output = [0.0];
        evaluator.activate(input, &mut output);
        *value = output[0];
    }
    values
}

// the paper's fitness: (4 − Σ|error|)², 16 for a perfect network
fn fitness(network: &Network) -> f64 {
    let error: f64 = outputs(network)
        .iter()
        .zip(&CASES)
        .map(|(output, (_, target))| (output - target).abs())
        .sum();
    (4.0 - error).powi(2)
}

// the paper's success criterion: every output on the right side of 0.5
fn solves(network: &Network) -> bool {
    let outputs = outputs(network);
    outputs
        .iter()
        .zip(&CASES)
        .all(|(output, (_, target))| (*output >= 0.5) == (*target == 1.0))
}

fn main() -> Result<()> {
    // the paper's settings, with its fitness sharing: the fitness is maximized and non-negative
    let neat = Neat::builder(2, 1)
        .population_size(150)
        .sharing(Sharing::Raw)
        .seed(1)
        .build()?;
    println!("XOR by NEAT: 150 networks, from 2 inputs and a bias connected to the output");
    // stop at the first generation with a network that solves it
    let abort = Arc::new(AtomicBool::new(false));
    let mut solution: Option<Network> = None;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(neat, fitness)
        .stop_when(Stop::generations(1000))
        .abort_flag(Arc::clone(&abort))
        .on_generation(|snapshot| {
            trace.record(snapshot);
            if solution.is_none() {
                let population = snapshot.population();
                if let Some(found) = population.iter().find(|i| solves(i.genome())) {
                    solution = Some(found.genome().clone());
                    abort.store(true, Ordering::Relaxed);
                }
            }
        })
        .run()?;
    let Some(network) = solution else {
        println!("not solved after {} generations", outcome.generations());
        trace.write();
        return Ok(());
    };
    println!(
        "solved in generation {} after {} evaluations\n",
        outcome.generations(),
        outcome.evaluations()
    );
    println!(
        "the network: {} hidden nodes, {} enabled connections of {}",
        network.hidden(),
        network.enabled(),
        network.connections().len()
    );
    let name = |id: u32| match network
        .nodes()
        .iter()
        .find(|n| n.id() == id)
        .map(|n| n.kind())
    {
        Some(NodeKind::Input) => format!("in{id}"),
        Some(NodeKind::Bias) => "bias".to_string(),
        Some(NodeKind::Output) => "out".to_string(),
        _ => format!("h{id}"),
    };
    for connection in network.connections().iter().filter(|c| c.is_enabled()) {
        println!(
            "  {:>4} -> {:<4} {:>8.3}",
            name(connection.from()),
            name(connection.to()),
            connection.weight()
        );
    }
    println!();
    for (output, ([a, b], expected)) in outputs(&network).iter().zip(CASES) {
        println!("{a:.0} xor {b:.0} = {expected:.0}: {output:.3}");
    }
    trace.write();
    Ok(())
}
