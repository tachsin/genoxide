//! Koza's 11-multiplexer: find the Boolean function that uses 3 address bits to select one of 8
//! data bits, from all 2048 cases of its truth table, by genetic programming.
//!
//! Trees of Koza's functions (and, or, not, if) and the 11 inputs, evolved by a genetic
//! algorithm with subtree crossover and a mix of mutations, and double tournaments against
//! bloat. The fitness is the number of the 2048 cases a tree gets wrong; the run stops when it
//! gets all of them right.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example multiplexer_11
//! ```

mod trace;

use genoxide::gp::boolean::Multiplexer;
use genoxide::gp::{Gp, Mutations, SubtreeCrossover, Tree};
use genoxide::prelude::*;

const POPULATION: usize = 4000;

fn main() -> Result<()> {
    let problem = Multiplexer::new(3)?;
    let set = problem.primitives().clone();
    println!("Koza's 11-multiplexer: 3 address bits select one of 8 data bits, 2048 cases");
    println!("{POPULATION} trees, until every case is right\n");

    // Koza's limits and initialization: depth 17, ramped half-and-half of depths 2 to 6
    let gp = Gp::builder(set.clone()).build()?;
    let initial = gp.ramped_half_and_half(POPULATION, &mut StreamRng::seed_from_u64(1))?;
    let ga = Ga::builder(gp)
        .population_size(POPULATION)
        .initial_genomes(initial)
        .select(DoubleTournament::new(7, 1.4)?)
        .crossover(SubtreeCrossover::new())
        .mutate(
            Mutations::builder()
                .subtree(0.5)
                .point(0.3)
                .hoist(0.1)
                .shrink(0.1)
                .build()?,
        )
        .crossover_rate(0.9)
        .mutation_rate(0.1)
        .minimize()
        .seed(1)
        .build()?;
    let mut trace = trace::Trace::from_env(&problem);
    let outcome = Engine::new(ga, |tree: &Tree| problem.errors(tree) as f64)
        .stop_when(Stop::target(0.0).or(Stop::generations(50)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best = outcome.best_genome();
    println!(
        "{:?} after {} generations and {} evaluations",
        outcome.stop_reason(),
        outcome.generations(),
        outcome.evaluations()
    );
    println!(
        "cases right: {} of {}",
        problem.cases() - problem.errors(best),
        problem.cases()
    );
    println!(
        "\nthe function, {} nodes of depth {}:\n{}",
        best.len(),
        best.depth(&set),
        best.display(&set)
    );
    trace.write();
    Ok(())
}
