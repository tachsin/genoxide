//! The cantilever beam (Fleury and Braibant, 1986): the lightest beam of five hollow square
//! segments that carries a load at its free end, subject to its deflection. A constrained
//! continuous problem with a proven minimum, 1.339956361.
//!
//! The variables are the widths of the five segments, from the support to the free end. The
//! weight grows with their sum, and one constraint limits the deflection. The fitness is the
//! weight and the constraint violation, which Deb's feasibility rules compare. CMA-ES searches
//! the widths, and the example prints the best design next to the minimum.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cantilever_beam
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::engineering::CantileverBeam;

fn main() -> Result<()> {
    let problem = CantileverBeam;
    let optimum = problem.optimum().expect("known");
    let cmaes = Cmaes::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(cmaes, problem)
        .stop_when(Stop::evaluations(8_000))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best = outcome.best_fitness();
    let weight = best.score().unwrap_or(f64::NAN);
    let minimum = optimum.value();
    println!(
        "weight {weight:.9} after {} evaluations (the minimum: {minimum:.9})",
        outcome.evaluations()
    );
    println!("violation {:.6}", best.violation());
    // the widths, from the support to the free end, next to the minimum's
    println!("segment  width     the minimum's");
    let x = outcome.best_genome();
    for (i, (width, exact)) in x.iter().zip(optimum.solutions()[0].iter()).enumerate() {
        println!("{:>7}  {width:.6}  {exact:.6}", i + 1);
    }
    // the constraint is 61/x1³ + 37/x2³ + 19/x3³ + 7/x4³ + 1/x5³ ≤ 1, as g = that sum − 1 ≤ 0
    let deflection = 1.0 + problem.constraints(x).inequalities()[0];
    println!("61/x1³ + 37/x2³ + 19/x3³ + 7/x4³ + 1/x5³ = {deflection:.6} (at most 1)");
    trace.write();
    Ok(())
}
