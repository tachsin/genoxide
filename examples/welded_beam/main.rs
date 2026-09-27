//! Welded beam design: the cheapest beam welded to a support that carries 6000 lb at 14 inches,
//! subject to its weld's shear stress, its bending stress, its buckling load and its deflection.
//! A constrained continuous problem, in the two forms of the literature.
//!
//! `WeldedBeam` is the form with seven constraints (Rao, 1996, as restated by Coello Coello,
//! 2000), `WeldedBeamRagsdell` the one with five (Ragsdell and Phillips, 1976, as restated by
//! Deb, 2000). Their fitness is the cost and the constraint violation, which Deb's feasibility
//! rules compare. SHADE, a differential evolution, solves each with the same budget, and the
//! example prints the best design next to the best known cost.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example welded_beam
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::engineering::{WeldedBeam, WeldedBeamRagsdell};
use genoxide::problems::{self, DynProblem};

fn main() -> Result<()> {
    let forms: [Box<dyn DynProblem>; 2] = [
        problems::boxed(WeldedBeam),
        problems::boxed(WeldedBeamRagsdell),
    ];
    // with GENOXIDE_TRACE=<file>, a trace of the first form's run for the plot on the example's
    // page
    let mut trace = trace::Trace::from_env();
    for problem in &forms {
        let best_known = problem.optimum().expect("known").value();
        let de = De::builder(problem.real()).minimize().seed(1).build()?;
        let outcome = Engine::new(de, |x: &Reals| problem.evaluate(x))
            .stop_when(Stop::evaluations(40_000))
            .on_generation(|snapshot| trace.record(snapshot))
            .run()?;
        let best = outcome.best_fitness();
        let x = outcome.best_genome();
        println!(
            "{}: cost {:.6}, violation {:.6} (the best known: {best_known})",
            problem.name(),
            best.score().unwrap_or(f64::NAN),
            best.violation()
        );
        println!(
            "  h {:.6}, l {:.6}, t {:.6}, b {:.6}",
            x[0], x[1], x[2], x[3]
        );
    }
    trace.write();
    Ok(())
}
