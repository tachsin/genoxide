//! CEC 2006 g22: the linear function x1 of 22 variables under 1 nonlinear inequality and 19
//! equality constraints, from the CEC 2006 special session on constrained optimization (Liang et
//! al., 2006). The best known value is 236.430975504001, with the equalities met within the
//! report's tolerance of 0.0001.
//!
//! genoxide's `G22` gives the value of a solution and its constraint violation, which Deb's
//! feasibility rules compare: a feasible solution beats an infeasible one. L-SHADE on the 22
//! variables, for the report's budget of 500,000 evaluations, doesn't find a feasible solution.
//! The 19 equalities can be solved in order, though, from x1, x8 and x9: SHADE on those three,
//! with the other 19 variables solved from them, finds the least value with every equality met,
//! 236.370313, below the report's best known. The example prints both runs, and the second's
//! solution and constraints.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of the second run for the plot on the
//! example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cec2006_g22
//! ```

mod trace;

use genoxide::math;
use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::{EQUALITY_TOLERANCE, G22};

// the CEC 2006 report's budget of evaluations per run
const BUDGET: u64 = 500_000;
// the least value with every equality met exactly, at x8 = 130 and x9 = 170 (genoxide's docs)
const EXACT: f64 = 236.370_313_314_566;
// the second run stops once its best is feasible and within this of EXACT
const ERROR: f64 = 1e-8;
// a constraint within this of its boundary is active
const ACTIVE: f64 = 1e-6;

// the 22 variables from x1, x8 and x9, the other 19 solved from the equalities in order, each
// kept within its bounds: a variable that its bounds cut leaves its equality unmet
fn solve(genes: &Reals) -> Reals {
    let representation = G22::default().representation();
    let bounds = representation.bounds();
    // x[i] is x_i, with x[0] unused
    let mut x = [0.0; 23];
    let within = |i: usize, value: f64| value.clamp(*bounds[i - 1].start(), *bounds[i - 1].end());
    (x[1], x[8], x[9]) = (genes[0], genes[1], genes[2]);
    // h1 to h6, h10 and h11 are linear
    x[5] = within(5, 100_000.0 * x[8] - 1e7);
    x[6] = within(6, 100_000.0 * x[9] - 100_000.0 * x[8]);
    x[7] = within(7, 5e7 - 100_000.0 * x[9]);
    x[10] = within(10, (3.3e7 - x[5]) / 100_000.0);
    x[11] = within(11, (4.4e7 - x[6]) / 100_000.0);
    x[12] = within(12, (6.6e7 - x[7]) / 100_000.0);
    x[16] = within(16, x[11] - x[8]);
    x[17] = within(17, x[12] - x[9]);
    // h12 to h16 give the logarithms, h17 to h19 x13 to x15, and h7 to h9 x2 to x4
    x[18] = within(18, math::ln(x[10] - 100.0));
    x[19] = within(19, math::ln(-x[8] + 300.0));
    x[20] = within(20, math::ln(x[16]));
    x[21] = within(21, math::ln(-x[9] + 400.0));
    x[22] = within(22, math::ln(x[17]));
    x[13] = within(13, (x[8] + x[10] - 400.0) / (x[18] - x[19]));
    x[14] = within(14, (x[9] + x[11] - x[8] - 400.0) / (x[20] - x[21]));
    x[15] = within(15, (x[12] - x[9] - 100.0) / (x[22] - 4.60517));
    x[2] = within(2, x[5] / (120.0 * x[13]));
    x[3] = within(3, x[6] / (80.0 * x[14]));
    x[4] = within(4, x[7] / (40.0 * x[15]));
    Reals::from(x[1..].to_vec())
}

fn main() -> Result<()> {
    // the report's equality tolerance, EQUALITY_TOLERANCE
    let problem = G22::default();
    let optimum = problem.optimum().expect("known");

    // the 22 variables, as the report poses the problem: SHADE with a population that shrinks
    // over the budget, from 18 · 22 = 396 to 4
    let l_shade = De::l_shade(problem.representation(), BUDGET)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(l_shade, problem)
        .stop_when(Stop::evaluations(BUDGET))
        .run()?;
    println!("L-SHADE on the 22 variables with Deb's feasibility rules, seed 1");
    println!(
        "stopped after {} evaluations: {}",
        outcome.evaluations(),
        feasibility(outcome.best_fitness())
    );

    // x1, x8 and x9 within their bounds, and the other 19 variables solved from the equalities
    let representation = problem.representation();
    let bounds = representation.bounds();
    let free = Real::new([bounds[0].clone(), bounds[7].clone(), bounds[8].clone()])?;
    let shade = De::builder(free).minimize().seed(1).build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(shade, |genes: &Reals| problem.evaluate(&solve(genes)))
        .stop_when(Stop::target(EXACT + ERROR).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| trace.record(snapshot, solve))
        .run()?;
    let best = outcome.best_fitness();
    let value = best.score().expect("valid");
    let x = solve(outcome.best_genome());
    println!("SHADE on x1, x8 and x9, the other 19 variables solved from the equalities, seed 1");
    let stop = if outcome.stop_reason() == StopReason::Target {
        "stopped by the target"
    } else {
        "stopped"
    };
    let evaluations = outcome.evaluations();
    println!(
        "{stop} after {evaluations} evaluations: {}",
        feasibility(best)
    );
    println!(
        "f(x) {value:.6}, {:.1e} above {EXACT:.6}, the least with every equality met",
        value - EXACT
    );
    println!(
        "f(x) - f* {:.6}, where f* {:.6} is the report's best known",
        value - optimum.value(),
        optimum.value()
    );
    let genes: Vec<String> = (1..)
        .zip(&x[..])
        .map(|(i, &xi)| format!("x{i} {}", significant(xi, 6)))
        .collect();
    println!("{}", genes.join(", "));
    // g1, then h1 to h19, each equality as its excess over the tolerance, 0 when it's met
    let constraints = problem.constraints(&x);
    let inequalities = (1..)
        .zip(constraints.inequalities())
        .map(|(i, &g)| format!("g{i} {}", state(g)));
    let equalities = (1..)
        .zip(constraints.equalities())
        .map(|(i, &h)| format!("h{i} {}", state((h.abs() - EQUALITY_TOLERANCE).max(0.0))));
    let constraints: Vec<String> = inequalities.chain(equalities).collect();
    println!("{}", constraints.join(", "));
    trace.write();
    Ok(())
}

// "feasible", or "infeasible" with the violation
fn feasibility(fitness: Fitness) -> String {
    if fitness.is_feasible() {
        "feasible".to_string()
    } else {
        format!(
            "infeasible, violation {}",
            significant(fitness.violation(), 4)
        )
    }
}

// a constraint g(x) <= 0: "active" on its boundary, else its value
fn state(g: f64) -> String {
    if g.abs() <= ACTIVE {
        "active".to_string()
    } else {
        significant(g, 4)
    }
}

// `digits` significant digits, e.g. 29.9953 or -30665.5 for 6
fn significant(value: f64, digits: i32) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = (digits - 1 - magnitude).max(0) as usize;
    format!("{value:.decimals$}")
}
