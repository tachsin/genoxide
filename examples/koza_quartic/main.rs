//! Koza's quartic: find the formula x⁴ + x³ + x² + x from 20 points of it, by genetic
//! programming.
//!
//! The classic first problem of genetic programming (Koza 1992): trees of Koza's functions (+,
//! −, ×, protected division, sin, cos, exp and a protected logarithm) and the variable x, evolved
//! by a genetic algorithm with subtree crossover and mutation, fitted to the root mean squared
//! error on the points. The run stops at exact recovery: an error at the level of rounding, on
//! the 20 training points and on 101 test points across [−1, 1], with the expression printed.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example koza_quartic
//! ```

mod trace;

use genoxide::gp::{Columns, Gp, PrimitiveSet, SubtreeCrossover, SubtreeMutation, Tree};
use genoxide::math;
use genoxide::prelude::*;
use rand::RngExt;
use std::cell::RefCell;

// Koza's function set for symbolic regression, and the variable
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Cos,
    Exp,
    Rlog,
    X,
}

pub fn primitives() -> Result<PrimitiveSet<Op>> {
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    set.function("add", Op::Add, [real, real], real)
        .function("sub", Op::Sub, [real, real], real)
        .function("mul", Op::Mul, [real, real], real)
        .function("div", Op::Div, [real, real], real)
        .function("sin", Op::Sin, [real], real)
        .function("cos", Op::Cos, [real], real)
        .function("exp", Op::Exp, [real], real)
        .function("rlog", Op::Rlog, [real], real)
        .terminal("x", Op::X, real);
    set.build(real)
}

pub fn quartic(x: f64) -> f64 {
    x * x * x * x + x * x * x + x * x + x
}

// the column of `op` from its arguments' columns, at the points `xs`; with genoxide's `math`, the
// same bits on every platform
fn apply(op: Op, args: &[&[f64]], out: &mut [f64], xs: &[f64]) {
    let unary = |out: &mut [f64], f: fn(f64) -> f64| {
        for (out, &a) in out.iter_mut().zip(args[0]) {
            *out = f(a);
        }
    };
    let binary = |out: &mut [f64], f: fn(f64, f64) -> f64| {
        for ((out, &a), &b) in out.iter_mut().zip(args[0]).zip(args[1]) {
            *out = f(a, b);
        }
    };
    match op {
        Op::Add => binary(out, |a, b| a + b),
        Op::Sub => binary(out, |a, b| a - b),
        Op::Mul => binary(out, |a, b| a * b),
        // Koza's protected division: 1 for a zero denominator
        Op::Div => binary(out, |a, b| if b == 0.0 { 1.0 } else { a / b }),
        Op::Sin => unary(out, math::sin),
        Op::Cos => unary(out, math::cos),
        Op::Exp => unary(out, math::exp),
        // Koza's protected logarithm: ln |a|, and 0 at 0
        Op::Rlog => unary(out, |a| if a == 0.0 { 0.0 } else { math::ln(a.abs()) }),
        Op::X => out.copy_from_slice(xs),
    }
}

// points and the quartic's values at them
pub struct Data {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

impl Data {
    fn new(xs: Vec<f64>) -> Self {
        let ys = xs.iter().map(|&x| quartic(x)).collect();
        Self { xs, ys }
    }

    // the tree's values at the points
    pub fn predict(&self, set: &PrimitiveSet<Op>, tree: &Tree) -> Vec<f64> {
        thread_local! {
            static COLUMNS: RefCell<Columns> = RefCell::default();
        }
        COLUMNS.with_borrow_mut(|columns| {
            if columns.points() != self.xs.len() {
                *columns = Columns::new(self.xs.len());
            }
            let values =
                tree.evaluate_columns(set, columns, |op, args, out| apply(op, args, out, &self.xs));
            values.to_vec()
        })
    }

    // the root mean squared error of the tree, None if a value isn't finite
    pub fn rmse(&self, set: &PrimitiveSet<Op>, tree: &Tree) -> Option<f64> {
        let predictions = self.predict(set, tree);
        let mut sum = 0.0;
        for (prediction, y) in predictions.iter().zip(&self.ys) {
            let error = prediction - y;
            sum += error * error;
        }
        let rmse = (sum / self.ys.len() as f64).sqrt();
        rmse.is_finite().then_some(rmse)
    }

    // the standard deviation of the values
    fn deviation(&self) -> f64 {
        let n = self.ys.len() as f64;
        let mean = self.ys.iter().sum::<f64>() / n;
        let squares: f64 = self.ys.iter().map(|y| (y - mean) * (y - mean)).sum();
        (squares / n).sqrt()
    }
}

// the islands, their trees, and the generations between migrations
const ISLANDS: u64 = 8;
const POPULATION: usize = 500;
const INTERVAL: u64 = 10;

fn main() -> Result<()> {
    // 20 training points uniform in [-1, 1], from a fixed seed, and 101 test points evenly spaced
    let mut rng = StreamRng::seed_from_u64(20);
    let training = Data::new((0..20).map(|_| rng.random_range(-1.0..=1.0)).collect());
    let test = Data::new((0..=100).map(|i| f64::from(i) / 50.0 - 1.0).collect());
    // exact recovery: an error of at most 1e-10 of the values' standard deviation
    let tolerance = 1e-10 * training.deviation();
    println!("Koza's quartic x^4 + x^3 + x^2 + x from 20 points in [-1, 1]");
    println!(
        "{ISLANDS} islands of {POPULATION} trees, until the RMSE is at most {tolerance:.2e}
"
    );

    // Koza's limits and initialization: depth 17, ramped half-and-half of depths 2 to 6
    let gp = Gp::builder(primitives()?).build()?;
    let set = gp.primitives().clone();
    let islands = (0..ISLANDS)
        .map(|island| {
            let seed = 100 + island;
            // Koza's even division among the depths and methods, without duplicates
            let initial =
                gp.ramped_half_and_half(POPULATION, &mut StreamRng::seed_from_u64(seed))?;
            Ga::builder(gp.clone())
                .population_size(POPULATION)
                .initial_genomes(initial)
                .select(Tournament::new(7)?)
                .crossover(SubtreeCrossover::new())
                .mutate(SubtreeMutation::new())
                .crossover_rate(0.9)
                .mutation_rate(0.1)
                .minimize()
                .seed(seed)
                .build()
        })
        .collect::<Result<Vec<_>>>()?;
    let islands = Islands::builder(islands)
        .interval(INTERVAL)
        .migrants(2)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(&training, &set);
    let outcome = Engine::new(islands, |tree: &Tree| training.rmse(&set, tree))
        .stop_when(Stop::target(tolerance).or(Stop::generations(200)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best = outcome.best_genome();
    let rmse = |data: &Data| data.rmse(&set, best).unwrap_or(f64::NAN);
    println!(
        "{:?} after {} generations and {} evaluations",
        outcome.stop_reason(),
        outcome.generations(),
        outcome.evaluations()
    );
    println!("RMSE on the 20 training points: {:.2e}", rmse(&training));
    println!("RMSE on the 101 test points:    {:.2e}", rmse(&test));
    println!(
        "
the expression, {} nodes of depth {}:
{}",
        best.len(),
        best.depth(&set),
        best.display(&set)
    );
    trace.write();
    Ok(())
}
