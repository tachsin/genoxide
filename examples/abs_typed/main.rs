//! |x| by strongly typed genetic programming: find the absolute value from 20 points of it, with
//! a comparison that returns a Boolean and a conditional that takes one.
//!
//! Two types, real numbers and Booleans (Montana 1995): `less` compares two reals and returns a
//! Boolean, `if` takes a Boolean and two reals, and every tree genoxide makes puts a Boolean
//! where a Boolean goes. The run stops at exact recovery: an error at the level of rounding, on
//! the 20 training points and on 101 test points across [−1, 1], with the expression printed.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example abs_typed
//! ```

mod trace;

use genoxide::gp::{Constants, Gp, Mutations, PrimitiveSet, SubtreeCrossover, Tree};
use genoxide::prelude::*;
use rand::RngExt;

// the primitives: arithmetic on reals, a comparison, Boolean functions and a conditional
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Less,
    And,
    Or,
    Not,
    If,
    X,
}

pub fn primitives() -> Result<PrimitiveSet<Op>> {
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    let boolean = set.new_type("bool");
    set.function("add", Op::Add, [real, real], real)
        .function("sub", Op::Sub, [real, real], real)
        .function("mul", Op::Mul, [real, real], real)
        .function("less", Op::Less, [real, real], boolean)
        .function("and", Op::And, [boolean, boolean], boolean)
        .function("or", Op::Or, [boolean, boolean], boolean)
        .function("not", Op::Not, [boolean], boolean)
        .function("if", Op::If, [boolean, real, real], real)
        .terminal("x", Op::X, real)
        // ephemeral random constants: the integers -2 to 2
        .constants(real, Constants::integers(-2..=2)?);
    set.build(real)
}

// a value of either type; the set's types guarantee which one each primitive gets
#[derive(Clone, Copy, Debug)]
enum Value {
    Real(f64),
    Bool(bool),
}

impl Value {
    fn real(self) -> f64 {
        match self {
            Value::Real(value) => value,
            Value::Bool(_) => unreachable!("a real, by the set's types"),
        }
    }

    fn bool(self) -> bool {
        match self {
            Value::Bool(value) => value,
            Value::Real(_) => unreachable!("a Boolean, by the set's types"),
        }
    }
}

// the value of `op` from its arguments' values, at x
fn apply(op: Op, args: &[Value], x: f64) -> Value {
    let real = |i: usize| args[i].real();
    let bool = |i: usize| args[i].bool();
    match op {
        Op::Add => Value::Real(real(0) + real(1)),
        Op::Sub => Value::Real(real(0) - real(1)),
        Op::Mul => Value::Real(real(0) * real(1)),
        Op::Less => Value::Bool(real(0) < real(1)),
        Op::And => Value::Bool(bool(0) && bool(1)),
        Op::Or => Value::Bool(bool(0) || bool(1)),
        Op::Not => Value::Bool(!bool(0)),
        Op::If => args[if bool(0) { 1 } else { 2 }],
        Op::X => Value::Real(x),
    }
}

// points and |x| at them
pub struct Data {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

impl Data {
    fn new(xs: Vec<f64>) -> Self {
        let ys = xs.iter().map(|x| x.abs()).collect();
        Self { xs, ys }
    }

    // the tree's values at the points
    pub fn predict(&self, set: &PrimitiveSet<Op>, tree: &Tree) -> Vec<f64> {
        let mut stack = Vec::new();
        self.xs
            .iter()
            .map(|&x| {
                let value = tree.evaluate(
                    set,
                    &mut stack,
                    |op, args| apply(op, args, x),
                    |_, constant| Value::Real(constant), // only reals have constants
                );
                value.real()
            })
            .collect()
    }

    // the root mean squared error of the tree
    pub fn rmse(&self, set: &PrimitiveSet<Op>, tree: &Tree) -> f64 {
        let predictions = self.predict(set, tree);
        let mut sum = 0.0;
        for (prediction, y) in predictions.iter().zip(&self.ys) {
            let error = prediction - y;
            sum += error * error;
        }
        (sum / self.ys.len() as f64).sqrt()
    }

    // the standard deviation of the values
    fn deviation(&self) -> f64 {
        let n = self.ys.len() as f64;
        let mean = self.ys.iter().sum::<f64>() / n;
        let squares: f64 = self.ys.iter().map(|y| (y - mean) * (y - mean)).sum();
        (squares / n).sqrt()
    }
}

const POPULATION: usize = 1000;

fn main() -> Result<()> {
    // 20 training points uniform in [-1, 1], from a fixed seed, and 101 test points evenly spaced
    let mut rng = StreamRng::seed_from_u64(20);
    let training = Data::new((0..20).map(|_| rng.random_range(-1.0..=1.0)).collect());
    let test = Data::new((0..=100).map(|i| f64::from(i) / 50.0 - 1.0).collect());
    // exact recovery: an error of at most 1e-10 of the values' standard deviation
    let tolerance = 1e-10 * training.deviation();
    println!("|x| from 20 points in [-1, 1], with real and Boolean types");
    println!("{POPULATION} trees, until the RMSE is at most {tolerance:.2e}\n");

    let gp = Gp::builder(primitives()?).build()?;
    let set = gp.primitives().clone();
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
    let mut trace = trace::Trace::from_env(&training, &set);
    let outcome = Engine::new(ga, |tree: &Tree| training.rmse(&set, tree))
        .stop_when(Stop::target(tolerance).or(Stop::generations(100)))
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
        "RMSE on the 20 training points: {:.2e}",
        training.rmse(&set, best)
    );
    println!(
        "RMSE on the 101 test points:    {:.2e}",
        test.rmse(&set, best)
    );
    println!(
        "\nthe expression, {} nodes of depth {}:\n{}",
        best.len(),
        best.depth(&set),
        best.display(&set)
    );
    trace.write();
    Ok(())
}
