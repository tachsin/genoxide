//! `genoxide fitness <name>`: test functions that speak the fitness protocol, to try a run file
//! without writing a program, and as examples of the protocol.

use genoxide::engine::{Extras, FitnessFunction};
use genoxide::genome::Reals;
use genoxide::math;
use genoxide::multi::MultiFitnessFunction;
use genoxide::multi::problems::{Schaffer1, Zdt1};
use genoxide::problems::{Ackley, Rastrigin, Rosenbrock, Sphere};
use std::f64::consts::{E, TAU};
use std::io::{BufRead, Write};

/// A built-in function.
pub struct Function {
    pub name: &'static str,
    /// What it scores, for the list.
    pub description: &'static str,
    /// The objective values of the genes.
    pub score: fn(&[f64]) -> Vec<f64>,
}

/// The built-in functions.
pub const FUNCTIONS: &[Function] = &[
    Function {
        name: "one-max",
        description: "binary: the number of ones (maximize)",
        score: |x| vec![x.iter().sum()],
    },
    Function {
        name: "sphere",
        description: "real or integer: the sum of squares (minimize, 0 at the origin)",
        score: |x| vec![x.iter().map(|xi| xi * xi).sum()],
    },
    Function {
        name: "rastrigin",
        description: "real: Rastrigin's function (minimize, 0 at the origin)",
        score: |x| {
            vec![
                10.0 * x.len() as f64
                    + x.iter()
                        .map(|xi| xi * xi - 10.0 * math::cos(TAU * xi))
                        .sum::<f64>(),
            ]
        },
    },
    Function {
        name: "rosenbrock",
        description: "real: Rosenbrock's function (minimize, 0 at all ones)",
        score: |x| {
            vec![
                x.windows(2)
                    .map(|pair| {
                        100.0 * math::powi(pair[1] - pair[0] * pair[0], 2)
                            + math::powi(1.0 - pair[0], 2)
                    })
                    .sum(),
            ]
        },
    },
    Function {
        name: "ackley",
        description: "real: Ackley's function (minimize, 0 at the origin)",
        score: |x| {
            let n = x.len() as f64;
            let squares = x.iter().map(|xi| xi * xi).sum::<f64>() / n;
            let cosines = x.iter().map(|xi| math::cos(TAU * xi)).sum::<f64>() / n;
            vec![-20.0 * math::exp(-0.2 * squares.sqrt()) - math::exp(cosines) + 20.0 + E]
        },
    },
    Function {
        name: "inversions",
        description: "permutation: the number of pairs out of order (minimize, 0 when sorted)",
        score: |x| {
            let mut count = 0;
            for i in 0..x.len() {
                for j in i + 1..x.len() {
                    count += usize::from(x[i] > x[j]);
                }
            }
            vec![count as f64]
        },
    },
    Function {
        name: "zdt1",
        description: "real in [0, 1]: ZDT1's two objectives (minimize both)",
        score: |x| {
            if x.len() > 1 {
                Zdt1::new(x.len())
                    .evaluate(&Reals::from(x.to_vec()))
                    .to_vec()
            } else {
                // g = 1 without the other variables
                vec![x[0], 1.0 - x[0].sqrt()]
            }
        },
    },
    Function {
        name: "schaffer",
        description: "real: Schaffer's two objectives, x² and (x − 2)² of the first gene (minimize both)",
        score: |x| Schaffer1.evaluate(&Reals::from(vec![x[0]])).to_vec(),
    },
];

/// A function's value, its gradient written into the second argument.
pub type WithGradient = fn(&[f64], &mut [f64]) -> f64;

/// The value and gradient of a built-in function with the gradient protocol: genoxide's test
/// problem of the same name, with its analytic gradient, for the real-valued smooth ones.
pub fn gradient_of(name: &str) -> Option<WithGradient> {
    fn with<F: FitnessFunction<Reals, Output = f64>>(
        problem: F,
        x: &[f64],
        gradient: &mut [f64],
    ) -> f64 {
        problem.evaluate_with(
            &Reals::from(x.to_vec()),
            &mut Extras::with_gradient(gradient),
        )
    }
    match name {
        "sphere" => Some(|x, gradient| with(Sphere::new(x.len()), x, gradient)),
        "rastrigin" => Some(|x, gradient| with(Rastrigin::new(x.len()), x, gradient)),
        "rosenbrock" => Some(|x, gradient| with(Rosenbrock::new(x.len()), x, gradient)),
        "ackley" => Some(|x, gradient| with(Ackley::new(x.len()), x, gradient)),
        _ => None,
    }
}

/// Runs a built-in function on stdin until it closes; with `gradient`, writing the value and
/// then the gradient (the gradient protocol, `fitness.gradient`).
pub fn serve(name: &str, gradient: bool) -> Result<(), String> {
    let function = FUNCTIONS
        .iter()
        .find(|function| function.name == name)
        .ok_or_else(|| format!("no built-in fitness `{name}`; {}", list()))?;
    let with_gradient = if gradient {
        Some(gradient_of(name).ok_or_else(|| {
            format!(
                "the built-in fitness `{name}` has no gradient; sphere, rastrigin, rosenbrock and \
                 ackley do"
            )
        })?)
    } else {
        None
    };
    let mut derivatives = Vec::new();
    let stdin = std::io::stdin().lock();
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    let mut genes = Vec::new();
    for line in stdin.lines() {
        let line = line.map_err(|error| error.to_string())?;
        genes.clear();
        for word in line.split_whitespace() {
            genes.push(
                word.parse::<f64>()
                    .map_err(|_| format!("`{word}` isn't a number"))?,
            );
        }
        if genes.is_empty() {
            return Err("an empty genome".to_string());
        }
        let values: Vec<String> = match with_gradient {
            Some(evaluate) => {
                derivatives.clear();
                derivatives.resize(genes.len(), 0.0);
                let value = evaluate(&genes, &mut derivatives);
                std::iter::once(value)
                    .chain(derivatives.iter().copied())
                    .map(|number| format!("{number:?}"))
                    .collect()
            }
            None => (function.score)(&genes)
                .iter()
                .map(f64::to_string)
                .collect(),
        };
        writeln!(stdout, "{}", values.join(" "))
            .and_then(|()| stdout.flush())
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// The built-in functions, for help and errors.
pub fn list() -> String {
    let names: Vec<&str> = FUNCTIONS.iter().map(|function| function.name).collect();
    format!("the built-in ones are {}", names.join(", "))
}
