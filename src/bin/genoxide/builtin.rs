//! `genoxide fitness <name>`: test functions that speak the fitness protocol, to try a run file
//! without writing a program, and as examples of the protocol.

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
                        .map(|xi| xi * xi - 10.0 * (TAU * xi).cos())
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
                        100.0 * (pair[1] - pair[0] * pair[0]).powi(2) + (1.0 - pair[0]).powi(2)
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
            let cosines = x.iter().map(|xi| (TAU * xi).cos()).sum::<f64>() / n;
            vec![-20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + 20.0 + E]
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
            let g = if x.len() > 1 {
                1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64
            } else {
                1.0
            };
            vec![x[0], g * (1.0 - (x[0] / g).sqrt())]
        },
    },
    Function {
        name: "schaffer",
        description: "real: Schaffer's two objectives, x² and (x − 2)² of the first gene (minimize both)",
        score: |x| vec![x[0] * x[0], (x[0] - 2.0) * (x[0] - 2.0)],
    },
];

/// Runs a built-in function on stdin until it closes.
pub fn serve(name: &str) -> Result<(), String> {
    let function = FUNCTIONS
        .iter()
        .find(|function| function.name == name)
        .ok_or_else(|| format!("no built-in fitness `{name}`; {}", list()))?;
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
        let values: Vec<String> = (function.score)(&genes)
            .iter()
            .map(f64::to_string)
            .collect();
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
