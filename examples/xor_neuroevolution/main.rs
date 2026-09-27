//! XOR neuroevolution: evolve the 9 weights of a 2-2-1 neural network until it computes XOR.
//!
//! XOR isn't linearly separable, so the network needs its hidden layer: two sigmoid units, each
//! with a weight per input and a bias, and a sigmoid output unit with a weight per hidden unit
//! and a bias. The fitness is the sum of the squared errors over the four input pairs, minimized
//! by CMA-ES with IPOP restarts, which escape the flat regions where the network outputs 0.5 or
//! solves three of the four cases.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the run's trace for the plot on the example's page:
//! the best network's output over [0, 1]², on a grid of 21 × 21 points, in at most 64 generations.
//!
//! ```text
//! cargo run --release --example xor_neuroevolution
//! ```

use genoxide::genome::Genome;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Value, json};

// the inputs and the expected output
const CASES: [([f64; 2], f64); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];
// the points per side of the trace's grid
const GRID: usize = 21;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// the network's output: w[0..3] and w[3..6] are the hidden units' weights and biases, w[6..9] the
// output unit's
fn output(w: &[f64], [a, b]: [f64; 2]) -> f64 {
    let hidden_1 = sigmoid(w[0] * a + w[1] * b + w[2]);
    let hidden_2 = sigmoid(w[3] * a + w[4] * b + w[5]);
    sigmoid(w[6] * hidden_1 + w[7] * hidden_2 + w[8])
}

fn squared_error(w: &Reals) -> f64 {
    let mut error = 0.0;
    for (input, expected) in CASES {
        let difference = output(w, input) - expected;
        error += difference * difference;
    }
    error
}

fn main() -> Result<()> {
    let cmaes = Cmaes::builder(Real::uniform(9, -10.0..=10.0)?)
        .restarts(cmaes::Restarts::Ipop)
        .minimize()
        .seed(1)
        .build()?;
    let mut trace = std::env::var("GENOXIDE_TRACE")
        .ok()
        .map(|path| (path, Trace::new(64)));
    let outcome = Engine::new(cmaes, squared_error)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(20_000)))
        .on_generation(|snapshot| {
            if let Some((_, trace)) = &mut trace {
                let state = json!({ "grid": surface(snapshot.best().genome()) });
                trace.record(snapshot, state);
            }
        })
        .run()?;

    println!(
        "squared error {:.6} after {} evaluations",
        outcome.best_fitness(),
        outcome.evaluations()
    );
    for ([a, b], expected) in CASES {
        let value = output(outcome.best_genome(), [a, b]);
        println!("{a:.0} xor {b:.0} = {expected:.0}: {value:.3}");
    }
    if let Some((path, trace)) = trace {
        trace.write(
            &path,
            json!({
                "format": 1,
                "example": "xor_neuroevolution",
                "objective": "minimize",
                "x_label": "evaluations",
                "y_label": "squared error",
                "log_y": false,
                "optimum": 0.0,
                "plot": "surface",
                "problem": {
                    "inputs": CASES.map(|(input, _)| input),
                    "targets": CASES.map(|(_, target)| target),
                    "grid": GRID,
                },
            }),
        );
    }
    Ok(())
}

// the network's output over [0, 1]²: a row per value of the second input, from 0 to 1, and a
// column per value of the first
fn surface(w: &[f64]) -> Vec<Vec<f64>> {
    let point = |i: usize| i as f64 / (GRID - 1) as f64;
    let row = |b: f64| (0..GRID).map(|a| output(w, [point(a), b])).collect();
    (0..GRID).map(|b| row(point(b))).collect()
}

// ---- the trace of the run, for the plot on the example's page ----------------------------------

// a frame per recorded generation, at most `most`: every `every`-th generation, with `every`
// doubling whenever there are `most`, and the last generation
struct Trace {
    most: usize,
    every: u64,
    frames: Vec<(u64, Value)>,
    last: Option<(u64, Value)>,
}

impl Trace {
    fn new(most: usize) -> Self {
        let (every, frames, last) = (1, Vec::new(), None);
        Self {
            most,
            every,
            frames,
            last,
        }
    }

    // the generation's progress, the median score of its population and the plot's `state`
    fn record<G: Genome>(&mut self, snapshot: &Snapshot<'_, G>, state: Value) {
        let progress = snapshot.progress();
        let population = snapshot.population().iter();
        let scores = population.filter_map(|individual| individual.fitness()?.score());
        let frame = json!({
            "generation": progress.generation(),
            "evaluations": progress.evaluations(),
            "best": progress.best().and_then(Fitness::score),
            "median": median(scores.collect()),
            "state": state,
        });
        self.push(progress.generation(), frame);
    }

    // keeps `frame` if it's of the `every`-th generation, or as the last one
    fn push(&mut self, generation: u64, frame: Value) {
        if generation % self.every != 0 {
            self.last = Some((generation, frame));
            return;
        }
        self.frames.push((generation, frame));
        self.last = None;
        if self.frames.len() == self.most {
            self.every *= 2;
            let every = self.every;
            self.frames
                .retain(|(generation, _)| generation % every == 0);
        }
    }

    // writes the settings and the frames to `path`, a frame per line
    fn write(self, path: &str, settings: Value) {
        let frames = self.frames.iter().chain(&self.last);
        let frames: Vec<String> = frames.map(|(_, frame)| to_json(frame)).collect();
        let settings = to_json(&settings);
        let head = &settings[..settings.len() - 1];
        let text = format!("{head},\"frames\":[\n{}\n]}}\n", frames.join(",\n"));
        std::fs::write(path, text).expect("the trace is written");
    }
}

// the median of the scores, None without any
fn median(mut scores: Vec<f64>) -> Option<f64> {
    scores.sort_by(f64::total_cmp);
    let middle = scores.len() / 2;
    match scores.len() {
        0 => None,
        n if n % 2 == 1 => Some(scores[middle]),
        _ => Some((scores[middle - 1] + scores[middle]) / 2.0),
    }
}

// compact JSON with sorted keys, and numbers rounded to 6 significant digits and written as
// Python writes them (7542.0, 1e-08): the Python example writes the same file
fn to_json(value: &Value) -> String {
    let join = |items: Vec<String>| items.join(",");
    match value {
        Value::Number(number) if number.is_f64() => python_float(number.as_f64().expect("f64")),
        Value::Array(items) => format!("[{}]", join(items.iter().map(to_json).collect())),
        Value::Object(map) => {
            let entry =
                |(key, item): (&String, &Value)| format!("{}:{}", json!(key), to_json(item));
            format!("{{{}}}", join(map.iter().map(entry).collect()))
        }
        other => other.to_string(),
    }
}

fn python_float(value: f64) -> String {
    let rounded: f64 = format!("{value:.5e}").parse().expect("a number");
    let shortest = format!("{rounded:e}");
    let (mantissa, exponent) = shortest.split_once('e').expect("an exponent");
    let exponent: i32 = exponent.parse().expect("an exponent");
    if (-4..16).contains(&exponent) {
        let text = rounded.to_string();
        if text.contains('.') {
            text
        } else {
            text + ".0"
        }
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    }
}
