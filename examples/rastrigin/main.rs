//! Rastrigin: minimize a real-valued function with many local minima, in 30 dimensions.
//!
//! Compares CMA-ES with IPOP restarts (a population that doubles at each restart) and L-SHADE
//! (differential evolution with a population that shrinks over the budget). The global minimum is
//! 0, at the origin. The function is genoxide's `problems::Rastrigin`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace for the plot on the example's page, which
//! draws the population on the function's contour: that needs two dimensions, so the trace is of
//! a separate run of L-SHADE in 2 dimensions, made only then.
//!
//! ```text
//! cargo run --release --example rastrigin
//! ```

use genoxide::genome::Genome;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{Problem, Rastrigin};
use serde_json::{Value, json};

const DIMENSIONS: usize = 30;
const BUDGET: u64 = 1_000_000;

fn main() -> Result<()> {
    let problem = Rastrigin::new(DIMENSIONS);
    let target = problem.optimum().expect("known").value() + 1e-8;
    let stop = || Stop::target(target).or(Stop::evaluations(BUDGET));

    let cmaes = Cmaes::builder(problem.representation())
        .restarts(cmaes::Restarts::Ipop)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(cmaes, problem).stop_when(stop()).run()?;
    report("CMA-ES", &outcome);

    let l_shade = De::l_shade(problem.representation(), BUDGET)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(l_shade, problem).stop_when(stop()).run()?;
    report("L-SHADE", &outcome);

    if let Ok(path) = std::env::var("GENOXIDE_TRACE") {
        trace_2d(&path)?;
    }
    Ok(())
}

// L-SHADE in 2 dimensions, with a budget of 10,000 evaluations per dimension, recorded for the
// plot of the population on the contour
fn trace_2d(path: &str) -> Result<()> {
    let problem = Rastrigin::new(2);
    let budget = 20_000;
    let l_shade = De::l_shade(problem.representation(), budget)
        .minimize()
        .seed(1)
        .build()?;
    let mut trace = Trace::new(200);
    Engine::new(l_shade, problem)
        .stop_when(Stop::target(1e-8).or(Stop::evaluations(budget)))
        .on_generation(|snapshot| {
            let population: Vec<&[f64]> = snapshot
                .population()
                .iter()
                .map(|x| &x.genome()[..])
                .collect();
            let best = &snapshot.best().genome()[..];
            trace.record(snapshot, json!({ "population": population, "best": best }));
        })
        .run()?;
    trace.write(
        path,
        json!({
            "format": 1,
            "example": "rastrigin",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error",
            "log_y": true,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "rastrigin",
                "bounds": [[-5.12, 5.12], [-5.12, 5.12]],
                "minima": [[0.0, 0.0]],
            },
        }),
    );
    Ok(())
}

fn report(name: &str, outcome: &Outcome<Reals>) {
    println!(
        "{name}: {:.6} after {} evaluations",
        outcome.best_fitness(),
        outcome.evaluations()
    );
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
