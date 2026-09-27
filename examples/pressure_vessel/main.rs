//! Pressure vessel design (Sandgren, 1990): the cheapest cylindrical vessel with hemispherical
//! heads that holds 1,296,000 cubic inches, a constrained mixed discrete-continuous problem. The
//! minimum cost is 6059.714335.
//!
//! The variables are the thickness of the shell and of the heads, multiples of 0.0625 inch, and
//! the inner radius and the length of the shell. genoxide's `PressureVessel` rounds the first two
//! genes to whole plates, and its fitness is the cost and the violation of the four constraints,
//! which Deb's feasibility rules compare. SHADE, a differential evolution, searches the genes.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the run's trace for the plot on the example's page:
//! the best design so far and its constraint violations, in at most 200 generations.
//!
//! ```text
//! cargo run --release --example pressure_vessel
//! ```

use genoxide::genome::Genome;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::engineering::PressureVessel;
use serde_json::{Value, json};

fn main() -> Result<()> {
    let problem = PressureVessel;
    let minimum = problem.optimum().expect("known").value();
    let de = De::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    let mut trace = std::env::var("GENOXIDE_TRACE")
        .ok()
        .map(|path| (path, Trace::new(200)));
    let outcome = Engine::new(de, problem)
        .stop_when(Stop::evaluations(50_000))
        .on_generation(|snapshot| {
            if let Some((_, trace)) = &mut trace {
                let best = snapshot.best().genome();
                let constraints = problem.constraints(best);
                let violations: Vec<f64> = constraints
                    .inequalities()
                    .iter()
                    .map(|&g| g.max(0.0))
                    .collect();
                let state = json!({ "best": problem.design(best), "violations": violations });
                trace.record(snapshot, state);
            }
        })
        .run()?;

    let best = outcome.best_fitness();
    let [shell, head, radius, length] = problem.design(outcome.best_genome());
    println!(
        "cost {:.6} after {} evaluations (the minimum: {minimum:.6})",
        best.score().unwrap_or(f64::NAN),
        outcome.evaluations()
    );
    println!("violation {:.6}", best.violation());
    println!("shell {shell:.4}, heads {head:.4}, radius {radius:.6}, length {length:.6}");
    if let Some((path, trace)) = trace {
        let variable = |name, bounds| json!({ "name": name, "unit": "in", "bounds": bounds });
        let (plates, inches) = ([0.0625, 6.1875], [10.0, 200.0]);
        trace.write(
            &path,
            json!({
                "format": 1,
                "example": "pressure_vessel",
                "objective": "minimize",
                "x_label": "evaluations",
                "y_label": "cost",
                "log_y": false,
                "optimum": minimum,
                "plot": "design",
                "problem": {
                    "variables": [
                        variable("Ts", plates),
                        variable("Th", plates),
                        variable("R", inches),
                        variable("L", inches),
                    ],
                    "constraints": ["g1", "g2", "g3", "g4"],
                },
            }),
        );
    }
    Ok(())
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
