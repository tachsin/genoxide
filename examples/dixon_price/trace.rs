//! The trace of a run of the particle swarm with a ring topology in 10 dimensions for the plot on
//! the example's page, written to the file that `GENOXIDE_TRACE` names: the best point so far, each
//! gene on its range with the minimum's value marked, and the error of the best and of the
//! population's median to the minimum, in at most 100 generations. The Python example writes the
//! same file.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{DixonPrice, Problem};
use serde_json::{Value, json};

// the dimensions of the run
const DIMENSIONS: usize = 10;

pub struct Trace {
    path: Option<String>,
    frames: Frames,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        let frames = Frames::new(100);
        Self { path, frames }
    }

    // records a generation: the errors of the best so far and of the population's median, and
    // the best point so far
    pub fn record(&mut self, snapshot: &Snapshot<'_, Reals>) {
        if self.path.is_none() {
            return;
        }
        let progress = snapshot.progress();
        let best = snapshot.best();
        let mut errors: Vec<f64> = snapshot
            .population()
            .iter()
            .filter_map(|individual| individual.fitness().and_then(Fitness::score))
            .map(error)
            .collect();
        errors.sort_by(f64::total_cmp);
        self.frames.push(json!({
            "generation": progress.generation(),
            "evaluations": progress.evaluations(),
            "best": best.fitness().and_then(Fitness::score).map(error),
            "median": median(&errors),
            "state": { "best": &best.genome()[..] },
        }));
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let optimum = DixonPrice::new(DIMENSIONS).optimum().expect("known");
        let minimum = &optimum.solutions()[0];
        let variables: Vec<Value> = (1..)
            .zip(DixonPrice::new(DIMENSIONS).representation().bounds())
            .zip(minimum.iter())
            .map(|((i, bounds), &optimum)| {
                let bounds = [*bounds.start(), *bounds.end()];
                json!({ "name": format!("x{i}"), "unit": "", "bounds": bounds, "optimum": optimum })
            })
            .collect();
        let settings = json!({
            "format": 1,
            "example": "dixon_price",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error to the minimum",
            "log_y": true,
            "optimum": 0.0,
            "plot": "design",
            "problem": {
                "variables": variables,
                "constraints": [],
                "optimum_label": "minimum",
            },
        });
        write(&path, settings, self.frames.into_vec());
    }
}

// the error to the minimum: rounding can put a solution a few ulps below it
fn error(value: f64) -> f64 {
    let minimum = DixonPrice::new(DIMENSIONS)
        .optimum()
        .expect("known")
        .value();
    (value - minimum).max(0.0)
}

// the median of sorted errors, None without any
fn median(errors: &[f64]) -> Option<f64> {
    let middle = errors.len() / 2;
    match errors.len() {
        0 => None,
        n if n % 2 == 1 => Some(errors[middle]),
        _ => Some((errors[middle - 1] + errors[middle]) / 2.0),
    }
}

// ---- the same in every example's trace ---------------------------------------------------------

// the frames of at most `most` generations: every `every`-th one, with `every` doubling whenever
// there are `most`, and the last one
struct Frames {
    most: usize,
    every: u64,
    kept: Vec<(u64, Value)>,
    last: Option<(u64, Value)>,
}

impl Frames {
    fn new(most: usize) -> Self {
        let (every, kept, last) = (1, Vec::new(), None);
        Self {
            most,
            every,
            kept,
            last,
        }
    }

    fn push(&mut self, frame: Value) {
        let generation = frame["generation"].as_u64().expect("a generation");
        if !generation.is_multiple_of(self.every) {
            self.last = Some((generation, frame));
            return;
        }
        self.kept.push((generation, frame));
        self.last = None;
        if self.kept.len() == self.most {
            self.every *= 2;
            let every = self.every;
            self.kept.retain(|(generation, _)| generation % every == 0);
        }
    }

    fn into_vec(self) -> Vec<Value> {
        let frames = self.kept.into_iter().chain(self.last);
        frames.map(|(_, frame)| frame).collect()
    }
}

// writes the settings and the frames to `path`, a frame per line
fn write(path: &str, settings: Value, frames: Vec<Value>) {
    let frames: Vec<String> = frames.iter().map(to_json).collect();
    let settings = to_json(&settings);
    let head = &settings[..settings.len() - 1];
    let text = format!("{head},\"frames\":[\n{}\n]}}\n", frames.join(",\n"));
    std::fs::write(path, text).expect("the trace is written");
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
