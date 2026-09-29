//! The trace of the run for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the best board so far and its attacking pairs, in at most 200
//! generations. The Python example writes the same file.

use crate::N;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Value, json};

pub struct Trace {
    path: Option<String>,
    frames: Frames,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        let frames = Frames::new(200);
        Self { path, frames }
    }

    // records a generation: the best board so far and its attacking pairs
    pub fn record(&mut self, snapshot: &Snapshot<'_, Order>) {
        if self.path.is_none() {
            return;
        }
        let best = snapshot.best().genome();
        let state = json!({ "best": &best[..], "attacks": attacks(best) });
        self.frames.push(frame(snapshot, state));
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let settings = json!({
            "format": 1,
            "example": "n_queens_128",
            "objective": "minimize",
            "x_label": "generations",
            "y_label": "conflicts",
            "log_y": false,
            "optimum": 0.0,
            "plot": "board",
            "problem": { "n": N },
        });
        write(&path, settings, self.frames.into_vec());
    }
}

// the pairs of rows whose queens attack each other, on a diagonal
fn attacks(order: &Order) -> Vec<[usize; 2]> {
    let mut pairs = Vec::new();
    for (a, &column_a) in order.iter().enumerate() {
        for (b, &column_b) in order.iter().enumerate().skip(a + 1) {
            if column_a.abs_diff(column_b) == b - a {
                pairs.push([a, b]);
            }
        }
    }
    pairs
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

// the frame of a generation: its progress, the median score of its population and `state`
fn frame<G: Genome>(snapshot: &Snapshot<'_, G>, state: Value) -> Value {
    let progress = snapshot.progress();
    let population = snapshot.population().iter();
    let scores = population.filter_map(|individual| individual.fitness()?.score());
    json!({
        "generation": progress.generation(),
        "evaluations": progress.evaluations(),
        "best": progress.best().and_then(Fitness::score),
        "median": median(scores.collect()),
        "state": state,
    })
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
