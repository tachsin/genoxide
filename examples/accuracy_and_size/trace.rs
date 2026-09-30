//! The trace of the run for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the front of each generation, in at most 100 generations, as the
//! logarithm of the RMSE (base 10, from the error at the level of rounding, 10⁻¹⁶, up) against the
//! size, and its hypervolume in those terms, up to a log RMSE of 1 and a size of 1025 (the limit
//! of 1024 nodes and one more).

use genoxide::Objective::Minimize;
use genoxide::gp::Tree;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::hypervolume;
use genoxide::prelude::*;
use serde_json::{Value, json};

// the reference point of the hypervolume
const REFERENCE: [f64; 2] = [1.0, 1025.0];

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

    // records a generation: the front, as the log RMSE and the size, and its hypervolume
    pub fn record(&mut self, snapshot: &MultiSnapshot<'_, Tree, 2>) {
        if self.path.is_none() {
            return;
        }
        let front = snapshot
            .front()
            .iter()
            .filter_map(|tree| tree.fitness()?.values())
            .map(|[error, size]| [error.max(1e-16).log10(), size]);
        let front: Vec<[f64; 2]> = front.collect();
        let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
        let state = json!({ "fronts": { "NSGA-II": front }, "hypervolume": { "NSGA-II": volume } });
        self.frames.push(frame(snapshot, state));
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let settings = json!({
            "format": 1,
            "example": "accuracy_and_size",
            "objective": ["minimize", "minimize"],
            "x_label": "evaluations",
            "y_label": "hypervolume",
            "log_y": false,
            "optimum": null,
            "plot": "front-2d",
            "problem": {
                "objectives": ["log10 RMSE", "size"],
                "series": ["NSGA-II"],
            },
        });
        write(&path, settings, self.frames.into_vec());
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

// the frame of a generation: its progress and `state`, whose hypervolume is the curve
fn frame<G: Genome, const M: usize>(snapshot: &MultiSnapshot<'_, G, M>, state: Value) -> Value {
    let progress = snapshot.progress();
    json!({
        "generation": progress.generation(),
        "evaluations": progress.evaluations(),
        "best": null,
        "median": null,
        "state": state,
    })
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
