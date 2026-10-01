//! The trace of the run for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the iterates of L-BFGS-B so far on the contour of Rosenbrock's function
//! in the box, a frame per round. The Python example writes the same file.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Value, json};

pub struct Trace {
    path: Option<String>,
    frames: Vec<Value>,
    // the points the run has stood at, in order
    iterates: Vec<[f64; 2]>,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        Self {
            path,
            frames: Vec::new(),
            iterates: Vec::new(),
        }
    }

    // records a round: every point the run has stood at so far, and the best one
    pub fn record(&mut self, snapshot: &Snapshot<'_, Reals>) {
        if self.path.is_none() {
            return;
        }
        let progress = snapshot.progress();
        let current = snapshot.population()[0].genome();
        let point = [current[0], current[1]];
        if self.iterates.last() != Some(&point) {
            self.iterates.push(point);
        }
        let best = snapshot.best().genome();
        self.frames.push(json!({
            "generation": progress.generation(),
            "evaluations": progress.evaluations(),
            "best": progress.best().and_then(Fitness::score),
            "state": {
                "population": self.iterates,
                "best": [best[0], best[1]],
            },
        }));
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let settings = json!({
            "format": 1,
            "example": "lbfgsb_bounds",
            "objective": "minimize",
            "x_label": "rounds",
            "y_label": "best value",
            "log_y": false,
            "optimum": 0.25,
            "plot": "contour",
            "problem": {
                "function": "rosenbrock",
                "bounds": [[-2.0, 0.5], [-1.0, 3.0]],
                "minima": [[0.5, 0.25]],
                "minima_label": "minimum in the box",
                "population_label": "iterates",
            },
        });
        write(&path, settings, self.frames);
    }
}

// ---- the same in every example's trace ---------------------------------------------------------

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
