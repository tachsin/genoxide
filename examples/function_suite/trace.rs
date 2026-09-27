//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the error of each function and algorithm after every 1,000
//! evaluations. The Python example writes the same file.

use crate::BUDGET;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Map, Value, json};

// the frames, evenly spaced over the budget
const FRAMES: u64 = 100;

pub struct Trace {
    path: Option<String>,
    // per function and algorithm: the evaluations and the error after each generation
    series: Vec<(String, Vec<(u64, f64)>)>,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        Self {
            path,
            series: Vec::new(),
        }
    }

    // the callback that records the run `name`'s error to `minimum` after each generation
    pub fn errors(&mut self, name: String, minimum: f64) -> impl FnMut(&Snapshot<'_, Reals>) {
        let tracing = self.path.is_some();
        self.series.push((name, Vec::new()));
        let (_, history) = self.series.last_mut().expect("a series");
        move |snapshot| {
            let progress = snapshot.progress();
            if let Some(best) = progress.best().and_then(Fitness::score).filter(|_| tracing) {
                history.push((progress.evaluations(), (best - minimum).max(0.0)));
            }
        }
    }

    // writes the trace, if there's one: the runs side by side, a frame per 1,000 evaluations with
    // each run's error after its last generation within them
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let frame = |evaluations: u64| {
            let mut values = Map::new();
            for (name, history) in &self.series {
                let done = history.iter().rev().find(|(done, _)| *done <= evaluations);
                values.insert(name.clone(), json!(done.map(|(_, error)| error)));
            }
            json!({
                "generation": null,
                "evaluations": evaluations,
                "best": null,
                "median": null,
                "state": { "values": values },
            })
        };
        let frames = (1..=FRAMES).map(|k| frame(k * BUDGET / FRAMES)).collect();
        let names: Vec<&String> = self.series.iter().map(|(name, _)| name).collect();
        let settings = json!({
            "format": 1,
            "example": "function_suite",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error to the minimum",
            "log_y": true,
            "optimum": 0.0,
            "plot": "multi-curve",
            "problem": { "series": names },
        });
        write(&path, settings, frames);
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
