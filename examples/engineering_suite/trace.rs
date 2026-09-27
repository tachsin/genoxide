//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: each problem's error f - f* of the best feasible solution, every 1,000
//! evaluations up to 50,000, then every 9,000 up to 500,000. The Python example writes the same
//! file.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Map, Value, json};

// the evaluations of the frames: every 1,000 up to 50,000, where the engineering designs' runs
// end, then every 9,000 up to the CEC 2006 budget of 500,000
fn frame_evaluations() -> Vec<u64> {
    let early = (1..=50).map(|k| k * 1_000);
    let late = (1..=50).map(|k| 50_000 + k * 9_000);
    early.chain(late).collect()
}

// a run's evaluations and error after each generation, None while no solution is feasible
type History = Vec<(u64, Option<f64>)>;

pub struct Trace {
    path: Option<String>,
    // per problem: its run's history
    series: Vec<(String, History)>,
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

    // the callback that records the run `name`'s error to `best_known` after each generation
    pub fn errors(&mut self, name: String, best_known: f64) -> impl FnMut(&Snapshot<'_, Reals>) {
        let tracing = self.path.is_some();
        self.series.push((name, Vec::new()));
        let (_, history) = self.series.last_mut().expect("a series");
        move |snapshot| {
            if !tracing {
                return;
            }
            let progress = snapshot.progress();
            // under Deb's rules, the best is feasible once any solution is; a solution below a
            // best known value counts as an error of 0
            let error = progress
                .best()
                .filter(|best| best.is_feasible())
                .and_then(Fitness::score)
                .map(|best| (best - best_known).max(0.0));
            history.push((progress.evaluations(), error));
        }
    }

    // writes the trace, if there's one: the runs side by side, each run's error after its last
    // generation within a frame's evaluations, and null from the frame after the run's end
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let evaluations = frame_evaluations();
        let frame = |k: usize| {
            let previous = if k == 0 { 0 } else { evaluations[k - 1] };
            let mut values = Map::new();
            for (name, history) in &self.series {
                let end = history.last().map_or(0, |(done, _)| *done);
                let done = history
                    .iter()
                    .rev()
                    .find(|(done, _)| *done <= evaluations[k]);
                let error = done
                    .and_then(|(_, error)| *error)
                    .filter(|_| previous < end);
                values.insert(name.clone(), json!(error));
            }
            json!({
                "generation": null,
                "evaluations": evaluations[k],
                "best": null,
                "median": null,
                "state": { "values": values },
            })
        };
        let frames = (0..evaluations.len()).map(frame).collect();
        let names: Vec<&String> = self.series.iter().map(|(name, _)| name).collect();
        let settings = json!({
            "format": 1,
            "example": "engineering_suite",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error f - f* of the best feasible solution",
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
