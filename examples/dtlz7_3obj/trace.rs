//! The trace of the last run for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the front and its hypervolume, in at most 100 generations, with at
//! most 100 points of the front a frame but the whole front in the last. The Python example writes
//! the same file.

use crate::REFERENCE;
use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{Dtlz7, MultiProblem};
use genoxide::prelude::*;
use serde_json::{Value, json};

pub struct Trace {
    path: Option<String>,
    frames: Frames,
    // the last front recorded, whole
    last: Vec<[f64; 3]>,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        let frames = Frames::new(100);
        let last = Vec::new();
        Self { path, frames, last }
    }

    // records a generation: the front and its hypervolume
    pub fn record(&mut self, snapshot: &MultiSnapshot<'_, Reals, 3>) {
        if self.path.is_none() {
            return;
        }
        let front = snapshot
            .front()
            .iter()
            .filter_map(|x| x.fitness()?.values());
        let front: Vec<[f64; 3]> = front.collect();
        let volume = hypervolume(&front, &REFERENCE, &[Minimize; 3]);
        // every k-th point of a front of more than 100, for a file of the usual size
        let every = front.len().div_ceil(100);
        let shown: Vec<[f64; 3]> = front.iter().step_by(every.max(1)).copied().collect();
        self.frames.push(frame(
            snapshot,
            json!({ "front": shown, "hypervolume": volume }),
        ));
        self.last = front;
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let settings = json!({
            "format": 1,
            "example": "dtlz7_3obj",
            "objective": ["minimize", "minimize", "minimize"],
            "x_label": "generations",
            "y_label": "hypervolume",
            "log_y": false,
            "optimum": 1.7392,
            "plot": "front-3d",
            "problem": {
                "objectives": ["f1", "f2", "f3"],
                // the four regions, as a grid of 32 × 32 values of f₁ and f₂
                "true_front": Dtlz7::<3>::default().optimal_front(1000),
            },
        });
        // the last frame shows the whole front
        let mut frames = self.frames.into_vec();
        if let Some(last) = frames.last_mut() {
            last["state"]["front"] = json!(self.last);
        }
        write(&path, settings, frames);
    }
}

// ---- the same in every example's trace ---------------------------------------------------------

// the frames of at most `most` generations, from the part of the run where what the page plots
// changes: the frames after the last change are left out (a run that reached its target, or a
// front that no longer moves), and the rest are spread evenly over the generations up to it. While
// the run goes, up to 8 × `most` frames are kept: every `every`-th generation, with `every`
// doubling whenever there are that many, and the last one.
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
        if self.kept.len() == 8 * self.most {
            self.every *= 2;
            let every = self.every;
            self.kept.retain(|(generation, _)| generation % every == 0);
        }
    }

    fn into_vec(self) -> Vec<Value> {
        let frames = self.kept.into_iter().chain(self.last);
        let frames: Vec<Value> = frames.map(|(_, frame)| frame).collect();
        let active = &frames[..=last_change(&frames)];
        let (count, most) = (active.len(), self.most.max(2));
        if count <= most {
            return active.to_vec();
        }
        let at = |i: usize| active[(i * (count - 1) + (most - 1) / 2) / (most - 1)].clone();
        (0..most).map(at).collect()
    }
}

// the index of the frame after which nothing the page plots changes. To 3 significant digits, as
// a plot shows them: the best, the median and, for a single objective (a numeric best), the state;
// to within a hundredth of their range over the run: a front's hypervolumes, in the state or in a
// grid's series
fn last_change(frames: &[Value]) -> usize {
    let Some(last) = frames.len().checked_sub(1) else {
        return 0;
    };
    let single = frames.iter().any(|frame| frame["best"].is_number());
    let measures = |frame: &Value| -> Vec<f64> {
        let mut values = Vec::new();
        for value in [&frame["state"]["hypervolume"], &frame["series"]] {
            match value {
                Value::Number(number) => values.extend(number.as_f64()),
                Value::Object(map) => values.extend(map.values().filter_map(Value::as_f64)),
                _ => {}
            }
        }
        values
    };
    let measured: Vec<Vec<f64>> = frames.iter().map(measures).collect();
    let end = &measured[last];
    let tolerance: Vec<f64> = (0..end.len())
        .map(|k| {
            let values = measured.iter().filter_map(|values| values.get(k).copied());
            let (low, high) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), v| {
                (low.min(v), high.max(v))
            });
            (high - low) / 100.0
        })
        .collect();
    let settled = |i: usize| {
        let (frame, final_frame) = (&frames[i], &frames[last]);
        let same =
            |key: &str, flush: bool| coarse(&frame[key], flush) == coarse(&final_frame[key], flush);
        same("best", false)
            && same("median", false)
            && (!single || same("state", true))
            && measured[i].len() == end.len()
            && measured[i]
                .iter()
                .zip(end)
                .zip(&tolerance)
                .all(|((value, end), tolerance)| (value - end).abs() <= *tolerance)
    };
    let mut first = last;
    while first > 0 && settled(first - 1) {
        first -= 1;
    }
    first
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

// `value` with its numbers to 3 significant digits, as precisely as a plot shows them: two frames
// whose plotted values agree to that precision look the same. With `flush`, for the solutions a
// plot draws on their ranges, numbers below 1e-6 in size count as 0
fn coarse(value: &Value, flush: bool) -> String {
    let join = |items: Vec<String>| items.join(",");
    match value {
        Value::Number(number) if number.is_f64() => {
            let number = number.as_f64().expect("f64");
            let number = if flush && number.abs() < 1e-6 {
                0.0
            } else {
                number
            };
            format!("{number:.2e}")
        }
        Value::Array(items) => {
            format!(
                "[{}]",
                join(items.iter().map(|item| coarse(item, flush)).collect())
            )
        }
        Value::Object(map) => {
            let entry = |(key, item): (&String, &Value)| format!("{key}:{}", coarse(item, flush));
            format!("{{{}}}", join(map.iter().map(entry).collect()))
        }
        other => other.to_string(),
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
