//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: both fronts and their hypervolumes, in at most 64 generations. A
//! frame's evaluations are the two runs' together. The Python example writes the same file.

use crate::REFERENCE;
use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::hypervolume;
use genoxide::prelude::*;
use serde_json::{Map, Value, json};

// the evaluations, the front and its hypervolume after a generation
type Front = (u64, Vec<[f64; 2]>, f64);

pub struct Trace {
    path: Option<String>,
    // per algorithm, its front after each generation
    series: Vec<(&'static str, Vec<Front>)>,
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

    // the callback that records the run `name`'s front and its hypervolume after each generation
    pub fn fronts(&mut self, name: &'static str) -> impl FnMut(&MultiSnapshot<'_, Reals, 2>) {
        let tracing = self.path.is_some();
        self.series.push((name, Vec::new()));
        let (_, history) = self.series.last_mut().expect("a series");
        move |snapshot| {
            if tracing {
                let front = snapshot
                    .front()
                    .iter()
                    .filter_map(|x| x.fitness()?.values());
                let front: Vec<[f64; 2]> = front.collect();
                let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
                history.push((snapshot.progress().evaluations(), front, volume));
            }
        }
    }

    // writes the trace, if there's one: the runs side by side, a frame per generation
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let mut frames = Frames::new(64);
        let generations = self.series.iter().map(|(_, history)| history.len()).max();
        for generation in 0..generations.unwrap_or(0) {
            let (mut fronts, mut volumes, mut evaluations) = (Map::new(), Map::new(), 0);
            for (name, history) in &self.series {
                let (done, front, volume) = &history[generation.min(history.len() - 1)];
                fronts.insert(name.to_string(), json!(front));
                volumes.insert(name.to_string(), json!(volume));
                evaluations += done;
            }
            frames.push(json!({
                "generation": generation,
                "evaluations": evaluations,
                "best": null,
                "median": null,
                "state": { "fronts": fronts, "hypervolume": volumes },
            }));
        }
        let names: Vec<&str> = self.series.iter().map(|(name, _)| *name).collect();
        let settings = json!({
            "format": 1,
            "example": "kursawe",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "hypervolume",
            "log_y": false,
            "optimum": null,
            "plot": "front-2d",
            "problem": { "objectives": ["f1", "f2"], "true_front": null, "series": names },
        });
        write(&path, settings, frames.into_vec());
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
