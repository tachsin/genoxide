//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: a panel per problem with its front, and each front's normalized
//! hypervolume, in at most 50 generations. A frame's evaluations are the four runs' together.
//! The Python example writes the same file.

use genoxide::multi::MultiSnapshot;
use genoxide::prelude::*;
use serde_json::{Map, Value, json};

// the evaluations, the front and its normalized hypervolume after a generation
type Front = (u64, Vec<[f64; 2]>, f64);

// a problem's run: its name, its optimal front if it's known, and its fronts
struct Run {
    name: String,
    optimal: Option<Vec<[f64; 2]>>,
    history: Vec<Front>,
}

pub struct Trace {
    path: Option<String>,
    runs: Vec<Run>,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        Self {
            path,
            runs: Vec::new(),
        }
    }

    // the callback that records the run of problem `name`, whose optimal front is `optimal`: its
    // front and the front's `volume` after each generation
    pub fn fronts(
        &mut self,
        name: &str,
        optimal: Option<Vec<[f64; 2]>>,
        volume: impl Fn(&[[f64; 2]]) -> f64,
    ) -> impl FnMut(&MultiSnapshot<'_, Reals, 2>) {
        let tracing = self.path.is_some();
        let name = name.to_string();
        let history = Vec::new();
        self.runs.push(Run {
            name,
            optimal,
            history,
        });
        let run = self.runs.last_mut().expect("a run");
        move |snapshot| {
            if tracing {
                let front = snapshot
                    .front()
                    .iter()
                    .filter_map(|x| x.fitness()?.values());
                let front: Vec<[f64; 2]> = front.collect();
                let volume = volume(&front);
                run.history
                    .push((snapshot.progress().evaluations(), front, volume));
            }
        }
    }

    // writes the trace, if there's one: the runs side by side, a frame per generation
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let mut frames = Frames::new(50);
        let generations = self.runs.iter().map(|run| run.history.len()).max();
        for generation in 0..generations.unwrap_or(0) {
            let (mut series, mut panels, mut evaluations) = (Map::new(), Vec::new(), 0);
            for run in &self.runs {
                let at = generation.min(run.history.len() - 1);
                let (done, front, volume) = &run.history[at];
                series.insert(run.name.clone(), json!(volume));
                panels.push(json!({ "fronts": { "NSGA-II": front } }));
                evaluations += done;
            }
            frames.push(json!({
                "generation": generation,
                "evaluations": evaluations,
                "series": series,
                "state": { "panels": panels },
            }));
        }
        let panel = |run: &Run| {
            let mut problem = json!({ "objectives": ["f1", "f2"], "series": ["NSGA-II"] });
            if let Some(optimal) = &run.optimal {
                problem["true_front"] = json!(optimal);
            }
            json!({ "title": run.name, "problem": problem })
        };
        let names: Vec<&str> = self.runs.iter().map(|run| run.name.as_str()).collect();
        let settings = json!({
            "format": 1,
            "example": "classic_fronts",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "normalized hypervolume",
            "log_y": false,
            "optimum": null,
            "plot": "grid",
            "problem": {
                "panel_plot": "front-2d",
                "panels": self.runs.iter().map(panel).collect::<Vec<_>>(),
                "series": names,
            },
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
        if generation % self.every != 0 {
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
