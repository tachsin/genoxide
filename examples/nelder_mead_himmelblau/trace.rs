//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the simplex on the contour of Himmelblau's function, a frame per
//! round, with the points where the runs so far converged. The Python example writes the same
//! file.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{Himmelblau, Problem};
use serde_json::{Value, json};

pub struct Trace {
    path: Option<String>,
    frames: Vec<Value>,
    // where the runs so far converged
    ends: Vec<[f64; 2]>,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        Self {
            path,
            frames: Vec::new(),
            ends: Vec::new(),
        }
    }

    // records a round: the vertices of the simplex, best first, the best point so far, and where
    // the runs before converged
    pub fn record(&mut self, snapshot: &Snapshot<'_, Reals>) {
        if self.path.is_none() {
            return;
        }
        let progress = snapshot.progress();
        let vertices: Vec<[f64; 2]> = snapshot
            .population()
            .iter()
            .map(|vertex| [vertex.genome()[0], vertex.genome()[1]])
            .collect();
        let values = snapshot.population().iter();
        let values = values.filter_map(|vertex| vertex.fitness().and_then(Fitness::score));
        let best = snapshot.best().genome();
        self.frames.push(json!({
            "generation": progress.generation(),
            "evaluations": progress.evaluations(),
            "best": progress.best().and_then(Fitness::score),
            "median": median(values.collect()),
            "state": {
                "population": vertices,
                "simplex": vertices,
                "best": [best[0], best[1]],
                "ends": self.ends,
            },
        }));
    }

    // records where a run converged, shown from the next round on
    pub fn end(&mut self, point: &Reals) {
        self.ends.push([point[0], point[1]]);
    }

    // writes the trace, if there's one, with at most 300 of its frames
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let mut frames = Frames::new(300);
        for frame in self.frames {
            frames.push(frame);
        }
        let optimum = Himmelblau.optimum().expect("known");
        let minima: Vec<&[f64]> = optimum.solutions().iter().map(|x| &x[..]).collect();
        let settings = json!({
            "format": 1,
            "example": "nelder_mead_himmelblau",
            "objective": "minimize",
            "x_label": "rounds",
            "y_label": "best value",
            "log_y": true,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "himmelblau",
                "bounds": [[-5.0, 5.0], [-5.0, 5.0]],
                "minima": minima,
                "population_label": "simplex",
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
