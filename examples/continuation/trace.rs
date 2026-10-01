//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the distance to the global minimum and the stage's σ at every round, of
//! the run through the stages, of σ = 0 from the start, and of the stages with L-BFGS-B's pairs
//! kept. The Python example writes the same file.

use serde_json::{Value, json};

// the lines of the plot: a panel per quantity, a line per run
const RUNS: [&str; 3] = ["stages", "σ = 0 from the start", "pairs kept"];
const QUANTITIES: [&str; 2] = ["distance to the global minimum", "σ"];

pub struct Trace {
    path: Option<String>,
    // per run, per round: the distance to the global minimum and σ
    runs: [Vec<(f64, f64)>; 3],
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        Self {
            path: std::env::var("GENOXIDE_TRACE").ok(),
            runs: Default::default(),
        }
    }

    // records a round of run `run`
    pub fn record(&mut self, run: usize, step: u64, distance: f64, sigma: f64) {
        if self.path.is_some() {
            assert_eq!(self.runs[run].len() as u64, step);
            self.runs[run].push((distance, sigma));
        }
    }

    // writes the trace, if there's one, with at most 200 of its frames
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let steps = self.runs.iter().map(Vec::len).max().unwrap_or(0);
        let mut frames = Frames::new(200);
        for step in 0..steps {
            let mut values = serde_json::Map::new();
            for (run, name) in self.runs.iter().zip(RUNS) {
                let at = run.get(step);
                values.insert(
                    format!("{}/{name}", QUANTITIES[0]),
                    json!(at.map(|&(distance, _)| distance)),
                );
                values.insert(
                    format!("{}/{name}", QUANTITIES[1]),
                    json!(at.map(|&(_, sigma)| sigma)),
                );
            }
            let kept = self.runs[0].get(step).map(|&(distance, _)| distance);
            frames.push(json!({
                "generation": step,
                "evaluations": null,
                "best": kept,
                "state": { "values": values },
            }));
        }
        let series: Vec<String> = QUANTITIES
            .iter()
            .flat_map(|quantity| RUNS.map(|run| format!("{quantity}/{run}")))
            .collect();
        let settings = json!({
            "format": 1,
            "example": "continuation",
            "objective": "minimize",
            "x_label": "rounds",
            "y_label": "distance to the global minimum",
            "log_y": true,
            "optimum": 0.0,
            "plot": "multi-curve",
            "problem": { "series": series },
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
