//! The trace of the run for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: a frame per step, with the points evaluated so far and, from the first
//! point the model chose, the model that chose it on a grid of 25 × 25 points: its posterior mean,
//! shaded on a log scale as the page shades the function, and the log expected improvement, the
//! 25 nats below its highest value shaded. Both are rounded to thousandths of their range, which
//! is all the page draws. A last frame shows the polished point. The Python example writes the
//! same file.

use genoxide::engine::Progress;
use genoxide::math::ln_1p;
use genoxide::prelude::*;
use genoxide::problems::{Branin, Problem};
use serde_json::{Value, json};

// the points per side of the grid, and the span of the acquisition that is shaded, in nats
const GRID: usize = 25;
const SPAN: f64 = 25.0;

pub struct Trace {
    path: Option<String>,
    frames: Vec<Value>,
    minimum: f64,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        let minimum = Branin.optimum().expect("known").value();
        Self {
            path,
            frames: Vec::new(),
            minimum,
        }
    }

    // records a step: the points so far, the newest one, the best, and the model that chose the
    // newest on the grid
    pub fn record(&mut self, bo: &Bo, progress: &Progress) {
        if self.path.is_none() {
            return;
        }
        let points: Vec<[f64; 2]> = bo
            .population()
            .iter()
            .map(|individual| [individual.genome()[0], individual.genome()[1]])
            .collect();
        let best = bo.best().expect("evaluated").genome();
        let mut state = json!({
            "population": points,
            "newest": points[points.len() - 1],
            "best": [best[0], best[1]],
        });
        if let Some(model) = bo.model() {
            let mean = grid(|x| model.predict(x).mean());
            let acquisition = grid(|x| bo.acquisition_at(x).expect("a model"));
            state["mean"] = json!(shaded(&mean, |lo, hi, v| ln_1p(v - lo) / ln_1p(hi - lo)));
            state["acquisition"] = json!(shaded(&acquisition, |lo, hi, v| {
                let lo = lo.max(hi - SPAN);
                ((v - lo) / (hi - lo)).max(0.0)
            }));
        }
        let error = progress.best().and_then(Fitness::score).expect("valid") - self.minimum;
        self.frames.push(json!({
            "generation": progress.generation(),
            "evaluations": progress.evaluations(),
            "best": error,
            "state": state,
        }));
    }

    // writes the trace, if there's one, with a last frame for the polished point `x` and its value
    pub fn write(mut self, x: &[f64], value: f64) {
        let Some(path) = self.path.take() else { return };
        let mut last = self.frames.last().expect("a frame").clone();
        let generation = last["generation"].as_u64().expect("a generation") + 1;
        let evaluations = last["evaluations"].as_u64().expect("evaluations") + 1;
        let best = last["best"]
            .as_f64()
            .expect("a best")
            .min(value - self.minimum);
        last["generation"] = json!(generation);
        last["evaluations"] = json!(evaluations);
        last["best"] = json!(best);
        last["state"]["polished"] = json!([x[0], x[1]]);
        self.frames.push(last);
        let optimum = Branin.optimum().expect("known");
        let minima: Vec<&[f64]> = optimum.solutions().iter().map(|x| &x[..]).collect();
        let settings = json!({
            "format": 1,
            "example": "bayesian_optimization",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error to the global minimum",
            "log_y": true,
            "optimum": 0.0,
            "plot": "surrogate",
            "problem": {
                "function": "branin",
                "bounds": [[-5.0, 10.0], [0.0, 15.0]],
                "minima": minima,
                "grid": GRID,
            },
        });
        write(&path, settings, self.frames);
    }
}

// `f` on the grid over Branin's box: a row per value of x₂, from 0 to 15, and a column per value
// of x₁, from −5 to 10
fn grid(f: impl Fn(&[f64]) -> f64) -> Vec<Vec<f64>> {
    let at = |low: f64, high: f64, i: usize| low + (high - low) * i as f64 / (GRID - 1) as f64;
    (0..GRID)
        .map(|row| {
            (0..GRID)
                .map(|column| f(&[at(-5.0, 10.0, column), at(0.0, 15.0, row)]))
                .collect()
        })
        .collect()
}

// the grid's values as thousandths of their range: `shade(lo, hi, v)` in [0, 1], rounded
fn shaded(grid: &[Vec<f64>], shade: impl Fn(f64, f64, f64) -> f64) -> Vec<Vec<u64>> {
    let values = grid.iter().flatten();
    let lo = values.clone().fold(f64::INFINITY, |a, &b| a.min(b));
    let hi = values.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|&v| (1000.0 * shade(lo, hi, v) + 0.5).floor() as u64)
                .collect()
        })
        .collect()
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
