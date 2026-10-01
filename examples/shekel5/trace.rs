//! The trace of the runs with the global topology for the plot on the example's page, written to
//! the file that `GENOXIDE_TRACE` names: the best point of each run side by side, drawn at
//! (x₁, x₂) on the contour of the plane x₃ = x₁, x₄ = x₂, and the best value's and the median's
//! error to the best known minimum, in at most 100 of their generations. The Python example
//! writes the same file.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{Problem, Shekel5};
use serde_json::{Value, json};

// a run's evaluations, and its best value and point, after a generation
type Step = (u64, Option<f64>, [f64; 2]);

pub struct Trace {
    path: Option<String>,
    runs: Vec<Vec<Step>>,
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

    // records a generation of a run, which starts at generation 0: its evaluations, and its
    // best value and point so far
    pub fn record(&mut self, snapshot: &Snapshot<'_, Reals>) {
        if self.path.is_none() {
            return;
        }
        let progress = snapshot.progress();
        if progress.generation() == 0 {
            self.runs.push(Vec::new());
        }
        let best = snapshot.best();
        let value = best.fitness().and_then(Fitness::score);
        // the point on the plot: (x₁, x₂)
        let point = [best.genome()[0], best.genome()[1]];
        let run = self.runs.last_mut().expect("a run");
        run.push((progress.evaluations(), value, point));
    }

    // writes the trace, if there's one: the runs side by side, a frame per generation, with
    // their points, and the best value's and the median's error to the best known minimum
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let optimum = Shekel5.optimum().expect("known");
        let minimum = optimum.value();
        // rounding can put a solution a few ulps below the minimum
        let error = |value: f64| (value - minimum).max(0.0);
        let mut frames = Frames::new(100);
        let generations = self.runs.iter().map(Vec::len).max().unwrap_or(0);
        for generation in 0..generations {
            let runs = self.runs.iter();
            let runs = runs.map(|run| run[generation.min(run.len() - 1)]);
            let runs: Vec<Step> = runs.collect();
            let values: Vec<f64> = runs.iter().filter_map(|run| run.1).collect();
            let best = runs.iter().filter(|run| run.1.is_some());
            let best = best.min_by(|a, b| a.1.unwrap().total_cmp(&b.1.unwrap()));
            let points: Vec<[f64; 2]> = runs.iter().map(|run| run.2).collect();
            frames.push(json!({
                "generation": generation,
                "evaluations": runs.iter().map(|run| run.0).sum::<u64>(),
                "best": best.and_then(|best| best.1).map(error),
                "median": median(values).map(error),
                "state": { "population": points, "best": best.map(|best| best.2) },
            }));
        }
        let bounds = Shekel5.representation().bounds().to_vec();
        let bounds: Vec<[f64; 2]> = bounds[..2].iter().map(|b| [*b.start(), *b.end()]).collect();
        let minima: Vec<&[f64]> = optimum.solutions().iter().map(|x| &x[..2]).collect();
        let settings = json!({
            "format": 1,
            "example": "shekel5",
            "objective": "minimize",
            "x_label": "generations",
            "y_label": "error to the best known minimum",
            "log_y": true,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "shekel5",
                "bounds": bounds,
                "minima": minima,
                "minima_label": "best known minimum",
                "labels": { "x": "x₁", "y": "x₂", "f": "f(x₁, x₂, x₁, x₂)" },
            },
        });
        write(&path, settings, frames.into_vec());
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
// to within a thousandth of their range over the run: a front's hypervolumes, in the state or in a
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
            (high - low) / 1000.0
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
