//! The trace for the plot on the example's page, written to the file that `GENOXIDE_TRACE` names.
//! The plot draws the population on the function's contour, which needs two dimensions: the trace
//! is of a separate run of L-SHADE in 2 dimensions, with a budget of 10,000 evaluations per
//! dimension, in at most 100 generations. The best value and the median are the errors to the
//! minimum, −837.97. The Python example writes the same file.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{Problem, Schwefel2_26};
use serde_json::{Value, json};

// runs L-SHADE in 2 dimensions and writes its trace, if GENOXIDE_TRACE is set
pub fn record_2d() -> Result<()> {
    let Ok(path) = std::env::var("GENOXIDE_TRACE") else {
        return Ok(());
    };
    let problem = Schwefel2_26::new(2);
    let optimum = problem.optimum().expect("known");
    let minimum = optimum.value();
    let budget = 20_000;
    let l_shade = De::l_shade(problem.representation(), budget)
        .minimize()
        .seed(1)
        .build()?;
    let mut frames = Frames::new(100);
    Engine::new(l_shade, problem)
        .stop_when(Stop::target(minimum + 1e-8).or(Stop::evaluations(budget)))
        .on_generation(|snapshot| {
            let population = snapshot.population().iter().map(|x| &x.genome()[..]);
            let population: Vec<&[f64]> = population.collect();
            let best = &snapshot.best().genome()[..];
            let state = json!({ "population": population, "best": best });
            let mut frame = frame(snapshot, state);
            // the errors to the minimum: rounding can put a solution a few ulps below it
            for key in ["best", "median"] {
                if let Some(value) = frame[key].as_f64() {
                    frame[key] = json!((value - minimum).max(0.0));
                }
            }
            frames.push(frame);
        })
        .run()?;
    let minima: Vec<&[f64]> = optimum.solutions().iter().map(|x| &x[..]).collect();
    let settings = json!({
        "format": 1,
        "example": "schwefel_2_26",
        "objective": "minimize",
        "x_label": "evaluations",
        "y_label": "error",
        "log_y": true,
        "optimum": 0.0,
        "plot": "contour",
        "problem": {
            "function": "schwefel_2_26",
            "bounds": [[-500.0, 500.0], [-500.0, 500.0]],
            "minima": minima,
        },
    });
    write(&path, settings, frames.into_vec());
    Ok(())
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

// the frame of a generation: its progress, the median score of its population and `state`
fn frame<G: Genome>(snapshot: &Snapshot<'_, G>, state: Value) -> Value {
    let progress = snapshot.progress();
    let population = snapshot.population().iter();
    let scores = population.filter_map(|individual| individual.fitness()?.score());
    json!({
        "generation": progress.generation(),
        "evaluations": progress.evaluations(),
        "best": progress.best().and_then(Fitness::score),
        "median": median(scores.collect()),
        "state": state,
    })
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
