//! The trace of the run for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the best solution so far and its constraint violation, in at most 100
//! generations. The Python example writes the same file.
//!
//! The curve is the error f − f* of the best solution and of the population's median, on a log
//! scale, measured without the run's ε level: null while they're infeasible, whose values aren't
//! comparable to f*.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::{EQUALITY_TOLERANCE, G13};
use serde_json::{Value, json};

pub struct Trace {
    path: Option<String>,
    frames: Frames,
    // the last generation recorded: a re-evaluation repeats it
    last: Option<u64>,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        let frames = Frames::new(100);
        let last = None;
        Self { path, frames, last }
    }

    // records a generation: the errors of the best and the median, the best solution so far and
    // its constraint violations, 0 when met
    pub fn record(&mut self, snapshot: &Snapshot<'_, Reals>) {
        let generation = snapshot.progress().generation();
        if self.path.is_none() || self.last == Some(generation) {
            return;
        }
        self.last = Some(generation);
        // each solution's value and violation, without the ε level
        let problem = G13::default();
        let best = snapshot.best().genome();
        let (value, violation) = problem.evaluate(best);
        let best_error = (violation == 0.0).then(|| error(value));
        // the population in the order of Deb's rules: the feasible solutions by value, then the
        // infeasible ones
        let population = snapshot.population();
        let mut feasible: Vec<f64> = population
            .iter()
            .map(|individual| problem.evaluate(individual.genome()))
            .filter(|&(_, violation)| violation == 0.0)
            .map(|(value, _)| error(value))
            .collect();
        feasible.sort_by(f64::total_cmp);
        let median = median(&feasible, population.len());
        // an equality h = 0 is met when |h| ≤ δ: its violation max(0, |h| − δ), 0 when met
        let violations: Vec<f64> = G13::default()
            .constraints(best)
            .equalities()
            .iter()
            .map(|h| (h.abs() - EQUALITY_TOLERANCE).max(0.0))
            .collect();
        let state = json!({ "best": &best[..], "violations": violations });
        self.frames.push(frame(snapshot, best_error, median, state));
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let variables: Vec<Value> = (1..)
            .zip(G13::default().representation().bounds())
            .map(|(i, bounds)| {
                let bounds = [*bounds.start(), *bounds.end()];
                json!({ "name": format!("x{i}"), "unit": "", "bounds": bounds })
            })
            .collect();
        let settings = json!({
            "format": 1,
            "example": "cec2006_g13",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error f - f* of the best feasible solution",
            "log_y": true,
            "optimum": 0.0,
            "plot": "design",
            // equalities only, recorded as their excess over the tolerance: 0 when met, which is
            // active
            "problem": {
                "variables": variables,
                "constraints": ["h1", "h2", "h3"],
                "signed": true,
            },
        });
        write(&path, settings, self.frames.into_vec());
    }
}

// the error f - f* of a feasible solution's value, 0 at or below f*
fn error(value: f64) -> f64 {
    let optimum = G13::default().optimum().expect("known").value();
    (value - optimum).max(0.0)
}

// the median error of a population of `size` in the order of Deb's rules, given the sorted
// errors of its feasible solutions: None if the median is infeasible
fn median(feasible: &[f64], size: usize) -> Option<f64> {
    let middle = size / 2;
    if size == 0 || middle >= feasible.len() {
        None
    } else if size % 2 == 1 {
        Some(feasible[middle])
    } else {
        Some((feasible[middle - 1] + feasible[middle]) / 2.0)
    }
}

// the frame of a generation: its progress, the errors of the best and the median, and `state`
fn frame<G: Genome>(
    snapshot: &Snapshot<'_, G>,
    best: Option<f64>,
    median: Option<f64>,
    state: Value,
) -> Value {
    let progress = snapshot.progress();
    json!({
        "generation": progress.generation(),
        "evaluations": progress.evaluations(),
        "best": best,
        "median": median,
        "state": state,
    })
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
