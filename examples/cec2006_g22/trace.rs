//! The trace of the run on x1, x8 and x9 for the plot on the example's page, written to the file
//! that `GENOXIDE_TRACE` names: the best solution so far, all 22 variables, and its constraints, in
//! at most 100 generations. The Python example writes the same file.
//!
//! The curve is the error f − 236.370313314566 of the best feasible solution and of the
//! population's median, on a log scale, against the least value with every equality met exactly:
//! null while they're infeasible, whose values aren't comparable to it.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::{EQUALITY_TOLERANCE, G22};
use serde_json::{Value, json};

pub struct Trace {
    path: Option<String>,
    frames: Frames,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        let frames = Frames::new(100);
        Self { path, frames }
    }

    // records a generation: the errors of the best and the median, the best solution so far, its
    // 22 variables from `solve`, and its constraints, g(x) for the inequalities, satisfied at or
    // below 0 and active at 0, and for the equalities the excess max(0, |h(x)| - 0.0001), 0
    // (active) when met (the page shows each one's state)
    pub fn record(&mut self, snapshot: &Snapshot<'_, Reals>, solve: fn(&Reals) -> Reals) {
        if self.path.is_none() {
            return;
        }
        let best = solve(snapshot.best().genome());
        let fitness = snapshot.best().fitness().expect("evaluated");
        let best_error = fitness.is_feasible().then(|| error(fitness));
        // the population in the order of Deb's rules: the feasible solutions by value, then the
        // infeasible ones
        let population: Vec<Fitness> = snapshot
            .population()
            .iter()
            .filter_map(|individual| individual.fitness())
            .collect();
        let mut feasible: Vec<f64> = population
            .iter()
            .filter(|fitness| fitness.is_feasible())
            .map(|&fitness| error(fitness))
            .collect();
        feasible.sort_by(f64::total_cmp);
        let median = median(&feasible, population.len());
        let constraints = G22::default().constraints(&best);
        let equalities = constraints.equalities().iter();
        let equalities = equalities.map(|h| (h.abs() - EQUALITY_TOLERANCE).max(0.0));
        let values: Vec<f64> = constraints
            .inequalities()
            .iter()
            .copied()
            .chain(equalities)
            .collect();
        let state = json!({ "best": &best[..], "violations": values });
        self.frames.push(frame(snapshot, best_error, median, state));
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let variables: Vec<Value> = (1..)
            .zip(G22::default().representation().bounds())
            .map(|(i, bounds)| {
                let bounds = [*bounds.start(), *bounds.end()];
                json!({ "name": format!("x{i}"), "unit": "", "bounds": bounds })
            })
            .collect();
        let constraints: Vec<String> = std::iter::once("g1".to_string())
            .chain((1..=19).map(|i| format!("h{i}")))
            .collect();
        let settings = json!({
            "format": 1,
            "example": "cec2006_g22",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error f - 236.370313314566 of the best feasible solution",
            "log_y": true,
            "optimum": 0.0,
            "plot": "design",
            // the inequalities as g(x), the equalities as their excess over the tolerance: 0 when
            // met, which is active
            "problem": {
                "variables": variables,
                "constraints": constraints,
                "signed": true,
            },
        });
        write(&path, settings, self.frames.into_vec());
    }
}

// the error f - 236.370313314566 of a feasible solution, 0 at or below it
fn error(fitness: Fitness) -> f64 {
    (fitness.score().expect("valid") - 236.370_313_314_566).max(0.0)
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
