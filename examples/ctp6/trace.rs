//! The trace of the run for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the front, the infeasible solutions of the population and its feasible
//! share, and the front's hypervolume in scaled objectives, in at most 100 generations, over the
//! problem's feasible region. The Python example writes the same file.

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{Ctp6, MultiProblem};
use genoxide::multi::{MultiFitnessFunction, MultiSnapshot, Scores};
use genoxide::prelude::*;
use serde_json::{Value, json};

const SERIES: &str = "NSGA-II";

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

    // records a generation: the front and its hypervolume, and the population's infeasible
    // solutions and feasible share
    pub fn record<G: Genome>(&mut self, snapshot: &MultiSnapshot<'_, G, 2>) {
        if self.path.is_none() {
            return;
        }
        let population = snapshot.population().as_slice();
        let values = |x: &Individual<G, Scores<2>>| x.fitness()?.values();
        let feasible = |x: &&Individual<G, Scores<2>>| x.fitness().is_some_and(|x| x.is_feasible());
        let front: Vec<[f64; 2]> = snapshot
            .front()
            .iter()
            .filter(feasible)
            .filter_map(values)
            .collect();
        let infeasible = population.iter().filter(|x| !feasible(x));
        let infeasible: Vec<[f64; 2]> = infeasible.filter_map(values).collect();
        let share = population.iter().filter(feasible).count() as f64 / population.len() as f64;
        let volume = hypervolume(&super::scaled(&front), &[1.1, 1.1], &[Minimize; 2]);
        let state = json!({
            "fronts": { SERIES: front },
            "infeasible": { SERIES: infeasible },
            "hypervolume": { SERIES: volume },
            "feasible": { SERIES: share },
        });
        self.frames.push(frame(snapshot, state));
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let front = Ctp6.optimal_front(400).expect("known");
        // the optimal front, in its pieces, over the feasible region
        let problem = json!({
            "objectives": ["f1", "f2"],
            "true_front": pieces(&front),
            "feasible_region": region(),
            "series": [SERIES],
        });
        let settings = json!({
            "format": 1,
            "example": "ctp6",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "hypervolume (scaled objectives)",
            "log_y": false,
            "optimum": super::whole_front_hypervolume(),
            "plot": "front-2d",
            "problem": problem,
        });
        write(&path, settings, self.frames.into_vec());
    }
}

// the feasible region over f₁ in [0, 1] and f₂ in [0, TOP], in COLUMNS × ROWS cells: a cell is
// feasible when the genome with its center's objectives meets the constraints. x₁ = f₁, and x₂ is
// found by bisection, as f₂ grows with x₂; rows from the bottom up, each the pairs of columns
// [start, end) where it's feasible
const TOP: f64 = 11.0;
const COLUMNS: usize = 400;
const ROWS: usize = 880;

fn region() -> Value {
    let problem = Ctp6;
    let high = *problem.representation().bounds()[1].end();
    let f2 = |x1: f64, x2: f64| problem.evaluate(&Reals::from(vec![x1, x2])).0[1];
    let rows: Vec<Vec<usize>> = (0..ROWS)
        .map(|row| {
            let target = TOP * (row as f64 + 0.5) / ROWS as f64;
            let feasible: Vec<bool> = (0..COLUMNS)
                .map(|column| {
                    let x1 = (column as f64 + 0.5) / COLUMNS as f64;
                    if target < f2(x1, 0.0) || target > f2(x1, high) {
                        return false;
                    }
                    let (mut low, mut up) = (0.0, high);
                    for _ in 0..50 {
                        let middle = 0.5 * (low + up);
                        if f2(x1, middle) < target {
                            low = middle;
                        } else {
                            up = middle;
                        }
                    }
                    problem.evaluate(&Reals::from(vec![x1, up])).1 == 0.0
                })
                .collect();
            let mut runs = Vec::new();
            for (column, &inside) in feasible.iter().enumerate() {
                let before = column > 0 && feasible[column - 1];
                if inside != before {
                    runs.push(column);
                }
            }
            if feasible[COLUMNS - 1] {
                runs.push(COLUMNS);
            }
            runs
        })
        .collect();
    json!({ "x": [0.0, 1.0], "y": [0.0, TOP], "columns": COLUMNS, "rows": rows })
}

// the pieces of the optimal front, split where neighbors are more than 0.01 apart
fn pieces(front: &[[f64; 2]]) -> Vec<Vec<[f64; 2]>> {
    super::pieces(front)
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
