//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: both fronts, each population's infeasible solutions and feasible share,
//! and the fronts' hypervolumes with normalized objectives, in at most 64 generations; and the
//! optimal front, in its pieces, and the feasible region, sampled. A frame's evaluations are the
//! two runs' together. The Python example writes the same file.

use crate::{REFERENCE, normalized};
use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{MultiProblem, Mw3};
use genoxide::multi::{MultiFitnessFunction, MultiSnapshot, Scores};
use genoxide::prelude::*;
use serde_json::{Map, Value, json};

// after a generation: the evaluations, the feasible front, the infeasible solutions, the
// hypervolume and the feasible share
type Record = (u64, Vec<[f64; 2]>, Vec<[f64; 2]>, f64, f64);

// the samples of the feasible region along x₁ and along the last variable, and its grid
const SAMPLES: usize = 400;
const CELLS: usize = 80;

pub struct Trace {
    path: Option<String>,
    // per run, its record after each generation
    series: Vec<(&'static str, Vec<Record>)>,
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

    // the callback that records the run `name` after each generation
    pub fn fronts(&mut self, name: &'static str) -> impl FnMut(&MultiSnapshot<'_, Reals, 2>) {
        let tracing = self.path.is_some();
        self.series.push((name, Vec::new()));
        let (_, history) = self.series.last_mut().expect("a series");
        move |snapshot| {
            if !tracing {
                return;
            }
            let values = |x: &Individual<Reals, Scores<2>>| x.fitness()?.values();
            let feasible = |x: &&Individual<Reals, Scores<2>>| {
                x.fitness().is_some_and(|scores| scores.is_feasible())
            };
            let front: Vec<[f64; 2]> = snapshot
                .front()
                .iter()
                .filter(feasible)
                .filter_map(values)
                .collect();
            let population = snapshot.population().as_slice();
            let infeasible = population.iter().filter(|x| !feasible(x));
            let infeasible: Vec<[f64; 2]> = infeasible.filter_map(values).collect();
            let share = population.iter().filter(feasible).count() as f64 / population.len() as f64;
            let volume = hypervolume(
                &normalized(&Mw3::default(), &front),
                &REFERENCE,
                &[Minimize; 2],
            );
            let evaluations = snapshot.progress().evaluations();
            history.push((evaluations, front, infeasible, volume, share));
        }
    }

    // writes the trace, if there's one: the runs side by side, a frame per generation
    pub fn write(self, problem: &Mw3) {
        let Some(path) = self.path else { return };
        let mut frames = Frames::new(64);
        let generations = self.series.iter().map(|(_, history)| history.len()).max();
        for generation in 0..generations.unwrap_or(0) {
            let mut state = [Map::new(), Map::new(), Map::new(), Map::new()];
            let mut evaluations = 0;
            for (name, history) in &self.series {
                let (done, front, infeasible, volume, share) =
                    &history[generation.min(history.len() - 1)];
                let values = [json!(front), json!(infeasible), json!(volume), json!(share)];
                for (map, value) in state.iter_mut().zip(values) {
                    map.insert(name.to_string(), value);
                }
                evaluations += done;
            }
            let [fronts, infeasible, volumes, shares] = state;
            frames.push(json!({
                "generation": generation,
                "evaluations": evaluations,
                "best": null,
                "median": null,
                "state": {
                    "fronts": fronts,
                    "infeasible": infeasible,
                    "hypervolume": volumes,
                    "feasible": shares,
                },
            }));
        }
        let names: Vec<&str> = self.series.iter().map(|(name, _)| *name).collect();
        let whole = normalized(problem, &problem.optimal_front(20_000).expect("known"));
        let settings = json!({
            "format": 1,
            "example": "mw3",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "hypervolume",
            "log_y": false,
            "optimum": hypervolume(&whole, &REFERENCE, &[Minimize; 2]),
            "plot": "front-2d",
            "problem": {
                "objectives": ["f1", "f2"],
                "true_front": pieces(&problem.optimal_front(400).expect("known")),
                "series": names,
                "feasible_region": region(problem),
            },
        });
        write(&path, settings, frames.into_vec());
    }
}

// the front in pieces, split where two points are more than 2% of its extent apart, each point
// once
fn pieces(front: &[[f64; 2]]) -> Vec<Vec<[f64; 2]>> {
    let (first, last) = (front[0], front[front.len() - 1]);
    let extent = (last[0] - first[0]).hypot(first[1] - last[1]);
    let mut pieces: Vec<Vec<[f64; 2]>> = Vec::new();
    for &point in front {
        match pieces.last_mut() {
            Some(piece) => {
                let previous = piece[piece.len() - 1];
                if (point[0] - previous[0]).hypot(point[1] - previous[1]) > 0.02 * extent {
                    pieces.push(vec![point]);
                } else if point != previous {
                    piece.push(point);
                }
            }
            None => pieces.push(vec![point]),
        }
    }
    pieces
}

// the feasible region in objective space: genomes with x₁ evenly spread over its range and the last
// variable swept over its range, the other distance variables at their optimal values, which moves
// g from 1 up; a cell of a grid of 80 × 80 over [0, 1.25 f₁ᵐᵃˣ] × [0, 1.25 f₂ᵐᵃˣ] (the nadir
// point's) is feasible if a feasible genome lands in it, and the rows go up from f₂ = 0
fn region(problem: &Mw3) -> Value {
    let n = problem.variables();
    let upper = 1.0;
    let nadir = problem.nadir_point().expect("known");
    let (width, height) = (1.25 * nadir[0], 1.25 * nadir[1]);
    let mut cells = vec![vec![false; CELLS]; CELLS];
    for a in 0..SAMPLES {
        let x1 = upper * a as f64 / (SAMPLES - 1) as f64;
        let mut x = vec![x1];
        x.extend(optimal(x1, n));
        for b in 0..SAMPLES {
            x[n - 1] = upper * b as f64 / (SAMPLES - 1) as f64;
            let (f, violation) = problem.evaluate(&Reals::from(x.clone()));
            let i = (f[0] / width * CELLS as f64).floor();
            let j = (f[1] / height * CELLS as f64).floor();
            if violation == 0.0 && i >= 0.0 && j >= 0.0 && i < CELLS as f64 && j < CELLS as f64 {
                cells[j as usize][i as usize] = true;
            }
        }
    }
    // each row as the pairs of columns [start, end) where it's feasible
    let rows: Vec<Vec<usize>> = cells.iter().map(Vec::as_slice).map(runs).collect();
    json!({"x": [0.0, width], "y": [0.0, height], "columns": CELLS, "rows": rows})
}

// the columns where a row of cells turns feasible or infeasible, as pairs [start, end)
fn runs(row: &[bool]) -> Vec<usize> {
    let mut runs = Vec::new();
    for (column, &inside) in row.iter().enumerate() {
        if inside != (column > 0 && row[column - 1]) {
            runs.push(column);
        }
    }
    if row.last() == Some(&true) {
        runs.push(row.len());
    }
    runs
}

// the distance variables where g₃ is 1: each is 1 − (the previous − 0.5)², from x₁
fn optimal(x1: f64, n: usize) -> Vec<f64> {
    let mut previous = x1;
    (1..n)
        .map(|_| {
            previous = 1.0 - (previous - 0.5) * (previous - 0.5);
            previous
        })
        .collect()
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
