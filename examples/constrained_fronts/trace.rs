//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: a panel per problem with its front, the infeasible solutions of the
//! population and its feasible share, and a curve of each front's normalized hypervolume, in at
//! most 50 generations. The Python example writes the same file.

use crate::normalized_hypervolume;
use genoxide::multi::problems::DynMultiProblem;
use genoxide::multi::{MultiSnapshot, Scores};
use genoxide::prelude::*;
use serde_json::{Map, Value, json};

pub struct Trace {
    path: Option<String>,
    // per problem, the frames of its run: the generation, evaluations, normalized hypervolume
    // and panel state; the runs are as long, so each keeps the same generations
    runs: Vec<Frames>,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        let runs = Vec::new();
        Self { path, runs }
    }

    // the callback that records the run on `problem` after each generation: its front, the
    // population's infeasible solutions and feasible share, and the feasible front's normalized
    // hypervolume
    pub fn panel<'a>(
        &'a mut self,
        problem: &'a dyn DynMultiProblem<2>,
    ) -> impl FnMut(&MultiSnapshot<'_, Reals, 2>) + 'a {
        let tracing = self.path.is_some();
        self.runs.push(Frames::new(50));
        let frames = self.runs.last_mut().expect("a run");
        move |snapshot| {
            if !tracing {
                return;
            }
            let population = snapshot.population().as_slice();
            let values = |x: &Individual<Reals, Scores<2>>| x.fitness()?.values();
            let feasible =
                |x: &&Individual<Reals, Scores<2>>| x.fitness().is_some_and(|x| x.is_feasible());
            let front = snapshot.front();
            let feasible_front: Vec<[f64; 2]> =
                front.iter().filter(feasible).filter_map(values).collect();
            let front: Vec<[f64; 2]> = front.iter().filter_map(values).collect();
            let infeasible = population.iter().filter(|x| !feasible(x));
            let infeasible: Vec<[f64; 2]> = infeasible.filter_map(values).collect();
            let share = population.iter().filter(feasible).count() as f64 / population.len() as f64;
            let progress = snapshot.progress();
            frames.push(json!({
                "generation": progress.generation(),
                "evaluations": progress.evaluations(),
                "hypervolume": normalized_hypervolume(problem, &feasible_front),
                "state": {
                    "fronts": { "NSGA-II": front },
                    "infeasible": { "NSGA-II": infeasible },
                    "feasible": { "NSGA-II": share },
                },
            }));
        }
    }

    // writes the trace, if there's one: a frame per recorded generation, with each problem's
    // normalized hypervolume and panel
    pub fn write(self, problems: &[Box<dyn DynMultiProblem<2>>]) {
        let Some(path) = self.path else { return };
        let names: Vec<&str> = problems.iter().map(|problem| problem.name()).collect();
        let runs: Vec<Vec<Value>> = self.runs.into_iter().map(Frames::into_vec).collect();
        let frame = |k: usize| {
            let mut series = Map::new();
            for (name, run) in names.iter().zip(&runs) {
                series.insert(name.to_string(), run[k]["hypervolume"].clone());
            }
            let panels: Vec<&Value> = runs.iter().map(|run| &run[k]["state"]).collect();
            json!({
                "generation": runs[0][k]["generation"],
                "evaluations": runs[0][k]["evaluations"],
                "series": series,
                "state": { "panels": panels },
            })
        };
        let frames = (0..runs[0].len()).map(frame).collect();
        let panel = |problem: &dyn DynMultiProblem<2>| {
            json!({
                "title": problem.name(),
                "problem": {
                    "objectives": ["f1", "f2"],
                    "true_front": problem.optimal_front(100),
                    "series": ["NSGA-II"],
                },
            })
        };
        let panels: Vec<Value> = problems.iter().map(|x| panel(x.as_ref())).collect();
        let settings = json!({
            "format": 1,
            "example": "constrained_fronts",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "normalized hypervolume",
            "log_y": false,
            "optimum": null,
            "plot": "grid",
            "problem": {
                "panel_plot": "front-2d",
                "panels": panels,
                "series": names,
            },
        });
        write(&path, settings, frames);
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
