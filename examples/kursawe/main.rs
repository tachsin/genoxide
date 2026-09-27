//! Kursawe: minimize two objectives whose Pareto front is in disconnected pieces, with SPEA2 and
//! NSGA-II.
//!
//! Kursawe's problem in 3 variables, from genoxide's `multi::problems::Kursawe`. Its front isn't
//! known in closed form, so the example compares the two algorithms' fronts by their hypervolume,
//! and counts the pieces each finds: a new piece starts where two neighbors on the front are
//! more than 0.5 apart.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the runs' trace for the plot on the example's page:
//! both fronts and their hypervolumes, in at most 64 generations. A frame's evaluations are the two
//! runs' together.
//!
//! ```text
//! cargo run --release --example kursawe
//! ```

use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{Kursawe, MultiProblem};
use genoxide::prelude::*;
use serde_json::{Map, Value, json};

const REFERENCE: [f64; 2] = [-14.0, 1.0];

fn main() -> Result<()> {
    let problem = Kursawe::new(3);
    let spea2 = Spea2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0)?)
        .seed(1)
        .build()?;
    let tracing = std::env::var("GENOXIDE_TRACE").ok();
    // per algorithm, for the trace: the evaluations, the front and its hypervolume after each
    // generation
    let (mut spea2_history, mut nsga2_history) = (Vec::new(), Vec::new());
    let spea2 = MultiEngine::new(spea2, problem)
        .stop_when(Stop::generations(250))
        .on_generation(fronts(&mut spea2_history, tracing.is_some()))
        .run()?;
    report("SPEA2", &spea2.front_values());

    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0)?)
        .seed(1)
        .build()?;
    let nsga2 = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::generations(250))
        .on_generation(fronts(&mut nsga2_history, tracing.is_some()))
        .run()?;
    report("NSGA-II", &nsga2.front_values());
    if let Some(path) = tracing {
        write_trace(
            &path,
            [("SPEA2", &spea2_history), ("NSGA-II", &nsga2_history)],
        );
    }
    Ok(())
}

// the evaluations, the front and its hypervolume after a generation
type Front = (u64, Vec<[f64; 2]>, f64);

// records the front after each generation, for the trace
fn fronts<G: Genome>(
    history: &mut Vec<Front>,
    tracing: bool,
) -> impl FnMut(&MultiSnapshot<'_, G, 2>) + '_ {
    move |snapshot| {
        if tracing {
            let front: Vec<[f64; 2]> = snapshot
                .front()
                .iter()
                .filter_map(|x| x.fitness()?.values())
                .collect();
            let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
            history.push((snapshot.progress().evaluations(), front, volume));
        }
    }
}

// the runs side by side, a frame per generation
fn write_trace(path: &str, series: [(&str, &Vec<Front>); 2]) {
    let mut trace = Trace::new(64);
    let generations = series
        .iter()
        .map(|(_, history)| history.len())
        .max()
        .unwrap_or(0);
    for generation in 0..generations {
        let (mut fronts, mut volumes, mut evaluations) = (Map::new(), Map::new(), 0);
        for (name, history) in series {
            let (done, front, volume) = &history[generation.min(history.len() - 1)];
            fronts.insert(name.to_string(), json!(front));
            volumes.insert(name.to_string(), json!(volume));
            evaluations += done;
        }
        let frame = json!({
            "generation": generation,
            "evaluations": evaluations,
            "best": null,
            "median": null,
            "state": { "fronts": fronts, "hypervolume": volumes },
        });
        trace.push(generation as u64, frame);
    }
    trace.write(
        path,
        json!({
            "format": 1,
            "example": "kursawe",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "hypervolume",
            "log_y": false,
            "optimum": null,
            "plot": "front-2d",
            "problem": {
                "objectives": ["f1", "f2"],
                "true_front": null,
                "series": ["SPEA2", "NSGA-II"],
            },
        }),
    );
}

// the size of the front, its pieces and its hypervolume
fn report(name: &str, front: &[[f64; 2]]) {
    let mut sorted = front.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let pieces = 1 + sorted
        .windows(2)
        .filter(|pair| {
            let (d1, d2) = (pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]);
            (d1 * d1 + d2 * d2).sqrt() > 0.5
        })
        .count();
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    println!(
        "{name:<8} {} solutions in {pieces} pieces, hypervolume {volume:.4}",
        front.len()
    );
}

// ---- the trace of the runs, for the plot on the example's page ---------------------------------

// a frame per recorded generation, at most `most`: every `every`-th generation, with `every`
// doubling whenever there are `most`, and the last generation
struct Trace {
    most: usize,
    every: u64,
    frames: Vec<(u64, Value)>,
    last: Option<(u64, Value)>,
}

impl Trace {
    fn new(most: usize) -> Self {
        let (every, frames, last) = (1, Vec::new(), None);
        Self {
            most,
            every,
            frames,
            last,
        }
    }

    // keeps `frame` if it's of the `every`-th generation, or as the last one
    fn push(&mut self, generation: u64, frame: Value) {
        if generation % self.every != 0 {
            self.last = Some((generation, frame));
            return;
        }
        self.frames.push((generation, frame));
        self.last = None;
        if self.frames.len() == self.most {
            self.every *= 2;
            let every = self.every;
            self.frames
                .retain(|(generation, _)| generation % every == 0);
        }
    }

    // writes the settings and the frames to `path`, a frame per line
    fn write(self, path: &str, settings: Value) {
        let frames = self.frames.iter().chain(&self.last);
        let frames: Vec<String> = frames.map(|(_, frame)| to_json(frame)).collect();
        let settings = to_json(&settings);
        let head = &settings[..settings.len() - 1];
        let text = format!("{head},\"frames\":[\n{}\n]}}\n", frames.join(",\n"));
        std::fs::write(path, text).expect("the trace is written");
    }
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
