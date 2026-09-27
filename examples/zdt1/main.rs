//! ZDT1: minimize two conflicting objectives over 30 variables in [0, 1].
//!
//! Shows NSGA-II on genoxide's ZDT1 test problem, and the hypervolume of the final non-dominated
//! front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the run's trace for the plot on the example's page:
//! the front and its hypervolume, in at most 100 generations.
//!
//! ```text
//! cargo run --release --example zdt1
//! ```

use genoxide::Objective::Minimize;
use genoxide::genome::Genome;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{MultiProblem, Zdt1};
use genoxide::prelude::*;
use serde_json::{Value, json};

fn main() -> Result<()> {
    let problem = Zdt1::new(30);
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
        .seed(1)
        .build()?;

    let mut trace = std::env::var("GENOXIDE_TRACE")
        .ok()
        .map(|path| (path, Trace::new(100)));
    let outcome = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::evaluations(25_000))
        .on_generation(|snapshot| {
            if let Some((_, trace)) = &mut trace {
                let front: Vec<[f64; 2]> = snapshot
                    .front()
                    .iter()
                    .filter_map(|x| x.fitness()?.values())
                    .collect();
                let volume = hypervolume(&front, &[1.1, 1.1], &[Minimize; 2]);
                let state =
                    json!({ "fronts": { "NSGA-II": front }, "hypervolume": { "NSGA-II": volume } });
                trace.record(snapshot, state);
            }
        })
        .run()?;

    // the hypervolume of the front, with the reference point (1.1, 1.1)
    let front = outcome.front_values();
    let volume = hypervolume(&front, &[1.1, 1.1], &[Minimize; 2]);
    println!(
        "{} solutions on the front, hypervolume {volume:.4} (the whole front: 0.8767)",
        front.len()
    );
    if let Some((path, trace)) = trace {
        trace.write(
            &path,
            json!({
                "format": 1,
                "example": "zdt1",
                "objective": ["minimize", "minimize"],
                "x_label": "evaluations",
                "y_label": "hypervolume",
                "log_y": false,
                "optimum": 0.8767,
                "plot": "front-2d",
                "problem": {
                    "objectives": ["f1", "f2"],
                    "true_front": problem.optimal_front(100),
                    "series": ["NSGA-II"],
                },
            }),
        );
    }
    Ok(())
}

// ---- the trace of the run, for the plot on the example's page ----------------------------------

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

    // the generation's progress and the plot's `state`, whose hypervolume is the curve
    fn record<G: Genome, const M: usize>(
        &mut self,
        snapshot: &MultiSnapshot<'_, G, M>,
        state: Value,
    ) {
        let progress = snapshot.progress();
        let frame = json!({
            "generation": progress.generation(),
            "evaluations": progress.evaluations(),
            "best": null,
            "median": null,
            "state": state,
        });
        self.push(progress.generation(), frame);
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
