//! Gear train design (Sandgren, 1990): the numbers of teeth of a compound gear train of four
//! gears, from 12 to 60 each, whose ratio is closest to 1/6.931. An integer problem.
//!
//! The problem is genoxide's `GearTrain`, on integer genes; its score is the squared error of
//! the ratio. A genetic algorithm with uniform crossover and a mutation that redraws each gene
//! with probability 0.25 searches the 49⁴ ≈ 5.8 million designs, until it reaches the minimum,
//! 2.700857e-12, known by evaluating them all.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the run's trace for the plot on the example's page:
//! the best design so far, in at most 200 generations.
//!
//! ```text
//! cargo run --release --example gear_train
//! ```

use genoxide::genome::Genome;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::engineering::GearTrain;
use serde_json::{Value, json};

fn main() -> Result<()> {
    let problem = GearTrain;
    let minimum = problem.optimum().expect("known").value();
    let ga = Ga::builder(problem.representation())
        .population_size(100)
        .select(Tournament::new(2)?)
        .crossover(UniformCrossover::new())
        .mutate(UniformMutation::per_gene(0.25)?)
        .minimize()
        .seed(1)
        .build()?;
    let mut trace = std::env::var("GENOXIDE_TRACE")
        .ok()
        .map(|path| (path, Trace::new(200)));
    let outcome = Engine::new(ga, problem)
        .stop_when(Stop::target(minimum).or(Stop::generations(2_000)))
        .on_generation(|snapshot| {
            if let Some((_, trace)) = &mut trace {
                let best = &snapshot.best().genome()[..];
                // no constraints: no violations
                trace.record(snapshot, json!({ "best": best, "violations": [] }));
            }
        })
        .run()?;

    let teeth = outcome.best_genome();
    let ratio = (teeth[0] * teeth[1]) as f64 / (teeth[2] * teeth[3]) as f64;
    println!(
        "error {:.6e} after {} generations (the minimum: {minimum:.6e})",
        outcome.best_fitness().score().unwrap_or(f64::NAN),
        outcome.generations()
    );
    println!(
        "teeth ({}, {}, {}, {}), ratio {ratio:.8} (the target: {:.8})",
        teeth[0],
        teeth[1],
        teeth[2],
        teeth[3],
        1.0 / 6.931
    );
    if let Some((path, trace)) = trace {
        let gear = |name| json!({ "name": name, "unit": "teeth", "bounds": [12, 60] });
        trace.write(
            &path,
            json!({
                "format": 1,
                "example": "gear_train",
                "objective": "minimize",
                "x_label": "generations",
                "y_label": "squared error of the ratio",
                "log_y": true,
                "optimum": minimum,
                "plot": "design",
                "problem": {
                    "variables": [gear("Td"), gear("Tb"), gear("Ta"), gear("Tf")],
                    "constraints": [],
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

    // the generation's progress, the median score of its population and the plot's `state`
    fn record<G: Genome>(&mut self, snapshot: &Snapshot<'_, G>, state: Value) {
        let progress = snapshot.progress();
        let population = snapshot.population().iter();
        let scores = population.filter_map(|individual| individual.fitness()?.score());
        let frame = json!({
            "generation": progress.generation(),
            "evaluations": progress.evaluations(),
            "best": progress.best().and_then(Fitness::score),
            "median": median(scores.collect()),
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
