//! Travelling salesman: the shortest round trip through the 52 locations in Berlin of TSPLIB's
//! berlin52, whose optimal tour has length 7542.
//!
//! A permutation genome is the order of the visits. Local search with inversion neighbors (a
//! random 2-opt move: a reversed segment of the tour) and simulated annealing, which also accepts
//! worse tours, less and less often as the temperature cools.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the run's trace for the plot on the example's page:
//! the best tour so far, in at most 200 generations.
//!
//! ```text
//! cargo run --release --example tsp_berlin52
//! ```

use genoxide::genome::Genome;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Value, json};

// the coordinates of the locations, from berlin52.tsp
const LOCATIONS: [(f64, f64); 52] = [
    (565.0, 575.0),
    (25.0, 185.0),
    (345.0, 750.0),
    (945.0, 685.0),
    (845.0, 655.0),
    (880.0, 660.0),
    (25.0, 230.0),
    (525.0, 1000.0),
    (580.0, 1175.0),
    (650.0, 1130.0),
    (1605.0, 620.0),
    (1220.0, 580.0),
    (1465.0, 200.0),
    (1530.0, 5.0),
    (845.0, 680.0),
    (725.0, 370.0),
    (145.0, 665.0),
    (415.0, 635.0),
    (510.0, 875.0),
    (560.0, 365.0),
    (300.0, 465.0),
    (520.0, 585.0),
    (480.0, 415.0),
    (835.0, 625.0),
    (975.0, 580.0),
    (1215.0, 245.0),
    (1320.0, 315.0),
    (1250.0, 400.0),
    (660.0, 180.0),
    (410.0, 250.0),
    (420.0, 555.0),
    (575.0, 665.0),
    (1150.0, 1160.0),
    (700.0, 580.0),
    (685.0, 595.0),
    (685.0, 610.0),
    (770.0, 610.0),
    (795.0, 645.0),
    (720.0, 635.0),
    (760.0, 650.0),
    (475.0, 960.0),
    (95.0, 260.0),
    (875.0, 920.0),
    (700.0, 500.0),
    (555.0, 815.0),
    (830.0, 485.0),
    (1170.0, 65.0),
    (830.0, 610.0),
    (605.0, 625.0),
    (595.0, 360.0),
    (1340.0, 725.0),
    (1740.0, 245.0),
];
const OPTIMUM: f64 = 7542.0;

// TSPLIB's EUC_2D distance: the Euclidean distance, rounded to the nearest integer
fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (a.0 - b.0, a.1 - b.1);
    ((dx * dx + dy * dy).sqrt() + 0.5).floor()
}

fn main() -> Result<()> {
    let distances: Vec<Vec<f64>> = LOCATIONS
        .iter()
        .map(|&a| LOCATIONS.iter().map(|&b| distance(a, b)).collect())
        .collect();
    let tour_length = |order: &Order| {
        (0..order.len())
            .map(|i| distances[order[i]][order[(i + 1) % order.len()]])
            .sum::<f64>()
    };

    let search = LocalSearch::builder(Permutation::new(LOCATIONS.len())?)
        .neighbor(InversionMutation)
        .acceptance(Acceptance::Annealing {
            initial_temperature: 100.0,
            cooling: 0.99996,
        })
        .minimize()
        .seed(1)
        .build()?;
    let mut trace = std::env::var("GENOXIDE_TRACE")
        .ok()
        .map(|path| (path, Trace::new(200)));
    let outcome = Engine::new(search, tour_length)
        .stop_when(Stop::target(OPTIMUM).or(Stop::evaluations(200_000)))
        .on_generation(|snapshot| {
            if let Some((_, trace)) = &mut trace {
                let state = json!({ "best": &snapshot.best().genome()[..] });
                trace.record(snapshot, state);
            }
        })
        .run()?;

    println!(
        "tour length {} after {} evaluations (the optimum: {OPTIMUM})",
        outcome.best_fitness(),
        outcome.evaluations()
    );
    // the tour from location 1, numbered from 1 as in TSPLIB
    let order = outcome.best_genome();
    let start = order.iter().position(|&i| i == 0).unwrap_or(0);
    let tour: Vec<usize> = order[start..]
        .iter()
        .chain(&order[..start])
        .map(|location| location + 1)
        .collect();
    println!("tour {tour:?}");
    if let Some((path, trace)) = trace {
        trace.write(
            &path,
            json!({
                "format": 1,
                "example": "tsp_berlin52",
                "objective": "minimize",
                "x_label": "evaluations",
                "y_label": "tour length",
                "log_y": false,
                "optimum": OPTIMUM,
                "plot": "tour",
                "problem": { "points": LOCATIONS.to_vec() },
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
