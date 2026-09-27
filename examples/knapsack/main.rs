//! 0/1 knapsack: choose items with the highest total value that fit in the knapsack.
//!
//! Shows a constraint with Deb's feasibility rules: the fitness function returns the value and how
//! far the weight exceeds the capacity, so overweight selections still guide the search towards
//! the feasible ones. The result is checked against the optimum found by dynamic programming.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the run's trace for the plot on the example's page:
//! the best selection so far, in at most 200 generations.
//!
//! ```text
//! cargo run --release --example knapsack
//! ```

use genoxide::genome::Genome;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Value, json};

// (weight, value)
const ITEMS: [(u32, u32); 20] = [
    (23, 92),
    (31, 57),
    (29, 49),
    (44, 68),
    (53, 60),
    (38, 43),
    (63, 67),
    (85, 84),
    (89, 87),
    (82, 72),
    (12, 31),
    (17, 29),
    (41, 52),
    (35, 38),
    (27, 44),
    (58, 61),
    (19, 26),
    (46, 55),
    (71, 70),
    (33, 41),
];
const CAPACITY: u32 = 400;

// the total weight and value of the selected items
fn totals(selection: &Bits) -> (u32, u32) {
    let (mut weight, mut value) = (0, 0);
    for (item, selected) in selection.iter().enumerate() {
        if selected {
            weight += ITEMS[item].0;
            value += ITEMS[item].1;
        }
    }
    (weight, value)
}

// the total value, and how much the weight exceeds the capacity (0 if the items fit)
fn value(selection: &Bits) -> (f64, f64) {
    let (weight, value) = totals(selection);
    (
        f64::from(value),
        constraint::at_most(f64::from(weight), f64::from(CAPACITY)),
    )
}

// the best value that fits, by dynamic programming over the capacities
fn optimum() -> u32 {
    let mut best = [0; CAPACITY as usize + 1];
    for (weight, value) in ITEMS {
        for capacity in (weight as usize..=CAPACITY as usize).rev() {
            best[capacity] = best[capacity].max(best[capacity - weight as usize] + value);
        }
    }
    best[CAPACITY as usize]
}

fn main() -> Result<()> {
    let ga = Ga::builder(Binary::new(ITEMS.len())?)
        .population_size(60)
        .select(Tournament::new(3)?)
        .crossover(PointCrossover::two_point())
        .mutate(BitFlip::per_gene(1.0 / ITEMS.len() as f64)?)
        .seed(7)
        .build()?;

    let mut trace = std::env::var("GENOXIDE_TRACE")
        .ok()
        .map(|path| (path, Trace::new(200)));
    let outcome = Engine::new(ga, value)
        .stop_when(Stop::stagnation(200).or(Stop::generations(2_000)))
        .on_generation(|snapshot| {
            if let Some((_, trace)) = &mut trace {
                let best: Vec<u8> = snapshot.best().genome().iter().map(u8::from).collect();
                trace.record(snapshot, json!({ "best": best }));
            }
        })
        .run()?;

    let best = outcome.best_genome();
    let items: Vec<usize> = (0..ITEMS.len())
        .filter(|&item| best.get(item) == Some(true))
        .collect();
    let (weight, value) = totals(best);
    println!("items {items:?}");
    println!("value {value}, weight {weight} of {CAPACITY}");
    println!(
        "after {} evaluations; the optimum is {}",
        outcome.evaluations(),
        optimum()
    );
    if let Some((path, trace)) = trace {
        let items: Vec<Value> = ITEMS
            .iter()
            .map(|&(weight, value)| json!({ "weight": weight, "value": value }))
            .collect();
        trace.write(
            &path,
            json!({
                "format": 1,
                "example": "knapsack",
                "objective": "maximize",
                "x_label": "generations",
                "y_label": "value",
                "log_y": false,
                "optimum": f64::from(optimum()),
                "plot": "knapsack",
                "problem": { "capacity": CAPACITY, "items": items },
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
