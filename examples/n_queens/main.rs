//! N-Queens: place N queens on an N×N board so that no two attack each other.
//!
//! A permutation genome puts one queen in each row and each column (queen `row` is in column
//! `order[row]`), so only the diagonals can conflict. Permutations have no position-wise
//! crossover, so this uses (μ+λ) with swap mutation only.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the run's trace for the plot on the example's page:
//! the best board so far and its attacking pairs, in at most 200 generations.
//!
//! ```text
//! cargo run --release --example n_queens
//! ```

use genoxide::genome::Genome;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Value, json};

const N: usize = 64;

// the number of pairs of queens on the same diagonal
fn conflicts(order: &Order) -> f64 {
    let mut diagonals = vec![0usize; 2 * N];
    let mut anti_diagonals = vec![0usize; 2 * N];
    for (row, &column) in order.iter().enumerate() {
        diagonals[row + N - column] += 1;
        anti_diagonals[row + column] += 1;
    }
    let pairs = |count: &usize| count * count.saturating_sub(1) / 2;
    (diagonals.iter().map(pairs).sum::<usize>() + anti_diagonals.iter().map(pairs).sum::<usize>())
        as f64
}

fn main() -> Result<()> {
    let ga = Ga::builder(Permutation::new(N)?)
        .population_size(20)
        .select(Tournament::new(2)?)
        .crossover(NoCrossover)
        .mutate(SwapMutation::new())
        .scheme(Scheme::MuPlusLambda { lambda: 20 })
        .minimize()
        .seed(1)
        .build()?;

    let mut trace = std::env::var("GENOXIDE_TRACE")
        .ok()
        .map(|path| (path, Trace::new(200)));
    let outcome = Engine::new(ga, conflicts)
        .stop_when(Stop::target(0.0).or(Stop::generations(50_000)))
        .on_generation(|snapshot| {
            if let Some((_, trace)) = &mut trace {
                let best = snapshot.best().genome();
                let state = json!({ "best": &best[..], "attacks": attacks(best) });
                trace.record(snapshot, state);
            }
        })
        .run()?;

    println!(
        "{} conflicts after {} generations and {} evaluations",
        outcome.best_fitness(),
        outcome.generations(),
        outcome.evaluations()
    );
    println!("columns {:?}", &outcome.best_genome()[..]);
    if let Some((path, trace)) = trace {
        trace.write(
            &path,
            json!({
                "format": 1,
                "example": "n_queens",
                "objective": "minimize",
                "x_label": "generations",
                "y_label": "conflicts",
                "log_y": false,
                "optimum": 0.0,
                "plot": "board",
                "problem": { "n": N },
            }),
        );
    }
    Ok(())
}

// the pairs of rows whose queens attack each other, on a diagonal
fn attacks(order: &Order) -> Vec<[usize; 2]> {
    let mut pairs = Vec::new();
    for (a, &column_a) in order.iter().enumerate() {
        for (b, &column_b) in order.iter().enumerate().skip(a + 1) {
            if column_a.abs_diff(column_b) == b - a {
                pairs.push([a, b]);
            }
        }
    }
    pairs
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
