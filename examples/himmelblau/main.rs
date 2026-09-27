//! Himmelblau: find the four global minima of a two-dimensional function by restarting a local
//! search from random points.
//!
//! Each search is a hill climber with Gaussian steps; it ends in the minimum whose basin it
//! started in. The known minima come from genoxide's `problems::Himmelblau`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the searches' trace for the plot on the example's
//! page: their points side by side on the contour, in at most 200 of their generations.
//!
//! ```text
//! cargo run --release --example himmelblau
//! ```

use genoxide::prelude::*;
use genoxide::problems::{Himmelblau, Problem};
use serde_json::{Value, json};

const SEARCHES: u64 = 20;

fn main() -> Result<()> {
    let problem = Himmelblau;
    let optimum = problem.optimum().expect("known");
    let minima = optimum.solutions();
    // per known minimum: the searches that ended nearest it, and the best and worst values they
    // reached
    let mut found = vec![(0, f64::INFINITY, 0.0f64); minima.len()];
    let trace = std::env::var("GENOXIDE_TRACE").ok();
    // per search, for the trace: its evaluations, best value and point after each generation
    let mut histories = Vec::new();
    for seed in 1..=SEARCHES {
        let mut history = Vec::new();
        let search = LocalSearch::builder(problem.representation())
            .neighbor(GaussianMutation::per_gene(1.0, 0.001)?)
            .neighbors(10)
            .acceptance(Acceptance::Improving)
            .minimize()
            .seed(seed)
            .build()?;
        let outcome = Engine::new(search, problem)
            .stop_when(Stop::generations(1_000))
            .on_generation(|snapshot| {
                if trace.is_some() {
                    let best = snapshot.best();
                    let value = best.fitness().and_then(Fitness::score);
                    let point = [best.genome()[0], best.genome()[1]];
                    history.push((snapshot.progress().evaluations(), value, point));
                }
            })
            .run()?;
        histories.push(history);
        let end = outcome.best_genome();
        let nearest = (0..minima.len())
            .min_by(|&a, &b| distance(end, &minima[a]).total_cmp(&distance(end, &minima[b])))
            .expect("four minima");
        let value = outcome.best_fitness().score().expect("valid");
        let (searches, best, worst) = &mut found[nearest];
        *searches += 1;
        *best = best.min(value);
        *worst = worst.max(value);
    }

    println!("{SEARCHES} local searches from random points in [-5, 5] x [-5, 5]");
    println!("minimum                  searches  values reached");
    for (minimum, (searches, best, worst)) in minima.iter().zip(found) {
        println!(
            "({:>9.6}, {:>9.6})  {searches:>8}  {} to {}",
            minimum[0],
            minimum[1],
            scientific(best),
            scientific(worst)
        );
    }
    if let Some(path) = trace {
        write_trace(&path, &histories, minima);
    }
    Ok(())
}

// a search's evaluations, best value and point after a generation
type Step = (u64, Option<f64>, [f64; 2]);

// the searches side by side, a frame per generation: their points, the best value and the median
fn write_trace(path: &str, histories: &[Vec<Step>], minima: &[Reals]) {
    let mut trace = Trace::new(200);
    let generations = histories.iter().map(Vec::len).max().unwrap_or(0);
    for generation in 0..generations {
        let searches = histories
            .iter()
            .map(|history| history[generation.min(history.len() - 1)]);
        let searches: Vec<Step> = searches.collect();
        let values: Vec<f64> = searches.iter().filter_map(|search| search.1).collect();
        let best = searches.iter().filter(|search| search.1.is_some());
        let best = best.min_by(|a, b| a.1.unwrap().total_cmp(&b.1.unwrap()));
        let points: Vec<[f64; 2]> = searches.iter().map(|search| search.2).collect();
        let frame = json!({
            "generation": generation,
            "evaluations": searches.iter().map(|search| search.0).sum::<u64>(),
            "best": best.and_then(|best| best.1),
            "median": median(values),
            "state": { "population": points, "best": best.map(|best| best.2) },
        });
        trace.push(generation as u64, frame);
    }
    let minima: Vec<&[f64]> = minima.iter().map(|minimum| &minimum[..]).collect();
    trace.write(
        path,
        json!({
            "format": 1,
            "example": "himmelblau",
            "objective": "minimize",
            "x_label": "generations",
            "y_label": "best value",
            "log_y": true,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "himmelblau",
                "bounds": [[-5.0, 5.0], [-5.0, 5.0]],
                "minima": minima,
            },
        }),
    );
}

fn distance(a: &Reals, b: &Reals) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
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
