//! The trace for the plot on the example's page, written to the file that `GENOXIDE_TRACE` names.
//! The plot draws the population on the function's contour, which needs two dimensions: the trace
//! is of a separate run of CMA-ES in 2 dimensions, with a budget of 10,000 evaluations per
//! dimension, in at most 100 frames. The Python example writes the same file.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{Problem, Schwefel2_22};
use serde_json::{Value, json};

// runs CMA-ES in 2 dimensions and writes its trace, if GENOXIDE_TRACE is set
pub fn record_small() -> Result<()> {
    let Ok(path) = std::env::var("GENOXIDE_TRACE") else {
        return Ok(());
    };
    let problem = Schwefel2_22::new(2);
    let optimum = problem.optimum().expect("known");
    let minimum = optimum.value();
    let budget = 20_000;
    let cmaes = Cmaes::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    let mut frames = Frames::new(100);
    Engine::new(cmaes, problem)
        .stop_when(Stop::target(minimum + 1e-8).or(Stop::evaluations(budget)))
        .on_generation(|snapshot| {
            let population = snapshot.population().iter().map(|x| &x.genome()[..]);
            let population: Vec<&[f64]> = population.collect();
            let best = &snapshot.best().genome()[..];
            let state = json!({ "population": population, "best": best });
            frames.push(frame(snapshot, state, minimum));
        })
        .run()?;
    let bounds = problem.representation().bounds().to_vec();
    let bounds: Vec<[f64; 2]> = bounds.iter().map(|b| [*b.start(), *b.end()]).collect();
    let minima: Vec<&[f64]> = optimum.solutions().iter().map(|x| &x[..]).collect();
    let settings = json!({
        "format": 1,
        "example": "schwefel_2_22",
        "objective": "minimize",
        "x_label": "evaluations",
        "y_label": "error",
        "log_y": true,
        "optimum": 0.0,
        "plot": "contour",
        "problem": {
            "function": "schwefel_2_22",
            "bounds": bounds,
            "minima": minima,
        },
    });
    write(&path, settings, frames.into_vec());
    Ok(())
}

// the frame of a generation: its progress, the errors of its best and of its population's median
// to `minimum` (rounding can put a solution a few ulps below it), and `state`
fn frame(snapshot: &Snapshot<'_, Reals>, state: Value, minimum: f64) -> Value {
    let progress = snapshot.progress();
    let error = |value: f64| (value - minimum).max(0.0);
    let population = snapshot.population().iter();
    let scores = population.filter_map(|individual| individual.fitness()?.score());
    json!({
        "generation": progress.generation(),
        "evaluations": progress.evaluations(),
        "best": progress.best().and_then(Fitness::score).map(error),
        "median": median(scores.collect()).map(error),
        "state": state,
    })
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
        if !generation.is_multiple_of(self.every) {
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
