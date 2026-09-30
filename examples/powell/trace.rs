//! The trace for the plot on the example's page, written to the file that `GENOXIDE_TRACE` names: a
//! separate run of CMA-ES in 4 dimensions, Powell's own, with a budget of 10,000 evaluations per
//! dimension: the best point so far, each gene on its range with the minimum's value marked, and
//! the error of the best and of the population's median, in at most 100 generations. The Python
//! example writes the same file.

use genoxide::prelude::*;
use genoxide::problems::{Powell, Problem};
use serde_json::{Value, json};

// runs CMA-ES in 4 dimensions and writes its trace, if GENOXIDE_TRACE is set
pub fn record_small() -> Result<()> {
    let Ok(path) = std::env::var("GENOXIDE_TRACE") else {
        return Ok(());
    };
    let problem = Powell::new(4);
    let budget = 40_000;
    let cmaes = Cmaes::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    let mut frames = Frames::new(100);
    Engine::new(cmaes, problem)
        .stop_when(Stop::target(1e-8).or(Stop::evaluations(budget)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            let mut scores: Vec<f64> = snapshot
                .population()
                .iter()
                .filter_map(|individual| individual.fitness().and_then(Fitness::score))
                .collect();
            scores.sort_by(f64::total_cmp);
            let best = snapshot.best();
            frames.push(json!({
                "generation": progress.generation(),
                "evaluations": progress.evaluations(),
                "best": best.fitness().and_then(Fitness::score),
                "median": median(&scores),
                "state": { "best": &best.genome()[..] },
            }));
        })
        .run()?;
    let optimum = problem.optimum().expect("known");
    let variables: Vec<Value> = (1..)
        .zip(problem.representation().bounds())
        .zip(optimum.solutions()[0].iter())
        .map(|((i, bounds), &optimum)| {
            let bounds = [*bounds.start(), *bounds.end()];
            json!({ "name": format!("x{i}"), "unit": "", "bounds": bounds, "optimum": optimum })
        })
        .collect();
    let settings = json!({
        "format": 1,
        "example": "powell",
        "objective": "minimize",
        "x_label": "evaluations",
        "y_label": "error",
        "log_y": true,
        "optimum": 0.0,
        "plot": "design",
        "problem": {
            "variables": variables,
            "constraints": [],
        },
    });
    write(&path, settings, frames.into_vec());
    Ok(())
}

// the median of sorted scores, None without any
fn median(scores: &[f64]) -> Option<f64> {
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
