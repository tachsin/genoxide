//! The trace of the GPU run for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: when each batch was evaluated, in at most 64 generations.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct Trace {
    path: Option<String>,
    start: Instant,
    // the batches so far: [worker, start, end], in seconds since the run started; the GPU is
    // worker 0
    batches: Arc<Mutex<Vec<(usize, f64, f64)>>>,
    frames: Frames,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        Self {
            path: std::env::var("GENOXIDE_TRACE").ok(),
            start: Instant::now(),
            batches: Arc::default(),
            frames: Frames::new(64),
        }
    }

    // `evaluate`, recording when each batch starts and ends
    pub fn timed<F>(&self, evaluate: F) -> impl Fn(&[&Reals]) -> Vec<f64> + Send + Sync + use<F>
    where
        F: Fn(&[&Reals]) -> Vec<f64> + Send + Sync,
    {
        let (tracing, start, batches) = (self.path.is_some(), self.start, self.batches.clone());
        move |genomes: &[&Reals]| {
            let begin = start.elapsed().as_secs_f64();
            let errors = evaluate(genomes);
            if tracing {
                let end = start.elapsed().as_secs_f64();
                batches.lock().expect("the batches").push((0, begin, end));
            }
            errors
        }
    }

    // records a generation: its progress, the seconds since the start and the last 200 batches
    pub fn record(&mut self, snapshot: &Snapshot<'_, Reals>) {
        if self.path.is_none() {
            return;
        }
        let batches = self.batches.lock().expect("the batches");
        let events = &batches[batches.len().saturating_sub(200)..];
        let seconds = self.start.elapsed().as_secs_f64();
        let frame = frame(snapshot, seconds, json!({ "events": events }));
        drop(batches);
        self.frames.push(frame);
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let settings = json!({
            "format": 1,
            "example": "gpu",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "mean squared error",
            "log_y": false,
            "optimum": null,
            "plot": "timeline",
            "problem": { "workers": 1 },
        });
        write(&path, settings, self.frames.into_vec());
    }
}

// the frame of a generation: its progress, the seconds since the start, the median score of its
// population and `state`
fn frame<G: Genome>(snapshot: &Snapshot<'_, G>, seconds: f64, state: Value) -> Value {
    let progress = snapshot.progress();
    let population = snapshot.population().iter();
    let scores = population.filter_map(|individual| individual.fitness()?.score());
    json!({
        "generation": progress.generation(),
        "evaluations": progress.evaluations(),
        "seconds": seconds,
        "best": progress.best().and_then(Fitness::score),
        "median": median(scores.collect()),
        "state": state,
    })
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
            self.kept
                .retain(|(generation, _)| generation.is_multiple_of(every));
        }
    }

    fn into_vec(self) -> Vec<Value> {
        let frames = self.kept.into_iter().chain(self.last);
        frames.map(|(_, frame)| frame).collect()
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
