//! The trace of the asynchronous run for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: when each worker evaluated what, a frame per generation (the initial
//! design's size in evaluations).

use genoxide::observer::{Observer, Snapshot};
use genoxide::prelude::*;
use genoxide::problems::{Hartmann3, Problem};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::Instant;

pub struct Trace {
    path: Option<String>,
    workers: usize,
    start: Instant,
    timeline: Arc<Mutex<Timeline>>,
    frames: Vec<Value>,
}

impl Trace {
    // a trace of a run on `workers`, for the file that GENOXIDE_TRACE names, or nothing to record
    // if it isn't set
    pub fn from_env(workers: usize) -> Self {
        Self {
            path: std::env::var("GENOXIDE_TRACE").ok(),
            workers,
            start: Instant::now(),
            timeline: Arc::default(),
            frames: Vec::new(),
        }
    }

    // `fitness`, recording when each evaluation starts and ends, and on which worker
    pub fn timed<F>(&self, fitness: F) -> impl Fn(&Reals) -> f64 + Send + Sync + use<F>
    where
        F: Fn(&Reals) -> f64 + Send + Sync,
    {
        let (tracing, start, timeline) = (self.path.is_some(), self.start, self.timeline.clone());
        move |x: &Reals| {
            let begin = start.elapsed().as_secs_f64();
            let value = fitness(x);
            if tracing {
                let end = start.elapsed().as_secs_f64();
                timeline.lock().expect("the timeline").record(begin, end);
            }
            value
        }
    }

    // writes the trace, if there's one
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let settings = json!({
            "format": 1,
            "example": "bo_asynchronous",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "best value's distance above the global minimum",
            "log_y": true,
            "optimum": 0.0,
            "plot": "timeline",
            "problem": { "workers": self.workers },
        });
        write(&path, settings, self.frames);
    }
}

impl Observer<Reals> for Trace {
    // records a generation: its progress, the seconds since the start and every evaluation so far
    fn observe(&mut self, snapshot: &Snapshot<'_, Reals>) {
        if self.path.is_none() {
            return;
        }
        let timeline = self.timeline.lock().expect("the timeline");
        let progress = snapshot.progress();
        let minimum = Hartmann3.optimum().expect("known").value();
        self.frames.push(json!({
            "generation": progress.generation(),
            "evaluations": progress.evaluations(),
            "seconds": self.start.elapsed().as_secs_f64(),
            "best": progress.best().and_then(Fitness::score).map(|best| best - minimum),
            "state": { "events": timeline.events },
        }));
    }
}

// the evaluations so far: [worker, start, end], in seconds since the run started, with the workers
// numbered in the order they first evaluated
#[derive(Default)]
struct Timeline {
    events: Vec<(usize, f64, f64)>,
    workers: Vec<ThreadId>,
}

impl Timeline {
    // the evaluation that the current thread did
    fn record(&mut self, start: f64, end: f64) {
        let thread = thread::current().id();
        let worker = match self.workers.iter().position(|&worker| worker == thread) {
            Some(worker) => worker,
            None => {
                self.workers.push(thread);
                self.workers.len() - 1
            }
        };
        self.events.push((worker, start, end));
    }
}

// ---- the same in every example's trace ---------------------------------------------------------

// writes the settings and the frames to `path`, a frame per line
fn write(path: &str, settings: Value, frames: Vec<Value>) {
    let frames: Vec<String> = frames.iter().map(to_json).collect();
    let settings = to_json(&settings);
    let head = &settings[..settings.len() - 1];
    let text = format!("{head},\"frames\":[\n{}\n]}}\n", frames.join(",\n"));
    std::fs::write(path, text).expect("the trace is written");
}

// compact JSON with sorted keys, and numbers rounded to 6 significant digits and written as
// Python writes them (7542.0, 1e-08)
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
