//! Asynchronous evaluation for a fitness function that takes a varying time, like a simulation:
//! a generational GA evaluating in parallel waits for the slowest evaluation of every generation,
//! while a steady-state GA with asynchronous evaluation gives every worker a new genome as soon as
//! it's done.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the steady-state run's trace for the plot on the
//! example's page: when each worker evaluated what, in at most 64 generations.
//!
//! cargo run --release --example asynchronous

use genoxide::observer::{Observer, Snapshot};
use genoxide::prelude::*;
use serde_json::{Value, json};
use std::f64::consts::TAU;
use std::sync::Mutex;
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

const DIMENSIONS: usize = 6;
const EVALUATIONS: u64 = 2_000;

// Rastrigin, taking 1 to 8 ms depending on the genome
fn simulation(x: &Reals) -> f64 {
    let value = 10.0 * x.len() as f64
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (TAU * xi).cos())
            .sum::<f64>();
    let millis = 1 + (x[0].to_bits() % 8);
    std::thread::sleep(Duration::from_millis(millis));
    value
}

fn main() -> genoxide::Result<()> {
    let workers = rayon::current_num_threads();
    let real = || Real::uniform(DIMENSIONS, -5.12..=5.12);
    println!("{EVALUATIONS} evaluations of 1 to 8 ms, {workers} at a time\n");

    let ga = Ga::builder(real()?)
        .population_size(40)
        .select(Tournament::new(3)?)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / DIMENSIONS as f64, 20.0)?)
        .minimize()
        .seed(1)
        .build()?;
    let start = Instant::now();
    let outcome = Engine::new(ga, simulation)
        .parallel(true)
        .stop_when(Stop::evaluations(EVALUATIONS))
        .run()?;
    report("generational, parallel", &outcome, start.elapsed());

    let steady = Ga::builder(real()?)
        .population_size(40)
        .select(Tournament::new(3)?)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / DIMENSIONS as f64, 20.0)?)
        .minimize()
        .seed(1)
        .build_steady()?;
    let start = Instant::now();
    let trace = std::env::var("GENOXIDE_TRACE").ok();
    let timeline = Mutex::new(Timeline::default());
    let mut frames = Frames {
        trace: Trace::new(64),
        timeline: &timeline,
        start,
    };
    let outcome = AsyncEngine::new(steady, |x: &Reals| {
        let begin = start.elapsed().as_secs_f64();
        let value = simulation(x);
        if trace.is_some() {
            let end = start.elapsed().as_secs_f64();
            timeline.lock().expect("the timeline").record(begin, end);
        }
        value
    })
    .workers(workers)
    .stop_when(Stop::evaluations(EVALUATIONS))
    .observe(&mut frames)
    .run()?;
    report("steady-state, asynchronous", &outcome, start.elapsed());
    if let Some(path) = trace {
        frames.trace.write(
            &path,
            json!({
                "format": 1,
                "example": "asynchronous",
                "objective": "minimize",
                "x_label": "evaluations",
                "y_label": "best value",
                "log_y": false,
                "optimum": 0.0,
                "plot": "timeline",
                "problem": { "workers": workers },
            }),
        );
    }
    Ok(())
}

// ---- the trace of the steady-state run, for the plot on the example's page ----------------------

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

// a frame per generation: its progress, the seconds since the start and the last 200 evaluations
struct Frames<'a> {
    trace: Trace,
    timeline: &'a Mutex<Timeline>,
    start: Instant,
}

impl Observer<Reals> for Frames<'_> {
    fn observe(&mut self, snapshot: &Snapshot<'_, Reals>) {
        let progress = snapshot.progress();
        let population = snapshot.population().iter();
        let scores = population.filter_map(|individual| individual.fitness()?.score());
        let timeline = self.timeline.lock().expect("the timeline");
        let events = &timeline.events[timeline.events.len().saturating_sub(200)..];
        let frame = json!({
            "generation": progress.generation(),
            "evaluations": progress.evaluations(),
            "seconds": self.start.elapsed().as_secs_f64(),
            "best": progress.best().and_then(Fitness::score),
            "median": median(scores.collect()),
            "state": { "events": events },
        });
        self.trace.push(progress.generation(), frame);
    }
}

fn report(name: &str, outcome: &Outcome<Reals>, elapsed: Duration) {
    println!(
        "{name:>27}: {:.2} s, {:.0} evaluations/s, best {:.3}",
        elapsed.as_secs_f64(),
        outcome.evaluations() as f64 / elapsed.as_secs_f64(),
        outcome.best_fitness()
    );
}

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
