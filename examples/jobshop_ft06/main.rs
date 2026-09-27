//! Job shop scheduling: Fisher and Thompson's 6×6 instance (ft06), whose shortest makespan is 55.
//!
//! Six jobs each go through the six machines in their own order, and a machine does one operation
//! at a time. A permutation of the 36 operations, where operation `k` counts for job `k / 6`, is
//! read as a sequence of jobs (a permutation with repetition): the n-th time a job appears, its
//! n-th operation starts as early as its job and its machine allow (a semi-active schedule). A
//! genetic algorithm with order crossover and swap mutation searches the sequences.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the run's trace for the plot on the example's page:
//! the best schedule so far, in at most 200 generations.
//!
//! ```text
//! cargo run --release --example jobshop_ft06
//! ```

use genoxide::genome::Genome;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use serde_json::{Value, json};

const JOBS: usize = 6;
const MACHINES: usize = 6;
// each job's operations in order: (machine, duration)
const INSTANCE: [[(usize, u32); MACHINES]; JOBS] = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
];
const OPTIMUM: f64 = 55.0;

// the start time of each job's operations in the semi-active schedule of `order`
fn schedule(order: &Order) -> [[u32; MACHINES]; JOBS] {
    let mut starts = [[0; MACHINES]; JOBS];
    let mut next = [0; JOBS];
    let mut job_free = [0; JOBS];
    let mut machine_free = [0; MACHINES];
    for &operation in order.iter() {
        let job = operation / MACHINES;
        let (machine, duration) = INSTANCE[job][next[job]];
        let start = job_free[job].max(machine_free[machine]);
        starts[job][next[job]] = start;
        next[job] += 1;
        job_free[job] = start + duration;
        machine_free[machine] = start + duration;
    }
    starts
}

// when the last operation ends
fn makespan(order: &Order) -> f64 {
    let starts = schedule(order);
    let last = MACHINES - 1;
    let end = |job: usize| starts[job][last] + INSTANCE[job][last].1;
    f64::from((0..JOBS).map(end).max().unwrap_or(0))
}

fn main() -> Result<()> {
    let ga = Ga::builder(Permutation::new(JOBS * MACHINES)?)
        .population_size(100)
        .select(Tournament::new(3)?)
        .crossover(OrderCrossover)
        .mutate(SwapMutation::new())
        .minimize()
        .seed(1)
        .build()?;

    let mut trace = std::env::var("GENOXIDE_TRACE")
        .ok()
        .map(|path| (path, Trace::new(200)));
    let outcome = Engine::new(ga, makespan)
        .stop_when(Stop::target(OPTIMUM).or(Stop::generations(1_000)))
        .on_generation(|snapshot| {
            if let Some((_, trace)) = &mut trace {
                let state = json!({ "best": operations(snapshot.best().genome()) });
                trace.record(snapshot, state);
            }
        })
        .run()?;

    println!(
        "makespan {} after {} generations (the optimum: {OPTIMUM})",
        outcome.best_fitness(),
        outcome.generations()
    );
    // each machine's jobs, in the order it does them
    let starts = schedule(outcome.best_genome());
    for machine in 0..MACHINES {
        let mut jobs: Vec<(u32, usize)> = Vec::new();
        for (job, operations) in INSTANCE.iter().enumerate() {
            for (step, &(on, _)) in operations.iter().enumerate() {
                if on == machine {
                    jobs.push((starts[job][step], job));
                }
            }
        }
        jobs.sort_unstable();
        let jobs: Vec<usize> = jobs.into_iter().map(|(_, job)| job).collect();
        println!("machine {machine}: jobs {jobs:?}");
    }
    if let Some((path, trace)) = trace {
        trace.write(
            &path,
            json!({
                "format": 1,
                "example": "jobshop_ft06",
                "objective": "minimize",
                "x_label": "generations",
                "y_label": "makespan",
                "log_y": false,
                "optimum": OPTIMUM,
                "plot": "gantt",
                "problem": { "machines": MACHINES, "jobs": INSTANCE },
            }),
        );
    }
    Ok(())
}

// the operations of the schedule of `order`: [job, operation, machine, start, end]
fn operations(order: &Order) -> Vec<[u32; 5]> {
    let starts = schedule(order);
    let mut operations = Vec::new();
    for (job, steps) in INSTANCE.iter().enumerate() {
        for (step, &(machine, duration)) in steps.iter().enumerate() {
            let start = starts[job][step];
            operations.push([
                job as u32,
                step as u32,
                machine as u32,
                start,
                start + duration,
            ]);
        }
    }
    operations
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
