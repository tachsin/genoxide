//! Welded beam design: the cheapest beam welded to a support that carries 6000 lb at 14 inches,
//! subject to its weld's shear stress, its bending stress, its buckling load and its deflection.
//! A constrained continuous problem, in the two forms of the literature.
//!
//! `WeldedBeam` is the form with seven constraints (Rao, 1996, as restated by Coello Coello,
//! 2000), `WeldedBeamRagsdell` the one with five (Ragsdell and Phillips, 1976, as restated by
//! Deb, 2000). Their fitness is the cost and the constraint violation, which Deb's feasibility
//! rules compare. SHADE, a differential evolution, solves each with the same budget, and the
//! example prints the best design next to the best known cost.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the trace of the first form's run for the plot on
//! the example's page: the best design so far and its constraint violations, in at most 200
//! generations.
//!
//! ```text
//! cargo run --release --example welded_beam
//! ```

use genoxide::genome::Genome;
use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::engineering::{WeldedBeam, WeldedBeamRagsdell};
use genoxide::problems::{self, DynProblem};
use serde_json::{Value, json};

fn main() -> Result<()> {
    let forms: [Box<dyn DynProblem>; 2] = [
        problems::boxed(WeldedBeam),
        problems::boxed(WeldedBeamRagsdell),
    ];
    // the trace is of the first form's run
    let mut trace = std::env::var("GENOXIDE_TRACE")
        .ok()
        .map(|path| (path, Trace::new(200)));
    for (form, problem) in forms.iter().enumerate() {
        let best_known = problem.optimum().expect("known").value();
        let de = De::builder(problem.real()).minimize().seed(1).build()?;
        let outcome = Engine::new(de, |x: &Reals| problem.evaluate(x))
            .stop_when(Stop::evaluations(40_000))
            .on_generation(|snapshot| {
                if let Some((_, trace)) = trace.as_mut().filter(|_| form == 0) {
                    let best = snapshot.best().genome();
                    let constraints = problem.constraints(best);
                    let violations: Vec<f64> = constraints
                        .inequalities()
                        .iter()
                        .map(|&g| g.max(0.0))
                        .collect();
                    trace.record(
                        snapshot,
                        json!({ "best": &best[..], "violations": violations }),
                    );
                }
            })
            .run()?;
        let best = outcome.best_fitness();
        let x = outcome.best_genome();
        println!(
            "{}: cost {:.6}, violation {:.6} (the best known: {best_known})",
            problem.name(),
            best.score().unwrap_or(f64::NAN),
            best.violation()
        );
        println!(
            "  h {:.6}, l {:.6}, t {:.6}, b {:.6}",
            x[0], x[1], x[2], x[3]
        );
    }
    if let Some((path, trace)) = trace {
        let variable = |name, bounds| json!({ "name": name, "unit": "in", "bounds": bounds });
        let (short, long) = ([0.1, 2.0], [0.1, 10.0]);
        let constraints: Vec<String> = (1..=7).map(|g| format!("g{g}")).collect();
        trace.write(
            &path,
            json!({
                "format": 1,
                "example": "welded_beam",
                "objective": "minimize",
                "x_label": "evaluations",
                "y_label": "cost",
                "log_y": false,
                "optimum": forms[0].optimum().expect("known").value(),
                "plot": "design",
                "problem": {
                    "variables": [
                        variable("h", short),
                        variable("l", long),
                        variable("t", long),
                        variable("b", short),
                    ],
                    "constraints": constraints,
                },
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
