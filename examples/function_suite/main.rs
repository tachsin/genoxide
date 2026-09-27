//! Function suite: CMA-ES, SHADE and PSO on twelve classic test functions in 10 dimensions.
//!
//! Each algorithm has a budget of 10,000 evaluations per dimension and stops early within 1e-8
//! of the known minimum. The table gives the error to the minimum: the best value found minus
//! the minimum. The functions, their bounds and their minima come from genoxide's `problems`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes the runs' trace for the plot on the example's
//! page: the error of each function and algorithm after every 1,000 evaluations.
//!
//! ```text
//! cargo run --release --example function_suite
//! ```

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{self, DynProblem};
use serde_json::{Map, Value, json};

const DIMENSIONS: usize = 10;
const BUDGET: u64 = 10_000 * DIMENSIONS as u64;
// the frames of the trace, evenly spaced over the budget
const FRAMES: u64 = 100;

fn main() -> Result<()> {
    let functions: Vec<Box<dyn DynProblem>> = vec![
        problems::boxed(problems::Sphere::new(DIMENSIONS)),
        problems::boxed(problems::AxisParallelEllipsoid::new(DIMENSIONS)),
        problems::boxed(problems::Schwefel1_2::new(DIMENSIONS)),
        problems::boxed(problems::Zakharov::new(DIMENSIONS)),
        problems::boxed(problems::Rosenbrock::new(DIMENSIONS)),
        problems::boxed(problems::Rastrigin::new(DIMENSIONS)),
        problems::boxed(problems::Ackley::new(DIMENSIONS)),
        problems::boxed(problems::Griewank::new(DIMENSIONS)),
        problems::boxed(problems::Schwefel2_26::new(DIMENSIONS)),
        problems::boxed(problems::Levy::new(DIMENSIONS)),
        problems::boxed(problems::StyblinskiTang::new(DIMENSIONS)),
        problems::boxed(problems::Michalewicz::new(DIMENSIONS)),
    ];

    println!("Error to the minimum in {DIMENSIONS} dimensions, {BUDGET} evaluations at most");
    println!(
        "{:<22}{:>10}{:>10}{:>10}",
        "function", "CMA-ES", "SHADE", "PSO"
    );
    let trace = std::env::var("GENOXIDE_TRACE").ok();
    // per function and algorithm, for the trace: the evaluations and the error after each
    // generation
    let mut series = Vec::new();
    for problem in &functions {
        let minimum = problem.optimum().expect("known").value();
        let stop = || Stop::target(minimum + 1e-8).or(Stop::evaluations(BUDGET));
        let fitness = |x: &Reals| problem.evaluate(x);
        let mut histories = [Vec::new(), Vec::new(), Vec::new()];
        let [cmaes_errors, shade_errors, pso_errors] = &mut histories;
        let tracing = trace.is_some();

        let cmaes = Cmaes::builder(problem.real())
            .restarts(cmaes::Restarts::Ipop)
            .minimize()
            .seed(1)
            .build()?;
        let cmaes = Engine::new(cmaes, fitness)
            .stop_when(stop())
            .on_generation(errors(cmaes_errors, minimum, tracing))
            .run()?;

        let shade = De::builder(problem.real()).minimize().seed(1).build()?;
        let shade = Engine::new(shade, fitness)
            .stop_when(stop())
            .on_generation(errors(shade_errors, minimum, tracing))
            .run()?;

        let pso = Pso::builder(problem.real())
            .population_size(40)
            .minimize()
            .seed(1)
            .build()?;
        let pso = Engine::new(pso, fitness)
            .stop_when(stop())
            .on_generation(errors(pso_errors, minimum, tracing))
            .run()?;

        let error = |outcome: &Outcome<Reals>| {
            let best = outcome.best_fitness().score().expect("valid");
            // rounding can put a solution a few ulps below the minimum
            scientific((best - minimum).max(0.0))
        };
        println!(
            "{:<22}{:>10}{:>10}{:>10}",
            problem.name(),
            error(&cmaes),
            error(&shade),
            error(&pso)
        );
        for (algorithm, history) in ["CMA-ES", "SHADE", "PSO"].into_iter().zip(histories) {
            series.push((format!("{}/{algorithm}", problem.name()), history));
        }
    }
    if let Some(path) = trace {
        write_trace(&path, &series);
    }
    Ok(())
}

// records the error to the minimum after each generation, for the trace
fn errors(
    history: &mut Vec<(u64, f64)>,
    minimum: f64,
    tracing: bool,
) -> impl FnMut(&Snapshot<'_, Reals>) + '_ {
    move |snapshot| {
        let progress = snapshot.progress();
        if let Some(best) = progress.best().and_then(Fitness::score).filter(|_| tracing) {
            history.push((progress.evaluations(), (best - minimum).max(0.0)));
        }
    }
}

// the runs side by side, a frame per 1,000 evaluations: each run's error after its last
// generation within them
fn write_trace(path: &str, series: &[(String, Vec<(u64, f64)>)]) {
    let frame = |evaluations: u64| {
        let mut values = Map::new();
        for (name, history) in series {
            let done = history.iter().rev().find(|(done, _)| *done <= evaluations);
            values.insert(name.clone(), json!(done.map(|(_, error)| error)));
        }
        json!({
            "generation": null,
            "evaluations": evaluations,
            "best": null,
            "median": null,
            "state": { "values": values },
        })
    };
    let frames: Vec<String> = (1..=FRAMES)
        .map(|k| to_json(&frame(k * BUDGET / FRAMES)))
        .collect();
    let names: Vec<&String> = series.iter().map(|(name, _)| name).collect();
    let settings = to_json(&json!({
        "format": 1,
        "example": "function_suite",
        "objective": "minimize",
        "x_label": "evaluations",
        "y_label": "error to the minimum",
        "log_y": true,
        "optimum": 0.0,
        "plot": "multi-curve",
        "problem": { "series": names },
    }));
    let head = &settings[..settings.len() - 1];
    let text = format!("{head},\"frames\":[\n{}\n]}}\n", frames.join(",\n"));
    std::fs::write(path, text).expect("the trace is written");
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}

// ---- the trace of the runs, for the plot on the example's page ---------------------------------

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
