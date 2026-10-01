//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: per round, the best value and the evaluations of the run with the
//! analytic gradient and of the run with forward differences. The Python example writes the same
//! file.

use serde_json::{Value, json};

// the lines of the plot: a panel per quantity, a line per source of the gradient
const SERIES: [&str; 4] = [
    "best value/analytic gradient",
    "best value/forward differences",
    "evaluations/analytic gradient",
    "evaluations/forward differences",
];

// a frame every this many rounds, and the last round of each run
const EVERY: usize = 4;

// writes the trace, if GENOXIDE_TRACE names a file: `(evaluations, best value)` per round of each
// run, every 4 rounds and the last of each run
pub fn write_runs(analytic: &[(u64, f64)], forward: &[(u64, f64)]) {
    let Ok(path) = std::env::var("GENOXIDE_TRACE") else {
        return;
    };
    let mut frames = Vec::new();
    for round in 0..analytic.len().max(forward.len()) {
        let last = round + 1 == analytic.len() || round + 1 == forward.len();
        if !round.is_multiple_of(EVERY) && !last {
            continue;
        }
        let best = |run: &[(u64, f64)]| run.get(round).map(|&(_, best)| best);
        let evaluations = |run: &[(u64, f64)]| run.get(round).map(|&(evaluations, _)| evaluations);
        let values = [
            json!(best(analytic)),
            json!(best(forward)),
            json!(evaluations(analytic)),
            json!(evaluations(forward)),
        ];
        let values: serde_json::Map<String, Value> = SERIES
            .iter()
            .zip(values)
            .map(|(name, value)| (name.to_string(), value))
            .collect();
        frames.push(json!({
            "generation": round,
            "evaluations": evaluations(analytic),
            "best": best(analytic),
            "state": { "values": values },
        }));
    }
    let settings = json!({
        "format": 1,
        "example": "lbfgsb",
        "objective": "minimize",
        "x_label": "rounds",
        "y_label": "best value",
        "log_y": true,
        "optimum": 0.0,
        "plot": "multi-curve",
        "problem": { "series": SERIES },
    });
    write_trace(&path, settings, frames);
}

// ---- the same in every example's trace ---------------------------------------------------------

// writes the settings and the frames to `path`, a frame per line
fn write_trace(path: &str, settings: Value, frames: Vec<Value>) {
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
