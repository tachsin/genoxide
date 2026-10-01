//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the best value over the evaluations of SHADE followed by L-BFGS-B, and
//! of SHADE alone. The Python example writes the same file.

use serde_json::{Value, json};

// the lines of the plot
const SERIES: [&str; 2] = ["SHADE, then L-BFGS-B", "SHADE alone"];
// a frame every this many generations of SHADE
const EVERY: usize = 4;

// writes the trace, if GENOXIDE_TRACE names a file, from `(evaluations, best value)` after each
// generation of SHADE alone, whose first `global` evaluations are the global search, and of
// L-BFGS-B, which starts after them
pub fn write_runs(alone: &[(u64, f64)], global: u64, local: &[(u64, f64)]) {
    let Ok(path) = std::env::var("GENOXIDE_TRACE") else {
        return;
    };
    let polished = local.last().map_or(f64::NAN, |&(_, value)| value);
    let mut frames = Vec::new();
    let mut frame = |evaluations: u64, pipeline: f64, shade: f64| {
        let values: serde_json::Map<String, Value> = SERIES
            .iter()
            .zip([pipeline, shade])
            .map(|(name, value)| (name.to_string(), json!(value)))
            .collect();
        frames.push(json!({
            "generation": frames.len(),
            "evaluations": evaluations,
            "best": pipeline,
            "state": { "values": values },
        }));
    };
    for (k, &(evaluations, value)) in alone.iter().enumerate() {
        let start = evaluations == global;
        if k % EVERY == 0 || k + 1 == alone.len() || start {
            let pipeline = if evaluations <= global {
                value
            } else {
                polished
            };
            frame(evaluations, pipeline, value);
        }
        if start {
            // L-BFGS-B's rounds, from SHADE's best: SHADE alone hasn't moved yet
            for &(more, polishing) in local {
                frame(global + more, polishing, value);
            }
        }
    }
    let settings = json!({
        "format": 1,
        "example": "polish",
        "objective": "minimize",
        "x_label": "evaluations",
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
