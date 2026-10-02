//! The trace of the runs for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: the best value's distance above the global minimum after each round, of
//! the search 4 points a round and of the search one point a round, which goes on for more rounds.
//! The Python example writes the same file.

use serde_json::{Value, json};

// the lines of the plot
const SERIES: [&str; 2] = ["4 points a round", "1 point a round"];

// writes the trace, if GENOXIDE_TRACE names a file, from the distances above the minimum after
// each round of both searches; a search that has stopped keeps its last value
pub fn write(batched: &[f64], single: &[f64]) {
    let Ok(path) = std::env::var("GENOXIDE_TRACE") else {
        return;
    };
    let rounds = batched.len().max(single.len());
    let at = |values: &[f64], round: usize| values[round.min(values.len() - 1)];
    let frames = (0..rounds)
        .map(|round| {
            let values: serde_json::Map<String, Value> = SERIES
                .iter()
                .zip([at(batched, round), at(single, round)])
                .map(|(name, value)| (name.to_string(), json!(value)))
                .collect();
            json!({
                "generation": round,
                "best": at(batched, round),
                "state": { "values": values },
            })
        })
        .collect();
    let settings = json!({
        "format": 1,
        "example": "bo_hartmann6",
        "objective": "minimize",
        "x_label": "rounds",
        "y_label": "best value's distance above the global minimum",
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
