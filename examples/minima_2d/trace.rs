//! The trace of the searches for the plot on the example's page, written to the file that
//! `GENOXIDE_TRACE` names: a contour panel per function, with its searches' points side by side,
//! and each function's error to its global minimum, in at most 100 of their generations. The
//! Python example writes the same file.

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::Optimum;
use serde_json::{Map, Value, json};
use std::ops::RangeInclusive;

// a search's evaluations, and its best value and point, after a generation
type Step = (u64, Option<f64>, [f64; 2]);

// a function's panel: its title, the contour plot's problem, its global minimum, and its searches
struct Panel {
    title: String,
    problem: Value,
    minimum: f64,
    searches: Vec<Vec<Step>>,
}

pub struct Trace {
    path: Option<String>,
    panels: Vec<Panel>,
}

impl Trace {
    // a trace for the file that GENOXIDE_TRACE names, or nothing to record if it isn't set
    pub fn from_env() -> Self {
        let path = std::env::var("GENOXIDE_TRACE").ok();
        Self {
            path,
            panels: Vec::new(),
        }
    }

    // starts the panel of a function, whose searches `record` records next: its title, its name
    // for the contour plot, its bounds and its global minima
    pub fn function(
        &mut self,
        title: &str,
        function: &str,
        bounds: &[RangeInclusive<f64>],
        optimum: &Optimum<Reals>,
    ) {
        let bounds: Vec<[f64; 2]> = bounds.iter().map(|b| [*b.start(), *b.end()]).collect();
        let minima: Vec<&[f64]> = optimum.solutions().iter().map(|x| &x[..]).collect();
        self.panels.push(Panel {
            title: title.to_string(),
            problem: json!({ "function": function, "bounds": bounds, "minima": minima }),
            minimum: optimum.value(),
            searches: Vec::new(),
        });
    }

    // records a generation of a search, which starts at generation 0: its evaluations, and its
    // best value and point so far
    pub fn record(&mut self, snapshot: &Snapshot<'_, Reals>) {
        if self.path.is_none() {
            return;
        }
        let progress = snapshot.progress();
        let panel = self.panels.last_mut().expect("a function");
        if progress.generation() == 0 {
            panel.searches.push(Vec::new());
        }
        let best = snapshot.best();
        let value = best.fitness().and_then(Fitness::score);
        let point = [best.genome()[0], best.genome()[1]];
        let search = panel.searches.last_mut().expect("a search");
        search.push((progress.evaluations(), value, point));
    }

    // writes the trace, if there's one: the functions' searches side by side, a frame per
    // generation, with each panel's points and best point, and each function's error
    pub fn write(self) {
        let Some(path) = self.path else { return };
        let mut frames = Frames::new(100);
        let searches = self.panels.iter().flat_map(|panel| &panel.searches);
        let generations = searches.map(Vec::len).max().unwrap_or(0);
        for generation in 0..generations {
            let mut evaluations = 0;
            let mut series = Map::new();
            let mut panels = Vec::new();
            for panel in &self.panels {
                let searches = panel.searches.iter();
                let searches = searches.map(|search| search[generation.min(search.len() - 1)]);
                let searches: Vec<Step> = searches.collect();
                evaluations += searches.iter().map(|search| search.0).sum::<u64>();
                let best = searches.iter().filter(|search| search.1.is_some());
                let best = best.min_by(|a, b| a.1.unwrap().total_cmp(&b.1.unwrap()));
                // rounding can put a solution a few ulps below the minimum
                let error = best.and_then(|best| best.1);
                let error = error.map(|value| (value - panel.minimum).max(0.0));
                series.insert(panel.title.clone(), json!(error));
                let points: Vec<[f64; 2]> = searches.iter().map(|search| search.2).collect();
                panels.push(json!({ "population": points, "best": best.map(|best| best.2) }));
            }
            frames.push(json!({
                "generation": generation,
                "evaluations": evaluations,
                "series": series,
                "state": { "panels": panels },
            }));
        }
        let titles: Vec<&String> = self.panels.iter().map(|panel| &panel.title).collect();
        let panels = self.panels.iter();
        let panels = panels.map(|panel| json!({ "title": panel.title, "problem": panel.problem }));
        let settings = json!({
            "format": 1,
            "example": "minima_2d",
            "objective": "minimize",
            "optimum": null,
            "x_label": "generations",
            "y_label": "error to the global minimum",
            "log_y": true,
            "plot": "grid",
            "problem": {
                "panel_plot": "contour",
                "panels": panels.collect::<Vec<Value>>(),
                "series": titles,
            },
        });
        write(&path, settings, frames.into_vec());
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
        if generation % self.every != 0 {
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
