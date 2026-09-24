//! The events of the `tracing` feature.
#![cfg(feature = "tracing")]

use genoxide::Objective::Minimize;
use genoxide::prelude::*;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

// a span or an event: its target, name or message, and fields
#[derive(Debug)]
struct Entry {
    target: String,
    name: String,
    fields: BTreeMap<String, String>,
}

#[derive(Clone, Default)]
struct Recorder {
    spans: Arc<Mutex<Vec<Entry>>>,
    events: Arc<Mutex<Vec<Entry>>>,
    ids: Arc<AtomicU64>,
}

struct Fields<'a>(&'a mut BTreeMap<String, String>);

impl Visit for Fields<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

impl Subscriber for Recorder {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, span: &Attributes<'_>) -> Id {
        let mut fields = BTreeMap::new();
        span.record(&mut Fields(&mut fields));
        self.spans.lock().unwrap().push(Entry {
            target: span.metadata().target().to_string(),
            name: span.metadata().name().to_string(),
            fields,
        });
        Id::from_u64(self.ids.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut fields = BTreeMap::new();
        event.record(&mut Fields(&mut fields));
        let name = fields.remove("message").unwrap_or_default();
        self.events.lock().unwrap().push(Entry {
            target: event.metadata().target().to_string(),
            name,
            fields,
        });
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

#[test]
fn a_run_has_a_span_and_events_per_generation() {
    let recorder = Recorder::default();
    let ga = Ga::builder(Binary::new(16).unwrap())
        .population_size(10)
        .select(Tournament::new(2).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::count(1).unwrap())
        .seed(1)
        .build()
        .unwrap();
    let outcome = tracing::subscriber::with_default(recorder.clone(), || {
        Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
            .stop_when(Stop::generations(5))
            .run()
            .unwrap()
    });

    let spans = recorder.spans.lock().unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].target, "genoxide");
    assert_eq!(spans[0].name, "run");
    assert!(spans[0].fields["algorithm"].contains("Ga<"));

    let events = recorder.events.lock().unwrap();
    assert!(events.iter().all(|event| event.target == "genoxide"));
    // generation 0 to 5, then the end
    assert_eq!(events.len(), 7);
    for (generation, event) in events[..6].iter().enumerate() {
        assert_eq!(event.name, "generation");
        assert_eq!(event.fields["generation"], generation.to_string());
        assert!(event.fields.contains_key("best"));
        assert!(!event.fields.contains_key("front"));
    }
    let end = &events[6];
    assert_eq!(end.name, "run finished");
    assert_eq!(end.fields["reason"], "Generations");
    assert_eq!(end.fields["evaluations"], outcome.evaluations().to_string());
    assert_eq!(end.fields["best"], outcome.best_fitness().to_string());
}

#[test]
fn a_multi_objective_run_reports_its_front() {
    let recorder = Recorder::default();
    let nsga2 = Nsga2::builder(Real::uniform(2, 0.0..=1.0).unwrap(), [Minimize; 2])
        .population_size(12)
        .crossover(UniformCrossover::new())
        .mutate(GaussianMutation::per_gene(0.5, 0.1).unwrap())
        .seed(2)
        .build()
        .unwrap();
    let outcome = tracing::subscriber::with_default(recorder.clone(), || {
        MultiEngine::new(nsga2, |x: &Reals| [x[0], 1.0 - x[0] + x[1]])
            .stop_when(Stop::generations(3))
            .run()
            .unwrap()
    });
    let events = recorder.events.lock().unwrap();
    assert_eq!(events.len(), 5);
    assert!(
        events
            .iter()
            .all(|event| event.fields.contains_key("front"))
    );
    assert!(
        events
            .iter()
            .all(|event| !event.fields.contains_key("best"))
    );
    assert_eq!(events[4].fields["front"], outcome.front().len().to_string());
}
