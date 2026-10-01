//! Wall-time benchmarks of the first-order methods' steps at 10³, 10⁵ and 10⁶ genes: the method's
//! own work per step (an ask and a tell with a gradient already computed), without the fitness
//! function's.
//!
//! ```text
//! cargo bench --bench first_order
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use genoxide::algorithm::first_order::Step;
use genoxide::engine::Evaluations;
use genoxide::gradient::Gradients;
use genoxide::prelude::*;
use std::hint::black_box;
use std::time::Duration;

fn steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_order_step");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    for genes in [1_000, 100_000, 1_000_000] {
        let real = Real::uniform(genes, -1.0..=1.0).unwrap();
        // a gradient that keeps the points inside the bounds: small, of alternating sign
        let gradient: Vec<f64> = (0..genes)
            .map(|i| if i % 2 == 0 { 1e-6 } else { -1e-6 })
            .collect();
        let fitness = [Fitness::new(1.0)];
        for (name, step) in [
            ("gradient", Step::gradient(1e-3)),
            ("momentum", Step::momentum(1e-3, 0.9)),
            ("nesterov", Step::nesterov(1e-3, 0.9)),
            ("adam", Step::adam(1e-6)),
            ("adamw", Step::adamw(1e-6, 1e-6)),
        ] {
            let mut algorithm = FirstOrder::builder(real.clone())
                .step(step)
                .gradients(Gradients::Supplied)
                .gradient_tolerance(0.0)
                .step_tolerance(0.0)
                .minimize()
                .seed(1)
                .build()
                .unwrap();
            let evaluations = Evaluations::with_gradients(&fitness, &gradient, genes).unwrap();
            // the first step allocates the best
            for _ in 0..2 {
                algorithm.ask();
                algorithm.tell_evaluations(&evaluations).unwrap();
            }
            group.bench_function(format!("{name}_{genes}"), |b| {
                b.iter(|| {
                    black_box(algorithm.ask().len());
                    algorithm.tell_evaluations(&evaluations).unwrap();
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, steps);
criterion_main!(benches);
