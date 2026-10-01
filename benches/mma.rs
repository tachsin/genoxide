//! The first-order methods at scale: the time of one iteration of MMA at 10³, 10⁵ and 10⁶ genes
//! with one constraint, the fitness function's evaluation included, sequentially and with the
//! dual's sums in parallel. Peak memory is checked by `tests/mma_allocations.rs`: 15 values per
//! gene besides the representation.
//!
//! ```text
//! cargo bench --bench mma
//! ```

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use genoxide::engine::{Evaluations, Provided};
use genoxide::prelude::*;
use std::hint::black_box;

// minimize Σ cⱼ/xⱼ subject to Σ xⱼ ≤ n, with cⱼ from 1 to 9
struct Volume {
    c: Vec<f64>,
    gradient: Vec<f64>,
    jacobian: Vec<f64>,
}

impl Volume {
    fn new(n: usize) -> Self {
        Self {
            c: (0..n).map(|j| 1.0 + (j % 9) as f64).collect(),
            gradient: vec![0.0; n],
            jacobian: vec![0.0; n],
        }
    }

    // one ask and its tell
    fn step(&mut self, mma: &mut Mma) {
        let n = self.c.len();
        let x = mma.ask().get(0).expect("a point");
        let (mut value, mut sum) = (0.0, 0.0);
        for j in 0..n {
            value += self.c[j] / x[j];
            self.gradient[j] = -self.c[j] / (x[j] * x[j]);
            self.jacobian[j] = 1.0;
            sum += x[j];
        }
        let g = [sum - n as f64];
        let fitness = [Fitness::constrained(value, g[0].max(0.0))];
        let evaluations = Evaluations::with_extras(
            &fitness,
            Some(&self.gradient),
            Some(&g),
            Some(&self.jacobian),
            n,
            1,
        )
        .expect("rows of the right lengths");
        mma.tell_evaluations(&evaluations).expect("a valid tell");
    }
}

// a run after its initial point and 3 iterations
fn started(n: usize, parallel: bool, volume: &mut Volume) -> Mma {
    let mut mma = Mma::builder(Real::uniform(n, 0.01..=10.0).unwrap())
        .initial_genome(Reals::from(vec![0.5; n]))
        .parallel_sums(parallel)
        .minimize()
        .build()
        .unwrap();
    mma.prepare(
        Provided::GRADIENT
            .with_inequalities(1)
            .with_constraint_jacobian(),
    )
    .unwrap();
    for _ in 0..4 {
        volume.step(&mut mma);
    }
    mma
}

fn mma_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("mma_iteration");
    group.sample_size(10);
    for (n, parallel) in [
        (1_000, false),
        (100_000, false),
        (1_000_000, false),
        (1_000_000, true),
    ] {
        let mut volume = Volume::new(n);
        let mma = started(n, parallel, &mut volume);
        let name = if parallel {
            format!("{n}_parallel")
        } else {
            n.to_string()
        };
        group.bench_function(name, |b| {
            b.iter_batched_ref(
                || mma.clone(),
                |mma| volume.step(black_box(mma)),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, mma_iteration);
criterion_main!(benches);
