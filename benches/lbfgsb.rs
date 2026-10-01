//! Wall-time benchmarks of L-BFGS-B at scale: the time of one round (an ask, the evaluation with
//! the analytic gradient, and the tell), at n = 10³, 10⁵ and 10⁶ genes with the default memory of
//! 10 pairs, on Rosenbrock's chained function, which takes thousands of iterations; and the peak
//! memory of a run, measured by the allocator (docs/optimization-plan.md, section 2.13).
//!
//! ```text
//! cargo bench --bench lbfgsb
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use genoxide::engine::{Evaluations, Extras, Provided};
use genoxide::prelude::*;
use genoxide::problems::{Problem, Rosenbrock};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// the system allocator, keeping the bytes in use and their peak
struct Peak;

static IN_USE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let now = IN_USE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every call is passed on to the system allocator unchanged
unsafe impl GlobalAlloc for Peak {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        // SAFETY: the caller's contract, passed on
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        IN_USE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        IN_USE.fetch_sub(layout.size(), Ordering::Relaxed);
        grew(new_size);
        // SAFETY: the caller's contract, passed on
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Peak = Peak;

const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

// an L-BFGS-B on Rosenbrock's function from the classic start, and the buffers of its rounds
struct Run {
    lbfgsb: Lbfgsb,
    problem: Rosenbrock,
    fitness: Vec<Fitness>,
    gradient: Vec<f64>,
}

impl Run {
    fn new(n: usize) -> Run {
        let problem = Rosenbrock::new(n);
        let start: Reals = (0..n)
            .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
            .collect();
        let mut lbfgsb = Lbfgsb::builder(problem.representation())
            .initial_genome(start)
            .gradient_tolerance(0.0)
            .function_tolerance(0.0)
            .minimize()
            .build()
            .unwrap();
        lbfgsb.prepare(Provided::GRADIENT).unwrap();
        Run {
            lbfgsb,
            problem,
            fitness: Vec::with_capacity(1),
            gradient: vec![0.0; n],
        }
    }

    // one ask / tell round
    fn round(&mut self) {
        let candidates = self.lbfgsb.ask();
        self.fitness.clear();
        if let Some(x) = candidates.get(0) {
            self.gradient.fill(0.0);
            let value = self
                .problem
                .evaluate_with(x, &mut Extras::with_gradient(&mut self.gradient));
            self.fitness.push(Fitness::new(value));
        }
        let n = self.gradient.len();
        let gradient = &self.gradient[..self.fitness.len() * n];
        let evaluations = Evaluations::with_gradients(&self.fitness, gradient, n).unwrap();
        self.lbfgsb.tell_evaluations(&evaluations).unwrap();
    }
}

fn rounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbfgsb_round");
    group.sample_size(20);
    for n in SIZES {
        // the peak memory of a run of 30 rounds, the memory full of pairs
        let before = IN_USE.load(Ordering::Relaxed);
        PEAK.store(before, Ordering::Relaxed);
        let mut run = Run::new(n);
        for _ in 0..30 {
            run.round();
        }
        let peak = PEAK.load(Ordering::Relaxed) - before;
        println!(
            "lbfgsb n = {n}: peak memory {:.1} MB ({:.1} f64 per gene), {} pairs",
            peak as f64 / 1e6,
            peak as f64 / 8.0 / n as f64,
            run.lbfgsb.pairs()
        );
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            // rounds of a run under way: a run that has converged (at 10³ genes, after a few
            // thousand rounds) is replaced, outside the timing, by a new one with its pairs
            b.iter_custom(|rounds| {
                let mut time = Duration::ZERO;
                for _ in 0..rounds {
                    if run.lbfgsb.is_finished() {
                        run = Run::new(n);
                        for _ in 0..30 {
                            run.round();
                        }
                    }
                    let start = Instant::now();
                    run.round();
                    time += start.elapsed();
                }
                time
            });
        });
    }
    group.finish();
}

criterion_group!(benches, rounds);
criterion_main!(benches);
