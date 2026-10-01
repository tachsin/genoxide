//! The first-order methods, MMA and continuations of them at scale: after their first iteration,
//! their steps allocate nothing, here at a million genes (docs/optimization-plan.md, section 2.13).
//! A test binary of its own, for its counting allocator.

use genoxide::algorithm::first_order::Step;
use genoxide::algorithm::mma::Method;
use genoxide::gradient::Differentiable;
use genoxide::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

// the system allocator, counting the allocations and reallocations, and the bytes in use and
// their peak
struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static IN_USE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// the counts are global: the tests of this binary run one at a time
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn allocated(size: usize) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    let in_use = IN_USE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(in_use, Ordering::Relaxed);
}

fn freed(size: usize) {
    IN_USE.fetch_sub(size, Ordering::Relaxed);
}

// SAFETY: every call is passed on to the system allocator unchanged
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocated(layout.size());
        // SAFETY: the caller's contract, passed on
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocated(layout.size());
        // SAFETY: the caller's contract, passed on
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        freed(layout.size());
        allocated(new_size);
        // SAFETY: the caller's contract, passed on
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        freed(layout.size());
        // SAFETY: the caller's contract, passed on
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const N: usize = 1_000_000;

#[test]
fn lbfgsb_steps_allocate_nothing_after_the_first() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Σ wᵢ (xᵢ − cᵢ)² with weights from 1 to 10 and centers beyond the box for every tenth gene:
    // the steps keep bounds active, so the Cauchy point passes breakpoints and the subspace
    // step works on both the free and the bound genes
    let weight = |i: usize| 1.0 + (i % 10) as f64;
    let center = |i: usize| if i.is_multiple_of(10) { 2.0 } else { 0.5 };
    let function = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        let mut value = 0.0;
        for i in 0..x.len() {
            let d = x[i] - center(i);
            gradient[i] = 2.0 * weight(i) * d;
            value += weight(i) * d * d;
        }
        value
    });
    let lbfgsb = Lbfgsb::builder(Real::uniform(N, -1.0..=1.0).unwrap())
        .gradient_tolerance(0.0)
        .function_tolerance(0.0)
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    // the allocations up to the end of each generation, in a vector allocated before the run
    let counts = RefCell::new(Vec::with_capacity(16));
    let mut engine = Engine::new(lbfgsb, function)
        .stop_when(Stop::generations(8))
        .on_generation(|_| {
            counts
                .borrow_mut()
                .push(ALLOCATIONS.load(Ordering::Relaxed));
        });
    engine.run().unwrap();
    let lbfgsb = engine.into_algorithm();
    let counts = counts.into_inner();
    assert_eq!(counts.len(), 9);
    // steps taken, and pairs stored
    assert!(lbfgsb.iterations() >= 4, "{}", lbfgsb.iterations());
    assert!(lbfgsb.pairs() >= 4);
    // generation 1 is the first iteration; nothing is allocated after it
    assert_eq!(counts[1..], [counts[1]; 8], "{counts:?}");
}

#[test]
fn first_order_steps_allocate_nothing_after_the_first() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    const GENERATIONS: u64 = 8;
    // the sphere, its gradient supplied
    let sphere = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        let mut value = 0.0;
        for (g, &xi) in gradient.iter_mut().zip(x.iter()) {
            *g = 2.0 * xi;
            value += xi * xi;
        }
        value
    });
    for step in [
        Step::gradient(0.1),
        Step::momentum(0.1, 0.9),
        Step::nesterov(0.1, 0.9),
        Step::adam(0.01),
        Step::adamw(0.01, 0.001),
    ] {
        let real = Real::uniform(N, -1.0..=1.0).unwrap();
        // the peak from here, the representation's 24 bytes a gene aside
        let start = IN_USE.load(Ordering::Relaxed);
        PEAK.store(start, Ordering::Relaxed);
        let algorithm = FirstOrder::builder(real)
            .step(step)
            .gradient_tolerance(0.0)
            .minimize()
            .seed(1)
            .build()
            .unwrap();
        // the allocations counted at each generation's control, kept without allocating
        let mut counts = Vec::with_capacity(GENERATIONS as usize + 1);
        let outcome = Engine::new(algorithm, sphere)
            .stop_when(Stop::generations(GENERATIONS))
            .control(|_, _| {
                counts.push(ALLOCATIONS.load(Ordering::Relaxed));
                Ok(())
            })
            .run()
            .unwrap();
        let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start);
        assert_eq!(outcome.generations(), GENERATIONS);
        // generation 0 evaluates the start, and generation 1 takes the first step: from then on,
        // nothing (the outcome's copy of the best is made at the last generation, before its
        // control)
        assert_eq!(counts.len(), GENERATIONS as usize + 1);
        let steps = &counts[1..GENERATIONS as usize];
        assert!(
            steps.windows(2).all(|pair| pair[0] == pair[1]),
            "{step:?}: allocations {counts:?}"
        );
        // memory linear in the genes: 8 bytes a gene for each of the iterate, the point, the
        // asked genome, the best, the gradient, the engine's buffer of gradients and the start's
        // copy, and for the velocity or each of Adam's averages: 56 to 72 bytes a gene
        let per_gene = peak as f64 / N as f64;
        assert!(per_gene <= 72.5, "{step:?}: {per_gene} bytes a gene");
        println!("{step:?}: peak {per_gene:.1} bytes a gene");
    }
}

#[test]
fn mma_iterations_allocate_nothing_after_the_first() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    const GENERATIONS: u64 = 6;
    // minimize Σ cⱼ / xⱼ subject to Σ xⱼ ≤ n, with cⱼ from 1 to 9: one constraint, active
    let c: Vec<f64> = (0..N).map(|j| 1.0 + (j % 9) as f64).collect();
    let volume = N as f64;
    for method in [Method::Mma, Method::Gcmma] {
        let problem = Constrained::differentiable(
            1,
            |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
                let (mut value, mut sum) = (0.0, 0.0);
                for j in 0..x.len() {
                    value += c[j] / x[j];
                    gradient[j] = -c[j] / (x[j] * x[j]);
                    jacobian[j] = 1.0;
                    sum += x[j];
                }
                g[0] = sum - volume;
                value
            },
        );
        let real = Real::uniform(N, 0.01..=10.0).unwrap();
        let initial = Reals::from(vec![0.5; N]);
        // the peak from here, the representation and the initial genome aside
        let start = IN_USE.load(Ordering::Relaxed);
        PEAK.store(start, Ordering::Relaxed);
        let mma = Mma::builder(real)
            .method(method)
            .initial_genome(initial)
            .minimize()
            .build()
            .unwrap();
        // the allocations counted at each generation's control, kept without allocating
        let mut counts = Vec::with_capacity(GENERATIONS as usize + 1);
        let outcome = Engine::new(mma, problem)
            .stop_when(Stop::generations(GENERATIONS))
            .control(|_, _| {
                counts.push(ALLOCATIONS.load(Ordering::Relaxed));
                Ok(())
            })
            .run()
            .unwrap();
        let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start);
        assert_eq!(outcome.generations(), GENERATIONS);
        // generation 0 evaluates the start, and generation 1 takes the first step and sizes the
        // buffers: from then on, nothing (the outcome's copy of the best is made at the last
        // generation, before its control)
        assert_eq!(counts.len(), GENERATIONS as usize + 1);
        let steps = &counts[1..GENERATIONS as usize];
        assert!(
            steps.windows(2).all(|pair| pair[0] == pair[1]),
            "{method:?}: allocations {counts:?}"
        );
        // O(n) memory: the run's vectors, the engine's buffers of the gradient and the Jacobian,
        // and the subproblem's table of the genes: 16 values a gene
        let per_gene = peak as f64 / N as f64;
        assert!(per_gene <= 136.5, "{method:?}: {per_gene} bytes a gene");
        println!("{method:?}: peak {per_gene:.1} bytes a gene");
    }
}

#[test]
fn continuations_allocate_nothing_after_the_first_step() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // 3 stages of 4 generations, each from the last: generations 4 and 8 end a stage, and the
    // next ones start with a re-evaluation of the point on the changed function
    const STAGES: usize = 3;
    const BUDGET: u64 = 4;
    const GENERATIONS: u64 = STAGES as u64 * BUDGET;
    // the weight of the stage, shared with the fitness functions
    static WEIGHT: AtomicU64 = AtomicU64::new(0);
    let weight = |stage: usize| {
        WEIGHT.store((1.0 + stage as f64).to_bits(), Ordering::Relaxed);
        Ok(())
    };
    let weighted = || f64::from_bits(WEIGHT.load(Ordering::Relaxed));

    // Adam on w Σ xᵢ², its gradient supplied
    let sphere = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        let w = weighted();
        let mut value = 0.0;
        for (g, &xi) in gradient.iter_mut().zip(x.iter()) {
            *g = 2.0 * w * xi;
            value += w * xi * xi;
        }
        value
    });
    let adam = FirstOrder::builder(Real::uniform(N, -1.0..=1.0).unwrap())
        .step(Step::adam(0.01))
        .gradient_tolerance(0.0)
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let continuation = Continuation::builder(adam)
        .stages(STAGES)
        .generations(BUDGET)
        .on_stage(move |stage, _| weight(stage))
        .build()
        .unwrap();
    let mut counts = Vec::with_capacity(GENERATIONS as usize + 1);
    let mut engine = Engine::new(continuation, sphere)
        .stop_when(Stop::generations(10 * GENERATIONS))
        .control(|_, _| {
            counts.push(ALLOCATIONS.load(Ordering::Relaxed));
            Ok(())
        });
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(outcome.generations(), GENERATIONS);
    assert_eq!(engine.algorithm().stages().len(), STAGES);
    drop(engine);
    // the control runs once per generation, not after a re-evaluation; nothing is allocated
    // after the first step, up to the last generation (whose outcome copies the best)
    assert_eq!(counts.len(), GENERATIONS as usize + 1);
    let steps = &counts[1..GENERATIONS as usize];
    assert!(
        steps.windows(2).all(|pair| pair[0] == pair[1]),
        "Adam: allocations {counts:?}"
    );

    // MMA on Σ cⱼ / xⱼ subject to w Σ xⱼ ≤ n
    let c: Vec<f64> = (0..N).map(|j| 1.0 + (j % 9) as f64).collect();
    let volume = N as f64;
    for method in [Method::Mma, Method::Gcmma] {
        let problem = Constrained::differentiable(
            1,
            |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
                let w = weighted();
                let (mut value, mut sum) = (0.0, 0.0);
                for j in 0..x.len() {
                    value += c[j] / x[j];
                    gradient[j] = -c[j] / (x[j] * x[j]);
                    jacobian[j] = w;
                    sum += x[j];
                }
                g[0] = w * sum - volume;
                value
            },
        );
        let mma = Mma::builder(Real::uniform(N, 0.01..=10.0).unwrap())
            .method(method)
            .initial_genome(Reals::from(vec![0.5; N]))
            .minimize()
            .build()
            .unwrap();
        let continuation = Continuation::builder(mma)
            .stages(STAGES)
            .generations(BUDGET)
            .on_stage(move |stage, _| weight(stage))
            .build()
            .unwrap();
        let mut counts = Vec::with_capacity(GENERATIONS as usize + 1);
        let mut engine = Engine::new(continuation, problem)
            .stop_when(Stop::generations(10 * GENERATIONS))
            .control(|_, _| {
                counts.push(ALLOCATIONS.load(Ordering::Relaxed));
                Ok(())
            });
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.generations(), GENERATIONS, "{method:?}");
        assert_eq!(engine.algorithm().stages().len(), STAGES);
        drop(engine);
        assert_eq!(counts.len(), GENERATIONS as usize + 1);
        let steps = &counts[1..GENERATIONS as usize];
        assert!(
            steps.windows(2).all(|pair| pair[0] == pair[1]),
            "{method:?}: allocations {counts:?}"
        );
    }
}
