//! The first-order methods at scale: after their first iteration, their steps allocate nothing,
//! here at a million genes (docs/optimization-plan.md, section 2.13). A test binary of its own,
//! for its counting allocator.

use genoxide::gradient::Differentiable;
use genoxide::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

// the system allocator, counting the allocations and reallocations
struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

// SAFETY: every call is passed on to the system allocator unchanged
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, passed on
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const N: usize = 1_000_000;

#[test]
fn lbfgsb_steps_allocate_nothing_after_the_first() {
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
