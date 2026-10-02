//! Wall-time benchmarks of the linear algebra: the blocked Cholesky factorization and `gemm` of
//! `n × n` matrices, on one thread and on eight.
//!
//! ```text
//! cargo bench --bench linalg
//! ```
//!
//! The module is internal to the crate, so the benchmark compiles its source as a module of its
//! own (`cargo bench` sets `cfg(test)`, which brings its test modules in without their tests).

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

#[path = "../src"]
#[expect(dead_code, unused_imports)]
mod source {
    pub mod linalg;
}

use source::linalg::blas::gemm;
use source::linalg::cholesky::cholesky;

const SIZES: [usize; 4] = [100, 500, 1000, 2000];
const THREADS: [usize; 2] = [1, 8];

// a seeded matrix with entries in [−1, 1)
fn random(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

// M Mᵀ / n + I: symmetric positive definite
fn spd(n: usize) -> Vec<f64> {
    let m = random(n * n, 1);
    let mut a = vec![0.0; n * n];
    gemm_transposed(n, &m, &mut a);
    for (i, row) in a.chunks_exact_mut(n).enumerate() {
        for x in row.iter_mut() {
            *x /= n as f64;
        }
        row[i] += 1.0;
    }
    a
}

// M Mᵀ by gemm on the transpose
fn gemm_transposed(n: usize, m: &[f64], out: &mut [f64]) {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = m[i * n + j];
        }
    }
    gemm(n, n, n, 1.0, m, &t, 0.0, out);
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn cholesky_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky");
    group.sample_size(10);
    for n in SIZES {
        let a = spd(n);
        for threads in THREADS {
            let pool = pool(threads);
            group.bench_with_input(
                BenchmarkId::new(format!("{threads}_threads"), n),
                &n,
                |b, &n| {
                    b.iter_batched_ref(
                        || a.clone(),
                        |l| pool.install(|| cholesky(l, n).unwrap()),
                        BatchSize::LargeInput,
                    )
                },
            );
        }
    }
    group.finish();
}

fn gemm_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");
    group.sample_size(10);
    for n in SIZES {
        let a = random(n * n, 2);
        let b = random(n * n, 3);
        let mut out = vec![0.0; n * n];
        for threads in THREADS {
            let pool = pool(threads);
            group.bench_with_input(
                BenchmarkId::new(format!("{threads}_threads"), n),
                &n,
                |bench, &n| {
                    bench.iter(|| pool.install(|| gemm(n, n, n, 1.0, &a, &b, 0.0, &mut out)))
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, cholesky_bench, gemm_bench);
criterion_main!(benches);
