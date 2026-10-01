// The compact representation, the Cauchy point and the subspace step against dense computations:
// the BFGS updates of θI one pair at a time, the piecewise quadratic along the projected path
// segment by segment, and the reduced Newton step by Gaussian elimination.

use super::*;
use crate::StreamRng;
use rand::RngExt;

fn random(rng: &mut StreamRng, n: usize, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|_| scale * (2.0 * rng.random::<f64>() - 1.0))
        .collect()
}

// a symmetric positive definite matrix A = Rᵀ R + I
fn spd(rng: &mut StreamRng, n: usize) -> Vec<f64> {
    let r = random(rng, n * n, 1.0);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (0..n).map(|k| r[k * n + i] * r[k * n + j]).sum::<f64>();
        }
        a[i * n + i] += 1.0;
    }
    a
}

fn times(a: &[f64], v: &[f64]) -> Vec<f64> {
    let n = v.len();
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * v[j]).sum())
        .collect()
}

// `count` pairs (s, A s) of a quadratic with Hessian A, in a memory of `capacity`
fn memory_of(rng: &mut StreamRng, a: &[f64], n: usize, capacity: usize, count: usize) -> Memory {
    let mut memory = Memory::new(n, capacity);
    let (mut x, mut g) = (vec![0.0; n], vec![0.0; n]);
    for _ in 0..count {
        let s = random(rng, n, 1.0);
        let y = times(a, &s);
        let x_new: Vec<f64> = x.iter().zip(&s).map(|(x, s)| x + s).collect();
        let g_new: Vec<f64> = g.iter().zip(&y).map(|(g, y)| g + y).collect();
        assert!(memory.update(&x_new, &x, &g_new, &g, f64::EPSILON));
        x = x_new;
        g = g_new;
    }
    memory
}

// B by the BFGS formula from θI, one stored pair at a time, oldest first
fn dense_b(memory: &Memory, n: usize) -> Vec<f64> {
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        b[i * n + i] = memory.theta();
    }
    for k in 0..memory.len() {
        let (s, y) = (memory.s(k).to_vec(), memory.y(k).to_vec());
        let bs = times(&b, &s);
        let sbs: f64 = s.iter().zip(&bs).map(|(s, b)| s * b).sum();
        let ys: f64 = y.iter().zip(&s).map(|(y, s)| y * s).sum();
        for i in 0..n {
            for j in 0..n {
                b[i * n + j] += y[i] * y[j] / ys - bs[i] * bs[j] / sbs;
            }
        }
    }
    b
}

// B v from the compact form: θ v − W M Wᵀ v
fn compact_times(memory: &Memory, workspace: &mut Workspace, v: &[f64]) -> Vec<f64> {
    let len = memory.len();
    let mut wtv = vec![0.0; 2 * len];
    memory.w_transpose(v, &mut wtv);
    let mut mwtv = vec![0.0; 2 * len];
    workspace.factor_middle(memory).unwrap();
    Workspace::apply_middle(memory, &workspace.middle, &wtv, &mut mwtv);
    (0..v.len())
        .map(|i| memory.theta() * v[i] - memory.w_times_at(i, &mwtv))
        .collect()
}

fn close(a: &[f64], b: &[f64], tolerance: f64) {
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.0),
            "{i}: {a} against {b}"
        );
    }
}

#[test]
fn the_compact_form_is_the_bfgs_matrix() {
    let mut rng = StreamRng::seed_from_u64(1);
    let n = 7;
    let a = spd(&mut rng, n);
    // fewer pairs than the capacity, as many, and more: the oldest dropped
    for count in [1, 3, 4, 9] {
        let memory = memory_of(&mut rng, &a, n, 4, count);
        assert_eq!(memory.len(), count.min(4));
        let b = dense_b(&memory, n);
        let mut workspace = Workspace::default();
        for _ in 0..3 {
            let v = random(&mut rng, n, 1.0);
            close(
                &compact_times(&memory, &mut workspace, &v),
                &times(&b, &v),
                1e-10,
            );
        }
    }
}

#[test]
fn the_inner_products_follow_the_ring() {
    let mut rng = StreamRng::seed_from_u64(2);
    let n = 5;
    let a = spd(&mut rng, n);
    let memory = memory_of(&mut rng, &a, n, 3, 8);
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(memory.sy(i, j), dot(memory.s(i), memory.y(j)));
            assert_eq!(memory.ss(i, j), dot(memory.s(i), memory.s(j)));
            assert_eq!(memory.yy(i, j), dot(memory.y(i), memory.y(j)));
        }
    }
    let newest = memory.len() - 1;
    assert_eq!(
        memory.theta(),
        dot(memory.y(newest), memory.y(newest)) / dot(memory.s(newest), memory.y(newest))
    );
    // a smaller memory keeps the newest pairs
    let smaller = memory.with_capacity(2);
    assert_eq!(smaller.len(), 2);
    assert_eq!(smaller.s(1), memory.s(2));
    assert_eq!(smaller.sy(0, 1), memory.sy(1, 2));
    assert_eq!(smaller.theta(), memory.theta());
    let larger = memory.with_capacity(5);
    assert_eq!(larger.len(), 3);
    assert_eq!(larger.y(0), memory.y(0));
}

#[test]
fn a_pair_without_curvature_is_skipped() {
    let n = 2;
    let mut memory = Memory::new(n, 2);
    let zero = [0.0; 2];
    // sᵀy = 0: skipped, and nothing stored
    assert!(!memory.update(&[1.0, 0.0], &zero, &[0.0, 1.0], &zero, f64::EPSILON));
    // sᵀy < 0
    assert!(!memory.update(&[1.0, 0.0], &zero, &[-1.0, 0.0], &zero, f64::EPSILON));
    // y = 0
    assert!(!memory.update(&[1.0, 0.0], &zero, &zero, &zero, f64::EPSILON));
    assert_eq!(memory.len(), 0);
    assert_eq!(memory.theta(), 1.0);
    assert!(memory.update(&[1.0, 0.0], &zero, &[2.0, 0.0], &zero, f64::EPSILON));
    assert_eq!(memory.theta(), 2.0);
    // a full memory keeps its oldest pair when the new one is skipped
    assert!(memory.update(
        &[1.0, 1.0],
        &[1.0, 0.0],
        &[2.0, 3.0],
        &[2.0, 0.0],
        f64::EPSILON
    ));
    assert!(!memory.update(
        &[2.0, 1.0],
        &[1.0, 1.0],
        &[2.0, 3.0],
        &[2.0, 3.0],
        f64::EPSILON
    ));
    assert_eq!(memory.s(0), [1.0, 0.0]);
    assert_eq!(memory.len(), 2);
}

// the generalized Cauchy point the long way: the first local minimizer of m(x(t)) along the
// projected path, segment by segment with the dense B
fn dense_cauchy(b: &[f64], x: &[f64], g: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    let n = x.len();
    let breakpoint = |i: usize| {
        if g[i] < 0.0 {
            (x[i] - upper[i]) / g[i]
        } else if g[i] > 0.0 {
            (x[i] - lower[i]) / g[i]
        } else {
            f64::INFINITY
        }
    };
    let path = |t: f64| -> Vec<f64> {
        (0..n)
            .map(|i| (x[i] - t.min(breakpoint(i)) * g[i]).clamp(lower[i], upper[i]))
            .collect()
    };
    let mut times_list: Vec<f64> = (0..n).map(breakpoint).filter(|t| t.is_finite()).collect();
    times_list.push(0.0);
    times_list.sort_by(f64::total_cmp);
    times_list.dedup();
    times_list.push(f64::INFINITY);
    for pair in times_list.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        // the direction on the segment
        let d: Vec<f64> = (0..n)
            .map(|i| if breakpoint(i) > start { -g[i] } else { 0.0 })
            .collect();
        let z: Vec<f64> = path(start).iter().zip(x).map(|(p, x)| p - x).collect();
        let bz = times(b, &z);
        let bd = times(b, &d);
        let f1: f64 = (0..n).map(|i| g[i] * d[i] + d[i] * bz[i]).sum();
        let f2: f64 = (0..n).map(|i| d[i] * bd[i]).sum();
        if f1 >= 0.0 {
            return path(start);
        }
        if f2 > 0.0 && start + -f1 / f2 < end {
            return path(start + -f1 / f2);
        }
        if end.is_infinite() {
            return path(start);
        }
    }
    unreachable!()
}

#[test]
fn the_cauchy_point_is_the_first_minimizer_along_the_projected_path() {
    let mut rng = StreamRng::seed_from_u64(3);
    let n = 8;
    for trial in 0..40 {
        let a = spd(&mut rng, n);
        let pairs = trial % 4;
        let memory = memory_of(&mut rng, &a, n, 3, pairs);
        let b = dense_b(&memory, n);
        // boxes of every width: the path bends at a few bounds or many
        let width = [0.2, 1.0, 5.0, 100.0][trial % 4];
        let lower: Vec<f64> = random(&mut rng, n, 1.0).iter().map(|l| l - width).collect();
        let upper: Vec<f64> = lower.iter().map(|l| l + 2.0 * width).collect();
        let mut x: Vec<f64> = (0..n)
            .map(|i| lower[i] + rng.random::<f64>() * (upper[i] - lower[i]))
            .collect();
        // a gene at a bound, its gradient pointing out
        x[0] = upper[0];
        let mut g = random(&mut rng, n, 3.0);
        g[0] = -1.0;
        let fixed = vec![false; n];
        let mut workspace = Workspace::default();
        if memory.len() > 0 {
            workspace.factor_middle(&memory).unwrap();
        }
        let mut xc = vec![0.0; n];
        workspace.cauchy_point(&memory, &x, &g, &lower, &upper, &fixed, &mut xc);
        close(&xc, &dense_cauchy(&b, &x, &g, &lower, &upper), 1e-9);
        assert!(workspace.active[0]);
        // c = Wᵀ(xc − x)
        let z: Vec<f64> = xc.iter().zip(&x).map(|(c, x)| c - x).collect();
        let mut c = vec![0.0; 2 * memory.len()];
        memory.w_transpose(&z, &mut c);
        close(&workspace.c, &c, 1e-9);
    }
}

// solves the dense system A x = b by Gaussian elimination with partial pivoting
fn solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .unwrap();
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        b.swap(k, pivot);
        for i in k + 1..n {
            let factor = a[i * n + k] / a[k * n + k];
            for j in k..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|j| a[i * n + j] * x[j]).sum();
        x[i] = (b[i] - sum) / a[i * n + i];
    }
    x
}

#[test]
fn the_subspace_step_minimizes_the_model_over_the_free_genes() {
    let mut rng = StreamRng::seed_from_u64(4);
    let n = 9;
    let mut checked = 0;
    for trial in 0..24 {
        let a = spd(&mut rng, n);
        let memory = memory_of(&mut rng, &a, n, 4, 1 + trial % 6);
        let b = dense_b(&memory, n);
        // a narrow box on the first genes, so the Cauchy point holds a few or many at a bound,
        // and a wide one on the rest, so the subspace minimizer is inside it
        let narrow = [2, 6][trial % 2];
        let lower: Vec<f64> = (0..n)
            .map(|i| if i < narrow { -0.05 } else { -1e6 })
            .collect();
        let upper: Vec<f64> = lower.iter().map(|l| -l).collect();
        let x = vec![0.0; n];
        let g = random(&mut rng, n, 2.0);
        let fixed = vec![false; n];
        let mut workspace = Workspace::default();
        workspace.factor_middle(&memory).unwrap();
        let mut xc = vec![0.0; n];
        workspace.cauchy_point(&memory, &x, &g, &lower, &upper, &fixed, &mut xc);
        let free: Vec<usize> = (0..n).filter(|&i| !workspace.active[i]).collect();
        let mut xbar = vec![0.0; n];
        workspace
            .subspace_step(&memory, &x, &g, &lower, &upper, &fixed, &xc, &mut xbar)
            .unwrap();
        // the minimizer of m over xc + Z d: B̂ d = −Zᵀ(g + B(xc − x))
        let z: Vec<f64> = xc.iter().zip(&x).map(|(c, x)| c - x).collect();
        let bz = times(&b, &z);
        let t = free.len();
        let reduced: Vec<f64> = (0..t * t)
            .map(|k| b[free[k / t] * n + free[k % t]])
            .collect();
        let rhs: Vec<f64> = free.iter().map(|&i| -(g[i] + bz[i])).collect();
        let d = solve(reduced, rhs);
        let inside = free
            .iter()
            .enumerate()
            .all(|(k, &i)| lower[i] < xc[i] + d[k] && xc[i] + d[k] < upper[i]);
        if inside {
            let mut expected = xc.clone();
            for (k, &i) in free.iter().enumerate() {
                expected[i] = xc[i] + d[k];
            }
            close(&xbar, &expected, 1e-8);
            checked += 1;
        }
        for i in 0..n {
            assert!(lower[i] <= xbar[i] && xbar[i] <= upper[i]);
            if workspace.active[i] {
                assert_eq!(xbar[i], xc[i]);
            }
        }
    }
    assert!(checked >= 12, "{checked}");
}

#[test]
fn the_heap_gives_the_breakpoints_in_order() {
    let mut rng = StreamRng::seed_from_u64(5);
    let mut points: Vec<(f64, usize)> = (0..200)
        .map(|i| ((rng.random::<f64>() * 10.0).floor(), i))
        .collect();
    let mut expected = points.clone();
    expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    build_heap(&mut points);
    let mut popped = Vec::new();
    while let Some(point) = pop_heap(&mut points) {
        popped.push(point);
    }
    assert_eq!(popped, expected);
}
