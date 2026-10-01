use super::*;
use crate::engine::Evaluations;

// a run on `n` genes in [−1, 2] with `m` constraints, at a point with the values and derivatives
// of a deterministic pseudo-random problem, the asymptotes of its first iteration and ρ
fn prepared(n: usize, m: usize, method: Method) -> Mma {
    let mut mma = Mma::builder(Real::uniform(n, -1.0..=2.0).unwrap())
        .method(method)
        .minimize()
        .seed(7)
        .build()
        .unwrap();
    mma.prepare(
        Provided::GRADIENT
            .with_inequalities(m)
            .with_constraint_jacobian(),
    )
    .unwrap();
    let mut rng = StreamRng::seed_from_u64(3);
    let mut uniform = |low: f64, high: f64| low + (high - low) * rng.unit_f64();
    let x: Vec<f64> = (0..n).map(|_| uniform(-0.5, 1.5)).collect();
    let gradient: Vec<f64> = (0..n).map(|_| uniform(-1.0, 1.0)).collect();
    let g: Vec<f64> = (0..m).map(|_| uniform(-0.2, 0.5)).collect();
    let jacobian: Vec<f64> = (0..m * n).map(|_| uniform(-1.0, 1.0)).collect();
    let mut x_reals = Reals::from(x);
    std::mem::swap(mma.current[0].genome_mut(), &mut x_reals);
    let fitness = [Fitness::constrained(
        1.5,
        g.iter().map(|g| g.max(0.0)).sum(),
    )];
    let evaluations =
        Evaluations::with_extras(&fitness, Some(&gradient), Some(&g), Some(&jacobian), n, m)
            .unwrap();
    mma.ask();
    mma.tell_evaluations(&evaluations).unwrap();
    mma
}

// f̃ᵢ at `x`, from the approximations around the current point: eqs. 3.2-3.5 as differences
fn approximate(mma: &Mma, i: usize, x: &[f64]) -> f64 {
    let genes: Vec<Gene> = {
        let places = mma.approximation(&[]);
        (0..mma.n()).map(|j| places.place(j)).collect()
    };
    let approximation = mma.approximation(&genes);
    let mut coefficients = vec![0.0; 2 * (mma.m() + 1)];
    let mut value = mma.values[i];
    for (j, (xj, current)) in x.iter().zip(mma.current[0].genome().iter()).enumerate() {
        let gene = approximation.gene(j);
        approximation.coefficients(j, &gene, &mut coefficients);
        let (sp, sq) = (coefficients[2 * i], coefficients[2 * i + 1]);
        let step = xj - current;
        value += step * (sp * gene.a / (gene.a - step) - sq * gene.b / (gene.b + step));
    }
    value
}

#[test]
fn the_approximations_have_the_functions_value_and_gradient() {
    let mma = prepared(6, 2, Method::Mma);
    let x = mma.current[0].genome().to_vec();
    for i in 0..3 {
        assert_eq!(approximate(&mma, i, &x), mma.values[i]);
        for j in 0..6 {
            let h = 1e-6;
            let (mut above, mut below) = (x.clone(), x.clone());
            above[j] += h;
            below[j] -= h;
            let slope = (approximate(&mma, i, &above) - approximate(&mma, i, &below)) / (2.0 * h);
            let exact = if i == 0 {
                mma.gradient[j]
            } else {
                mma.jacobian[(i - 1) * 6 + j]
            };
            assert!(
                (slope - exact).abs() < 1e-8,
                "f{i}, gene {j}: {slope} and {exact}"
            );
        }
    }
}

// the subproblem's solution: within the move limits, the multipliers of the right sign, the
// constraints of the subproblem met where a multiplier is 0 and active where it isn't, and the
// Lagrangian at its minimum in every gene inside its limits
#[test]
fn the_subproblem_is_solved() {
    for m in [0, 1, 2, 3, 6] {
        for method in [Method::Mma, Method::Gcmma] {
            let mut mma = prepared(40, m, method);
            mma.solve();
            let x = mma.trial.genome().to_vec();
            let cost = mma.settings.constraint_cost;
            for (i, &lambda) in mma.multipliers.iter().enumerate() {
                let approximation = approximate(&mma, i + 1, &x);
                assert!((approximation - mma.approximations[i + 1]).abs() < 1e-13);
                let relaxed = (lambda - cost).max(0.0) / D;
                assert!(lambda >= 0.0);
                if lambda > 0.0 {
                    assert!(
                        (approximation - relaxed).abs() < 1e-10,
                        "{m}, {i}: {approximation}"
                    );
                } else {
                    assert!(approximation <= 1e-10, "{m}, {i}: {approximation}");
                }
            }
            let lagrangian = |x: &[f64]| {
                let mut value = approximate(&mma, 0, x);
                for (i, lambda) in mma.multipliers.iter().enumerate() {
                    value += lambda * approximate(&mma, i + 1, x);
                }
                value
            };
            let genes: Vec<Gene> = {
                let places = mma.approximation(&[]);
                (0..40).map(|j| places.place(j)).collect()
            };
            let at = lagrangian(&x);
            for (j, gene) in genes.iter().enumerate() {
                let step = x[j] - mma.current[0].genome()[j];
                for h in [-1e-4, 1e-4] {
                    if step + h > gene.low && step + h < gene.high {
                        let mut moved = x.clone();
                        moved[j] += h;
                        assert!(lagrangian(&moved) >= at - 1e-12, "{m}, gene {j}");
                    }
                }
            }
        }
    }
}

// the chunks of the dual's sums in registers, for up to 4 constraints, and for any number: the
// same bits
#[test]
fn the_dual_kernels_agree_to_the_bit() {
    let mma = prepared(100, 4, Method::Mma);
    let genes: Vec<Gene> = {
        let places = mma.approximation(&[]);
        (0..100).map(|j| places.place(j)).collect()
    };
    let approximation = mma.approximation(&genes);
    let lambda = [0.3, 0.0, 1.7, 0.05];
    let (mut fixed, mut any) = (vec![0.0; 1 + 8 + 16], vec![0.0; 1 + 8 + 16]);
    approximation.dual_fixed::<4>(0..100, &lambda, &mut fixed);
    let (mut coefficients, mut slopes) = (vec![0.0; 10], vec![0.0; 4]);
    approximation.dual_any(0..100, &lambda, &mut any, &mut coefficients, &mut slopes);
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&fixed), bits(&any));
}

#[test]
fn cholesky_solves_and_rejects() {
    // [[4, 2], [2, 3]] z = [2, 1]: z = [0.5, 0]
    let mut matrix = [4.0, 2.0, 2.0, 3.0];
    let mut rhs = [2.0, 1.0];
    assert!(cholesky_solve(&mut matrix, &mut rhs, 2));
    assert_eq!(rhs, [0.5, 0.0]);
    let mut singular = [1.0, 1.0, 1.0, 1.0];
    assert!(!cholesky_solve(&mut singular, &mut [1.0, 1.0], 2));
    let mut nan = [f64::NAN];
    assert!(!cholesky_solve(&mut nan, &mut [1.0], 1));
}

// Svanberg (2007), eqs. 3.11-3.14: away from the point by 0.5 of the range in the first two
// iterations; then 0.7 of the last distance where a gene turned back, 1.2 where it went on, the
// same where it stood still, within 0.01 and 10 of the range
#[test]
fn the_asymptotes_move_as_the_notes_say() {
    let mut mma = prepared(4, 0, Method::Mma);
    let width = 3.0;
    let x = mma.current[0].genome().to_vec();
    for (j, xj) in x.iter().enumerate() {
        assert_eq!(mma.asymptotes.lower[j], xj - 0.5 * width);
        assert_eq!(mma.asymptotes.upper[j], xj + 0.5 * width);
    }
    // iteration 3, from x⁽¹⁾ = 0 and x⁽²⁾ = 1 with asymptotes 1 ∓ 1
    mma.asymptotes.iteration = 3;
    mma.asymptotes.before_previous = vec![0.0; 4];
    mma.asymptotes.previous = vec![1.0; 4];
    mma.asymptotes.lower = vec![0.0; 4];
    mma.asymptotes.upper = vec![2.0; 4];
    // gene 0 turns back, gene 1 goes on, gene 2 stands still, gene 3 goes on with a tiny step,
    // where 1.2 would leave the asymptotes within 0.01 of the range
    let now = [0.5, 1.5, 1.0, 1.0];
    mma.current[0].genome_mut().copy_from_slice(&now);
    mma.asymptotes.lower[3] = 1.0 - 0.001;
    mma.asymptotes.upper[3] = 1.0 + 1e-3;
    mma.asymptotes.before_previous[3] = 1.0 - 1e-9;
    mma.current[0].genome_mut()[3] = 1.0 + 1e-9;
    mma.update_asymptotes();
    let (lower, upper) = (&mma.asymptotes.lower, &mma.asymptotes.upper);
    assert_eq!((lower[0], upper[0]), (0.5 - 0.7, 0.5 + 0.7));
    assert_eq!((lower[1], upper[1]), (1.5 - 1.2, 1.5 + 1.2));
    assert_eq!((lower[2], upper[2]), (0.0, 2.0));
    let x3 = 1.0 + 1e-9;
    assert_eq!((lower[3], upper[3]), (x3 - 0.01 * width, x3 + 0.01 * width));
}
