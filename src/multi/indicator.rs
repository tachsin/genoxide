//! Quality indicators: how good a front is, alone or compared with a reference front.
//!
//! Every indicator takes the objective values of a front, e.g. from
//! [`MultiOutcome::front_values`](super::MultiOutcome::front_values), and the direction of each
//! objective. The hypervolume needs only the front and a reference point; the others compare it
//! with a reference front, typically points sampled from the true Pareto front of a test
//! problem.
//!
//! | Indicator | Better | Measures |
//! |---|---|---|
//! | [`hypervolume`] | larger | convergence and spread, Pareto compliant |
//! | [`igd_plus`] | smaller | convergence and spread, weakly Pareto compliant |
//! | [`igd`] | smaller | convergence and spread |
//! | [`gd`] | smaller | convergence only |
//! | [`spread`] | smaller | spread only |
//!
//! The indicators only add, subtract, multiply, divide and take square roots, so they give the
//! same results on every platform.

use crate::Objective;

// the values with every objective turned into one to minimize
fn minimized<const M: usize>(point: &[f64; M], objectives: &[Objective; M]) -> [f64; M] {
    std::array::from_fn(|j| match objectives[j] {
        Objective::Minimize => point[j],
        Objective::Maximize => -point[j],
    })
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// The hypervolume of a front: the volume of the region that its points dominate, bounded by
/// the `reference` point, which every point of interest should dominate (e.g. a little worse
/// than the worst value of each objective). Points that don't dominate the reference point add
/// nothing. Larger is better, and a front that dominates another has a larger hypervolume
/// (Pareto compliance), which makes it the most widely used indicator.
///
/// The computation is exact. It takes O(N log N) time for up to 2 objectives, and slices the
/// last objective for more (HSO, O(N^(M−1) log N)): fast for 3 and 4 objectives, slow for
/// hundreds of points in 5 or more.
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::indicator::hypervolume;
///
/// let front = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]];
/// // 3 + 2 + 1 unit squares below the reference point (4, 4)
/// assert_eq!(hypervolume(&front, &[4.0, 4.0], &[Minimize, Minimize]), 6.0);
/// ```
pub fn hypervolume<const M: usize>(
    front: &[[f64; M]],
    reference: &[f64; M],
    objectives: &[Objective; M],
) -> f64 {
    let reference = minimized(reference, objectives);
    let points: Vec<Vec<f64>> = front
        .iter()
        .map(|point| minimized(point, objectives))
        .filter(|point| point.iter().zip(&reference).all(|(x, r)| x < r))
        .map(|point| point.to_vec())
        .collect();
    if points.is_empty() {
        return 0.0;
    }
    hypervolume_of(points, &reference)
}

// the hypervolume of minimized points that all dominate the reference point
fn hypervolume_of(mut points: Vec<Vec<f64>>, reference: &[f64]) -> f64 {
    let dimensions = reference.len();
    match dimensions {
        0 => 0.0,
        1 => points
            .iter()
            .map(|p| reference[0] - p[0])
            .fold(0.0, f64::max),
        2 => {
            points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
            let (mut volume, mut ceiling) = (0.0, reference[1]);
            for point in &points {
                if point[1] < ceiling {
                    volume += (reference[0] - point[0]) * (ceiling - point[1]);
                    ceiling = point[1];
                }
            }
            volume
        }
        _ => {
            // slices along the last objective: between consecutive values, the volume is the
            // thickness times the hypervolume of the points below, in one dimension fewer
            let last = dimensions - 1;
            points.sort_by(|a, b| a[last].total_cmp(&b[last]));
            let mut volume = 0.0;
            let mut below: Vec<Vec<f64>> = Vec::with_capacity(points.len());
            for (index, point) in points.iter().enumerate() {
                below.push(point[..last].to_vec());
                let top = points
                    .get(index + 1)
                    .map_or(reference[last], |next| next[last]);
                let thickness = top - point[last];
                if thickness > 0.0 {
                    volume += thickness * hypervolume_of(below.clone(), &reference[..last]);
                }
            }
            volume
        }
    }
}

/// The inverted generational distance: the mean distance from each point of the reference front
/// to the nearest point of `front`. Smaller is better; 0 means that `front` covers every point of
/// the reference front. It measures both convergence and spread, but isn't Pareto compliant: a
/// better front can have a larger IGD, see [`igd_plus`].
///
/// Infinite for an empty front, and NaN for an empty reference front. The directions don't
/// matter for distances, so IGD takes none.
///
/// ```
/// use genoxide::multi::indicator::igd;
///
/// let reference = [[0.0, 1.0], [1.0, 0.0]];
/// assert_eq!(igd(&[[0.0, 1.0], [1.0, 0.0]], &reference), 0.0);
/// assert_eq!(igd(&[[0.0, 1.0]], &reference), 2f64.sqrt() / 2.0);
/// ```
pub fn igd<const M: usize>(front: &[[f64; M]], reference_front: &[[f64; M]]) -> f64 {
    mean_nearest(reference_front, front, |a, b| distance(a, b))
}

/// The generational distance: the mean distance from each point of `front` to the nearest point
/// of the reference front. Smaller is better; it measures convergence only, so a single point on
/// the true front scores a perfect 0.
///
/// Infinite for an empty reference front, and NaN for an empty front.
pub fn gd<const M: usize>(front: &[[f64; M]], reference_front: &[[f64; M]]) -> f64 {
    mean_nearest(front, reference_front, |a, b| distance(a, b))
}

/// IGD+ (Ishibuchi et al., 2015): like [`igd`], but a point of `front` only counts as far from a
/// reference point by how much it's worse in each objective, `√Σ max(aᵢ − zᵢ, 0)²` when
/// minimizing. A front that dominates or equals every reference point scores 0, and a front that
/// dominates another never scores worse (weak Pareto compliance). Smaller is better.
///
/// Infinite for an empty front, and NaN for an empty reference front.
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::indicator::{igd, igd_plus};
///
/// let reference = [[0.0, 1.0], [1.0, 0.0]];
/// // better than the reference front: IGD sees a distance, IGD+ doesn't
/// let front = [[0.0, 0.0]];
/// assert_eq!(igd(&front, &reference), 1.0);
/// assert_eq!(igd_plus(&front, &reference, &[Minimize, Minimize]), 0.0);
/// ```
pub fn igd_plus<const M: usize>(
    front: &[[f64; M]],
    reference_front: &[[f64; M]],
    objectives: &[Objective; M],
) -> f64 {
    let front: Vec<[f64; M]> = front.iter().map(|p| minimized(p, objectives)).collect();
    let reference: Vec<[f64; M]> = reference_front
        .iter()
        .map(|p| minimized(p, objectives))
        .collect();
    mean_nearest(&reference, &front, |z, a| {
        a.iter()
            .zip(z)
            .map(|(a, z)| (a - z).max(0.0) * (a - z).max(0.0))
            .sum::<f64>()
            .sqrt()
    })
}

// the mean over `from` of the smallest distance to a point of `to`
fn mean_nearest<const M: usize>(
    from: &[[f64; M]],
    to: &[[f64; M]],
    distance: impl Fn(&[f64; M], &[f64; M]) -> f64,
) -> f64 {
    let total: f64 = from
        .iter()
        .map(|a| {
            to.iter()
                .map(|b| distance(a, b))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    total / from.len() as f64
}

/// The generalized spread Δ (Zhou et al., 2006; Deb's spread for any number of objectives): how
/// evenly `front` covers the reference front, from the distances of its points to their nearest
/// neighbors and of the reference front's extremes to `front`. 0 for a front of evenly spaced
/// points that reaches the extremes; larger is worse. It says nothing about convergence.
///
/// `Δ = (Σₘ d(eₘ, front) + Σ |d(x) − d̄|) / (Σₘ d(eₘ, front) + N d̄)`, where `eₘ` is the point of
/// the reference front that is worst in objective `m`, `d(x)` the distance of a point of `front`
/// to its nearest neighbor in `front`, and `d̄` their mean. 1 for fronts of fewer than 2 points,
/// or when every distance is 0.
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::indicator::spread;
///
/// let reference = [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]];
/// // evenly spaced from extreme to extreme
/// assert_eq!(spread(&reference, &reference, &[Minimize, Minimize]), 0.0);
/// let clustered = [[0.0, 2.0], [0.1, 1.9], [2.0, 0.0]];
/// assert!(spread(&clustered, &reference, &[Minimize, Minimize]) > 0.5);
/// ```
pub fn spread<const M: usize>(
    front: &[[f64; M]],
    reference_front: &[[f64; M]],
    objectives: &[Objective; M],
) -> f64 {
    if front.len() < 2 || reference_front.is_empty() {
        return 1.0;
    }
    let front: Vec<[f64; M]> = front.iter().map(|p| minimized(p, objectives)).collect();
    let reference: Vec<[f64; M]> = reference_front
        .iter()
        .map(|p| minimized(p, objectives))
        .collect();
    let nearest = |point: &[f64; M], skip: Option<usize>| {
        front
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != skip)
            .map(|(_, other)| distance(point, other))
            .fold(f64::INFINITY, f64::min)
    };
    let extremes: f64 = (0..M)
        .map(|m| {
            let extreme =
                reference.iter().fold(
                    &reference[0],
                    |worst, p| if p[m] > worst[m] { p } else { worst },
                );
            nearest(extreme, None)
        })
        .sum();
    let neighbors: Vec<f64> = front
        .iter()
        .enumerate()
        .map(|(index, point)| nearest(point, Some(index)))
        .collect();
    let mean = neighbors.iter().sum::<f64>() / neighbors.len() as f64;
    let deviation: f64 = neighbors.iter().map(|d| (d - mean).abs()).sum();
    let denominator = extremes + front.len() as f64 * mean;
    if denominator > 0.0 {
        (extremes + deviation) / denominator
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::{Maximize, Minimize};
    use proptest::prelude::*;

    // the hypervolume by inclusion–exclusion over every subset of points
    fn brute_force<const M: usize>(front: &[[f64; M]], reference: &[f64; M]) -> f64 {
        let n = front.len();
        let mut volume = 0.0;
        for subset in 1u32..(1 << n) {
            let mut corner = [f64::NEG_INFINITY; M];
            for (index, point) in front.iter().enumerate() {
                if subset >> index & 1 == 1 {
                    for j in 0..M {
                        corner[j] = corner[j].max(point[j]);
                    }
                }
            }
            let box_volume: f64 = (0..M)
                .map(|j| (reference[j] - corner[j]).max(0.0))
                .product();
            let sign = if subset.count_ones() % 2 == 1 {
                1.0
            } else {
                -1.0
            };
            volume += sign * box_volume;
        }
        volume
    }

    #[test]
    fn hypervolume_special_cases() {
        let min2 = [Minimize, Minimize];
        assert_eq!(hypervolume::<2>(&[], &[1.0, 1.0], &min2), 0.0);
        // on or beyond the reference point: nothing
        assert_eq!(
            hypervolume(&[[1.0, 0.0], [2.0, 2.0]], &[1.0, 1.0], &min2),
            0.0
        );
        // dominated points and duplicates add nothing
        let front = [[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]];
        assert_eq!(hypervolume(&front, &[3.0, 3.0], &min2), 4.0);
        assert_eq!(hypervolume(&[[2.0]], &[5.0], &[Minimize]), 3.0);
        // two unit cubes overlapping in half a cube
        let cubes = [[0.0, 0.0, 0.5], [0.0, 0.5, 0.0]];
        let min3 = [Minimize; 3];
        assert_eq!(hypervolume(&cubes, &[1.0; 3], &min3), 0.75);
        // maximizing: the reference point is below
        let front = [[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]];
        assert_eq!(hypervolume(&front, &[0.0, 0.0], &[Maximize, Maximize]), 6.0);
        let mixed = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]];
        let negated = [[1.0, -3.0], [2.0, -2.0], [3.0, -1.0]];
        assert_eq!(
            hypervolume(&mixed, &[4.0, 4.0], &min2),
            hypervolume(&negated, &[4.0, -4.0], &[Minimize, Maximize])
        );
    }

    #[test]
    fn distances() {
        let reference = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]];
        assert_eq!(igd(&reference, &reference), 0.0);
        assert_eq!(gd(&[[0.5, 0.5]], &reference), 0.0);
        assert!(igd(&[[0.5, 0.5]], &reference) > 0.0);
        assert_eq!(igd::<2>(&[], &reference), f64::INFINITY);
        assert!(igd(&reference, &[]).is_nan());
        assert_eq!(gd(&[[1.0, 1.0]], &[[1.0, 0.0]]), 1.0);
        // IGD+ only counts how much worse
        let min2 = [Minimize, Minimize];
        assert_eq!(igd_plus(&[[1.0, 2.0]], &[[1.0, 1.0]], &min2), 1.0);
        assert_eq!(igd_plus(&[[0.0, 2.0]], &[[1.0, 1.0]], &min2), 1.0);
        assert_eq!(igd_plus(&[[0.0, 0.0]], &reference, &min2), 0.0);
        assert_eq!(
            igd_plus(&[[2.0, 2.0]], &[[1.0, 1.0]], &[Maximize, Maximize]),
            0.0
        );
    }

    #[test]
    fn spread_special_cases() {
        let min2 = [Minimize, Minimize];
        let reference = [[0.0, 1.0], [1.0, 0.0]];
        assert_eq!(spread(&[[0.0, 1.0]], &reference, &min2), 1.0);
        assert_eq!(spread(&[[0.5, 0.5], [0.5, 0.5]], &[[0.5, 0.5]], &min2), 1.0);
        // missing an extreme
        let front = [[0.0, 1.0], [0.5, 0.5]];
        assert!(spread(&front, &reference, &min2) > 0.0);
    }

    fn any_front<const M: usize>(max: usize) -> impl Strategy<Value = Vec<[f64; M]>> {
        prop::collection::vec(
            prop::array::uniform::<_, M>((0..8).prop_map(|v| v as f64 / 2.0)),
            0..max,
        )
    }

    proptest! {
        #[test]
        fn hypervolume_matches_inclusion_exclusion(
            two in any_front::<2>(10),
            three in any_front::<3>(9),
            four in any_front::<4>(8),
        ) {
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
            let reference = |v| [v; 4];
            prop_assert!(close(hypervolume(&two, &[4.0; 2], &[Minimize; 2]), brute_force(&two, &[4.0; 2])));
            prop_assert!(close(hypervolume(&three, &[4.0; 3], &[Minimize; 3]), brute_force(&three, &[4.0; 3])));
            prop_assert!(close(hypervolume(&four, &reference(4.0), &[Minimize; 4]), brute_force(&four, &reference(4.0))));
        }

        #[test]
        fn a_dominating_front_is_never_worse(
            front in any_front::<3>(10),
            reference_front in any_front::<3>(10),
            shift in 0usize..3,
        ) {
            // improving every point of a front never lowers its hypervolume or raises its IGD+
            let better: Vec<[f64; 3]> = front
                .iter()
                .map(|p| { let mut p = *p; p[shift] -= 0.5; p })
                .collect();
            let min3 = [Minimize; 3];
            prop_assert!(hypervolume(&better, &[4.0; 3], &min3) >= hypervolume(&front, &[4.0; 3], &min3));
            if !reference_front.is_empty() {
                prop_assert!(igd_plus(&better, &reference_front, &min3) <= igd_plus(&front, &reference_front, &min3));
                // IGD+ never exceeds IGD
                prop_assert!(igd_plus(&front, &reference_front, &min3) <= igd(&front, &reference_front) + 1e-12);
            }
            let s = spread(&front, &reference_front, &min3);
            prop_assert!(s >= 0.0 && s.is_finite());
        }
    }
}
