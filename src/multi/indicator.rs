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

/// The exclusive hypervolume contribution of each point of a front, in its order: the volume
/// that it dominates and no other point does, bounded by the `reference` point. A dominated point
/// (or one of two equal points) contributes 0, as does a point that doesn't dominate the reference
/// point. SMS-EMOA removes the smallest contributor.
///
/// It takes O(N log N) time for up to 2 objectives, and slices the last objective for more:
/// O(N²) for 3, O(N³) for 4.
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::indicator::hypervolume_contributions;
///
/// let front = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [3.0, 3.0]];
/// let contributions = hypervolume_contributions(&front, &[4.0, 4.0], &[Minimize, Minimize]);
/// // each of the first three alone dominates a unit square; (3, 3) is dominated
/// assert_eq!(contributions, [1.0, 1.0, 1.0, 0.0]);
/// ```
pub fn hypervolume_contributions<const M: usize>(
    front: &[[f64; M]],
    reference: &[f64; M],
    objectives: &[Objective; M],
) -> Vec<f64> {
    let reference = minimized(reference, objectives);
    let inside: Vec<usize> = (0..front.len())
        .filter(|&i| {
            minimized(&front[i], objectives)
                .iter()
                .zip(&reference)
                .all(|(x, r)| x < r)
        })
        .collect();
    let points: Vec<Vec<f64>> = inside
        .iter()
        .map(|&i| minimized(&front[i], objectives).to_vec())
        .collect();
    let mut contributions = vec![0.0; front.len()];
    if M > 0 && !points.is_empty() {
        for (&index, contribution) in inside.iter().zip(contributions_of(&points, &reference)) {
            contributions[index] = contribution;
        }
    }
    contributions
}

// the exclusive contributions of minimized points that all dominate the reference point
fn contributions_of(points: &[Vec<f64>], reference: &[f64]) -> Vec<f64> {
    let n = points.len();
    let dimensions = reference.len();
    let mut contributions = vec![0.0; n];
    match dimensions {
        1 => {
            // the smallest point alone dominates up to the second smallest, unless they're equal
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| points[a][0].total_cmp(&points[b][0]));
            let best = order[0];
            let next = order.get(1).map_or(reference[0], |&i| points[i][0]);
            contributions[best] = if next == points[best][0] {
                0.0
            } else {
                next - points[best][0]
            };
        }
        2 => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                lexicographic(&[points[a][0], points[a][1]], &[points[b][0], points[b][1]])
            });
            let sorted: Vec<[f64; 2]> = order
                .iter()
                .map(|&i| [points[i][0], points[i][1]])
                .collect();
            let mut slab = vec![0.0; n];
            staircase_contributions(&sorted, [reference[0], reference[1]], &mut slab);
            for (&i, contribution) in order.iter().zip(slab) {
                contributions[i] = contribution;
            }
        }
        3 => {
            // slabs along the third objective, with the points below kept sorted in the first two
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| points[a][2].total_cmp(&points[b][2]));
            let mut below: Vec<[f64; 2]> = Vec::with_capacity(n);
            let mut members: Vec<usize> = Vec::with_capacity(n);
            let mut slab = vec![0.0; n];
            for (position, &i) in order.iter().enumerate() {
                let point = [points[i][0], points[i][1]];
                let at = below.partition_point(|other| lexicographic(other, &point).is_le());
                below.insert(at, point);
                members.insert(at, i);
                let top = order
                    .get(position + 1)
                    .map_or(reference[2], |&next| points[next][2]);
                let thickness = top - points[i][2];
                if thickness > 0.0 {
                    let slab = &mut slab[..below.len()];
                    staircase_contributions(&below, [reference[0], reference[1]], slab);
                    for (&member, &contribution) in members.iter().zip(slab.iter()) {
                        // a zero contribution adds nothing, even to an infinite slab
                        if contribution > 0.0 {
                            contributions[member] += thickness * contribution;
                        }
                    }
                }
            }
        }
        _ => {
            // between consecutive values of the last objective, the points below share a slab:
            // each one's exclusive part of it is the slab's thickness times its exclusive
            // contribution in one dimension fewer
            let last = dimensions - 1;
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| points[a][last].total_cmp(&points[b][last]));
            let mut below: Vec<Vec<f64>> = Vec::with_capacity(n);
            let mut members: Vec<usize> = Vec::with_capacity(n);
            for (position, &i) in order.iter().enumerate() {
                below.push(points[i][..last].to_vec());
                members.push(i);
                let top = order
                    .get(position + 1)
                    .map_or(reference[last], |&next| points[next][last]);
                let thickness = top - points[i][last];
                if thickness > 0.0 {
                    let slab = contributions_of(&below, &reference[..last]);
                    for (&member, contribution) in members.iter().zip(slab) {
                        if contribution > 0.0 {
                            contributions[member] += thickness * contribution;
                        }
                    }
                }
            }
        }
    }
    contributions
}

fn lexicographic(a: &[f64; 2], b: &[f64; 2]) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1]))
}

// the exclusive contributions of 2-dimensional minimized points, sorted by the first objective
// and then the second, into `contributions`: the staircase of the non-dominated points, x
// increasing and y decreasing; a step's rectangle reaches to the next step's x and the previous
// step's y, and its exclusive part is what the points it dominates (duplicates included) leave
// uncovered; those come right after it in the order
fn staircase_contributions(points: &[[f64; 2]], reference: [f64; 2], contributions: &mut [f64]) {
    contributions.fill(0.0);
    let mut step = 0;
    let mut above = reference[1];
    while step < points.len() {
        // the next step: the first later point strictly below this one
        let mut next = step + 1;
        while next < points.len() && points[next][1] >= points[step][1] {
            next += 1;
        }
        let right = points.get(next).map_or(reference[0], |point| point[0]);
        let [x, y] = points[step];
        let mut covered = 0.0;
        let mut ceiling = above;
        for &[qx, qy] in &points[step + 1..next] {
            if qx >= right {
                break;
            }
            if qy < ceiling {
                covered += (right - qx) * (ceiling - qy);
                ceiling = qy;
            }
        }
        let area = (right - x) * (above - y) - covered;
        contributions[step] = if area.is_nan() {
            // infinite coordinates: the uncovered strips one by one, which never subtract an
            // infinity from an infinity (finite ones take the faster formula above)
            exclusive_strips(&points[step..next], right, above)
        } else {
            area
        };
        above = y;
        step = next;
    }
}

// the part of a step's rectangle (up to `right` and `above`) that the points it dominates, which
// follow it, leave uncovered, as strips between their x coordinates
fn exclusive_strips(points: &[[f64; 2]], right: f64, above: f64) -> f64 {
    let [mut x, y] = points[0];
    let mut ceiling = above;
    let mut area = 0.0;
    let mut strip = |from: f64, to: f64, top: f64| {
        if to > from && top > y {
            area += (to - from) * (top - y);
        }
    };
    for &[qx, qy] in &points[1..] {
        if qx >= right {
            break;
        }
        if qy < ceiling {
            strip(x, qx, ceiling);
            x = qx;
            ceiling = qy;
        }
    }
    strip(x, right, ceiling);
    area
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
/// an empty reference front, or when every distance is 0, and NaN with infinite values.
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
    fn contributions_special_cases() {
        let min2 = [Minimize, Minimize];
        // duplicates contribute nothing, but still bound their neighbors
        let front = [[1.0, 3.0], [2.0, 2.0], [2.0, 2.0], [3.0, 1.0]];
        let contributions = hypervolume_contributions(&front, &[4.0, 4.0], &min2);
        assert_eq!(contributions, [1.0, 0.0, 0.0, 1.0]);
        // weakly dominated points contribute nothing, but they cover part of their dominator's
        // box: (2, 2) takes 1 of the 2 that (1, 2) dominates
        let front = [[1.0, 2.0], [1.0, 3.0], [2.0, 2.0]];
        let contributions = hypervolume_contributions(&front, &[3.0, 3.0], &min2);
        assert_eq!(contributions, [1.0, 0.0, 0.0]);
        assert_eq!(
            hypervolume_contributions(&[[5.0, 0.0]], &[4.0, 4.0], &min2),
            [0.0]
        );
        let one = [Minimize];
        assert_eq!(
            hypervolume_contributions(&[[1.0], [3.0], [1.0]], &[4.0], &one),
            [0.0; 3]
        );
        assert_eq!(
            hypervolume_contributions(&[[1.0], [3.0]], &[4.0], &one),
            [2.0, 0.0]
        );
        // a dominated point covers part of its dominator's exclusive volume in every slice:
        // 4⁴ − 3 · 2 · 4 · 4 = 160
        let front = [[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]];
        let contributions = hypervolume_contributions(&front, &[4.0; 4], &[Minimize; 4]);
        assert_eq!(contributions, [0.0, 160.0]);
        // three unit cubes along the diagonal of a 2 × 2 × 2 box
        let cubes = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let contributions = hypervolume_contributions(&cubes, &[2.0; 3], &[Minimize; 3]);
        assert_eq!(contributions, [1.0; 3]);
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

    // each contribution is the hypervolume lost without that point
    fn check_contributions<const M: usize>(front: &[[f64; M]]) -> Result<(), TestCaseError> {
        let (reference, objectives) = ([4.0; M], [Minimize; M]);
        let total = hypervolume(front, &reference, &objectives);
        let contributions = hypervolume_contributions(front, &reference, &objectives);
        for (i, contribution) in contributions.iter().enumerate() {
            let mut others = front.to_vec();
            others.remove(i);
            let difference = total - hypervolume(&others, &reference, &objectives);
            let tolerance = 1e-9 * total.max(1.0);
            prop_assert!(
                (contribution - difference).abs() <= tolerance,
                "{contribution} {difference}"
            );
        }
        Ok(())
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
        fn contributions_are_hypervolume_differences(
            two in any_front::<2>(12),
            three in any_front::<3>(10),
            four in any_front::<4>(8),
        ) {
            check_contributions(&two)?;
            check_contributions(&three)?;
            check_contributions(&four)?;
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
