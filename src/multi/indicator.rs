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
/// nothing, and neither do points with a NaN value. Larger is better, and a front that dominates
/// another has a larger hypervolume
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
    let mut points: Vec<[f64; M]> = front
        .iter()
        .map(|point| minimized(point, objectives))
        .filter(|point| point.iter().zip(&reference).all(|(x, r)| x < r))
        .collect();
    if points.is_empty() {
        return 0.0;
    }
    hypervolume_of(&mut points, M, &reference)
}

// the hypervolume of minimized points that all dominate the reference point, in their first
// `dimensions` values. The result depends only on the points, not on their order: every sort
// is by one value at a time, and points with equal values add nothing between them.
fn hypervolume_of<const M: usize>(
    points: &mut [[f64; M]],
    dimensions: usize,
    reference: &[f64; M],
) -> f64 {
    match dimensions {
        0 => 0.0,
        1 => points
            .iter()
            .map(|p| reference[0] - p[0])
            .fold(0.0, f64::max),
        2 => {
            points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
            let (mut volume, mut ceiling) = (0.0, reference[1]);
            for point in points.iter() {
                if point[1] < ceiling {
                    volume += (reference[0] - point[0]) * (ceiling - point[1]);
                    ceiling = point[1];
                }
            }
            volume
        }
        3 => {
            // slices along the third objective, with the points below kept sorted as the
            // 2-dimensional case sorts them, instead of sorting a copy for each slice
            points.sort_by(|a, b| a[2].total_cmp(&b[2]));
            let mut volume = 0.0;
            let mut below: Vec<[f64; 2]> = Vec::with_capacity(points.len());
            for (index, point) in points.iter().enumerate() {
                let flat = [point[0], point[1]];
                let at = below.partition_point(|other| {
                    other[0]
                        .total_cmp(&flat[0])
                        .then(other[1].total_cmp(&flat[1]))
                        .is_le()
                });
                below.insert(at, flat);
                let top = points.get(index + 1).map_or(reference[2], |next| next[2]);
                let thickness = top - point[2];
                if thickness > 0.0 {
                    let (mut area, mut ceiling) = (0.0, reference[1]);
                    for &[x, y] in &below {
                        if y < ceiling {
                            area += (reference[0] - x) * (ceiling - y);
                            ceiling = y;
                        }
                    }
                    volume += thickness * area;
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
            let mut below: Vec<[f64; M]> = Vec::with_capacity(points.len());
            for index in 0..points.len() {
                let top = points
                    .get(index + 1)
                    .map_or(reference[last], |next| next[last]);
                let thickness = top - points[index][last];
                if thickness > 0.0 {
                    below.clear();
                    below.extend_from_slice(&points[..=index]);
                    volume += thickness * hypervolume_of(&mut below, last, reference);
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
    if M == 2 || M == 3 {
        let points = front
            .iter()
            .map(|point| minimized(point, objectives))
            .collect();
        return Shrinking::new(points, reference).contributions().to_vec();
    }
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

/// The exclusive hypervolume contributions of the points of a front of 2 or 3 minimized
/// objectives, as [`hypervolume_contributions`] gives them, kept up to date as points are removed:
/// SMS-EMOA removes the smallest contributor again and again. The points are sorted once, and a
/// removed point leaves the sorted order: a stable sort of the points left would give the same
/// order, so each contribution is the same to the bit, every sum being taken in the same order,
/// without a sort or an allocation per removal.
pub(crate) struct Shrinking<const M: usize> {
    points: Vec<[f64; M]>,
    reference: [f64; M],
    // the points left that dominate the reference point, by the first objective and then the
    // second for 2 objectives, by the third for 3; ties in the order of `points`
    order: Vec<usize>,
    // 2 objectives: the points of `order`
    sorted: Vec<[f64; 2]>,
    // 3 objectives: the points below the current slab, sorted in the first two objectives, and
    // their indices
    below: Vec<[f64; 2]>,
    members: Vec<usize>,
    slab: Vec<f64>,
    // by index into `points`; 0 for a point removed or outside the reference point
    contributions: Vec<f64>,
}

impl<const M: usize> Shrinking<M> {
    // M is 2 or 3
    pub(crate) fn new(points: Vec<[f64; M]>, reference: [f64; M]) -> Self {
        debug_assert!(M == 2 || M == 3);
        let mut order: Vec<usize> = (0..points.len())
            .filter(|&i| points[i].iter().zip(&reference).all(|(x, r)| x < r))
            .collect();
        // stable: equal points keep their order, as in `contributions_of`
        if M == 2 {
            order.sort_by(|&a, &b| {
                lexicographic(&[points[a][0], points[a][1]], &[points[b][0], points[b][1]])
            });
        } else {
            order.sort_by(|&a, &b| points[a][2].total_cmp(&points[b][2]));
        }
        let sorted = if M == 2 {
            order
                .iter()
                .map(|&i| [points[i][0], points[i][1]])
                .collect()
        } else {
            Vec::new()
        };
        let n = order.len();
        Self {
            contributions: vec![0.0; points.len()],
            points,
            reference,
            order,
            sorted,
            below: Vec::with_capacity(n),
            members: Vec::with_capacity(n),
            slab: vec![0.0; n],
        }
    }

    /// The contribution of each point, by its index; 0 for a removed point.
    pub(crate) fn contributions(&mut self) -> &[f64] {
        self.contributions.fill(0.0);
        let reference = [self.reference[0], self.reference[1]];
        if M == 2 {
            let slab = &mut self.slab[..self.order.len()];
            staircase_contributions(&self.sorted, reference, slab);
            for (&i, &contribution) in self.order.iter().zip(slab.iter()) {
                self.contributions[i] = contribution;
            }
            return &self.contributions;
        }
        // slabs along the third objective, as `contributions_of` takes them
        self.below.clear();
        self.members.clear();
        for (position, &i) in self.order.iter().enumerate() {
            let point = [self.points[i][0], self.points[i][1]];
            let at = self
                .below
                .partition_point(|other| lexicographic(other, &point).is_le());
            self.below.insert(at, point);
            self.members.insert(at, i);
            let top = self
                .order
                .get(position + 1)
                .map_or(self.reference[2], |&next| self.points[next][2]);
            let thickness = top - self.points[i][2];
            if thickness > 0.0 {
                let slab = &mut self.slab[..self.below.len()];
                staircase_contributions(&self.below, reference, slab);
                for (&member, &contribution) in self.members.iter().zip(slab.iter()) {
                    // a zero contribution adds nothing, even to an infinite slab
                    if contribution > 0.0 {
                        self.contributions[member] += thickness * contribution;
                    }
                }
            }
        }
        &self.contributions
    }

    /// Removes the point at `index`.
    pub(crate) fn remove(&mut self, index: usize) {
        if let Some(position) = self.order.iter().position(|&i| i == index) {
            self.order.remove(position);
            if M == 2 {
                self.sorted.remove(position);
            }
        }
    }
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
/// Infinite for an empty front, and NaN for an empty reference front. A point of `front` with a
/// NaN value, an invalid solution, is skipped: a front of only such points scores infinity. The
/// directions don't matter for distances, so IGD takes none.
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
/// Infinite for an empty reference front, and NaN for an empty front. A point of `front` with a
/// NaN value, an invalid solution, is infinitely far: the result is infinite.
pub fn gd<const M: usize>(front: &[[f64; M]], reference_front: &[[f64; M]]) -> f64 {
    mean_nearest(front, reference_front, |a, b| distance(a, b))
}

/// IGD+ (Ishibuchi et al., 2015): like [`igd`], but a point of `front` only counts as far from a
/// reference point by how much it's worse in each objective, `√Σ max(aᵢ − zᵢ, 0)²` when
/// minimizing. A front that dominates or equals every reference point scores 0, and a front that
/// dominates another never scores worse (weak Pareto compliance). Smaller is better.
///
/// Infinite for an empty front, and NaN for an empty reference front. A point of `front` with a
/// NaN value, an invalid solution, is skipped, as in [`igd`]: a front of only such points scores
/// infinity.
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
            .map(|(a, z)| {
                // NaN stays NaN, so that a point with a NaN value is skipped, as in `igd`:
                // `f64::max` would turn it into 0, the best distance
                let worse = a - z;
                if worse > 0.0 || worse.is_nan() {
                    worse * worse
                } else {
                    0.0
                }
            })
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
            let sign = if !subset.count_ones().is_multiple_of(2) {
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
    fn points_with_nan_values() {
        let reference = [[0.0, 1.0], [1.0, 0.0]];
        let minimize = [Minimize, Minimize];
        let good = [[1.0, 2.0]];
        let with_nan = [[f64::NAN, f64::NAN], [1.0, 2.0]];
        let only_nan = [[f64::NAN, f64::NAN]];
        // skipped by IGD and IGD+: a NaN point isn't a perfect one
        assert_eq!(igd(&with_nan, &reference), igd(&good, &reference));
        assert_eq!(
            igd_plus(&with_nan, &reference, &minimize),
            igd_plus(&good, &reference, &minimize)
        );
        assert!(igd_plus(&good, &reference, &minimize) > 0.0);
        assert_eq!(igd(&only_nan, &reference), f64::INFINITY);
        assert_eq!(igd_plus(&only_nan, &reference, &minimize), f64::INFINITY);
        // one NaN value is enough, in either direction
        let half_nan = [[0.0, f64::NAN]];
        assert_eq!(igd_plus(&half_nan, &reference, &minimize), f64::INFINITY);
        assert_eq!(
            igd_plus(&half_nan, &reference, &[Maximize, Maximize]),
            f64::INFINITY
        );
        // infinitely far for GD, and nothing for the hypervolume
        assert_eq!(gd(&with_nan, &reference), f64::INFINITY);
        assert_eq!(
            hypervolume(&with_nan, &[4.0, 4.0], &minimize),
            hypervolume(&good, &[4.0, 4.0], &minimize)
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

    // the hypervolume by slicing, as it was computed before its slices reused buffers: a copy
    // of the points below each slice, sorted again. The faster one must give the same bits.
    fn sliced(mut points: Vec<Vec<f64>>, reference: &[f64]) -> f64 {
        let dimensions = reference.len();
        match dimensions {
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
                let last = dimensions - 1;
                points.sort_by(|a, b| a[last].total_cmp(&b[last]));
                let mut volume = 0.0;
                let mut below: Vec<Vec<f64>> = Vec::new();
                for (index, point) in points.iter().enumerate() {
                    below.push(point[..last].to_vec());
                    let top = points
                        .get(index + 1)
                        .map_or(reference[last], |next| next[last]);
                    let thickness = top - point[last];
                    if thickness > 0.0 {
                        volume += thickness * sliced(below.clone(), &reference[..last]);
                    }
                }
                volume
            }
        }
    }

    fn same_as_sliced<const M: usize>(front: &[[f64; M]], reference: f64) -> bool {
        let inside: Vec<Vec<f64>> = front
            .iter()
            .filter(|p| p.iter().all(|&x| x < reference))
            .map(|p| p.to_vec())
            .collect();
        let expected = if inside.is_empty() {
            0.0
        } else {
            sliced(inside, &[reference; M])
        };
        hypervolume(front, &[reference; M], &[Minimize; M]).to_bits() == expected.to_bits()
    }

    // removes the points of `front` in the order of `removals` (indices into the points left), and
    // checks after each removal that `Shrinking` gives every point left the contribution that
    // `contributions_of` gives it from scratch, as SMS-EMOA computed it before, to the bit
    fn shrinks_as_from_scratch<const M: usize>(
        front: &[[f64; M]],
        removals: &[usize],
    ) -> Result<(), TestCaseError> {
        let reference = [4.0; M];
        let mut shrinking = Shrinking::new(front.to_vec(), reference);
        let mut left: Vec<usize> = (0..front.len()).collect();
        let mut removals = removals.iter();
        loop {
            let inside: Vec<usize> = left
                .iter()
                .copied()
                .filter(|&i| front[i].iter().zip(&reference).all(|(x, r)| x < r))
                .collect();
            let points: Vec<Vec<f64>> = inside.iter().map(|&i| front[i].to_vec()).collect();
            let mut expected = vec![0.0; front.len()];
            if !points.is_empty() {
                for (&i, contribution) in inside.iter().zip(contributions_of(&points, &reference)) {
                    expected[i] = contribution;
                }
            }
            let got = shrinking.contributions();
            for &i in &left {
                prop_assert_eq!(
                    got[i].to_bits(),
                    expected[i].to_bits(),
                    "point {} of {:?}",
                    i,
                    front
                );
            }
            match removals.next() {
                Some(&removal) if !left.is_empty() => {
                    shrinking.remove(left.remove(removal % left.len()));
                }
                _ => return Ok(()),
            }
        }
    }

    // values with ties, and infinities and NaN, which lie outside the reference point or beyond
    // the ideal one
    fn any_value() -> impl Strategy<Value = f64> {
        prop_oneof![
            8 => (0..8).prop_map(|v| f64::from(v) / 2.0),
            8 => 0.0..4.0,
            1 => Just(f64::NEG_INFINITY),
            1 => Just(f64::INFINITY),
            1 => Just(f64::NAN),
        ]
    }

    #[test]
    fn contributions_of_2_and_3_objectives_are_as_before() {
        // the public function, now through `Shrinking`, on the doc example and a 3-objective one
        let two = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [3.0, 3.0]];
        let three = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [3.0, 3.0, 1.0],
        ];
        for (got, expected) in [
            (
                hypervolume_contributions(&two, &[4.0; 2], &[Minimize; 2]),
                contributions_of(&two.map(|p| p.to_vec()), &[4.0; 2]),
            ),
            (
                hypervolume_contributions(&three, &[4.0; 3], &[Minimize; 3]),
                contributions_of(&three.map(|p| p.to_vec()), &[4.0; 3]),
            ),
        ] {
            let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&expected));
        }
    }

    proptest! {
        #[test]
        fn shrinking_contributions_are_the_ones_from_scratch_to_the_bit(
            two in prop::collection::vec(prop::array::uniform::<_, 2>(any_value()), 0..40),
            three in prop::collection::vec(prop::array::uniform::<_, 3>(any_value()), 0..25),
            removals in prop::collection::vec(0usize..100, 0..40),
        ) {
            shrinks_as_from_scratch(&two, &removals)?;
            shrinks_as_from_scratch(&three, &removals)?;
        }

        #[test]
        fn hypervolume_is_the_sliced_one_to_the_bit(
            three in prop::collection::vec(prop::array::uniform::<_, 3>(prop_oneof![(0..6).prop_map(f64::from), 0.0..4.0]), 0..60),
            four in prop::collection::vec(prop::array::uniform::<_, 4>(prop_oneof![(0..6).prop_map(f64::from), 0.0..4.0]), 0..30),
            five in prop::collection::vec(prop::array::uniform::<_, 5>(prop_oneof![(0..6).prop_map(f64::from), 0.0..4.0]), 0..15),
        ) {
            prop_assert!(same_as_sliced(&three, 4.5));
            prop_assert!(same_as_sliced(&four, 4.5));
            prop_assert!(same_as_sliced(&five, 4.5));
        }

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
