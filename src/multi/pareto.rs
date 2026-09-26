//! Pareto dominance, non-dominated sorting and crowding distance.

use super::Scores;
use crate::Objective;
use std::cmp::Ordering;

/// Whether `a` dominates `b`, by constrained dominance (Deb, 2000):
///
/// 1. Every valid solution dominates an [invalid](Scores::invalid) one.
/// 2. A feasible solution dominates an infeasible one.
/// 3. Between infeasible solutions, the smaller constraint violation dominates, whatever the
///    objective values.
/// 4. Between feasible solutions, Pareto dominance: `a` is at least as good as `b` in every
///    objective and better in at least one, in the direction of each objective in `objectives`.
///
/// Equal scores don't dominate each other.
///
/// ```
/// use genoxide::Objective::{Maximize, Minimize};
/// use genoxide::multi::{Scores, dominates};
///
/// let objectives = [Minimize, Maximize];
/// let (a, b) = (Scores::new([1.0, 5.0]), Scores::new([2.0, 5.0]));
/// assert!(dominates(&a, &b, &objectives));
/// assert!(!dominates(&b, &a, &objectives));
/// // a trade-off: neither dominates
/// let c = Scores::new([0.0, 1.0]);
/// assert!(!dominates(&a, &c, &objectives) && !dominates(&c, &a, &objectives));
/// // feasibility comes first
/// let infeasible = Scores::constrained([0.0, 9.0], 0.1);
/// assert!(dominates(&b, &infeasible, &objectives));
/// ```
pub fn dominates<const M: usize>(
    a: &Scores<M>,
    b: &Scores<M>,
    objectives: &[Objective; M],
) -> bool {
    match (a.is_valid(), b.is_valid()) {
        (true, false) => return true,
        (false, _) => return false,
        (true, true) => {}
    }
    if a.violation() != b.violation() {
        return a.violation() < b.violation();
    }
    if a.violation() > 0.0 {
        // equally infeasible
        return false;
    }
    pareto_dominates(
        &minimized(a.raw(), objectives),
        &minimized(b.raw(), objectives),
    )
}

// the values with every objective turned into one to minimize
fn minimized<const M: usize>(values: &[f64; M], objectives: &[Objective; M]) -> [f64; M] {
    std::array::from_fn(|j| match objectives[j] {
        Objective::Minimize => values[j],
        Objective::Maximize => -values[j],
    })
}

// Pareto dominance between minimized, non-NaN values
fn pareto_dominates(a: &[f64], b: &[f64]) -> bool {
    let mut better = false;
    for (x, y) in a.iter().zip(b) {
        if x > y {
            return false;
        }
        better |= x < y;
    }
    better
}

// whether `new` has a solution that no solution of `old` dominates or equals: the front improved
pub(crate) fn gains<const M: usize>(
    new: &[Scores<M>],
    old: &[Scores<M>],
    objectives: &[Objective; M],
) -> bool {
    new.iter().any(|candidate| {
        !old.iter()
            .any(|other| other == candidate || dominates(other, candidate, objectives))
    })
}

/// The fronts of non-dominated sorting by [constrained dominance](dominates): the first front
/// holds the indices of the solutions that no other solution dominates, the second those that
/// only the first front dominates, and so on. Each front lists its indices in ascending order.
///
/// The feasible solutions come first, sorted by Pareto dominance; then the infeasible ones, one
/// front per constraint violation, smallest first; then the invalid ones, in one front.
///
/// Sorting takes O(N log N) time for 2 objectives, and uses ENS-BS (Zhang et al., 2015) for
/// more, which is much faster than the O(M N²) of Deb's fast non-dominated sort in practice.
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::{Scores, non_dominated_sort};
///
/// let scores = [
///     Scores::new([1.0, 4.0]),
///     Scores::new([2.0, 2.0]),
///     Scores::new([3.0, 3.0]), // dominated by [2, 2]
///     Scores::new([4.0, 1.0]),
///     Scores::constrained([0.0, 0.0], 1.0), // infeasible
/// ];
/// let fronts = non_dominated_sort(&scores, &[Minimize, Minimize]);
/// assert_eq!(fronts, [vec![0, 1, 3], vec![2], vec![4]]);
/// ```
pub fn non_dominated_sort<const M: usize>(
    scores: &[Scores<M>],
    objectives: &[Objective; M],
) -> Vec<Vec<usize>> {
    let mut feasible = Vec::new();
    let mut infeasible = Vec::new();
    let mut invalid = Vec::new();
    for (index, score) in scores.iter().enumerate() {
        if !score.is_valid() {
            invalid.push(index);
        } else if score.violation() > 0.0 {
            infeasible.push(index);
        } else {
            feasible.push(index);
        }
    }
    let points: Vec<[f64; M]> = feasible
        .iter()
        .map(|&index| minimized(scores[index].raw(), objectives))
        .collect();
    let ranks = if M == 2 {
        two_objective_ranks(&points)
    } else {
        ens_ranks(&points)
    };
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    for (position, &rank) in ranks.iter().enumerate() {
        if rank >= fronts.len() {
            fronts.resize(rank + 1, Vec::new());
        }
        fronts[rank].push(feasible[position]);
    }
    // one front per violation, smallest first; the sort is stable, so indices stay ascending
    infeasible.sort_by(|&a, &b| scores[a].violation().total_cmp(&scores[b].violation()));
    for chunk in infeasible.chunk_by(|&a, &b| scores[a].violation() == scores[b].violation()) {
        fronts.push(chunk.to_vec());
    }
    if !invalid.is_empty() {
        fronts.push(invalid);
    }
    for front in &mut fronts {
        front.sort_unstable();
    }
    fronts
}

// the positions of `points`, lexicographically sorted, the earlier position on ties
fn lexicographic_order<const M: usize>(points: &[[f64; M]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| {
        points[a]
            .iter()
            .zip(&points[b])
            .map(|(x, y)| x.partial_cmp(y).unwrap_or(Ordering::Equal))
            .find(|ordering| ordering.is_ne())
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

// the front of each of the minimized points with 2 objectives, in O(N log N): in lexicographic
// order, the second value of the points added to a front only decreases, so a point belongs to
// the first front whose last point has a larger second value, or to the front of its duplicate
fn two_objective_ranks<const M: usize>(points: &[[f64; M]]) -> Vec<usize> {
    let mut ranks = vec![0; points.len()];
    // the second value of the last point of each front, non-decreasing from front to front
    let mut lasts: Vec<f64> = Vec::new();
    let mut previous: Option<usize> = None;
    for position in lexicographic_order(points) {
        let point = &points[position];
        let rank = match previous {
            Some(previous) if points[previous] == *point => ranks[previous],
            _ => lasts.partition_point(|&last| last <= point[1]),
        };
        if rank == lasts.len() {
            lasts.push(point[1]);
        } else {
            lasts[rank] = point[1];
        }
        ranks[position] = rank;
        previous = Some(position);
    }
    ranks
}

// the front of each of the minimized points by ENS-BS (Zhang et al., 2015): in lexicographic
// order, no point is dominated by a later one, and if a front dominates a point, every earlier
// front does too, so a binary search finds the first front that doesn't dominate it
fn ens_ranks<const M: usize>(points: &[[f64; M]]) -> Vec<usize> {
    let mut ranks = vec![0; points.len()];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    for position in lexicographic_order(points) {
        let point = &points[position];
        let dominated_by = |front: &Vec<usize>| {
            // the most recently added points are the most likely to dominate
            front
                .iter()
                .rev()
                .any(|&other| pareto_dominates(&points[other], point))
        };
        let rank = fronts.partition_point(dominated_by);
        if rank == fronts.len() {
            fronts.push(Vec::new());
        }
        fronts[rank].push(position);
        ranks[position] = rank;
    }
    ranks
}

/// The crowding distance of each solution of a front (Deb et al., 2002), in the order of
/// `front`: for every objective, the distance between the neighbors on either side, relative to
/// the range of the front, summed over the objectives. The solutions at the ends of each
/// objective get an infinite distance. Larger distances mean less crowded regions, which
/// NSGA-II prefers to keep a spread-out front.
///
/// An objective whose range is 0 or not finite adds nothing. If any solution of the front is
/// invalid, every distance is 0.
///
/// # Panics
///
/// If an index of `front` is out of range of `scores`.
///
/// ```
/// use genoxide::multi::{Scores, crowding_distance};
///
/// let scores = [
///     Scores::new([0.0, 4.0]),
///     Scores::new([1.0, 3.0]),
///     Scores::new([3.0, 1.0]),
///     Scores::new([4.0, 0.0]),
/// ];
/// let distances = crowding_distance(&scores, &[0, 1, 2, 3]);
/// assert_eq!(distances, [f64::INFINITY, 1.5, 1.5, f64::INFINITY]);
/// ```
pub fn crowding_distance<const M: usize>(scores: &[Scores<M>], front: &[usize]) -> Vec<f64> {
    let mut distances = vec![0.0; front.len()];
    if front.iter().any(|&index| !scores[index].is_valid()) {
        return distances;
    }
    if front.len() <= 2 {
        distances.fill(f64::INFINITY);
        return distances;
    }
    let mut order: Vec<usize> = (0..front.len()).collect();
    for objective in 0..M {
        let value = |position: usize| scores[front[position]].raw()[objective];
        order.sort_by(|&a, &b| value(a).total_cmp(&value(b)).then(a.cmp(&b)));
        let (first, last) = (order[0], order[order.len() - 1]);
        let range = value(last) - value(first);
        // a flat objective has no ends: with every value equal, the first and last in the order
        // would be arbitrary
        if !(range > 0.0 && range.is_finite()) {
            continue;
        }
        distances[first] = f64::INFINITY;
        distances[last] = f64::INFINITY;
        for window in order.windows(3) {
            let (previous, middle, next) = (window[0], window[1], window[2]);
            distances[middle] += (value(next) - value(previous)) / range;
        }
    }
    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::{Maximize, Minimize};
    use proptest::prelude::*;

    // Deb's fast non-dominated sort, O(M N²): the reference for the faster sorts
    fn reference_sort<const M: usize>(
        scores: &[Scores<M>],
        objectives: &[Objective; M],
    ) -> Vec<Vec<usize>> {
        let n = scores.len();
        let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut counts = vec![0usize; n];
        for a in 0..n {
            for b in 0..n {
                if dominates(&scores[a], &scores[b], objectives) {
                    dominated[a].push(b);
                } else if dominates(&scores[b], &scores[a], objectives) {
                    counts[a] += 1;
                }
            }
        }
        let mut fronts = Vec::new();
        let mut current: Vec<usize> = (0..n).filter(|&index| counts[index] == 0).collect();
        while !current.is_empty() {
            let mut next = Vec::new();
            for &a in &current {
                for &b in &dominated[a] {
                    counts[b] -= 1;
                    if counts[b] == 0 {
                        next.push(b);
                    }
                }
            }
            current.sort_unstable();
            fronts.push(current);
            current = next;
        }
        fronts
    }

    #[test]
    fn dominance() {
        let objectives = [Minimize, Minimize];
        let s = |a: f64, b: f64| Scores::new([a, b]);
        assert!(dominates(&s(1.0, 1.0), &s(1.0, 2.0), &objectives));
        assert!(!dominates(&s(1.0, 1.0), &s(1.0, 1.0), &objectives));
        assert!(!dominates(&s(1.0, 2.0), &s(2.0, 1.0), &objectives));
        assert!(dominates(
            &s(0.0, 0.0),
            &s(-0.0, f64::INFINITY),
            &objectives
        ));
        assert!(dominates(
            &s(f64::NEG_INFINITY, 0.0),
            &s(0.0, 0.0),
            &objectives
        ));
        let maximize = [Maximize, Maximize];
        assert!(dominates(&s(1.0, 2.0), &s(1.0, 1.0), &maximize));
        // invalid, then infeasible
        let invalid = Scores::<2>::invalid();
        let infeasible = |violation| Scores::constrained([0.0, 0.0], violation);
        assert!(dominates(&infeasible(f64::INFINITY), &invalid, &objectives));
        assert!(!dominates(&invalid, &invalid, &objectives));
        assert!(!dominates(&invalid, &s(9.0, 9.0), &objectives));
        assert!(dominates(&s(9.0, 9.0), &infeasible(0.1), &objectives));
        assert!(dominates(&infeasible(0.1), &infeasible(0.2), &objectives));
        assert!(!dominates(
            &infeasible(0.1),
            &Scores::constrained([5.0, 5.0], 0.1),
            &objectives
        ));
    }

    #[test]
    fn sorting_special_cases() {
        let objectives = [Minimize, Minimize];
        assert!(non_dominated_sort::<2>(&[], &objectives).is_empty());
        // duplicates share a front
        let duplicates = [Scores::new([1.0, 1.0]); 3];
        assert_eq!(
            non_dominated_sort(&duplicates, &objectives),
            [vec![0, 1, 2]]
        );
        // a chain
        let chain: Vec<Scores<2>> = (0..5).rev().map(|i| Scores::new([i as f64; 2])).collect();
        let fronts = non_dominated_sort(&chain, &objectives);
        assert_eq!(fronts, [[4], [3], [2], [1], [0]]);
        // equal second values: [0, 1] dominates [1, 1], which dominates [2, 1]
        let flat = [
            Scores::new([2.0, 1.0]),
            Scores::new([0.0, 1.0]),
            Scores::new([1.0, 1.0]),
        ];
        assert_eq!(non_dominated_sort(&flat, &objectives), [[1], [2], [0]]);
        // constraints and invalid solutions
        let mixed = [
            Scores::invalid(),
            Scores::constrained([0.0, 0.0], 2.0),
            Scores::new([5.0, 5.0]),
            Scores::constrained([9.0, 9.0], 1.0),
            Scores::constrained([1.0, 1.0], 2.0),
            Scores::invalid(),
        ];
        assert_eq!(
            non_dominated_sort(&mixed, &objectives),
            [vec![2], vec![3], vec![1, 4], vec![0, 5]]
        );
        // one objective: a total order, with ties together
        let single = [Scores::new([3.0]), Scores::new([1.0]), Scores::new([3.0])];
        assert_eq!(
            non_dominated_sort(&single, &[Maximize]),
            [vec![0, 2], vec![1]]
        );
    }

    #[test]
    fn crowding() {
        let s = |a: f64, b: f64| Scores::new([a, b]);
        assert_eq!(
            crowding_distance(&[s(0.0, 0.0), s(1.0, 1.0)], &[0, 1]),
            [f64::INFINITY; 2]
        );
        // one objective with range 0 adds nothing
        let flat = [s(0.0, 5.0), s(1.0, 5.0), s(3.0, 5.0), s(4.0, 5.0)];
        let distances = crowding_distance(&flat, &[0, 1, 2, 3]);
        assert_eq!(distances, [f64::INFINITY, 0.75, 0.75, f64::INFINITY]);
        // only the front's members count, in its order
        let distances = crowding_distance(&flat, &[3, 1, 0]);
        assert_eq!(distances, [f64::INFINITY, 1.0, f64::INFINITY]);
        // a flat objective, unsorted: the ends of the other objectives are the only infinite ones
        let flat = [
            Scores::new([1.0, 3.0, 7.0]),
            Scores::new([0.0, 4.0, 7.0]),
            Scores::new([4.0, 0.0, 7.0]),
            Scores::new([3.0, 1.0, 7.0]),
            Scores::new([2.0, 2.0, 7.0]),
        ];
        let distances = crowding_distance(&flat, &[0, 1, 2, 3, 4]);
        assert_eq!(distances, [1.0, f64::INFINITY, f64::INFINITY, 1.0, 1.0]);
        let invalid = [Scores::<2>::invalid(); 4];
        assert_eq!(crowding_distance(&invalid, &[0, 1, 2, 3]), [0.0; 4]);
        // an infinite value makes the range infinite: that objective adds nothing
        let infinite = [s(0.0, 0.0), s(1.0, 1.0), s(f64::INFINITY, 2.0)];
        let distances = crowding_distance(&infinite, &[0, 1, 2]);
        assert_eq!(distances, [f64::INFINITY, 1.0, f64::INFINITY]);
    }

    fn any_scores<const M: usize>() -> impl Strategy<Value = Scores<M>> {
        // few distinct values, so that ties and duplicates are common
        let value = prop_oneof![(0..4).prop_map(f64::from), Just(f64::INFINITY)];
        (
            prop::array::uniform::<_, M>(value),
            prop_oneof![8 => Just(0.0), 1 => (1..3).prop_map(f64::from), 1 => Just(f64::NAN)],
        )
            .prop_map(|(values, violation)| {
                if violation.is_nan() {
                    Scores::invalid()
                } else {
                    Scores::constrained(values, violation)
                }
            })
    }

    fn any_objectives<const M: usize>() -> impl Strategy<Value = [Objective; M]> {
        prop::array::uniform::<_, M>(prop_oneof![Just(Minimize), Just(Maximize)])
    }

    proptest! {
        #[test]
        fn two_objective_sort_matches_the_reference(
            scores in prop::collection::vec(any_scores::<2>(), 0..60),
            objectives in any_objectives::<2>(),
        ) {
            prop_assert_eq!(non_dominated_sort(&scores, &objectives), reference_sort(&scores, &objectives));
        }

        #[test]
        fn ens_sort_matches_the_reference(
            three in prop::collection::vec(any_scores::<3>(), 0..60),
            five in prop::collection::vec(any_scores::<5>(), 0..40),
            one in prop::collection::vec(any_scores::<1>(), 0..20),
            objectives in any_objectives::<3>(),
        ) {
            prop_assert_eq!(non_dominated_sort(&three, &objectives), reference_sort(&three, &objectives));
            let five_objectives = [Minimize, Maximize, Minimize, Minimize, Maximize];
            prop_assert_eq!(non_dominated_sort(&five, &five_objectives), reference_sort(&five, &five_objectives));
            prop_assert_eq!(non_dominated_sort(&one, &[Minimize]), reference_sort(&one, &[Minimize]));
        }

        #[test]
        fn dominance_is_a_strict_partial_order(
            a in any_scores::<3>(), b in any_scores::<3>(), c in any_scores::<3>(),
            objectives in any_objectives::<3>(),
        ) {
            prop_assert!(!dominates(&a, &a, &objectives));
            prop_assert!(!(dominates(&a, &b, &objectives) && dominates(&b, &a, &objectives)));
            if dominates(&a, &b, &objectives) && dominates(&b, &c, &objectives) {
                prop_assert!(dominates(&a, &c, &objectives));
            }
        }

        #[test]
        fn crowding_distances_are_not_negative(
            scores in prop::collection::vec(any_scores::<3>(), 1..30),
        ) {
            let front: Vec<usize> = (0..scores.len()).collect();
            let distances = crowding_distance(&scores, &front);
            prop_assert_eq!(distances.len(), scores.len());
            prop_assert!(distances.iter().all(|&d| d >= 0.0));
        }
    }
}
