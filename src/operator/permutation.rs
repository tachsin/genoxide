//! Crossover and mutation for [`Permutation`] genomes: every child is a permutation.

use super::{Crossover, Mutate};
use crate::StreamRng;
use crate::genome::{Order, Permutation};
use std::cell::Cell;

// Scratch space of the genome's length for the operators, reused instead of allocated on every
// call. Each thread has its own, so breeding in parallel shares nothing.
#[derive(Default)]
struct Scratch {
    genes: Vec<usize>,
    positions: Vec<usize>,
    flags: Vec<bool>,
    edges: [Vec<Neighbors>; 2],
}

thread_local! {
    static SCRATCH: Cell<Scratch> = const {
        Cell::new(Scratch {
            genes: Vec::new(),
            positions: Vec::new(),
            flags: Vec::new(),
            edges: [Vec::new(), Vec::new()],
        })
    };
}

// `f` with this thread's scratch space, taken out of the cell meanwhile
fn with_scratch<T>(f: impl FnOnce(&mut Scratch) -> T) -> T {
    SCRATCH.with(|cell| {
        let mut scratch = cell.take();
        let result = f(&mut scratch);
        cell.set(scratch);
        result
    })
}

// the position of every gene, into `positions`
fn positions(genes: &[usize], positions: &mut Vec<usize>) {
    positions.resize(genes.len(), 0);
    for (position, &gene) in genes.iter().enumerate() {
        positions[gene] = position;
    }
}

// a random segment `start..end` of at least `min_len` genes that isn't the whole genome, for
// genomes of at least `min_len + 1` genes
fn segment(len: usize, min_len: usize, rng: &mut StreamRng) -> (usize, usize) {
    debug_assert!(len > min_len, "segment of {min_len} in {len}");
    loop {
        let (start, end) = rng.sample_pair(len + 1);
        if end - start >= min_len && end - start < len {
            return (start, end);
        }
    }
}

/// Partially mapped crossover (PMX) for [`Permutation`] genomes.
///
/// Each child takes a random segment from the other parent, at the same positions. The rest comes
/// from its own parent, where a gene that is now duplicated is replaced through the mapping
/// between the two segments. The segment is never the whole genome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PartiallyMappedCrossover;

// `child` takes the segment `other_segment` from `start` on, repaired by swaps (the mapping of
// PMX); `positions` is scratch space of the genome's length
fn partially_mapped(
    child: &mut [usize],
    other_segment: &[usize],
    start: usize,
    positions: &mut [usize],
) {
    for (position, &gene) in child.iter().enumerate() {
        positions[gene] = position;
    }
    for (position, &gene) in (start..).zip(other_segment) {
        let at = positions[gene];
        let displaced = child[position];
        child.swap(position, at);
        positions[gene] = position;
        positions[displaced] = at;
    }
}

impl Crossover<Permutation> for PartiallyMappedCrossover {
    fn crossover(&self, _: &Permutation, a: &mut Order, b: &mut Order, rng: &mut StreamRng) {
        if a.len() < 2 {
            return;
        }
        let (start, end) = segment(a.len(), 1, rng);
        with_scratch(|scratch| {
            // in place: the second child needs only the first parent's segment
            let segment_of_a = &mut scratch.genes;
            segment_of_a.clear();
            segment_of_a.extend_from_slice(&a[start..end]);
            let positions = &mut scratch.positions;
            positions.resize(a.len(), 0);
            partially_mapped(a.genes_mut(), &b[start..end], start, positions);
            partially_mapped(b.genes_mut(), segment_of_a, start, positions);
        });
    }
}

/// Order crossover (OX1) for [`Permutation`] genomes.
///
/// Each child keeps a random segment of its own parent in place. The other genes follow in the
/// order they have in the other parent, starting after the segment and wrapping around. It keeps
/// relative order, which suits sequencing and tours. The segment is never the whole genome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OrderCrossover;

// `child` keeps its segment `start..end`, and takes the other genes in the order of `other`, from
// `end` on; `in_segment` is scratch space of the genome's length
fn ordered(
    child: &mut [usize],
    other: &[usize],
    start: usize,
    end: usize,
    in_segment: &mut [bool],
) {
    let len = child.len();
    in_segment.fill(false);
    for &gene in &child[start..end] {
        in_segment[gene] = true;
    }
    // the positions after the segment, wrapping around: `end..len`, then `0..start`
    let mut position = end % len;
    for &gene in other[end..].iter().chain(&other[..end]) {
        if !in_segment[gene] {
            child[position] = gene;
            position += 1;
            if position == len {
                position = 0;
            }
        }
    }
}

impl Crossover<Permutation> for OrderCrossover {
    fn crossover(&self, _: &Permutation, a: &mut Order, b: &mut Order, rng: &mut StreamRng) {
        if a.len() < 2 {
            return;
        }
        let (start, end) = segment(a.len(), 1, rng);
        with_scratch(|scratch| {
            // in place: the first child only overwrites genes outside its segment, and the
            // second child reads the first parent
            let first_parent = &mut scratch.genes;
            first_parent.clear();
            first_parent.extend_from_slice(a);
            let in_segment = &mut scratch.flags;
            in_segment.resize(a.len(), false);
            ordered(a.genes_mut(), b, start, end, in_segment);
            ordered(b.genes_mut(), first_parent, start, end, in_segment);
        });
    }
}

/// Cycle crossover (CX) for [`Permutation`] genomes.
///
/// The positions split into cycles: following a gene of one parent to its position in the other
/// parent leads back to the start. The children take the cycles alternately from each parent, so
/// every gene keeps the position it has in one of the parents. It's deterministic, and when the
/// parents form a single cycle, the children are copies of the parents.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CycleCrossover;

impl Crossover<Permutation> for CycleCrossover {
    fn crossover(&self, _: &Permutation, a: &mut Order, b: &mut Order, _: &mut StreamRng) {
        with_scratch(|scratch| {
            positions(a, &mut scratch.positions);
            scratch.flags.clear();
            scratch.flags.resize(a.len(), false);
            let (positions_in_a, visited) = (&scratch.positions[..], &mut scratch.flags[..]);
            let (first, second) = (a.genes_mut(), b.genes_mut());
            let mut odd = false;
            for start in 0..first.len() {
                if visited[start] {
                    continue;
                }
                let mut position = start;
                while !visited[position] {
                    visited[position] = true;
                    // the next position of the cycle, from the parents' genes: a position is
                    // exchanged only once it's visited, and never read again
                    let next = positions_in_a[second[position]];
                    if odd {
                        std::mem::swap(&mut first[position], &mut second[position]);
                    }
                    position = next;
                }
                odd = !odd;
            }
        });
    }
}

/// Edge recombination crossover (ERX) for [`Permutation`] genomes that are tours: the genome is a
/// cycle, and what matters is which genes are neighbors.
///
/// A child starts with the first gene of its parent, and then always moves to a neighbor that the
/// current gene has in either parent, picking the one with the fewest remaining neighbors (ties at
/// random). Only when there's none left, it continues with a random unvisited gene. So children
/// consist almost entirely of their parents' edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeRecombinationCrossover;

// The distinct neighbors of a gene in two tours, at most 4, in the order they were found.
#[derive(Clone, Copy, Default)]
struct Neighbors {
    genes: [usize; 4],
    len: usize,
}

impl Neighbors {
    fn as_slice(&self) -> &[usize] {
        &self.genes[..self.len]
    }

    fn add(&mut self, gene: usize) {
        if !self.as_slice().contains(&gene) {
            self.genes[self.len] = gene;
            self.len += 1;
        }
    }

    // removes `gene`, keeping the order of the others
    fn remove(&mut self, gene: usize) {
        if let Some(at) = self.as_slice().iter().position(|&other| other == gene) {
            self.genes.copy_within(at + 1..self.len, at);
            self.len -= 1;
        }
    }
}

// the neighbors of every gene in the tours `a` and `b`, without duplicates, `a`'s first, into
// `neighbors`
fn edges(a: &[usize], b: &[usize], neighbors: &mut Vec<Neighbors>) {
    let len = a.len();
    neighbors.clear();
    neighbors.resize(len, Neighbors::default());
    for tour in [a, b] {
        for position in 0..len {
            let gene = tour[position];
            for neighbor in [tour[(position + len - 1) % len], tour[(position + 1) % len]] {
                if neighbor != gene {
                    neighbors[gene].add(neighbor);
                }
            }
        }
    }
}

// writes into `child` a tour through the edges in `neighbors`, from `start`; `unvisited` and
// `index` are scratch space
fn edge_recombination(
    neighbors: &mut [Neighbors],
    start: usize,
    child: &mut [usize],
    unvisited: &mut Vec<usize>,
    index: &mut Vec<usize>,
    rng: &mut StreamRng,
) {
    let len = child.len();
    // the unvisited genes, for a random jump, and the position of each gene in it
    unvisited.clear();
    unvisited.extend(0..len);
    index.clear();
    index.extend(0..len);
    let mut current = start;
    for slot in child.iter_mut() {
        *slot = current;
        let at = index[current];
        unvisited.swap_remove(at);
        if at < unvisited.len() {
            index[unvisited[at]] = at;
        }
        if unvisited.is_empty() {
            return;
        }
        // the current gene is visited: no longer anyone's candidate
        let candidates = std::mem::take(&mut neighbors[current]);
        for &neighbor in candidates.as_slice() {
            neighbors[neighbor].remove(current);
        }
        current = if candidates.len == 0 {
            unvisited[rng.below(unvisited.len())]
        } else {
            // the candidate with the fewest remaining neighbors, ties at random
            let fewest = candidates
                .as_slice()
                .iter()
                .map(|&gene| neighbors[gene].len)
                .min();
            let mut tied = Neighbors::default();
            for &gene in candidates.as_slice() {
                if Some(neighbors[gene].len) == fewest {
                    tied.add(gene);
                }
            }
            if tied.len == 1 {
                tied.genes[0]
            } else {
                tied.genes[rng.below(tied.len)]
            }
        };
    }
}

impl Crossover<Permutation> for EdgeRecombinationCrossover {
    fn crossover(&self, _: &Permutation, a: &mut Order, b: &mut Order, rng: &mut StreamRng) {
        if a.len() < 3 {
            return;
        }
        with_scratch(|scratch| {
            // both children's edges before either parent is overwritten
            let [first, second] = &mut scratch.edges;
            edges(a, b, first);
            edges(b, a, second);
            let (first_start, second_start) = (a[0], b[0]);
            let (unvisited, index) = (&mut scratch.genes, &mut scratch.positions);
            edge_recombination(first, first_start, a.genes_mut(), unvisited, index, rng);
            edge_recombination(second, second_start, b.genes_mut(), unvisited, index, rng);
        });
    }
}

/// Inversion mutation (2-opt) for [`Permutation`] genomes: reverses a random segment of at least
/// two genes.
///
/// In a tour, this replaces two edges and keeps all the others, which makes it the standard move
/// for routing problems. The genome always changes (unless it has fewer than 2 genes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InversionMutation;

impl Mutate<Permutation> for InversionMutation {
    fn mutate(&self, _: &Permutation, genome: &mut Order, rng: &mut StreamRng) {
        let len = genome.len();
        if len < 2 {
            return;
        }
        // reversing the whole genome changes it too, for a sequence
        let (start, end) = loop {
            let (start, end) = rng.sample_pair(len + 1);
            if end - start >= 2 {
                break (start, end);
            }
        };
        genome.genes_mut()[start..end].reverse();
    }
}

/// Insertion mutation for [`Permutation`] genomes: moves a random gene to another random position,
/// shifting the genes in between.
///
/// The genome always changes (unless it has a single gene).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InsertionMutation;

impl Mutate<Permutation> for InsertionMutation {
    fn mutate(&self, _: &Permutation, genome: &mut Order, rng: &mut StreamRng) {
        let len = genome.len();
        if len < 2 {
            return;
        }
        let from = rng.below(len);
        let to = rng.below(len - 1);
        let to = if to >= from { to + 1 } else { to };
        let genes = genome.genes_mut();
        if from < to {
            genes[from..=to].rotate_left(1);
        } else {
            genes[to..=from].rotate_right(1);
        }
    }
}

/// Scramble mutation for [`Permutation`] genomes: shuffles a random segment of at least two genes.
///
/// The genome always changes (unless it has a single gene): a shuffle that happens to give the
/// same order is shuffled again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScrambleMutation;

impl Mutate<Permutation> for ScrambleMutation {
    fn mutate(&self, _: &Permutation, genome: &mut Order, rng: &mut StreamRng) {
        let len = genome.len();
        if len < 2 {
            return;
        }
        let (start, end) = loop {
            let (start, end) = rng.sample_pair(len + 1);
            if end - start >= 2 {
                break (start, end);
            }
        };
        let segment = &mut genome.genes_mut()[start..end];
        with_scratch(|scratch| {
            let original = &mut scratch.genes;
            original.clear();
            original.extend_from_slice(segment);
            while segment == original.as_slice() {
                // Fisher-Yates
                for position in (1..segment.len()).rev() {
                    segment.swap(position, rng.below(position + 1));
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Representation;
    use proptest::prelude::*;

    fn is_permutation(order: &Order) -> bool {
        Order::new(order.to_vec()).is_ok()
    }

    fn parents(len: usize, seed: u64) -> (Permutation, Order, Order, StreamRng) {
        let permutation = Permutation::new(len).unwrap();
        let mut rng = StreamRng::seed_from_u64(seed);
        let a = permutation.random_genome(&mut rng);
        let b = permutation.random_genome(&mut rng);
        (permutation, a, b, rng)
    }

    // the child of OX that keeps `own`'s segment
    fn ordered_child(own: &[usize], other: &[usize], start: usize, end: usize) -> Vec<usize> {
        let mut child = own.to_vec();
        ordered(&mut child, other, start, end, &mut vec![false; own.len()]);
        child
    }

    // the child of PMX that takes `other`'s segment
    fn mapped_child(own: &[usize], other: &[usize], start: usize, end: usize) -> Vec<usize> {
        let mut child = own.to_vec();
        partially_mapped(
            &mut child,
            &other[start..end],
            start,
            &mut vec![0; own.len()],
        );
        child
    }

    // the classic example, 0-based: 1 2 3 | 4 5 6 7 | 8 9 and 4 5 2 | 1 8 7 6 | 9 3
    const A: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const B: [usize; 9] = [3, 4, 1, 0, 7, 6, 5, 8, 2];

    #[test]
    fn order_crossover_example() {
        // 2 1 8 | 4 5 6 7 | 9 3
        assert_eq!(ordered_child(&A, &B, 3, 7), [1, 0, 7, 3, 4, 5, 6, 8, 2]);
    }

    #[test]
    fn partially_mapped_example() {
        // 4 2 3 | 1 8 7 6 | 5 9
        assert_eq!(mapped_child(&A, &B, 3, 7), [3, 1, 2, 0, 7, 6, 5, 4, 8]);
    }

    #[test]
    fn cycle_crossover_example() {
        // cycles {0, 3, 6, 7} (from A), {1, 2, 4} (from B), {5} (from A)
        let a = Order::new(vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let b = Order::new(vec![7, 4, 1, 0, 2, 5, 3, 6]).unwrap();
        let (mut x, mut y) = (a.clone(), b.clone());
        let permutation = Permutation::new(8).unwrap();
        CycleCrossover.crossover(
            &permutation,
            &mut x,
            &mut y,
            &mut StreamRng::seed_from_u64(0),
        );
        assert_eq!(&x[..], &[0, 4, 1, 3, 2, 5, 6, 7]);
        assert_eq!(&y[..], &[7, 1, 2, 0, 4, 5, 3, 6]);
    }

    // a crossover of two genomes, whatever its type
    type Recombine<'a> = &'a dyn Fn(&mut Order, &mut Order, &mut StreamRng);

    #[test]
    fn scratch_space_of_earlier_calls_changes_nothing() {
        // every operator on a pair of genomes, on a new thread, after `before` genes if any
        let children = |before: Option<usize>| {
            std::thread::spawn(move || {
                let mut children = Vec::new();
                for len in before.into_iter().chain([20]) {
                    let (permutation, a, b, mut rng) = parents(len, 1);
                    let crossovers: [Recombine; 4] = [
                        &|x, y, rng| PartiallyMappedCrossover.crossover(&permutation, x, y, rng),
                        &|x, y, rng| OrderCrossover.crossover(&permutation, x, y, rng),
                        &|x, y, rng| CycleCrossover.crossover(&permutation, x, y, rng),
                        &|x, y, rng| EdgeRecombinationCrossover.crossover(&permutation, x, y, rng),
                    ];
                    for crossover in crossovers {
                        let (mut x, mut y) = (a.clone(), b.clone());
                        crossover(&mut x, &mut y, &mut rng);
                        children.extend([x, y]);
                    }
                    let mut genome = a.clone();
                    ScrambleMutation.mutate(&permutation, &mut genome, &mut rng);
                    children.push(genome);
                }
                children.split_off(children.len() - 9)
            })
            .join()
            .unwrap()
        };
        let fresh = children(None);
        assert_eq!(children(Some(50)), fresh);
        assert_eq!(children(Some(5)), fresh);
    }

    fn edges(tour: &[usize]) -> Vec<(usize, usize)> {
        let len = tour.len();
        (0..len)
            .map(|i| {
                let (x, y) = (tour[i], tour[(i + 1) % len]);
                (x.min(y), x.max(y))
            })
            .collect()
    }

    proptest! {
        #[test]
        fn crossovers_give_permutations(len in 1usize..40, seed: u64) {
            let (permutation, a, b, mut rng) = parents(len, seed);
            let crossovers: [Recombine; 4] = [
                &|x, y, rng| PartiallyMappedCrossover.crossover(&permutation, x, y, rng),
                &|x, y, rng| OrderCrossover.crossover(&permutation, x, y, rng),
                &|x, y, rng| CycleCrossover.crossover(&permutation, x, y, rng),
                &|x, y, rng| EdgeRecombinationCrossover.crossover(&permutation, x, y, rng),
            ];
            for crossover in crossovers {
                let (mut x, mut y) = (a.clone(), b.clone());
                crossover(&mut x, &mut y, &mut rng);
                prop_assert!(is_permutation(&x) && is_permutation(&y));
                prop_assert_eq!(x.len(), len);
            }
        }

        #[test]
        fn segment_crossovers(len in 2usize..40, seed: u64) {
            let (_, a, b, mut rng) = parents(len, seed);
            let (start, end) = segment(len, 1, &mut rng);
            prop_assert!(start < end && end - start < len);
            // OX keeps its own segment, and the other genes in the order of the other parent
            let child = ordered_child(&a, &b, start, end);
            prop_assert_eq!(&child[start..end], &a[start..end]);
            let rest: Vec<usize> = (end..end + len).map(|i| child[i % len]).take(len - (end - start)).collect();
            let expected: Vec<usize> = (end..end + len).map(|i| b[i % len]).filter(|gene| !a[start..end].contains(gene)).collect();
            prop_assert_eq!(rest, expected);
            // PMX takes the other segment, and keeps its own genes where they don't conflict
            let child = mapped_child(&a, &b, start, end);
            prop_assert_eq!(&child[start..end], &b[start..end]);
            for position in (0..start).chain(end..len) {
                if !b[start..end].contains(&a[position]) {
                    prop_assert_eq!(child[position], a[position]);
                }
            }
        }

        #[test]
        fn cycle_crossover_keeps_positions(len in 1usize..40, seed: u64) {
            let (permutation, a, b, mut rng) = parents(len, seed);
            let (mut x, mut y) = (a.clone(), b.clone());
            CycleCrossover.crossover(&permutation, &mut x, &mut y, &mut rng);
            for position in 0..len {
                prop_assert!(x[position] == a[position] || x[position] == b[position]);
                // complementary: y has the other parent's gene
                prop_assert_eq!(x[position] == a[position], y[position] == b[position]);
            }
        }

        #[test]
        fn edge_recombination_of_a_tour_with_itself(len in 3usize..40, seed: u64) {
            let (permutation, a, _, mut rng) = parents(len, seed);
            let (mut x, mut y) = (a.clone(), a.clone());
            EdgeRecombinationCrossover.crossover(&permutation, &mut x, &mut y, &mut rng);
            let mut parent_edges = edges(&a);
            parent_edges.sort_unstable();
            for child in [&x, &y] {
                let mut child_edges = edges(child);
                child_edges.sort_unstable();
                prop_assert_eq!(&child_edges, &parent_edges);
            }
        }

        #[test]
        fn mutations_change_and_keep_permutations(len in 2usize..40, seed: u64) {
            let (permutation, original, _, mut rng) = parents(len, seed);
            let changed = |genome: &Order| (0..len).filter(|&i| genome[i] != original[i]).collect::<Vec<_>>();

            // inversion: exactly one segment, reversed
            let mut genome = original.clone();
            InversionMutation.mutate(&permutation, &mut genome, &mut rng);
            prop_assert!(is_permutation(&genome));
            let positions = changed(&genome);
            prop_assert!(!positions.is_empty());
            let (first, last) = (positions[0], *positions.last().unwrap());
            for position in first..=last {
                prop_assert_eq!(genome[position], original[first + last - position]);
            }

            // insertion: one gene moved, the others in the same order
            let mut genome = original.clone();
            InsertionMutation.mutate(&permutation, &mut genome, &mut rng);
            prop_assert!(is_permutation(&genome));
            prop_assert!(!changed(&genome).is_empty());
            let moved = (0..len).any(|gene| {
                let without = |order: &Order| order.iter().copied().filter(|&g| g != gene).collect::<Vec<_>>();
                without(&genome) == without(&original)
            });
            prop_assert!(moved);

            // scramble: changes within one segment
            let mut genome = original.clone();
            ScrambleMutation.mutate(&permutation, &mut genome, &mut rng);
            prop_assert!(is_permutation(&genome));
            prop_assert!(!changed(&genome).is_empty());
        }
    }
}
