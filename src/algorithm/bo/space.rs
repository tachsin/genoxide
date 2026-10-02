//! The representations Bayesian optimization searches: [`Real`] genomes, and [`Integer`] genomes
//! with their genes rounded inside the kernel.

// the sealed trait's methods take the crate's own unit-cube map: unnameable outside the crate
#![allow(private_interfaces)]

use crate::genome::{Integer, Integers, Real, Reals, Representation};
use crate::model::gp::Scaling;
use crate::{Error, Result, StreamRng};

/// A representation that [`Bo`](super::Bo) searches: [`Real`], or [`Integer`]. Sealed: its
/// methods are genoxide's own.
///
/// - **[`Real`]**: the model's inputs are the genes, scaled to the unit cube by the bounds, and
///   the acquisition function is maximized by L-BFGS-B with its gradient.
/// - **[`Integer`]**: each gene's values are scaled to the unit cube by the bounds alike, and
///   every point the model is given or asked about is a point of the integer lattice: Garrido-
///   Merchán and Hernández-Lobato's (2020) transformation, the genes rounded to the nearest
///   integer inside the kernel, so that the model is constant between integers and certain at
///   an evaluated point (their eq. 7). The acquisition function is then a function of the lattice,
///   maximized there: at the raw samples, random lattice points (or every point, for a lattice of
///   at most that many), then by a hill climb from the best of them and from the best point
///   evaluated, each step to the best of the neighbors one away in one gene, until none is
///   better. Once every point of the lattice is evaluated, the search
///   [has finished](crate::algorithm::Algorithm::is_finished).
///
/// Reference: Garrido-Merchán, E. C. and Hernández-Lobato, D. (2020). Dealing with categorical and
/// integer-valued variables in Bayesian optimization with Gaussian processes. *Neurocomputing*
/// 380: 20-35.
pub trait Space: sealed::Sealed {}

impl Space for Real {}
impl Space for Integer {}

pub(crate) mod sealed {
    use super::*;

    /// What Bayesian optimization needs of a representation.
    pub trait Sealed: Representation {
        // the lower and upper bounds of each gene, as reals
        fn real_bounds(&self) -> Vec<(f64, f64)>;

        // the map between genomes and the model's unit cube
        fn scaling(&self) -> Scaling {
            Scaling::from_bounds(&self.real_bounds())
        }

        // the unit-cube coordinates of `genome`
        fn to_unit(scaling: &Scaling, genome: &Self::Genome, unit: &mut [f64]);

        // the genome at the unit-cube coordinates `unit`, in the bounds: on the lattice, the
        // nearest point
        fn genome_at(&self, scaling: &Scaling, unit: &[f64]) -> Self::Genome;

        // the genes of `genome` as reals, for the model's predictions
        fn gene_values(genome: &Self::Genome) -> Vec<f64>;

        // `n` genomes of a Latin hypercube
        fn design(&self, n: usize, rng: &mut StreamRng) -> Result<Vec<Self::Genome>>;

        // the number of points of an integer lattice (saturating), None for reals
        fn lattice(&self) -> Option<u128>;

        // every point of the lattice, if there are at most `limit`
        fn enumerate(&self, limit: usize) -> Option<Vec<Self::Genome>>;

        // the lattice's neighbors of `genome`: one away in one gene, within the bounds
        fn neighbors(&self, genome: &Self::Genome, neighbors: &mut Vec<Self::Genome>);
    }
}

impl sealed::Sealed for Real {
    fn real_bounds(&self) -> Vec<(f64, f64)> {
        self.bounds()
            .iter()
            .map(|range| (*range.start(), *range.end()))
            .collect()
    }

    fn scaling(&self) -> Scaling {
        Scaling::new(self)
    }

    fn to_unit(scaling: &Scaling, genome: &Reals, unit: &mut [f64]) {
        scaling.to_unit(genome, unit);
    }

    fn genome_at(&self, scaling: &Scaling, unit: &[f64]) -> Reals {
        scaling.to_genome(unit)
    }

    fn gene_values(genome: &Reals) -> Vec<f64> {
        genome.to_vec()
    }

    fn design(&self, n: usize, rng: &mut StreamRng) -> Result<Vec<Reals>> {
        self.latin_hypercube(n, rng)
    }

    fn lattice(&self) -> Option<u128> {
        None
    }

    fn enumerate(&self, _limit: usize) -> Option<Vec<Reals>> {
        None
    }

    fn neighbors(&self, _genome: &Reals, _neighbors: &mut Vec<Reals>) {}
}

// the value of an integer gene nearest to `x`, in [lower, upper]
fn nearest(x: f64, lower: i64, upper: i64) -> i64 {
    // `as` saturates, and a NaN becomes 0, which the clamp moves into the bounds
    (x.round() as i64).clamp(lower, upper)
}

impl sealed::Sealed for Integer {
    fn real_bounds(&self) -> Vec<(f64, f64)> {
        self.bounds()
            .iter()
            .map(|range| (*range.start() as f64, *range.end() as f64))
            .collect()
    }

    fn to_unit(scaling: &Scaling, genome: &Integers, unit: &mut [f64]) {
        scaling.to_unit_by(|i| genome[i] as f64, unit);
    }

    fn genome_at(&self, scaling: &Scaling, unit: &[f64]) -> Integers {
        let bounds = self.bounds();
        let mut genome: Vec<i64> = bounds.iter().map(|range| *range.start()).collect();
        for (k, &i) in scaling.variable().iter().enumerate() {
            let (low, high) = (*bounds[i].start(), *bounds[i].end());
            genome[i] = nearest(scaling.gene_at(k, unit[k]), low, high);
        }
        Integers::from(genome)
    }

    fn gene_values(genome: &Integers) -> Vec<f64> {
        genome.iter().map(|&gene| gene as f64).collect()
    }

    fn design(&self, n: usize, rng: &mut StreamRng) -> Result<Vec<Integers>> {
        if n == 0 {
            return Err(Error::InvalidSetting {
                setting: "n",
                reason: "a Latin hypercube needs at least 1 genome".to_string(),
            });
        }
        crate::operator::check_size("n", n)?;
        let mut genomes = vec![Vec::with_capacity(self.genome_len()); n];
        let mut strata: Vec<usize> = Vec::with_capacity(n);
        for range in self.bounds() {
            let (low, high) = (*range.start(), *range.end());
            if low == high {
                for genome in &mut genomes {
                    genome.push(low);
                }
                continue;
            }
            strata.clear();
            strata.extend(0..n);
            for last in (1..n).rev() {
                let other = rng.below(last + 1);
                strata.swap(last, other);
            }
            // the values low..=high as the intervals [low − ½, high + ½), stratified like reals
            let values = (high - low) as f64 + 1.0;
            for (genome, &stratum) in genomes.iter_mut().zip(&strata) {
                let u = (stratum as f64 + rng.unit_f64()) / n as f64;
                let x = low as f64 - 0.5 + u * values;
                genome.push(nearest(x, low, high));
            }
        }
        Ok(genomes.into_iter().map(Integers::from).collect())
    }

    fn lattice(&self) -> Option<u128> {
        let mut points: u128 = 1;
        for range in self.bounds() {
            let values = u128::from(range.end().abs_diff(*range.start())) + 1;
            points = points.saturating_mul(values);
        }
        Some(points)
    }

    fn enumerate(&self, limit: usize) -> Option<Vec<Integers>> {
        let points = self.lattice()?;
        if points > limit as u128 {
            return None;
        }
        let bounds = self.bounds();
        let mut genome: Vec<i64> = bounds.iter().map(|range| *range.start()).collect();
        let mut all = Vec::with_capacity(points as usize);
        loop {
            all.push(Integers::from(genome.clone()));
            // the next point, the last gene fastest
            let mut gene = genome.len();
            loop {
                if gene == 0 {
                    return Some(all);
                }
                gene -= 1;
                if genome[gene] < *bounds[gene].end() {
                    genome[gene] += 1;
                    break;
                }
                genome[gene] = *bounds[gene].start();
            }
        }
    }

    fn neighbors(&self, genome: &Integers, neighbors: &mut Vec<Integers>) {
        neighbors.clear();
        for (gene, range) in self.bounds().iter().enumerate() {
            let value = genome[gene];
            if value > *range.start() {
                let mut neighbor = genome.clone();
                neighbor[gene] = value - 1;
                neighbors.push(neighbor);
            }
            if value < *range.end() {
                let mut neighbor = genome.clone();
                neighbor[gene] = value + 1;
                neighbors.push(neighbor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sealed::Sealed;
    use super::*;

    #[test]
    fn integer_genomes_round_trip_through_the_unit_cube() {
        let integer = Integer::new([-3..=4, 7..=7, 0..=1]).unwrap();
        let scaling = integer.scaling();
        assert_eq!(scaling.dims(), 2);
        for genome in integer.enumerate(100).unwrap() {
            let mut unit = [0.0; 2];
            Integer::to_unit(&scaling, &genome, &mut unit);
            assert!(unit.iter().all(|u| (0.0..=1.0).contains(u)));
            assert_eq!(integer.genome_at(&scaling, &unit), genome);
        }
        assert_eq!(integer.lattice(), Some(16));
        assert_eq!(integer.enumerate(15), None);
        // the nearest lattice point of any unit-cube point
        let genome = integer.genome_at(&scaling, &[0.5, 0.49]);
        assert_eq!(genome[..], [1, 7, 0]);
    }

    #[test]
    fn an_integer_latin_hypercube_stratifies_each_gene() {
        let integer = Integer::new([0..=9, 5..=5]).unwrap();
        let design = integer
            .design(10, &mut StreamRng::seed_from_u64(3))
            .unwrap();
        let mut first: Vec<i64> = design.iter().map(|genome| genome[0]).collect();
        first.sort_unstable();
        assert_eq!(first, (0..=9).collect::<Vec<_>>());
        assert!(design.iter().all(|genome| genome[1] == 5));
    }

    #[test]
    fn neighbors_stay_in_the_bounds() {
        let integer = Integer::new([0..=2, -1..=1]).unwrap();
        let mut neighbors = Vec::new();
        integer.neighbors(&Integers::from(vec![0, 1]), &mut neighbors);
        assert_eq!(
            neighbors,
            [Integers::from(vec![1, 1]), Integers::from(vec![0, 0])]
        );
    }
}
