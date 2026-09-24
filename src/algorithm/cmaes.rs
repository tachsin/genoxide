//! The covariance matrix adaptation evolution strategy (CMA-ES), with IPOP and BIPOP restarts.

use super::{Algorithm, Candidates};
use crate::genome::{Real, Reals, Representation};
use crate::math::{exp, log};
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use std::collections::VecDeque;

/// Whether a [`Cmaes`] starts a new run when the current one has converged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Restarts {
    /// No restarts: a converged run goes on sampling around the same point (the default). Stop
    /// it with a stop condition such as [`Stop::stagnation`](crate::Stop::stagnation).
    #[default]
    Never,
    /// IPOP-CMA-ES (Auger and Hansen, 2005): each restart doubles the population size and starts
    /// from a random point with the initial step size. Larger populations smooth out local
    /// optima, which suits multimodal functions with a global structure, like Rastrigin.
    Ipop,
    /// BIPOP-CMA-ES (Hansen, 2009): restarts alternate between a large population regime, which
    /// doubles the population like IPOP, and a small one with a random smaller population and a
    /// random step size down to 1/100 of the initial one. The regime that has used fewer
    /// evaluations goes next; the first run counts as a large one. Good on a wider range of
    /// multimodal functions than IPOP.
    Bipop,
}

/// Why a run of a [`Cmaes`] has converged: the stop criteria of Hansen (2009), with the
/// population's fitness and the search distribution in coordinates scaled to `0..=1` per gene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Criterion {
    /// The best scores of the last `10 + ⌈30 n / λ⌉` generations, and all scores of the last
    /// generation, are within 1e-12.
    TolHistFun,
    /// In more than a third of the last `n` generations, the best fitness equals the fitness of
    /// the `1 + ⌈0.1 + λ / 4⌉`-th best: a plateau.
    EqualFunValues,
    /// The step size times every standard deviation and evolution path component is below 1e-12
    /// times the initial step size.
    TolX,
    /// The step size times the largest standard deviation grew above 1000 times the initial step
    /// size: the function is probably unbounded, or the step size diverged.
    TolUpX,
    /// Adding 0.1 standard deviations along a principal axis doesn't change the mean.
    NoEffectAxis,
    /// Adding 0.2 standard deviations in a coordinate doesn't change the mean.
    NoEffectCoord,
    /// The condition number of the covariance matrix exceeds 1e14.
    ConditionCov,
}

// the population of every restart is at most this many times the initial one
const MAX_GROWTH: usize = 1024;

// draws of a sample before an out-of-bounds sample is clipped
const RESAMPLES: usize = 100;

// the strategy parameters for `n` dimensions and `lambda` samples, the defaults of Hansen's 2016
// tutorial with positive recombination weights
#[derive(Clone, Debug)]
struct Parameters {
    lambda: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    // the expected length of a standard normal vector
    chi_n: f64,
    // generations between eigendecompositions
    eigen_interval: u64,
    // generations of best fitness values for TolHistFun
    history: usize,
}

impl Parameters {
    fn new(n: usize, lambda: usize) -> Self {
        let dimensions = n as f64;
        let mu = lambda / 2;
        let raw: Vec<f64> = (1..=mu)
            .map(|i| log((lambda as f64 + 1.0) / 2.0) - log(i as f64))
            .collect();
        let sum: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / sum).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        let c_sigma = (mu_eff + 2.0) / (dimensions + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (((mu_eff - 1.0) / (dimensions + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / dimensions) / (dimensions + 4.0 + 2.0 * mu_eff / dimensions);
        let c_1 = 2.0 / ((dimensions + 1.3) * (dimensions + 1.3) + mu_eff);
        let c_mu = (1.0 - c_1).min(
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
                / ((dimensions + 2.0) * (dimensions + 2.0) + mu_eff),
        );
        let chi_n = dimensions.sqrt()
            * (1.0 - 1.0 / (4.0 * dimensions) + 1.0 / (21.0 * dimensions * dimensions));
        let eigen_interval = (1.0 / ((c_1 + c_mu) * dimensions * 10.0)).max(1.0) as u64;
        let history = 10 + (30.0 * dimensions / lambda as f64).ceil() as usize;
        Self {
            lambda,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
            eigen_interval,
            history,
        }
    }
}

/// The covariance matrix adaptation evolution strategy on [`Real`] genomes, as an ask / tell
/// [`Algorithm`]: the state of the art for continuous optimization without derivatives, when the
/// problem has up to a few hundred genes.
///
/// Every generation samples `λ` genomes from a multivariate normal distribution `m + σ · N(0, C)`,
/// moves the mean `m` to a weighted average of the best half, and adapts the step size `σ`
/// (cumulative step-size adaptation) and the covariance matrix `C` (rank-one and rank-μ updates)
/// so that good steps become more likely. `C` learns the scaling and the correlations of the
/// genes, which makes the search invariant to rotations and to the conditioning of the problem.
/// The parameters are the defaults of Hansen's tutorial (2016); only the population size and the
/// initial step size are settings.
///
/// The search works in coordinates scaled to `0..=1` per gene, so genes with different ranges
/// start with the same relative step size, and genes with a single value are left out. A sample
/// outside the bounds is drawn again, up to 100 times, and then clipped; the distribution learns
/// from the samples as evaluated. When a run meets one of the stop criteria ([`Criterion`]),
/// [`Restarts`] can start a new one.
///
/// Every random decision and every math function (a Householder and QL eigendecomposition,
/// fdlibm's `log` and `exp`) is portable, so a seed gives the same run on every platform.
///
/// Built with [`Cmaes::builder`], run with an [`Engine`](crate::Engine).
///
/// ```
/// use genoxide::prelude::*;
///
/// // a rotated, badly conditioned function
/// let ellipsoid = |x: &Reals| {
///     x.windows(2).map(|w| (w[0] + w[1]).powi(2) + 1e4 * (w[0] - w[1]).powi(2)).sum::<f64>()
/// };
/// let cmaes = Cmaes::builder(Real::uniform(8, -5.0..=5.0)?).minimize().seed(1).build()?;
/// let outcome = Engine::new(cmaes, ellipsoid)
///     .stop_when(Stop::target(1e-10).or(Stop::evaluations(50_000)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Cmaes {
    real: Real,
    // the indices of the genes with more than one value, the dimensions of the search
    free: Vec<usize>,
    restarts: Restarts,
    initial_step: f64,
    initial_lambda: usize,
    objective: Objective,
    seed: u64,
    rng: StreamRng,
    parameters: Parameters,
    // the distribution in scaled coordinates: C = B · diag(D²) · Bᵀ, matrices row-major
    mean: Vec<f64>,
    sigma: f64,
    run_sigma: f64,
    covariance: Vec<f64>,
    basis: Vec<f64>,
    deviations: Vec<f64>,
    path_sigma: Vec<f64>,
    path_c: Vec<f64>,
    // (1 − c_σ)^(2 (g + 1)), for the bias correction of h_σ
    decay: f64,
    run_generation: u64,
    eigen_generation: u64,
    // the scaled samples of the population and their steps (x − m) / σ
    samples: Vec<Vec<f64>>,
    steps: Vec<Vec<f64>>,
    best_scores: VecDeque<Option<f64>>,
    equal_values: VecDeque<bool>,
    converged: Option<Criterion>,
    // restarts: how many, the large population, and the evaluations of each BIPOP regime
    restart_count: u64,
    large_lambda: usize,
    large_evaluations: u64,
    small_evaluations: u64,
    small_run: bool,
    run_evaluations: u64,
    population: Population<Reals>,
    started: bool,
    asked: bool,
    pending: Vec<usize>,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Reals>>,
    best_generation: u64,
}

impl Cmaes {
    /// A builder for a CMA-ES on `real`.
    pub fn builder(real: Real) -> CmaesBuilder {
        CmaesBuilder {
            real,
            population_size: None,
            initial_step: 0.3,
            initial_mean: None,
            restarts: Restarts::Never,
            objective: Objective::default(),
            seed: None,
        }
    }

    /// The default population size for `n` genes, `4 + ⌊3 ln n⌋`: 10 for 10 genes, 14 for 100.
    pub fn default_population_size(n: usize) -> usize {
        4 + (3.0 * log(n.max(1) as f64)) as usize
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// The mean of the search distribution, the current estimate of the optimum.
    pub fn mean(&self) -> Reals {
        self.to_genome(&self.mean)
    }

    /// The step size `σ`, as a fraction of the range of each gene.
    pub fn step_size(&self) -> f64 {
        self.sigma
    }

    /// The standard deviations of the search distribution along its principal axes, as fractions
    /// of the ranges: `σ` times the square roots of the eigenvalues of `C`, from the latest
    /// eigendecomposition.
    pub fn axis_deviations(&self) -> Vec<f64> {
        self.deviations.iter().map(|d| self.sigma * d).collect()
    }

    /// Why the current run has converged, or `None` if it hasn't. With [`Restarts`], the next
    /// [`ask`](Algorithm::ask) starts a new run.
    pub fn converged(&self) -> Option<Criterion> {
        self.converged
    }

    /// The number of restarts so far.
    pub fn restart_count(&self) -> u64 {
        self.restart_count
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn dimensions(&self) -> usize {
        self.free.len()
    }

    // the genome of the scaled coordinates `u`
    fn to_genome(&self, u: &[f64]) -> Reals {
        let bounds = self.real.bounds();
        let mut genes: Vec<f64> = bounds.iter().map(|range| *range.start()).collect();
        for (&gene, &value) in self.free.iter().zip(u) {
            let (start, end) = (*bounds[gene].start(), *bounds[gene].end());
            genes[gene] = (start + (end - start) * value).clamp(start, end);
        }
        Reals::from(genes)
    }

    // a new run with `lambda` samples, step size `sigma` and a mean at `mean`
    fn start_run(&mut self, lambda: usize, sigma: f64, mean: Vec<f64>) {
        let n = self.dimensions();
        self.parameters = Parameters::new(n, lambda);
        self.mean = mean;
        self.sigma = sigma;
        self.run_sigma = sigma;
        self.covariance = identity(n);
        self.basis = identity(n);
        self.deviations = vec![1.0; n];
        self.path_sigma = vec![0.0; n];
        self.path_c = vec![0.0; n];
        self.decay = 1.0;
        self.run_generation = 0;
        self.eigen_generation = 0;
        self.best_scores.clear();
        self.equal_values.clear();
        self.converged = None;
        self.run_evaluations = 0;
    }

    fn restart(&mut self) {
        let finished = self.run_evaluations;
        if self.small_run {
            self.small_evaluations += finished;
        } else {
            self.large_evaluations += finished;
        }
        self.restart_count += 1;
        let (lambda, sigma, small) = match self.restarts {
            Restarts::Bipop if self.small_evaluations < self.large_evaluations => {
                // λ = ⌊λ₀ (λ_large / (2 λ₀))^(U²)⌋ and σ = σ₀ · 10^(−2 U), U uniform in 0..1
                let u = self.rng.unit_f64();
                let ratio = 0.5 * self.large_lambda as f64 / self.initial_lambda as f64;
                let lambda = (self.initial_lambda as f64 * exp(u * u * log(ratio))) as usize;
                let u = self.rng.unit_f64();
                let sigma = self.initial_step * exp(-2.0 * u * std::f64::consts::LN_10);
                (lambda.max(2), sigma, true)
            }
            _ => {
                let limit = self.initial_lambda.saturating_mul(MAX_GROWTH);
                self.large_lambda = self.large_lambda.saturating_mul(2).min(limit);
                (self.large_lambda, self.initial_step, false)
            }
        };
        self.small_run = small;
        let mean = (0..self.dimensions())
            .map(|_| self.rng.unit_f64())
            .collect();
        self.start_run(lambda, sigma, mean);
    }

    // samples a new population
    fn sample(&mut self) {
        let n = self.dimensions();
        let lambda = self.parameters.lambda;
        self.samples.clear();
        self.steps.clear();
        let mut z = vec![0.0; n];
        for _ in 0..lambda {
            let mut accepted = None;
            let mut last = (Vec::new(), Vec::new());
            for _ in 0..RESAMPLES {
                for value in z.iter_mut() {
                    *value = self.rng.normal();
                }
                // y = B · D · z
                let step: Vec<f64> = (0..n)
                    .map(|i| {
                        let row = &self.basis[i * n..(i + 1) * n];
                        row.iter()
                            .zip(&self.deviations)
                            .zip(&z)
                            .map(|((b, d), z)| b * d * z)
                            .sum()
                    })
                    .collect();
                let u: Vec<f64> = self
                    .mean
                    .iter()
                    .zip(&step)
                    .map(|(m, y)| m + self.sigma * y)
                    .collect();
                if u.iter().all(|u| (0.0..=1.0).contains(u)) {
                    accepted = Some((u, step));
                    break;
                }
                last = (u, step);
            }
            let (u, step) = accepted.unwrap_or_else(|| {
                // clipped, and the step it takes from the mean
                let u: Vec<f64> = last
                    .0
                    .iter()
                    .zip(&self.mean)
                    .map(|(&u, &m)| if u.is_nan() { m } else { u.clamp(0.0, 1.0) })
                    .collect();
                let step = if self.sigma > 0.0 {
                    u.iter()
                        .zip(&self.mean)
                        .map(|(u, m)| (u - m) / self.sigma)
                        .collect()
                } else {
                    vec![0.0; n]
                };
                (u, step)
            });
            self.samples.push(u);
            self.steps.push(step);
        }
        let genomes: Vec<Reals> = self.samples.iter().map(|u| self.to_genome(u)).collect();
        self.population = Population::from_genomes(genomes);
    }

    // the distribution update from the evaluated population
    fn update(&mut self) {
        let n = self.dimensions();
        let objective = self.objective;
        let fitness: Vec<Fitness> = self
            .population
            .iter()
            .map(|individual| individual.fitness().unwrap_or(Fitness::invalid()))
            .collect();
        // best first, the earlier one on ties
        let mut order: Vec<usize> = (0..fitness.len()).collect();
        order.sort_by(|&a, &b| objective.compare(fitness[b], fitness[a]));
        let p = &self.parameters;
        let (c_sigma, c_c, c_1, c_mu, mu_eff) = (p.c_sigma, p.c_c, p.c_1, p.c_mu, p.mu_eff);
        let weights = p.weights.clone();
        // the weighted mean step, and the new mean
        let mut mean_step = vec![0.0; n];
        for (&weight, &index) in weights.iter().zip(&order) {
            for (total, y) in mean_step.iter_mut().zip(&self.steps[index]) {
                *total += weight * y;
            }
        }
        for (m, y) in self.mean.iter_mut().zip(&mean_step) {
            *m = (*m + self.sigma * y).clamp(0.0, 1.0);
        }
        // the evolution path of σ, with C^(−1/2) · y_w = B · D⁻¹ · Bᵀ · y_w
        let rotated: Vec<f64> = (0..n)
            .map(|k| {
                let projection: f64 = (0..n).map(|i| self.basis[i * n + k] * mean_step[i]).sum();
                projection / self.deviations[k]
            })
            .collect();
        let factor = (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt();
        for i in 0..n {
            let whitened: f64 = (0..n).map(|k| self.basis[i * n + k] * rotated[k]).sum();
            self.path_sigma[i] = (1.0 - c_sigma) * self.path_sigma[i] + factor * whitened;
        }
        let norm = self.path_sigma.iter().map(|p| p * p).sum::<f64>().sqrt();
        self.decay *= (1.0 - c_sigma) * (1.0 - c_sigma);
        let chi_n = self.parameters.chi_n;
        // h_σ: whether the path is short enough for the rank-one update to be unbiased
        let h_sigma = norm / (1.0 - self.decay).sqrt() < (1.4 + 2.0 / (n as f64 + 1.0)) * chi_n;
        // the evolution path of C
        let factor = if h_sigma {
            (c_c * (2.0 - c_c) * mu_eff).sqrt()
        } else {
            0.0
        };
        for (path, y) in self.path_c.iter_mut().zip(&mean_step) {
            *path = (1.0 - c_c) * *path + factor * y;
        }
        // the rank-one and rank-μ updates
        let delta = if h_sigma { 0.0 } else { c_c * (2.0 - c_c) };
        let keep = 1.0 + c_1 * delta - c_1 - c_mu;
        for i in 0..n {
            for j in 0..=i {
                let rank_mu: f64 = weights
                    .iter()
                    .zip(&order)
                    .map(|(w, &index)| w * self.steps[index][i] * self.steps[index][j])
                    .sum();
                let value = keep * self.covariance[i * n + j]
                    + c_1 * self.path_c[i] * self.path_c[j]
                    + c_mu * rank_mu;
                self.covariance[i * n + j] = value;
                self.covariance[j * n + i] = value;
            }
        }
        // cumulative step-size adaptation, at most a factor e per generation
        let change = (c_sigma / self.parameters.d_sigma) * (norm / chi_n - 1.0);
        self.sigma *= exp(change.min(1.0));
        self.run_generation += 1;
        if self.run_generation - self.eigen_generation >= self.parameters.eigen_interval {
            self.decompose();
        }
        self.check(&fitness, &order);
    }

    // the eigendecomposition of C, conditioned to at most 1e14
    fn decompose(&mut self) {
        let n = self.dimensions();
        self.eigen_generation = self.run_generation;
        let (mut values, vectors) = eigen(&self.covariance, n);
        let max = values.iter().copied().fold(f64::MIN, f64::max);
        let min = values.iter().copied().fold(f64::MAX, f64::min);
        if !(min > 0.0 && max <= 1e14 * min) {
            self.converged = Some(Criterion::ConditionCov);
            let shift = (max / 1e14 - min).max(0.0) + f64::MIN_POSITIVE;
            for i in 0..n {
                self.covariance[i * n + i] += shift;
            }
            for value in &mut values {
                *value += shift;
            }
        }
        self.basis = vectors;
        self.deviations = values.iter().map(|value| value.max(0.0).sqrt()).collect();
    }

    // records the fitness of a generation and checks the stop criteria
    fn check(&mut self, fitness: &[Fitness], order: &[usize]) {
        let n = self.dimensions();
        let (history, lambda) = (self.parameters.history, self.parameters.lambda);
        let best = fitness[order[0]];
        self.best_scores.push_back(feasible_score(best));
        if self.best_scores.len() > history {
            self.best_scores.pop_front();
        }
        let k = (1 + (0.1 + lambda as f64 / 4.0).ceil() as usize).min(lambda);
        self.equal_values.push_back(best == fitness[order[k - 1]]);
        if self.equal_values.len() > n {
            self.equal_values.pop_front();
        }
        if self.converged.is_none() {
            self.converged = self.criterion(fitness);
        }
    }

    fn criterion(&self, fitness: &[Fitness]) -> Option<Criterion> {
        let n = self.dimensions();
        let sigma = self.sigma;
        if self.best_scores.len() == self.parameters.history
            && range(self.best_scores.iter().copied()).is_some_and(|range| range < 1e-12)
            && range(fitness.iter().map(|&f| feasible_score(f))).is_some_and(|range| range < 1e-12)
        {
            return Some(Criterion::TolHistFun);
        }
        let equal = self.equal_values.iter().filter(|&&equal| equal).count();
        if self.equal_values.len() == n && 3 * equal > n {
            return Some(Criterion::EqualFunValues);
        }
        let tol_x = 1e-12 * self.run_sigma;
        let deviation = |i: usize| self.covariance[i * n + i].sqrt();
        if (0..n).all(|i| sigma * self.path_c[i].abs() < tol_x && sigma * deviation(i) < tol_x) {
            return Some(Criterion::TolX);
        }
        let largest = self.deviations.iter().copied().fold(0.0, f64::max);
        if sigma * largest > 1e3 * self.run_sigma {
            return Some(Criterion::TolUpX);
        }
        let axis = (self.run_generation % n as u64) as usize;
        let scale = 0.1 * sigma * self.deviations[axis];
        if (0..n).all(|i| self.mean[i] + scale * self.basis[i * n + axis] == self.mean[i]) {
            return Some(Criterion::NoEffectAxis);
        }
        if (0..n).any(|i| self.mean[i] + 0.2 * sigma * deviation(i) == self.mean[i]) {
            return Some(Criterion::NoEffectCoord);
        }
        None
    }
}

// the score of a feasible fitness
fn feasible_score(fitness: Fitness) -> Option<f64> {
    fitness.score().filter(|_| fitness.is_feasible())
}

// the difference between the largest and the smallest value, or `None` if one is missing
fn range(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for value in values {
        let value = value?;
        low = low.min(value);
        high = high.max(value);
    }
    Some(high - low)
}

fn identity(n: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; n * n];
    for i in 0..n {
        matrix[i * n + i] = 1.0;
    }
    matrix
}

// the eigenvalues and eigenvectors (the columns of a row-major matrix) of the symmetric row-major
// `n × n` matrix: a Householder reduction to tridiagonal form and the implicit QL method, as
// tred2 and tql2 of JAMA (public domain), which Hansen's Java CMA-ES uses too. Only +, −, ×, ÷
// and sqrt, so the same on every platform.
fn eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = matrix.to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    tridiagonalize(&mut v, &mut d, &mut e, n);
    diagonalize(&mut v, &mut d, &mut e, n);
    (d, v)
}

// sqrt(a² + b²) without overflow or underflow
fn hypot(a: f64, b: f64) -> f64 {
    let (a, b) = (a.abs(), b.abs());
    let (large, small) = if a > b { (a, b) } else { (b, a) };
    if large == 0.0 {
        0.0
    } else {
        let ratio = small / large;
        large * (1.0 + ratio * ratio).sqrt()
    }
}

// tred2: the Householder reduction of the symmetric `v` to a tridiagonal matrix with diagonal
// `d` and subdiagonal `e[1..]`, and the transformation in `v`
fn tridiagonalize(v: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    d.copy_from_slice(&v[(n - 1) * n..]);
    for i in (1..n).rev() {
        let scale: f64 = d[..i].iter().map(|x| x.abs()).sum();
        let mut h = 0.0;
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
                v[j * n + i] = 0.0;
            }
        } else {
            // the Householder vector
            for x in &mut d[..i] {
                *x /= scale;
                h += *x * *x;
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            e[..i].fill(0.0);
            // the similarity transformation of the remaining columns
            for j in 0..i {
                f = d[j];
                v[j * n + i] = f;
                g = e[j] + v[j * n + j] * f;
                for k in j + 1..i {
                    g += v[k * n + j] * d[k];
                    e[k] += v[k * n + j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[k * n + j] -= f * e[k] + g * d[k];
                }
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
            }
        }
        d[i] = h;
    }
    // the accumulated transformations
    for i in 0..n - 1 {
        v[(n - 1) * n + i] = v[i * n + i];
        v[i * n + i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[k * n + i + 1] / h;
            }
            for j in 0..=i {
                let g: f64 = (0..=i).map(|k| v[k * n + i + 1] * v[k * n + j]).sum();
                for k in 0..=i {
                    v[k * n + j] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[k * n + i + 1] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[(n - 1) * n + j];
        v[(n - 1) * n + j] = 0.0;
    }
    v[(n - 1) * n + n - 1] = 1.0;
    e[0] = 0.0;
}

// tql2: the eigenvalues (in `d`) and eigenvectors (the columns of `v`) of the tridiagonal matrix
// from `tridiagonalize`, by the implicit QL method
fn diagonalize(v: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    e.copy_within(1.., 0);
    e[n - 1] = 0.0;
    let mut f = 0.0;
    let mut largest = 0.0f64;
    for l in 0..n {
        // a small subdiagonal element splits the matrix
        largest = largest.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 && e[m].abs() > f64::EPSILON * largest {
            m += 1;
        }
        if m > l {
            for _ in 0..100 {
                // the implicit shift
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = hypot(p, 1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for x in &mut d[l + 2..] {
                    *x -= h;
                }
                f += h;
                // the implicit QL transformation
                p = d[m];
                let (mut c, mut c2, mut c3) = (1.0, 1.0, 1.0);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (0.0, 0.0);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for k in 0..n {
                        h = v[k * n + i + 1];
                        v[k * n + i + 1] = s * v[k * n + i] + c * h;
                        v[k * n + i] = c * v[k * n + i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= f64::EPSILON * largest {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
}

impl Algorithm for Cmaes {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        if !self.asked {
            if self.started {
                if self.converged.is_some() && self.restarts != Restarts::Never {
                    self.restart();
                }
                self.sample();
            }
            self.pending.clear();
            self.pending.extend(0..self.population.len());
            self.asked = true;
        }
        Candidates::new(self.population.as_slice(), &self.pending)
    }

    fn tell(&mut self, fitness: &[Fitness]) -> Result<()> {
        if !self.asked {
            return Err(Error::TellWithoutAsk);
        }
        if fitness.len() != self.pending.len() {
            return Err(Error::FitnessCount {
                expected: self.pending.len(),
                got: fitness.len(),
            });
        }
        self.asked = false;
        self.evaluations += fitness.len() as u64;
        self.run_evaluations += fitness.len() as u64;
        if self.started {
            self.generation += 1;
        }
        let objective = self.objective;
        for (individual, &fitness) in self.population.iter_mut().zip(fitness) {
            individual.set_fitness(fitness);
        }
        // the best so far, the first one on ties
        let mut improved = false;
        for candidate in self.population.iter() {
            let fitness = candidate.fitness().unwrap_or(Fitness::invalid());
            let better = self.best.as_ref().is_none_or(|best| {
                objective.is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
            });
            if better {
                self.best = Some(candidate.clone());
                improved = true;
            }
        }
        if improved {
            self.best_generation = self.generation;
        }
        self.update();
        self.started = true;
        Ok(())
    }

    fn population(&self) -> &Population<Reals> {
        &self.population
    }

    fn best(&self) -> Option<&Individual<Reals>> {
        self.best.as_ref()
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn best_generation(&self) -> u64 {
        self.best_generation
    }
}

/// A builder for a [`Cmaes`], from [`Cmaes::builder`].
///
/// Defaults: the population size of [`Cmaes::default_population_size`], an initial step size of
/// 0.3 of each gene's range, a random initial mean, no restarts, maximize and a random seed.
#[derive(Clone, Debug)]
pub struct CmaesBuilder {
    real: Real,
    population_size: Option<usize>,
    initial_step: f64,
    initial_mean: Option<Reals>,
    restarts: Restarts,
    objective: Objective,
    seed: Option<u64>,
}

impl CmaesBuilder {
    /// The number of samples per generation `λ`, at least 2. Larger populations are slower but
    /// more global. `4 + ⌊3 ln n⌋` for `n` genes by default; with restarts, the size of the first
    /// run.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// The initial step size `σ₀` as a fraction of each gene's range, greater than 0 and at most
    /// 1: about a third of the distance to the optimum. 0.3 by default.
    pub fn initial_step(mut self, fraction: f64) -> Self {
        self.initial_step = fraction;
        self
    }

    /// The initial mean of the search distribution, valid for the representation. Random by
    /// default; restarts always start from a random point.
    pub fn initial_mean(mut self, genome: Reals) -> Self {
        self.initial_mean = Some(genome);
        self
    }

    /// Whether to start a new run when the current one has converged. [`Restarts::Never`] by
    /// default.
    pub fn restarts(mut self, restarts: Restarts) -> Self {
        self.restarts = restarts;
        self
    }

    /// Whether higher or lower fitness is better. Maximize by default.
    pub fn objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }

    /// Higher fitness is better (the default).
    pub fn maximize(self) -> Self {
        self.objective(Objective::Maximize)
    }

    /// Lower fitness is better.
    pub fn minimize(self) -> Self {
        self.objective(Objective::Minimize)
    }

    /// The seed of the random numbers, for a reproducible run. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the settings and creates the algorithm, with its first samples.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for a population size below 2, an initial step size out of
    ///   range, or a representation without a gene that has more than one value.
    /// - [`Error::InvalidGenome`] for an initial mean that doesn't fit the representation.
    pub fn build(self) -> Result<Cmaes> {
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        let bounds = self.real.bounds();
        let free: Vec<usize> = (0..bounds.len())
            .filter(|&gene| bounds[gene].start() < bounds[gene].end())
            .collect();
        if free.is_empty() {
            return invalid(
                "real",
                "CMA-ES needs a gene with more than one value".to_string(),
            );
        }
        let lambda = self
            .population_size
            .unwrap_or_else(|| Cmaes::default_population_size(free.len()));
        if lambda < 2 {
            return invalid(
                "population_size",
                format!("CMA-ES needs at least 2 samples, got {lambda}"),
            );
        }
        if !(self.initial_step > 0.0 && self.initial_step <= 1.0) {
            return invalid(
                "initial_step",
                format!(
                    "must be greater than 0 and at most 1, got {}",
                    self.initial_step
                ),
            );
        }
        if let Some(genome) = &self.initial_mean {
            self.real.validate(genome)?;
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let mean: Vec<f64> = match &self.initial_mean {
            Some(genome) => free
                .iter()
                .map(|&gene| {
                    let (start, end) = (*bounds[gene].start(), *bounds[gene].end());
                    ((genome[gene] - start) / (end - start)).clamp(0.0, 1.0)
                })
                .collect(),
            None => free.iter().map(|_| rng.unit_f64()).collect(),
        };
        let n = free.len();
        let mut cmaes = Cmaes {
            real: self.real,
            free,
            restarts: self.restarts,
            initial_step: self.initial_step,
            initial_lambda: lambda,
            objective: self.objective,
            seed,
            rng,
            parameters: Parameters::new(n, lambda),
            mean: Vec::new(),
            sigma: 0.0,
            run_sigma: 0.0,
            covariance: Vec::new(),
            basis: Vec::new(),
            deviations: Vec::new(),
            path_sigma: Vec::new(),
            path_c: Vec::new(),
            decay: 1.0,
            run_generation: 0,
            eigen_generation: 0,
            samples: Vec::new(),
            steps: Vec::new(),
            best_scores: VecDeque::new(),
            equal_values: VecDeque::new(),
            converged: None,
            restart_count: 0,
            large_lambda: lambda,
            large_evaluations: 0,
            small_evaluations: 0,
            small_run: false,
            run_evaluations: 0,
            population: Population::new(Vec::new()),
            started: false,
            asked: false,
            pending: Vec::new(),
            generation: 0,
            evaluations: 0,
            best: None,
            best_generation: 0,
        };
        cmaes.start_run(lambda, self.initial_step, mean);
        cmaes.sample();
        Ok(cmaes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Engine, Stop, StopReason};
    use proptest::prelude::*;

    fn sphere(x: &Reals) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn ellipsoid(x: &Reals) -> f64 {
        let n = x.len() as f64;
        x.iter()
            .enumerate()
            .map(|(i, xi)| crate::math::pow(10.0, 6.0 * i as f64 / (n - 1.0)) * xi * xi)
            .sum()
    }

    fn rosenbrock(x: &Reals) -> f64 {
        x.windows(2)
            .map(|w| {
                let (a, b) = (w[1] - w[0] * w[0], 1.0 - w[0]);
                100.0 * a * a + b * b
            })
            .sum()
    }

    fn builder(n: usize, seed: u64) -> CmaesBuilder {
        Cmaes::builder(Real::uniform(n, -5.0..=5.0).unwrap())
            .minimize()
            .seed(seed)
    }

    fn step(cmaes: &mut Cmaes, f: impl Fn(&Reals) -> f64) {
        let fitness: Vec<Fitness> = cmaes.ask().iter().map(|x| Fitness::new(f(x))).collect();
        cmaes.tell(&fitness).unwrap();
    }

    fn setting(result: Result<Cmaes>) -> &'static str {
        match result {
            Err(Error::InvalidSetting { setting, .. } | Error::MissingSetting { setting }) => {
                setting
            }
            other => panic!("expected a setting error, got {other:?}"),
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs()
    }

    #[test]
    fn parameters_are_hansens_defaults() {
        // mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu and chi_n, computed independently from the
        // formulas of Hansen's tutorial
        let cases = [
            (
                10,
                10,
                [
                    3.1672992814107017,
                    0.28442858794636744,
                    1.2844285879463675,
                    0.29499038303562225,
                    0.015283824524751714,
                    0.02015428276120837,
                    3.0847265651690123,
                ],
            ),
            (
                2,
                6,
                [
                    2.0286114646100617,
                    0.44620498737831715,
                    1.4462049873783172,
                    0.6245545390268264,
                    0.1548153998964136,
                    0.057859085071916304,
                    1.254272742818995,
                ],
            ),
            (
                30,
                50,
                [
                    13.951320940285154,
                    0.3258608886110323,
                    1.3258608886110324,
                    0.12782802102673924,
                    0.002012798741207134,
                    0.023166787301663005,
                    5.431871828887874,
                ],
            ),
        ];
        for (n, lambda, expected) in cases {
            let p = Parameters::new(n, lambda);
            let actual = [
                p.mu_eff, p.c_sigma, p.d_sigma, p.c_c, p.c_1, p.c_mu, p.chi_n,
            ];
            for (a, e) in actual.iter().zip(expected) {
                assert!(close(*a, e), "n {n}: {actual:?}");
            }
            assert_eq!(p.weights.len(), lambda / 2);
            assert!(close(p.weights.iter().sum(), 1.0));
            assert!(p.weights.windows(2).all(|w| w[0] > w[1]));
        }
        assert_eq!(Cmaes::default_population_size(1), 4);
        assert_eq!(Cmaes::default_population_size(10), 10);
        assert_eq!(Cmaes::default_population_size(100), 17);
    }

    #[test]
    fn validation() {
        let real = || Real::uniform(2, 0.0..=1.0).unwrap();
        let with_step = |step| Cmaes::builder(real()).initial_step(step).build();
        assert_eq!(
            setting(Cmaes::builder(real()).population_size(1).build()),
            "population_size"
        );
        assert_eq!(setting(with_step(0.0)), "initial_step");
        assert_eq!(setting(with_step(1.5)), "initial_step");
        assert_eq!(setting(with_step(f64::NAN)), "initial_step");
        let fixed = Real::new([1.0..=1.0, 2.0..=2.0]).unwrap();
        assert_eq!(setting(Cmaes::builder(fixed).build()), "real");
        assert!(matches!(
            Cmaes::builder(real())
                .initial_mean(Reals::from(vec![0.5]))
                .build(),
            Err(Error::InvalidGenome { .. })
        ));
        assert!(
            Cmaes::builder(real())
                .population_size(2)
                .initial_step(1.0)
                .build()
                .is_ok()
        );
    }

    #[test]
    fn ask_tell_protocol() {
        let mut cmaes = builder(10, 0).build().unwrap();
        assert_eq!(cmaes.tell(&[]), Err(Error::TellWithoutAsk));
        let first: Vec<Reals> = cmaes.ask().iter().cloned().collect();
        assert_eq!(first.len(), 10);
        assert_eq!(cmaes.ask().iter().cloned().collect::<Vec<_>>(), first);
        assert_eq!(
            cmaes.tell(&[Fitness::new(0.0)]),
            Err(Error::FitnessCount {
                expected: 10,
                got: 1
            })
        );
        step(&mut cmaes, sphere);
        assert_eq!((cmaes.generation(), cmaes.evaluations()), (0, 10));
        step(&mut cmaes, sphere);
        assert_eq!((cmaes.generation(), cmaes.evaluations()), (1, 20));
        assert!(cmaes.discarded().is_empty());
    }

    #[test]
    fn initial_mean_and_fixed_genes() {
        let real = Real::new([2.0..=2.0, -1.0..=1.0, 0.0..=10.0]).unwrap();
        let mut cmaes = Cmaes::builder(real)
            .initial_mean(Reals::from(vec![2.0, 0.5, 7.5]))
            .initial_step(0.1)
            .seed(0)
            .build()
            .unwrap();
        assert_eq!(cmaes.mean().to_vec(), [2.0, 0.5, 7.5]);
        assert_eq!(cmaes.axis_deviations(), [0.1, 0.1]);
        for _ in 0..10 {
            assert!(cmaes.ask().iter().all(|x| x[0] == 2.0));
            step(&mut cmaes, sphere);
        }
        assert_eq!(cmaes.mean()[0], 2.0);
    }

    #[test]
    fn as_efficient_as_the_reference_implementation() {
        // pycma (without active CMA) needs about 1,800, 5,800 and 6,400 evaluations
        let functions: [fn(&Reals) -> f64; 3] = [sphere, ellipsoid, rosenbrock];
        for (f, budget) in functions.into_iter().zip([2_500, 8_000, 10_000]) {
            let cmaes = builder(10, 1).build().unwrap();
            let outcome = Engine::new(cmaes, f)
                .stop_when(Stop::target(1e-10).or(Stop::evaluations(budget)))
                .run()
                .unwrap();
            assert_eq!(
                outcome.stop_reason(),
                StopReason::Target,
                "{}",
                outcome.evaluations()
            );
        }
    }

    #[test]
    fn converged_runs_are_detected() {
        let run = |f: fn(&Reals) -> f64| {
            let mut cmaes = builder(4, 2).build().unwrap();
            while cmaes.converged().is_none() && cmaes.generation() < 5_000 {
                step(&mut cmaes, f);
            }
            cmaes
        };
        let cmaes = run(sphere);
        assert_eq!(cmaes.converged(), Some(Criterion::TolHistFun));
        assert!(sphere(cmaes.best().unwrap().genome()) < 1e-12);
        let cmaes = run(|_| 1.0);
        assert_eq!(cmaes.converged(), Some(Criterion::EqualFunValues));
        assert!(cmaes.generation() < 10);
        // steps too small or too large
        let mut cmaes = builder(4, 2).build().unwrap();
        step(&mut cmaes, sphere);
        cmaes.sigma = 1e-14 * cmaes.run_sigma;
        assert_eq!(cmaes.criterion(&[]), Some(Criterion::TolX));
        cmaes.sigma = 1e4 * cmaes.run_sigma;
        assert_eq!(cmaes.criterion(&[]), Some(Criterion::TolUpX));
    }

    #[test]
    fn ill_conditioned_covariance_is_repaired() {
        let mut cmaes = builder(3, 0).build().unwrap();
        cmaes.covariance = vec![1.0, 0.0, 0.0, 0.0, 1e-20, 0.0, 0.0, 0.0, 1.0];
        cmaes.decompose();
        assert_eq!(cmaes.converged(), Some(Criterion::ConditionCov));
        let max = cmaes.deviations.iter().copied().fold(0.0, f64::max);
        let min = cmaes.deviations.iter().copied().fold(f64::MAX, f64::min);
        assert!(min > 0.0 && max / min <= 1.0001e7, "{:?}", cmaes.deviations);
    }

    #[test]
    fn ipop_doubles_the_population_up_to_a_limit() {
        // a flat function converges after one generation in one dimension
        let mut cmaes = Cmaes::builder(Real::uniform(1, 0.0..=1.0).unwrap())
            .restarts(Restarts::Ipop)
            .seed(0)
            .build()
            .unwrap();
        let mut sizes = Vec::new();
        for _ in 0..14 {
            sizes.push(cmaes.ask().len());
            step(&mut cmaes, |_| 0.0);
        }
        assert_eq!(sizes[..4], [4, 8, 16, 32]);
        assert_eq!(sizes[10..], [4096; 4]);
        assert_eq!(cmaes.restart_count(), 13);
    }

    #[test]
    fn bipop_balances_the_budgets_of_both_regimes() {
        let mut cmaes = Cmaes::builder(Real::uniform(1, 0.0..=1.0).unwrap())
            .restarts(Restarts::Bipop)
            .seed(0)
            .build()
            .unwrap();
        for _ in 0..200 {
            let size = cmaes.ask().len();
            if cmaes.small_run {
                assert!(size >= 2 && size <= cmaes.large_lambda / 2 + 2, "{size}");
                assert!(cmaes.run_sigma <= 0.3 && cmaes.run_sigma >= 0.003);
            } else {
                assert_eq!(size, cmaes.large_lambda);
                assert_eq!(cmaes.run_sigma, 0.3);
            }
            step(&mut cmaes, |_| 0.0);
        }
        let (small, large) = (cmaes.small_evaluations, cmaes.large_evaluations);
        assert!(small > 0 && large > 0);
        // the small regime catches up with the large one, and stays behind it
        assert!(
            small <= large && large - small <= cmaes.large_lambda as u64,
            "{small} {large}"
        );
        assert!(cmaes.large_lambda > 4);
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut cmaes = builder(5, seed).restarts(Restarts::Bipop).build().unwrap();
            for _ in 0..50 {
                step(&mut cmaes, rosenbrock);
            }
            cmaes.population().clone()
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }

    fn symmetric(n: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-100.0..100.0f64, n * n).prop_map(move |mut matrix| {
            for i in 0..n {
                for j in 0..i {
                    matrix[i * n + j] = matrix[j * n + i];
                }
            }
            matrix
        })
    }

    // A = V · diag(values) · Vᵀ, and Vᵀ · V = I
    fn assert_decomposes(matrix: &[f64], n: usize) {
        let (values, vectors) = eigen(matrix, n);
        let scale = matrix.iter().fold(1.0f64, |max, x| max.max(x.abs()));
        for i in 0..n {
            for j in 0..n {
                let rebuilt: f64 = (0..n)
                    .map(|k| vectors[i * n + k] * values[k] * vectors[j * n + k])
                    .sum();
                assert!((rebuilt - matrix[i * n + j]).abs() < 1e-12 * scale * n as f64);
                let dot: f64 = (0..n)
                    .map(|k| vectors[k * n + i] * vectors[k * n + j])
                    .sum();
                let identity = if i == j { 1.0 } else { 0.0 };
                assert!((dot - identity).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn eigen_decomposes_special_matrices() {
        for n in [1, 2, 5, 60] {
            assert_decomposes(&vec![0.0; n * n], n);
            assert_decomposes(&identity(n), n);
            // diagonal with repeated values, and rank one
            let diagonal: Vec<f64> = (0..n * n)
                .map(|k| {
                    if k % (n + 1) == 0 {
                        (k % 3) as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            assert_decomposes(&diagonal, n);
            let y: Vec<f64> = (0..n).map(|i| i as f64 - 2.5).collect();
            let rank_one: Vec<f64> = (0..n * n).map(|k| y[k / n] * y[k % n]).collect();
            assert_decomposes(&rank_one, n);
            // badly conditioned
            let scaled: Vec<f64> = (0..n * n)
                .map(|k| {
                    if k % (n + 1) == 0 {
                        crate::math::pow(10.0, -((k % 15) as f64))
                    } else {
                        1e-9
                    }
                })
                .collect();
            assert_decomposes(&scaled, n);
        }
    }

    proptest! {
        #[test]
        fn eigen_decomposes_symmetric_matrices(
            (n, matrix) in (1usize..30).prop_flat_map(|n| (Just(n), symmetric(n)))
        ) {
            assert_decomposes(&matrix, n);
        }

        #[test]
        fn samples_and_mean_stay_in_bounds_and_the_distribution_stays_valid(
            seed: u64,
            lambda in 2usize..12,
            initial_step in 0.001..=1.0f64,
            restarts in prop::sample::select(vec![Restarts::Never, Restarts::Ipop, Restarts::Bipop]),
        ) {
            // one fixed gene, bounds of different widths, and a random fitness
            let real = Real::new([3.0..=3.0, -1.0..=1.0, 0.0..=100.0, -1e-3..=1e-3, -8e307..=8e307])
                .unwrap();
            let mut cmaes = Cmaes::builder(real.clone())
                .population_size(lambda)
                .initial_step(initial_step)
                .restarts(restarts)
                .minimize()
                .seed(seed)
                .build()
                .unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            for _ in 0..30 {
                let samples: Vec<Reals> = cmaes.ask().iter().cloned().collect();
                for sample in &samples {
                    prop_assert!(real.validate(sample).is_ok(), "{sample:?}");
                }
                let fitness: Vec<Fitness> =
                    samples.iter().map(|_| Fitness::new(rng.below(3) as f64)).collect();
                cmaes.tell(&fitness).unwrap();
                prop_assert!(real.validate(&cmaes.mean()).is_ok());
                prop_assert!(cmaes.sigma.is_finite() && cmaes.sigma > 0.0);
                prop_assert!(cmaes.deviations.iter().all(|d| d.is_finite() && *d > 0.0));
                let n = cmaes.dimensions();
                for i in 0..n {
                    for j in 0..n {
                        prop_assert_eq!(cmaes.covariance[i * n + j], cmaes.covariance[j * n + i]);
                    }
                }
            }
        }
    }
}
