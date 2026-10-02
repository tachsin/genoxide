//! Gaussian process regression with a constant mean and a stationary kernel, its hyperparameters
//! fitted by maximum marginal likelihood: the surrogate model of
//! [Bayesian optimization](crate::algorithm::bo).
//!
//! **Unstable for one release.** This module is public from genoxide 0.13 so that a Gaussian
//! process can be fitted and queried on its own, but its API and the bits of its fits may still
//! change in 0.14, as the batch and constrained Bayesian optimization of that release use it.
//!
//! A [`GaussianProcess`] models a function `f` of the genes of a [`Real`] genome as
//! `f(x) ~ GP(m, σ_f² k(x, x′))` (Rasmussen and Williams, 2006, eq. 2.37), observed with
//! independent normal noise of variance `σ_n²` (eq. 2.20), and gives the posterior mean and
//! variance of `f` at any point (eq. 2.25 and 2.26), with their gradients. Built with
//! [`GaussianProcess::builder`], fitted with [`fit`](GaussianProcessBuilder::fit):
//!
//! ```
//! use genoxide::model::gp::GaussianProcess;
//! use genoxide::prelude::*;
//!
//! // a smooth function of one gene, from 8 evaluations
//! let f = |x: f64| x.sin() + 0.1 * x * x;
//! let points: Vec<Reals> = (0..8).map(|i| Reals::from(vec![f64::from(i)])).collect();
//! let values: Vec<f64> = points.iter().map(|x| f(x[0])).collect();
//! let gp = GaussianProcess::builder(Real::uniform(1, 0.0..=7.0)?).fit(&points, &values)?;
//! // close to f between the points, and sure of itself there
//! let prediction = gp.predict(&[3.5]);
//! assert!((prediction.mean() - f(3.5)).abs() < 0.05);
//! assert!(prediction.sd() < 0.1);
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! # The model
//!
//! - **Inputs** are scaled to the unit cube by the bounds of the [`Real`] genome: each gene's
//!   range becomes [0, 1], so the length scales of every gene start alike. A gene whose bounds are
//!   equal takes no part in the model.
//! - **Outputs** are standardized: the model fits `(y − ȳ) / s`, with `ȳ` and `s` the mean and the
//!   standard deviation of the values, and its predictions are mapped back. With given
//!   [hyperparameters](GaussianProcessBuilder::hyperparameters), that changes nothing but the
//!   rounding: the predictions are those of the model in the values' own units.
//! - **The kernel** ([`Kernel`]) is stationary with a length scale per gene (automatic relevance
//!   determination, Rasmussen and Williams, eq. 5.1 and 5.2): `k(x, x′) = k(r)` of the scaled
//!   distance `r² = Σᵢ (xᵢ − x′ᵢ)² / ℓᵢ²`. The Matérn kernel with ν = 5/2 by default (eq. 4.17),
//!   twice differentiable, as Snoek, Larochelle and Adams (2012) advise for Bayesian optimization
//!   against the squared exponential's infinitely smooth functions (eq. 4.9; Stein, 1999).
//! - **The noise** ([`Noise`]) is none by default: the model interpolates the values, as suits
//!   the deterministic functions genoxide optimizes, with the jitter below as the only nugget.
//!   For a noisy function, it's learned with the other hyperparameters, or fixed.
//! - **The constant mean** `m` is, for given kernel hyperparameters, the one that maximizes the
//!   marginal likelihood: the generalized least squares estimate `m = 1ᵀK⁻¹y / 1ᵀK⁻¹1` (setting
//!   the derivative of eq. 2.30 with `y − m` for `y` to zero, as eq. 2.38 models a fixed mean).
//!   The hyperparameters therefore maximize the likelihood over the mean too, and the mean's
//!   derivative being zero there, eq. 5.9 is the gradient of the likelihood so maximized.
//!
//! # Fitting
//!
//! [`fit`](GaussianProcessBuilder::fit) maximizes the log marginal likelihood (eq. 2.30 and 5.8)
//! `ln p(y | X, θ) = −½ (y − m)ᵀ K_y⁻¹ (y − m) − ½ ln |K_y| − (n/2) ln 2π`, `K_y = σ_f² K + σ_n² I`,
//! over the logarithms of the length scales, of `σ_f²` and of `σ_n²`, with genoxide's
//! [`Lbfgsb`] and the analytic gradient of eq. 5.9,
//! `∂/∂θⱼ ln p = ½ tr((ααᵀ − K_y⁻¹) ∂K_y/∂θⱼ)`, `α = K_y⁻¹(y − m)`. The search runs from
//! [several starts](GaussianProcessBuilder::starts): the first from fixed values (length scales of
//! 0.5 of each range, `σ_f²` the values' variance, a learned `σ_n²` 1e-4 of it or its least), the
//! others from random points of the box below, drawn from streams derived from the
//! [seed](GaussianProcessBuilder::seed), independent of each other. The likelihood's best wins,
//! the earlier start on ties, so the fit is the same on any number of threads (with the
//! `parallel` feature, the starts run on rayon). The box, in the scaled units: length scales from
//! 0.01 to 100, `σ_f²` from 1e-3 to 1e3, `σ_n²` from the noise's least to 1.
//!
//! Every matrix is factored by Cholesky (Rasmussen and Williams, Algorithm 2.1). A kernel matrix
//! that rounding makes indefinite, e.g. with points very close together and little noise, gets a
//! jitter on its diagonal: none, then 1e-10 of the diagonal's scale, raised tenfold until it
//! factors ([`GaussianProcess::jitter`]).
//!
//! Every operation is a sum, product, quotient or square root in a fixed order, with
//! [`math`](crate::math)'s `exp` and `ln`, so a fit gives the same bits on every platform.
//!
//! References: Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine
//! Learning.* MIT Press, ch. 2, 4 and 5. Snoek, J., Larochelle, H. and Adams, R. P. (2012).
//! Practical Bayesian optimization of machine learning algorithms. *NeurIPS 25*,
//! arXiv:1206.2944. Stein, M. L. (1999). *Interpolation of Spatial Data.* Springer.

use crate::algorithm::Lbfgsb;
use crate::genome::{Real, Reals, Representation};
use crate::linalg::cholesky::cholesky_with_jitter;
use crate::linalg::triangular::{
    cholesky_solve, cholesky_solve_multi, solve_lower, solve_lower_transposed,
};
use crate::math::{exp, ln};
use crate::{Error, Result, StreamRng};
use std::ops::RangeInclusive;

/// The kernel of a [`GaussianProcess`]: the correlation of `f` at two points as a function of
/// their scaled distance `r`, with `r² = Σᵢ (xᵢ − x′ᵢ)² / ℓᵢ²`, a length scale `ℓᵢ` per gene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Kernel {
    /// The Matérn kernel with ν = 5/2, `(1 + √5 r + 5r²/3) exp(−√5 r)` (Rasmussen and Williams,
    /// 2006, eq. 4.17): functions twice differentiable. The default, as for most Bayesian
    /// optimization (Snoek, Larochelle and Adams, 2012).
    #[default]
    Matern52,
    /// The squared exponential, `exp(−r²/2)` (Rasmussen and Williams, eq. 4.9): infinitely
    /// differentiable functions, smoother than most that are optimized.
    SquaredExponential,
}

impl Kernel {
    /// `k(r)` and its derivative with respect to `r²`, at `r²`.
    #[inline]
    fn eval(self, r2: f64) -> (f64, f64) {
        match self {
            Kernel::Matern52 => {
                // s = √5 r: k = (1 + s + s²/3) e^(−s), dk/d(r²) = −(5/6)(1 + s) e^(−s)
                let s = (5.0 * r2).sqrt();
                let e = exp(-s);
                ((1.0 + s + s * s / 3.0) * e, -5.0 / 6.0 * (1.0 + s) * e)
            }
            Kernel::SquaredExponential => {
                let k = exp(-0.5 * r2);
                (k, -0.5 * k)
            }
        }
    }
}

/// The observation noise of a [`GaussianProcess`]: its variance `σ_n²`, as a fraction of the
/// values' variance (the model's outputs are standardized).
///
/// None by default (`Fixed(0.0)`): genoxide's fitness functions are deterministic, and a model
/// that interpolates them resolves the small differences near a minimum that a learned noise
/// smooths over. Measured with [`Bo`](crate::algorithm::Bo) over 20 seeds and 80 evaluations, to
/// f* + 1e-4: Branin reached in 20 runs, the six-hump camel in [−3, 3] × [−2, 2] in 20 and
/// Hartmann 3 in 20, against 14, 17 and 12 with noise learned from a least of 1e-6, which settles
/// above that least, at an absolute standard deviation of about 0.03 on Branin.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Noise {
    /// Learned with the other hyperparameters, at least `min` (0 < `min` < 1), e.g. 1e-6: for a
    /// noisy function, whose values the model shouldn't interpolate.
    Learned {
        /// The least noise variance, a fraction of the values' variance.
        min: f64,
    },
    /// A fixed variance, at least 0: 0 (the default) interpolates the values, with the jitter the
    /// Cholesky factorization may need.
    Fixed(f64),
}

impl Default for Noise {
    fn default() -> Self {
        Noise::Fixed(0.0)
    }
}

impl Noise {
    pub(crate) fn validate(self) -> Result<()> {
        let valid = match self {
            Noise::Learned { min } => min > 0.0 && min < 1.0,
            Noise::Fixed(variance) => variance >= 0.0 && variance.is_finite(),
        };
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidSetting {
                setting: "noise",
                reason: format!(
                    "a learned noise's least variance must be between 0 and 1 (exclusive), and a \
                     fixed one finite and at least 0; got {self:?}"
                ),
            })
        }
    }
}

/// The hyperparameters of a [`GaussianProcess`], in the units of the genes and of the values.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Hyperparameters {
    mean: f64,
    length_scales: Vec<f64>,
    signal_variance: f64,
    noise_variance: f64,
}

impl Hyperparameters {
    /// Hyperparameters: the constant mean `m`, a length scale `ℓᵢ` per gene (in the gene's own
    /// units; ignored for genes whose bounds are equal), the signal variance `σ_f²` and the noise
    /// variance `σ_n²`.
    pub fn new(
        mean: f64,
        length_scales: Vec<f64>,
        signal_variance: f64,
        noise_variance: f64,
    ) -> Self {
        Self {
            mean,
            length_scales,
            signal_variance,
            noise_variance,
        }
    }

    /// The constant mean `m`.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// The length scales `ℓᵢ`, one per gene, in the genes' units: infinite for a gene whose bounds
    /// are equal.
    pub fn length_scales(&self) -> &[f64] {
        &self.length_scales
    }

    /// The signal variance `σ_f²`, the variance of `f` far from any data.
    pub fn signal_variance(&self) -> f64 {
        self.signal_variance
    }

    /// The noise variance `σ_n²`.
    pub fn noise_variance(&self) -> f64 {
        self.noise_variance
    }
}

/// A [`GaussianProcess`]'s prediction at a point: the posterior mean and variance of `f` there
/// (Rasmussen and Williams, 2006, eq. 2.25 and 2.26), without the noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    mean: f64,
    variance: f64,
}

impl Prediction {
    /// The posterior mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// The posterior variance, at least 0: about 0 at the data (up to the noise), and the signal
    /// variance far from it.
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// The posterior standard deviation, `√variance`.
    pub fn sd(&self) -> f64 {
        self.variance.sqrt()
    }
}

// the map between genomes and the model's unit cube: the genes with more than one value, each
// range scaled to [0, 1]
#[derive(Clone, Debug)]
pub(crate) struct Scaling {
    // the genes' values: the lower bounds, which are the values of the fixed genes
    template: Vec<f64>,
    variable: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    width: Vec<f64>,
}

impl Scaling {
    pub(crate) fn new(real: &Real) -> Self {
        let bounds = real.bounds();
        let variable = real.variable_genes().to_vec();
        let lower: Vec<f64> = variable.iter().map(|&i| *bounds[i].start()).collect();
        let upper: Vec<f64> = variable.iter().map(|&i| *bounds[i].end()).collect();
        let width = lower.iter().zip(&upper).map(|(l, u)| u - l).collect();
        Self {
            template: bounds.iter().map(|range| *range.start()).collect(),
            variable,
            lower,
            upper,
            width,
        }
    }

    // the model's dimensions: the genes with more than one value
    pub(crate) fn dims(&self) -> usize {
        self.variable.len()
    }

    pub(crate) fn genes(&self) -> usize {
        self.template.len()
    }

    // the unit-cube coordinates of a genome
    pub(crate) fn to_unit(&self, genome: &[f64], unit: &mut [f64]) {
        for (k, &i) in self.variable.iter().enumerate() {
            unit[k] = (genome[i] - self.lower[k]) / self.width[k];
        }
    }

    // the genome at unit-cube coordinates, in the box: the ends map to the bounds exactly
    pub(crate) fn to_genome(&self, unit: &[f64]) -> Reals {
        let mut genome = self.template.clone();
        for (k, &i) in self.variable.iter().enumerate() {
            let u = unit[k];
            genome[i] = if u >= 1.0 {
                self.upper[k]
            } else if u <= 0.0 {
                self.lower[k]
            } else {
                (self.lower[k] + u * self.width[k]).clamp(self.lower[k], self.upper[k])
            };
        }
        Reals::from(genome)
    }

    // a gradient with respect to the unit-cube coordinates, as one with respect to the genes
    fn to_gene_gradient(&self, unit_gradient: &[f64], scale: f64, gradient: &mut [f64]) {
        gradient.fill(0.0);
        for (k, &i) in self.variable.iter().enumerate() {
            gradient[i] = scale * unit_gradient[k] / self.width[k];
        }
    }
}

// the settings of a fit
#[derive(Clone, Copy, Debug)]
pub(crate) struct Settings {
    pub(crate) kernel: Kernel,
    pub(crate) noise: Noise,
    pub(crate) starts: usize,
    pub(crate) seed: u64,
}

// the box of the log-hyperparameters, in the scaled units: length scales, σ_f², σ_n²
const LENGTH_SCALES: RangeInclusive<f64> = 0.01..=100.0;
const SIGNAL_VARIANCE: RangeInclusive<f64> = 1e-3..=1e3;
const MAX_NOISE_VARIANCE: f64 = 1.0;
// the first start's values
const INITIAL_LENGTH_SCALE: f64 = 0.5;
const INITIAL_NOISE_VARIANCE: f64 = 1e-4;
// the evaluations of the likelihood per start
const MAX_EVALUATIONS: u64 = 200;
// the first jitter tried, a fraction of the diagonal's scale
const INITIAL_JITTER: f64 = 1e-10;

/// A Gaussian process fitted to evaluations of a function: its posterior mean and variance at any
/// point, with their gradients. See the [module](self) for the model and how it's fitted.
///
/// Built with [`GaussianProcess::builder`].
#[derive(Clone, Debug)]
pub struct GaussianProcess {
    kernel: Kernel,
    scaling: Scaling,
    // the points in the unit cube, count × dims, and the standardization of the values
    x: Vec<f64>,
    count: usize,
    y_mean: f64,
    y_scale: f64,
    // the hyperparameters in the scaled units, and their logarithms (the fitted parameters)
    mean: f64,
    length_scales: Vec<f64>,
    signal: f64,
    noise: f64,
    log: Vec<f64>,
    jitter: f64,
    // the Cholesky factor L of K_y, α = K_y⁻¹(y − m), and the log marginal likelihood
    factor: Vec<f64>,
    alpha: Vec<f64>,
    log_likelihood: f64,
}

impl GaussianProcess {
    /// A builder for a Gaussian process on the genes of `real`, whose bounds scale the inputs.
    pub fn builder(real: Real) -> GaussianProcessBuilder {
        GaussianProcessBuilder {
            real,
            kernel: Kernel::default(),
            noise: Noise::default(),
            starts: 5,
            seed: 0,
            hyperparameters: None,
        }
    }

    /// The kernel.
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// The number of points the model was fitted to.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the model has no points: never, as fitting needs at least one.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The hyperparameters, fitted or given, in the units of the genes and of the values.
    pub fn hyperparameters(&self) -> Hyperparameters {
        let s2 = self.y_scale * self.y_scale;
        let mut length_scales = vec![f64::INFINITY; self.scaling.genes()];
        for (k, &i) in self.scaling.variable.iter().enumerate() {
            length_scales[i] = self.length_scales[k] * self.scaling.width[k];
        }
        Hyperparameters {
            mean: self.y_mean + self.y_scale * self.mean,
            length_scales,
            signal_variance: s2 * self.signal,
            noise_variance: s2 * self.noise,
        }
    }

    /// The jitter added to the diagonal of the kernel matrix for its Cholesky factorization, in
    /// the values' units: usually 0.
    pub fn jitter(&self) -> f64 {
        self.y_scale * self.y_scale * self.jitter
    }

    /// The log marginal likelihood of the values at the hyperparameters (Rasmussen and Williams,
    /// 2006, eq. 2.30), in the values' own units.
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.log_likelihood - self.count as f64 * ln(self.y_scale)
    }

    /// The posterior mean and variance of `f` at `genome` (Rasmussen and Williams, 2006, eq. 2.25
    /// and 2.26, as their Algorithm 2.1 computes them).
    ///
    /// # Panics
    ///
    /// If `genome` doesn't have a value per gene.
    pub fn predict(&self, genome: &[f64]) -> Prediction {
        let mut unit = vec![0.0; self.scaling.dims()];
        self.check_len(genome);
        self.scaling.to_unit(genome, &mut unit);
        let (mean, variance) = self.predict_unit(&unit, None);
        self.unscaled(mean, variance)
    }

    /// The posterior mean and variance at `genome`, with their gradients with respect to the
    /// genes written into `mean_gradient` and `variance_gradient` (0 for a gene whose bounds are
    /// equal). Where rounding would make the variance negative, it's 0, with a gradient of 0.
    ///
    /// # Panics
    ///
    /// If `genome`, `mean_gradient` or `variance_gradient` doesn't have a value per gene.
    pub fn predict_with_gradient(
        &self,
        genome: &[f64],
        mean_gradient: &mut [f64],
        variance_gradient: &mut [f64],
    ) -> Prediction {
        self.check_len(genome);
        self.check_len(mean_gradient);
        self.check_len(variance_gradient);
        let dims = self.scaling.dims();
        let mut unit = vec![0.0; dims];
        self.scaling.to_unit(genome, &mut unit);
        let mut dmean = vec![0.0; dims];
        let mut dvariance = vec![0.0; dims];
        let (mean, variance) = self.predict_unit(&unit, Some((&mut dmean, &mut dvariance)));
        self.scaling
            .to_gene_gradient(&dmean, self.y_scale, mean_gradient);
        self.scaling
            .to_gene_gradient(&dvariance, self.y_scale * self.y_scale, variance_gradient);
        self.unscaled(mean, variance)
    }

    fn check_len(&self, values: &[f64]) {
        assert_eq!(
            values.len(),
            self.scaling.genes(),
            "a Gaussian process of {} genes",
            self.scaling.genes()
        );
    }

    fn unscaled(&self, mean: f64, variance: f64) -> Prediction {
        Prediction {
            mean: self.y_mean + self.y_scale * mean,
            variance: self.y_scale * self.y_scale * variance,
        }
    }

    // the scaled units' mean and variance at the unit-cube point `u`, with their gradients
    // with respect to `u`
    pub(crate) fn predict_unit(
        &self,
        u: &[f64],
        gradients: Option<(&mut [f64], &mut [f64])>,
    ) -> (f64, f64) {
        let (n, dims) = (self.count, self.scaling.dims());
        // k(x, xⱼ) and σ_f² dk/d(r²) for each point xⱼ
        let mut k = vec![0.0; n];
        let mut dk = vec![0.0; n];
        let mut mean = self.mean;
        for j in 0..n {
            let xj = &self.x[j * dims..(j + 1) * dims];
            let r2 = scaled_distance(u, xj, &self.length_scales);
            let (value, derivative) = self.kernel.eval(r2);
            k[j] = self.signal * value;
            dk[j] = self.signal * derivative;
            mean += self.alpha[j] * k[j];
        }
        // v = L⁻¹k, V[f] = σ_f² − vᵀv (Algorithm 2.1, lines 5 and 6)
        let mut v = k;
        solve_lower(&self.factor, n, &mut v);
        let mut vv = 0.0;
        for &vj in &v {
            vv += vj * vj;
        }
        let raw_variance = self.signal - vv;
        let variance = raw_variance.max(0.0);
        if let Some((dmean, dvariance)) = gradients {
            // ∂k(x, xⱼ)/∂xᵢ = σ_f² dk/d(r²) · 2 (xᵢ − xⱼᵢ) / ℓᵢ²; ∂V/∂xᵢ = −2 (K_y⁻¹k)ᵀ ∂k/∂xᵢ
            let mut w = v;
            solve_lower_transposed(&self.factor, n, &mut w);
            dmean.fill(0.0);
            dvariance.fill(0.0);
            for j in 0..n {
                let xj = &self.x[j * dims..(j + 1) * dims];
                for i in 0..dims {
                    let l = self.length_scales[i];
                    let dki = dk[j] * 2.0 * (u[i] - xj[i]) / (l * l);
                    dmean[i] += self.alpha[j] * dki;
                    dvariance[i] -= 2.0 * w[j] * dki;
                }
            }
            if raw_variance <= 0.0 {
                dvariance.fill(0.0);
            }
        }
        (mean, variance)
    }

    // the unit-cube coordinates of the model's point `index`
    pub(crate) fn unit_point(&self, index: usize) -> &[f64] {
        let dims = self.scaling.dims();
        &self.x[index * dims..(index + 1) * dims]
    }

    // the logarithms of the fitted hyperparameters, for a warm start of the next fit
    pub(crate) fn log_parameters(&self) -> &[f64] {
        &self.log
    }

    // the standardization of the values: the mean and the scale
    pub(crate) fn standardization(&self) -> (f64, f64) {
        (self.y_mean, self.y_scale)
    }

    // fits the model to the unit-cube points `x` (count × dims) and `values`, all finite, from
    // the warm start `warm` (the logarithms of an earlier fit's hyperparameters) or the fixed
    // first start
    pub(crate) fn fit_unit(
        settings: Settings,
        scaling: Scaling,
        x: Vec<f64>,
        values: &[f64],
        warm: Option<&[f64]>,
    ) -> Result<GaussianProcess> {
        let count = values.len();
        let (y_mean, y_scale, y) = standardize(values);
        let dims = scaling.dims();
        let data = Data {
            x: &x,
            y: &y,
            count,
            dims,
            kernel: settings.kernel,
            noise: settings.noise,
        };
        let bounds = parameter_bounds(dims, settings.noise);
        let first = initial_parameters(dims, settings.noise, &bounds, warm);
        let best = maximize_likelihood(&data, &bounds, first.clone(), settings);
        let log = best.unwrap_or(first);
        let (length_scales, signal, noise) = unpack(&log, dims, settings.noise);
        let mut workspace = Workspace::new(count);
        // the box's parameters factor with jitter up to the diagonal's scale, a positive definite
        // K_y + (σ_f² + σ_n²) I, unless rounding breaks even that
        let Some(jitter) = data.factor(&length_scales, signal, noise, &mut workspace) else {
            return Err(Error::InvalidSetting {
                setting: "values",
                reason: "the kernel matrix of the points doesn't factor, even with a jitter of                          its diagonal's scale"
                    .to_string(),
            });
        };
        let (mean, log_likelihood) = data.solve(&mut workspace, None);
        let Workspace { l, b, .. } = workspace;
        Ok(GaussianProcess {
            kernel: settings.kernel,
            scaling,
            x,
            count,
            y_mean,
            y_scale,
            mean,
            length_scales,
            signal,
            noise,
            log,
            jitter,
            factor: l,
            alpha: b,
            log_likelihood,
        })
    }
}

// (y − ȳ) / s, with s the standard deviation, 1 if the values are all equal
fn standardize(values: &[f64]) -> (f64, f64, Vec<f64>) {
    let n = values.len() as f64;
    let mut sum = 0.0;
    for &y in values {
        sum += y;
    }
    let mean = sum / n;
    let mut squares = 0.0;
    for &y in values {
        squares += (y - mean) * (y - mean);
    }
    let scale = (squares / n).sqrt();
    let scale = if scale > 0.0 && scale.is_finite() {
        scale
    } else {
        1.0
    };
    (
        mean,
        scale,
        values.iter().map(|y| (y - mean) / scale).collect(),
    )
}

// Σᵢ (aᵢ − bᵢ)² / ℓᵢ²
#[inline]
fn scaled_distance(a: &[f64], b: &[f64], length_scales: &[f64]) -> f64 {
    let mut r2 = 0.0;
    for i in 0..a.len() {
        let t = (a[i] - b[i]) / length_scales[i];
        r2 += t * t;
    }
    r2
}

// the box of the log-hyperparameters: ln ℓᵢ for each gene, ln σ_f², and ln σ_n² when learned
fn parameter_bounds(dims: usize, noise: Noise) -> Vec<RangeInclusive<f64>> {
    let log = |range: RangeInclusive<f64>| ln(*range.start())..=ln(*range.end());
    let mut bounds = vec![log(LENGTH_SCALES); dims];
    bounds.push(log(SIGNAL_VARIANCE));
    if let Noise::Learned { min } = noise {
        bounds.push(log(min..=MAX_NOISE_VARIANCE));
    }
    bounds
}

// the first start: the warm start in the box, or the fixed values
fn initial_parameters(
    dims: usize,
    noise: Noise,
    bounds: &[RangeInclusive<f64>],
    warm: Option<&[f64]>,
) -> Vec<f64> {
    if let Some(warm) = warm.filter(|warm| warm.len() == bounds.len()) {
        return warm
            .iter()
            .zip(bounds)
            .map(|(&p, range)| p.clamp(*range.start(), *range.end()))
            .collect();
    }
    let mut start = vec![ln(INITIAL_LENGTH_SCALE); dims];
    start.push(0.0);
    if let Noise::Learned { min } = noise {
        start.push(ln(INITIAL_NOISE_VARIANCE.max(min)));
    }
    start
}

// the length scales, σ_f² and σ_n² of the log-hyperparameters
fn unpack(log: &[f64], dims: usize, noise: Noise) -> (Vec<f64>, f64, f64) {
    let length_scales = log[..dims].iter().map(|&p| exp(p)).collect();
    let signal = exp(log[dims]);
    let noise = match noise {
        Noise::Learned { .. } => exp(log[dims + 1]),
        Noise::Fixed(variance) => variance,
    };
    (length_scales, signal, noise)
}

// the best of the starts, by L-BFGS-B on the negated log marginal likelihood; None if no start
// gave a finite likelihood
fn maximize_likelihood(
    data: &Data<'_>,
    bounds: &[RangeInclusive<f64>],
    first: Vec<f64>,
    settings: Settings,
) -> Option<Vec<f64>> {
    let Ok(real) = Real::new(bounds.iter().cloned()) else {
        return None;
    };
    let starts = settings.starts.max(1);
    let root = StreamRng::seed_from_u64(settings.seed);
    let results = map_in_order(starts, |start| {
        let point = if start == 0 {
            first.clone()
        } else {
            real.random_genome(&mut root.derive(start as u64))
                .into_vec()
        };
        let mut workspace = Workspace::new(data.count);
        Lbfgsb::minimize_with(
            real.clone(),
            Reals::from(point),
            MAX_EVALUATIONS,
            |log, gradient| {
                let value = data.log_likelihood(log, &mut workspace, Some(gradient));
                for g in gradient.iter_mut() {
                    *g = -*g;
                }
                -value
            },
        )
    });
    let mut best: Option<(Reals, f64)> = None;
    for (point, value) in results.into_iter().flatten() {
        if best.as_ref().is_none_or(|(_, best)| value < *best) {
            best = Some((point, value));
        }
    }
    best.map(|(point, _)| point.into_vec())
}

// `f(i)` for i in 0..n, in order: on rayon with the `parallel` feature, the same results
pub(crate) fn map_in_order<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

// the matrices of a likelihood's evaluation, reused between evaluations
struct Workspace {
    k: Vec<f64>,
    l: Vec<f64>,
    inverse: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    diff: Vec<f64>,
}

impl Workspace {
    fn new(n: usize) -> Self {
        Self {
            k: vec![0.0; n * n],
            l: vec![0.0; n * n],
            inverse: Vec::new(),
            a: vec![0.0; n],
            b: vec![0.0; n],
            diff: Vec::new(),
        }
    }
}

// the standardized data of a fit
struct Data<'a> {
    x: &'a [f64],
    y: &'a [f64],
    count: usize,
    dims: usize,
    kernel: Kernel,
    noise: Noise,
}

impl Data<'_> {
    // factors K_y = σ_f² K + σ_n² I (plus the jitter it needs) into `workspace.l`: the jitter,
    // or None if no jitter up to the diagonal's scale factors it
    fn factor(
        &self,
        length_scales: &[f64],
        signal: f64,
        noise: f64,
        workspace: &mut Workspace,
    ) -> Option<f64> {
        let (n, dims) = (self.count, self.dims);
        for p in 0..n {
            let xp = &self.x[p * dims..(p + 1) * dims];
            for q in 0..p {
                let xq = &self.x[q * dims..(q + 1) * dims];
                let (k, _) = self.kernel.eval(scaled_distance(xp, xq, length_scales));
                workspace.k[p * n + q] = signal * k;
            }
            workspace.k[p * n + p] = signal + noise;
        }
        let scale = signal + noise;
        if !(scale > 0.0 && scale.is_finite()) {
            return None;
        }
        cholesky_with_jitter(
            &workspace.k,
            n,
            INITIAL_JITTER * scale,
            scale,
            &mut workspace.l,
        )
        .ok()
    }

    // from the factor in `workspace.l`: the mean (the generalized least squares estimate, or
    // `fixed`), α = K_y⁻¹(y − m) into `workspace.b`, and the log marginal likelihood (eq. 2.30)
    fn solve(&self, workspace: &mut Workspace, fixed: Option<f64>) -> (f64, f64) {
        let n = self.count;
        let l = &workspace.l;
        // b = K_y⁻¹y and a = K_y⁻¹1
        workspace.b.copy_from_slice(self.y);
        cholesky_solve(l, n, &mut workspace.b);
        let mean = match fixed {
            Some(mean) => {
                workspace.a.fill(mean);
                cholesky_solve(l, n, &mut workspace.a);
                mean
            }
            None => {
                workspace.a.fill(1.0);
                cholesky_solve(l, n, &mut workspace.a);
                let (mut sa, mut sb) = (0.0, 0.0);
                for i in 0..n {
                    sa += workspace.a[i];
                    sb += workspace.b[i];
                }
                let mean = sb / sa;
                for a in &mut workspace.a {
                    *a *= mean;
                }
                mean
            }
        };
        // α = K_y⁻¹y − K_y⁻¹(m 1)
        let mut fit = 0.0;
        for i in 0..n {
            workspace.b[i] -= workspace.a[i];
            fit += (self.y[i] - mean) * workspace.b[i];
        }
        let mut half_log_det = 0.0;
        for i in 0..n {
            half_log_det += ln(l[i * n + i]);
        }
        let log_likelihood = -0.5 * fit - half_log_det - 0.5 * n as f64 * ln(std::f64::consts::TAU);
        (mean, log_likelihood)
    }

    // the log marginal likelihood at the log-hyperparameters `log`, the mean at its best, with
    // its gradient (eq. 5.9) into `gradient`; NaN if K_y doesn't factor
    fn log_likelihood(
        &self,
        log: &[f64],
        workspace: &mut Workspace,
        gradient: Option<&mut [f64]>,
    ) -> f64 {
        let (n, dims) = (self.count, self.dims);
        let (length_scales, signal, noise) = unpack(log, dims, self.noise);
        if self
            .factor(&length_scales, signal, noise, workspace)
            .is_none()
        {
            return f64::NAN;
        }
        let (_, value) = self.solve(workspace, None);
        let Some(gradient) = gradient else {
            return value;
        };
        // K_y⁻¹ from its factor, and W = ααᵀ − K_y⁻¹: ∂/∂θⱼ = ½ Σ_pq W_pq ∂K_pq/∂θⱼ, each
        // pair p ≠ q twice by symmetry
        workspace.inverse.clear();
        workspace.inverse.resize(n * n, 0.0);
        for i in 0..n {
            workspace.inverse[i * n + i] = 1.0;
        }
        cholesky_solve_multi(&workspace.l, n, &mut workspace.inverse, n);
        workspace.diff.clear();
        workspace.diff.resize(dims, 0.0);
        let alpha = &workspace.b;
        gradient.fill(0.0);
        let mut trace = 0.0;
        for p in 0..n {
            let xp = &self.x[p * dims..(p + 1) * dims];
            for q in 0..p {
                let xq = &self.x[q * dims..(q + 1) * dims];
                let w = alpha[p] * alpha[q] - workspace.inverse[p * n + q];
                let mut r2 = 0.0;
                for i in 0..dims {
                    let t = (xp[i] - xq[i]) / length_scales[i];
                    workspace.diff[i] = t * t;
                    r2 += t * t;
                }
                let (k, dk) = self.kernel.eval(r2);
                // ∂K_pq/∂ln ℓᵢ = σ_f² dk/d(r²) · (−2 uᵢ²), uᵢ = (x_pᵢ − x_qᵢ)/ℓᵢ
                let factor = w * signal * dk * -2.0;
                for (g, &u2) in gradient.iter_mut().zip(&workspace.diff) {
                    *g += factor * u2;
                }
                // ∂K_pq/∂ln σ_f² = σ_f² k_pq
                gradient[dims] += w * signal * k;
            }
            trace += alpha[p] * alpha[p] - workspace.inverse[p * n + p];
        }
        // the diagonal: ∂K_pp/∂ln σ_f² = σ_f², ∂K_pp/∂ln σ_n² = σ_n²
        gradient[dims] += 0.5 * signal * trace;
        if let Noise::Learned { .. } = self.noise {
            gradient[dims + 1] = 0.5 * noise * trace;
        }
        value
    }
}

/// A builder for a [`GaussianProcess`], from [`GaussianProcess::builder`].
///
/// Defaults: the [Matérn 5/2 kernel](Kernel::Matern52), no noise ([`Noise::Fixed`] of 0), 5 starts
/// of the likelihood's maximization, seed 0.
#[derive(Clone, Debug)]
pub struct GaussianProcessBuilder {
    real: Real,
    kernel: Kernel,
    noise: Noise,
    starts: usize,
    seed: u64,
    hyperparameters: Option<Hyperparameters>,
}

impl GaussianProcessBuilder {
    /// The kernel: [`Kernel::Matern52`] by default.
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// The observation noise: none by default ([`Noise::Fixed`] of 0), the model interpolating
    /// the values; [`Noise::Learned`] for a noisy function.
    pub fn noise(mut self, noise: Noise) -> Self {
        self.noise = noise;
        self
    }

    /// The number of starts of the likelihood's maximization, at least 1: 5 by default. The
    /// first from fixed values, the others from random points.
    pub fn starts(mut self, starts: usize) -> Self {
        self.starts = starts;
        self
    }

    /// The seed of the random starts: 0 by default, so a fit is reproducible unless told
    /// otherwise.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Hyperparameters to use as they are, instead of fitting them: the model is then the
    /// posterior of Rasmussen and Williams (2006, eq. 2.38, 2.25 and 2.26) with this mean and
    /// these variances and length scales. The [noise](GaussianProcessBuilder::noise) setting is
    /// ignored.
    pub fn hyperparameters(mut self, hyperparameters: Hyperparameters) -> Self {
        self.hyperparameters = Some(hyperparameters);
        self
    }

    /// Fits the model to the `values` of a function at `points`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for no points, a different number of values, a point without
    ///   a value per gene, a point or value that isn't finite, starts of 0, an invalid
    ///   [`Noise`], or given hyperparameters without a length scale per gene, with a length scale
    ///   or signal variance that isn't positive and finite, a negative or infinite noise
    ///   variance, or a mean that isn't finite.
    pub fn fit(&self, points: &[Reals], values: &[f64]) -> Result<GaussianProcess> {
        let invalid =
            |setting: &'static str, reason: String| Err(Error::InvalidSetting { setting, reason });
        if points.is_empty() {
            return invalid("points", "a Gaussian process needs at least 1 point".into());
        }
        if values.len() != points.len() {
            return invalid(
                "values",
                format!(
                    "expected a value per point, {}, got {}",
                    points.len(),
                    values.len()
                ),
            );
        }
        let genes = self.real.genome_len();
        for (index, point) in points.iter().enumerate() {
            if point.len() != genes || !point.iter().all(|x| x.is_finite()) {
                return invalid(
                    "points",
                    format!("point {index} must have {genes} finite genes, got {point:?}"),
                );
            }
        }
        if let Some(index) = values.iter().position(|y| !y.is_finite()) {
            return invalid(
                "values",
                format!("value {index} isn't finite: {}", values[index]),
            );
        }
        if self.starts == 0 {
            return invalid("starts", "must be at least 1, got 0".into());
        }
        self.noise.validate()?;
        let scaling = Scaling::new(&self.real);
        let dims = scaling.dims();
        let mut x = vec![0.0; points.len() * dims];
        for (point, unit) in points.iter().zip(x.chunks_exact_mut(dims.max(1))) {
            scaling.to_unit(point, unit);
        }
        if dims == 0 {
            x.clear();
        }
        let settings = Settings {
            kernel: self.kernel,
            noise: self.noise,
            starts: self.starts,
            seed: self.seed,
        };
        match &self.hyperparameters {
            None => GaussianProcess::fit_unit(settings, scaling, x, values, None),
            Some(h) => self.with_hyperparameters(h, settings, scaling, x, values),
        }
    }

    // the model with given hyperparameters, in the scaled units
    fn with_hyperparameters(
        &self,
        h: &Hyperparameters,
        settings: Settings,
        scaling: Scaling,
        x: Vec<f64>,
        values: &[f64],
    ) -> Result<GaussianProcess> {
        let invalid = |reason: String| {
            Err(Error::InvalidSetting {
                setting: "hyperparameters",
                reason,
            })
        };
        if h.length_scales.len() != scaling.genes() {
            return invalid(format!(
                "expected a length scale per gene, {}, got {}",
                scaling.genes(),
                h.length_scales.len()
            ));
        }
        let positive = |x: f64| x > 0.0 && x.is_finite();
        if !scaling
            .variable
            .iter()
            .all(|&i| positive(h.length_scales[i]))
            || !positive(h.signal_variance)
            || !(h.noise_variance >= 0.0 && h.noise_variance.is_finite())
            || !h.mean.is_finite()
        {
            return invalid(format!(
                "length scales and the signal variance must be positive and finite, the noise \
                 variance finite and at least 0, and the mean finite; got {h:?}"
            ));
        }
        let count = values.len();
        let (y_mean, y_scale, y) = standardize(values);
        let s2 = y_scale * y_scale;
        let length_scales: Vec<f64> = scaling
            .variable
            .iter()
            .enumerate()
            .map(|(k, &i)| h.length_scales[i] / scaling.width[k])
            .collect();
        let (signal, noise) = (h.signal_variance / s2, h.noise_variance / s2);
        let mean = (h.mean - y_mean) / y_scale;
        let data = Data {
            x: &x,
            y: &y,
            count,
            dims: scaling.dims(),
            kernel: settings.kernel,
            noise: Noise::Fixed(noise),
        };
        let mut workspace = Workspace::new(count);
        let Some(jitter) = data.factor(&length_scales, signal, noise, &mut workspace) else {
            return invalid(format!(
                "the kernel matrix doesn't factor with these hyperparameters: {h:?}"
            ));
        };
        let (mean, log_likelihood) = data.solve(&mut workspace, Some(mean));
        let mut log: Vec<f64> = length_scales.iter().map(|&l| ln(l)).collect();
        log.push(ln(signal));
        let Workspace { l, b, .. } = workspace;
        Ok(GaussianProcess {
            kernel: settings.kernel,
            scaling,
            x,
            count,
            y_mean,
            y_scale,
            mean,
            length_scales,
            signal,
            noise,
            log,
            jitter,
            factor: l,
            alpha: b,
            log_likelihood,
        })
    }
}

#[cfg(test)]
mod tests;
