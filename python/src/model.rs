//! The Gaussian process of `genoxide::model::gp`, for `gx.model.gp`: fitted in Rust from the
//! description the Python package makes, and queried on numpy arrays.

use crate::config::{Acquisition, KernelName, NoiseConfig};
use crate::errors::{genome_setting, setting};
use genoxide::algorithm::bo;
use genoxide::genome::{Real, Reals};
use genoxide::model::gp::{GaussianProcess, Hyperparameters, Kernel, Noise};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;

/// The kernel of a name.
pub fn kernel(name: KernelName) -> Kernel {
    match name {
        KernelName::Matern52 => Kernel::Matern52,
        KernelName::SquaredExponential => Kernel::SquaredExponential,
    }
}

/// The noise of a description.
pub fn noise(noise: NoiseConfig) -> Noise {
    match noise {
        NoiseConfig::Fixed { variance } => Noise::Fixed(variance),
        NoiseConfig::Learned { min } => Noise::Learned { min },
    }
}

/// The acquisition function of a description.
pub fn acquisition(acquisition: Acquisition) -> bo::Acquisition {
    match acquisition {
        Acquisition::ExpectedImprovement => bo::Acquisition::ExpectedImprovement,
        Acquisition::LogExpectedImprovement => bo::Acquisition::LogExpectedImprovement,
        Acquisition::ProbabilityOfImprovement { xi } => {
            bo::Acquisition::ProbabilityOfImprovement { xi }
        }
        Acquisition::UpperConfidenceBound { beta } => {
            bo::Acquisition::UpperConfidenceBound { beta }
        }
    }
}

/// A fit, as `gx.model.gp.GaussianProcess.fit` describes it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Description {
    bounds: Vec<(f64, f64)>,
    kernel: KernelName,
    noise: NoiseConfig,
    starts: usize,
    seed: u64,
    hyperparameters: Option<GivenHyperparameters>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GivenHyperparameters {
    mean: f64,
    length_scales: Vec<f64>,
    signal_variance: f64,
    noise_variance: f64,
}

// two arrays of a value per point
type Columns<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// A fitted Gaussian process.
#[pyclass(frozen, module = "genoxide._genoxide", name = "GaussianProcess")]
pub struct PyGaussianProcess {
    model: GaussianProcess,
    genes: usize,
}

impl PyGaussianProcess {
    /// A fitted model of `genes` genes, for Python.
    pub fn new(model: GaussianProcess, genes: usize) -> Self {
        Self { model, genes }
    }

    // a ValueError unless `point` has a value per gene
    fn check(&self, what: &str, len: usize) -> PyResult<()> {
        if len == self.genes {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "{what} has {len} values, not one per gene of the model's {}",
                self.genes
            )))
        }
    }
}

#[pymethods]
impl PyGaussianProcess {
    /// Fits a model to `values` at `points`, a point per row, as `description` (JSON) says.
    #[new]
    fn fit(
        py: Python<'_>,
        description: &str,
        points: PyReadonlyArray2<'_, f64>,
        values: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let description: Description = serde_json::from_str(description)
            .map_err(|error| PyValueError::new_err(format!("invalid model: {error}")))?;
        let real = genome_setting(
            Real::new(description.bounds.iter().map(|&(low, high)| low..=high)),
            "Real",
        )
        .map_err(PyValueError::new_err)?;
        let genes = description.bounds.len();
        let points = points.as_array();
        if points.ncols() != genes {
            return Err(PyValueError::new_err(format!(
                "points is a point per row, a value per gene of the genome's {genes}, not of shape \
                 ({}, {})",
                points.nrows(),
                points.ncols()
            )));
        }
        let points: Vec<Reals> = points
            .rows()
            .into_iter()
            .map(|row| row.iter().copied().collect())
            .collect();
        let values: Vec<f64> = values.as_array().iter().copied().collect();
        let mut builder = GaussianProcess::builder(real)
            .kernel(kernel(description.kernel))
            .noise(noise(description.noise))
            .starts(description.starts)
            .seed(description.seed);
        if let Some(h) = description.hyperparameters {
            builder = builder.hyperparameters(Hyperparameters::new(
                h.mean,
                h.length_scales,
                h.signal_variance,
                h.noise_variance,
            ));
        }
        let model = py
            .detach(|| setting(builder.fit(&points, &values)))
            .map_err(PyValueError::new_err)?;
        Ok(Self::new(model, genes))
    }

    /// The posterior means and variances at `points`, a point per row.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Columns<'py>> {
        let points = points.as_array();
        self.check("a point", points.ncols())?;
        let mut point = vec![0.0; self.genes];
        let (mut means, mut variances) = (Vec::new(), Vec::new());
        for row in points.rows() {
            for (x, &value) in point.iter_mut().zip(row.iter()) {
                *x = value;
            }
            let prediction = self.model.predict(&point);
            means.push(prediction.mean());
            variances.push(prediction.variance());
        }
        Ok((
            PyArray1::from_vec(py, means),
            PyArray1::from_vec(py, variances),
        ))
    }

    /// The posterior mean and variance at `point`, with their gradients.
    #[expect(clippy::type_complexity)]
    fn predict_with_gradient<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(
        f64,
        f64,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let point: Vec<f64> = point.as_array().iter().copied().collect();
        self.check("point", point.len())?;
        let (mut dmean, mut dvariance) = (vec![0.0; self.genes], vec![0.0; self.genes]);
        let prediction = self
            .model
            .predict_with_gradient(&point, &mut dmean, &mut dvariance);
        Ok((
            prediction.mean(),
            prediction.variance(),
            PyArray1::from_vec(py, dmean),
            PyArray1::from_vec(py, dvariance),
        ))
    }

    /// The hyperparameters: the mean, the length scales, the signal and noise variances.
    #[getter]
    fn hyperparameters(&self) -> (f64, Vec<f64>, f64, f64) {
        let h = self.model.hyperparameters();
        (
            h.mean(),
            h.length_scales().to_vec(),
            h.signal_variance(),
            h.noise_variance(),
        )
    }

    #[getter]
    fn log_marginal_likelihood(&self) -> f64 {
        self.model.log_marginal_likelihood()
    }

    #[getter]
    fn jitter(&self) -> f64 {
        self.model.jitter()
    }

    #[getter]
    fn kernel(&self) -> &'static str {
        match self.model.kernel() {
            Kernel::SquaredExponential => "squared_exponential",
            _ => "matern52",
        }
    }

    #[getter]
    fn genes(&self) -> usize {
        self.genes
    }

    fn __len__(&self) -> usize {
        self.model.len()
    }
}
