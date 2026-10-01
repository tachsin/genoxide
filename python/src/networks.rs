//! The neural networks of `genoxide::nn`, as the Python package describes them in JSON: the
//! networks of `gx.nn`, their forward passes, and their policies for the control tasks.

use crate::errors::setting;
use crate::neat::Evaluator;
use genoxide::nn::{Activation, Elman, Mlp};
use genoxide::problems::control::Policy;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;

type Result<T> = std::result::Result<T, String>;

/// An activation function, by its Python name.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationName {
    Identity,
    Tanh,
    Sigmoid,
    Relu,
    SteepSigmoid,
}

impl From<ActivationName> for Activation {
    fn from(name: ActivationName) -> Self {
        match name {
            ActivationName::Identity => Activation::Identity,
            ActivationName::Tanh => Activation::Tanh,
            ActivationName::Sigmoid => Activation::Sigmoid,
            ActivationName::Relu => Activation::Relu,
            ActivationName::SteepSigmoid => Activation::SteepSigmoid,
        }
    }
}

/// A network, as `_describe()` of a `gx.nn` class gives it.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum NetworkConfig {
    Mlp {
        layers: Vec<usize>,
        activation: ActivationName,
        output_activation: ActivationName,
        bias: bool,
    },
    Elman {
        inputs: usize,
        hidden: usize,
        outputs: usize,
        activation: ActivationName,
        output_activation: ActivationName,
        bias: bool,
    },
}

/// A network of `genoxide::nn`, without its weights.
#[derive(Clone, Debug)]
pub enum Network {
    Mlp(Mlp),
    Elman(Elman),
}

impl Network {
    /// The network that `config` describes.
    pub fn build(config: NetworkConfig) -> Result<Self> {
        Ok(match config {
            NetworkConfig::Mlp {
                layers,
                activation,
                output_activation,
                bias,
            } => Network::Mlp(
                setting(Mlp::new(layers, activation.into()))?
                    .output_activation(output_activation.into())
                    .bias(bias),
            ),
            NetworkConfig::Elman {
                inputs,
                hidden,
                outputs,
                activation,
                output_activation,
                bias,
            } => Network::Elman(
                setting(Elman::new(inputs, hidden, outputs, activation.into()))?
                    .output_activation(output_activation.into())
                    .bias(bias),
            ),
        })
    }

    /// The number of weights: the length of a genome.
    pub fn parameters(&self) -> usize {
        match self {
            Network::Mlp(mlp) => mlp.parameters(),
            Network::Elman(elman) => elman.parameters(),
        }
    }

    pub fn inputs(&self) -> usize {
        match self {
            Network::Mlp(mlp) => mlp.inputs(),
            Network::Elman(elman) => elman.inputs(),
        }
    }

    pub fn outputs(&self) -> usize {
        match self {
            Network::Mlp(mlp) => mlp.outputs(),
            Network::Elman(elman) => elman.outputs(),
        }
    }

    // an error unless there are as many weights as the network has
    fn check(&self, weights: &[f64]) -> Result<()> {
        let parameters = self.parameters();
        if weights.len() == parameters {
            Ok(())
        } else {
            Err(format!(
                "the network has {parameters} weights, not {}",
                weights.len()
            ))
        }
    }

    /// `act` with the network of `weights` as a policy, a recurrent one with a context of 0.
    pub fn with_policy<R>(
        &self,
        weights: &[f64],
        act: impl FnOnce(&mut dyn Policy) -> R,
    ) -> Result<R> {
        self.check(weights)?;
        Ok(match self {
            Network::Mlp(mlp) => act(&mut setting(mlp.with(weights))?),
            Network::Elman(elman) => act(&mut setting(elman.with(weights))?),
        })
    }

    // the outputs for `inputs`, a row of `inputs()` values each, into `outputs`, a row of
    // `outputs()` each; for an Elman network, the rows are the steps of a sequence
    fn forward(&self, weights: &[f64], inputs: &[f64], outputs: &mut [f64]) -> Result<()> {
        let (n, m) = (self.inputs(), self.outputs());
        let rows = inputs.chunks_exact(n).zip(outputs.chunks_exact_mut(m));
        self.with_policy(weights, |policy| {
            for (input, output) in rows {
                policy.act(input, output);
            }
        })
    }
}

/// The network that `description` (JSON) describes.
pub fn parse(description: &str) -> PyResult<Network> {
    let mut json = serde_json::Deserializer::from_str(description);
    let config: NetworkConfig = serde_path_to_error::deserialize(&mut json).map_err(|error| {
        let (path, error) = (error.path().to_string(), error.into_inner());
        PyValueError::new_err(format!("invalid network `{path}`: {error}"))
    })?;
    Network::build(config).map_err(PyValueError::new_err)
}

/// A network of `gx.nn`, from its description.
#[pyclass(frozen, module = "genoxide._genoxide", name = "Network")]
pub struct PyNetwork {
    network: Network,
}

#[pymethods]
impl PyNetwork {
    #[new]
    fn new(description: &str) -> PyResult<Self> {
        Ok(Self {
            network: parse(description)?,
        })
    }

    /// The number of weights.
    #[getter]
    fn parameters(&self) -> usize {
        self.network.parameters()
    }

    #[getter]
    fn inputs(&self) -> usize {
        self.network.inputs()
    }

    #[getter]
    fn outputs(&self) -> usize {
        self.network.outputs()
    }

    /// The outputs for `inputs`, a row each, with `weights`, computed without the GIL.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        weights: PyReadonlyArray1<'py, f64>,
        inputs: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let network = &self.network;
        let inputs = inputs.as_array();
        if inputs.ncols() != network.inputs() {
            return Err(PyValueError::new_err(format!(
                "the network has {} inputs, but the rows have {} values",
                network.inputs(),
                inputs.ncols()
            )));
        }
        // copies: Python may change the arrays while the GIL is released
        let weights = weights.as_array().to_vec();
        let inputs: Vec<f64> = inputs.iter().copied().collect();
        let rows = inputs.len() / network.inputs();
        let outputs = py.detach(|| {
            let mut outputs = vec![0.0; rows * network.outputs()];
            network
                .forward(&weights, &inputs, &mut outputs)
                .map(|()| outputs)
        });
        let outputs = outputs.map_err(PyValueError::new_err)?;
        let outputs = Array2::from_shape_vec((rows, network.outputs()), outputs)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(outputs.into_pyarray(py))
    }

    /// The network with `weights`, as a policy for the control tasks.
    fn policy(&self, weights: PyReadonlyArray1<'_, f64>) -> PyResult<NetworkPolicy> {
        let weights = weights.as_array().to_vec();
        self.network
            .check(&weights)
            .map_err(PyValueError::new_err)?;
        Ok(NetworkPolicy::new(Driver::Weights {
            network: self.network.clone(),
            weights,
        }))
    }
}

/// What drives a policy: a network of `gx.nn` with its weights, or a NEAT network's evaluator
/// whose actions are `scale × output + offset`.
#[derive(Clone, Debug)]
pub enum Driver {
    Weights {
        network: Network,
        weights: Vec<f64>,
    },
    Neat {
        evaluator: Evaluator,
        inputs: usize,
        outputs: usize,
        scale: f64,
        offset: f64,
    },
}

impl Driver {
    /// The number of inputs: what the policy observes.
    pub fn inputs(&self) -> usize {
        match self {
            Driver::Weights { network, .. } => network.inputs(),
            Driver::Neat { inputs, .. } => *inputs,
        }
    }

    /// The number of outputs: its actions.
    pub fn outputs(&self) -> usize {
        match self {
            Driver::Weights { network, .. } => network.outputs(),
            Driver::Neat { outputs, .. } => *outputs,
        }
    }

    /// `act` with the policy, from its start: a recurrent network with a context of 0.
    pub fn act<R>(&mut self, act: impl FnOnce(&mut dyn Policy) -> R) -> Result<R> {
        match self {
            Driver::Weights { network, weights } => network.with_policy(weights, act),
            Driver::Neat {
                evaluator,
                scale,
                offset,
                ..
            } => Ok(evaluator.act(*scale, *offset, act)),
        }
    }
}

/// A network with its weights, or a NEAT network's evaluator: a policy for the control tasks
/// that runs in Rust.
#[pyclass(frozen, module = "genoxide._genoxide", name = "Policy")]
pub struct NetworkPolicy {
    pub driver: Driver,
}

impl NetworkPolicy {
    pub fn new(driver: Driver) -> Self {
        Self { driver }
    }
}

#[pymethods]
impl NetworkPolicy {
    /// The number of inputs: what the policy observes.
    #[getter]
    fn inputs(&self) -> usize {
        self.driver.inputs()
    }

    /// The number of outputs: its actions.
    #[getter]
    fn outputs(&self) -> usize {
        self.driver.outputs()
    }

    fn __repr__(&self) -> String {
        match &self.driver {
            Driver::Weights { network, .. } => {
                let kind = match network {
                    Network::Mlp(_) => "Mlp",
                    Network::Elman(_) => "Elman",
                };
                format!("<policy of an {kind} of {} weights>", network.parameters())
            }
            Driver::Neat {
                evaluator,
                scale,
                offset,
                ..
            } => {
                let kind = match evaluator {
                    Evaluator::FeedForward(_) => "feed-forward",
                    Evaluator::Recurrent(_) => "recurrent",
                };
                format!("<policy of a {kind} NEAT network, actions {scale} × output + {offset}>")
            }
        }
    }
}
