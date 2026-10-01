//! NEAT's networks (`genoxide::neat`) as Python sees them: `gx.neat.Network`, the genome a NEAT
//! run's fitness function gets, with its genes and its feed-forward and recurrent evaluators,
//! which are also policies of the control tasks that run in Rust.

use crate::genes::{GenomeContext, PyGenome};
use crate::networks::{Driver, NetworkPolicy};
use crate::snapshot::Kept;
use genoxide::neat::{FeedForward, Network, NodeKind, Recurrent};
use genoxide::nn::Activation;
use genoxide::problems::control::Policy;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// The Python name of an activation function, as `gx.nn` names them.
pub fn activation_name(activation: Activation) -> &'static str {
    match activation {
        Activation::Identity => "identity",
        Activation::Tanh => "tanh",
        Activation::Sigmoid => "sigmoid",
        Activation::Relu => "relu",
        Activation::SteepSigmoid => "steep_sigmoid",
    }
}

/// A NEAT network: node genes and connection genes. A NEAT run's genome.
#[pyclass(
    frozen,
    eq,
    hash,
    skip_from_py_object,
    module = "genoxide.neat",
    name = "Network"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NeatNetwork {
    pub network: Network,
}

#[pymethods]
impl NeatNetwork {
    /// The network of `inputs` inputs and the bias, each connected to each of `outputs` outputs
    /// with the steepened sigmoid, the weights output by output: the inputs', then the bias's.
    #[staticmethod]
    fn fully_connected(inputs: usize, outputs: usize, weights: Vec<f64>) -> PyResult<Self> {
        let limit = 1 << 24;
        if !(1..=limit).contains(&inputs) || !(1..=limit).contains(&outputs) {
            return Err(PyValueError::new_err(format!(
                "inputs and outputs are 1 to 2^24, not {inputs} and {outputs}"
            )));
        }
        let needed = (inputs + 1) * outputs;
        if weights.len() != needed {
            return Err(PyValueError::new_err(format!(
                "a network of {inputs} inputs and {outputs} outputs has {needed} weights, a \
                 weight for each input and the bias, for each output, not {}",
                weights.len()
            )));
        }
        Ok(Self {
            network: Network::fully_connected(inputs, outputs, &weights),
        })
    }

    /// The number of inputs, without the bias.
    #[getter]
    fn inputs(&self) -> usize {
        self.network.inputs()
    }

    /// The number of outputs.
    #[getter]
    fn outputs(&self) -> usize {
        self.network.outputs()
    }

    /// The number of hidden nodes.
    fn hidden(&self) -> usize {
        self.network.hidden()
    }

    /// The number of enabled connections.
    fn enabled(&self) -> usize {
        self.network.enabled()
    }

    /// The node genes, by id.
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let nodes = self.network.nodes().iter().map(|node| NodeGene {
            id: node.id(),
            kind: match node.kind() {
                NodeKind::Input => "input",
                NodeKind::Bias => "bias",
                NodeKind::Output => "output",
                NodeKind::Hidden => "hidden",
            },
            activation: activation_name(node.activation()),
        });
        PyTuple::new(py, nodes)
    }

    /// The connection genes, by innovation number.
    fn connections<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let connections = self
            .network
            .connections()
            .iter()
            .map(|connection| ConnectionGene {
                innovation: connection.innovation(),
                from_: connection.from(),
                to: connection.to(),
                weight: connection.weight(),
                enabled: connection.is_enabled(),
            });
        PyTuple::new(py, connections)
    }

    /// The number of connection genes, enabled or not: the genome's length.
    fn __len__(&self) -> usize {
        genoxide::genome::Genome::len(&self.network)
    }

    /// The feed-forward evaluator: a `ValueError` if the enabled connections form a cycle.
    fn feed_forward(&self) -> PyResult<PyFeedForward> {
        let evaluator = self
            .network
            .feed_forward()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PyFeedForward {
            evaluator,
            inputs: self.network.inputs(),
            outputs: self.network.outputs(),
        })
    }

    /// The recurrent evaluator.
    fn recurrent(&self) -> PyResult<PyRecurrent> {
        let evaluator = self
            .network
            .recurrent()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PyRecurrent {
            evaluator,
            inputs: self.network.inputs(),
            outputs: self.network.outputs(),
        })
    }

    fn __repr__(&self) -> String {
        let network = &self.network;
        format!(
            "Network(inputs={}, outputs={}, hidden={}, enabled={} of {} connections)",
            network.inputs(),
            network.outputs(),
            network.hidden(),
            network.enabled(),
            network.connections().len()
        )
    }

    // pickled as its JSON, made again by `neat_network`
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let json = serde_json::to_string(&self.network)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let module = py.import("genoxide._genoxide")?;
        let make = module.getattr("neat_network")?;
        PyTuple::new(py, [make, PyTuple::new(py, [json])?.into_any()])
    }
}

/// The network that `json` describes, as `__reduce__` writes it: for pickling.
#[pyfunction]
pub fn neat_network(json: &str) -> PyResult<NeatNetwork> {
    let network: Network = serde_json::from_str(json)
        .map_err(|error| PyValueError::new_err(format!("not a NEAT network: {error}")))?;
    Ok(NeatNetwork { network })
}

/// A node gene: its id, its kind ("input", "bias", "output" or "hidden") and its activation.
#[pyclass(frozen, get_all, skip_from_py_object, module = "genoxide.neat")]
#[derive(Clone, Debug)]
pub struct NodeGene {
    id: u32,
    kind: &'static str,
    activation: &'static str,
}

#[pymethods]
impl NodeGene {
    fn __repr__(&self) -> String {
        format!(
            "NodeGene(id={}, kind='{}', activation='{}')",
            self.id, self.kind, self.activation
        )
    }
}

/// A connection gene: its innovation number, the ids of the nodes it leaves (`from_`) and enters
/// (`to`), its weight, and whether it's enabled.
#[pyclass(frozen, get_all, skip_from_py_object, module = "genoxide.neat")]
#[derive(Clone, Debug)]
pub struct ConnectionGene {
    innovation: u32,
    from_: u32,
    to: u32,
    weight: f64,
    enabled: bool,
}

#[pymethods]
impl ConnectionGene {
    fn __repr__(&self) -> String {
        format!(
            "ConnectionGene(innovation={}, from_={}, to={}, weight={}, enabled={})",
            self.innovation,
            self.from_,
            self.to,
            self.weight,
            if self.enabled { "True" } else { "False" }
        )
    }
}

/// A network's evaluator: the outputs for inputs, step by step.
trait Evaluate {
    fn activate(&mut self, input: &[f64], output: &mut [f64]);
}

impl Evaluate for FeedForward {
    fn activate(&mut self, input: &[f64], output: &mut [f64]) {
        FeedForward::activate(self, input, output);
    }
}

impl Evaluate for Recurrent {
    fn activate(&mut self, input: &[f64], output: &mut [f64]) {
        Recurrent::activate(self, input, output);
    }
}

// the outputs for `input`: a 1-D array of `inputs` values, or a 2-D array of rows of them, each
// row a step; into `out`, a 1-D float64 array of `outputs` values, for a 1-D input
fn activate<'py>(
    py: Python<'py>,
    evaluator: &mut impl Evaluate,
    (inputs, outputs): (usize, usize),
    input: &Bound<'py, PyAny>,
    out: Option<PyReadwriteArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let wrong = |values: usize| {
        PyValueError::new_err(format!(
            "the network has {inputs} inputs, not {values} input values"
        ))
    };
    let not_numbers = || {
        let text = input
            .repr()
            .map_or_else(|_| "this".to_string(), |r| r.to_string());
        PyValueError::new_err(format!(
            "the input is a 1-D or a 2-D array of numbers, not {text}"
        ))
    };
    // a float64 array as it is; anything else, e.g. a list or an array of ints, through numpy
    let converted;
    let input = if input.cast::<PyArray1<f64>>().is_ok() || input.cast::<PyArray2<f64>>().is_ok() {
        input
    } else {
        converted = py
            .import("numpy")?
            .call_method1("ascontiguousarray", (input, "float64"))
            .map_err(|_| not_numbers())?;
        &converted
    };
    if let Ok(rows) = input.extract::<PyReadonlyArray2<'py, f64>>() {
        if out.is_some() {
            return Err(PyValueError::new_err(
                "out is for a 1-D input: a 2-D input gets a new array",
            ));
        }
        let rows = rows.as_array();
        if rows.ncols() != inputs {
            return Err(wrong(rows.ncols()));
        }
        let mut values = vec![0.0; rows.nrows() * outputs];
        let mut input = vec![0.0; inputs];
        for (row, output) in rows
            .rows()
            .into_iter()
            .zip(values.chunks_exact_mut(outputs))
        {
            for (value, x) in input.iter_mut().zip(row) {
                *value = *x;
            }
            evaluator.activate(&input, output);
        }
        let array = PyArray1::from_vec(py, values);
        return Ok(array.reshape([rows.nrows(), outputs])?.into_any());
    }
    let values: Vec<f64> = input
        .extract::<PyReadonlyArray1<'py, f64>>()
        .map_err(|_| not_numbers())?
        .as_array()
        .to_vec();
    if values.len() != inputs {
        return Err(wrong(values.len()));
    }
    match out {
        Some(mut out) => {
            if out.len() != outputs {
                return Err(PyValueError::new_err(format!(
                    "out has {} values, but the network has {outputs} outputs",
                    out.len()
                )));
            }
            let slice = out
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("out is a contiguous array"))?;
            evaluator.activate(&values, slice);
            // the same array, written
            let array: Bound<'py, PyArray1<f64>> = (**out).clone();
            drop(out);
            Ok(array.into_any())
        }
        None => {
            let mut output = vec![0.0; outputs];
            evaluator.activate(&values, &mut output);
            Ok(PyArray1::from_vec(py, output).into_any())
        }
    }
}

// the map of a policy's outputs to its actions, `scale × output + offset`, checked
fn map(scale: f64, offset: f64) -> PyResult<(f64, f64)> {
    if !(scale.is_finite() && offset.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "scale and offset are finite numbers, not {scale} and {offset}"
        )));
    }
    Ok((scale, offset))
}

/// A NEAT network compiled for feed-forward evaluation, from `Network.feed_forward()`; a policy
/// of the control tasks too, run in Rust.
#[pyclass(skip_from_py_object, module = "genoxide.neat", name = "FeedForward")]
#[derive(Clone, Debug)]
pub struct PyFeedForward {
    pub evaluator: FeedForward,
    inputs: usize,
    outputs: usize,
}

#[pymethods]
impl PyFeedForward {
    #[getter]
    fn inputs(&self) -> usize {
        self.inputs
    }

    #[getter]
    fn outputs(&self) -> usize {
        self.outputs
    }

    /// The outputs for `input`, a 1-D array (into `out` if given), or for each row of a 2-D
    /// array.
    #[pyo3(signature = (input, out = None))]
    fn activate<'py>(
        &mut self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
        out: Option<PyReadwriteArray1<'py, f64>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = (self.inputs, self.outputs);
        activate(py, &mut self.evaluator, shape, input, out)
    }

    /// A policy of the control tasks: the actions are `scale × output + offset`.
    #[pyo3(signature = (*, scale = 1.0, offset = 0.0))]
    fn policy(&self, scale: f64, offset: f64) -> PyResult<NetworkPolicy> {
        let (scale, offset) = map(scale, offset)?;
        Ok(NetworkPolicy::new(self.driver(scale, offset)))
    }

    fn __repr__(&self) -> String {
        format!(
            "FeedForward(inputs={}, outputs={})",
            self.inputs, self.outputs
        )
    }
}

impl PyFeedForward {
    /// The evaluator, copied, as a policy's driver.
    pub fn driver(&self, scale: f64, offset: f64) -> Driver {
        Driver::Neat {
            evaluator: Evaluator::FeedForward(self.evaluator.clone()),
            inputs: self.inputs,
            outputs: self.outputs,
            scale,
            offset,
        }
    }
}

/// A NEAT network compiled for recurrent evaluation, from `Network.recurrent()`: a step of time
/// per activation, until `reset`; a policy of the control tasks too, run in Rust.
#[pyclass(skip_from_py_object, module = "genoxide.neat", name = "Recurrent")]
#[derive(Clone, Debug)]
pub struct PyRecurrent {
    pub evaluator: Recurrent,
    inputs: usize,
    outputs: usize,
}

#[pymethods]
impl PyRecurrent {
    #[getter]
    fn inputs(&self) -> usize {
        self.inputs
    }

    #[getter]
    fn outputs(&self) -> usize {
        self.outputs
    }

    /// One step: the outputs for `input`, a 1-D array (into `out` if given); or a step for each
    /// row of a 2-D array, in order.
    #[pyo3(signature = (input, out = None))]
    fn activate<'py>(
        &mut self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
        out: Option<PyReadwriteArray1<'py, f64>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = (self.inputs, self.outputs);
        activate(py, &mut self.evaluator, shape, input, out)
    }

    /// Sets every node back to 0, as before the first step.
    fn reset(&mut self) {
        self.evaluator.reset();
    }

    /// A policy of the control tasks: the actions are `scale × output + offset`. Each episode
    /// starts from a reset network.
    #[pyo3(signature = (*, scale = 1.0, offset = 0.0))]
    fn policy(&self, scale: f64, offset: f64) -> PyResult<NetworkPolicy> {
        let (scale, offset) = map(scale, offset)?;
        Ok(NetworkPolicy::new(self.driver(scale, offset)))
    }

    fn __repr__(&self) -> String {
        format!(
            "Recurrent(inputs={}, outputs={})",
            self.inputs, self.outputs
        )
    }
}

impl PyRecurrent {
    /// The evaluator, copied, as a policy's driver.
    pub fn driver(&self, scale: f64, offset: f64) -> Driver {
        Driver::Neat {
            evaluator: Evaluator::Recurrent(self.evaluator.clone()),
            inputs: self.inputs,
            outputs: self.outputs,
            scale,
            offset,
        }
    }
}

/// A NEAT network's evaluator, for a policy.
#[derive(Clone, Debug)]
pub enum Evaluator {
    FeedForward(FeedForward),
    Recurrent(Recurrent),
}

impl Evaluator {
    /// `act` with the evaluator as a policy whose actions are `scale × output + offset`.
    pub fn act<R>(&mut self, scale: f64, offset: f64, act: impl FnOnce(&mut dyn Policy) -> R) -> R {
        match self {
            Evaluator::FeedForward(policy) => act(&mut Mapped {
                policy,
                scale,
                offset,
            }),
            Evaluator::Recurrent(policy) => act(&mut Mapped {
                policy,
                scale,
                offset,
            }),
        }
    }
}

// a policy whose actions are mapped: `scale × output + offset`; the outputs themselves for 1
// and 0, so NEAT's sigmoid outputs in (0, 1) become forces in (−1, 1) with 2 and −1
struct Mapped<'a, P> {
    policy: &'a mut P,
    scale: f64,
    offset: f64,
}

impl<P: Policy> Policy for Mapped<'_, P> {
    fn act(&mut self, observation: &[f64], action: &mut [f64]) {
        self.policy.act(observation, action);
        if self.scale != 1.0 || self.offset != 0.0 {
            for value in action.iter_mut() {
                *value = self.scale * *value + self.offset;
            }
        }
    }

    fn reset(&mut self) {
        self.policy.reset();
    }
}

/// The networks of a generation, kept for a progress object: a tuple of `gx.neat.Network`s when
/// read.
struct KeptNetworks(Vec<Network>);

impl Kept for KeptNetworks {
    fn genomes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let networks = self.0.iter().map(|network| NeatNetwork {
            network: network.clone(),
        });
        Ok(PyTuple::new(py, networks)?.into_any())
    }
}

impl PyGenome for Network {
    fn object<'py>(&self, py: Python<'py>, _: &GenomeContext) -> PyResult<Bound<'py, PyAny>> {
        let network = NeatNetwork {
            network: self.clone(),
        };
        Ok(Bound::new(py, network)?.into_any())
    }

    fn batch<'py>(
        py: Python<'py>,
        genomes: &[&Self],
        _: &GenomeContext,
    ) -> PyResult<Bound<'py, PyAny>> {
        let networks = genomes.iter().map(|&network| NeatNetwork {
            network: network.clone(),
        });
        Ok(PyTuple::new(py, networks)?.into_any())
    }

    fn keep<'a>(
        genomes: impl ExactSizeIterator<Item = &'a Self>,
        _: &GenomeContext,
    ) -> Box<dyn Kept>
    where
        Self: 'a,
    {
        Box::new(KeptNetworks(genomes.cloned().collect()))
    }
}
