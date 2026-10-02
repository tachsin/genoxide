//! Neural networks whose weights are a genome: a multilayer perceptron and an Elman recurrent
//! network, for neuroevolution with a fixed topology.
//!
//! A network's architecture is fixed: its layer sizes and activation functions. Its weights are
//! a [`Real`] genome of [`parameters`](Mlp::parameters) genes, which an algorithm such as
//! [`Cmaes`](crate::algorithm::Cmaes) evolves; [`representation`](Mlp::representation) gives that
//! genome's representation. In a fitness function, [`with`](Mlp::with) puts a genome's weights in
//! the network, and [`forward`](MlpNetwork::forward) computes its outputs, without allocating.
//!
//! | Network | What it is |
//! |---|---|
//! | [`Mlp`] | A multilayer perceptron (Rumelhart, Hinton and Williams 1986): layers of units, each the activation of a weighted sum of the previous layer's outputs and a bias |
//! | [`Elman`] | A simple recurrent network (Elman 1990): a hidden layer that also receives its own outputs of the previous step, for tasks where the network must remember, such as a control task without velocities |
//!
//! Both are driven by a control task of [`problems::control`](crate::problems::control) through
//! the [`Policy`](crate::problems::control::Policy) trait. This is not a training framework: there
//! is no gradient, only the forward pass that a fitness function needs.
//!
//! ```
//! use genoxide::nn::{Activation, Mlp};
//! use genoxide::prelude::*;
//!
//! // XOR with a 2-2-1 network of sigmoid units
//! const CASES: [([f64; 2], f64); 4] = [
//!     ([0.0, 0.0], 0.0),
//!     ([0.0, 1.0], 1.0),
//!     ([1.0, 0.0], 1.0),
//!     ([1.0, 1.0], 0.0),
//! ];
//! let mlp = Mlp::new([2, 2, 1], Activation::Sigmoid)?.output_activation(Activation::Sigmoid);
//! assert_eq!(mlp.parameters(), 9);
//! let squared_error = |weights: &Reals| -> Option<f64> {
//!     let mut network = mlp.with(weights).ok()?;
//!     let mut output = [0.0];
//!     let mut error = 0.0;
//!     for (input, target) in CASES {
//!         network.forward(&input, &mut output);
//!         error += (output[0] - target) * (output[0] - target);
//!     }
//!     Some(error)
//! };
//! let cmaes = Cmaes::builder(mlp.representation(-10.0..=10.0)?)
//!     .restarts(cmaes::Restarts::Bipop)
//!     .minimize()
//!     .seed(1)
//!     .build()?;
//! let outcome = Engine::new(cmaes, squared_error)
//!     .stop_when(Stop::target(0.01).or(Stop::evaluations(100_000)))
//!     .run()?;
//! assert_eq!(outcome.stop_reason(), StopReason::Target);
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! # The same bits on every platform
//!
//! The activation functions use [`math`] (`tanh`, `exp`), and every weighted sum is
//! taken in a fixed order: the inputs in order, then the bias, without fused multiply-adds. A
//! network's outputs are then the same bits on every platform, and so are the episodes of a
//! control task that it drives.
//!
//! # References
//!
//! - Rumelhart, D. E., Hinton, G. E. and Williams, R. J. (1986). Learning representations by
//!   back-propagating errors. *Nature* 323: 533-536. doi:10.1038/323533a0
//! - Elman, J. L. (1990). Finding structure in time. *Cognitive Science* 14(2): 179-211.
//!   doi:10.1207/s15516709cog1402_1
//! - Igel, C. (2003). Neuroevolution for reinforcement learning using evolution strategies.
//!   CEC 2003: 2588-2595. doi:10.1109/CEC.2003.1299414 (CMA-ES on the weights of fixed
//!   networks; networks without biases for the symmetric pole-balancing tasks)

use crate::genome::Real;
use crate::operator::check_size;
use crate::{Error, Result, math};
use std::ops::RangeInclusive;

/// An activation function: what a unit outputs for its weighted sum `x`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Activation {
    /// `x`: a linear unit.
    Identity,
    /// `tanh(x)`, in (−1, 1).
    #[default]
    Tanh,
    /// The logistic function `1 / (1 + e^−x)`, in (0, 1).
    Sigmoid,
    /// `max(0, x)`, a rectified linear unit.
    Relu,
    /// The NEAT paper's steepened sigmoid `1 / (1 + e^−4.9x)`, in (0, 1): close to linear near
    /// 0 over most of the range of [−1, 1] (Stanley and Miikkulainen 2002).
    SteepSigmoid,
}

impl Activation {
    /// The activation of `x`, the same bits on every platform.
    #[inline]
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Activation::Identity => x,
            Activation::Tanh => math::tanh(x),
            Activation::Sigmoid => 1.0 / (1.0 + math::exp(-x)),
            Activation::SteepSigmoid => 1.0 / (1.0 + math::exp(-4.9 * x)),
            // NaN stays NaN
            Activation::Relu => {
                if x < 0.0 {
                    0.0
                } else {
                    x
                }
            }
        }
    }
}

// a size of a network, at least 1 and at most 2^24
fn check_units(setting: &'static str, units: usize) -> Result<usize> {
    if units == 0 {
        return Err(Error::InvalidSetting {
            setting,
            reason: "must be at least 1, got 0".into(),
        });
    }
    check_size(setting, units)
}

// the number of weights, at most 2^24
fn check_parameters(parameters: Option<usize>) -> Result<usize> {
    let too_many = || Error::InvalidSetting {
        setting: "layers",
        reason: "the network must have at most 16777216 (2^24) weights".into(),
    };
    check_size("layers", parameters.ok_or_else(too_many)?).map_err(|_| too_many())
}

// the weights for a network of `parameters` weights
fn check_weights(parameters: usize, weights: &[f64]) -> Result<()> {
    if weights.len() == parameters {
        Ok(())
    } else {
        Err(Error::InvalidGenome {
            reason: format!("expected {parameters} weights, got {}", weights.len()),
        })
    }
}

// `sum` plus the products of `weights` and `inputs`, added in order
#[inline]
fn accumulate(mut sum: f64, weights: &[f64], inputs: &[f64]) -> f64 {
    for (weight, input) in weights.iter().zip(inputs) {
        sum += weight * input;
    }
    sum
}

// the weighted sum of `inputs` with `weights`, one per input and then the bias, if `bias`: the
// products added in order, then the bias
#[inline]
fn weighted_sum(weights: &[f64], inputs: &[f64], bias: bool) -> f64 {
    let sum = accumulate(0.0, weights, inputs);
    if bias {
        sum + weights[inputs.len()]
    } else {
        sum
    }
}

/// A multilayer perceptron: fully connected layers of units, from the inputs to the outputs.
///
/// Each unit outputs the activation of a weighted sum of the previous layer's outputs plus a
/// bias. The hidden layers share an activation function; the output layer has its own,
/// [`Identity`](Activation::Identity) unless set with
/// [`output_activation`](Mlp::output_activation).
///
/// The weights come layer by layer, and within a layer unit by unit: each unit's weights for the
/// previous layer's outputs in order, then its bias. An `[a, b, c]` network has
/// `b (a + 1) + c (b + 1)` weights, or `b a + c b` without biases.
///
/// ```
/// use genoxide::nn::{Activation, Mlp};
///
/// let mlp = Mlp::new([2, 3, 1], Activation::Tanh)?;
/// assert_eq!(mlp.parameters(), 3 * 3 + 1 * 4);
/// // one hidden unit with weights 1 and 2 and bias 0.5, the other two zero; an output unit that
/// // adds the hidden units' outputs
/// let weights = [1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
/// let mut output = [0.0];
/// mlp.with(&weights)?.forward(&[0.25, -0.5], &mut output);
/// assert_eq!(output[0], genoxide::math::tanh(0.25 - 1.0 + 0.5));
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mlp {
    layers: Vec<usize>,
    hidden: Activation,
    output: Activation,
    bias: bool,
}

impl Mlp {
    /// A network with the given layer sizes, from the inputs to the outputs, e.g. `[4, 16, 1]`,
    /// and `activation` for the hidden layers; with biases, and a linear output layer.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for fewer than two layers, a layer of no units or of more than
    /// 2^24, or more than 2^24 weights.
    pub fn new(layers: impl AsRef<[usize]>, activation: Activation) -> Result<Self> {
        let layers = layers.as_ref();
        if layers.len() < 2 {
            return Err(Error::InvalidSetting {
                setting: "layers",
                reason: format!(
                    "must have an input and an output layer, got {} layers",
                    layers.len()
                ),
            });
        }
        for &units in layers {
            check_units("layers", units)?;
        }
        let mlp = Self {
            layers: layers.to_vec(),
            hidden: activation,
            output: Activation::Identity,
            bias: true,
        };
        check_parameters(mlp.count_parameters(true))?;
        Ok(mlp)
    }

    /// Sets the output layer's activation function, e.g. [`Tanh`](Activation::Tanh) for outputs
    /// in (−1, 1). [`Identity`](Activation::Identity) by default.
    #[must_use]
    pub fn output_activation(mut self, activation: Activation) -> Self {
        self.output = activation;
        self
    }

    /// Sets whether the units have biases: `true` by default. Without, a network is an odd
    /// function of its inputs when its activations are (`tanh`, the identity), which suits a
    /// symmetric task: Igel (2003) balanced poles with fewer evaluations without biases.
    #[must_use]
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// The layer sizes, from the inputs to the outputs.
    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// The number of inputs.
    pub fn inputs(&self) -> usize {
        self.layers[0]
    }

    /// The number of outputs.
    pub fn outputs(&self) -> usize {
        self.layers[self.layers.len() - 1]
    }

    /// Whether the units have biases.
    pub fn has_bias(&self) -> bool {
        self.bias
    }

    /// The number of weights (and biases): the length of a genome.
    pub fn parameters(&self) -> usize {
        // at most 2^24, checked in `new` with biases, the most
        self.count_parameters(self.bias).unwrap_or(usize::MAX)
    }

    fn count_parameters(&self, bias: bool) -> Option<usize> {
        self.layers
            .array_windows()
            .try_fold(0_usize, |sum, &[inputs, units]| {
                let per_unit = inputs.checked_add(usize::from(bias))?;
                sum.checked_add(units.checked_mul(per_unit)?)
            })
    }

    /// The representation of this network's weights, [`parameters`](Mlp::parameters) genes, each
    /// in `bounds`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for bounds that [`Real::uniform`] rejects.
    pub fn representation(&self, bounds: RangeInclusive<f64>) -> Result<Real> {
        Real::uniform(self.parameters(), bounds)
    }

    /// The network with `weights`, ready for [`forward`](MlpNetwork::forward). It holds the
    /// buffers of the layers' outputs, so that the forward passes don't allocate: make it once
    /// per evaluation, not per pass.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidGenome`] unless there are [`parameters`](Mlp::parameters) weights.
    pub fn with<'a>(&'a self, weights: &'a [f64]) -> Result<MlpNetwork<'a>> {
        check_weights(self.parameters(), weights)?;
        let widest = self.layers[1..].iter().copied().max().unwrap_or(0);
        Ok(MlpNetwork {
            mlp: self,
            weights,
            buffers: [vec![0.0; widest], vec![0.0; widest]],
        })
    }
}

/// An [`Mlp`] with its weights, from [`Mlp::with`].
#[derive(Clone, Debug)]
pub struct MlpNetwork<'a> {
    mlp: &'a Mlp,
    weights: &'a [f64],
    buffers: [Vec<f64>; 2],
}

impl MlpNetwork<'_> {
    /// The network.
    pub fn mlp(&self) -> &Mlp {
        self.mlp
    }

    /// Computes the outputs for `input`, into `output`.
    ///
    /// # Panics
    ///
    /// If `input` isn't [`inputs`](Mlp::inputs) long, or `output`
    /// [`outputs`](Mlp::outputs) long.
    pub fn forward(&mut self, input: &[f64], output: &mut [f64]) {
        let mlp = self.mlp;
        assert_eq!(input.len(), mlp.inputs(), "the network's inputs");
        assert_eq!(output.len(), mlp.outputs(), "the network's outputs");
        let bias = usize::from(mlp.bias);
        let layers = mlp.layers.len();
        let [a, b] = &mut self.buffers;
        let (mut previous, mut next) = (a, b);
        let mut weights = self.weights;
        for layer in 1..layers {
            let (inputs, units) = (mlp.layers[layer - 1], mlp.layers[layer]);
            let values: &[f64] = if layer == 1 {
                input
            } else {
                &previous[..inputs]
            };
            let last = layer == layers - 1;
            let activation = if last { mlp.output } else { mlp.hidden };
            let target: &mut [f64] = if last { output } else { &mut next[..units] };
            let per_unit = inputs + bias;
            // tanh, the usual activation, in a loop of its own: without choosing the function
            // for each unit
            if activation == Activation::Tanh {
                for (unit, value) in target.iter_mut().enumerate() {
                    let unit_weights = &weights[unit * per_unit..(unit + 1) * per_unit];
                    *value = math::tanh(weighted_sum(unit_weights, values, mlp.bias));
                }
            } else {
                for (unit, value) in target.iter_mut().enumerate() {
                    let unit_weights = &weights[unit * per_unit..(unit + 1) * per_unit];
                    *value = activation.apply(weighted_sum(unit_weights, values, mlp.bias));
                }
            }
            weights = &weights[units * per_unit..];
            std::mem::swap(&mut previous, &mut next);
        }
    }
}

/// An Elman network (Elman 1990): a recurrent network with one hidden layer that receives the
/// inputs and its own outputs of the previous step (the context), and an output layer fed by the
/// hidden layer.
///
/// Each step, hidden unit `j` outputs `f(Σᵢ wⱼᵢ xᵢ + Σₖ uⱼₖ hₖ(t − 1) + bⱼ)`, and output unit `o`
/// outputs `g(Σⱼ vₒⱼ hⱼ(t) + cₒ)`, where `f` is the hidden activation and `g` the output's
/// ([`Identity`](Activation::Identity) unless set with
/// [`output_activation`](Elman::output_activation)). The context starts at 0, and
/// [`reset`](ElmanNetwork::reset) sets it to 0 again.
///
/// The weights come unit by unit: each hidden unit's weights for the inputs, then for the
/// context, then its bias; then each output unit's weights for the hidden units, then its bias.
/// A network of `n` inputs, `h` hidden and `m` output units has `h (n + h + 1) + m (h + 1)`
/// weights, or `h (n + h) + m h` without biases.
///
/// ```
/// use genoxide::nn::{Activation, Elman};
///
/// // one input, one hidden unit that adds the input to its previous output: a running sum
/// let elman = Elman::new(1, 1, 1, Activation::Identity)?.bias(false);
/// let weights = [1.0, 1.0, 1.0]; // input, context, output
/// let mut network = elman.with(&weights)?;
/// let mut output = [0.0];
/// for (input, sum) in [(1.0, 1.0), (2.0, 3.0), (3.0, 6.0)] {
///     network.forward(&[input], &mut output);
///     assert_eq!(output[0], sum);
/// }
/// network.reset();
/// network.forward(&[5.0], &mut output);
/// assert_eq!(output[0], 5.0);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Elman {
    inputs: usize,
    hidden: usize,
    outputs: usize,
    hidden_activation: Activation,
    output_activation: Activation,
    bias: bool,
}

impl Elman {
    /// A network of `inputs` inputs, `hidden` hidden units with `activation`, and `outputs`
    /// linear output units, with biases.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a layer of no units or of more than 2^24, or more than 2^24
    /// weights.
    pub fn new(
        inputs: usize,
        hidden: usize,
        outputs: usize,
        activation: Activation,
    ) -> Result<Self> {
        check_units("inputs", inputs)?;
        check_units("hidden", hidden)?;
        check_units("outputs", outputs)?;
        let elman = Self {
            inputs,
            hidden,
            outputs,
            hidden_activation: activation,
            output_activation: Activation::Identity,
            bias: true,
        };
        check_parameters(elman.count_parameters(true))?;
        Ok(elman)
    }

    /// Sets the output layer's activation function. [`Identity`](Activation::Identity) by
    /// default.
    #[must_use]
    pub fn output_activation(mut self, activation: Activation) -> Self {
        self.output_activation = activation;
        self
    }

    /// Sets whether the units have biases: `true` by default.
    #[must_use]
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// The number of inputs.
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// The number of hidden units, and of context units.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The number of outputs.
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Whether the units have biases.
    pub fn has_bias(&self) -> bool {
        self.bias
    }

    /// The number of weights (and biases): the length of a genome.
    pub fn parameters(&self) -> usize {
        // at most 2^24, checked in `new` with biases, the most
        self.count_parameters(self.bias).unwrap_or(usize::MAX)
    }

    fn count_parameters(&self, bias: bool) -> Option<usize> {
        let bias = usize::from(bias);
        let per_hidden = self.inputs.checked_add(self.hidden)?.checked_add(bias)?;
        let per_output = self.hidden.checked_add(bias)?;
        self.hidden
            .checked_mul(per_hidden)?
            .checked_add(self.outputs.checked_mul(per_output)?)
    }

    /// The representation of this network's weights, [`parameters`](Elman::parameters) genes,
    /// each in `bounds`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for bounds that [`Real::uniform`] rejects.
    pub fn representation(&self, bounds: RangeInclusive<f64>) -> Result<Real> {
        Real::uniform(self.parameters(), bounds)
    }

    /// The network with `weights` and a context of 0, ready for
    /// [`forward`](ElmanNetwork::forward).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidGenome`] unless there are [`parameters`](Elman::parameters) weights.
    pub fn with<'a>(&'a self, weights: &'a [f64]) -> Result<ElmanNetwork<'a>> {
        check_weights(self.parameters(), weights)?;
        Ok(ElmanNetwork {
            elman: self,
            weights,
            context: vec![0.0; self.hidden],
            hidden: vec![0.0; self.hidden],
        })
    }
}

/// An [`Elman`] network with its weights and its context, from [`Elman::with`].
#[derive(Clone, Debug)]
pub struct ElmanNetwork<'a> {
    elman: &'a Elman,
    weights: &'a [f64],
    context: Vec<f64>,
    hidden: Vec<f64>,
}

impl ElmanNetwork<'_> {
    /// The network.
    pub fn elman(&self) -> &Elman {
        self.elman
    }

    /// The hidden units' outputs of the last step: the context of the next.
    pub fn context(&self) -> &[f64] {
        &self.context
    }

    /// Sets the context to 0, as at the start.
    pub fn reset(&mut self) {
        self.context.fill(0.0);
    }

    /// A step: computes the outputs for `input` into `output`, and keeps the hidden units'
    /// outputs as the next step's context.
    ///
    /// # Panics
    ///
    /// If `input` isn't [`inputs`](Elman::inputs) long, or `output`
    /// [`outputs`](Elman::outputs) long.
    pub fn forward(&mut self, input: &[f64], output: &mut [f64]) {
        let elman = self.elman;
        assert_eq!(input.len(), elman.inputs, "the network's inputs");
        assert_eq!(output.len(), elman.outputs, "the network's outputs");
        let bias = usize::from(elman.bias);
        let per_hidden = elman.inputs + elman.hidden + bias;
        let (hidden_weights, output_weights) = self.weights.split_at(elman.hidden * per_hidden);
        for (unit, value) in self.hidden.iter_mut().enumerate() {
            let weights = &hidden_weights[unit * per_hidden..(unit + 1) * per_hidden];
            let (from_inputs, rest) = weights.split_at(elman.inputs);
            let mut sum = accumulate(0.0, from_inputs, input);
            sum = accumulate(sum, rest, &self.context);
            if elman.bias {
                sum += rest[elman.hidden];
            }
            *value = elman.hidden_activation.apply(sum);
        }
        let per_output = elman.hidden + bias;
        for (unit, value) in output.iter_mut().enumerate() {
            let weights = &output_weights[unit * per_output..(unit + 1) * per_output];
            let sum = weighted_sum(weights, &self.hidden, elman.bias);
            *value = elman.output_activation.apply(sum);
        }
        self.context.copy_from_slice(&self.hidden);
    }
}

#[cfg(test)]
// the weights written out, -1 included
#[allow(clippy::neg_multiply)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use rand::RngExt;

    const ACTIVATIONS: [Activation; 5] = [
        Activation::Identity,
        Activation::Tanh,
        Activation::Sigmoid,
        Activation::Relu,
        Activation::SteepSigmoid,
    ];

    fn random_values(n: usize, rng: &mut StreamRng) -> Vec<f64> {
        (0..n).map(|_| rng.random_range(-2.0..2.0)).collect()
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    // a unit's output, written out: the products in order, then the bias
    fn unit(activation: Activation, weights: &[f64], inputs: &[f64], bias: Option<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..inputs.len() {
            sum += weights[i] * inputs[i];
        }
        if let Some(bias) = bias {
            sum += bias;
        }
        activation.apply(sum)
    }

    // an MLP as weight matrices, one row per unit, evaluated a layer at a time
    fn naive_mlp(mlp: &Mlp, weights: &[f64], input: &[f64]) -> Vec<f64> {
        let mut rest = weights;
        let mut values = input.to_vec();
        let layers = mlp.layers();
        for layer in 1..layers.len() {
            let activation = if layer == layers.len() - 1 {
                mlp.output
            } else {
                mlp.hidden
            };
            let mut rows = Vec::new();
            for _ in 0..layers[layer] {
                let width = layers[layer - 1] + usize::from(mlp.has_bias());
                rows.push(rest[..width].to_vec());
                rest = &rest[width..];
            }
            values = rows
                .iter()
                .map(|row| {
                    let bias = mlp.has_bias().then(|| row[values.len()]);
                    unit(activation, row, &values, bias)
                })
                .collect();
        }
        assert!(rest.is_empty());
        values
    }

    #[test]
    fn activations_by_hand() {
        assert_eq!(Activation::Identity.apply(-1.5), -1.5);
        assert_eq!(Activation::Tanh.apply(0.0), 0.0);
        assert_eq!(Activation::Tanh.apply(1.0), math::tanh(1.0));
        assert!((Activation::Tanh.apply(1.0) - 0.761_594_155_955_764_9).abs() < 1e-15);
        assert_eq!(Activation::Sigmoid.apply(0.0), 0.5);
        assert!((Activation::Sigmoid.apply(2.0) - 0.880_797_077_977_882_3).abs() < 1e-15);
        assert_eq!(Activation::Sigmoid.apply(-1000.0), 0.0);
        assert_eq!(Activation::Sigmoid.apply(1000.0), 1.0);
        assert_eq!(Activation::Relu.apply(-2.0), 0.0);
        assert_eq!(Activation::Relu.apply(3.0), 3.0);
        assert!(Activation::Relu.apply(f64::NAN).is_nan());
        assert_eq!(Activation::SteepSigmoid.apply(0.0), 0.5);
        assert_eq!(
            Activation::SteepSigmoid.apply(0.5).to_bits(),
            Activation::Sigmoid.apply(2.45).to_bits()
        );
    }

    #[test]
    fn mlp_by_hand() {
        // 2 inputs, 2 hidden tanh units, 1 sigmoid output
        let mlp = Mlp::new([2, 2, 1], Activation::Tanh)
            .unwrap()
            .output_activation(Activation::Sigmoid);
        assert_eq!(mlp.parameters(), 9);
        let weights = [0.5, -1.0, 0.25, 2.0, 1.0, -0.5, 1.5, -2.0, 0.125];
        let (a, b) = (0.75, -0.5);
        let h1 = math::tanh(0.5 * a + -1.0 * b + 0.25);
        let h2 = math::tanh(2.0 * a + 1.0 * b + -0.5);
        let expected = 1.0 / (1.0 + math::exp(-(1.5 * h1 + -2.0 * h2 + 0.125)));
        let mut output = [0.0];
        mlp.with(&weights).unwrap().forward(&[a, b], &mut output);
        assert_eq!(output[0], expected);
        // h1 = tanh(1.125), h2 = tanh(0.5): 1 / (1 + e^-(1.5 h1 - 2 h2 + 0.125))
        let (h1, h2) = (0.809_301_070_201_781_7, 0.462_117_157_260_009_8);
        let by_hand = 1.0 / (1.0 + (-(1.5 * h1 - 2.0 * h2 + 0.125_f64)).exp());
        assert!((output[0] - by_hand).abs() < 1e-12, "{}", output[0]);

        // without biases: 6 weights
        let mlp = mlp.bias(false);
        assert_eq!(mlp.parameters(), 6);
        let weights = [0.5, -1.0, 2.0, 1.0, 1.5, -2.0];
        let h1 = math::tanh(0.5 * a + -1.0 * b);
        let h2 = math::tanh(2.0 * a + 1.0 * b);
        let expected = 1.0 / (1.0 + math::exp(-(1.5 * h1 + -2.0 * h2)));
        mlp.with(&weights).unwrap().forward(&[a, b], &mut output);
        assert_eq!(output[0], expected);
    }

    #[test]
    fn mlp_matches_a_naive_implementation_to_the_bit() {
        let mut rng = StreamRng::seed_from_u64(1);
        let shapes: [&[usize]; 5] = [
            &[1, 1],
            &[3, 1],
            &[4, 7, 2],
            &[5, 3, 6, 4],
            &[2, 8, 8, 8, 3],
        ];
        for layers in shapes {
            for bias in [true, false] {
                for hidden in ACTIVATIONS {
                    for output in ACTIVATIONS {
                        let mlp = Mlp::new(layers, hidden)
                            .unwrap()
                            .output_activation(output)
                            .bias(bias);
                        let weights = random_values(mlp.parameters(), &mut rng);
                        let mut network = mlp.with(&weights).unwrap();
                        let mut out = vec![0.0; mlp.outputs()];
                        // the same network on several inputs: the buffers don't carry over
                        for _ in 0..3 {
                            let input = random_values(mlp.inputs(), &mut rng);
                            network.forward(&input, &mut out);
                            let naive = naive_mlp(&mlp, &weights, &input);
                            assert_eq!(bits(&out), bits(&naive), "{layers:?} {bias}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mlp_parameters() {
        let mlp = Mlp::new([4, 16, 1], Activation::Tanh).unwrap();
        assert_eq!(mlp.parameters(), 16 * 5 + 17);
        assert_eq!(mlp.bias(false).parameters(), 16 * 4 + 16);
        // Igel's (2003) network for the double pole with velocities: 42 weights
        let mlp = Mlp::new(vec![6, 6, 1], Activation::Tanh)
            .unwrap()
            .bias(false);
        assert_eq!(mlp.parameters(), 42);
        let real = mlp.representation(-1.0..=1.0).unwrap();
        assert_eq!(real.bounds().len(), 42);
        assert_eq!((mlp.inputs(), mlp.outputs()), (6, 1));
    }

    #[test]
    fn invalid_networks_are_errors() {
        let invalid = |result: Result<Mlp>| {
            matches!(
                result,
                Err(Error::InvalidSetting {
                    setting: "layers",
                    ..
                })
            )
        };
        assert!(invalid(Mlp::new([3], Activation::Tanh)));
        assert!(invalid(Mlp::new([] as [usize; 0], Activation::Tanh)));
        assert!(invalid(Mlp::new([3, 0, 1], Activation::Tanh)));
        assert!(invalid(Mlp::new([0, 1], Activation::Tanh)));
        assert!(invalid(Mlp::new([(1 << 24) + 1, 1], Activation::Tanh)));
        // each layer within 2^24, but 5000 × 5001 weights
        assert!(invalid(Mlp::new([5000, 5000], Activation::Tanh)));
        assert!(invalid(Mlp::new([usize::MAX, 2], Activation::Tanh)));
        assert!(Mlp::new([1 << 23, 1], Activation::Tanh).is_ok());

        let elman = |i, h, o| Elman::new(i, h, o, Activation::Tanh);
        for (setting, result) in [
            ("inputs", elman(0, 1, 1)),
            ("hidden", elman(1, 0, 1)),
            ("outputs", elman(1, 1, 0)),
            ("hidden", elman(1, (1 << 24) + 1, 1)),
            ("layers", elman(1, 5000, 1)),
        ] {
            assert!(
                matches!(result, Err(Error::InvalidSetting { setting: s, .. }) if s == setting),
                "{setting}"
            );
        }

        let mlp = Mlp::new([2, 1], Activation::Tanh).unwrap();
        assert!(matches!(
            mlp.with(&[1.0; 2]),
            Err(Error::InvalidGenome { .. })
        ));
        let elman = Elman::new(2, 2, 1, Activation::Tanh).unwrap();
        assert!(matches!(
            elman.with(&[1.0; 12]),
            Err(Error::InvalidGenome { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "the network's inputs")]
    fn a_wrong_input_length_panics() {
        let mlp = Mlp::new([2, 1], Activation::Tanh).unwrap();
        mlp.with(&[1.0; 3]).unwrap().forward(&[1.0], &mut [0.0]);
    }

    #[test]
    fn elman_by_hand() {
        // 1 input, 2 hidden tanh units, 1 linear output
        let elman = Elman::new(1, 2, 1, Activation::Tanh).unwrap();
        assert_eq!(elman.parameters(), 2 * (1 + 2 + 1) + (2 + 1));
        // hidden 1: input 0.5, context 1.0 and -0.5, bias 0.1; hidden 2: input -1, context 0.25
        // and 2, bias 0; output: 1, -1, bias 0.5
        let weights = [0.5, 1.0, -0.5, 0.1, -1.0, 0.25, 2.0, 0.0, 1.0, -1.0, 0.5];
        let mut network = elman.with(&weights).unwrap();
        let mut output = [0.0];
        // the first step: a context of 0
        network.forward(&[0.8], &mut output);
        let h1 = math::tanh(0.5 * 0.8 + 1.0 * 0.0 + -0.5 * 0.0 + 0.1);
        let h2 = math::tanh(-1.0 * 0.8 + 0.25 * 0.0 + 2.0 * 0.0 + 0.0);
        assert_eq!(output[0], 1.0 * h1 + -1.0 * h2 + 0.5);
        assert_eq!(network.context(), [h1, h2]);
        // the second step sees the first's hidden outputs
        network.forward(&[-0.3], &mut output);
        let g1 = math::tanh(0.5 * -0.3 + 1.0 * h1 + -0.5 * h2 + 0.1);
        let g2 = math::tanh(-1.0 * -0.3 + 0.25 * h1 + 2.0 * h2 + 0.0);
        assert_eq!(output[0], 1.0 * g1 + -1.0 * g2 + 0.5);
        // reset: as the first step again
        network.reset();
        network.forward(&[0.8], &mut output);
        assert_eq!(output[0], 1.0 * h1 + -1.0 * h2 + 0.5);
    }

    // an Elman network written out: hidden(t) from the inputs and hidden(t − 1)
    fn naive_elman(elman: &Elman, weights: &[f64], inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (n, h, m) = (elman.inputs(), elman.hidden(), elman.outputs());
        let b = usize::from(elman.has_bias());
        let mut context = vec![0.0; h];
        let mut outputs = Vec::new();
        for input in inputs {
            let mut both = input.clone();
            both.extend(&context);
            let hidden: Vec<f64> = (0..h)
                .map(|j| {
                    let row = &weights[j * (n + h + b)..(j + 1) * (n + h + b)];
                    let bias = elman.has_bias().then(|| row[n + h]);
                    unit(elman.hidden_activation, row, &both, bias)
                })
                .collect();
            let start = h * (n + h + b);
            let output = (0..m)
                .map(|o| {
                    let row = &weights[start + o * (h + b)..start + (o + 1) * (h + b)];
                    let bias = elman.has_bias().then(|| row[h]);
                    unit(elman.output_activation, row, &hidden, bias)
                })
                .collect();
            outputs.push(output);
            context = hidden;
        }
        outputs
    }

    #[test]
    fn elman_matches_a_naive_implementation_to_the_bit() {
        let mut rng = StreamRng::seed_from_u64(2);
        for (n, h, m) in [(1, 1, 1), (3, 3, 1), (2, 5, 3), (6, 4, 2)] {
            for bias in [true, false] {
                for activation in ACTIVATIONS {
                    let elman = Elman::new(n, h, m, activation)
                        .unwrap()
                        .output_activation(Activation::Tanh)
                        .bias(bias);
                    let weights = random_values(elman.parameters(), &mut rng);
                    let inputs: Vec<Vec<f64>> =
                        (0..10).map(|_| random_values(n, &mut rng)).collect();
                    let expected = naive_elman(&elman, &weights, &inputs);
                    let mut network = elman.with(&weights).unwrap();
                    let mut output = vec![0.0; m];
                    for (input, expected) in inputs.iter().zip(&expected) {
                        network.forward(input, &mut output);
                        assert_eq!(bits(&output), bits(expected));
                    }
                }
            }
        }
    }
}
