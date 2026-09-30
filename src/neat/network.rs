//! NEAT's genome: node and connection genes, and the networks they make.

use crate::genome::Genome;
use crate::nn::Activation;
use crate::{Error, Result};
use std::hash::{Hash, Hasher};

/// What a node of a [`Network`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NodeKind {
    /// An input: its value is the input's.
    Input,
    /// The bias: an input fixed at 1, as in the NEAT paper.
    Bias,
    /// An output.
    Output,
    /// A hidden node, added by mutation.
    Hidden,
}

/// A node gene: the node's id, its kind, and its activation (the identity for inputs and the
/// bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeGene {
    id: u32,
    kind: NodeKind,
    activation: Activation,
}

impl NodeGene {
    pub(crate) fn new(id: u32, kind: NodeKind, activation: Activation) -> Self {
        Self {
            id,
            kind,
            activation,
        }
    }

    /// The node's id: inputs first, then the bias, the outputs, and the hidden nodes in the order
    /// the run added them.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Its kind.
    pub fn kind(&self) -> NodeKind {
        self.kind
    }

    /// Its activation function.
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

/// A connection gene: its innovation number, its ends, its weight, and whether it's enabled.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConnectionGene {
    innovation: u32,
    from: u32,
    to: u32,
    weight: f64,
    enabled: bool,
}

impl ConnectionGene {
    pub(crate) fn new(innovation: u32, from: u32, to: u32, weight: f64, enabled: bool) -> Self {
        Self {
            innovation,
            from,
            to,
            weight,
            enabled,
        }
    }

    /// The innovation number: the same for the same structure throughout a run, so crossover
    /// aligns genes by history.
    pub fn innovation(&self) -> u32 {
        self.innovation
    }

    /// The id of the node it leaves.
    pub fn from(&self) -> u32 {
        self.from
    }

    /// The id of the node it enters.
    pub fn to(&self) -> u32 {
        self.to
    }

    /// Its weight.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Whether it's enabled: a disabled gene is inherited but not expressed.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }

    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

// compared and hashed by the weight's bits, as `Reals` are: every genome equals itself
impl PartialEq for ConnectionGene {
    fn eq(&self, other: &Self) -> bool {
        self.innovation == other.innovation
            && self.from == other.from
            && self.to == other.to
            && self.weight.to_bits() == other.weight.to_bits()
            && self.enabled == other.enabled
    }
}

impl Eq for ConnectionGene {}

impl Hash for ConnectionGene {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.innovation.hash(state);
        self.from.hash(state);
        self.to.hash(state);
        self.weight.to_bits().hash(state);
        self.enabled.hash(state);
    }
}

/// A NEAT genome: a network of node genes, sorted by id (the inputs, the bias, the outputs, then
/// the hidden nodes), and connection genes, sorted by innovation number.
///
/// Its [`len`](Genome::len) is the number of connection genes, enabled or not. Evaluate it with
/// [`feed_forward`](Network::feed_forward).
///
/// ```
/// use genoxide::neat::Network;
///
/// // two inputs fully connected to one output, with the paper's steepened sigmoid
/// let network = Network::fully_connected(2, 1, &[0.5, -0.5, 0.0]);
/// let mut evaluator = network.feed_forward()?;
/// let mut output = [0.0];
/// evaluator.activate(&[1.0, 1.0], &mut output);
/// assert_eq!(output[0], 0.5); // 0.5 - 0.5 + 0 × bias, through the sigmoid at 0
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Network {
    inputs: u32,
    outputs: u32,
    nodes: Vec<NodeGene>,
    connections: Vec<ConnectionGene>,
}

impl Network {
    // the network of `inputs`, the bias and `outputs`, without connections
    pub(crate) fn minimal(inputs: u32, outputs: u32, activation: Activation) -> Self {
        let mut nodes = Vec::with_capacity((inputs + 1 + outputs) as usize);
        nodes
            .extend((0..inputs).map(|id| NodeGene::new(id, NodeKind::Input, Activation::Identity)));
        nodes.push(NodeGene::new(inputs, NodeKind::Bias, Activation::Identity));
        nodes.extend(
            (0..outputs).map(|k| NodeGene::new(inputs + 1 + k, NodeKind::Output, activation)),
        );
        Self {
            inputs,
            outputs,
            nodes,
            connections: Vec::new(),
        }
    }

    /// The network of `inputs` inputs and the bias, each connected to each of `outputs` outputs
    /// with the paper's steepened sigmoid, the weights in order: output by output, the inputs'
    /// then the bias's. Its innovation numbers are those a [`Neat`](super::Neat) run gives its
    /// initial networks.
    ///
    /// # Panics
    ///
    /// If `weights` doesn't have (inputs + 1) × outputs values.
    pub fn fully_connected(inputs: usize, outputs: usize, weights: &[f64]) -> Self {
        let (ins, outs) = (inputs as u32, outputs as u32);
        assert_eq!(
            weights.len(),
            (inputs + 1) * outputs,
            "a weight for each input and the bias, for each output"
        );
        let mut network = Self::minimal(ins, outs, Activation::SteepSigmoid);
        let mut weights = weights.iter();
        for k in 0..outs {
            for from in 0..=ins {
                let innovation = network.connections.len() as u32;
                let weight = *weights.next().expect("a weight");
                network.connections.push(ConnectionGene::new(
                    innovation,
                    from,
                    ins + 1 + k,
                    weight,
                    true,
                ));
            }
        }
        network
    }

    pub(crate) fn from_genes(
        inputs: u32,
        outputs: u32,
        nodes: Vec<NodeGene>,
        connections: Vec<ConnectionGene>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            nodes,
            connections,
        }
    }

    /// The number of inputs, without the bias.
    pub fn inputs(&self) -> usize {
        self.inputs as usize
    }

    /// The number of outputs.
    pub fn outputs(&self) -> usize {
        self.outputs as usize
    }

    /// The node genes, sorted by id.
    pub fn nodes(&self) -> &[NodeGene] {
        &self.nodes
    }

    /// The connection genes, sorted by innovation number.
    pub fn connections(&self) -> &[ConnectionGene] {
        &self.connections
    }

    /// The number of hidden nodes.
    pub fn hidden(&self) -> usize {
        self.nodes.len() - (self.inputs + 1 + self.outputs) as usize
    }

    /// The number of enabled connections.
    pub fn enabled(&self) -> usize {
        self.connections.iter().filter(|c| c.enabled).count()
    }

    pub(crate) fn connections_mut(&mut self) -> &mut [ConnectionGene] {
        &mut self.connections
    }

    pub(crate) fn node(&self, id: u32) -> Option<&NodeGene> {
        self.nodes
            .binary_search_by_key(&id, |node| node.id)
            .ok()
            .map(|index| &self.nodes[index])
    }

    pub(crate) fn has_node(&self, id: u32) -> bool {
        self.node(id).is_some()
    }

    // the connection from `from` to `to`, enabled or not
    pub(crate) fn connection_between(&self, from: u32, to: u32) -> Option<usize> {
        self.connections
            .iter()
            .position(|c| c.from == from && c.to == to)
    }

    pub(crate) fn add_node(&mut self, node: NodeGene) {
        let index = self.nodes.partition_point(|n| n.id < node.id);
        self.nodes.insert(index, node);
    }

    pub(crate) fn add_connection(&mut self, connection: ConnectionGene) {
        let index = self
            .connections
            .partition_point(|c| c.innovation < connection.innovation);
        self.connections.insert(index, connection);
    }

    // whether a path of enabled or disabled connections leads from `from` to `to`: adding
    // `to -> from` would then close a cycle
    pub(crate) fn reaches(&self, from: u32, to: u32) -> bool {
        if from == to {
            return true;
        }
        let mut seen = vec![from];
        let mut stack = vec![from];
        while let Some(node) = stack.pop() {
            for connection in self.connections.iter().filter(|c| c.from == node) {
                if connection.to == to {
                    return true;
                }
                if !seen.contains(&connection.to) {
                    seen.push(connection.to);
                    stack.push(connection.to);
                }
            }
        }
        false
    }

    /// The enabled connections compiled into a feed-forward evaluator: each node computed once,
    /// after the nodes it depends on, its inputs summed in innovation order.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidGenome`] if the enabled connections form a cycle (a network of a run with
    /// recurrent connections).
    pub fn feed_forward(&self) -> Result<FeedForward> {
        let index_of = |id: u32| {
            self.nodes
                .binary_search_by_key(&id, |node| node.id)
                .expect("a connection's nodes are the network's")
        };
        let n = self.nodes.len();
        // Kahn's algorithm over the enabled connections, the ready nodes taken in id order
        let mut incoming = vec![0usize; n];
        for c in self.connections.iter().filter(|c| c.enabled) {
            incoming[index_of(c.to)] += 1;
        }
        let mut ready: std::collections::BTreeSet<usize> =
            (0..n).filter(|&i| incoming[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(i) = ready.pop_first() {
            order.push(i);
            let id = self.nodes[i].id;
            for c in self
                .connections
                .iter()
                .filter(|c| c.enabled && c.from == id)
            {
                let j = index_of(c.to);
                incoming[j] -= 1;
                if incoming[j] == 0 {
                    ready.insert(j);
                }
            }
        }
        if order.len() < n {
            return Err(Error::InvalidGenome {
                reason: "the enabled connections form a cycle: not a feed-forward network"
                    .to_string(),
            });
        }
        // each computed node: its activation, and its incoming connections in innovation order
        let mut steps = Vec::new();
        let mut links = Vec::new();
        for &i in &order {
            let node = self.nodes[i];
            if matches!(node.kind, NodeKind::Input | NodeKind::Bias) {
                continue;
            }
            let start = links.len();
            for c in self
                .connections
                .iter()
                .filter(|c| c.enabled && c.to == node.id)
            {
                links.push((index_of(c.from) as u32, c.weight));
            }
            steps.push(Step {
                node: i as u32,
                links: start as u32..links.len() as u32,
                activation: node.activation,
            });
        }
        let outputs = (0..self.outputs)
            .map(|k| index_of(self.inputs + 1 + k) as u32)
            .collect();
        Ok(FeedForward {
            inputs: self.inputs as usize,
            bias: index_of(self.inputs) as u32,
            steps,
            links,
            outputs,
            values: vec![0.0; n],
        })
    }
}

impl Genome for Network {
    /// The number of connection genes, enabled or not.
    fn len(&self) -> usize {
        self.connections.len()
    }
}

// a computed node: where its value goes, its incoming links and its activation
#[derive(Clone, Debug)]
struct Step {
    node: u32,
    links: std::ops::Range<u32>,
    activation: Activation,
}

/// A [`Network`] compiled for evaluation, from [`Network::feed_forward`]: the nodes in
/// topological order, each the activation of the weighted sum of its inputs. A node without
/// enabled inputs has the activation of 0, and an output that nothing reaches too.
///
/// [`activate`](FeedForward::activate) doesn't allocate.
#[derive(Clone, Debug)]
pub struct FeedForward {
    inputs: usize,
    bias: u32,
    steps: Vec<Step>,
    links: Vec<(u32, f64)>,
    outputs: Vec<u32>,
    values: Vec<f64>,
}

impl FeedForward {
    /// The outputs for `input`, the same bits on every platform.
    ///
    /// # Panics
    ///
    /// If `input` or `output` has another length than the network's inputs or outputs.
    pub fn activate(&mut self, input: &[f64], output: &mut [f64]) {
        assert_eq!(input.len(), self.inputs, "an input value per input");
        assert_eq!(
            output.len(),
            self.outputs.len(),
            "an output value per output"
        );
        self.values[..self.inputs].copy_from_slice(input);
        self.values[self.bias as usize] = 1.0;
        for step in &self.steps {
            let mut sum = 0.0;
            for &(from, weight) in &self.links[step.links.start as usize..step.links.end as usize] {
                sum += weight * self.values[from as usize];
            }
            self.values[step.node as usize] = step.activation.apply(sum);
        }
        for (value, &node) in output.iter_mut().zip(&self.outputs) {
            *value = self.values[node as usize];
        }
    }
}
