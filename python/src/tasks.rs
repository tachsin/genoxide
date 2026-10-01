//! The control tasks of `genoxide::problems::control`, as the Python package describes them in
//! JSON, driven by a network's policy in Rust or by a Python callable; and `Balance`, the fitness
//! of a network's weights on a task, evaluated in Rust.

use crate::config;
use crate::neat::{PyFeedForward, PyRecurrent};
use crate::networks::{Network, NetworkConfig, NetworkPolicy};
use genoxide::Fitness;
use genoxide::problems::control::{CartPole, DoublePole, Policy};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;

type Result<T> = std::result::Result<T, String>;

/// A task, as `_describe()` of a `gx.problems.control` class gives it.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskConfig {
    CartPole {},
    DoublePole { velocities: bool },
}

/// A control task.
#[derive(Clone, Copy, Debug)]
pub enum Task {
    CartPole(CartPole),
    DoublePole(DoublePole),
}

impl Task {
    fn build(config: TaskConfig) -> Self {
        match config {
            TaskConfig::CartPole {} => Task::CartPole(CartPole::new()),
            TaskConfig::DoublePole { velocities: true } => Task::DoublePole(DoublePole::new()),
            TaskConfig::DoublePole { velocities: false } => {
                Task::DoublePole(DoublePole::without_velocities())
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Task::CartPole(_) => "CartPole",
            Task::DoublePole(_) => "DoublePole",
        }
    }

    /// The number of variables a policy observes.
    pub fn observations(&self) -> usize {
        match self {
            Task::CartPole(_) => 4,
            Task::DoublePole(task) => task.observations(),
        }
    }

    fn run(&self, policy: &mut dyn Policy, steps: u32) -> u32 {
        match self {
            Task::CartPole(task) => task.run(policy, steps),
            Task::DoublePole(task) => task.run(policy, steps),
        }
    }

    fn solved(&self, policy: &mut dyn Policy) -> bool {
        match self {
            Task::CartPole(task) => task.solved(policy),
            Task::DoublePole(task) => task.solved(policy),
        }
    }

    // the double pole, for its damping fitness and generalization test
    fn double_pole(&self, what: &str) -> Result<&DoublePole> {
        match self {
            Task::DoublePole(task) => Ok(task),
            Task::CartPole(_) => Err(format!("{what} is DoublePole's, not CartPole's")),
        }
    }

    // an error unless a network of `inputs` inputs and `outputs` outputs fits the task: an
    // input per observation, and one output
    fn check(&self, inputs: usize, outputs: usize) -> Result<()> {
        let observations = self.observations();
        if inputs != observations || outputs != 1 {
            return Err(format!(
                "{} needs a network of {observations} inputs and 1 output, not {inputs} inputs and {outputs} outputs",
                self.name(),
            ));
        }
        Ok(())
    }

    /// The state after each step of an episode of at most `steps` steps from the initial state,
    /// after resetting `policy`, the step that failed included, one after the other.
    fn episode(&self, policy: &mut dyn Policy, steps: u32) -> Vec<f64> {
        let mut states = Vec::new();
        policy.reset();
        let mut action = [0.0];
        match self {
            Task::CartPole(task) => {
                let (mut task, mut observation) = (*task, [0.0; 4]);
                for _ in 0..steps {
                    task.observe(&mut observation);
                    policy.act(&observation, &mut action);
                    let balanced = task.step(action[0]);
                    states.extend(task.state());
                    if !balanced {
                        break;
                    }
                }
            }
            Task::DoublePole(task) => {
                let (mut task, mut observation) = (*task, [0.0; 6]);
                let observation = &mut observation[..task.observations()];
                for _ in 0..steps {
                    task.observe(observation);
                    policy.act(observation, &mut action);
                    let balanced = task.step(action[0]);
                    states.extend(task.state());
                    if !balanced {
                        break;
                    }
                }
            }
        }
        states
    }

    // the number of variables of a state
    fn state_len(&self) -> usize {
        match self {
            Task::CartPole(_) => 4,
            Task::DoublePole(_) => 6,
        }
    }
}

fn parse<T: for<'de> Deserialize<'de>>(what: &str, description: &str) -> PyResult<T> {
    let mut json = serde_json::Deserializer::from_str(description);
    serde_path_to_error::deserialize(&mut json).map_err(|error| {
        let (path, error) = (error.path().to_string(), error.into_inner());
        PyValueError::new_err(format!("invalid {what} `{path}`: {error}"))
    })
}

// `act` with `policy`: a network's policy, in Rust without the GIL, or a Python callable
// `policy(observation, action)` that writes the action into its numpy array. The first exception
// of the callable is raised, and the episode ends at its next step.
fn with_policy<'py, R: Send>(
    py: Python<'py>,
    task: &Task,
    policy: &Bound<'py, PyAny>,
    act: impl FnOnce(&mut dyn Policy) -> R + Send,
) -> PyResult<R> {
    // a network's policy, or a NEAT network's evaluator, whose outputs are the actions; a copy,
    // run without the GIL
    let driver = if let Ok(policy) = policy.cast::<NetworkPolicy>() {
        Some(policy.get().driver.clone())
    } else if let Ok(evaluator) = policy.cast::<PyFeedForward>() {
        Some(evaluator.borrow().driver(1.0, 0.0))
    } else if let Ok(evaluator) = policy.cast::<PyRecurrent>() {
        Some(evaluator.borrow().driver(1.0, 0.0))
    } else {
        None
    };
    if let Some(mut driver) = driver {
        task.check(driver.inputs(), driver.outputs())
            .map_err(PyValueError::new_err)?;
        return py
            .detach(move || driver.act(act))
            .map_err(PyValueError::new_err);
    }
    if !policy.is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "a policy is a network's policy, e.g. mlp.policy(weights), a NEAT network's \
             evaluator, e.g. network.feed_forward(), or a callable policy(observation, action), \
             not {}",
            policy.repr()?
        )));
    }
    let mut error = None;
    let mut call = |observation: &[f64], action: &mut [f64]| {
        if error.is_none() {
            let called = (|| {
                let written = PyArray1::<f64>::zeros(py, action.len(), false);
                policy.call1((PyArray1::from_slice(py, observation), &written))?;
                action.copy_from_slice(written.readonly().as_slice()?);
                Ok::<(), PyErr>(())
            })();
            if let Err(raised) = called {
                error = Some(raised);
            }
        }
        // a NaN force ends the episode
        if error.is_some() {
            action.fill(f64::NAN);
        }
    };
    let result = act(&mut call);
    match error {
        Some(error) => Err(error),
        None => Ok(result),
    }
}

/// A control task of `gx.problems.control`, from its description.
#[pyclass(frozen, module = "genoxide._genoxide", name = "Task")]
pub struct PyTask {
    task: Task,
}

#[pymethods]
impl PyTask {
    #[new]
    fn new(description: &str) -> PyResult<Self> {
        Ok(Self {
            task: Task::build(parse("task", description)?),
        })
    }

    /// The number of variables a policy observes.
    #[getter]
    fn observations(&self) -> usize {
        self.task.observations()
    }

    /// The steps balanced in an episode of at most `steps` steps.
    fn run(&self, py: Python<'_>, policy: &Bound<'_, PyAny>, steps: u32) -> PyResult<u32> {
        let task = self.task;
        with_policy(py, &task, policy, |policy| task.run(policy, steps))
    }

    /// The state after each step of an episode of at most `steps` steps, a row each, until the
    /// step that failed, included.
    fn episode<'py>(
        &self,
        py: Python<'py>,
        policy: &Bound<'py, PyAny>,
        steps: u32,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let task = self.task;
        let states = with_policy(py, &task, policy, |policy| task.episode(policy, steps))?;
        let rows = states.len() / task.state_len();
        PyArray1::from_vec(py, states).reshape([rows, task.state_len()])
    }

    /// Whether `policy` solves the task.
    fn solved(&self, py: Python<'_>, policy: &Bound<'_, PyAny>) -> PyResult<bool> {
        let task = self.task;
        with_policy(py, &task, policy, |policy| task.solved(policy))
    }

    /// Gruau, Whitley and Pyeatt's damping fitness of `policy`.
    fn damping_fitness(&self, py: Python<'_>, policy: &Bound<'_, PyAny>) -> PyResult<f64> {
        let task = *self
            .task
            .double_pole("damping_fitness")
            .map_err(PyValueError::new_err)?;
        with_policy(py, &self.task, policy, |policy| {
            task.damping_fitness(policy)
        })
    }

    /// The starts of the generalization test that `policy` balances for 1000 steps.
    fn generalization(&self, py: Python<'_>, policy: &Bound<'_, PyAny>) -> PyResult<u32> {
        let task = *self
            .task
            .double_pole("generalization")
            .map_err(PyValueError::new_err)?;
        with_policy(py, &self.task, policy, |policy| task.generalization(policy))
    }
}

/// What `Balance` scores: the steps balanced, or the damping fitness.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BalanceFitness {
    Steps,
    Damping,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BalanceConfig {
    task: TaskConfig,
    network: NetworkConfig,
    fitness: BalanceFitness,
    steps: u32,
}

/// The fitness of a network's weights on a task, to maximize, evaluated in Rust: the steps
/// balanced in an episode of at most `steps`, or the damping fitness.
pub struct Balance {
    task: Task,
    network: Network,
    fitness: BalanceFitness,
    steps: u32,
}

impl Balance {
    /// The `Balance` that `description` (JSON, with the type "balance") describes.
    pub fn parse(description: &str) -> PyResult<Self> {
        let mut config: serde_json::Value = parse("problem", description)?;
        if let Some(config) = config.as_object_mut() {
            config.remove("type");
        }
        let config: BalanceConfig = serde_path_to_error::deserialize(config).map_err(|error| {
            let (path, error) = (error.path().to_string(), error.into_inner());
            PyValueError::new_err(format!("invalid problem `{path}`: {error}"))
        })?;
        let task = Task::build(config.task);
        let network = Network::build(config.network).map_err(PyValueError::new_err)?;
        task.check(network.inputs(), network.outputs())
            .map_err(PyValueError::new_err)?;
        if let BalanceFitness::Damping = config.fitness {
            task.double_pole("the damping fitness")
                .map_err(PyValueError::new_err)?;
        }
        Ok(Self {
            task,
            network,
            fitness: config.fitness,
            steps: config.steps,
        })
    }

    /// Whether `description` (JSON) describes a `Balance`.
    pub fn describes(description: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(description)
            .is_ok_and(|config| config["type"] == "balance")
    }

    /// The fitness of `weights`: invalid for weights that aren't the network's.
    pub fn evaluate(&self, weights: &[f64]) -> Fitness {
        let task = &self.task;
        let score = self
            .network
            .with_policy(weights, |policy| match self.fitness {
                BalanceFitness::Steps => f64::from(task.run(policy, self.steps)),
                BalanceFitness::Damping => match task {
                    Task::DoublePole(task) => task.damping_fitness(policy),
                    // checked when parsed
                    Task::CartPole(_) => f64::NAN,
                },
            });
        score.map_or(Fitness::invalid(), Fitness::new)
    }

    /// An error unless a run fits: one objective, maximized, and a real genome of a gene per
    /// weight.
    pub fn check(&self, run: &config::Run) -> Result<()> {
        if run.objectives.len() != 1 {
            return Err("Balance has one objective: use a single-objective algorithm".to_string());
        }
        if run.objectives[0] != config::Objective::Maximize {
            return Err(
                "Balance maximizes its fitness: pass objective=\"maximize\" (the default)"
                    .to_string(),
            );
        }
        let parameters = self.network.parameters();
        match &run.genome {
            config::Genome::Real { bounds } | config::Genome::AdaptiveReal { bounds, .. }
                if bounds.len() == parameters =>
            {
                Ok(())
            }
            config::Genome::Real { bounds } | config::Genome::AdaptiveReal { bounds, .. } => {
                Err(format!(
                    "the network has {parameters} weights, but the genome has {} genes",
                    bounds.len()
                ))
            }
            _ => Err(format!(
                "Balance needs a Real genome of {parameters} genes, the network's weights"
            )),
        }
    }
}

/// The fitness of each row of `genomes` for the `Balance` that `description` (JSON) describes,
/// evaluated without the GIL.
#[pyfunction]
pub fn balance_evaluate<'py>(
    py: Python<'py>,
    description: &str,
    genomes: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let balance = Balance::parse(description)?;
    let genomes = genomes.as_array();
    let parameters = balance.network.parameters();
    if genomes.ncols() != parameters {
        return Err(PyValueError::new_err(format!(
            "the network has {parameters} weights, but the genomes have {} genes",
            genomes.ncols()
        )));
    }
    let rows: Vec<Vec<f64>> = genomes.rows().into_iter().map(|row| row.to_vec()).collect();
    let scores: Vec<f64> = py.detach(|| {
        rows.iter()
            .map(|weights| balance.evaluate(weights).score().unwrap_or(f64::NAN))
            .collect()
    });
    Ok(PyArray1::from_vec(py, scores))
}
