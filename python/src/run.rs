//! Building the algorithm a run describes, and running it with a Python fitness function.

use crate::checkpoint::Checkpoints;
use crate::config;
use crate::control::{
    CmaesSettings, DeSettings, EsSettings, GaSettings, IslandsSettings, LocalSearchSettings,
    NeatSettings, NelderMeadSettings, OpenEsSettings, PsoSettings, Running, Settings, Slot,
};
use crate::errors::{genome_setting, setting};
use crate::fitness::{Multi, Native, Shared, Single};
use crate::genes::{GenomeContext, PyGenome};
use crate::operators::{
    AnySelect, ListCrossover, OrderCrossovers, OrderMutation, RealCrossover, RealMutation,
    bit_flip, integer_mutation, self_adaptive,
};
use crate::problems;
use crate::snapshot::{Snapshot, objective_values};
use crate::tasks::Balance;
use crate::tree_problems::TreeFitness;
use crate::trees::tree_algorithm;
use genoxide::algorithm::islands::{Migrate, Topology};
use genoxide::algorithm::nelder_mead::Coefficients;
use genoxide::algorithm::{GaBuilder, Islands, Reevaluate, cmaes, es, pso};
use genoxide::engine::Progress;
use genoxide::genome::{AdaptiveReal, Representation};
use genoxide::multi::{self, Decomposition, MultiObjectiveAlgorithm, MultiSnapshot, SmsEmoa};
use genoxide::neat::{self, Neat};
use genoxide::operator::{Crossover, Mutate};
use genoxide::prelude::*;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

type Result<T> = std::result::Result<T, String>;

/// Runs the optimization that `config` (JSON, from the Python package) describes, with
/// `fitness`, called with a genome or, with `batch`, a generation of genomes; with `parallel`,
/// from several threads at once. With `problem`, a test problem of `genoxide::problems` or
/// `genoxide::multi::problems` (JSON), the problem is evaluated in Rust instead, and `fitness` and
/// `batch` aren't used.
/// `on_generation` is called after every generation with the generation, the evaluations, the
/// seconds and the best fitness (or the size of the front), and returns False to stop the run.
/// `control`, for a single-objective algorithm, is called once per generation after it (after
/// the last one too, and not after a re-evaluation), with a [`Running`] handle to the algorithm
/// and the same arguments as `on_generation`, to change the algorithm's settings or re-evaluate
/// it. With `checkpoint`, the run saves a checkpoint there every `checkpoint_every` generations
/// and when it stops; with `resume`, it continues from the checkpoint there, saved with the same
/// settings. Returns the result as a dict.
#[pyfunction]
#[pyo3(signature = (config, fitness, batch = false, parallel = false, on_generation = None, problem = None, control = None, checkpoint = None, checkpoint_every = None, resume = None))]
#[allow(clippy::too_many_arguments)]
pub fn run<'py>(
    py: Python<'py>,
    config: &str,
    fitness: Py<PyAny>,
    batch: bool,
    parallel: bool,
    on_generation: Option<Py<PyAny>>,
    problem: Option<&str>,
    control: Option<Py<PyAny>>,
    checkpoint: Option<PathBuf>,
    checkpoint_every: Option<u64>,
    resume: Option<PathBuf>,
) -> PyResult<Bound<'py, PyDict>> {
    // the error names the setting, e.g. `stop.generations`
    let mut json = serde_json::Deserializer::from_str(config);
    let run: config::Run = serde_path_to_error::deserialize(&mut json).map_err(|error| {
        let (path, error) = (error.path().to_string(), error.into_inner());
        PyValueError::new_err(format!("invalid setting `{path}`: {error}"))
    })?;
    let checkpoints = checkpoints(config, checkpoint, checkpoint_every, resume)?;
    // a network's weights balancing poles, or a test problem
    let (balance, problem) = match problem {
        Some(problem) if Balance::describes(problem) => (Some(Balance::parse(problem)?), None),
        problem => (None, problem.map(problems::parse).transpose()?),
    };
    if let Some(problem) = &problem {
        check_problem(problem, &run).map_err(PyValueError::new_err)?;
    }
    if let Some(balance) = &balance {
        balance.check(&run).map_err(PyValueError::new_err)?;
    }
    // a fitness of trees evaluated in Rust, and the primitive set of a run's trees
    let tree = TreeFitness::from_object(fitness.bind(py));
    let genome_context = match &run.genome {
        config::Genome::Gp(gp) => GenomeContext::Tree(Arc::new(gp.primitives.clone())),
        _ if tree.is_some() => {
            return Err(PyValueError::new_err(
                "a fitness of trees needs a gx.gp.Gp genome",
            ));
        }
        _ => GenomeContext::None,
    };
    let context = Context {
        shared: Shared::new(fitness, batch, parallel, on_generation, genome_context),
        objectives: run
            .objectives
            .iter()
            .map(|objective| match objective {
                config::Objective::Maximize => Objective::Maximize,
                config::Objective::Minimize => Objective::Minimize,
            })
            .collect(),
        stop: run.stop,
        parallel,
        problem,
        balance,
        tree,
        control,
        checkpoints,
    };
    let result = match run.genome {
        config::Genome::Binary { length } => with_operators(
            py,
            genome_setting(Binary::new(length), "Binary"),
            run.algorithm,
            &context,
            |crossover| ListCrossover::new(crossover, "binary"),
            bit_flip,
        ),
        config::Genome::Integer { bounds } => with_operators(
            py,
            genome_setting(
                Integer::new(bounds.iter().map(|&(low, high)| low..=high)),
                "Integer",
            ),
            run.algorithm,
            &context,
            |crossover| ListCrossover::new(crossover, "integer"),
            integer_mutation,
        ),
        config::Genome::Real { bounds } => {
            let real = genome_setting(
                Real::new(bounds.iter().map(|&(low, high)| low..=high)),
                "Real",
            );
            real_algorithm(py, real, run.algorithm, &context)
        }
        config::Genome::Permutation { length } => with_operators(
            py,
            genome_setting(Permutation::new(length), "Permutation"),
            run.algorithm,
            &context,
            OrderCrossovers::new,
            OrderMutation::new,
        ),
        config::Genome::AdaptiveReal {
            bounds,
            initial_step,
        } => with_operators(
            py,
            genome_setting(
                Real::new(bounds.iter().map(|&(low, high)| low..=high)),
                "Real",
            )
            .and_then(|real| genome_setting(AdaptiveReal::new(real, initial_step), "AdaptiveReal")),
            run.algorithm,
            &context,
            |crossover| ListCrossover::new(crossover, "adaptive real"),
            self_adaptive,
        ),
        config::Genome::Network { inputs, outputs } => {
            neat_algorithm(py, inputs, outputs, run.algorithm, &context)
        }
        config::Genome::Gp(gp) => {
            tree_algorithm(py, crate::trees::build_gp(*gp), run.algorithm, &context)
        }
    };
    result.map_err(|error| match error {
        Failure::Setting(message) => PyValueError::new_err(message),
        Failure::Python(error) => error,
    })
}

// the checkpoints of a run: its settings, the genome, the objectives and the algorithm's (not the
// stop conditions, which can change), where to save and what to resume from
fn checkpoints(
    config: &str,
    save: Option<PathBuf>,
    every: Option<u64>,
    resume: Option<PathBuf>,
) -> PyResult<Checkpoints> {
    let mut settings: serde_json::Value =
        serde_json::from_str(config).map_err(|error| PyValueError::new_err(error.to_string()))?;
    if let Some(settings) = settings.as_object_mut() {
        settings.remove("stop");
    }
    let save = match (save, every) {
        (Some(path), Some(every)) => Some((path, every)),
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "checkpoint and checkpoint_every go together",
            ));
        }
    };
    // an OSError such as FileNotFoundError if it can't be read
    let resume = match resume {
        Some(path) => {
            let bytes = std::fs::read(&path)?;
            Some((path, bytes))
        }
        None => None,
    };
    Ok(Checkpoints {
        settings: settings.to_string(),
        save,
        resume,
    })
}

// why a run failed: its description, or Python (the fitness function, the progress callback,
// Ctrl+C)
pub(crate) enum Failure {
    Setting(String),
    Python(PyErr),
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Failure::Setting(message)
    }
}

impl From<PyErr> for Failure {
    fn from(error: PyErr) -> Self {
        Failure::Python(error)
    }
}

pub(crate) type Returns<'py> = std::result::Result<Bound<'py, PyDict>, Failure>;

// a single-objective test problem is minimized: maximizing it by mistake (the algorithms' default)
// would optimize the wrong way without a warning
fn minimized(name: &str, run: &config::Run) -> Result<()> {
    if run.objectives.contains(&config::Objective::Maximize) {
        return Err(format!(
            "{name} minimizes its objective: pass objective=\"minimize\" (the problem's objective)"
        ));
    }
    Ok(())
}

// a test problem runs with its objectives, minimized, and a genome of its type and dimensions
fn check_problem(problem: &problems::Problem, run: &config::Run) -> Result<()> {
    if let problems::Problem::Integer(problem) = problem {
        let name = problem.name();
        if run.objectives.len() != 1 {
            return Err(format!(
                "{name} has one objective: use a single-objective algorithm"
            ));
        }
        minimized(name, run)?;
        let dimensions = problem.integer().genome_len();
        return match &run.genome {
            config::Genome::Integer { bounds } if bounds.len() == dimensions => Ok(()),
            config::Genome::Integer { bounds } => Err(format!(
                "{name} has {dimensions} dimensions, but the genome has {} genes",
                bounds.len()
            )),
            _ => Err(format!("{name} needs an Integer genome")),
        };
    }
    let (name, objectives, dimensions, binary) = match problem {
        problems::Problem::Single(problem) => {
            (problem.name(), 1, problem.real().genome_len(), false)
        }
        // checked above
        problems::Problem::Integer(_) => return Ok(()),
        problems::Problem::Multi(config) => {
            let (name, dimensions, binary) = problems::name_and_dimensions(*config);
            (name, config.objectives(), dimensions, binary)
        }
    };
    match (objectives, run.objectives.len()) {
        (1, 1) => minimized(name, run)?,
        (1, _) => {
            return Err(format!(
                "{name} has one objective: use a single-objective algorithm"
            ));
        }
        (_, 1) => {
            return Err(format!(
                "{name} has {objectives} objectives: use a multi-objective algorithm"
            ));
        }
        (objectives, count) if objectives != count => {
            return Err(format!(
                "{name} has {objectives} objectives, but the algorithm has {count}"
            ));
        }
        _ => {
            if run.objectives.contains(&config::Objective::Maximize) {
                return Err(format!("{name} minimizes its objectives"));
            }
        }
    }
    match &run.genome {
        config::Genome::Binary { length } if binary && *length == dimensions => Ok(()),
        config::Genome::Binary { length } if binary => Err(format!(
            "{name} has {dimensions} bits, but the genome has {length}"
        )),
        _ if binary => Err(format!("{name} needs a Binary genome")),
        config::Genome::Real { bounds } | config::Genome::AdaptiveReal { bounds, .. }
            if bounds.len() == dimensions =>
        {
            Ok(())
        }
        config::Genome::Real { bounds } | config::Genome::AdaptiveReal { bounds, .. } => {
            Err(format!(
                "{name} has {dimensions} dimensions, but the genome has {} genes",
                bounds.len()
            ))
        }
        _ => Err(format!("{name} needs a Real genome")),
    }
}

// what a run needs besides the algorithm
pub(crate) struct Context {
    shared: Shared,
    pub objectives: Vec<Objective>,
    stop: config::Stop,
    parallel: bool,
    // a test problem, evaluated in Rust instead of the Python function
    problem: Option<problems::Problem>,
    // a network's weights balancing poles, evaluated in Rust instead of the Python function
    balance: Option<Balance>,
    // a fitness of trees, evaluated in Rust instead of the Python function
    pub tree: Option<TreeFitness>,
    // called with the running algorithm once per generation
    control: Option<Py<PyAny>>,
    checkpoints: Checkpoints,
}

impl Context {
    // the objective of a single-objective algorithm
    fn single_objective(&self) -> Result<Objective> {
        match self.objectives.as_slice() {
            [objective] => Ok(*objective),
            objectives => Err(format!(
                "this algorithm optimizes one objective, not {}; use Nsga2, Nsga3, Spea2, Moead or SmsEmoa for several",
                objectives.len()
            )),
        }
    }

    // the stop conditions
    fn stop(&self, single: bool) -> Result<Stop> {
        let config = &self.stop;
        let mut conditions = Vec::new();
        if let Some(generations) = config.generations {
            conditions.push(Stop::generations(generations));
        }
        if let Some(evaluations) = config.evaluations {
            conditions.push(Stop::evaluations(evaluations));
        }
        if let Some(target) = config.target {
            if !single {
                return Err("`target` needs a single objective".to_string());
            }
            if target.is_nan() {
                return Err("`target` is NaN".to_string());
            }
            conditions.push(Stop::target(target));
        }
        if let Some(seconds) = config.seconds {
            let time = Duration::try_from_secs_f64(seconds)
                .map_err(|error| format!("`time` of {seconds} seconds: {error}"))?;
            conditions.push(Stop::time(time));
        }
        if let Some(stagnation) = config.stagnation {
            if stagnation == 0 {
                return Err("`stagnation` must be at least 1 generation".to_string());
            }
            conditions.push(Stop::stagnation(stagnation));
        }
        let mut conditions = conditions.into_iter();
        let first = conditions.next().ok_or(
            "a run needs a stop condition: generations, evaluations, target, time or stagnation",
        )?;
        Ok(conditions.fold(first, Stop::or))
    }
}

// differential evolution, as `de` describes it
fn build_de(real: Real, de: config::De, context: &Context) -> std::result::Result<De, Failure> {
    let config::De {
        population_size,
        seed,
        l_shade,
        strategy,
        control,
        restarts,
        parallel_breeding,
    } = de;
    let mut builder = match l_shade {
        Some(evaluations) => De::l_shade(real, evaluations),
        None => De::builder(real),
    };
    if let Some(size) = population_size {
        builder = builder.population_size(size);
    }
    if let Some(seed) = seed {
        builder = builder.seed(seed);
    }
    if let Some(strategy) = strategy {
        builder = builder.strategy(de_strategy(strategy));
    }
    if let Some(control) = control {
        builder = builder.control(de_control(control));
    }
    if let Some(restarts) = restarts {
        builder = builder.restarts(de_restarts(restarts));
    }
    if let Some(parallel_breeding) = parallel_breeding {
        builder = builder.parallel_breeding(parallel_breeding);
    }
    let builder = builder.objective(context.single_objective()?);
    Ok(setting(builder.build())?)
}

// the island model of `islands`, with its settings
pub(crate) fn build_islands<A: Migrate>(
    islands: Vec<A>,
    topology: Option<config::Topology>,
    interval: Option<u64>,
    migrants: Option<usize>,
    seed: Option<u64>,
) -> Result<Islands<A>> {
    let mut builder = Islands::builder(islands);
    if let Some(topology) = topology {
        builder = builder.topology(match topology {
            config::Topology::Ring => Topology::Ring,
            config::Topology::FullyConnected => Topology::FullyConnected,
            config::Topology::Random => Topology::Random,
            config::Topology::Isolated => Topology::Isolated,
        });
    }
    if let Some(interval) = interval {
        builder = builder.interval(interval);
    }
    if let Some(migrants) = migrants {
        builder = builder.migrants(migrants);
    }
    if let Some(seed) = seed {
        builder = builder.seed(seed);
    }
    setting(builder.build())
}

pub fn de_strategy(strategy: config::DeStrategy) -> de::Strategy {
    match strategy {
        config::DeStrategy::Named(config::DeStrategyName::Rand1) => de::Strategy::Rand1,
        config::DeStrategy::Named(config::DeStrategyName::Best1) => de::Strategy::Best1,
        config::DeStrategy::CurrentToPBest(config::CurrentToPBest { p, archive }) => {
            de::Strategy::CurrentToPBest { p, archive }
        }
        config::DeStrategy::CurrentToPBestRandomP(config::CurrentToPBestRandomP {
            max_p,
            archive,
        }) => de::Strategy::CurrentToPBestRandomP { max_p, archive },
    }
}

pub fn de_control(control: config::DeControl) -> de::Control {
    match control {
        config::DeControl::Fixed(config::Fixed { f, cr }) => de::Control::Fixed { f, cr },
        config::DeControl::Dither(config::Dither { min_f, max_f, cr }) => {
            de::Control::Dither { min_f, max_f, cr }
        }
        config::DeControl::Jade(config::Jade { c }) => de::Control::Jade { c },
        config::DeControl::Shade(config::Shade { memory }) => de::Control::Shade { memory },
    }
}

fn nelder_mead_coefficients(coefficients: config::NelderMeadCoefficients) -> Coefficients {
    match coefficients {
        config::NelderMeadCoefficients::Named(config::NelderMeadCoefficientsName::Adaptive) => {
            Coefficients::Adaptive
        }
        config::NelderMeadCoefficients::Named(config::NelderMeadCoefficientsName::Standard) => {
            Coefficients::Standard
        }
        config::NelderMeadCoefficients::Custom(config::CustomCoefficients {
            reflection,
            expansion,
            contraction,
            shrink,
        }) => Coefficients::Custom {
            reflection,
            expansion,
            contraction,
            shrink,
        },
    }
}

fn de_restarts(restarts: config::DeRestarts) -> de::Restarts {
    match restarts {
        config::DeRestarts::Named(config::DeRestartsName::Never) => de::Restarts::Never,
        config::DeRestarts::OnStagnation(config::OnStagnation {
            tolerance,
            patience,
        }) => de::Restarts::OnStagnation {
            tolerance,
            patience,
        },
    }
}

// a NEAT run's genome is its networks, and its networks are NEAT's
const NEAT_GENOME: &str = "Neat evolves its own networks: it takes no genome";

// NEAT, on networks of `inputs` inputs and `outputs` outputs
fn neat_algorithm<'py>(
    py: Python<'py>,
    inputs: usize,
    outputs: usize,
    algorithm: config::Algorithm,
    context: &Context,
) -> Returns<'py> {
    let config::Algorithm::Neat(settings) = algorithm else {
        return Err("a network genome is Neat's: use gx.Neat".to_string().into());
    };
    let mut builder = Neat::builder(inputs, outputs);
    builder = match context.single_objective()? {
        Objective::Maximize => builder.maximize(),
        Objective::Minimize => builder.minimize(),
    };
    if let Some(size) = settings.population_size {
        builder = builder.population_size(size);
    }
    if let Some((c1, c2, c3, threshold)) = settings.compatibility {
        builder = builder.compatibility(c1, c2, c3, threshold);
    }
    if let Some((rate, replace)) = settings.weight_mutation {
        builder = builder.weight_mutation(rate, replace);
    }
    if let Some((perturbation, new)) = settings.weight_deviations {
        builder = builder.weight_deviations(perturbation, new);
    }
    if let Some((add_node, add_connection)) = settings.structural_mutation {
        builder = builder.structural_mutation(add_node, add_connection);
    }
    if let Some((mutation_only, interspecies, disable)) = settings.reproduction {
        builder = builder.reproduction(mutation_only, interspecies, disable);
    }
    if let Some((elitism_size, survival)) = settings.selection {
        builder = builder.selection(elitism_size, survival);
    }
    if let Some(generations) = settings.stagnation {
        builder = builder.stagnation(generations);
    }
    if let Some(activation) = settings.activation {
        builder = builder.activation(activation.into());
    }
    if let Some(feed_forward) = settings.feed_forward {
        builder = builder.feed_forward(feed_forward);
    }
    if let Some(initial) = settings.initial {
        builder = builder.initial(match initial {
            config::NeatInitial::FullyConnected => neat::Initial::FullyConnected,
            config::NeatInitial::Unconnected => neat::Initial::Unconnected,
        });
    }
    if let Some(sharing) = settings.sharing {
        builder = builder.sharing(match sharing {
            config::NeatSharing::Normalized => neat::Sharing::Normalized,
            config::NeatSharing::Raw => neat::Sharing::Raw,
        });
    }
    if let Some(seed) = settings.seed {
        builder = builder.seed(seed);
    }
    generational(py, setting(builder.build())?, NeatSettings, context)
}

// the algorithms only for real genomes, and the others
fn real_algorithm<'py>(
    py: Python<'py>,
    real: Result<Real>,
    algorithm: config::Algorithm,
    context: &Context,
) -> Returns<'py> {
    let real = real?;
    match algorithm {
        config::Algorithm::De(de) => {
            generational(py, build_de(real, de, context)?, DeSettings, context)
        }
        config::Algorithm::Es {
            parents,
            offspring,
            recombination,
            rho,
            selection,
            step_sizes,
            initial_step,
            parallel_breeding,
            seed,
        } => {
            let mut builder = Es::builder(real)
                .parents(parents)
                .offspring(offspring)
                .objective(context.single_objective()?);
            if recombination.is_some() || rho.is_some() {
                // all the parents by default
                let rho = rho.unwrap_or(parents);
                builder = builder.recombination(match recombination {
                    Some(config::Recombination::Dominant) => es::Recombination::Dominant { rho },
                    _ => es::Recombination::Intermediate { rho },
                });
            }
            if let Some(selection) = selection {
                builder = builder.selection(match selection {
                    config::EsSelection::Comma => es::Selection::Comma,
                    config::EsSelection::Plus => es::Selection::Plus,
                });
            }
            if let Some(step_sizes) = step_sizes {
                builder = builder.step_sizes(match step_sizes {
                    config::StepSizes::One => es::StepSizes::One,
                    config::StepSizes::PerGene => es::StepSizes::PerGene,
                });
            }
            if let Some(step) = initial_step {
                builder = builder.initial_step(step);
            }
            if let Some(parallel_breeding) = parallel_breeding {
                builder = builder.parallel_breeding(parallel_breeding);
            }
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            generational(py, setting(builder.build())?, EsSettings, context)
        }
        config::Algorithm::Islands {
            islands,
            topology,
            interval,
            migrants,
            seed,
        } if islands
            .iter()
            .all(|island| matches!(island, config::Algorithm::De(_))) =>
        {
            let islands = islands
                .into_iter()
                .map(|island| match island {
                    config::Algorithm::De(de) => build_de(real.clone(), de, context),
                    _ => unreachable!("checked above: all De"),
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let islands = build_islands(islands, topology, interval, migrants, seed)?;
            generational(py, islands, IslandsSettings(DeSettings), context)
        }
        config::Algorithm::Cmaes {
            population_size,
            seed,
            restarts,
            initial_step,
            covariance,
            min_step,
        } => {
            let mut builder = Cmaes::builder(real).objective(context.single_objective()?);
            if let Some(size) = population_size {
                builder = builder.population_size(size);
            }
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            if let Some(restarts) = restarts {
                builder = builder.restarts(match restarts {
                    config::Restarts::Never => cmaes::Restarts::Never,
                    config::Restarts::Ipop => cmaes::Restarts::Ipop,
                    config::Restarts::Bipop => cmaes::Restarts::Bipop,
                });
            }
            if let Some(step) = initial_step {
                builder = builder.initial_step(step);
            }
            if let Some(covariance) = covariance {
                builder = builder.covariance(match covariance {
                    config::Covariance::Full => cmaes::Covariance::Full,
                    config::Covariance::Diagonal => cmaes::Covariance::Diagonal,
                });
            }
            if let Some(fraction) = min_step {
                builder = builder.min_step(fraction);
            }
            generational(py, setting(builder.build())?, CmaesSettings, context)
        }
        config::Algorithm::Pso {
            population_size,
            seed,
            ring,
        } => {
            let mut builder = Pso::builder(real).objective(context.single_objective()?);
            if let Some(size) = population_size {
                builder = builder.population_size(size);
            }
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            if let Some(neighbors) = ring {
                builder = builder.topology(pso::Topology::Ring { neighbors });
            }
            generational(py, setting(builder.build())?, PsoSettings, context)
        }
        config::Algorithm::OpenEs {
            population_size,
            sigma,
            optimizer,
            weight_decay,
            evaluate_mean,
            initial_mean,
            parallel_breeding,
            seed,
        } => {
            let open_es = build_open_es(
                real,
                population_size,
                sigma,
                optimizer,
                weight_decay,
                evaluate_mean,
                initial_mean,
                parallel_breeding,
                seed,
                context.single_objective()?,
            )?;
            generational(py, open_es, OpenEsSettings, context)
        }
        config::Algorithm::NelderMead {
            coefficients,
            initial_step,
            initial_step_absolute,
            tolerance,
            restarts,
            speculative,
            initial_genome,
            seed,
        } => {
            let mut builder = NelderMead::builder(real).objective(context.single_objective()?);
            if let Some(coefficients) = coefficients {
                builder = builder.coefficients(nelder_mead_coefficients(coefficients));
            }
            if let Some(step) = initial_step {
                builder = builder.initial_step(step);
            }
            if let Some(distance) = initial_step_absolute {
                builder = builder.initial_step_absolute(distance);
            }
            if let Some(tolerance) = tolerance {
                builder = builder.tolerance(tolerance);
            }
            if let Some(times) = restarts {
                builder = builder.restarts(local::Restarts::Random { times });
            }
            if let Some(speculative) = speculative {
                builder = builder.speculative(speculative);
            }
            if let Some(genome) = initial_genome {
                builder = builder.initial_genome(Reals::from(genome));
            }
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            // the only genome the builder checks is the initial one
            let nelder_mead = setting(builder.build().map_err(|error| match error {
                genoxide::Error::InvalidGenome { reason } => genoxide::Error::InvalidSetting {
                    setting: "initial_genome",
                    reason,
                },
                error => error,
            }))?;
            generational(py, nelder_mead, NelderMeadSettings, context)
        }
        algorithm => with_operators(
            py,
            Ok(real),
            algorithm,
            context,
            RealCrossover::new,
            RealMutation::new,
        ),
    }
}

// OpenAI's evolution strategy, with its settings
#[allow(clippy::too_many_arguments)]
fn build_open_es(
    real: Real,
    population_size: usize,
    sigma: Option<f64>,
    optimizer: Option<config::Optimizer>,
    weight_decay: Option<f64>,
    evaluate_mean: Option<bool>,
    initial_mean: Option<Vec<f64>>,
    parallel_breeding: Option<bool>,
    seed: Option<u64>,
    objective: Objective,
) -> Result<OpenEs> {
    let mut builder = OpenEs::builder(real)
        .population_size(population_size)
        .objective(objective);
    if let Some(sigma) = sigma {
        builder = builder.sigma(sigma);
    }
    if let Some(optimizer) = optimizer {
        builder = builder.optimizer(match optimizer {
            config::Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
            } => open_es::Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
            },
            config::Optimizer::Sgd {
                learning_rate,
                momentum,
            } => open_es::Optimizer::sgd(learning_rate, momentum),
        });
    }
    if let Some(decay) = weight_decay {
        builder = builder.weight_decay(decay);
    }
    if let Some(evaluate) = evaluate_mean {
        builder = builder.evaluate_mean(evaluate);
    }
    if let Some(mean) = initial_mean {
        builder = builder.initial_mean(Reals::from(mean));
    }
    if let Some(parallel_breeding) = parallel_breeding {
        builder = builder.parallel_breeding(parallel_breeding);
    }
    if let Some(seed) = seed {
        builder = builder.seed(seed);
    }
    let open_es = builder.build();
    // the initial mean, outside the genome's bounds or of another length
    if let Err(genoxide::Error::InvalidGenome { reason }) = &open_es {
        return Err(format!("invalid setting `initial_mean`: {reason}"));
    }
    setting(open_es)
}

// the algorithms for any genome, with its operators
fn with_operators<'py, R, C, M>(
    py: Python<'py>,
    representation: Result<R>,
    algorithm: config::Algorithm,
    context: &Context,
    crossover: fn(config::Crossover) -> Result<C>,
    mutate: fn(config::Mutate) -> Result<M>,
) -> Returns<'py>
where
    R: Representation + Clone + PartialEq + Send + Serialize + DeserializeOwned + 'static,
    R::Genome: PyGenome + Serialize + DeserializeOwned,
    C: Crossover<R> + Clone + Send + Serialize + DeserializeOwned + 'static,
    M: Mutate<R> + Clone + Send + Serialize + DeserializeOwned + 'static,
{
    let representation = representation?;
    match algorithm {
        config::Algorithm::Ga(ga) => {
            let builder = ga_builder(representation, &ga, context, crossover, mutate)?;
            let settings = GaSettings { crossover, mutate };
            generational(py, setting(builder.build())?, settings, context)
        }
        config::Algorithm::Islands {
            islands,
            topology,
            interval,
            migrants,
            seed,
        } => {
            let islands = islands
                .into_iter()
                .map(|island| match island {
                    config::Algorithm::Ga(ga) => {
                        let builder =
                            ga_builder(representation.clone(), &ga, context, crossover, mutate)?;
                        Ok(setting(builder.build())?)
                    }
                    config::Algorithm::De(_) => {
                        Err(Failure::from("De needs a Real genome".to_string()))
                    }
                    _ => Err(Failure::from(
                        "the islands are all Ga or all De".to_string(),
                    )),
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let islands = build_islands(islands, topology, interval, migrants, seed)?;
            let settings = IslandsSettings(GaSettings { crossover, mutate });
            generational(py, islands, settings, context)
        }
        config::Algorithm::LocalSearch {
            seed,
            neighbor,
            neighbors,
            acceptance,
            restart,
        } => {
            let mut builder = LocalSearch::builder(representation)
                .neighbor(mutate(neighbor)?)
                .objective(context.single_objective()?);
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            if let Some(neighbors) = neighbors {
                builder = builder.neighbors(neighbors);
            }
            if let Some(acceptance) = acceptance {
                builder = builder.acceptance(match acceptance {
                    config::Acceptance::Improving {} => Acceptance::Improving,
                    config::Acceptance::NotWorse {} => Acceptance::NotWorse,
                    config::Acceptance::Annealing {
                        initial_temperature,
                        cooling,
                    } => Acceptance::Annealing {
                        initial_temperature,
                        cooling,
                    },
                    config::Acceptance::Tabu { tenure } => Acceptance::Tabu { tenure },
                });
            }
            if let Some((patience, kicks)) = restart {
                builder = builder.restart(patience, kicks);
            }
            let settings = LocalSearchSettings { neighbor: mutate };
            generational(py, setting(builder.build())?, settings, context)
        }
        config::Algorithm::Nsga2 { ref variation, .. }
        | config::Algorithm::Nsga3 { ref variation, .. }
        | config::Algorithm::Spea2 { ref variation, .. }
        | config::Algorithm::Moead { ref variation, .. }
        | config::Algorithm::SmsEmoa { ref variation, .. } => {
            let multi = MultiObjective {
                py,
                representation,
                crossover: crossover(variation.crossover)?,
                mutate: mutate(variation.mutate.clone())?,
                algorithm,
                context,
            };
            with_objectives(context.objectives.len(), multi).unwrap_or_else(|count| {
                let message =
                    format!("multi-objective algorithms take 2 to 6 objectives, not {count}");
                Err(message.into())
            })
        }
        config::Algorithm::De(_) => Err("De needs a Real genome".to_string().into()),
        config::Algorithm::Es { .. } => Err("Es needs a Real genome".to_string().into()),
        config::Algorithm::Cmaes { .. } => Err("Cmaes needs a Real genome".to_string().into()),
        config::Algorithm::Pso { .. } => Err("Pso needs a Real genome".to_string().into()),
        config::Algorithm::OpenEs { .. } => Err("OpenEs needs a Real genome".to_string().into()),
        config::Algorithm::Neat(_) => Err(NEAT_GENOME.to_string().into()),
        config::Algorithm::NelderMead { .. } => {
            Err("NelderMead needs a Real genome".to_string().into())
        }
    }
}

// initial genomes are trees: other genomes start from random ones
const INITIAL_GENOMES: &str = "initial_genomes are the trees of a gx.gp.Gp genome";

pub(crate) fn ga_builder<R, C, M>(
    representation: R,
    ga: &config::Ga,
    context: &Context,
    crossover: fn(config::Crossover) -> Result<C>,
    mutate: fn(config::Mutate) -> Result<M>,
) -> Result<GaBuilder<R, AnySelect, C, M>>
where
    R: Representation,
{
    if ga.initial_genomes.is_some() {
        return Err(INITIAL_GENOMES.to_string());
    }
    let mut builder = Ga::builder(representation)
        .population_size(ga.population_size)
        .select(AnySelect::new(ga.select.clone())?)
        .crossover(crossover(ga.crossover)?)
        .mutate(mutate(ga.mutate.clone())?)
        .objective(context.single_objective()?);
    if let Some(seed) = ga.seed {
        builder = builder.seed(seed);
    }
    if let Some(rate) = ga.crossover_rate {
        builder = builder.crossover_rate(rate);
    }
    if let Some(rate) = ga.mutation_rate {
        builder = builder.mutation_rate(rate);
    }
    if let Some(parallel_breeding) = ga.parallel_breeding {
        builder = builder.parallel_breeding(parallel_breeding);
    }
    if let Some(scheme) = ga.scheme {
        builder = builder.scheme(match scheme {
            config::Scheme::Generational { elitism } => Scheme::Generational { elitism },
            config::Scheme::SteadyState { replacements } => Scheme::SteadyState { replacements },
            config::Scheme::MuPlusLambda { lambda } => Scheme::MuPlusLambda { lambda },
            config::Scheme::MuCommaLambda { lambda } => Scheme::MuCommaLambda { lambda },
        });
    }
    Ok(builder)
}

/// Something to do with the number of objectives as a constant.
pub trait WithObjectives {
    type Output;

    fn with<const N: usize>(self) -> Self::Output;
}

/// Does `task` with `count` objectives, 2 to 6, or returns `Err(count)`.
pub fn with_objectives<T: WithObjectives>(
    count: usize,
    task: T,
) -> std::result::Result<T::Output, usize> {
    match count {
        2 => Ok(task.with::<2>()),
        3 => Ok(task.with::<3>()),
        4 => Ok(task.with::<4>()),
        5 => Ok(task.with::<5>()),
        6 => Ok(task.with::<6>()),
        count => Err(count),
    }
}

/// Das-Dennis points for `objectives` objectives, 2 to 6, with `divisions` divisions: the
/// reference directions of NSGA-III and the weights of MOEA/D, a point per row.
#[pyfunction]
pub fn das_dennis(
    py: Python<'_>,
    objectives: usize,
    divisions: usize,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    if !(2..=6).contains(&objectives) {
        return Err(PyValueError::new_err(format!(
            "das_dennis takes 2 to 6 objectives, not {objectives}"
        )));
    }
    // checked before allocating: a failed allocation would abort the interpreter
    let count = das_dennis_count(objectives, divisions);
    if count > MAX_POINTS {
        return Err(PyValueError::new_err(format!(
            "das_dennis({objectives}, {divisions}) would have {count} points, more than 2^24"
        )));
    }
    let points = with_objectives(objectives, DasDennis(divisions)).map_err(|count| {
        PyValueError::new_err(format!("das_dennis takes 2 to 6 objectives, not {count}"))
    })?;
    Ok(points.into_pyarray(py))
}

/// The most points that `das_dennis` and `optimal_front` return, as genoxide's sizes.
pub const MAX_POINTS: u128 = 1 << 24;

// the number of Das-Dennis points, C(divisions + objectives − 1, objectives − 1); 0 for 0
// divisions, as `das_dennis` returns none
fn das_dennis_count(objectives: usize, divisions: usize) -> u128 {
    if divisions == 0 {
        return 0;
    }
    let (n, k) = (
        (divisions + objectives - 1) as u128,
        (objectives - 1) as u128,
    );
    // exact at each step: C(n, i + 1) = C(n, i) (n − i) / (i + 1); saturates far above the limit
    (0..k).fold(1u128, |count, i| count.saturating_mul(n - i) / (i + 1))
}

// Das-Dennis points with this many divisions
struct DasDennis(usize);

impl WithObjectives for DasDennis {
    type Output = Array2<f64>;

    fn with<const N: usize>(self) -> Array2<f64> {
        let points = multi::das_dennis::<N>(self.0);
        Array2::from_shape_fn((points.len(), N), |(point, objective)| {
            points[point][objective]
        })
    }
}

// a multi-objective algorithm to build and run, with its operators
struct MultiObjective<'a, 'py, R, C, X> {
    py: Python<'py>,
    representation: R,
    crossover: C,
    mutate: X,
    algorithm: config::Algorithm,
    context: &'a Context,
}

// the settings every multi-objective builder has
macro_rules! variation {
    ($builder:expr, $crossover:expr, $mutate:expr, $variation:expr, $seed:expr) => {{
        let mut builder = $builder.crossover($crossover).mutate($mutate);
        if let Some(rate) = $variation.crossover_rate {
            builder = builder.crossover_rate(rate);
        }
        if let Some(rate) = $variation.mutation_rate {
            builder = builder.mutation_rate(rate);
        }
        if let Some(seed) = $seed {
            builder = builder.seed(seed);
        }
        builder
    }};
}

// duplicate elimination, for the algorithms that have it
macro_rules! duplicates {
    ($builder:expr, $variation:expr) => {{
        let mut builder = $builder;
        if let Some(eliminate) = $variation.eliminate_duplicates {
            builder = builder.eliminate_duplicates(eliminate);
        }
        builder
    }};
}

impl<'py, R, C, X> WithObjectives for MultiObjective<'_, 'py, R, C, X>
where
    R: Representation + Clone + Serialize + DeserializeOwned,
    R::Genome: PyGenome + Serialize + DeserializeOwned,
    C: Crossover<R> + Clone + Serialize + DeserializeOwned,
    X: Mutate<R> + Clone + Serialize + DeserializeOwned,
{
    type Output = Returns<'py>;

    fn with<const N: usize>(self) -> Returns<'py> {
        let MultiObjective {
            py,
            representation,
            crossover,
            mutate,
            algorithm,
            context,
        } = self;
        let objectives: [Objective; N] = context
            .objectives
            .clone()
            .try_into()
            .map_err(|_| "the number of objectives changed".to_string())?;
        match algorithm {
            config::Algorithm::Nsga2 {
                population_size,
                seed,
                variation,
                initial_genomes,
            } => {
                if initial_genomes.is_some() {
                    return Err(INITIAL_GENOMES.to_string().into());
                }
                let builder =
                    Nsga2::builder(representation, objectives).population_size(population_size);
                let builder = variation!(builder, crossover, mutate, variation, seed);
                let builder = duplicates!(builder, variation);
                multi_objective(py, setting(builder.build())?, context)
            }
            config::Algorithm::Nsga3 {
                reference_directions,
                population_size,
                seed,
                variation,
            } => {
                let directions = rows(reference_directions, "reference_directions")?;
                let mut builder = Nsga3::builder(representation, objectives, directions);
                if let Some(size) = population_size {
                    builder = builder.population_size(size);
                }
                let builder = variation!(builder, crossover, mutate, variation, seed);
                let builder = duplicates!(builder, variation);
                multi_objective(py, setting(builder.build())?, context)
            }
            config::Algorithm::Spea2 {
                population_size,
                seed,
                variation,
            } => {
                let builder =
                    Spea2::builder(representation, objectives).population_size(population_size);
                let builder = variation!(builder, crossover, mutate, variation, seed);
                let builder = duplicates!(builder, variation);
                multi_objective(py, setting(builder.build())?, context)
            }
            config::Algorithm::Moead {
                weights,
                neighbors,
                neighbor_mating,
                max_replacements,
                decomposition,
                seed,
                variation,
            } => {
                let mut builder =
                    Moead::builder(representation, objectives, rows(weights, "weights")?);
                if let Some(neighbors) = neighbors {
                    builder = builder.neighbors(neighbors);
                }
                if let Some(probability) = neighbor_mating {
                    builder = builder.neighbor_mating(probability);
                }
                if let Some(count) = max_replacements {
                    builder = builder.max_replacements(count);
                }
                if let Some(decomposition) = decomposition {
                    builder = builder.decomposition(match decomposition {
                        config::Decomposition::Tchebycheff {} => Decomposition::Tchebycheff,
                        config::Decomposition::Pbi { theta } => Decomposition::Pbi { theta },
                    });
                }
                if variation.eliminate_duplicates.is_some() {
                    return Err(
                        "Moead has no eliminate_duplicates: it replaces its neighbors \
                                one child at a time"
                            .to_string()
                            .into(),
                    );
                }
                let builder = variation!(builder, crossover, mutate, variation, seed);
                multi_objective(py, setting(builder.build())?, context)
            }
            config::Algorithm::SmsEmoa {
                population_size,
                offspring,
                seed,
                variation,
            } => {
                let mut builder =
                    SmsEmoa::builder(representation, objectives).population_size(population_size);
                if let Some(count) = offspring {
                    builder = builder.offspring(count);
                }
                let builder = variation!(builder, crossover, mutate, variation, seed);
                let builder = duplicates!(builder, variation);
                multi_objective(py, setting(builder.build())?, context)
            }
            _ => Err("not a multi-objective algorithm".to_string().into()),
        }
    }
}

// reference directions or weight vectors: a row each, with a value per objective
fn rows<const N: usize>(rows: Vec<Vec<f64>>, setting: &str) -> Result<Vec<[f64; N]>> {
    rows.into_iter()
        .map(|row| {
            <[f64; N]>::try_from(row).map_err(|row| {
                format!(
                    "each row of `{setting}` has a value per objective, {N}, not {}",
                    row.len()
                )
            })
        })
        .collect()
}

// runs a single-objective algorithm, detached from Python so that other threads, and the fitness
// function on rayon's threads, can run; with the control of the run, if any, which can change
// `settings` of the algorithm
pub(crate) fn generational<'py, A, S>(
    py: Python<'py>,
    algorithm: A,
    settings: S,
    context: &Context,
) -> Returns<'py>
where
    A: Algorithm + Reevaluate + Clone + Send + Serialize + DeserializeOwned + 'static,
    A::Genome: PyGenome,
    S: Settings<A>,
{
    let stop = context.stop(true)?;
    let algorithm = context.checkpoints.resume(algorithm)?;
    let shared = &context.shared;
    let cx = shared.context();
    let parallel = context.parallel;
    let problem = match &context.problem {
        Some(problems::Problem::Single(problem)) => Some(Native::Real(problem.as_ref())),
        Some(problems::Problem::Integer(problem)) => Some(Native::Integer(problem.as_ref())),
        _ => match &context.tree {
            Some(tree) => Some(Native::Tree(tree)),
            None => context.balance.as_ref().map(Native::Balance),
        },
    };
    let fitness = Single { shared, problem };
    // the control, the handle it gets, and the slot that holds the algorithm during its call
    let control = match &context.control {
        Some(callback) => {
            let slot = Arc::new(Slot::new(settings));
            let handle = Py::new(py, Running::new(Arc::clone(&slot)))?;
            // takes the algorithm's place in the engine during the call
            let spare = algorithm.clone();
            Some((callback, slot, handle, spare))
        }
        None => None,
    };
    let outcome = py.detach(|| {
        let mut engine = Engine::new(algorithm, fitness)
            .stop_when(stop)
            .abort_flag(shared.abort_flag())
            .parallel(parallel)
            .on_generation(|snapshot| {
                shared.after_generation(snapshot.progress(), |py| {
                    single_state(
                        py,
                        snapshot.population(),
                        snapshot.best(),
                        snapshot.progress(),
                        cx,
                    )
                });
            });
        if let Some((callback, slot, handle, spare)) = control {
            let mut spare = Some(spare);
            engine = engine.control(move |algorithm, progress| {
                let called = shared.control(progress, |py, arguments| {
                    let best = algorithm.best().expect("a best individual after a tell");
                    let state = single_state(py, algorithm.population(), best, progress, cx)?;
                    // the algorithm moves to the slot for the call, and back
                    let filler = spare.take().expect("the spare is back after each call");
                    slot.fill(std::mem::replace(algorithm, filler));
                    let mut all = vec![handle.bind(py).clone().into_any()];
                    all.extend(arguments);
                    all.extend(state);
                    let result = callback.bind(py).call1(pyo3::types::PyTuple::new(py, all)?);
                    let moved = slot.empty().expect("the algorithm stays in its slot");
                    spare = Some(std::mem::replace(algorithm, moved));
                    result.map(drop)
                });
                if called {
                    Ok(())
                } else {
                    // the run raises the control's exception; this error only stops the engine
                    Err(genoxide::Error::InvalidSetting {
                        setting: "control",
                        reason: "the control raised an exception".to_string(),
                    })
                }
            });
        }
        if let Some((path, every)) = &context.checkpoints.save {
            engine = engine.checkpoint_every(*every, |algorithm| save(context, algorithm, path));
        }
        engine.run()
    });
    if let Some(error) = shared.take_error() {
        return Err(error.into());
    }
    let outcome = outcome.map_err(engine_error)?;
    let fitness = outcome.best_fitness();
    let result = PyDict::new(py);
    result.set_item("best_genome", outcome.best_genome().object(py, cx)?)?;
    result.set_item("best_fitness", fitness.score())?;
    // no violation without a valid solution: NaN, as 0 means feasible
    let violation = fitness.score().map_or(f64::NAN, |_| fitness.violation());
    result.set_item("violation", violation)?;
    result.set_item("generations", outcome.generations())?;
    result.set_item("evaluations", outcome.evaluations())?;
    result.set_item("seconds", outcome.elapsed().as_secs_f64())?;
    result.set_item("stop_reason", stop_reason(outcome.stop_reason()))?;
    Ok(result)
}

// runs a multi-objective algorithm, detached from Python like `generational`
pub(crate) fn multi_objective<'py, A, const N: usize>(
    py: Python<'py>,
    algorithm: A,
    context: &Context,
) -> Returns<'py>
where
    A: MultiObjectiveAlgorithm<N> + Clone + Send + Serialize + DeserializeOwned,
    A::Genome: PyGenome,
{
    let stop = context.stop(false)?;
    let algorithm = context.checkpoints.resume(algorithm)?;
    let shared = &context.shared;
    let cx = shared.context();
    let parallel = context.parallel;
    let problem = match &context.problem {
        Some(problems::Problem::Multi(config)) => Some(config.build::<N>()),
        _ => None,
    };
    let fitness = Multi {
        shared,
        problem: problem.as_ref(),
        tree: context.tree.as_ref(),
    };
    let outcome = py.detach(|| {
        let mut engine = MultiEngine::new(algorithm, fitness)
            .stop_when(stop)
            .abort_flag(shared.abort_flag())
            .parallel(parallel)
            .on_generation(|snapshot| {
                shared.after_generation(snapshot.progress(), |py| multi_state(py, snapshot, cx));
            });
        if let Some((path, every)) = &context.checkpoints.save {
            engine = engine.checkpoint_every(*every, |algorithm| save(context, algorithm, path));
        }
        engine.run()
    });
    if let Some(error) = shared.take_error() {
        return Err(error.into());
    }
    let outcome = outcome.map_err(engine_error)?;
    // the front, each genome once, as in the last generation's progress
    let front = outcome.front();
    let genomes: Vec<&A::Genome> = front.iter().map(Individual::genome).collect();
    let (objectives, violations) = objective_rows(py, front)?;
    let result = PyDict::new(py);
    result.set_item("front_genomes", PyGenome::batch(py, &genomes, cx)?)?;
    result.set_item("front_objectives", objectives)?;
    result.set_item("front_violations", violations)?;
    result.set_item("generations", outcome.generations())?;
    result.set_item("evaluations", outcome.evaluations())?;
    result.set_item("seconds", outcome.elapsed().as_secs_f64())?;
    result.set_item("stop_reason", stop_reason(outcome.stop_reason()))?;
    Ok(result)
}

// saves a checkpoint, unless an exception stopped the run: its generation's genomes left got an
// invalid fitness without a call, and the last good checkpoint stays
fn save<A: Clone + Serialize>(
    context: &Context,
    algorithm: &A,
    path: &Path,
) -> genoxide::Result<()> {
    if context.shared.failed() {
        return Ok(());
    }
    context.checkpoints.save(algorithm, path)
}

// the error that stopped an engine: an OSError for a checkpoint that couldn't be saved
fn engine_error(error: genoxide::Error) -> Failure {
    match error {
        genoxide::Error::Checkpoint { .. } => {
            Failure::Python(PyOSError::new_err(error.to_string()))
        }
        error => Failure::Setting(error.to_string()),
    }
}

// the arguments of the progress callback after a single-objective generation: the best score
// and genome so far, and the population, made into arrays when Python reads them
fn single_state<'py, G: PyGenome>(
    py: Python<'py>,
    population: &Population<G>,
    best: &Individual<G>,
    progress: &Progress,
    cx: &GenomeContext,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let score = progress.best().and_then(Fitness::score);
    Ok(vec![
        score.into_pyobject(py)?.into_any(),
        best.genome().object(py, cx)?,
        Bound::new(py, Snapshot::single(population.iter(), cx))?.into_any(),
    ])
}

// the arguments of the progress callback after a multi-objective generation: the size of the
// front, and the population and the front, made into arrays when Python reads them
fn multi_state<'py, G: PyGenome, const N: usize>(
    py: Python<'py>,
    snapshot: &MultiSnapshot<'_, G, N>,
    cx: &GenomeContext,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    Ok(vec![
        snapshot.front().len().into_pyobject(py)?.into_any(),
        Bound::new(py, Snapshot::multi(snapshot.population().iter(), true, cx))?.into_any(),
        Bound::new(py, Snapshot::multi(snapshot.front().iter(), false, cx))?.into_any(),
    ])
}

// objective values, a row per individual, and constraint violations
type ObjectiveRows<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);

// the objective values of individuals, a row each, and their constraint violations; NaN for an
// invalid solution
fn objective_rows<'a, 'py, G: Genome + 'a, const N: usize>(
    py: Python<'py>,
    individuals: impl IntoIterator<Item = &'a Individual<G, Scores<N>>>,
) -> PyResult<ObjectiveRows<'py>> {
    let (objectives, violations) = objective_values(individuals);
    let objectives = Array2::from_shape_vec((violations.len(), N), objectives)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py);
    Ok((objectives, PyArray1::from_vec(py, violations)))
}

fn stop_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Target => "target",
        StopReason::Generations => "generations",
        StopReason::Evaluations => "evaluations",
        StopReason::Time => "time",
        StopReason::Stagnation => "stagnation",
        StopReason::Aborted => "aborted",
        StopReason::Stalled => "stalled",
        StopReason::Converged => "converged",
        _ => "other",
    }
}
