//! Building the algorithm a run describes, and running it with a Python fitness function.

use crate::config;
use crate::fitness::{Multi, Shared, Single};
use crate::genes::{self, Genes};
use crate::operators::{
    AnySelect, ListCrossover, OrderCrossovers, OrderMutation, RealCrossover, RealMutation,
    bit_flip, integer_mutation,
};
use genoxide::algorithm::{GaBuilder, cmaes, pso};
use genoxide::genome::Representation;
use genoxide::multi::MultiObjectiveAlgorithm;
use genoxide::operator::{Crossover, Mutate};
use genoxide::prelude::*;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::Duration;

type Result<T> = std::result::Result<T, String>;

fn setting<T>(result: genoxide::Result<T>) -> Result<T> {
    result.map_err(|error| error.to_string())
}

/// Runs the optimization that `config` (JSON, from the Python package) describes, with
/// `fitness`, called with a genome or, with `batch`, a generation of genomes; with `parallel`,
/// from several threads at once. Returns the result as a dict.
#[pyfunction]
#[pyo3(signature = (config, fitness, batch = false, parallel = false))]
pub fn run<'py>(
    py: Python<'py>,
    config: &str,
    fitness: Py<PyAny>,
    batch: bool,
    parallel: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let run: config::Run = serde_json::from_str(config)
        .map_err(|error| PyValueError::new_err(format!("invalid run description: {error}")))?;
    let context = Context {
        shared: Shared::new(fitness, batch),
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
    };
    let result = match run.genome {
        config::Genome::Binary { length } => with_operators(
            py,
            setting(Binary::new(length)),
            run.algorithm,
            &context,
            |crossover| ListCrossover::new(crossover, "binary"),
            bit_flip,
        ),
        config::Genome::Integer { bounds } => with_operators(
            py,
            setting(Integer::new(bounds.iter().map(|&(low, high)| low..=high))),
            run.algorithm,
            &context,
            |crossover| ListCrossover::new(crossover, "integer"),
            integer_mutation,
        ),
        config::Genome::Real { bounds } => {
            let real = setting(Real::new(bounds.iter().map(|&(low, high)| low..=high)));
            real_algorithm(py, real, run.algorithm, &context)
        }
        config::Genome::Permutation { length } => with_operators(
            py,
            setting(Permutation::new(length)),
            run.algorithm,
            &context,
            OrderCrossovers::new,
            OrderMutation::new,
        ),
    };
    result.map_err(|error| match error {
        Failure::Setting(message) => PyValueError::new_err(message),
        Failure::Python(error) => error,
    })
}

// why a run failed: its description, or Python (the fitness function, Ctrl+C)
enum Failure {
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

type Returns<'py> = std::result::Result<Bound<'py, PyDict>, Failure>;

// what a run needs besides the algorithm
struct Context {
    shared: Shared,
    objectives: Vec<Objective>,
    stop: config::Stop,
    parallel: bool,
}

impl Context {
    // the objective of a single-objective algorithm
    fn single_objective(&self) -> Result<Objective> {
        match self.objectives.as_slice() {
            [objective] => Ok(*objective),
            objectives => Err(format!(
                "this algorithm optimizes one objective, not {}; use Nsga2 for several",
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

// the algorithms only for real genomes, and the others
fn real_algorithm<'py>(
    py: Python<'py>,
    real: Result<Real>,
    algorithm: config::Algorithm,
    context: &Context,
) -> Returns<'py> {
    let real = real?;
    match algorithm {
        config::Algorithm::De {
            population_size,
            seed,
            l_shade,
        } => {
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
            let builder = builder.objective(context.single_objective()?);
            generational(py, setting(builder.build())?, context)
        }
        config::Algorithm::Cmaes {
            population_size,
            seed,
            restarts,
            initial_step,
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
            generational(py, setting(builder.build())?, context)
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
            generational(py, setting(builder.build())?, context)
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

// the algorithms for any genome, with its operators
fn with_operators<'py, R, C, M>(
    py: Python<'py>,
    representation: Result<R>,
    algorithm: config::Algorithm,
    context: &Context,
    crossover: impl Fn(config::Crossover) -> Result<C>,
    mutate: impl Fn(config::Mutate) -> Result<M>,
) -> Returns<'py>
where
    R: Representation,
    R::Genome: Genes,
    C: Crossover<R>,
    M: Mutate<R>,
{
    let representation = representation?;
    match algorithm {
        config::Algorithm::Ga(ga) => {
            let builder = ga_builder(representation, &ga, context, &crossover, &mutate)?;
            generational(py, setting(builder.build())?, context)
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
            generational(py, setting(builder.build())?, context)
        }
        config::Algorithm::Nsga2 {
            population_size,
            seed,
            crossover: crossover_setting,
            mutate: mutate_setting,
            crossover_rate,
        } => {
            let settings = Nsga2Settings {
                population_size,
                seed,
                crossover_rate,
            };
            let crossover = crossover(crossover_setting)?;
            let mutate = mutate(mutate_setting)?;
            match context.objectives.len() {
                2 => nsga2::<_, _, _, 2>(py, representation, crossover, mutate, &settings, context),
                3 => nsga2::<_, _, _, 3>(py, representation, crossover, mutate, &settings, context),
                4 => nsga2::<_, _, _, 4>(py, representation, crossover, mutate, &settings, context),
                5 => nsga2::<_, _, _, 5>(py, representation, crossover, mutate, &settings, context),
                6 => nsga2::<_, _, _, 6>(py, representation, crossover, mutate, &settings, context),
                count => Err(format!("Nsga2 takes 2 to 6 objectives, not {count}").into()),
            }
        }
        config::Algorithm::De { .. } => Err("De needs a Real genome".to_string().into()),
        config::Algorithm::Cmaes { .. } => Err("Cmaes needs a Real genome".to_string().into()),
        config::Algorithm::Pso { .. } => Err("Pso needs a Real genome".to_string().into()),
    }
}

fn ga_builder<R, C, M>(
    representation: R,
    ga: &config::Ga,
    context: &Context,
    crossover: &impl Fn(config::Crossover) -> Result<C>,
    mutate: &impl Fn(config::Mutate) -> Result<M>,
) -> Result<GaBuilder<R, AnySelect, C, M>>
where
    R: Representation,
{
    let mut builder = Ga::builder(representation)
        .population_size(ga.population_size)
        .select(AnySelect::new(ga.select)?)
        .crossover(crossover(ga.crossover)?)
        .mutate(mutate(ga.mutate)?)
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

struct Nsga2Settings {
    population_size: usize,
    seed: Option<u64>,
    crossover_rate: Option<f64>,
}

fn nsga2<'py, R, C, X, const N: usize>(
    py: Python<'py>,
    representation: R,
    crossover: C,
    mutate: X,
    settings: &Nsga2Settings,
    context: &Context,
) -> Returns<'py>
where
    R: Representation,
    R::Genome: Genes,
    C: Crossover<R>,
    X: Mutate<R>,
{
    let objectives: [Objective; N] = context
        .objectives
        .clone()
        .try_into()
        .map_err(|_| "the number of objectives changed".to_string())?;
    let mut builder = Nsga2::builder(representation, objectives)
        .population_size(settings.population_size)
        .crossover(crossover)
        .mutate(mutate);
    if let Some(seed) = settings.seed {
        builder = builder.seed(seed);
    }
    if let Some(rate) = settings.crossover_rate {
        builder = builder.crossover_rate(rate);
    }
    multi_objective(py, setting(builder.build())?, context)
}

// runs a single-objective algorithm, detached from Python so that other threads, and the fitness
// function on rayon's threads, can run
fn generational<'py, A>(py: Python<'py>, algorithm: A, context: &Context) -> Returns<'py>
where
    A: Algorithm + Send,
    A::Genome: Genes,
{
    let stop = context.stop(true)?;
    let shared = &context.shared;
    let parallel = context.parallel;
    let outcome = py.detach(|| {
        Engine::new(algorithm, Single(shared))
            .stop_when(stop)
            .abort_flag(shared.abort_flag())
            .parallel(parallel)
            .on_generation(|_| shared.check_signals())
            .run()
    });
    if let Some(error) = shared.take_error() {
        return Err(error.into());
    }
    let outcome = outcome.map_err(|error| error.to_string())?;
    let fitness = outcome.best_fitness();
    let result = PyDict::new(py);
    result.set_item("best_genome", genes::array(py, outcome.best_genome()))?;
    result.set_item("best_fitness", fitness.score())?;
    result.set_item("violation", fitness.violation())?;
    result.set_item("generations", outcome.generations())?;
    result.set_item("evaluations", outcome.evaluations())?;
    result.set_item("seconds", outcome.elapsed().as_secs_f64())?;
    result.set_item("stop_reason", stop_reason(outcome.stop_reason()))?;
    Ok(result)
}

// runs a multi-objective algorithm, detached from Python like `generational`
fn multi_objective<'py, A, const N: usize>(
    py: Python<'py>,
    algorithm: A,
    context: &Context,
) -> Returns<'py>
where
    A: MultiObjectiveAlgorithm<N> + Send,
    A::Genome: Genes,
{
    let stop = context.stop(false)?;
    let shared = &context.shared;
    let parallel = context.parallel;
    let outcome = py.detach(|| {
        MultiEngine::new(algorithm, Multi(shared))
            .stop_when(stop)
            .abort_flag(shared.abort_flag())
            .parallel(parallel)
            .on_generation(|_| shared.check_signals())
            .run()
    });
    if let Some(error) = shared.take_error() {
        return Err(error.into());
    }
    let outcome = outcome.map_err(|error| error.to_string())?;
    // the front, without copies of a genome
    let mut members: Vec<&Individual<A::Genome, Scores<N>>> = Vec::new();
    for individual in outcome.front() {
        if members
            .iter()
            .all(|member| member.genome() != individual.genome())
        {
            members.push(individual);
        }
    }
    let genomes: Vec<&A::Genome> = members.iter().map(|member| member.genome()).collect();
    let mut objectives = Vec::with_capacity(members.len() * N);
    let mut violations = Vec::with_capacity(members.len());
    for member in &members {
        let scores = member.fitness();
        objectives.extend(
            scores
                .and_then(|scores| scores.values())
                .unwrap_or([f64::NAN; N]),
        );
        violations.push(scores.map_or(f64::NAN, |scores| scores.violation()));
    }
    let objectives = Array2::from_shape_vec((members.len(), N), objectives)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py);
    let result = PyDict::new(py);
    result.set_item("front_genomes", genes::matrix(py, &genomes)?)?;
    result.set_item("front_objectives", objectives)?;
    result.set_item("front_violations", PyArray1::from_vec(py, violations))?;
    result.set_item("generations", outcome.generations())?;
    result.set_item("evaluations", outcome.evaluations())?;
    result.set_item("seconds", outcome.elapsed().as_secs_f64())?;
    result.set_item("stop_reason", stop_reason(outcome.stop_reason()))?;
    Ok(result)
}

fn stop_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Target => "target",
        StopReason::Generations => "generations",
        StopReason::Evaluations => "evaluations",
        StopReason::Time => "time",
        StopReason::Stagnation => "stagnation",
        StopReason::Aborted => "aborted",
        _ => "other",
    }
}
