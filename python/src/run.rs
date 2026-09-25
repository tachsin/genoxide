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
use genoxide::multi::{self, Decomposition, MultiObjectiveAlgorithm, SmsEmoa};
use genoxide::operator::{Crossover, Mutate};
use genoxide::prelude::*;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::Duration;

type Result<T> = std::result::Result<T, String>;

fn setting<T>(result: genoxide::Result<T>) -> Result<T> {
    result.map_err(|error| match error {
        genoxide::Error::MissingSetting { setting } => format!("`{setting}` is needed"),
        error => error.to_string(),
    })
}

/// Runs the optimization that `config` (JSON, from the Python package) describes, with
/// `fitness`, called with a genome or, with `batch`, a generation of genomes; with `parallel`,
/// from several threads at once. `on_generation` is called after every generation with the
/// generation, the evaluations, the seconds and the best fitness (or the size of the front), and
/// returns False to stop the run. Returns the result as a dict.
#[pyfunction]
#[pyo3(signature = (config, fitness, batch = false, parallel = false, on_generation = None))]
pub fn run<'py>(
    py: Python<'py>,
    config: &str,
    fitness: Py<PyAny>,
    batch: bool,
    parallel: bool,
    on_generation: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    // the error names the setting, e.g. `stop.generations`
    let mut json = serde_json::Deserializer::from_str(config);
    let run: config::Run = serde_path_to_error::deserialize(&mut json).map_err(|error| {
        let (path, error) = (error.path().to_string(), error.into_inner());
        PyValueError::new_err(format!("invalid setting `{path}`: {error}"))
    })?;
    let context = Context {
        shared: Shared::new(fitness, batch, on_generation),
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

// why a run failed: its description, or Python (the fitness function, the progress callback,
// Ctrl+C)
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
        config::Algorithm::Nsga2 { variation, .. }
        | config::Algorithm::Nsga3 { variation, .. }
        | config::Algorithm::Spea2 { variation, .. }
        | config::Algorithm::Moead { variation, .. }
        | config::Algorithm::SmsEmoa { variation, .. } => {
            let multi = MultiObjective {
                py,
                representation,
                crossover: crossover(variation.crossover)?,
                mutate: mutate(variation.mutate)?,
                algorithm,
                context,
            };
            with_objectives(context.objectives.len(), multi).unwrap_or_else(|count| {
                let message =
                    format!("multi-objective algorithms take 2 to 6 objectives, not {count}");
                Err(message.into())
            })
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

// something to do with the number of objectives as a constant
trait WithObjectives {
    type Output;

    fn with<const N: usize>(self) -> Self::Output;
}

// does `task` with `count` objectives, 2 to 6, or returns `Err(count)`
fn with_objectives<T: WithObjectives>(
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
    let points = with_objectives(objectives, DasDennis(divisions)).map_err(|count| {
        PyValueError::new_err(format!("das_dennis takes 2 to 6 objectives, not {count}"))
    })?;
    Ok(points.into_pyarray(py))
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
    R: Representation,
    R::Genome: Genes,
    C: Crossover<R>,
    X: Mutate<R>,
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
            } => {
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
            .on_generation(|snapshot| {
                let progress = snapshot.progress();
                shared.after_generation(progress, progress.best().and_then(Fitness::score));
            })
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
            .on_generation(|snapshot| {
                shared.after_generation(snapshot.progress(), snapshot.front().len());
            })
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
        // no violation for an invalid solution: NaN, as 0 means feasible
        let valid = scores.filter(|scores| scores.is_valid());
        violations.push(valid.map_or(f64::NAN, |scores| scores.violation()));
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
        StopReason::Stalled => "stalled",
        _ => "other",
    }
}
