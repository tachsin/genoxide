//! Building what a run file describes, and running it.

use crate::config;
use crate::operators::{
    AnySelect, ListCrossover, OrderCrossovers, OrderMutation, RealCrossover, RealMutation,
    bit_flip, integer_mutation,
};
use crate::process::{Genes, Multi, Pool, Single, Timeout};
use genoxide::algorithm::{Incremental, cmaes, local, mma, pso};
use genoxide::checkpoint;
use genoxide::engine::asynchronous::MAX_WORKERS;
use genoxide::genome::Representation;
use genoxide::gradient::Gradients;
use genoxide::model::gp;
use genoxide::multi::MultiObjectiveAlgorithm;
use genoxide::observer::Report;
use genoxide::operator::{Crossover, Mutate};
use genoxide::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, String>;

fn setting<T>(result: genoxide::Result<T>) -> Result<T> {
    result.map_err(|error| match error {
        genoxide::Error::MissingSetting { setting } => {
            format!("`algorithm.{setting}` is needed")
        }
        error => error.to_string(),
    })
}

/// How to run.
pub struct Options {
    /// Resume from the checkpoint.
    pub resume: bool,
    /// Only build the algorithm, to check the run file.
    pub check: bool,
}

// what a run needs besides the algorithm
struct Context {
    command: Vec<String>,
    directory: PathBuf,
    workers: usize,
    objectives: Vec<Objective>,
    nan: NanPolicy,
    timeout: Option<Timeout>,
    stop: config::Stop,
    report: config::Report,
    checkpoint: Option<(PathBuf, u64)>,
    // whether the fitness program writes the gradient after the value
    gradient: bool,
    // the inequality constraints whose values and Jacobian it writes after the gradient
    constraints: usize,
    // the genome and algorithm settings, as JSON: a checkpoint resumes only with the same
    settings: String,
    options: Options,
}

// a checkpoint of the program: the settings it was made with, and the algorithm
#[derive(Serialize, serde::Deserialize)]
struct Saved<A> {
    settings: String,
    algorithm: A,
}

/// Runs the run file at `path`, whose contents are `run`: the result as JSON, or `null` when
/// only checking.
pub fn run(run: config::Run, path: &Path, options: Options) -> Result<Value> {
    let directory = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let gradient = run.fitness.gradient;
    let constraints = run.fitness.constraints;
    if constraints > 0 && !gradient {
        return Err(
            "`fitness.constraints` needs `fitness.gradient = true`: the program writes the \
             constraints' values and Jacobian after the gradient"
                .to_string(),
        );
    }
    let command = match (run.fitness.command, run.fitness.builtin) {
        (Some(command), None) if !command.is_empty() => command,
        (None, Some(name)) => {
            if !crate::builtin::FUNCTIONS
                .iter()
                .any(|function| function.name == name)
            {
                return Err(format!(
                    "no built-in fitness `{name}`; {}",
                    crate::builtin::list()
                ));
            }
            if gradient && crate::builtin::gradient_of(&name).is_none() {
                return Err(format!(
                    "`fitness.gradient`: the built-in fitness `{name}` has no gradient; sphere, \
                     rastrigin, rosenbrock, ackley and volume do"
                ));
            }
            let builtin = crate::builtin::constraints_of(&name);
            if gradient && constraints != builtin {
                return Err(format!(
                    "`fitness.constraints`: the built-in fitness `{name}` writes {builtin} \
                     constraint{} with its gradient",
                    if builtin == 1 { "" } else { "s" }
                ));
            }
            let program = std::env::current_exe()
                .map_err(|error| format!("can't find the genoxide program: {error}"))?;
            let mut command = vec![
                program.to_string_lossy().into_owned(),
                "fitness".to_string(),
                name,
            ];
            if gradient {
                command.push("--gradient".to_string());
            }
            command
        }
        (Some(_), None) => return Err("`fitness.command` is empty".to_string()),
        _ => {
            return Err(
                "`fitness` needs either `command` (a program) or `builtin` (a test function)"
                    .to_string(),
            );
        }
    };
    let objectives: Vec<Objective> = run
        .fitness
        .objectives
        .iter()
        .map(|objective| match objective {
            config::Objective::Maximize => Objective::Maximize,
            config::Objective::Minimize => Objective::Minimize,
        })
        .collect();
    if objectives.is_empty() {
        return Err("`fitness.objectives` is empty".to_string());
    }
    if gradient && objectives.len() != 1 {
        return Err(
            "`fitness.gradient` needs a single objective: the gradient is of its value".to_string(),
        );
    }
    let stop = &run.stop;
    if stop.generations.is_none()
        && stop.evaluations.is_none()
        && stop.target.is_none()
        && stop.time.is_none()
        && stop.stagnation.is_none()
    {
        return Err(
            "`stop` needs at least one of generations, evaluations, target, time and stagnation"
                .to_string(),
        );
    }
    let workers = match run.fitness.workers {
        Some(workers) if workers == 0 || workers > MAX_WORKERS => {
            return Err(format!(
                "`fitness.workers` must be between 1 and {MAX_WORKERS}, got {workers}"
            ));
        }
        Some(workers) => workers,
        None => std::thread::available_parallelism().map_or(1, usize::from),
    };
    if run
        .checkpoint
        .as_ref()
        .is_some_and(|checkpoint| checkpoint.every == 0)
    {
        return Err("`checkpoint.every` must be at least 1 generation".to_string());
    }
    // the directory of a checkpoint must exist before the run, not at its first save
    if let Some(checkpoint) = &run.checkpoint {
        let path = directory.join(&checkpoint.path);
        // an empty parent is the current directory
        let parent = path.parent().unwrap_or(Path::new(""));
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            return Err(format!(
                "`checkpoint.path`: the directory {} doesn't exist",
                parent.display()
            ));
        }
    }
    if options.resume && run.checkpoint.is_none() {
        return Err("`--resume` needs a `[checkpoint]` in the run file".to_string());
    }
    if run.fitness.timeout == Some(Duration::ZERO) {
        return Err("`fitness.timeout` must be longer than 0".to_string());
    }
    let settings = serde_json::to_string(&(&run.genome, &run.fitness.objectives, &run.algorithm))
        .map_err(|error| error.to_string())?;
    let context = Context {
        command,
        settings,
        workers,
        objectives,
        // a program that doesn't answer ends the run: at most `stop.time` without a timeout (a
        // time of 0 ends the run after its first generation, however long that takes)
        timeout: match (
            run.fitness.timeout,
            run.stop.time.filter(|time| !time.is_zero()),
        ) {
            (Some(limit), _) => Some(Timeout {
                limit,
                setting: "fitness.timeout",
            }),
            (None, Some(limit)) => Some(Timeout {
                limit,
                setting: "stop.time",
            }),
            (None, None) => None,
        },
        nan: match run.fitness.nan {
            config::Nan::Invalid => NanPolicy::Invalid,
            config::Nan::Error => NanPolicy::Error,
        },
        stop: run.stop,
        report: run.report,
        checkpoint: run
            .checkpoint
            .map(|checkpoint| (directory.join(checkpoint.path), checkpoint.every)),
        gradient,
        constraints,
        directory,
        options,
    };

    match run.genome {
        config::Genome::Binary { length } => with_operators(
            setting(Binary::new(length))?,
            run.algorithm,
            &context,
            |crossover| ListCrossover::new(crossover, "binary"),
            bit_flip,
        ),
        config::Genome::Integer { length, bounds } => {
            let bounds = bounds.per_gene(length)?;
            let integer = setting(Integer::new(bounds.map(|[low, high]| low..=high)))?;
            with_operators(
                integer,
                run.algorithm,
                &context,
                |crossover| ListCrossover::new(crossover, "integer"),
                integer_mutation,
            )
        }
        config::Genome::Real { length, bounds } => {
            let bounds = bounds.per_gene(length)?;
            let real = setting(Real::new(bounds.map(|[low, high]| low..=high)))?;
            real_algorithm(real, run.algorithm, &context)
        }
        config::Genome::Permutation { length } => with_operators(
            setting(Permutation::new(length))?,
            run.algorithm,
            &context,
            OrderCrossovers::new,
            OrderMutation::new,
        ),
    }
}

fn de_strategy(strategy: config::DeStrategy) -> de::Strategy {
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

fn de_control(control: config::DeControl) -> de::Control {
    match control {
        config::DeControl::Fixed(config::Fixed { f, cr }) => de::Control::Fixed { f, cr },
        config::DeControl::Dither(config::Dither { min_f, max_f, cr }) => {
            de::Control::Dither { min_f, max_f, cr }
        }
        config::DeControl::Jade(config::Jade { c }) => de::Control::Jade { c },
        config::DeControl::Shade(config::Shade { memory }) => de::Control::Shade { memory },
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

fn nelder_mead_coefficients(
    coefficients: config::NelderMeadCoefficients,
) -> nelder_mead::Coefficients {
    match coefficients {
        config::NelderMeadCoefficients::Named(config::NelderMeadCoefficientsName::Adaptive) => {
            nelder_mead::Coefficients::Adaptive
        }
        config::NelderMeadCoefficients::Named(config::NelderMeadCoefficientsName::Standard) => {
            nelder_mead::Coefficients::Standard
        }
        config::NelderMeadCoefficients::Custom(config::CustomCoefficients {
            reflection,
            expansion,
            contraction,
            shrink,
        }) => nelder_mead::Coefficients::Custom {
            reflection,
            expansion,
            contraction,
            shrink,
        },
    }
}

// where a gradient-based method's gradients come from, with the relative step of finite
// differences
fn gradient_source(
    source: Option<config::GradientSource>,
    difference_step: Option<f64>,
    context: &Context,
) -> Result<Gradients> {
    Ok(match source.unwrap_or(config::GradientSource::Auto) {
        config::GradientSource::Auto | config::GradientSource::Supplied
            if difference_step.is_some() =>
        {
            return Err(
                "`algorithm.difference_step` needs `gradients = \"forward\"` or `\"central\"`"
                    .to_string(),
            );
        }
        config::GradientSource::Auto => Gradients::Auto,
        config::GradientSource::Supplied if !context.gradient => {
            return Err(
                "`algorithm.gradients = \"supplied\"` needs `fitness.gradient = true`".to_string(),
            );
        }
        config::GradientSource::Supplied => Gradients::Supplied,
        config::GradientSource::Forward => Gradients::Forward {
            step: difference_step,
        },
        config::GradientSource::Central => Gradients::Central {
            step: difference_step,
        },
    })
}

fn first_order_step(step: config::FirstOrderStep) -> first_order::Step {
    use first_order::Step;
    // Kingma and Ba's, as `Step::adam(0.001)`'s
    let adam = |learning_rate: Option<f64>,
                beta1: Option<f64>,
                beta2: Option<f64>,
                epsilon: Option<f64>| {
        (
            learning_rate.unwrap_or(0.001),
            beta1.unwrap_or(0.9),
            beta2.unwrap_or(0.999),
            epsilon.unwrap_or(1e-8),
        )
    };
    match step {
        config::FirstOrderStep::Gradient { learning_rate } => Step::gradient(learning_rate),
        config::FirstOrderStep::Momentum {
            learning_rate,
            momentum,
        } => Step::momentum(learning_rate, momentum),
        config::FirstOrderStep::Nesterov {
            learning_rate,
            momentum,
        } => Step::nesterov(learning_rate, momentum),
        config::FirstOrderStep::Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        } => {
            let (learning_rate, beta1, beta2, epsilon) = adam(learning_rate, beta1, beta2, epsilon);
            Step::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            }
        }
        config::FirstOrderStep::Adamw {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        } => {
            let (learning_rate, beta1, beta2, epsilon) = adam(learning_rate, beta1, beta2, epsilon);
            Step::AdamW {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            }
        }
    }
}

// the algorithms only for real genomes, and the others
fn real_algorithm(real: Real, algorithm: config::Algorithm, context: &Context) -> Result<Value> {
    match algorithm {
        config::Algorithm::De {
            population_size,
            seed,
            l_shade,
            strategy,
            control,
            restarts,
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
            if let Some(strategy) = strategy {
                builder = builder.strategy(de_strategy(strategy));
            }
            if let Some(control) = control {
                builder = builder.control(de_control(control));
            }
            if let Some(restarts) = restarts {
                builder = builder.restarts(de_restarts(restarts));
            }
            let builder = builder.objective(context.single_objective()?);
            generational(setting(builder.build())?, context)
        }
        config::Algorithm::Cmaes {
            population_size,
            seed,
            restarts,
            initial_step,
            covariance,
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
                    config::Restarts::Stop => cmaes::Restarts::Stop,
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
            generational(setting(builder.build())?, context)
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
            generational(setting(builder.build())?, context)
        }
        config::Algorithm::Mma {
            seed,
            method,
            asymptote_initial,
            asymptote_decrease,
            asymptote_increase,
            move_limit,
            constraint_cost,
            kkt_tolerance,
            step_tolerance,
            restoration,
            parallel_sums,
        } => {
            if !context.gradient {
                return Err(
                    "`mma` needs `fitness.gradient = true`: it uses the gradient, and the                      constraints' Jacobian with `fitness.constraints`"
                        .to_string(),
                );
            }
            let mut builder = Mma::builder(real).objective(context.single_objective()?);
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            if let Some(method) = method {
                builder = builder.method(match method {
                    config::MmaMethod::Mma => mma::Method::Mma,
                    config::MmaMethod::Gcmma => mma::Method::Gcmma,
                });
            }
            if let Some(fraction) = asymptote_initial {
                builder = builder.asymptote_initial(fraction);
            }
            if let Some(factor) = asymptote_decrease {
                builder = builder.asymptote_decrease(factor);
            }
            if let Some(factor) = asymptote_increase {
                builder = builder.asymptote_increase(factor);
            }
            if let Some(fraction) = move_limit {
                builder = builder.move_limit(fraction);
            }
            if let Some(cost) = constraint_cost {
                builder = builder.constraint_cost(cost);
            }
            if let Some(tolerance) = kkt_tolerance {
                builder = builder.kkt_tolerance(tolerance);
            }
            if let Some(tolerance) = step_tolerance {
                builder = builder.step_tolerance(tolerance);
            }
            if let Some(restoration) = restoration {
                builder = builder.restoration(restoration);
            }
            if let Some(parallel) = parallel_sums {
                builder = builder.parallel_sums(parallel);
            }
            generational(setting(builder.build())?, context)
        }
        config::Algorithm::Lbfgsb {
            seed,
            memory,
            gradients,
            difference_step,
            gradient_tolerance,
            function_tolerance,
            max_line_search,
            restarts,
        } => {
            let mut builder = Lbfgsb::builder(real).objective(context.single_objective()?);
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            if let Some(memory) = memory {
                builder = builder.memory(memory);
            }
            builder = builder.gradients(gradient_source(gradients, difference_step, context)?);
            if let Some(tolerance) = gradient_tolerance {
                builder = builder.gradient_tolerance(tolerance);
            }
            if let Some(tolerance) = function_tolerance {
                builder = builder.function_tolerance(tolerance);
            }
            if let Some(trials) = max_line_search {
                builder = builder.max_line_search(trials);
            }
            match restarts {
                Some(0) => {
                    return Err(
                        "`algorithm.restarts` must be at least 1; leave it out for none"
                            .to_string(),
                    );
                }
                Some(times) => builder = builder.restarts(local::Restarts::Random { times }),
                None => {}
            }
            generational(setting(builder.build())?, context)
        }
        config::Algorithm::Bo {
            seed,
            initial_points,
            acquisition,
            kernel,
            noise,
            output,
            raw_samples,
            acquisition_starts,
            hyperparameter_starts,
        } => {
            let mut builder = Bo::builder(real).objective(context.single_objective()?);
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            if let Some(points) = initial_points {
                builder = builder.initial_points(points);
            }
            if let Some(acquisition) = acquisition {
                builder = builder.acquisition(match acquisition {
                    config::BoAcquisition::Named(config::BoAcquisitionName::LogEi) => {
                        bo::Acquisition::LogExpectedImprovement
                    }
                    config::BoAcquisition::Named(config::BoAcquisitionName::Ei) => {
                        bo::Acquisition::ExpectedImprovement
                    }
                    config::BoAcquisition::Table(config::BoAcquisitionTable::Pi { xi }) => {
                        bo::Acquisition::ProbabilityOfImprovement { xi }
                    }
                    config::BoAcquisition::Table(config::BoAcquisitionTable::Ucb { beta }) => {
                        bo::Acquisition::UpperConfidenceBound { beta }
                    }
                });
            }
            if let Some(kernel) = kernel {
                builder = builder.kernel(match kernel {
                    config::Kernel::Matern52 => gp::Kernel::Matern52,
                    config::Kernel::SquaredExponential => gp::Kernel::SquaredExponential,
                });
            }
            if let Some(noise) = noise {
                builder = builder.noise(match noise {
                    config::BoNoise::Fixed(variance) => gp::Noise::Fixed(variance),
                    config::BoNoise::Learned { learned } => gp::Noise::Learned { min: learned },
                });
            }
            if let Some(output) = output {
                builder = builder.output(match output {
                    config::BoOutput::Standardize => bo::Output::Standardize,
                    config::BoOutput::Log => bo::Output::Log,
                });
            }
            if let Some(samples) = raw_samples {
                builder = builder.raw_samples(samples);
            }
            if let Some(starts) = acquisition_starts {
                builder = builder.acquisition_starts(starts);
            }
            if let Some(starts) = hyperparameter_starts {
                builder = builder.hyperparameter_starts(starts);
            }
            generational(setting(builder.build())?, context)
        }
        config::Algorithm::NelderMead {
            seed,
            coefficients,
            initial_step,
            initial_step_absolute,
            tolerance,
            restarts,
            speculative,
        } => {
            if initial_step.is_some() && initial_step_absolute.is_some() {
                return Err(
                    "`algorithm.initial_step` and `algorithm.initial_step_absolute`: give one, not both"
                        .to_string(),
                );
            }
            let mut builder = NelderMead::builder(real).objective(context.single_objective()?);
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
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
            match restarts {
                Some(0) => {
                    return Err(
                        "`algorithm.restarts` must be at least 1; leave it out for none"
                            .to_string(),
                    );
                }
                Some(times) => builder = builder.restarts(local::Restarts::Random { times }),
                None => {}
            }
            if let Some(speculative) = speculative {
                builder = builder.speculative(speculative);
            }
            generational(setting(builder.build())?, context)
        }
        config::Algorithm::FirstOrder {
            seed,
            step,
            gradients,
            difference_step,
            gradient_tolerance,
            step_tolerance,
            restarts,
        } => {
            let mut builder = FirstOrder::builder(real).objective(context.single_objective()?);
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            if let Some(step) = step {
                builder = builder.step(first_order_step(step));
            }
            builder = builder.gradients(gradient_source(gradients, difference_step, context)?);
            if let Some(tolerance) = gradient_tolerance {
                builder = builder.gradient_tolerance(tolerance);
            }
            if let Some(tolerance) = step_tolerance {
                builder = builder.step_tolerance(tolerance);
            }
            match restarts {
                Some(0) => {
                    return Err(
                        "`algorithm.restarts` must be at least 1; leave it out for none"
                            .to_string(),
                    );
                }
                Some(times) => builder = builder.restarts(local::Restarts::Random { times }),
                None => {}
            }
            generational(setting(builder.build())?, context)
        }
        algorithm => with_operators(
            real,
            algorithm,
            context,
            RealCrossover::new,
            RealMutation::new,
        ),
    }
}

// the algorithms for any genome, with its operators
fn with_operators<R, C, M>(
    representation: R,
    algorithm: config::Algorithm,
    context: &Context,
    crossover: impl Fn(config::Crossover) -> Result<C>,
    mutate: impl Fn(config::Mutate) -> Result<M>,
) -> Result<Value>
where
    R: Representation + Serialize + DeserializeOwned,
    R::Genome: Genes + Serialize + DeserializeOwned + Send + Sync,
    C: Crossover<R> + Serialize + DeserializeOwned,
    M: Mutate<R> + Serialize + DeserializeOwned,
{
    match algorithm {
        config::Algorithm::Ga(ga) => {
            let builder = ga_builder(representation, &ga, context, &crossover, &mutate)?;
            generational(setting(builder.build())?, context)
        }
        config::Algorithm::SteadyGa(ga) => {
            if ga.scheme.is_some() {
                return Err("`steady-ga` has no scheme: each result replaces the worst individual when it's not worse".to_string());
            }
            let builder = ga_builder(representation, &ga, context, &crossover, &mutate)?;
            asynchronous(setting(builder.build_steady())?, context)
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
            generational(setting(builder.build())?, context)
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
                2 => nsga2::<_, _, _, 2>(representation, crossover, mutate, &settings, context),
                3 => nsga2::<_, _, _, 3>(representation, crossover, mutate, &settings, context),
                4 => nsga2::<_, _, _, 4>(representation, crossover, mutate, &settings, context),
                5 => nsga2::<_, _, _, 5>(representation, crossover, mutate, &settings, context),
                6 => nsga2::<_, _, _, 6>(representation, crossover, mutate, &settings, context),
                count => Err(format!(
                    "`nsga2` takes 2 to 6 objectives, got {count}: `fitness.objectives` has one entry per objective"
                )),
            }
        }
        config::Algorithm::De { .. } => Err("`de` needs a real genome".to_string()),
        config::Algorithm::Cmaes { .. } => Err("`cmaes` needs a real genome".to_string()),
        config::Algorithm::Pso { .. } => Err("`pso` needs a real genome".to_string()),
        config::Algorithm::NelderMead { .. } => {
            Err("`nelder-mead` needs a real genome".to_string())
        }
        config::Algorithm::Bo { .. } => Err("`bo` needs a real genome".to_string()),
        config::Algorithm::Lbfgsb { .. } => Err("`lbfgsb` needs a real genome".to_string()),
        config::Algorithm::Mma { .. } => Err("`mma` needs a real genome".to_string()),
        config::Algorithm::FirstOrder { .. } => {
            Err("`first-order` needs a real genome".to_string())
        }
    }
}

fn ga_builder<R, C, M>(
    representation: R,
    ga: &config::Ga,
    context: &Context,
    crossover: &impl Fn(config::Crossover) -> Result<C>,
    mutate: &impl Fn(config::Mutate) -> Result<M>,
) -> Result<genoxide::algorithm::GaBuilder<R, AnySelect, C, M>>
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

fn nsga2<R, C, X, const N: usize>(
    representation: R,
    crossover: C,
    mutate: X,
    settings: &Nsga2Settings,
    context: &Context,
) -> Result<Value>
where
    R: Representation + Serialize + DeserializeOwned,
    R::Genome: Genes + Serialize + DeserializeOwned + Send + Sync,
    C: Crossover<R> + Serialize + DeserializeOwned,
    X: Mutate<R> + Serialize + DeserializeOwned,
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
    multi_objective(setting(builder.build())?, context)
}

impl Context {
    // the objective of a single-objective algorithm
    fn single_objective(&self) -> Result<Objective> {
        match self.objectives.as_slice() {
            [objective] => Ok(*objective),
            objectives => Err(format!(
                "this algorithm optimizes one objective, but `fitness.objectives` has {}; use `nsga2` for several",
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
                return Err("`stop.target` needs a single objective".to_string());
            }
            if target.is_nan() {
                return Err("`stop.target` is NaN".to_string());
            }
            conditions.push(Stop::target(target));
        }
        if let Some(time) = config.time {
            conditions.push(Stop::time(time));
        }
        if let Some(stagnation) = config.stagnation {
            if stagnation == 0 {
                return Err("`stop.stagnation` must be at least 1 generation".to_string());
            }
            conditions.push(Stop::stagnation(stagnation));
        }
        let mut conditions = conditions.into_iter();
        let first = conditions
            .next()
            .ok_or("`stop` needs at least one condition")?;
        Ok(conditions.fold(first, Stop::or))
    }

    // the progress report, if any
    fn report(&self) -> Result<Option<Report>> {
        match &self.report {
            config::Report::Text(text) if text == "off" => Ok(None),
            config::Report::Text(text) => Ok(Some(Report::every(config::parse_duration(text)?))),
            config::Report::Generations(0) => {
                Err("`report` must be at least 1 generation".to_string())
            }
            config::Report::Generations(generations) => {
                setting(Report::every_generations(*generations)).map(Some)
            }
        }
    }

    // the algorithm from the checkpoint when resuming, or the new one
    fn resume<A: DeserializeOwned>(&self, algorithm: A) -> Result<A> {
        let (Some((path, _)), true) = (&self.checkpoint, self.options.resume) else {
            return Ok(algorithm);
        };
        let saved: Saved<A> = setting(checkpoint::load_file(path))?;
        if saved.settings != self.settings {
            return Err(format!(
                "{} was saved with other genome, objectives or algorithm settings; resume with the run file that made it, changing only `fitness.command`, `fitness.builtin`, `fitness.workers`, `fitness.nan`, `stop`, `report` and `checkpoint`",
                path.display()
            ));
        }
        Ok(saved.algorithm)
    }

    // saves a checkpoint of `algorithm`, unless a fitness program failed: its evaluations since
    // are NaN, and the last good checkpoint stays
    fn save<A: Serialize + Clone>(
        &self,
        algorithm: &A,
        path: &Path,
        pool: &Pool,
    ) -> genoxide::Result<()> {
        if pool.failure().is_some() {
            return Ok(());
        }
        let saved = Saved {
            settings: self.settings.clone(),
            algorithm: algorithm.clone(),
        };
        checkpoint::save_file(&saved, path)
    }

    // rayon's threads for parallel evaluation: one per worker. A run is the only one of its
    // process, so the global pool is set once.
    fn threads(&self) -> Result<()> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.workers)
            .build_global()
            .map_err(|error| format!("can't start {} threads: {error}", self.workers))
    }

    fn start(&self) -> Result<(Pool, Arc<AtomicBool>)> {
        let abort = Arc::new(AtomicBool::new(false));
        let pool = Pool::start(
            &self.command,
            &self.directory,
            self.workers,
            abort.clone(),
            self.timeout,
            self.gradient,
            self.constraints,
        )?;
        Ok((pool, abort))
    }
}

// runs a generational single-objective algorithm, evaluating in parallel
fn generational<A>(algorithm: A, context: &Context) -> Result<Value>
where
    A: Algorithm + Serialize + DeserializeOwned + Clone,
    A::Genome: Genes + Send + Sync,
{
    let stop = context.stop(true)?;
    let report = context.report()?;
    let algorithm = context.resume(algorithm)?;
    if context.options.check {
        return Ok(Value::Null);
    }
    context.threads()?;
    let (pool, abort) = context.start()?;
    let fitness = Single(&pool);
    let mut engine = Engine::new(algorithm, fitness)
        .stop_when(stop)
        .abort_flag(abort)
        .nan_policy(context.nan)
        .parallel(true);
    if let Some(report) = report {
        engine = engine.observe(report);
    }
    if let Some((path, every)) = &context.checkpoint {
        engine = engine.checkpoint_every(*every, |algorithm| context.save(algorithm, path, &pool));
    }
    let start = Instant::now();
    let outcome = engine.run();
    drop(engine);
    finish(&pool, outcome, start.elapsed())
}

// runs a steady-state GA with asynchronous evaluation
fn asynchronous<A>(algorithm: A, context: &Context) -> Result<Value>
where
    A: Incremental + Serialize + DeserializeOwned + Clone,
    A::Genome: Genes + Send + Sync,
{
    let stop = context.stop(true)?;
    let report = context.report()?;
    let algorithm = context.resume(algorithm)?;
    if context.options.check {
        return Ok(Value::Null);
    }
    let (pool, abort) = context.start()?;
    let mut engine = AsyncEngine::new(algorithm, Single(&pool))
        .workers(context.workers)
        .stop_when(stop)
        .abort_flag(abort)
        .nan_policy(context.nan);
    if let Some(report) = report {
        engine = engine.observe(report);
    }
    if let Some((path, every)) = &context.checkpoint {
        engine = engine.checkpoint_every(*every, |algorithm| context.save(algorithm, path, &pool));
    }
    let start = Instant::now();
    let outcome = engine.run();
    drop(engine);
    finish(&pool, outcome, start.elapsed())
}

// the result of a single-objective run, or its error
fn finish<G: Genes + genoxide::genome::Genome>(
    pool: &Pool,
    outcome: genoxide::Result<Outcome<G>>,
    elapsed: Duration,
) -> Result<Value> {
    if let Some(failure) = pool.close() {
        return Err(failure);
    }
    let outcome = setting(outcome)?;
    let fitness = outcome.best_fitness();
    Ok(json!({
        "stop_reason": stop_reason(outcome.stop_reason()),
        "generations": outcome.generations(),
        "evaluations": outcome.evaluations(),
        "seconds": elapsed.as_secs_f64(),
        "fitness": fitness.score().map(number),
        "violation": number(fitness.violation()),
        "genome": outcome.best_genome().to_json(),
    }))
}

// runs a multi-objective algorithm, evaluating in parallel
fn multi_objective<A, const N: usize>(algorithm: A, context: &Context) -> Result<Value>
where
    A: MultiObjectiveAlgorithm<N> + Serialize + DeserializeOwned + Clone,
    A::Genome: Genes + Send + Sync,
{
    let stop = context.stop(false)?;
    let mut report = context.report()?;
    let algorithm = context.resume(algorithm)?;
    if context.options.check {
        return Ok(Value::Null);
    }
    context.threads()?;
    let (pool, abort) = context.start()?;
    let mut engine = MultiEngine::new(algorithm, Multi(&pool))
        .stop_when(stop)
        .abort_flag(abort)
        .nan_policy(context.nan)
        .parallel(true);
    if report.is_some() {
        engine = engine.on_generation(move |snapshot| {
            if let Some(report) = &mut report {
                report.update(snapshot.progress());
            }
        });
    }
    if let Some((path, every)) = &context.checkpoint {
        engine = engine.checkpoint_every(*every, |algorithm| context.save(algorithm, path, &pool));
    }
    let start = Instant::now();
    let outcome = engine.run();
    drop(engine);
    if let Some(failure) = pool.close() {
        return Err(failure);
    }
    let outcome = setting(outcome)?;
    // the front, each genome once
    let front: Vec<Value> = outcome
        .front()
        .iter()
        .map(|individual| {
            let scores = individual.fitness();
            json!({
                "objectives": scores.and_then(|scores| scores.values()).map(|values| values.map(number).to_vec()),
                "violation": scores.map(|scores| number(scores.violation())),
                "genome": individual.genome().to_json(),
            })
        })
        .collect();
    Ok(json!({
        "stop_reason": stop_reason(outcome.stop_reason()),
        "generations": outcome.generations(),
        "evaluations": outcome.evaluations(),
        "seconds": start.elapsed().as_secs_f64(),
        "front": front,
    }))
}

fn stop_reason(reason: StopReason) -> String {
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
    .to_string()
}

// a number as JSON: infinities, which JSON has no numbers for, as the text "inf" and "-inf"
fn number(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else if value > 0.0 {
        json!("inf")
    } else {
        json!("-inf")
    }
}
