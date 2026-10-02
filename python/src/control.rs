//! Parameter control: a Python callable that gets the running algorithm once per generation, from
//! genoxide's `Engine::control`, to change its settings or re-evaluate it.
//!
//! The callable gets a [`Running`] handle. During the call, the algorithm is moved into the
//! handle's slot, and a spare copy takes its place in the engine; afterwards it's moved back. Out
//! of the call the slot is empty, and the handle raises a `RuntimeError`. Each setting is changed
//! with the algorithm's own validating setter: an invalid value raises a `ValueError` and changes
//! nothing.

use crate::config;
use crate::errors::setting;
use crate::model::PyGaussianProcess;
use crate::operators::AnySelect;
use crate::run::{de_control, de_strategy};
use genoxide::algorithm::islands::Migrate;
use genoxide::algorithm::{Islands, Reevaluate};
use genoxide::genome::Representation;
use genoxide::neat::Neat;
use genoxide::operator::{Crossover, Mutate};
use genoxide::prelude::*;
use numpy::ndarray::ArrayView2;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex, PoisonError};

type Result<T> = std::result::Result<T, String>;

/// The settings of an algorithm that a control can read and change, by their Python names.
pub trait Settings<A>: Send + Sync + 'static {
    /// The current value of a setting, as JSON.
    fn get(&self, algorithm: &A, name: &str) -> Result<Value>;

    /// Changes a setting to `value`, JSON; nothing changes on errors.
    fn set(&self, algorithm: &mut A, name: &str, value: &str) -> Result<()>;

    /// The surrogate model that chose the last point, and its number of genes, for the
    /// algorithms that have one: Bayesian optimization's. None by default.
    fn model(&self, algorithm: &A) -> Option<PyGaussianProcess> {
        let _ = algorithm;
        None
    }

    /// The acquisition function at `points`, a point per row, for the algorithms that have one.
    fn acquisition(&self, algorithm: &A, points: ArrayView2<'_, f64>) -> Result<Vec<f64>> {
        let _ = (algorithm, points);
        Err("the algorithm has no acquisition function".to_string())
    }

    /// The probability of feasibility at `points`, a point per row, for the algorithms that
    /// model constraints.
    fn feasibility(&self, algorithm: &A, points: ArrayView2<'_, f64>) -> Result<Vec<f64>> {
        let _ = (algorithm, points);
        Err("the algorithm has no model of the constraints".to_string())
    }
}

fn unknown(name: &str) -> String {
    format!("the algorithm has no setting `{name}` to control")
}

// a setting's value, from the JSON the Python package makes
fn parse<T: DeserializeOwned>(name: &str, value: &str) -> Result<T> {
    serde_json::from_str(value).map_err(|error| format!("invalid setting `{name}`: {error}"))
}

/// The builders of a GA's crossover and mutation, as for the genome of its run.
pub struct GaSettings<C, M> {
    pub crossover: fn(config::Crossover) -> Result<C>,
    pub mutate: fn(config::Mutate) -> Result<M>,
}

impl<R, C, M> Settings<Ga<R, AnySelect, C, M>> for GaSettings<C, M>
where
    R: Representation,
    C: Crossover<R> + 'static,
    M: Mutate<R> + 'static,
{
    fn get(&self, ga: &Ga<R, AnySelect, C, M>, name: &str) -> Result<Value> {
        match name {
            "crossover_rate" => Ok(json!(ga.crossover_rate())),
            "mutation_rate" => Ok(json!(ga.mutation_rate())),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, ga: &mut Ga<R, AnySelect, C, M>, name: &str, value: &str) -> Result<()> {
        match name {
            "crossover_rate" => setting(ga.set_crossover_rate(parse(name, value)?)),
            "mutation_rate" => setting(ga.set_mutation_rate(parse(name, value)?)),
            "select" => {
                *ga.select_mut() = AnySelect::new(parse(name, value)?)?;
                Ok(())
            }
            "crossover" => {
                let crossover = (self.crossover)(parse(name, value)?)?;
                let previous = std::mem::replace(ga.crossover_mut(), crossover);
                // a crossover that doesn't recombine needs a mutation rate above 0, as in the
                // builder: the rates are checked again with it
                let rate = ga.crossover_rate();
                setting(ga.set_crossover_rate(rate)).inspect_err(|_| {
                    *ga.crossover_mut() = previous;
                })
            }
            "mutation" => {
                *ga.mutate_mut() = (self.mutate)(parse(name, value)?)?;
                Ok(())
            }
            _ => Err(unknown(name)),
        }
    }
}

/// The builder of a local search's neighbor operator, as for the genome of its run.
pub struct LocalSearchSettings<M> {
    pub neighbor: fn(config::Mutate) -> Result<M>,
}

impl<R, M> Settings<LocalSearch<R, M>> for LocalSearchSettings<M>
where
    R: Representation,
    M: Mutate<R> + 'static,
{
    fn get(&self, search: &LocalSearch<R, M>, name: &str) -> Result<Value> {
        match name {
            "neighbors" => Ok(json!(search.neighbors())),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, search: &mut LocalSearch<R, M>, name: &str, value: &str) -> Result<()> {
        match name {
            "neighbors" => setting(search.set_neighbors(parse(name, value)?)),
            "neighbor" => {
                *search.neighbor_mut() = (self.neighbor)(parse(name, value)?)?;
                Ok(())
            }
            _ => Err(unknown(name)),
        }
    }
}

/// Differential evolution's strategy and control of `F` and `CR`.
pub struct DeSettings;

impl Settings<De> for DeSettings {
    fn get(&self, de: &De, name: &str) -> Result<Value> {
        match name {
            "strategy" => Ok(match de.strategy() {
                de::Strategy::Rand1 => json!("rand1"),
                de::Strategy::Best1 => json!("best1"),
                de::Strategy::CurrentToPBest { p, archive } => json!({"p": p, "archive": archive}),
                de::Strategy::CurrentToPBestRandomP { max_p, archive } => {
                    json!({"max_p": max_p, "archive": archive})
                }
                strategy => {
                    return Err(format!("a strategy the package doesn't know: {strategy:?}"));
                }
            }),
            "control" => Ok(match de.control() {
                de::Control::Fixed { f, cr } => json!({"f": f, "cr": cr}),
                de::Control::Dither { min_f, max_f, cr } => {
                    json!({"min_f": min_f, "max_f": max_f, "cr": cr})
                }
                de::Control::Jade { c } => json!({"c": c}),
                de::Control::Shade { memory } => json!({"memory": memory}),
                control => return Err(format!("a control the package doesn't know: {control:?}")),
            }),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, de: &mut De, name: &str, value: &str) -> Result<()> {
        match name {
            "strategy" => setting(de.set_strategy(de_strategy(parse(name, value)?))),
            "control" => setting(de.set_control(de_control(parse(name, value)?))),
            _ => Err(unknown(name)),
        }
    }
}

/// Particle swarm optimization's inertia and accelerations.
pub struct PsoSettings;

impl Settings<Pso> for PsoSettings {
    fn get(&self, pso: &Pso, name: &str) -> Result<Value> {
        match name {
            "inertia" => Ok(json!(pso.inertia())),
            "acceleration" => Ok(json!(pso.acceleration())),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, pso: &mut Pso, name: &str, value: &str) -> Result<()> {
        match name {
            "inertia" => setting(pso.set_inertia(parse(name, value)?)),
            "acceleration" => {
                let (cognitive, social) = parse(name, value)?;
                setting(pso.set_acceleration(cognitive, social))
            }
            _ => Err(unknown(name)),
        }
    }
}

/// CMA-ES, which adapts its own settings: only re-evaluation.
pub struct CmaesSettings;

impl Settings<Cmaes> for CmaesSettings {
    fn get(&self, _: &Cmaes, name: &str) -> Result<Value> {
        Err(unknown(name))
    }

    fn set(&self, _: &mut Cmaes, name: &str, _: &str) -> Result<()> {
        Err(unknown(name))
    }
}

/// An evolution strategy, which adapts its own step sizes: only re-evaluation.
pub struct EsSettings;

impl Settings<Es> for EsSettings {
    fn get(&self, _: &Es, name: &str) -> Result<Value> {
        Err(unknown(name))
    }

    fn set(&self, _: &mut Es, name: &str, _: &str) -> Result<()> {
        Err(unknown(name))
    }
}

/// NEAT's state, to read: its species (each with its representative network, as genoxide
/// serializes it) and the innovation numbers given so far. It has nothing to change.
pub struct NeatSettings;

impl Settings<Neat> for NeatSettings {
    fn get(&self, neat: &Neat, name: &str) -> Result<Value> {
        match name {
            "species" => neat
                .species()
                .iter()
                .map(|species| {
                    let representative = serde_json::to_value(species.representative())
                        .map_err(|error| error.to_string())?;
                    Ok(json!({
                        "id": species.id(),
                        "members": species.members(),
                        "best_fitness": species.best().and_then(Fitness::score),
                        "improved": species.improved(),
                        "created": species.created(),
                        "representative": representative,
                    }))
                })
                .collect::<Result<Vec<Value>>>()
                .map(Value::Array),
            "innovations" => Ok(json!(neat.innovations())),
            "seed" => Ok(json!(neat.seed())),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, _: &mut Neat, name: &str, _: &str) -> Result<()> {
        Err(unknown(name))
    }
}

/// Nelder-Mead, whose steps follow from its simplex: the state of the simplex, read-only, and
/// re-evaluation.
pub struct NelderMeadSettings;

impl Settings<NelderMead> for NelderMeadSettings {
    fn get(&self, nelder_mead: &NelderMead, name: &str) -> Result<Value> {
        match name {
            "converged" => Ok(json!(nelder_mead.converged())),
            "size" => Ok(json!(nelder_mead.size())),
            "iterations" => Ok(json!(nelder_mead.iterations())),
            "restart_count" => Ok(json!(nelder_mead.restart_count())),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, _: &mut NelderMead, name: &str, _: &str) -> Result<()> {
        Err(unknown(name))
    }
}

/// MMA, whose subproblems follow from its asymptotes: its state, read-only, and re-evaluation.
pub struct MmaSettings;

impl Settings<Mma> for MmaSettings {
    fn get(&self, mma: &Mma, name: &str) -> Result<Value> {
        match name {
            "converged" => Ok(json!(mma.converged().map(|criterion| match criterion {
                mma::Convergence::Kkt => "kkt",
                _ => "step",
            }))),
            "iterations" => Ok(json!(mma.iterations())),
            "inner_iterations" => Ok(json!(mma.inner_iterations())),
            "multipliers" => Ok(json!(mma.multipliers())),
            // NaN before the first iteration, which JSON can't hold
            "kkt_residual" => Ok(json!(
                Some(mma.kkt_residual()).filter(|residual| residual.is_finite())
            )),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, _: &mut Mma, name: &str, _: &str) -> Result<()> {
        Err(unknown(name))
    }
}

/// L-BFGS-B: its memory, which a control can change, and its state, read-only.
pub struct LbfgsbSettings;

impl Settings<Lbfgsb> for LbfgsbSettings {
    fn get(&self, lbfgsb: &Lbfgsb, name: &str) -> Result<Value> {
        match name {
            "memory" => Ok(json!(lbfgsb.memory())),
            "pairs" => Ok(json!(lbfgsb.pairs())),
            "converged" => Ok(json!(lbfgsb.converged().map(criterion))),
            "projected_gradient" => Ok(json!(lbfgsb.projected_gradient())),
            "iterations" => Ok(json!(lbfgsb.iterations())),
            "gradients" => Ok(json!(gradient_source(lbfgsb.gradients()))),
            "gradient_evaluations" => Ok(json!(lbfgsb.gradient_evaluations())),
            "stencil_evaluations" => Ok(json!(lbfgsb.stencil_evaluations())),
            "skipped_pairs" => Ok(json!(lbfgsb.skipped_pairs())),
            "memory_resets" => Ok(json!(lbfgsb.memory_resets())),
            "restart_count" => Ok(json!(lbfgsb.restart_count())),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, lbfgsb: &mut Lbfgsb, name: &str, value: &str) -> Result<()> {
        match name {
            "memory" => setting(lbfgsb.set_memory(parse(name, value)?)),
            _ => Err(unknown(name)),
        }
    }
}

// a convergence criterion's Python name
fn criterion(criterion: lbfgsb::Criterion) -> &'static str {
    match criterion {
        lbfgsb::Criterion::ProjectedGradient => "projected_gradient",
        lbfgsb::Criterion::RelativeDecrease => "relative_decrease",
        lbfgsb::Criterion::LineSearch => "line_search",
        lbfgsb::Criterion::NotFinite => "not_finite",
        _ => "other",
    }
}

// where the gradients come from, by its Python name
fn gradient_source(gradients: genoxide::gradient::Gradients) -> &'static str {
    use genoxide::gradient::Gradients;
    match gradients {
        Gradients::Supplied => "supplied",
        Gradients::Forward { .. } => "forward",
        Gradients::Central { .. } => "central",
        _ => "auto",
    }
}

/// A first-order method: its learning rate and schedule multiplier, and its state, read-only.
pub struct FirstOrderSettings;

impl Settings<FirstOrder> for FirstOrderSettings {
    fn get(&self, first_order: &FirstOrder, name: &str) -> Result<Value> {
        match name {
            "learning_rate" => Ok(json!(first_order.step().learning_rate())),
            "multiplier" => Ok(json!(first_order.multiplier())),
            "converged" => Ok(json!(first_order.converged().map(convergence))),
            // infinite before the first tell, which JSON can't hold
            "gradient_norm" => Ok(json!(
                Some(first_order.gradient_norm()).filter(|norm| norm.is_finite())
            )),
            "gradient" => Ok(json!(first_order.gradient())),
            "iterations" => Ok(json!(first_order.iterations())),
            "steps" => Ok(json!(first_order.steps())),
            "restart_count" => Ok(json!(first_order.restart_count())),
            "gradients" => Ok(json!(gradient_source(first_order.gradients()))),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, first_order: &mut FirstOrder, name: &str, value: &str) -> Result<()> {
        match name {
            "learning_rate" => setting(first_order.set_learning_rate(parse(name, value)?)),
            "multiplier" => setting(first_order.set_multiplier(parse(name, value)?)),
            _ => Err(unknown(name)),
        }
    }
}

// why a first-order method converged, by its Python name
fn convergence(convergence: first_order::Convergence) -> &'static str {
    match convergence {
        first_order::Convergence::Gradient => "gradient",
        first_order::Convergence::Step => "step",
        first_order::Convergence::Invalid => "invalid",
        _ => "other",
    }
}

/// OpenAI's evolution strategy's step: σ and the learning rate.
pub struct OpenEsSettings;

impl Settings<OpenEs> for OpenEsSettings {
    fn get(&self, open_es: &OpenEs, name: &str) -> Result<Value> {
        match name {
            "sigma" => Ok(json!(open_es.sigma())),
            "learning_rate" => Ok(json!(open_es.optimizer().learning_rate())),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, open_es: &mut OpenEs, name: &str, value: &str) -> Result<()> {
        match name {
            "sigma" => setting(open_es.set_sigma(parse(name, value)?)),
            "learning_rate" => setting(open_es.set_learning_rate(parse(name, value)?)),
            _ => Err(unknown(name)),
        }
    }
}

/// The settings of each island, those of its algorithm, named `index/setting`: `2/mutation_rate`
/// is the mutation rate of the third island.
pub struct IslandsSettings<S>(pub S);

// the index of the island and its setting's name, from `index/setting`, for `count` islands
fn island(name: &str, count: usize) -> Result<(usize, &str)> {
    let (index, setting) = name.split_once('/').ok_or_else(|| unknown(name))?;
    match index.parse::<usize>() {
        Ok(index) if index < count => Ok((index, setting)),
        _ => Err(format!("no island {index} of the {count}")),
    }
}

impl<A, S> Settings<Islands<A>> for IslandsSettings<S>
where
    A: Migrate,
    S: Settings<A>,
{
    fn get(&self, islands: &Islands<A>, name: &str) -> Result<Value> {
        let (index, setting) = island(name, islands.islands().len())?;
        self.0.get(&islands.islands()[index], setting)
    }

    fn set(&self, islands: &mut Islands<A>, name: &str, value: &str) -> Result<()> {
        let (index, setting) = island(name, islands.islands().len())?;
        self.0
            .set(&mut islands.islands_mut()[index], setting, value)
    }
}

/// Bayesian optimization's acquisition function, batch and fantasy, which can change, its model
/// and its acquisition's values and probability of feasibility, for a plot.
pub struct BoSettings;

impl<R: bo::Space> Settings<Bo<R>> for BoSettings {
    fn get(&self, bo: &Bo<R>, name: &str) -> Result<Value> {
        match name {
            "acquisition" => Ok(match bo.acquisition() {
                bo::Acquisition::ExpectedImprovement => json!({"type": "expected_improvement"}),
                bo::Acquisition::LogExpectedImprovement => {
                    json!({"type": "log_expected_improvement"})
                }
                bo::Acquisition::ProbabilityOfImprovement { xi } => {
                    json!({"type": "probability_of_improvement", "xi": xi})
                }
                bo::Acquisition::UpperConfidenceBound { beta } => {
                    json!({"type": "upper_confidence_bound", "beta": beta})
                }
                acquisition => {
                    return Err(format!(
                        "an acquisition the package doesn't know: {acquisition:?}"
                    ));
                }
            }),
            "initial_points" => Ok(json!(bo.initial_points())),
            "batch" => Ok(json!(bo.batch())),
            "fantasy" => Ok(json!(match bo.fantasy() {
                bo::Fantasy::KrigingBeliever => "believer",
                bo::Fantasy::ConstantLiar(bo::Lie::Min) => "liar-min",
                bo::Fantasy::ConstantLiar(bo::Lie::Mean) => "liar-mean",
                bo::Fantasy::ConstantLiar(bo::Lie::Max) => "liar-max",
                fantasy => {
                    return Err(format!("a fantasy the package doesn't know: {fantasy:?}"));
                }
            })),
            "constraints" => Ok(json!(bo.constraints())),
            _ => Err(unknown(name)),
        }
    }

    fn set(&self, bo: &mut Bo<R>, name: &str, value: &str) -> Result<()> {
        match name {
            "acquisition" => {
                let acquisition: config::Acquisition = parse(name, value)?;
                setting(bo.set_acquisition(crate::model::acquisition(acquisition)))
            }
            "batch" => setting(bo.set_batch(parse(name, value)?)),
            "fantasy" => {
                let fantasy: config::BoFantasy = parse(name, value)?;
                bo.set_fantasy(crate::run::bo_fantasy(fantasy));
                Ok(())
            }
            _ => Err(unknown(name)),
        }
    }

    fn model(&self, bo: &Bo<R>) -> Option<PyGaussianProcess> {
        let genes = bo.representation().genome_len();
        bo.model()
            .map(|model| PyGaussianProcess::new(model.clone(), genes))
    }

    fn acquisition(&self, bo: &Bo<R>, points: ArrayView2<'_, f64>) -> Result<Vec<f64>> {
        at_points(bo, points, |bo, point| bo.acquisition_at(point))
    }

    fn feasibility(&self, bo: &Bo<R>, points: ArrayView2<'_, f64>) -> Result<Vec<f64>> {
        at_points(bo, points, |bo, point| {
            bo.probability_of_feasibility_at(point)
        })
    }
}

// `f` of the model at `points`, a point per row: an error without a model
fn at_points<R: bo::Space>(
    bo: &Bo<R>,
    points: ArrayView2<'_, f64>,
    f: impl Fn(&Bo<R>, &[f64]) -> Option<f64>,
) -> Result<Vec<f64>> {
    let genes = bo.representation().genome_len();
    if points.ncols() != genes {
        return Err(format!(
            "points is a point per row, a value per gene of the genome's {genes}, not of shape              ({}, {})",
            points.nrows(),
            points.ncols()
        ));
    }
    let mut point = vec![0.0; genes];
    let mut values = Vec::with_capacity(points.nrows());
    for row in points.rows() {
        for (x, &value) in point.iter_mut().zip(row.iter()) {
            *x = value;
        }
        match f(bo, &point) {
            Some(value) => values.push(value),
            None => {
                return Err("no model yet: the first point after the initial design has its                             model"
                    .to_string());
            }
        }
    }
    Ok(values)
}

/// Where the algorithm is while the control runs, and its settings.
pub struct Slot<A, S> {
    algorithm: Mutex<Option<A>>,
    settings: S,
}

impl<A, S> Slot<A, S> {
    pub fn new(settings: S) -> Self {
        Self {
            algorithm: Mutex::new(None),
            settings,
        }
    }

    /// Moves the algorithm in, for the control's call.
    pub fn fill(&self, algorithm: A) {
        *self
            .algorithm
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(algorithm);
    }

    /// Moves the algorithm out, after the call.
    pub fn empty(&self) -> Option<A> {
        self.algorithm
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    // `change` with the algorithm, or a RuntimeError out of the control's call
    fn with<T>(&self, change: impl FnOnce(&mut A, &S) -> Result<T>) -> PyResult<T> {
        let mut algorithm = self
            .algorithm
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let algorithm = algorithm.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err(
                "the running algorithm can be changed only during its control call",
            )
        })?;
        change(algorithm, &self.settings).map_err(PyValueError::new_err)
    }
}

/// A slot of any algorithm, for the handle.
trait AnySlot: Send + Sync {
    fn get(&self, name: &str) -> PyResult<String>;
    fn set(&self, name: &str, value: &str) -> PyResult<()>;
    fn reevaluate(&self) -> PyResult<()>;
    fn model(&self) -> PyResult<Option<PyGaussianProcess>>;
    fn acquisition(&self, points: ArrayView2<'_, f64>) -> PyResult<Vec<f64>>;
    fn feasibility(&self, points: ArrayView2<'_, f64>) -> PyResult<Vec<f64>>;
}

impl<A, S> AnySlot for Slot<A, S>
where
    A: Reevaluate + Send,
    S: Settings<A>,
{
    fn get(&self, name: &str) -> PyResult<String> {
        self.with(|algorithm, settings| settings.get(algorithm, name))
            .map(|value| value.to_string())
    }

    fn set(&self, name: &str, value: &str) -> PyResult<()> {
        self.with(|algorithm, settings| settings.set(algorithm, name, value))
    }

    fn reevaluate(&self) -> PyResult<()> {
        self.with(|algorithm, _| setting(algorithm.reevaluate()))
    }

    fn model(&self) -> PyResult<Option<PyGaussianProcess>> {
        self.with(|algorithm, settings| Ok(settings.model(algorithm)))
    }

    fn acquisition(&self, points: ArrayView2<'_, f64>) -> PyResult<Vec<f64>> {
        self.with(|algorithm, settings| settings.acquisition(algorithm, points))
    }

    fn feasibility(&self, points: ArrayView2<'_, f64>) -> PyResult<Vec<f64>> {
        self.with(|algorithm, settings| settings.feasibility(algorithm, points))
    }
}

/// The running algorithm, for a control: valid during the control's call only. The Python
/// package wraps it in a class per algorithm, e.g. `RunningGa`.
#[pyclass(frozen, module = "genoxide._genoxide")]
pub struct Running {
    slot: Arc<dyn AnySlot>,
}

impl Running {
    pub fn new<A, S>(slot: Arc<Slot<A, S>>) -> Self
    where
        A: Reevaluate + Send + 'static,
        S: Settings<A>,
    {
        Self { slot }
    }
}

#[pymethods]
impl Running {
    /// The current value of a setting, as JSON.
    fn get(&self, name: &str) -> PyResult<String> {
        self.slot.get(name)
    }

    /// Changes a setting to `value`, JSON, with the algorithm's validating setter.
    fn set(&self, name: &str, value: &str) -> PyResult<()> {
        self.slot.set(name, value)
    }

    /// Marks what the algorithm keeps for evaluation by the next generation's ask.
    fn reevaluate(&self) -> PyResult<()> {
        self.slot.reevaluate()
    }

    /// The surrogate model that chose the last point, or None.
    fn model(&self) -> PyResult<Option<PyGaussianProcess>> {
        self.slot.model()
    }

    /// The acquisition function at `points`, a point per row.
    fn acquisition<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.slot.acquisition(points.as_array())?;
        Ok(PyArray1::from_vec(py, values))
    }

    /// The probability of feasibility at `points`, a point per row.
    fn feasibility<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.slot.feasibility(points.as_array())?;
        Ok(PyArray1::from_vec(py, values))
    }
}

/// A continuation: the wrapped method's settings, and the current stage, read-only.
pub struct ContinuationSettings<S>(pub S);

impl<A, S> Settings<Continuation<A>> for ContinuationSettings<S>
where
    A: Continue,
    S: Settings<A>,
{
    fn get(&self, continuation: &Continuation<A>, name: &str) -> Result<Value> {
        match name {
            "stage" => Ok(json!(continuation.stage())),
            _ => self.0.get(continuation.algorithm(), name),
        }
    }

    fn set(&self, continuation: &mut Continuation<A>, name: &str, value: &str) -> Result<()> {
        self.0.set(continuation.algorithm_mut(), name, value)
    }
}
