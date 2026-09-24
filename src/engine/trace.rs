// events of the `tracing` feature, with the target `genoxide`; nothing without it

use super::{Progress, StopReason};

// the span of a run, entered until dropped
pub(crate) struct RunSpan {
    #[cfg(feature = "tracing")]
    _span: tracing::span::EnteredSpan,
}

// enters the span of a run of `A`, at the info level
#[cfg_attr(not(feature = "tracing"), allow(clippy::extra_unused_type_parameters))]
pub(crate) fn run<A>() -> RunSpan {
    RunSpan {
        #[cfg(feature = "tracing")]
        _span:
            tracing::info_span!(target: "genoxide", "run", algorithm = std::any::type_name::<A>())
                .entered(),
    }
}

// a generation, at the debug level; `front` is the size of a multi-objective front
#[cfg_attr(not(feature = "tracing"), allow(unused_variables))]
pub(crate) fn generation(progress: &Progress, front: Option<usize>) {
    #[cfg(feature = "tracing")]
    tracing::debug!(
        target: "genoxide",
        generation = progress.generation(),
        evaluations = progress.evaluations(),
        best = progress.best().map(tracing::field::display),
        best_generation = progress.best_generation(),
        front,
        elapsed = ?progress.elapsed(),
        "generation"
    );
}

// the end of a run, at the info level
#[cfg_attr(not(feature = "tracing"), allow(unused_variables))]
pub(crate) fn finished(progress: &Progress, reason: StopReason, front: Option<usize>) {
    #[cfg(feature = "tracing")]
    tracing::info!(
        target: "genoxide",
        ?reason,
        generations = progress.generation(),
        evaluations = progress.evaluations(),
        best = progress.best().map(tracing::field::display),
        best_generation = progress.best_generation(),
        front,
        elapsed = ?progress.elapsed(),
        "run finished"
    );
}
