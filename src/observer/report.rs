//! Lines of progress, printed now and then.

use super::{Observer, Snapshot};
use crate::engine::Progress;
use crate::genome::Genome;
use crate::{Error, Result};
use std::io::{self, Write};
use std::time::Duration;

/// Prints a line of progress now and then: after the initial population (or the first generation
/// of a run that continues), then once a second by default, to stderr by default.
///
/// ```text
/// generation 0: 50 evaluations, best 29 (generation 0), 0.000 s
/// generation 212: 8530 evaluations, best 64 (generation 205), 1.001 s
/// ```
///
/// A multi-objective line has the last generation in which the front improved instead of a best
/// fitness. Writing errors are ignored: a report never stops a run.
///
/// ```
/// use genoxide::observer::Report;
/// use genoxide::prelude::*;
/// use std::time::Duration;
///
/// let ga = Ga::builder(Binary::new(64)?)
///     .population_size(50)
///     .select(Tournament::new(3)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::per_gene(1.0 / 64.0)?)
///     .seed(1)
///     .build()?;
/// Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
///     .stop_when(Stop::target(64.0))
///     .observe(Report::every(Duration::from_millis(500)))
///     .run()?;
///
/// // a multi-objective run reports from its callback
/// let mut report = Report::every_generations(10)?;
/// let nsga2 = Nsga2::builder(Real::uniform(2, 0.0..=1.0)?, [Objective::Minimize; 2])
///     .population_size(20)
///     .crossover(UniformCrossover::new())
///     .mutate(GaussianMutation::per_gene(0.5, 0.1)?)
///     .seed(1)
///     .build()?;
/// MultiEngine::new(nsga2, |x: &Reals| [x[0], 1.0 - x[0] + x[1]])
///     .stop_when(Stop::generations(30))
///     .on_generation(|snapshot| report.update(snapshot.progress()))
///     .run()?;
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Debug)]
pub struct Report<W = io::Stderr> {
    writer: W,
    every: Every,
    // the generation and time of the last line
    last: Option<(u64, Duration)>,
}

#[derive(Clone, Copy, Debug)]
enum Every {
    Time(Duration),
    Generations(u64),
}

impl Report {
    /// A line to stderr after the initial population, then once a second.
    pub fn new() -> Self {
        Self::every(Duration::from_secs(1))
    }

    /// A line to stderr after the initial population, then after the first generation to end in
    /// each new `interval` of the run (e.g. once past 1 s, 2 s, 3 s, ...), so the lines don't
    /// drift: after every generation with [`Duration::ZERO`].
    pub fn every(interval: Duration) -> Self {
        Self {
            writer: io::stderr(),
            every: Every::Time(interval),
            last: None,
        }
    }

    /// A line to stderr after the initial population and every `generations` generations
    /// (generation 0, `generations`, `2 * generations`, ...).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] if `generations` is 0.
    pub fn every_generations(generations: u64) -> Result<Self> {
        if generations == 0 {
            return Err(Error::InvalidSetting {
                setting: "every_generations",
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            writer: io::stderr(),
            every: Every::Generations(generations),
            last: None,
        })
    }
}

impl Default for Report {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: Write> Report<W> {
    /// The same report to `writer` instead, e.g. a file or a `Vec<u8>`.
    pub fn to<V: Write>(self, writer: V) -> Report<V> {
        Report {
            writer,
            every: self.every,
            last: self.last,
        }
    }

    /// The writer.
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// The writer, e.g. to read a `Vec<u8>` after the run.
    pub fn into_writer(self) -> W {
        self.writer
    }

    /// Prints a line if one is due: what an [`Observer`] does after a generation. Call it from
    /// [`MultiEngine::on_generation`](crate::multi::MultiEngine::on_generation) in a
    /// multi-objective run.
    pub fn update(&mut self, progress: &Progress) {
        let due = match (self.last, self.every) {
            (None, _) => true,
            // a new run, e.g. `Engine::run` called again: its time starts again
            (Some((_, last)), _) if progress.elapsed() < last => true,
            (Some((_, last)), Every::Time(interval)) => {
                let interval = interval.as_nanos();
                interval == 0
                    || progress.elapsed().as_nanos() / interval > last.as_nanos() / interval
            }
            (Some((last, _)), Every::Generations(generations)) => {
                progress.generation() != last && progress.generation() % generations == 0
            }
        };
        if due {
            self.last = Some((progress.generation(), progress.elapsed()));
            let _ = write_line(&mut self.writer, progress);
        }
    }
}

// one line of progress
fn write_line(writer: &mut impl Write, progress: &Progress) -> io::Result<()> {
    write!(
        writer,
        "generation {}: {} evaluations, ",
        progress.generation(),
        progress.evaluations()
    )?;
    match progress.best() {
        Some(best) => write!(
            writer,
            "best {best} (generation {})",
            progress.best_generation()
        )?,
        None => write!(
            writer,
            "front improved in generation {}",
            progress.best_generation()
        )?,
    }
    writeln!(writer, ", {:.3} s", progress.elapsed().as_secs_f64())
}

impl<G: Genome, W: Write> Observer<G> for Report<W> {
    fn observe(&mut self, snapshot: &Snapshot<'_, G>) {
        self.update(snapshot.progress());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective;
    use crate::engine::Progress;

    fn lines(report: Report<Vec<u8>>) -> Vec<String> {
        String::from_utf8(report.into_writer())
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn every_generations() {
        let mut report = Report::every_generations(3).unwrap().to(Vec::new());
        for generation in 0..8 {
            report.update(&Progress::for_test(generation, Objective::Maximize));
        }
        // the same generation twice is one line
        report.update(&Progress::for_test(6, Objective::Maximize));
        let lines = lines(report);
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("generation 0: "));
        assert!(lines[1].starts_with("generation 3: "));
        assert!(lines[2].starts_with("generation 6: "));
        assert!(Report::every_generations(0).is_err());
    }

    #[test]
    fn every_interval() {
        let progress = |generation: u64, millis: u64| {
            let mut progress = Progress::for_test(generation, Objective::Maximize);
            progress.elapsed = Duration::from_millis(millis);
            progress
        };
        let mut report = Report::every(Duration::from_millis(100)).to(Vec::new());
        for (generation, millis) in [(0, 5), (1, 60), (2, 104), (3, 150), (4, 204), (5, 400)] {
            report.update(&progress(generation, millis));
        }
        let generations: Vec<String> = lines(report)
            .iter()
            .map(|line| line.split(':').next().unwrap().to_string())
            .collect();
        assert_eq!(
            generations,
            [
                "generation 0",
                "generation 2",
                "generation 4",
                "generation 5"
            ]
        );
        // a run that continues starts with a line, and its own schedule
        let mut report = Report::every(Duration::from_millis(100)).to(Vec::new());
        for (generation, millis) in [(0, 0), (5, 400), (6, 10), (7, 60), (8, 120)] {
            report.update(&progress(generation, millis));
        }
        let generations: Vec<String> = lines(report)
            .iter()
            .map(|line| line.split(':').next().unwrap().to_string())
            .collect();
        assert_eq!(
            generations,
            [
                "generation 0",
                "generation 5",
                "generation 6",
                "generation 8"
            ]
        );
        // every generation
        let mut report = Report::every(Duration::ZERO).to(Vec::new());
        for generation in 0..4 {
            report.update(&progress(generation, 0));
        }
        assert_eq!(lines(report).len(), 4);
    }

    #[test]
    fn line_format() {
        let mut progress = Progress::for_test(12, Objective::Maximize);
        progress.evaluations = 650;
        progress.elapsed = Duration::from_millis(1500);
        progress.best_generation = 10;
        let mut report = Report::every(Duration::ZERO).to(Vec::new());
        report.update(&progress);
        progress.best = Some(crate::Fitness::new(31.5));
        progress.generation = 13;
        report.update(&progress);
        assert_eq!(
            lines(report),
            [
                "generation 12: 650 evaluations, front improved in generation 10, 1.500 s",
                "generation 13: 650 evaluations, best 31.5 (generation 10), 1.500 s",
            ]
        );
    }
}
