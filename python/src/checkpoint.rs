//! Checkpoints of a run, to resume it later with the results it would have had without the
//! interruption.
//!
//! A checkpoint is in genoxide's checkpoint format (`genoxide::checkpoint`: a header with the
//! genoxide version and the type, the state in a compact binary format that stores every `f64`
//! exactly, and a checksum), and holds the run's settings with the algorithm, as the `genoxide`
//! program's checkpoints do. The settings are the genome, the objectives and the algorithm's, as
//! the Python package describes them: they give the algorithm's type, so loading needs no type
//! from the user, and a checkpoint resumes only a run with the same settings. The fitness
//! function, the stop conditions, `batch`, `parallel`, the callbacks and where to save can
//! change.

use genoxide::checkpoint;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

// what a checkpoint holds: borrowed to save, owned when loaded, the same type either way (a
// checkpoint's header names the type)
#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "A: Serialize", deserialize = "A: DeserializeOwned"))]
struct Saved<'a, A: Clone> {
    settings: Cow<'a, str>,
    algorithm: Cow<'a, A>,
}

/// Where a run saves its checkpoints, and the checkpoint it resumes from.
pub struct Checkpoints {
    /// The run's settings: the genome, the objectives and the algorithm, as JSON.
    pub settings: String,
    /// The file to save to, and every how many generations.
    pub save: Option<(PathBuf, u64)>,
    /// The file to resume from, and its contents.
    pub resume: Option<(PathBuf, Vec<u8>)>,
}

impl Checkpoints {
    /// The algorithm of the checkpoint to resume from, or `algorithm`, built from the settings,
    /// without one.
    pub fn resume<A: Clone + DeserializeOwned>(&self, algorithm: A) -> Result<A> {
        let Some((path, bytes)) = &self.resume else {
            return Ok(algorithm);
        };
        let other = || {
            format!(
                "{} was saved by a run with other settings: resume with the genome, objectives and algorithm settings that saved it (the fitness function, the stop conditions, the callbacks and the checkpoint can change)",
                path.display()
            )
        };
        let saved: Saved<'_, A> = match checkpoint::load(bytes.as_slice()) {
            Ok(saved) => saved,
            // another algorithm, or another genome's
            Err(genoxide::Error::Checkpoint { reason }) if reason.starts_with("holds a ") => {
                return Err(other());
            }
            Err(error) => return Err(format!("can't resume from {}: {error}", path.display())),
        };
        if saved.settings != self.settings {
            return Err(other());
        }
        Ok(saved.algorithm.into_owned())
    }

    /// Saves a checkpoint of `algorithm` to `path`, atomically.
    pub fn save<A: Clone + Serialize>(&self, algorithm: &A, path: &Path) -> genoxide::Result<()> {
        let saved = Saved {
            settings: Cow::Borrowed(self.settings.as_str()),
            algorithm: Cow::Borrowed(algorithm),
        };
        checkpoint::save_file(&saved, path)
    }
}
