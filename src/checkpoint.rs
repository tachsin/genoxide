//! Checkpoints: an algorithm saved during a run, to resume the run later with exactly the results
//! it would have had without the interruption.
//!
//! A checkpoint is a small header (the genoxide version), the algorithm's state in a compact
//! binary format that stores every `f64` exactly (NaN and infinities too), and a checksum. It
//! resumes with the same genoxide version that saved it; a corrupted or truncated file, or one from
//! another version, is an [`Error::Checkpoint`], never a wrong run.
//!
//! [`Engine::checkpoint_every`](crate::Engine::checkpoint_every) saves the algorithm every few
//! generations and when the run stops. To resume, load it and run it in a new engine with the same
//! fitness function and stop conditions: generations and evaluations continue from the
//! checkpoint (the time of a [`Stop::time`](crate::engine::Stop::time) starts again).
//!
//! ```
//! use genoxide::checkpoint;
//! use genoxide::prelude::*;
//!
//! type OneMax = Ga<Binary, Tournament, UniformCrossover, BitFlip>;
//!
//! let path = std::env::temp_dir().join("genoxide-doc-one-max.ckpt");
//! let one_max = |genome: &Bits| genome.count_ones() as f64;
//! let ga: OneMax = Ga::builder(Binary::new(100)?)
//!     .population_size(50)
//!     .select(Tournament::new(3)?)
//!     .crossover(UniformCrossover::new())
//!     .mutate(BitFlip::per_gene(0.01)?)
//!     .seed(7)
//!     .build()?;
//! // the first 20 generations, saved every 10 and at the end
//! Engine::new(ga, one_max)
//!     .stop_when(Stop::generations(20))
//!     .checkpoint_every(10, |ga| checkpoint::save_file(ga, &path))
//!     .run()?;
//!
//! // later, maybe in another process: 30 more
//! let ga: OneMax = checkpoint::load_file(&path)?;
//! assert_eq!(ga.generation(), 20);
//! let outcome = Engine::new(ga, one_max).stop_when(Stop::generations(50)).run()?;
//! assert_eq!(outcome.generations(), 50);
//! # std::fs::remove_file(&path).unwrap();
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! Save between a [`tell`](crate::algorithm::Algorithm::tell) and the next
//! [`ask`](crate::algorithm::Algorithm::ask), as the engine does. Every algorithm, genome,
//! representation, operator and observer of genoxide also implements `serde`'s `Serialize` and
//! `Deserialize`, for other formats; deserializing validates fitness values, scores, genomes and
//! representations. JSON can't store NaN or infinities, which invalid fitness values and some
//! algorithms (e.g. crowding distances) use: prefer a binary format, or this module.

use crate::{Error, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"genoxide";
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Writes a checkpoint of `algorithm` to `writer`.
///
/// # Errors
///
/// [`Error::Checkpoint`] if it can't be serialized or written.
pub fn save<A: Serialize>(algorithm: &A, mut writer: impl Write) -> Result<()> {
    let payload = postcard::to_stdvec(algorithm)
        .map_err(|error| checkpoint_error(format!("can't serialize the algorithm: {error}")))?;
    let mut bytes = Vec::with_capacity(MAGIC.len() + 1 + VERSION.len() + 8 + payload.len() + 8);
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION.len() as u8);
    bytes.extend_from_slice(VERSION.as_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(&checksum(&payload).to_le_bytes());
    writer
        .write_all(&bytes)
        .and_then(|()| writer.flush())
        .map_err(|error| checkpoint_error(format!("can't write: {error}")))
}

/// Reads a checkpoint written by [`save`] from `reader`. The type `A` must be the type of the
/// saved algorithm, including its representation and operators.
///
/// # Errors
///
/// [`Error::Checkpoint`] if it can't be read, isn't a genoxide checkpoint, comes from another
/// genoxide version, is corrupted or truncated, or doesn't hold an `A`.
pub fn load<A: DeserializeOwned>(mut reader: impl Read) -> Result<A> {
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| checkpoint_error(format!("can't read: {error}")))?;
    let mut rest = bytes.as_slice();
    if take(&mut rest, MAGIC.len()) != Some(MAGIC) {
        return Err(checkpoint_error("not a genoxide checkpoint".to_string()));
    }
    let version = take(&mut rest, 1)
        .and_then(|len| take(&mut rest, usize::from(len[0])))
        .ok_or_else(truncated)?;
    if version != VERSION.as_bytes() {
        return Err(checkpoint_error(format!(
            "saved by genoxide {}, but this is genoxide {VERSION}: a checkpoint resumes with the \
             version that saved it",
            String::from_utf8_lossy(version)
        )));
    }
    let len = take(&mut rest, 8).ok_or_else(truncated)?;
    let len = u64::from_le_bytes(len.try_into().map_err(|_| truncated())?);
    let payload = usize::try_from(len)
        .ok()
        .and_then(|len| take(&mut rest, len))
        .ok_or_else(truncated)?;
    let expected = take(&mut rest, 8).ok_or_else(truncated)?;
    if expected != checksum(payload).to_le_bytes() || !rest.is_empty() {
        return Err(checkpoint_error(
            "corrupted: the checksum doesn't match".to_string(),
        ));
    }
    match postcard::take_from_bytes(payload) {
        Ok((algorithm, [])) => Ok(algorithm),
        Ok(_) => Err(checkpoint_error(
            "doesn't hold an algorithm of this type: bytes left over".to_string(),
        )),
        Err(error) => Err(checkpoint_error(format!(
            "doesn't hold an algorithm of this type: {error}"
        ))),
    }
}

/// Writes a checkpoint of `algorithm` to the file at `path`, atomically: to a temporary file
/// next to it (the same path, with `.tmp` added), synced to disk and then renamed over it, so a
/// crash while saving keeps the previous checkpoint.
///
/// # Errors
///
/// [`Error::Checkpoint`] if it can't be serialized or written.
pub fn save_file<A: Serialize>(algorithm: &A, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let written = (|| {
        let file = File::create(&temporary)
            .map_err(|error| checkpoint_error(format!("can't create {temporary:?}: {error}")))?;
        save(algorithm, BufWriter::new(&file))?;
        file.sync_all()
            .map_err(|error| checkpoint_error(format!("can't sync {temporary:?}: {error}")))?;
        fs::rename(&temporary, path)
            .map_err(|error| checkpoint_error(format!("can't replace {path:?}: {error}")))
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written
}

/// Reads a checkpoint written by [`save_file`] or [`save`] from the file at `path`.
///
/// # Errors
///
/// As [`load`], and if the file can't be opened.
pub fn load_file<A: DeserializeOwned>(path: impl AsRef<Path>) -> Result<A> {
    let path = path.as_ref();
    let file = File::open(path)
        .map_err(|error| checkpoint_error(format!("can't open {path:?}: {error}")))?;
    load(BufReader::new(file))
}

fn checkpoint_error(reason: String) -> Error {
    Error::Checkpoint { reason }
}

fn truncated() -> Error {
    checkpoint_error("truncated".to_string())
}

// the first `len` bytes of `bytes`, which keeps the rest; `None` if there are fewer
fn take<'a>(bytes: &mut &'a [u8], len: usize) -> Option<&'a [u8]> {
    if bytes.len() < len {
        return None;
    }
    let (first, rest) = bytes.split_at(len);
    *bytes = rest;
    Some(first)
}

// 64-bit FNV-1a, to detect accidental corruption
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}
