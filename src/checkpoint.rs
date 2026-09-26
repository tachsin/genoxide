//! Checkpoints: an algorithm saved during a run, to resume the run later with exactly the results
//! it would have had without the interruption.
//!
//! This module needs the `serde` feature.
//!
//! A checkpoint is a small header (the genoxide version and the algorithm's type), the algorithm's
//! state in a compact binary format that stores every `f64` exactly (NaN and infinities too), and
//! a checksum. It resumes with the same genoxide version that saved it, as the same type; a
//! corrupted or truncated file, one from another version, or one of another type is an
//! [`Error::Checkpoint`].
//!
//! Load only checkpoints you trust, like the program that saved them: the checksum detects
//! accidental damage, not tampering, and the algorithm's state isn't validated again. A crafted
//! checkpoint can make a run panic or loop, though never break memory safety: genoxide has no
//! unsafe code.
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
//! [`ask`](crate::algorithm::Algorithm::ask), as the engine does. Observers aren't part of a
//! checkpoint: a resumed run's statistics and hall of fame start empty.
//!
//! Every algorithm, genome, representation and operator of genoxide, and the statistics and hall
//! of fame, also implement `serde`'s `Serialize` and `Deserialize`, for other formats.
//! Deserializing checks fitness values, scores, genomes (a permutation is one, bits fit their
//! length) and representations (valid bounds) on their own, not whether they fit together.
//! JSON can't store NaN or infinities, which invalid fitness values and some algorithms (e.g.
//! crowding distances) use: prefer a binary format, or this module.

use crate::{Error, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::any::type_name;
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
    let kind = type_name::<A>().as_bytes();
    let kind_len = u16::try_from(kind.len())
        .map_err(|_| checkpoint_error(format!("the type name is too long: {}", kind.len())))?;
    let mut bytes = Vec::with_capacity(
        MAGIC.len() + 1 + VERSION.len() + 2 + kind.len() + 8 + payload.len() + 8,
    );
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION.len() as u8);
    bytes.extend_from_slice(VERSION.as_bytes());
    bytes.extend_from_slice(&kind_len.to_le_bytes());
    bytes.extend_from_slice(kind);
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&payload);
    // everything after the magic bytes
    let sum = checksum(&bytes[MAGIC.len()..]);
    bytes.extend_from_slice(&sum.to_le_bytes());
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
/// [`Error::Checkpoint`] if it can't be read, isn't a genoxide checkpoint, is corrupted or
/// truncated, comes from another genoxide version, or holds another type than `A`.
pub fn load<A: DeserializeOwned>(mut reader: impl Read) -> Result<A> {
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| checkpoint_error(format!("can't read: {error}")))?;
    let mut rest = bytes.as_slice();
    if take(&mut rest, MAGIC.len()) != Some(MAGIC) {
        return Err(checkpoint_error("not a genoxide checkpoint".to_string()));
    }
    // the checksum first, over everything between the magic bytes and it
    let summed = rest.len().checked_sub(8).ok_or_else(truncated)?;
    let (content, sum) = rest.split_at(summed);
    if sum != checksum(content).to_le_bytes() {
        return Err(checkpoint_error(
            "corrupted or truncated: the checksum doesn't match".to_string(),
        ));
    }
    let mut rest = content;
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
    let kind_len = take(&mut rest, 2).ok_or_else(truncated)?;
    let kind_len = u16::from_le_bytes([kind_len[0], kind_len[1]]);
    let kind = take(&mut rest, usize::from(kind_len)).ok_or_else(truncated)?;
    if kind != type_name::<A>().as_bytes() {
        return Err(checkpoint_error(format!(
            "holds a {}, not a {}",
            String::from_utf8_lossy(kind),
            type_name::<A>()
        )));
    }
    let len = take(&mut rest, 8).ok_or_else(truncated)?;
    let len = u64::from_le_bytes(len.try_into().map_err(|_| truncated())?);
    if usize::try_from(len) != Ok(rest.len()) {
        return Err(truncated());
    }
    let payload = rest;
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
        let file = File::create(&temporary).map_err(|error| {
            checkpoint_error(format!(
                "can't create {}: {error}",
                Path::new(&temporary).display()
            ))
        })?;
        save(algorithm, BufWriter::new(&file))?;
        file.sync_all().map_err(|error| {
            checkpoint_error(format!(
                "can't sync {}: {error}",
                Path::new(&temporary).display()
            ))
        })?;
        fs::rename(&temporary, path)
            .map_err(|error| checkpoint_error(format!("can't replace {}: {error}", path.display())))
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
        .map_err(|error| checkpoint_error(format!("can't open {}: {error}", path.display())))?;
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
