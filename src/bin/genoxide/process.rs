//! Fitness programs: long-lived processes, one per worker, that read a genome per line on stdin
//! and write its fitness per line on stdout.

use genoxide::engine::FitnessFunction;
use genoxide::genome::{Bits, Integers, Order, Reals};
use genoxide::multi::MultiFitnessFunction;
use std::fmt::Write as _;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// A genome as a line of text for a fitness program, and as JSON for the result.
pub trait Genes {
    /// The genes, separated by spaces.
    fn write_genes(&self, line: &mut String);

    /// The genes as a JSON array.
    fn to_json(&self) -> serde_json::Value;
}

impl Genes for Bits {
    fn write_genes(&self, line: &mut String) {
        for (index, bit) in self.iter().enumerate() {
            if index > 0 {
                line.push(' ');
            }
            line.push(if bit { '1' } else { '0' });
        }
    }

    fn to_json(&self) -> serde_json::Value {
        self.iter().map(u8::from).collect()
    }
}

// whole numbers
fn write_numbers<T: std::fmt::Display>(numbers: &[T], line: &mut String) {
    for (index, number) in numbers.iter().enumerate() {
        if index > 0 {
            line.push(' ');
        }
        let _ = write!(line, "{number}");
    }
}

impl Genes for Reals {
    // the shortest text that reads back as the same value, with an exponent when that's shorter
    fn write_genes(&self, line: &mut String) {
        for (index, number) in self.iter().enumerate() {
            if index > 0 {
                line.push(' ');
            }
            let _ = write!(line, "{number:?}");
        }
    }

    fn to_json(&self) -> serde_json::Value {
        self.iter().copied().collect()
    }
}

impl Genes for Integers {
    fn write_genes(&self, line: &mut String) {
        write_numbers(self, line);
    }

    fn to_json(&self) -> serde_json::Value {
        self.iter().copied().collect()
    }
}

impl Genes for Order {
    fn write_genes(&self, line: &mut String) {
        write_numbers(self, line);
    }

    fn to_json(&self) -> serde_json::Value {
        self.iter().copied().collect()
    }
}

struct Process {
    child: Child,
    stdin: Option<BufWriter<ChildStdin>>,
    // the lines of the program's stdout, read by a thread of their own so that a wait for an
    // answer can end; disconnected at the end of the output
    lines: Receiver<io::Result<String>>,
    line: String,
}

// reads the lines of `stdout` into `lines` until its end, an error, or the pool is gone
fn read_lines(stdout: ChildStdout, lines: &Sender<io::Result<String>>) {
    let mut stdout = BufReader::new(stdout);
    loop {
        let mut line = String::new();
        match stdout.read_line(&mut line) {
            Ok(0) => return,
            Ok(_) => {
                if lines.send(Ok(line)).is_err() {
                    return;
                }
            }
            Err(error) => {
                let _ = lines.send(Err(error));
                return;
            }
        }
    }
}

/// The longest wait for an answer, and the setting it comes from.
#[derive(Clone, Copy, Debug)]
pub struct Timeout {
    pub limit: Duration,
    pub setting: &'static str,
}

/// The fitness programs, one per worker. The first failure (a program that exits, writes
/// something that isn't a fitness or more lines than genomes, or doesn't answer in time) is
/// kept, sets the abort flag, and makes every later evaluation NaN without asking a program.
pub struct Pool {
    command: String,
    processes: Vec<Mutex<Process>>,
    free: Mutex<Vec<usize>>,
    available: Condvar,
    failure: Mutex<Option<String>>,
    abort: Arc<AtomicBool>,
    timeout: Option<Timeout>,
}

impl Pool {
    /// Starts `workers` copies of `command`, in `directory`, each answer awaited at most
    /// `timeout`.
    pub fn start(
        command: &[String],
        directory: &Path,
        workers: usize,
        abort: Arc<AtomicBool>,
        timeout: Option<Timeout>,
    ) -> Result<Self, String> {
        let (program, arguments) = command.split_first().ok_or("`fitness.command` is empty")?;
        // absolute: a relative program path joined to a relative directory would be resolved
        // from inside that directory again on Unix, after `current_dir`
        let directory = if directory.as_os_str().is_empty() {
            directory.to_path_buf()
        } else {
            std::path::absolute(directory).map_err(|error| {
                format!(
                    "can't find the run file's directory {}: {error}",
                    directory.display()
                )
            })?
        };
        let directory = directory.as_path();
        let mut processes = Vec::with_capacity(workers);
        for _ in 0..workers {
            // on Windows, a program path is looked up before changing directory
            let program = Path::new(program);
            let program = if program.is_relative() && program.components().count() > 1 {
                directory.join(program)
            } else {
                program.to_path_buf()
            };
            let mut command_line = Command::new(program);
            if !directory.as_os_str().is_empty() {
                command_line.current_dir(directory);
            }
            let mut child = command_line
                .args(arguments)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|error| format!("can't start `{}`: {error}", command.join(" ")))?;
            let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
                return Err(format!("can't talk to `{}`", command.join(" ")));
            };
            let (sender, lines) = mpsc::channel();
            std::thread::Builder::new()
                .name("fitness output".to_string())
                .spawn(move || read_lines(stdout, &sender))
                .map_err(|error| format!("can't start a thread: {error}"))?;
            processes.push(Mutex::new(Process {
                child,
                stdin: Some(BufWriter::new(stdin)),
                lines,
                line: String::new(),
            }));
        }
        Ok(Self {
            command: command.join(" "),
            free: Mutex::new((0..workers).rev().collect()),
            processes,
            available: Condvar::new(),
            failure: Mutex::new(None),
            abort,
            timeout,
        })
    }

    /// The first failure, if any.
    pub fn failure(&self) -> Option<String> {
        self.failure.lock().ok().and_then(|failure| failure.clone())
    }

    // the objective values and the constraint violation of a genome, NaN after a failure
    fn evaluate<G: Genes>(&self, genome: &G, values: &mut [f64]) -> f64 {
        if self.abort.load(Ordering::Relaxed) && self.failure().is_some() {
            values.fill(f64::NAN);
            return 0.0;
        }
        let index = self.checkout();
        let result = self.ask(index, genome, values);
        self.checkin(index);
        match result {
            Ok(violation) => violation,
            Err(failure) => {
                self.fail(failure);
                values.fill(f64::NAN);
                0.0
            }
        }
    }

    // keeps the first failure and stops the run
    fn fail(&self, failure: String) {
        if let Ok(mut first) = self.failure.lock() {
            first.get_or_insert(failure);
        }
        self.abort.store(true, Ordering::Relaxed);
    }

    // a line one too many: each answer after the extra one would be taken for the next genome's.
    // `line` is where it showed, not necessarily the extra one, e.g. the last answer after a
    // banner
    fn extra_line(&self, line: &str) -> String {
        format!(
            "`{}` wrote more lines than genomes (`{}` was one too many), so fitness values would belong to the wrong genomes: write one line per genome, and anything else to stderr",
            self.command,
            line.trim_end()
        )
    }

    // writes a genome to program `index` and reads its answer
    fn ask<G: Genes>(&self, index: usize, genome: &G, values: &mut [f64]) -> Result<f64, String> {
        let mut process = self.processes[index]
            .lock()
            .map_err(|_| "a worker panicked".to_string())?;
        let process = &mut *process;
        // a line before the genome is written answers nothing
        match process.lines.try_recv() {
            Ok(Ok(line)) => return Err(self.extra_line(&line)),
            Ok(Err(error)) => {
                return Err(format!("can't read from `{}`: {error}", self.command));
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {}
        }
        process.line.clear();
        genome.write_genes(&mut process.line);
        process.line.push('\n');
        let stdin = process
            .stdin
            .as_mut()
            .ok_or("the fitness program was closed")?;
        stdin
            .write_all(process.line.as_bytes())
            .and_then(|()| stdin.flush())
            .map_err(|error| {
                format!(
                    "`{}` exited or closed its input instead of reading a genome: {error}",
                    self.command
                )
            })?;
        let answer = match self.timeout {
            Some(timeout) => process.lines.recv_timeout(timeout.limit),
            None => process
                .lines
                .recv()
                .map_err(|_| RecvTimeoutError::Disconnected),
        };
        process.line = match answer {
            Ok(Ok(line)) => line,
            Ok(Err(error)) => {
                return Err(format!("can't read from `{}`: {error}", self.command));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(format!(
                    "`{}` exited instead of writing a fitness",
                    self.command
                ));
            }
            Err(RecvTimeoutError::Timeout) => {
                let (limit, setting) = self
                    .timeout
                    .map(|timeout| (timeout.limit, timeout.setting))
                    .unwrap_or_default();
                return Err(format!(
                    "`{}` didn't answer a genome within {limit:?} (`{setting}`)",
                    self.command
                ));
            }
        };
        parse(&process.line, values).map_err(|reason| {
            format!(
                "`{}` wrote `{}`: {reason}",
                self.command,
                process.line.trim_end()
            )
        })
    }

    fn checkout(&self) -> usize {
        let mut free = match self.free.lock() {
            Ok(free) => free,
            Err(poisoned) => poisoned.into_inner(),
        };
        loop {
            if let Some(index) = free.pop() {
                return index;
            }
            free = match self.available.wait(free) {
                Ok(free) => free,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
    }

    fn checkin(&self, index: usize) {
        let mut free = match self.free.lock() {
            Ok(free) => free,
            Err(poisoned) => poisoned.into_inner(),
        };
        free.push(index);
        self.available.notify_one();
    }
}

/// Parses an answer: the objective values, then optionally the constraint violation. Returns
/// the violation, 0 if there's none.
pub fn parse(line: &str, values: &mut [f64]) -> Result<f64, String> {
    let mut numbers = Vec::with_capacity(values.len() + 1);
    for word in line.split_whitespace() {
        let number: f64 = word
            .parse()
            .map_err(|_| format!("`{word}` isn't a number"))?;
        numbers.push(number);
    }
    let expected = values.len();
    match numbers.len() {
        len if len == expected => {
            values.copy_from_slice(&numbers);
            Ok(0.0)
        }
        len if len == expected + 1 => {
            values.copy_from_slice(&numbers[..expected]);
            let violation = numbers[expected];
            if violation < 0.0 {
                return Err(format!(
                    "the constraint violation, the number after the {expected} objective value{}, is negative",
                    if expected == 1 { "" } else { "s" }
                ));
            }
            Ok(violation)
        }
        len => Err(format!(
            "expected {expected} number{} (the objective value{}), and optionally a constraint violation, got {len}",
            if expected == 1 { "" } else { "s" },
            if expected == 1 { "" } else { "s" },
        )),
    }
}

impl Pool {
    /// Closes the programs' input, gives them a second to exit, then kills them. The first
    /// failure, if any: also a program that wrote more lines than genomes, found in what it
    /// writes until it exits.
    pub fn close(&self) -> Option<String> {
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut extra = None;
        for process in &self.processes {
            let mut process = match process.lock() {
                Ok(process) => process,
                Err(poisoned) => poisoned.into_inner(),
            };
            if process.stdin.take().is_none() {
                // closed already
                continue;
            }
            // the rest of the output, until the program closes it
            loop {
                let wait = deadline.saturating_duration_since(Instant::now());
                match process.lines.recv_timeout(wait) {
                    Ok(Ok(line)) => {
                        extra.get_or_insert_with(|| self.extra_line(&line));
                    }
                    Ok(Err(_))
                    | Err(RecvTimeoutError::Disconnected | RecvTimeoutError::Timeout) => {
                        break;
                    }
                }
            }
        }
        for process in &self.processes {
            let mut process = match process.lock() {
                Ok(process) => process,
                Err(poisoned) => poisoned.into_inner(),
            };
            let child = &mut process.child;
            while matches!(child.try_wait(), Ok(None)) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if matches!(child.try_wait(), Ok(None)) {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
        if let Some(extra) = extra {
            self.fail(extra);
        }
        self.failure()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.close();
    }
}

/// A single-objective fitness function asking the pool.
pub struct Single<'a>(pub &'a Pool);

impl<G: Genes> FitnessFunction<G> for Single<'_> {
    type Output = (f64, f64);

    fn evaluate(&self, genome: &G) -> (f64, f64) {
        let mut value = [0.0];
        let violation = self.0.evaluate(genome, &mut value);
        (value[0], violation)
    }
}

/// A multi-objective fitness function asking the pool.
pub struct Multi<'a>(pub &'a Pool);

impl<G: Genes, const M: usize> MultiFitnessFunction<G, M> for Multi<'_> {
    type Output = ([f64; M], f64);

    fn evaluate(&self, genome: &G) -> ([f64; M], f64) {
        let mut values = [0.0; M];
        let violation = self.0.evaluate(genome, &mut values);
        (values, violation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_parse() {
        let mut one = [0.0];
        assert_eq!(parse("1.5\n", &mut one), Ok(0.0));
        assert_eq!(one, [1.5]);
        assert_eq!(parse("  -2 0.25 ", &mut one), Ok(0.25));
        assert_eq!(one, [-2.0]);
        assert!(parse("nan", &mut one).is_ok() && one[0].is_nan());
        assert_eq!(parse("inf", &mut one), Ok(0.0));
        assert!(parse("", &mut one).unwrap_err().contains("got 0"));
        assert!(
            parse("1 2 3", &mut one)
                .unwrap_err()
                .contains("expected 1 number")
        );
        assert!(
            parse("one", &mut one)
                .unwrap_err()
                .contains("`one` isn't a number")
        );
        let mut two = [0.0; 2];
        assert_eq!(parse("1 2", &mut two), Ok(0.0));
        assert_eq!(parse("1 2 3", &mut two), Ok(3.0));
        assert!(
            parse("1", &mut two)
                .unwrap_err()
                .contains("expected 2 numbers")
        );
    }

    #[test]
    fn genes_are_written_exactly() {
        let mut line = String::new();
        Bits::from_iter([true, false, true]).write_genes(&mut line);
        assert_eq!(line, "1 0 1");
        line.clear();
        let reals = Reals::from(vec![0.1, -2.0, 1e-300]);
        reals.write_genes(&mut line);
        assert_eq!(line, "0.1 -2.0 1e-300");
        // they read back as the same values
        let read: Vec<f64> = line.split(' ').map(|word| word.parse().unwrap()).collect();
        assert_eq!(read, reals.to_vec());
    }
}
