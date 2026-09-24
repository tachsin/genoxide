//! The `genoxide` program: run files, fitness programs, results and errors.
#![cfg(feature = "cli")]

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const GENOXIDE: &str = env!("CARGO_BIN_EXE_genoxide");

// a directory of its own for each test
fn directory(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("genoxide-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn genoxide(arguments: &[&str]) -> Output {
    Command::new(GENOXIDE).args(arguments).output().unwrap()
}

// runs a run file; the result, or the error message
fn run(directory: &Path, name: &str, text: &str, extra: &[&str]) -> Result<Value, String> {
    let path = directory.join(name);
    std::fs::write(&path, text).unwrap();
    let path = path.to_string_lossy().into_owned();
    let mut arguments = vec!["run", path.as_str()];
    arguments.extend(extra);
    let output = genoxide(&arguments);
    if output.status.success() {
        Ok(serde_json::from_slice(&output.stdout).unwrap())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

fn check(directory: &Path, text: &str) -> Result<(), String> {
    let path = directory.join("check.toml");
    std::fs::write(&path, text).unwrap();
    let output = genoxide(&["check", &path.to_string_lossy()]);
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

const CMAES: &str = r#"
report = "off"
[genome]
type = "real"
length = 5
bounds = [-5.0, 5.0]
[fitness]
builtin = "sphere"
objectives = ["minimize"]
workers = 2
[algorithm]
type = "cmaes"
seed = 1
[stop]
target = 1e-8
evaluations = 50000
"#;

#[test]
fn every_kind_of_genome_and_algorithm_runs() {
    let directory = directory("kinds");
    let result = run(&directory, "cmaes.toml", CMAES, &[]).unwrap();
    assert_eq!(result["stop_reason"], "target");
    assert!(result["fitness"].as_f64().unwrap() <= 1e-8);
    assert_eq!(result["genome"].as_array().unwrap().len(), 5);

    let binary = r#"
report = "off"
[genome]
type = "binary"
length = 32
[fitness]
builtin = "one-max"
workers = 3
[algorithm]
type = "steady-ga"
population_size = 20
seed = 2
select = { type = "tournament", size = 3 }
crossover = { type = "uniform" }
mutate = { type = "bit-flip", rate = 0.03 }
[stop]
target = 32
evaluations = 20000
"#;
    let result = run(&directory, "binary.toml", binary, &[]).unwrap();
    assert_eq!(result["stop_reason"], "target");
    assert_eq!(result["genome"], Value::from(vec![1; 32]));

    let permutation = r#"
report = "off"
[genome]
type = "permutation"
length = 10
[fitness]
builtin = "inversions"
objectives = ["minimize"]
[algorithm]
type = "ga"
population_size = 30
seed = 3
select = { type = "tournament", size = 3 }
crossover = { type = "order" }
mutate = { type = "inversion" }
scheme = { type = "mu-plus-lambda", lambda = 30 }
[stop]
target = 0
generations = 1000
"#;
    let result = run(&directory, "permutation.toml", permutation, &[]).unwrap();
    assert_eq!(result["stop_reason"], "target");
    assert_eq!(result["genome"], Value::from((0..10).collect::<Vec<_>>()));

    let integer = r#"
report = "off"
[genome]
type = "integer"
bounds = [[-10, 10], [-10, 10], [0, 5]]
[fitness]
builtin = "sphere"
objectives = ["minimize"]
[algorithm]
type = "local-search"
seed = 4
neighbor = { type = "uniform", count = 1 }
neighbors = 8
[stop]
target = 0
generations = 1000
"#;
    let result = run(&directory, "integer.toml", integer, &[]).unwrap();
    assert_eq!(result["genome"], Value::from(vec![0, 0, 0]));

    for (algorithm, extra) in [
        ("de", "population_size = 30"),
        ("pso", "population_size = 30\nring = 1"),
    ] {
        let text = format!(
            "report = \"off\"\n[genome]\ntype = \"real\"\nlength = 4\nbounds = [-5.0, 5.0]\n[fitness]\nbuiltin = \"sphere\"\nobjectives = [\"minimize\"]\n[algorithm]\ntype = \"{algorithm}\"\nseed = 5\n{extra}\n[stop]\ntarget = 1e-4\ngenerations = 2000\n"
        );
        let result = run(&directory, "real.toml", &text, &[]).unwrap();
        assert_eq!(result["stop_reason"], "target", "{algorithm}");
    }

    // several objectives, in a JSON run file
    let nsga2 = r#"{
        "report": "off",
        "genome": { "type": "real", "length": 5, "bounds": [0.0, 1.0] },
        "fitness": { "builtin": "zdt1", "objectives": ["minimize", "minimize"] },
        "algorithm": {
            "type": "nsga2", "population_size": 20, "seed": 6,
            "crossover": { "type": "simulated-binary", "eta": 15.0 },
            "mutate": { "type": "polynomial", "rate": 0.2, "eta": 20.0 }
        },
        "stop": { "generations": 50 }
    }"#;
    let result = run(&directory, "nsga2.json", nsga2, &[]).unwrap();
    let front = result["front"].as_array().unwrap();
    assert!(front.len() > 5);
    for member in front {
        assert_eq!(member["objectives"].as_array().unwrap().len(), 2);
        assert_eq!(member["genome"].as_array().unwrap().len(), 5);
    }
    // no copies of a genome
    for (index, member) in front.iter().enumerate() {
        assert!(
            front[..index]
                .iter()
                .all(|other| other["genome"] != member["genome"])
        );
    }
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_seed_gives_the_same_result() {
    let directory = directory("seed");
    let mut first = run(&directory, "cmaes.toml", CMAES, &[]).unwrap();
    let mut second = run(&directory, "cmaes.toml", CMAES, &[]).unwrap();
    first["seconds"] = Value::Null;
    second["seconds"] = Value::Null;
    assert_eq!(first, second);
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn runs_resume_from_checkpoints() {
    let directory = directory("resume");
    let text = |generations: u64| {
        format!(
            "report = \"off\"\n[genome]\ntype = \"binary\"\nlength = 200\n[fitness]\nbuiltin = \"one-max\"\n[algorithm]\ntype = \"ga\"\npopulation_size = 20\nseed = 7\nselect = {{ type = \"tournament\", size = 3 }}\ncrossover = {{ type = \"uniform\" }}\nmutate = {{ type = \"bit-flip\", rate = 0.005 }}\n[stop]\ngenerations = {generations}\n[checkpoint]\npath = \"run.ckpt\"\nevery = 5\n"
        )
    };
    let whole = run(&directory, "whole.toml", &text(40), &[]).unwrap();
    let first = run(&directory, "part.toml", &text(15), &[]).unwrap();
    assert_eq!(first["generations"], 15);
    // the checkpoint is next to the run file
    assert!(directory.join("run.ckpt").exists());
    let resumed = run(&directory, "part.toml", &text(40), &["--resume"]).unwrap();
    assert_eq!(resumed["generations"], 40);
    assert_eq!(resumed["genome"], whole["genome"]);
    assert_eq!(resumed["evaluations"], whole["evaluations"]);
    // other settings are an error, even with the same types
    let other = text(40).replace("\"uniform\"", "\"one-point\"");
    let error = run(&directory, "part.toml", &other, &["--resume"]).unwrap_err();
    assert!(
        error.contains("other genome or algorithm settings"),
        "{error}"
    );
    // and so is another algorithm
    let other = text(40).replace("type = \"ga\"", "type = \"steady-ga\"");
    let error = run(&directory, "part.toml", &other, &["--resume"]).unwrap_err();
    assert!(error.contains("holds a"), "{error}");
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_failing_fitness_program_is_an_error() {
    let directory = directory("failing");
    // a program that exits at once
    let text = format!(
        "report = \"off\"\n[genome]\ntype = \"binary\"\nlength = 8\n[fitness]\ncommand = [{:?}, \"fitness\", \"no-such-function\"]\n[algorithm]\ntype = \"ga\"\npopulation_size = 10\nselect = {{ type = \"tournament\", size = 2 }}\ncrossover = {{ type = \"uniform\" }}\nmutate = {{ type = \"bit-flip\", rate = 0.1 }}\n[stop]\ngenerations = 5\n",
        GENOXIDE
    );
    let error = run(&directory, "failing.toml", &text, &[]).unwrap_err();
    // it exits before or after its input is written, so either message
    assert!(error.contains("exited"), "{error}");
    // one that writes the wrong number of values: sphere has one, two are expected
    let text = text
        .replace("no-such-function", "sphere")
        .replace(
            "[fitness]",
            "[fitness]\nobjectives = [\"minimize\", \"minimize\"]",
        )
        .replace(
            "type = \"ga\"\npopulation_size = 10\nselect = { type = \"tournament\", size = 2 }\n",
            "type = \"nsga2\"\npopulation_size = 10\n",
        );
    let error = run(&directory, "failing.toml", &text, &[]).unwrap_err();
    assert!(error.contains("expected 2 numbers"), "{error}");
    // a program that doesn't exist
    let text = CMAES.replace(
        "builtin = \"sphere\"",
        "command = [\"genoxide-no-such-program\"]",
    );
    let error = run(&directory, "missing.toml", &text, &[]).unwrap_err();
    assert!(error.contains("can't start"), "{error}");
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn run_files_are_checked() {
    let directory = directory("check");
    assert_eq!(check(&directory, CMAES), Ok(()));
    let expect = |text: &str, message: &str| {
        let error = check(&directory, text).unwrap_err();
        assert!(error.contains(message), "{message}: {error}");
    };
    expect(
        &CMAES.replace("type = \"cmaes\"", "type = \"cmaes\"\ncolor = 1"),
        "unknown field `color`",
    );
    expect(
        &CMAES.replace("type = \"cmaes\"", "type = \"de\""),
        "`algorithm.population_size` is needed",
    );
    expect(
        &CMAES.replace("[stop]\ntarget = 1e-8\nevaluations = 50000", "[stop]"),
        "`stop` needs at least one",
    );
    expect(
        &CMAES.replace("builtin = \"sphere\"", "builtin = \"nope\""),
        "no built-in fitness `nope`",
    );
    expect(
        &CMAES.replace(
            "builtin = \"sphere\"",
            "builtin = \"sphere\"\ncommand = [\"x\"]",
        ),
        "either `command`",
    );
    expect(
        &CMAES.replace("[\"minimize\"]", "[\"minimize\", \"minimize\"]"),
        "use `nsga2` for several",
    );
    expect(
        &CMAES
            .replace("type = \"real\"", "type = \"binary\"")
            .replace("bounds = [-5.0, 5.0]\n", ""),
        "`cmaes` needs a real genome",
    );
    expect(
        &CMAES.replace("length = 5\n", ""),
        "`genome.length` is needed",
    );
    let ga = CMAES.replace(
        "type = \"cmaes\"",
        "type = \"ga\"\npopulation_size = 10\nselect = { type = \"tournament\", size = 2 }\ncrossover = { type = \"order\" }\nmutate = { type = \"polynomial\", rate = 0.1, eta = 20.0 }",
    );
    expect(&ga, "crossover `order` doesn't work with real genomes");
    expect(
        &ga.replace("{ type = \"order\" }", "{ type = \"uniform\" }")
            .replace("rate = 0.1, eta", "eta"),
        "either `rate`",
    );
    expect(
        &ga.replace("{ type = \"order\" }", "{ type = \"uniform\" }")
            .replace("size = 2", "size = 0"),
        "size",
    );
    expect(
        &CMAES.replace("evaluations = 50000", "time = \"5 weeks\""),
        "unknown unit",
    );
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn the_built_in_fitness_programs_speak_the_protocol() {
    let mut child = Command::new(GENOXIDE)
        .args(["fitness", "zdt1"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "0 0 0").unwrap();
        writeln!(stdin, "1 0 0").unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert_eq!(String::from_utf8_lossy(&output.stdout), "0 1\n1 0\n");
    let list = genoxide(&["fitness"]);
    assert!(String::from_utf8_lossy(&list.stdout).contains("rastrigin"));
    let help = genoxide(&["--help"]);
    assert!(String::from_utf8_lossy(&help.stdout).contains("genoxide run <file>"));
    assert!(!genoxide(&["frobnicate"]).status.success());
}
