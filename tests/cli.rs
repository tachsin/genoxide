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
        (
            "de",
            "population_size = 30\nstrategy = \"rand1\"\ncontrol = { f = 0.5, cr = 0.9 }\nrestarts = \"never\"",
        ),
        (
            "de",
            "strategy = { p = 0.1, archive = 1.0 }\ncontrol = { c = 0.1 }\nrestarts = { tolerance = 1e-12, patience = 100 }",
        ),
        (
            "de",
            "strategy = \"best1\"\ncontrol = { min_f = 0.5, max_f = 1.0, cr = 0.9 }",
        ),
        (
            "de",
            "strategy = { max_p = 0.2, archive = 0.0 }\ncontrol = { memory = 6 }",
        ),
        ("pso", "population_size = 30\nring = 1"),
        ("cmaes", "covariance = \"diagonal\"\nrestarts = \"ipop\""),
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
    // "full" is the default; a diagonal covariance matrix is another run
    let full = CMAES.replace(
        "type = \"cmaes\"",
        "type = \"cmaes\"\ncovariance = \"full\"",
    );
    let mut full = run(&directory, "full.toml", &full, &[]).unwrap();
    full["seconds"] = Value::Null;
    assert_eq!(full, first);
    let diagonal = CMAES.replace(
        "type = \"cmaes\"",
        "type = \"cmaes\"\ncovariance = \"diagonal\"",
    );
    let diagonal = run(&directory, "diagonal.toml", &diagonal, &[]).unwrap();
    assert_eq!(diagonal["stop_reason"], "target");
    assert_ne!(diagonal["genome"], first["genome"]);
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
        error.contains("other genome, objectives or algorithm settings"),
        "{error}"
    );
    // and so is another algorithm
    let other = text(40).replace("type = \"ga\"", "type = \"steady-ga\"");
    let error = run(&directory, "part.toml", &other, &["--resume"]).unwrap_err();
    assert!(error.contains("holds a"), "{error}");
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_checkpoint_in_a_missing_directory_is_found_by_check() {
    let directory = directory("checkpoint-directory");
    let text = |path: &str| format!("{CMAES}[checkpoint]\npath = \"{path}\"\nevery = 5\n");
    let missing = text("missing/run.ckpt");
    let error = check(&directory, &missing).unwrap_err();
    assert!(
        error.contains("`checkpoint.path`: the directory") && error.contains("doesn't exist"),
        "{error}"
    );
    // a run stops before its first generation, not at its first save
    let error = run(&directory, "missing.toml", &missing, &[]).unwrap_err();
    assert!(error.contains("`checkpoint.path`"), "{error}");
    assert!(!directory.join("missing").exists());
    // a file isn't a directory
    std::fs::write(directory.join("file"), "").unwrap();
    let error = check(&directory, &text("file/run.ckpt")).unwrap_err();
    assert!(error.contains("`checkpoint.path`"), "{error}");
    // a directory that exists, relative to the run file, and the run file's own
    std::fs::create_dir(directory.join("saves")).unwrap();
    assert_eq!(check(&directory, &text("saves/run.ckpt")), Ok(()));
    assert_eq!(check(&directory, &text("run.ckpt")), Ok(()));
    let result = run(&directory, "saves.toml", &text("saves/run.ckpt"), &[]).unwrap();
    assert!(result["fitness"].is_number(), "{result}");
    assert!(directory.join("saves/run.ckpt").exists());
    std::fs::remove_dir_all(&directory).unwrap();
}

// a script in `directory` as a fitness command: `unix` for sh, `windows` for cmd
fn script(directory: &Path, name: &str, unix: &str, windows: &str) -> String {
    if cfg!(windows) {
        std::fs::write(directory.join(format!("{name}.cmd")), windows).unwrap();
        // `.\` too: cmd needn't look in the current directory
        format!("command = [\"cmd\", \"/c\", \".\\\\{name}.cmd\"]")
    } else {
        std::fs::write(directory.join(format!("{name}.sh")), unix).unwrap();
        format!("command = [\"sh\", \"{name}.sh\"]")
    }
}

#[test]
fn a_fitness_program_that_writes_extra_lines_is_an_error() {
    let directory = directory("extra");
    // a line before its answers: each answer would be taken for the next genome's
    let banner = script(
        &directory,
        "banner",
        &format!("echo 42\nexec '{GENOXIDE}' fitness sphere\n"),
        &format!("@echo 42\r\n@\"{GENOXIDE}\" fitness sphere\r\n"),
    );
    let text = CMAES.replace("builtin = \"sphere\"", &banner);
    let error = run(&directory, "banner.toml", &text, &[]).unwrap_err();
    assert!(error.contains("wrote more lines than genomes"), "{error}");
    assert!(error.contains("banner"), "{error}");

    // a line after every answer, from a program that answers each genome on its own
    #[cfg(unix)]
    {
        let chatty = script(
            &directory,
            "chatty",
            &format!(
                "while read -r line; do echo \"$line\" | '{GENOXIDE}' fitness sphere; echo 7; done\n"
            ),
            "",
        );
        let text = CMAES
            .replace("builtin = \"sphere\"", &chatty)
            .replace("evaluations = 50000", "evaluations = 50");
        let error = run(&directory, "chatty.toml", &text, &[]).unwrap_err();
        // found before a genome is written or when the programs close: the line it names
        // depends on which
        assert!(error.contains("wrote more lines than genomes"), "{error}");
    }
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_fitness_program_that_stops_answering_is_an_error() {
    let directory = directory("silent");
    // waits without answering, in the process that is killed: a child of its own would keep
    // genoxide's stderr open after that. Its stdout stays the pipe genoxide reads: closed, it
    // would be a program that exited, not one that doesn't answer
    let silent = script(
        &directory,
        "silent",
        "exec sleep 20 2>/dev/null\n",
        // the second waits for a genome that never comes
        "@set /p first=\r\n@set /p second=\r\n",
    );
    let text = CMAES.replace("builtin = \"sphere\"", &silent);
    // at most `stop.time` for an answer
    let started = std::time::Instant::now();
    let error = run(
        &directory,
        "time.toml",
        &text.replace("evaluations = 50000", "time = \"1s\""),
        &[],
    )
    .unwrap_err();
    assert!(
        error.contains("didn't answer a genome within 1s (`stop.time`)"),
        "{error}"
    );
    assert!(error.contains("silent"), "{error}");
    // or `fitness.timeout`, which comes first
    let error = run(
        &directory,
        "timeout.toml",
        &text
            .replace("workers = 2", "workers = 2\ntimeout = \"500ms\"")
            .replace("evaluations = 50000", "time = \"1h\""),
        &[],
    )
    .unwrap_err();
    assert!(
        error.contains("didn't answer a genome within 500ms (`fitness.timeout`)"),
        "{error}"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(15));
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
        &CMAES.replace(
            "type = \"cmaes\"",
            "type = \"cmaes\"\ncovariance = \"sparse\"",
        ),
        "unknown variant `sparse`, expected `full` or `diagonal`",
    );
    expect(
        &CMAES.replace("type = \"cmaes\"", "type = \"pso\""),
        "`algorithm.population_size` is needed",
    );
    expect(
        &CMAES.replace("[stop]\ntarget = 1e-8\nevaluations = 50000", "[stop]"),
        "`stop` needs at least one",
    );
    expect(
        &CMAES.replace("workers = 2", "workers = 2\ntimeout = \"0s\""),
        "`fitness.timeout` must be longer than 0",
    );
    expect(
        &CMAES.replace("builtin = \"sphere\"", "builtin = \"nope\""),
        "no built-in fitness `nope`",
    );
    // lengths above 2^24 are an error, not a failed allocation
    for kind in ["real", "integer"] {
        let text = CMAES
            .replace("type = \"real\"", &format!("type = \"{kind}\""))
            .replace("bounds = [-5.0, 5.0]", "bounds = [-5, 5]")
            .replace("type = \"cmaes\"", "type = \"de\"");
        expect(
            &text.replace("length = 5", "length = 1099511627776"),
            "at most 16777216 (2^24) genes, got 1099511627776",
        );
        expect(
            &text.replace("length = 5", "length = 33554432"),
            "at most 16777216 (2^24) genes",
        );
    }
    // bounds of the wrong kind say what bounds are
    expect(
        &CMAES.replace("bounds = [-5.0, 5.0]", "bounds = [\"a\", 5.0]"),
        "`bounds` is [low, high] for every gene, or a list of [low, high], one per gene, of numbers",
    );
    expect(
        &CMAES
            .replace("type = \"real\"", "type = \"integer\"")
            .replace("type = \"cmaes\"", "type = \"de\""),
        "of whole numbers",
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
    // differential evolution's settings: their forms, then genoxide's checks
    let de =
        |settings: &str| CMAES.replace("type = \"cmaes\"", &format!("type = \"de\"\n{settings}"));
    assert_eq!(
        check(
            &directory,
            &de("strategy = \"rand1\"\ncontrol = { f = 0.5, cr = 0.9 }\nrestarts = \"never\"")
        ),
        Ok(())
    );
    expect(
        &de("strategy = \"rand2\""),
        "`algorithm.strategy`: \"rand1\", \"best1\", { p, archive } or { max_p, archive }",
    );
    expect(
        &de("strategy = { p = 0.1 }"),
        "`algorithm.strategy`: \"rand1\", \"best1\"",
    );
    expect(
        &de("control = { f = 0.5 }"),
        "`algorithm.control`: { f, cr }, { min_f, max_f, cr }, { c } or { memory }",
    );
    expect(
        &de("restarts = \"ipop\""),
        "`algorithm.restarts`: \"never\" or { tolerance, patience }",
    );
    expect(&de("strategy = { p = 0.0, archive = 1.0 }"), "`p`");
    expect(&de("control = { f = 0.5, cr = 1.5 }"), "`cr`");
    expect(&de("control = { memory = 0 }"), "`memory`");
    expect(
        &de("restarts = { tolerance = 1e-12, patience = 0 }"),
        "`restarts`",
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

#[test]
fn review_fixes() {
    let directory = directory("review");
    // extra keys on operators without settings are errors
    let ga = CMAES.replace(
        "type = \"cmaes\"",
        "type = \"ga\"\npopulation_size = 10\nselect = { type = \"roulette\", size = 3 }\ncrossover = { type = \"uniform\" }\nmutate = { type = \"polynomial\", rate = 0.1, eta = 20.0 }",
    );
    let error = check(&directory, &ga).unwrap_err();
    assert!(error.contains("unknown field `size`"), "{error}");
    // and each operator's error says which operator it is
    assert!(error.contains("`algorithm.select`"), "{error}");
    let missing = ga.replace(
        "select = { type = \"roulette\", size = 3 }",
        "select = { type = \"tournament\" }",
    );
    let error = check(&directory, &missing).unwrap_err();
    assert!(
        error.contains("`algorithm.select`: missing field `size`"),
        "{error}"
    );
    let error = check(
        &directory,
        &ga.replace(
            "mutate = { type = \"polynomial\", rate = 0.1, eta = 20.0 }",
            "mutate = { type = \"polynomial\", rate = 0.1 }",
        ),
    )
    .unwrap_err();
    assert!(error.contains("`algorithm.mutate`"), "{error}");
    let error = check(
        &directory,
        &ga.replace(
            "{ type = \"roulette\", size = 3 }",
            "{ type = \"roulette\" }",
        )
        .replace("{ type = \"uniform\" }", "{ type = \"uniform\", eta = 3 }"),
    )
    .unwrap_err();
    assert!(error.contains("unknown field `eta`"), "{error}");
    // checkpoints every 0 generations
    let error = check(
        &directory,
        &format!("{CMAES}[checkpoint]\npath = \"x.ckpt\"\nevery = 0\n"),
    )
    .unwrap_err();
    assert!(error.contains("`checkpoint.every`"), "{error}");

    // a failed program leaves no checkpoint of its NaN evaluations
    let failing = format!(
        "{}[checkpoint]\npath = \"failed.ckpt\"\nevery = 1\n",
        CMAES
            .replace("[\"minimize\"]", "[\"minimize\", \"minimize\"]")
            .replace(
                "type = \"cmaes\"",
                "type = \"nsga2\"\npopulation_size = 10\ncrossover = { type = \"uniform\" }\nmutate = { type = \"polynomial\", rate = 0.1, eta = 20.0 }",
            )
            .replace("target = 1e-8\n", "")
    );
    let error = run(&directory, "failing.toml", &failing, &[]).unwrap_err();
    assert!(error.contains("expected 2 numbers"), "{error}");
    assert!(!directory.join("failed.ckpt").exists());

    // resuming with other objectives is an error
    let one_max = |objective: &str, generations: u64| {
        format!(
            "report = \"off\"\n[genome]\ntype = \"binary\"\nlength = 16\n[fitness]\nbuiltin = \"one-max\"\nobjectives = [\"{objective}\"]\n[algorithm]\ntype = \"ga\"\npopulation_size = 10\nseed = 1\nselect = {{ type = \"tournament\", size = 2 }}\ncrossover = {{ type = \"uniform\" }}\nmutate = {{ type = \"bit-flip\", rate = 0.1 }}\n[stop]\ngenerations = {generations}\n[checkpoint]\npath = \"one-max.ckpt\"\nevery = 5\n"
        )
    };
    run(&directory, "one-max.toml", &one_max("maximize", 5), &[]).unwrap();
    let error = run(
        &directory,
        "one-max.toml",
        &one_max("minimize", 10),
        &["--resume"],
    )
    .unwrap_err();
    assert!(error.contains("other genome, objectives"), "{error}");
    run(
        &directory,
        "one-max.toml",
        &one_max("maximize", 10),
        &["--resume"],
    )
    .unwrap();

    // a relative program path is relative to the run file
    let programs = directory.join("programs");
    std::fs::create_dir_all(&programs).unwrap();
    let copy = programs.join(Path::new(GENOXIDE).file_name().unwrap());
    std::fs::copy(GENOXIDE, &copy).unwrap();
    let relative = format!("./programs/{}", copy.file_name().unwrap().to_string_lossy());
    let text = CMAES.replace(
        "builtin = \"sphere\"",
        &format!("command = [{relative:?}, \"fitness\", \"sphere\"]"),
    );
    let result = run(&directory, "relative.toml", &text, &[]).unwrap();
    assert_eq!(result["stop_reason"], "target");
    // also with the run file given by a relative path, from another directory
    let output = Command::new(GENOXIDE)
        .current_dir(directory.parent().unwrap())
        .args([
            "run",
            &Path::new(directory.file_name().unwrap())
                .join("relative.toml")
                .to_string_lossy(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // infinite values are text in the result; null is only for an invalid fitness
    let infinite = CMAES
        .replace("bounds = [-5.0, 5.0]", "bounds = [1e200, 1e300]")
        .replace("[\"minimize\"]", "[\"maximize\"]")
        .replace("type = \"cmaes\"", "type = \"de\"\npopulation_size = 10")
        .replace("target = 1e-8\nevaluations = 50000", "generations = 2");
    let result = run(&directory, "infinite.toml", &infinite, &[]).unwrap();
    assert_eq!(result["fitness"], "inf");
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn stop_and_report_mistakes_are_found_by_check() {
    let directory = directory("messages");
    let expect = |text: &str, message: &str| {
        let error = check(&directory, text).unwrap_err();
        assert!(error.contains(message), "{message}: {error}");
    };
    expect(
        &CMAES.replace("evaluations = 50000", "stagnation = 0"),
        "`stop.stagnation` must be at least 1",
    );
    expect(
        &CMAES.replace("target = 1e-8", "target = nan"),
        "`stop.target` is NaN",
    );
    expect(
        &CMAES.replace("report = \"off\"", "report = 0"),
        "`report` must be at least 1",
    );
    expect(
        &CMAES.replace("report = \"off\"", "report = -3"),
        "a duration like \"2s\", a number of generations, or \"off\"",
    );
    // a second number for one objective is a violation, which can't be negative
    let text = CMAES
        .replace("builtin = \"sphere\"", "builtin = \"zdt1\"")
        .replace("bounds = [-5.0, 5.0]", "bounds = [0.0, 5.0]");
    let error = run(&directory, "zdt1.toml", &text, &[]).unwrap_err();
    assert!(error.contains("fitness zdt1"), "{error}");
    assert!(error.contains("the constraint violation"), "{error}");
    std::fs::remove_dir_all(&directory).unwrap();
}

const NELDER_MEAD: &str = r#"
report = "off"
[genome]
type = "real"
length = 4
bounds = [-5.0, 5.0]
[fitness]
builtin = "rosenbrock"
objectives = ["minimize"]
workers = 2
[algorithm]
type = "nelder-mead"
seed = 1
[stop]
evaluations = 100000
"#;

// a run file of `NELDER_MEAD` with these settings
fn nelder_mead(settings: &str) -> String {
    NELDER_MEAD.replace(
        "type = \"nelder-mead\"",
        &format!("type = \"nelder-mead\"\n{settings}"),
    )
}

// a result without its time, to compare runs
fn untimed(mut result: Value) -> Value {
    result["seconds"] = Value::Null;
    result
}

#[test]
fn nelder_mead_runs_until_it_converges() {
    let directory = directory("nelder-mead");
    let run = |text: &str| untimed(run(&directory, "run.toml", text, &[]).unwrap());
    let default = run(NELDER_MEAD);
    assert_eq!(default["stop_reason"], "converged");
    assert!(default["fitness"].as_f64().unwrap() < 1e-15, "{default}");
    assert!(default["evaluations"].as_u64().unwrap() < 100_000);
    for gene in default["genome"].as_array().unwrap() {
        assert!((gene.as_f64().unwrap() - 1.0).abs() < 1e-8, "{default}");
    }
    // the settings' defaults
    let explicit = run(&nelder_mead(
        "coefficients = \"adaptive\"\ninitial_step = 0.1\ntolerance = 1e-10\nspeculative = false",
    ));
    assert_eq!(explicit, default);
    // the standard coefficients, by name or as a table
    let standard = run(&nelder_mead("coefficients = \"standard\""));
    assert_eq!(standard["stop_reason"], "converged");
    assert_ne!(standard["evaluations"], default["evaluations"]);
    let table = run(&nelder_mead(
        "coefficients = { reflection = 1.0, expansion = 2.0, contraction = 0.5, shrink = 0.5 }",
    ));
    assert_eq!(table, standard);
    // restarts: more runs, each to convergence
    let restarts = run(&nelder_mead("restarts = 3"));
    assert_eq!(restarts["stop_reason"], "converged");
    assert!(restarts["evaluations"].as_u64().unwrap() > default["evaluations"].as_u64().unwrap());
    assert!(restarts["fitness"].as_f64().unwrap() < 1e-15, "{restarts}");
    // speculative: the same kind of search, in fewer rounds of more evaluations
    let speculative = run(&nelder_mead("speculative = true"));
    assert_eq!(speculative["stop_reason"], "converged");
    assert!(
        speculative["fitness"].as_f64().unwrap() < 1e-15,
        "{speculative}"
    );
    assert!(
        speculative["evaluations"].as_u64().unwrap()
            > speculative["generations"].as_u64().unwrap() * 2
    );
    // a stop condition can still come first
    let short = run(&NELDER_MEAD.replace("evaluations = 100000", "generations = 10"));
    assert_eq!(short["stop_reason"], "generations");
    assert_eq!(short["generations"], 10);
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn nelder_mead_resumes_from_checkpoints() {
    let directory = directory("nelder-mead-resume");
    let text = |generations: u64| {
        format!(
            "{}[checkpoint]\npath = \"run.ckpt\"\nevery = 25\n",
            nelder_mead("restarts = 2\nspeculative = true").replace(
                "evaluations = 100000",
                &format!("generations = {generations}")
            )
        )
    };
    let whole = run(&directory, "whole.toml", &text(100_000), &[]).unwrap();
    assert_eq!(whole["stop_reason"], "converged");
    let first = run(&directory, "part.toml", &text(60), &[]).unwrap();
    assert_eq!(first["generations"], 60);
    let resumed = run(&directory, "part.toml", &text(100_000), &["--resume"]).unwrap();
    assert_eq!(untimed(resumed), untimed(whole));
    // other settings are an error
    let other = text(100_000).replace("restarts = 2", "restarts = 3");
    let error = run(&directory, "part.toml", &other, &["--resume"]).unwrap_err();
    assert!(
        error.contains("other genome, objectives or algorithm settings"),
        "{error}"
    );
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn nelder_mead_settings_are_checked() {
    let directory = directory("nelder-mead-check");
    let expect = |text: &str, message: &str| {
        let error = check(&directory, text).unwrap_err();
        assert!(error.contains(message), "{message}: {error}");
    };
    assert_eq!(
        check(
            &directory,
            &nelder_mead(
                "coefficients = { reflection = 1.0, expansion = 2.5, contraction = 0.4, shrink = 0.6 }\ninitial_step = 0.5\ntolerance = 1e-6\nrestarts = 1\nspeculative = true"
            )
        ),
        Ok(())
    );
    expect(
        &nelder_mead("restarts = 0"),
        "`algorithm.restarts` must be at least 1; leave it out for none",
    );
    expect(
        &nelder_mead("tolerance = 0.1"),
        "invalid setting `tolerance`: must be greater than 0 and smaller than the initial step 0.1",
    );
    expect(
        &nelder_mead("initial_step = 0.001\ntolerance = 0.01"),
        "invalid setting `tolerance`",
    );
    expect(
        &nelder_mead("tolerance = 0.0"),
        "invalid setting `tolerance`",
    );
    expect(
        &nelder_mead("initial_step = 1.5"),
        "invalid setting `initial_step`",
    );
    expect(
        &nelder_mead("coefficients = \"golden\""),
        "`algorithm.coefficients`: \"adaptive\", \"standard\" or { reflection, expansion, contraction, shrink }",
    );
    expect(
        &nelder_mead("coefficients = { reflection = 1.0, expansion = 2.0 }"),
        "`algorithm.coefficients`",
    );
    expect(
        &nelder_mead(
            "coefficients = { reflection = 1.0, expansion = 2.0, contraction = 0.5, shrink = 1.5 }",
        ),
        "invalid setting `coefficients`",
    );
    expect(&nelder_mead("restarts = -1"), "expected u64");
    expect(&nelder_mead("color = 1"), "unknown field `color`");
    expect(&nelder_mead("neighbors = 4"), "unknown field `neighbors`");
    expect(
        &NELDER_MEAD
            .replace("type = \"real\"", "type = \"binary\"")
            .replace("bounds = [-5.0, 5.0]\n", ""),
        "`nelder-mead` needs a real genome",
    );
    expect(
        &NELDER_MEAD
            .replace("type = \"real\"", "type = \"permutation\"")
            .replace("bounds = [-5.0, 5.0]\n", ""),
        "`nelder-mead` needs a real genome",
    );
    expect(
        &NELDER_MEAD.replace("[\"minimize\"]", "[\"minimize\", \"minimize\"]"),
        "use `nsga2` for several",
    );
    // every gene fixed: nothing to search
    expect(
        &NELDER_MEAD.replace("bounds = [-5.0, 5.0]", "bounds = [1.0, 1.0]"),
        "Nelder-Mead needs a gene with more than one value",
    );
    // the list of types names it
    expect(
        &NELDER_MEAD.replace("type = \"nelder-mead\"", "type = \"simplex\""),
        "nelder-mead",
    );
    expect(
        &NELDER_MEAD.replace("evaluations = 100000", ""),
        "`stop` needs at least one",
    );
    std::fs::remove_dir_all(&directory).unwrap();
}
