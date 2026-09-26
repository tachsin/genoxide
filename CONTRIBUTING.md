# Contributing to genoxide

genoxide is alpha, pre-1.0: ideas and API feedback are as welcome as code. Open an issue first for anything larger than a small fix.

## Pull requests

- **One change per PR, with tests.** Operators get property tests: validity, bounds, exact rates, and no no-op mutations (a picked gene always changes).
- **The PR title is the changelog entry.** PRs are squash merged. The title becomes the commit message and the changelog line. Use [Conventional Commits](https://www.conventionalcommits.org/), with a short sentence as the subject:

  | Type | Use for | Changelog section |
  |---|---|---|
  | `feat` | new functionality | Added |
  | `fix` | bug fixes | Fixed |
  | `perf` | performance improvements | Performance |
  | `refactor` | internal changes | Changed |
  | `docs` | documentation | Documentation |
  | `test`, `ci`, `build`, `chore` | everything else | not listed |

  Add `!` after the type for a breaking change, e.g. `feat!: rename Ga::run to Ga::solve`. Explain the migration in the PR description.
- **Link the issue:** `Fixes #123` or `Part of #123` in the description.
- **Before pushing, run:** `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.

## Versioning

genoxide follows [Semantic Versioning](https://semver.org/). Before 1.0, Cargo's rules for `0.x` apply:

| Change | Before 1.0 | From 1.0 |
|---|---|---|
| Breaking change to the public API | minor: 0.1.x → 0.2.0 | major: 1.x → 2.0.0 |
| Different results for the same seed and settings | minor: 0.1.x → 0.2.0 | major: 1.x → 2.0.0 |
| New functionality, backwards compatible | patch: 0.1.0 → 0.1.1 | minor: 1.0 → 1.1.0 |
| Bug fix | patch | patch |

Reproducibility is part of the API: a seeded run gives the same results in every patch release. A change to the random choices is breaking, even an equally good one (`feat!:`, `perf!:`, …).

## Releases

[release-plz](https://release-plz.dev/) automates releases:

1. **Release PR.** After every merge to `main`, release-plz opens or updates a release PR. It holds the next version and the new CHANGELOG.md section, generated from the PR titles.
2. **Breaking-change check.** [cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks) compares the public API with the previous release. It catches a breaking change without a breaking version bump.
3. **Release.** A maintainer merges the release PR. That tags the version, creates the GitHub release with the same notes, and publishes to crates.io.

## The Python package

`python/` is a member of genoxide's Cargo workspace. It shares genoxide's version (`[workspace.package]` in `Cargo.toml`) and the workspace's `Cargo.lock`. A commit that touches only `python/` bumps genoxide's version in the release PR and appears in genoxide's changelog.

After a release to crates.io, the release workflow builds the wheels for Linux, macOS and Windows and tests them. It publishes them to PyPI once a maintainer approves the upload in the `pypi` environment.

## Benchmarks

Every minor release (0.6.0, 0.7.0, …) is benchmarked before it's released, and [docs/benchmarks/](docs/benchmarks/) is updated from it. Patch releases aren't benchmarked again.

- **Partial rerun:** genoxide and its Python package are always rerun. Another library is rerun only when its pinned version changes. `python run.py --update <the last results> --libraries <the changed ones>` keeps the other libraries' results.
- **Full rerun** (about 6 hours): when the scenarios, the fitness functions, the budgets, the machine, its operating system or a toolchain (Rust, Python, Java, Julia) change.
- Nothing else may run on the machine during a timed run.
