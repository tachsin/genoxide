# Contributing to genoxide

Thanks for your interest! genoxide is at the design stage, so ideas and API feedback are as welcome as code. Please open an issue first for anything larger than a small fix.

## Pull requests

- **One change per PR, with tests.** Operators get property tests: validity, bounds, exact rates, and no no-op mutations (a picked gene always changes).
- **The PR title is the changelog entry.** PRs are squash merged, and the title becomes the commit message and the line in the changelog. Use [Conventional Commits](https://www.conventionalcommits.org/) with a short sentence as the subject:

  | Type | Use for | Changelog section |
  |---|---|---|
  | `feat` | new functionality | Added |
  | `fix` | bug fixes | Fixed |
  | `perf` | performance improvements | Performance |
  | `refactor` | internal changes | Changed |
  | `docs` | documentation | Documentation |
  | `test`, `ci`, `build`, `chore` | everything else | not listed |

  Add `!` after the type for a breaking change, e.g. `feat!: rename Ga::run to Ga::solve`, and explain the migration in the PR description.
- **Link the issue.** Put `Fixes #123` (or `Part of #123`) in the description. The changelog links the PR, and the PR links the issue.
- **Before pushing, run:** `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.

## Versioning and releases

genoxide follows [Semantic Versioning](https://semver.org/). Before 1.0, Cargo's rules for `0.x` apply:

| Change | Before 1.0 | From 1.0 |
|---|---|---|
| Breaking change to the public API | minor: 0.1.x → 0.2.0 | major: 1.x → 2.0.0 |
| Different results for the same seed and settings | minor: 0.1.x → 0.2.0 | major: 1.x → 2.0.0 |
| New functionality, backwards compatible | patch: 0.1.0 → 0.1.1 | minor: 1.0 → 1.1.0 |
| Bug fix | patch | patch |

Reproducibility is part of the API: a seeded run gives the same results in every patch release. A change to the random choices, even an equally good one, is marked breaking (`feat!:`, `perf!:`, …).

Releases are automated with [release-plz](https://release-plz.dev/):

1. **Release PR.** After every merge to `main`, release-plz opens or updates a release PR. It contains the next version and the new CHANGELOG.md section, generated from the PR titles.
2. **Breaking-change check.** [cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks) compares the public API with the previous release. A breaking change without a breaking version bump is caught before it's released.
3. **Release.** Merging the release PR tags the version and creates the GitHub release with the same notes. From 0.1, it also publishes to crates.io.

Nothing is released by accident: a release only happens when a maintainer merges the release PR.

Every minor release (0.6.0, 0.7.0, …) is benchmarked before it's released, and the charts, [docs/benchmarks/](docs/benchmarks/) and the README's findings are updated from it. Patch releases (0.x.y) aren't benchmarked again.

- **What's rerun:** only what changed. genoxide and its Python package are rerun every time. Another library is rerun only when its pinned version changes. `python run.py --update <the last results> --libraries <the changed ones>` keeps the other libraries' results.
- **Full reruns:** all libraries are rerun, which takes about 6 hours, when something changes everyone's numbers: the scenarios, the fitness functions or the budgets, or the machine, its operating system or a toolchain (Rust, Python, Java, Julia).
- **While it runs:** nothing else may run on the machine while it measures times.

The Python package in `python/` is a member of genoxide's Cargo workspace, with genoxide's version (`[workspace.package]` in `Cargo.toml`) and the workspace's `Cargo.lock`. Its changes are genoxide's changes: a commit that touches only `python/` bumps genoxide's version in the release PR and appears in genoxide's changelog. After a release to crates.io, the release workflow builds the wheels for Linux, macOS and Windows, tests them, and publishes them to PyPI once a maintainer approves the upload in the `pypi` environment.
