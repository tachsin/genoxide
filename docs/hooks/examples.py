"""MkDocs hook: the example pages of the docs site, generated from the folders in examples/.

Each folder has a README.md with YAML front matter (title, category, summary, reference,
reference_url, optimum, languages, order) and a description, and the code in main.rs and main.py
(or src/main.rs, for an example that is a crate of its own). The hook makes a page per example with
the code in Rust and Python tabs, an index with a table of them, and their entries in the nav, in
the order of their `order`. The build fails for a README without valid front matter, or an example
missing from the table in examples/README.md.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

import yaml
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
GITHUB = "https://github.com/tachsin/genoxide/tree/main/examples"
CATEGORIES = {
    "binary",
    "permutation",
    "integer",
    "continuous",
    "multi-objective",
    "constrained",
    "neuroevolution",
    "engine",
}
FIELDS = {
    "title",
    "category",
    "summary",
    "reference",
    "reference_url",
    "optimum",
    "languages",
    "order",
}
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


@dataclass
class Example:
    name: str
    meta: dict
    body: str

    def source(self, language: str) -> str:
        """The path of the code in ``language``, from the root of the repository."""
        if language == "python":
            return f"examples/{self.name}/main.py"
        if (EXAMPLES / self.name / "main.rs").exists():
            return f"examples/{self.name}/main.rs"
        return f"examples/{self.name}/src/main.rs"

    def run(self, language: str) -> str:
        """The command that runs the example, from the root of the repository."""
        if language == "python":
            return f"python examples/{self.name}/main.py"
        if (EXAMPLES / self.name / "Cargo.toml").exists():
            return f"cargo run --release --manifest-path examples/{self.name}/Cargo.toml"
        return f"cargo run --release --example {self.name}"


def examples() -> list[Example]:
    """The examples, in their order."""
    found = []
    for folder in sorted(path for path in EXAMPLES.iterdir() if path.is_dir()):
        readme = folder / "README.md"
        if not readme.exists():
            raise PluginError(f"examples/{folder.name} has no README.md")
        text = readme.read_text(encoding="utf-8").replace("\r\n", "\n")
        match = FRONT_MATTER.match(text)
        if not match:
            raise PluginError(f"examples/{folder.name}/README.md has no YAML front matter")
        meta = yaml.safe_load(match.group(1))
        if not isinstance(meta, dict) or set(meta) != FIELDS:
            raise PluginError(
                f"the front matter of examples/{folder.name}/README.md has the fields "
                f"{sorted(FIELDS)}"
            )
        if meta["category"] not in CATEGORIES:
            raise PluginError(
                f"examples/{folder.name}: the category {meta['category']!r} isn't one of "
                f"{sorted(CATEGORIES)}"
            )
        languages = meta["languages"]
        if not languages or not set(languages) <= {"rust", "python"}:
            raise PluginError(f"examples/{folder.name}: languages are rust and python")
        example = Example(folder.name, meta, match.group(2).strip())
        for language in languages:
            if not (ROOT / example.source(language)).exists():
                raise PluginError(f"{example.source(language)} doesn't exist")
        found.append(example)
    return sorted(found, key=lambda example: (example.meta["order"], example.name))


def page(example: Example) -> str:
    """The page of an example: its README, facts, how to run it and the code."""
    meta = example.meta
    lines = [example.body, ""]
    reference = meta["reference"]
    if reference:
        if meta["reference_url"]:
            reference = f"[{reference}]({meta['reference_url']})"
        lines += [f"**Reference:** {reference}", ""]
    if meta["optimum"]:
        lines += [f"**Known optimum:** {meta['optimum']}", ""]
    lines += [f"**Source:** [examples/{example.name}]({GITHUB}/{example.name})", ""]
    for language in meta["languages"]:
        label, fence = {"rust": ("Rust", "rust"), "python": ("Python", "python")}[language]
        lines += [
            f'=== "{label}"',
            "",
            "    ```sh",
            f"    {example.run(language)}",
            "    ```",
            "",
            f"    ```{fence}",
            f'    --8<-- "{example.source(language)}"',
            "    ```",
            "",
        ]
    return "\n".join(lines)


def index(found: list[Example]) -> str:
    """The index of the examples: a table of them, and how to run them."""
    lines = [
        "# Examples",
        "",
        "Each example is the same program in Rust and Python, with a problem from the literature",
        "or a feature of genoxide. The code on these pages is the code in the repository's",
        f"[examples]({GITHUB}) folder, which CI runs. Choosing Rust or Python on one page selects",
        "it on every page.",
        "",
        "| Example | Category | Languages | Known optimum |",
        "|---|---|---|---|",
    ]
    for example in found:
        meta = example.meta
        languages = ", ".join(language.capitalize() for language in meta["languages"])
        optimum = meta["optimum"] or ""
        lines.append(
            f"| [{meta['title']}]({example.name}.md) | {meta['category']} | {languages} "
            f"| {optimum} |"
        )
    lines += [
        "",
        "Run a Rust example with `cargo run --release --example <name>` from the root of the",
        "repository, and a Python one with `python examples/<name>/main.py` after",
        "`pip install genoxide`.",
        "",
    ]
    return "\n".join(lines)


def on_config(config):
    """The examples' entries in the nav, after the index."""
    found = examples()
    table = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for example in found:
        if f"]({example.name}/)" not in table:
            raise PluginError(f"the table in examples/README.md lacks examples/{example.name}")
    pages = ["examples/index.md"]
    pages += [{example.meta["title"]: f"examples/{example.name}.md"} for example in found]
    for entry in config["nav"]:
        if isinstance(entry, dict) and "Examples" in entry:
            entry["Examples"] = pages
            break
    else:
        raise PluginError("mkdocs.yml's nav has no Examples section")
    return config


def on_files(files, config):
    """The generated pages."""
    found = examples()
    files.append(File.generated(config, "examples/index.md", content=index(found)))
    for example in found:
        files.append(File.generated(config, f"examples/{example.name}.md", content=page(example)))
    return files
