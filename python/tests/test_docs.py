"""The code in the documentation runs: the README's Python blocks and the examples in the
docstrings of the package and its submodules."""

import pathlib
import re
import textwrap

import genoxide as gx

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def test_the_readme_code_runs():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks
    # later blocks use what earlier ones define
    namespace = {}
    for block in blocks:
        exec(compile(block, str(README), "exec"), namespace)


def docstring_examples(module):
    """The literal blocks after "::" in the module's docstring, each up to the first line that
    isn't indented."""
    examples = []
    for after in module.__doc__.split("::\n")[1:]:
        lines = []
        for line in after.splitlines():
            if line and not line.startswith(" "):
                break
            lines.append(line)
        examples.append(textwrap.dedent("\n".join(lines)))
    return examples


def test_the_module_docstring_example_runs():
    example = docstring_examples(gx)[0]
    assert "gx.Ga(" in example
    exec(compile(example, "genoxide.__doc__", "exec"), {})


def test_the_submodule_docstring_examples_run():
    for module in (gx.problems, gx.indicators):
        # later examples use what earlier ones define
        namespace = {}
        for example in docstring_examples(module):
            assert "gx." in example
            exec(compile(example, f"{module.__name__}.__doc__", "exec"), namespace)
