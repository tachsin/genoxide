"""The code in the documentation runs: the README's Python blocks and the examples in the
docstrings of the package, its classes and its submodules."""

import inspect
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


def docstring_examples(documented):
    """The literal blocks after "::" in the docstring of a module or a class, each up to the first
    line that isn't indented."""
    examples = []
    for after in inspect.cleandoc(documented.__doc__).split("::\n")[1:]:
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
    modules = (gx.problems, gx.problems.cec2006, gx.problems.engineering, gx.neat)
    modules += (gx.gp, gx.gp.regression, gx.gp.regression.problems, gx.gp.boolean)
    for module in modules + (gx.problems.multi_engineering, gx.indicators, gx.model.gp):
        # later examples use what earlier ones define
        namespace = {}
        for example in docstring_examples(module):
            assert "gx." in example
            exec(compile(example, f"{module.__name__}.__doc__", "exec"), namespace)


def test_the_class_docstring_examples_run():
    classes = [getattr(gx, name) for name in gx.__all__]
    documented = [
        cls for cls in classes if isinstance(cls, type) and "::\n" in (inspect.getdoc(cls) or "")
    ]
    assert gx.NelderMead in documented
    for cls in documented:
        for example in docstring_examples(cls):
            assert "gx." in example
            exec(compile(example, f"{cls.__name__}.__doc__", "exec"), {})
