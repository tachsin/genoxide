"""The code in the documentation runs: the README's Python blocks and the module docstring's
example."""

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


def test_the_module_docstring_example_runs():
    # the literal block after "::", up to the first line that isn't indented
    _, after = gx.__doc__.split("::\n", 1)
    lines = []
    for line in after.splitlines():
        if line and not line.startswith(" "):
            break
        lines.append(line)
    example = textwrap.dedent("\n".join(lines))
    assert "gx.Ga(" in example
    exec(compile(example, "genoxide.__doc__", "exec"), {})
