from collections.abc import Callable
from typing import Any

__version__: str

def run(
    config: str,
    fitness: Callable[[Any], Any],
    batch: bool = False,
    parallel: bool = False,
) -> dict[str, Any]: ...
