from collections.abc import Callable
from typing import Any

import numpy as np

__version__: str

def run(
    config: str,
    fitness: Callable[[Any], Any],
    batch: bool = False,
    parallel: bool = False,
    on_generation: Callable[[int, int, float, Any], bool] | None = None,
) -> dict[str, Any]: ...
def das_dennis(objectives: int, divisions: int) -> np.ndarray: ...
