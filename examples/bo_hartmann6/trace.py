"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the best value's distance above the global minimum after each round, of
the search 4 points a round and of the search one point a round, which goes on for more rounds.
The Rust example writes the same file."""

import json
import math
import os

# the lines of the plot
SERIES = ["4 points a round", "1 point a round"]


def write(batched, single):
    """Writes the trace, if GENOXIDE_TRACE names a file, from the distances above the minimum
    after each round of both searches; a search that has stopped keeps its last value."""
    path = os.environ.get("GENOXIDE_TRACE")
    if not path:
        return
    rounds = max(len(batched), len(single))

    def at(values, round_):
        return values[min(round_, len(values) - 1)]

    frames = [
        {
            "generation": round_,
            "best": at(batched, round_),
            "state": {"values": dict(zip(SERIES, [at(batched, round_), at(single, round_)]))},
        }
        for round_ in range(rounds)
    ]
    settings = {
        "format": 1,
        "example": "bo_hartmann6",
        "objective": "minimize",
        "x_label": "rounds",
        "y_label": "best value's distance above the global minimum",
        "log_y": True,
        "optimum": 0.0,
        "plot": "multi-curve",
        "problem": {"series": SERIES},
    }
    write_trace(path, settings, frames)


# ---- the same in every example's trace ----------------------------------------------------------


def write_trace(path, settings, frames):
    """Writes the settings and the frames to ``path``, a frame per line."""
    lines = ",\n".join(map(to_json, frames))
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(f'{to_json(settings)[:-1]},"frames":[\n{lines}\n]}}\n')


def to_json(value):
    """Compact JSON with sorted keys, and numbers rounded to 6 significant digits, as the Rust
    example writes it."""
    return json.dumps(rounded(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.5e}") if math.isfinite(value) else None
    return value
