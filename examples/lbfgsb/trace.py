"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: per round, the best value and the evaluations of the run with the
analytic gradient and of the run with forward differences. The Rust example writes the same file."""

import json
import math
import os

# the lines of the plot: a panel per quantity, a line per source of the gradient
SERIES = [
    "best value/analytic gradient",
    "best value/forward differences",
    "evaluations/analytic gradient",
    "evaluations/forward differences",
]


# a frame every this many rounds, and the last round of each run
EVERY = 4


def write_runs(analytic, forward):
    """Writes the trace, if GENOXIDE_TRACE names a file: ``(evaluations, best value)`` per round
    of each run, every 4 rounds and the last of each run."""
    path = os.environ.get("GENOXIDE_TRACE")
    if not path:
        return
    frames = []

    def best(run, round_):
        return run[round_][1] if round_ < len(run) else None

    def evaluations(run, round_):
        return run[round_][0] if round_ < len(run) else None

    for round_ in range(max(len(analytic), len(forward))):
        last = round_ + 1 in (len(analytic), len(forward))
        if round_ % EVERY and not last:
            continue
        values = [
            best(analytic, round_),
            best(forward, round_),
            evaluations(analytic, round_),
            evaluations(forward, round_),
        ]
        frames.append(
            {
                "generation": round_,
                "evaluations": evaluations(analytic, round_),
                "best": best(analytic, round_),
                "state": {"values": dict(zip(SERIES, values))},
            }
        )
    settings = {
        "format": 1,
        "example": "lbfgsb",
        "objective": "minimize",
        "x_label": "rounds",
        "y_label": "best value",
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
