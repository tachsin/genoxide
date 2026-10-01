"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the best value over the evaluations of SHADE followed by L-BFGS-B, and of
SHADE alone. The Rust example writes the same file."""

import json
import math
import os

# the lines of the plot
SERIES = ["SHADE, then L-BFGS-B", "SHADE alone"]
# a frame every this many generations of SHADE
EVERY = 4


def write_runs(alone, global_, local):
    """Writes the trace, if GENOXIDE_TRACE names a file, from ``(evaluations, best value)`` after
    each generation of SHADE alone, whose first ``global_`` evaluations are the global search, and
    of L-BFGS-B, which starts after them."""
    path = os.environ.get("GENOXIDE_TRACE")
    if not path:
        return
    polished = local[-1][1] if local else math.nan
    frames = []

    def frame(evaluations, pipeline, shade):
        frames.append(
            {
                "generation": len(frames),
                "evaluations": evaluations,
                "best": pipeline,
                "state": {"values": dict(zip(SERIES, [pipeline, shade]))},
            }
        )

    for k, (evaluations, value) in enumerate(alone):
        start = evaluations == global_
        if k % EVERY == 0 or k + 1 == len(alone) or start:
            frame(evaluations, value if evaluations <= global_ else polished, value)
        if start:
            # L-BFGS-B's rounds, from SHADE's best: SHADE alone hasn't moved yet
            for more, polishing in local:
                frame(global_ + more, polishing, value)
    settings = {
        "format": 1,
        "example": "polish",
        "objective": "minimize",
        "x_label": "evaluations",
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
