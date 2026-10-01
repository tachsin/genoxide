"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the front and its hypervolume, in at most 100 generations. The Rust
example writes the same file."""

import json
import math
import os

import genoxide as gx


class Trace:
    """Records the run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, problem, reference):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = Frames(100)
        self.problem, self.reference = problem, reference

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation: the front and its hypervolume."""
        front = progress.front_objectives
        volume = gx.indicators.hypervolume(front, self.reference)
        state = {"fronts": {"NSGA-II": front.tolist()}, "hypervolume": {"NSGA-II": volume}}
        self.frames.push(frame(progress, state))

    def write(self):
        """Writes the trace, if there's one."""
        if self.path:
            settings = {
                "format": 1,
                "example": "zdt2",
                "objective": ["minimize", "minimize"],
                "x_label": "generations",
                "y_label": "hypervolume",
                "log_y": False,
                "optimum": 0.5433,
                "plot": "front-2d",
                "problem": {
                    "objectives": ["f1", "f2"],
                    "true_front": self.problem.optimal_front(100).tolist(),
                    "series": ["NSGA-II"],
                },
            }
            write(self.path, settings, self.frames.to_list())


# ---- the same in every example's trace ----------------------------------------------------------


class Frames:
    """The frames of at most ``most`` generations, from the part of the run where what the page
    plots changes: the frames after the last change are left out (a run that reached its target,
    or a front that no longer moves), and the rest are spread evenly over the generations up to
    it. While the run goes, up to 8 × ``most`` frames are kept: every ``every``-th generation,
    with ``every`` doubling whenever there are that many, and the last one."""

    def __init__(self, most):
        self.most, self.every, self.kept, self.last = most, 1, [], None

    def push(self, frame):
        if frame["generation"] % self.every:
            self.last = frame
            return
        self.kept.append(frame)
        self.last = None
        if len(self.kept) == 8 * self.most:
            self.every *= 2
            self.kept = [kept for kept in self.kept if kept["generation"] % self.every == 0]

    def to_list(self):
        frames = self.kept + ([self.last] if self.last else [])
        active = frames[: last_change(frames) + 1]
        count, most = len(active), max(self.most, 2)
        if count <= most:
            return active
        return [active[(i * (count - 1) + (most - 1) // 2) // (most - 1)] for i in range(most)]


def last_change(frames):
    """The index of the frame after which nothing the page plots changes. To 3 significant
    digits, as a plot shows them: the best, the median and, for a single objective (a numeric
    best), the state; to within a hundredth of their range over the run: a front's
    hypervolumes, in the state or in a grid's series."""
    if not frames:
        return 0
    last = len(frames) - 1
    number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
    single = any(number(frame.get("best")) for frame in frames)

    def measures(frame):
        values = []
        state = frame.get("state")
        hypervolume = state.get("hypervolume") if isinstance(state, dict) else None
        for value in (hypervolume, frame.get("series")):
            if number(value):
                values.append(float(value))
            elif isinstance(value, dict):
                values.extend(float(v) for _, v in sorted(value.items()) if number(v))
        return values

    measured = [measures(frame) for frame in frames]
    end = measured[last]
    tolerance = []
    for k in range(len(end)):
        values = [values[k] for values in measured if k < len(values)]
        tolerance.append((max(values) - min(values)) / 100.0)

    def same(frame, final, key, flush=False):
        return coarse(frame.get(key), flush) == coarse(final.get(key), flush)

    def settled(i):
        frame, final = frames[i], frames[last]
        return (
            same(frame, final, "best")
            and same(frame, final, "median")
            and (not single or same(frame, final, "state", flush=True))
            and len(measured[i]) == len(end)
            and all(abs(v - e) <= t for v, e, t in zip(measured[i], end, tolerance))
        )

    first = last
    while first > 0 and settled(first - 1):
        first -= 1
    return first


def frame(progress, state):
    """The frame of a generation: its progress and ``state``, whose hypervolume is the curve."""
    return {
        "generation": progress.generation,
        "evaluations": progress.evaluations,
        "best": None,
        "median": None,
        "state": state,
    }


def coarse(value, flush=False):
    """``value`` with its numbers to 3 significant digits, as precisely as a plot shows them: two
    frames whose plotted values agree to that precision look the same. With ``flush``, for the
    solutions a plot draws on their ranges, numbers below 1e-6 in size count as 0."""
    if isinstance(value, float):
        if flush and abs(value) < 1e-6:
            value = 0.0
        return f"{value:.2e}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(coarse(item, flush) for item in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items())
        return "{" + ",".join(f"{key}:{coarse(item, flush)}" for key, item in items) + "}"
    return json.dumps(value)


def write(path, settings, frames):
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
