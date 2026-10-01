"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names, for a new kind of plot, ``cart_poles``: the cart and its poles animated.
Each of at most 64 generations has the first 100 steps (2 s) of the best network so far, an
``[x, θ₁, θ₂]`` per step (m, degrees), from the initial state; the settings have the task's
geometry and the solution's first 500 steps (10 s). The page plays them.
The Rust example writes the same file."""

import json
import math
import os

# the steps of an episode in a frame, and in the solution's
FRAME_STEPS = 100
SOLUTION_STEPS = 500


def degrees(radians):
    """``radians`` in degrees, as Rust's ``to_degrees`` computes them: times 180 / π."""
    return radians * (180.0 / math.pi)


class Trace:
    """Records the run through ``record`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, mlp, task):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = Frames(64)
        self.mlp, self.task = mlp, task

    def record(self, progress):
        """Records a generation: the best network's first steps."""
        if self.path:
            episode = self.episode(progress.best_genome, FRAME_STEPS)
            self.frames.push(frame(progress, {"episode": episode}))

    def write(self, solution):
        """Writes the trace, if there's one, with the solution's episode."""
        if self.path:
            settings = {
                "format": 1,
                "example": "double_pole",
                "objective": "maximize",
                "x_label": "evaluations",
                "y_label": "steps balanced",
                "log_y": True,
                "optimum": 100000.0,
                "plot": "cart_poles",
                "problem": {
                    "track": 2.4,
                    "half_lengths": [0.5, 0.05],
                    "masses": [0.1, 0.01],
                    "failure_angle": 36.0,
                    "step": 0.02,
                    "episode": self.episode(solution, SOLUTION_STEPS),
                },
            }
            write(self.path, settings, self.frames.to_list())

    def episode(self, weights, steps):
        """``[x, θ₁, θ₂]`` after each step of an episode of at most ``steps`` steps from the
        initial state."""
        states = self.task.episode(self.mlp.policy(weights), steps).tolist()
        return [
            [x, degrees(theta_1), degrees(theta_2)] for x, _, theta_1, _, theta_2, _ in states
        ]


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
    """The frame of a generation: its progress, the median score of its population and
    ``state``."""
    return {
        "generation": progress.generation,
        "evaluations": progress.evaluations,
        "best": progress.best_fitness,
        "median": median(progress.scores),
        "state": state,
    }


def median(scores):
    """The median of the valid scores, None without any."""
    scores = sorted(float(score) for score in scores if not math.isnan(score))
    middle = len(scores) // 2
    if not scores:
        return None
    return scores[middle] if len(scores) % 2 else (scores[middle - 1] + scores[middle]) / 2


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
