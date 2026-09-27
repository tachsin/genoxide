"""The trace of the searches for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: a contour panel per function, with its searches' points side by side,
and each function's error to its global minimum, in at most 100 of their generations. The Rust
example writes the same file."""

import json
import math
import os


class Trace:
    """Records the searches through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self):
        self.path = os.environ.get("GENOXIDE_TRACE")
        # per function: its title, the contour plot's problem, its global minimum, and per
        # search, after each generation, its evaluations, and its best value and point
        self.panels = []

    def function(self, title, function, bounds, optimum):
        """Starts the panel of a function, whose searches ``record`` records next: its title, its
        name for the contour plot, its bounds and its global minima."""
        problem = {
            "function": function,
            "bounds": [[float(low), float(high)] for low, high in bounds],
            "minima": optimum.solutions.tolist(),
        }
        self.panels.append(
            {"title": title, "problem": problem, "minimum": optimum.value, "searches": []}
        )

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation of a search, which starts at generation 0: its evaluations, and
        its best value and point so far."""
        searches = self.panels[-1]["searches"]
        if progress.generation == 0:
            searches.append([])
        point = progress.best_genome.tolist()
        searches[-1].append((progress.evaluations, progress.best_fitness, point))

    def write(self):
        """Writes the trace, if there's one: the functions' searches side by side, a frame per
        generation, with each panel's points and best point, and each function's error."""
        if not self.path:
            return
        frames = Frames(100)
        every_search = [search for panel in self.panels for search in panel["searches"]]
        for generation in range(max(map(len, every_search), default=0)):
            evaluations, series, panels = 0, {}, []
            for panel in self.panels:
                searches = [
                    search[min(generation, len(search) - 1)] for search in panel["searches"]
                ]
                evaluations += sum(search[0] for search in searches)
                valid = [search for search in searches if not math.isnan(search[1])]
                best = min(valid, key=lambda search: search[1], default=None)
                # rounding can put a solution a few ulps below the minimum
                series[panel["title"]] = max(best[1] - panel["minimum"], 0.0) if best else None
                panels.append(
                    {
                        "population": [search[2] for search in searches],
                        "best": best[2] if best else None,
                    }
                )
            frames.push(
                {
                    "generation": generation,
                    "evaluations": evaluations,
                    "series": series,
                    "state": {"panels": panels},
                }
            )
        settings = {
            "format": 1,
            "example": "minima_2d",
            "objective": "minimize",
            "optimum": None,
            "x_label": "generations",
            "y_label": "error to the global minimum",
            "log_y": True,
            "plot": "grid",
            "problem": {
                "panel_plot": "contour",
                "panels": [
                    {"title": panel["title"], "problem": panel["problem"]} for panel in self.panels
                ],
                "series": [panel["title"] for panel in self.panels],
            },
        }
        write(self.path, settings, frames.to_list())


# ---- the same in every example's trace ----------------------------------------------------------


class Frames:
    """The frames of at most ``most`` generations: every ``every``-th one, with ``every`` doubling
    whenever there are ``most``, and the last one."""

    def __init__(self, most):
        self.most, self.every, self.kept, self.last = most, 1, [], None

    def push(self, frame):
        if frame["generation"] % self.every:
            self.last = frame
            return
        self.kept.append(frame)
        self.last = None
        if len(self.kept) == self.most:
            self.every *= 2
            self.kept = [kept for kept in self.kept if kept["generation"] % self.every == 0]

    def to_list(self):
        return self.kept + ([self.last] if self.last else [])


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
