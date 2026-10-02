"""Gaussian process regression, fitted in Rust: the surrogate model of :class:`genoxide.Bo`, which
can also be fitted and queried on its own.

**Unstable for one release.** The module is public from genoxide 0.13, but its API and the bits of
its fits may still change in 0.14.

A :class:`GaussianProcess` models a function of the genes of a :class:`genoxide.Real` genome from
its values at some points, and gives the posterior mean and variance of the function anywhere,
with their gradients (Rasmussen and Williams 2006, eq. 2.25 and 2.26)::

    import numpy as np
    import genoxide as gx

    # a smooth function of one gene, from 8 evaluations
    x = np.arange(8.0).reshape(-1, 1)
    y = np.sin(x[:, 0]) + 0.1 * x[:, 0] ** 2
    model = gx.model.gp.GaussianProcess.fit(gx.Real((0, 7), length=1), x, y)
    mean, variance = model.predict([[3.5]])

The model has a constant mean and a kernel with a length scale per gene, "matern52" (Matérn,
ν = 5/2, the default) or "squared_exponential"; inputs are scaled to the unit cube by the genome's
bounds, and values standardized. Without noise by default, the model interpolates the values, as
suits a deterministic function; :class:`Learned` noise is for a noisy one. The hyperparameters
maximize the log marginal likelihood (eq. 2.30), with genoxide's L-BFGS-B and the gradient of eq.
5.9, from ``starts`` starts (the first from fixed values, the others random, from ``seed``): the
same fit on every platform, and the one genoxide's Rust ``model::gp`` gives.

References: Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine
Learning.* MIT Press, ch. 2, 4 and 5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .. import Real, _describe_setting, _genoxide, _number, _whole

__all__ = ["GaussianProcess", "Hyperparameters", "Learned", "Kernel"]

Kernel = Literal["matern52", "squared_exponential"]
_KERNELS = ("matern52", "squared_exponential")


@dataclass(frozen=True)
class Learned:
    """Noise learned with the other hyperparameters, its variance at least ``min`` of the values'
    variance (between 0 and 1, exclusive): for a noisy function, whose values the model shouldn't
    interpolate."""

    min: float = 1e-6

    def _describe(self) -> dict[str, Any]:
        return {"type": "learned", "min": _number("Learned.min", self.min)}


@dataclass(frozen=True)
class Hyperparameters:
    """A Gaussian process's hyperparameters, in the units of the genes and of the values: the
    constant ``mean``, a length scale per gene (infinite for a gene whose bounds are equal), the
    signal variance (the function's variance far from any data) and the noise variance."""

    mean: float
    length_scales: tuple[float, ...]
    signal_variance: float
    noise_variance: float


def _noise(noise: Any) -> dict[str, Any]:
    """A noise setting: a fixed variance, a fraction of the values' variance, or :class:`Learned`."""
    if isinstance(noise, Learned):
        return noise._describe()
    return {"type": "fixed", "variance": _number("noise", noise)}


def _kernel(kernel: Any) -> str:
    if kernel not in _KERNELS:
        raise ValueError(f'kernel is "matern52" or "squared_exponential", not {kernel!r}')
    return str(kernel)


def _rows(name: str, points: Any, genes: int) -> np.ndarray:
    """Points as a 2-D float64 array, a point per row: a 1-D array is one point."""
    rows = np.ascontiguousarray(points, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != genes:
        raise ValueError(
            f"{name} are points of {genes} genes, a point per row, not of shape {np.shape(points)}"
        )
    return rows


class GaussianProcess:
    """A Gaussian process fitted to evaluations of a function, from :meth:`fit`, or the model of a
    running :class:`genoxide.Bo` (:attr:`genoxide.RunningBo.model`)."""

    __slots__ = ("_native",)

    def __init__(self, native: _genoxide.GaussianProcess) -> None:
        self._native = native

    @classmethod
    def fit(
        cls,
        genome: Real,
        points: Any,
        values: Any,
        *,
        kernel: Kernel = "matern52",
        noise: float | Learned = 0.0,
        starts: int = 5,
        seed: int = 0,
        hyperparameters: Hyperparameters | None = None,
    ) -> GaussianProcess:
        """Fits a model to ``values`` at ``points``.

        Parameters
        ----------
        genome : Real
            The genes, whose bounds scale the model's inputs to the unit cube.
        points : array-like of float
            A point per row, a value per gene, all finite; at least one point.
        values : array-like of float
            A finite value per point.
        kernel : "matern52" or "squared_exponential", default "matern52"
            The kernel: Matérn's with ν = 5/2 (twice differentiable functions), or the squared
            exponential (infinitely differentiable).
        noise : float or Learned, default 0.0
            The noise variance as a fraction of the values' variance: a fixed one, at least 0 (0
            interpolates the values), or :class:`Learned` with the other hyperparameters.
        starts : int, default 5
            The starts of the likelihood's maximization, at least 1.
        seed : int, default 0
            The seed of the random starts, 0 to 2^64 - 1.
        hyperparameters : Hyperparameters, optional
            Hyperparameters to use as they are, instead of fitting them; ``noise`` is then
            ignored.

        Raises
        ------
        ValueError
            For a wrong setting, no points, a point without a value per gene, or a point or value
            that isn't finite.
        """
        described = _describe_setting("genome", genome, "a Real")
        if described.get("type") != "real":
            raise ValueError(f"genome is a Real, not {genome!r}")
        bounds = described["bounds"]
        rows = _rows("points", points, len(bounds))
        array = np.ascontiguousarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(
                f"values is a 1-D array, a value per point, not of shape {array.shape}"
            )
        given = None
        if hyperparameters is not None:
            given = {
                "mean": _number("hyperparameters.mean", hyperparameters.mean),
                "length_scales": [
                    _number("hyperparameters.length_scales", length, plural=True)
                    for length in hyperparameters.length_scales
                ],
                "signal_variance": _number(
                    "hyperparameters.signal_variance", hyperparameters.signal_variance
                ),
                "noise_variance": _number(
                    "hyperparameters.noise_variance", hyperparameters.noise_variance
                ),
            }
        description = {
            "bounds": bounds,
            "kernel": _kernel(kernel),
            "noise": _noise(noise),
            "starts": _whole("starts", starts),
            "seed": _whole("seed", seed, minimum=0, maximum=2**64 - 1),
            "hyperparameters": given,
        }
        return cls(_genoxide.GaussianProcess(json.dumps(description), rows, array))

    def predict(self, points: Any) -> tuple[np.ndarray, np.ndarray]:
        """The posterior means and variances of the function at ``points``, a point per row (a
        1-D array is one point): two 1-D arrays, a value per point. The variances are of the
        function, without the noise, and at least 0."""
        rows = _rows("points", points, self.genes)
        means, variances = self._native.predict(rows)
        return np.asarray(means), np.asarray(variances)

    def predict_with_gradient(self, point: Any) -> tuple[float, float, np.ndarray, np.ndarray]:
        """The posterior mean and variance at ``point``, with their gradients with respect to the
        genes (0 for a gene whose bounds are equal)."""
        array = np.ascontiguousarray(point, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"point is a 1-D array, a value per gene, not of shape {array.shape}")
        mean, variance, dmean, dvariance = self._native.predict_with_gradient(array)
        return float(mean), float(variance), np.asarray(dmean), np.asarray(dvariance)

    @property
    def hyperparameters(self) -> Hyperparameters:
        """The hyperparameters, fitted or given."""
        mean, length_scales, signal, noise = self._native.hyperparameters
        return Hyperparameters(float(mean), tuple(length_scales), float(signal), float(noise))

    @property
    def log_marginal_likelihood(self) -> float:
        """The log marginal likelihood of the values at the hyperparameters (eq. 2.30)."""
        return float(self._native.log_marginal_likelihood)

    @property
    def jitter(self) -> float:
        """The jitter added to the kernel matrix's diagonal for its factorization, in the values'
        units: usually 0."""
        return float(self._native.jitter)

    @property
    def kernel(self) -> Kernel:
        """The kernel."""
        if self._native.kernel == "squared_exponential":
            return "squared_exponential"
        return "matern52"

    @property
    def genes(self) -> int:
        """The number of genes of a point."""
        return int(self._native.genes)

    def __len__(self) -> int:
        """The number of points the model was fitted to."""
        return len(self._native)

    def __repr__(self) -> str:
        return f"GaussianProcess(points={len(self)}, kernel={self.kernel!r})"
