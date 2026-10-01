"""Engineering design problems with several objectives, evaluated in Rust.

Every objective is minimized, in the units of the paper that states it. A constrained problem's
fitness is ``(objectives, violation)``, the violation adding up ``max(0, g(x))`` over its
constraints ``g(x) <= 0``, which keep the scale of the paper that states them::

    import genoxide as gx

    problem = gx.problems.multi_engineering.TwoBarTruss()
    nsga2 = gx.Nsga2(
        problem.genome,
        objectives=problem.objectives,
        population_size=100,
        crossover=gx.SimulatedBinaryCrossover(20),
        mutation=gx.PolynomialMutation(20, rate=1 / 3),
        seed=1,
    )
    result = nsga2.run(problem, generations=200)
    print(gx.indicators.igd_plus(result.front_objectives, problem.optimal_front(500)))

The fronts of :class:`TwoBarTruss`, :class:`FourBarTruss` and :class:`WaterResourcePlanning` are
derived from their definitions; the others aren't known, and their ``optimal_front`` is None.
Their ``ideal_point`` gives each objective's best value, and for two objectives ``nadir_point``
the other objective's value there: best known values, found with genoxide's SHADE.


:class:`DiscBrake` and :class:`SpeedReducer` each have an integer variable: their genome is real,
its integer gene is rounded to the nearest integer when it's evaluated, and ``design(x)`` gives the
rounded design.

The single-objective welded beam, speed reducer and car side impact are in
:mod:`genoxide.problems.engineering`. Each class names the pages of its original that were
checked; the disc brake's and the four-bar truss's originals couldn't be read, and those classes
name the paper that restates the definition used here, still to be checked against the original
(#168).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .. import Real, _genoxide
from . import MultiProblem

__all__ = [
    "TwoBarTruss",
    "WeldedBeam",
    "DiscBrake",
    "SpeedReducer",
    "FourBarTruss",
    "CarSideImpact",
    "RocketInjector",
    "VehicleCrashworthiness",
    "MarineDesign",
    "WaterResourcePlanning",
]


class _Mixed(MultiProblem[Real]):
    """A problem whose genome rounds its discrete genes."""

    def design(self, genome: Any) -> np.ndarray:
        """The design variables of ``genome``, its integer gene rounded to the nearest integer.

        Raises
        ------
        ValueError
            If ``genome`` isn't a 1-D array of numbers with a value per dimension.
        """
        genome = np.ascontiguousarray(genome, dtype=np.float64)
        if genome.ndim != 1:
            raise ValueError(f"a genome is a 1-D array, not an array of shape {genome.shape}")
        return _genoxide.design(self._json(), genome)


@dataclass(frozen=True)
class TwoBarTruss(MultiProblem[Real]):
    """The two-bar truss: the lightest truss of two bars that carries 100 kN, and the one whose
    bars are least stressed.

    The genes are the bars' cross-sections x₁ (AC) and x₂ (BC), in m², and the depth y of the
    loaded joint, in m. The objectives are the volume ``x₁ √(16 + y²) + x₂ √(1 + y²)`` and the
    larger stress ``max(σ_AC, σ_BC)``, in kPa, with ``σ_AC = 20 √(16 + y²) / (y x₁)`` and
    ``σ_BC = 80 √(1 + y²) / (y x₂)``, subject to a stress of at most 10⁵.

    Bounds x₁, x₂ in [0, 0.01], y in [1, 3]; a bar with no section has an infinite stress. The
    front, derived from the definition, has two pieces: y = 2 with volume × stress = 400, for
    stresses from 10⁵ down to 4000√5 ≈ 8944.27, where x₂ reaches 0.01; then x₂ = 0.01 and y from
    2 to 3, stress ``8000 √(1 + y²) / y`` and volume ``(4 + y²) / (80 √(1 + y²))``, down to
    8000√10/3 ≈ 8432.74 at a volume of 0.0513870.

    Deb, K., Pratap, A. and Moitra, S. (2000). Mechanical component design for multiple
    objectives using elitist non-dominated sorting GA. Parallel Problem Solving from Nature (PPSN
    VI), LNCS 1917: 859-868, eq. 1, checked in the authors' preprint (KanGAL report 200002); the
    front as derived by Deb and Srinivasan (2006, KanGAL report 2005007, eqs. 1-7).
    """

    _type: ClassVar[str] = "two_bar_truss"


@dataclass(frozen=True)
class WeldedBeam(MultiProblem[Real]):
    """The welded beam with two objectives: the cheapest beam welded to a support that carries
    6000 lb at 14 in, and the one whose end deflects least.

    The genes are x = (h, l, t, b), in inches. The objectives are the cost
    ``1.10471 h² l + 0.04811 t b (14 + l)`` and the deflection ``2.1952 / (t³ b)``, subject to
    the first four constraints of :class:`genoxide.problems.engineering.WeldedBeamRagsdell`: the
    shear stress, the bending stress, h ≤ b and the buckling load.

    Bounds h, b in [0.125, 5], l, t in [0.1, 10]. The front isn't known; its best known ends are
    a cost of 2.3811341169 with a deflection of 0.0157592, and a deflection of 0.00043904
    (t = 10, b = 5) at a cost of 36.421245.

    Deb, K., Pratap, A. and Moitra, S. (2000). Mechanical component design for multiple
    objectives using elitist non-dominated sorting GA. Parallel Problem Solving from Nature (PPSN
    VI), LNCS 1917: 859-868, eq. 2, checked in the authors' preprint (KanGAL report 200002).
    Tanabe and Ishibuchi's restatement (2020, problem CRE2-4-2) takes the other form's shear
    stress and buckling load: not the same front.
    """

    _type: ClassVar[str] = "multi_welded_beam"


@dataclass(frozen=True)
class DiscBrake(_Mixed):
    """The disc brake: the lightest multiple-disc brake, and the one that stops fastest.

    The genes are the inner and outer radii r and R, in mm, the engaging force F, in N, and the
    number of friction surfaces s, an integer: gene 3 is rounded to the nearest integer, and
    ``design(x)`` gives the rounded design. The objectives are the mass
    ``4.9·10⁻⁵ (R² − r²)(s − 1)`` and the stopping time ``9.82·10⁶ (R² − r²) / (F s (R³ − r³))``,
    subject to ``R − r >= 20``, ``2.5 (s + 1) <= 30``, ``F / (3.14 (R² − r²)) <= 0.4``,
    ``2.22·10⁻³ F (R³ − r³) / (R² − r²)² <= 1`` and ``0.0266 F s (R³ − r³) / (R² − r²) >= 900``.

    Bounds r in [55, 80], R in [75, 110], F in [1000, 3000], s in [2, 20]. The front isn't known;
    its ends follow from the definition: 0.1274 kg stopping in 16.654925 s, and 2.0710401 s at
    2.793 kg.

    Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria
    optimization problems using the simple genetic algorithm. Structural Optimization 10(2):
    94-99, not read. Definition and bounds as restated in Yang, X.-S., Karamanoglu, M. and He, X.
    (2013). Multi-objective flower algorithm for optimization. Procedia Computer Science 18:
    861-868 (eqs. 10-12); not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "disc_brake"


@dataclass(frozen=True)
class SpeedReducer(_Mixed):
    """The speed reducer with two objectives: the lightest gearbox, and the one whose first shaft
    is least stressed.

    The genes are those of Golinski's :class:`genoxide.problems.engineering.SpeedReducer`; the
    number of teeth, gene 2, is rounded to the nearest integer, and ``design(x)`` gives the rounded
    design. The objectives are the volume, with the constants 7.477 and 14.933, and the first
    shaft's stress ``√((745 x₄ / (x₂x₃))² + 1.69·10⁷) / (0.1 x₆³)``, subject to eleven
    constraints, among them stresses of at most 1300 and 1100 in the two shafts.

    Bounds x₁ in [2.6, 3.6], x₂ in [0.7, 0.8], x₃ in [17, 28], x₄, x₅ in [7.3, 8.3], x₆ in
    [2.9, 3.9], x₇ in [5, 5.5]. The front isn't known; its best known ends are a volume of
    2771.9151 at the stress limit of 1300, and a stress of 694.70574 at a volume of 5777.9203.

    Kurpati, A., Azarm, S. and Wu, J. (2002). Constraint handling improvements for multiobjective
    genetic algorithms. Structural and Multidisciplinary Optimization 23(3): 204-213, not read.
    Definition and bounds as restated in Tanabe and Ishibuchi (2020, Applied Soft Computing 89:
    106078, supplement, problem RE3-7-5); not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "multi_speed_reducer"


@dataclass(frozen=True)
class FourBarTruss(MultiProblem[Real]):
    """The four-bar truss: the truss of four bars with the least volume, and the one whose loaded
    joint moves least.

    The genes are the bars' cross-sections, in cm². The objectives are the volume
    ``200 (2x₁ + √2 x₂ + √2 x₃ + x₄)``, in cm³, and the displacement
    ``0.01 (2/x₁ + 2√2/x₂ − 2√2/x₃ + 2/x₄)``, in cm (F = 10 kN, L = 200 cm, E = 2·10⁵ kN/cm²).

    Bounds x₁, x₄ in [1, 3], x₂, x₃ in [√2, 3]. The front, derived from the definition, has
    x₃ = √2 and three pieces, from (1400, 0.04) to (2200 + 600√2, (2√2 − 2)/300) ≈ (3048.53,
    0.0027614).

    Stadler, W. and Dauer, J. (1992). Multicriteria optimization in engineering: a tutorial and
    survey. In Structural Optimization: Status and Promise, AIAA: 209-249, not read. Definition,
    constants and bounds as restated in Costa, M. F. P. and Fernandes, E. M. G. P. (2009). 8th
    World Congress on Structural and Multidisciplinary Optimization, Lisbon, problem (4-truss);
    not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "four_bar_truss"


@dataclass(frozen=True)
class CarSideImpact(MultiProblem[Real]):
    """The car side impact with three objectives: the car's weight, the pubic force a passenger
    feels, and the mean velocity of the B-pillar and the front door.

    The genes, bounds and ten constraints are those of
    :class:`genoxide.problems.engineering.CarSideImpact`. The objectives are its weight, the
    pubic force ``4.72 − 0.5x₄ − 0.19x₂x₃`` and ``(V_MBP + V_FD) / 2``. The front isn't known.

    Gu, L., Yang, R. J., Tho, C. H., Makowski, M., Faruque, O. and Li, Y. (2001). Optimisation and
    robustness for crashworthiness of side impact. International Journal of Vehicle Design 26(4):
    348-360 (not read), with the three objectives of Jain, H. and Deb, K. (2014). IEEE
    Transactions on Evolutionary Computation 18(4): 602-622 (appendix, checked in the authors'
    accepted manuscript).
    """

    _type: ClassVar[str] = "multi_car_side_impact"


@dataclass(frozen=True)
class RocketInjector(MultiProblem[Real]):
    """The rocket injector: the design of a single-element hydrogen-oxygen injector, through
    response surfaces fitted to CFD simulations.

    The genes are the hydrogen's flow angle, the changes in the hydrogen's and the oxygen's flow
    areas and the oxidizer post tip's thickness, each scaled to [0, 1]. The objectives are the
    face's highest temperature TF_max, the post tip's highest temperature TT_max and the
    combustion length X_cc, quadratic and cubic response surfaces; the paper's fourth objective,
    the wall temperature TW₄, is left out as in later studies. The front isn't known.

    Vaidyanathan, R., Tucker, P. K., Papila, N. and Shyy, W. (2003). CFD-based design optimization
    for single element rocket injector. 41st AIAA Aerospace Sciences Meeting, AIAA paper 2003-296,
    eqs. A1, A3 and A4, checked in the authors' copy.
    """

    _type: ClassVar[str] = "rocket_injector"


@dataclass(frozen=True)
class VehicleCrashworthiness(MultiProblem[Real]):
    """Vehicle crashworthiness: the lightest car front that best protects its occupants in frontal
    crashes, through response surfaces fitted to crash simulations.

    The genes are the thicknesses of five reinforcing members, in mm, each in [1, 3]. The
    objectives are the mass, the integral of the deceleration in the full frontal crash and the
    toe board's intrusion in the offset frontal crash. The front isn't known.

    Liao, X., Li, Q., Yang, X., Zhang, W. and Li, W. (2008). Multiobjective optimization for crash
    safety design of vehicles using stepwise regression model. Structural and Multidisciplinary
    Optimization 35(6): 561-569, not read. Definition and bounds as restated in Tanabe and
    Ishibuchi (2020, problem RE3-5-4) and de Carvalho and Sichman (2018, OptMAS, eqs. 1-3); not
    yet checked against the original (#168).
    """

    _type: ClassVar[str] = "vehicle_crashworthiness"


@dataclass(frozen=True)
class MarineDesign(MultiProblem[Real]):
    """Conceptual marine design: the bulk carrier that carries cargo the cheapest, with the
    lightest ship and the most cargo a year, in Parsons and Scott's Panamax case.

    The genes are the length L, beam B, depth D and draft T, in m, the block coefficient C_B and
    the speed V_k, in knots: x = (L, B, D, T, C_B, V_k). The objectives are the transportation
    cost, in £/t, the light ship weight, in t, and the annual cargo, maximized, so its negative,
    in t a year, from the parametric model of a bulk carrier in the paper's appendix (power by the
    Admiralty coefficient, steel, outfit and machinery weights, deadweight, round trips a year,
    capital, running and voyage costs). Nine constraints: ``L/B >= 6``, ``L/D <= 15``,
    ``L/T <= 19``, ``T <= 0.45 DWT^0.31``, ``T <= 0.7 D + 0.7``, ``25,000 <= DWT <= 500,000``,
    a Froude number of at most 0.32 and a metacentric height of at least 0.07 B.

    Bounds L in [150, 274.32], B in [20, 32.31], D in [10, 25], T in [8, 11.71], C_B in
    [0.63, 0.75], V_k in [14, 18]: the Panamax case's limits on L, B and T and the model's on C_B
    and V_k; the other bounds are genoxide's, and hold every feasible design. The front isn't
    known; its ideal point is (8.376894, 5240.3356, −700,552.76), the paper's single-criterion
    designs (table 4) computed again.

    Parsons, M. G. and Scott, R. L. (2004). Formulation of multicriterion design optimization
    problems for solution with scalar numerical optimization methods. Journal of Ship Research
    48(1): 61-76, the numerical example and its Panamax case 2 (pp. 68-69) and the appendix
    (p. 76), checked there: its designs give its printed criteria with this model. Tanabe and
    Ishibuchi's restatement (2020, problem RE4-6-2) differs: a fourth objective, a least
    deadweight of 3000, narrower bounds and, in its code, sea days of (5000/24) V_k.
    """

    _type: ClassVar[str] = "marine_design"


@dataclass(frozen=True)
class WaterResourcePlanning(MultiProblem[Real]):
    """Water resource planning (WATER): a storm drainage system with five costs and losses to
    minimize under seven constraints.

    The genes are the local detention storage capacity x₁, the maximum treatment rate x₂ and the
    maximum allowable overflow rate x₃. The objectives are the drainage network's, storage
    facility's and treatment facility's costs, the expected flood damage and the expected economic
    loss from floods. Bounds x₁ in [0.01, 0.45], x₂, x₃ in [0.01, 0.1].

    The front, derived from the definition: the optimal solutions are every (x₁, x₂, 0.01) with
    x₁x₂ ≥ 0.00139 / (1.08 − 0.0494), where g₁ is the only binding constraint, and the front is
    their image, a surface in five dimensions that ``optimal_front`` samples on a grid over x₁ and
    x₂ and along its edge.

    Musselman, K. and Talavage, J. (1980). A tradeoff cut approach to multiple objective
    optimization. Operations Research 28(6): 1424-1435, not read. Definition and bounds as
    restated in Deb, Pratap, Agarwal and Meyarivan (2002, NSGA-II, table V) and in Jain and Deb
    (2014, part II, appendix).
    """

    _type: ClassVar[str] = "water_resource_planning"
