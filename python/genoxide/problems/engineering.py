"""Engineering design problems with one objective, evaluated in Rust.

Every problem is minimized. A constrained problem's fitness is ``(score, violation)``, the
violation adding up ``max(0, g(x))`` over its constraints ``g(x) <= 0``, which keep the scale of
the paper that states them::

    import genoxide as gx

    problem = gx.problems.engineering.WeldedBeam()
    shade = gx.De(problem.genome, objective=problem.objective, seed=1)
    result = shade.run(problem, evaluations=30_000)
    print(result.best_fitness, result.violation)

Most originals aren't openly available: each class names its original and the later paper that
restates the definition used here, still to be checked against the original (#168).

:class:`GearTrain` has an :class:`genoxide.Integer` genome. :class:`PressureVessel` and
:class:`SpeedReducer` mix discrete and continuous variables: their genome is real, its discrete
genes are rounded to their grid when it's evaluated, and ``design(x)`` gives the rounded design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .. import _genoxide
from . import Problem

__all__ = [
    "WeldedBeam",
    "WeldedBeamRagsdell",
    "PressureVessel",
    "TensionCompressionSpring",
    "SpeedReducer",
    "GearTrain",
    "ThreeBarTruss",
    "CantileverBeam",
    "CarSideImpact",
]


class _Mixed(Problem):
    """A problem whose genome rounds its discrete genes."""

    def design(self, genome: Any) -> np.ndarray:
        """The design variables of ``genome``, its discrete genes rounded to their grid.

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
class WeldedBeam(Problem):
    """The welded beam, in the form with seven constraints: the cheapest beam welded to a support
    that carries 6000 lb at 14 in.

    The genes are x = (h, l, t, b), in inches; the cost is ``1.10471 h² l + 0.04811 t b (14 + l)``,
    subject to the weld's shear stress, the bar's bending stress, h ≤ b, a second cost limit,
    h ≥ 0.125, the end deflection and the buckling load.

    Bounds h, b in [0.1, 2], l, t in [0.1, 10]; best known 1.724852 at (0.205730, 3.470489,
    9.036624, 0.205729), from Cagnina, Esquivel and Coello Coello (2008, Informatica 32: 319-326),
    whose printed solution exceeds the bending and buckling limits by 0.09 psi and 0.06 lb, and
    has h above b by 1e-6.
    :class:`WeldedBeamRagsdell` is the other form in the literature.

    This form isn't :class:`WeldedBeamRagsdell`'s mechanics with other limits: its J is twice the
    weld throat's, and its buckling load has E where Ragsdell and Phillips's has √(EG), 1.58 times
    larger. It's the benchmark as published, so it stays as it is.

    Rao, S. S. (1996). Engineering Optimization. Wiley, third edition. Definition and bounds as
    restated in Coello Coello, C. A. (2000). Use of a self-adaptive penalty approach for
    engineering optimization problems. Computers in Industry 41(2): 113-127 (eqs. 22-37), and the
    same in Coello Coello and Mezura-Montes (2002, Advanced Engineering Informatics 16: 193-203,
    eqs. 8-23), after Rao; Cagnina, Esquivel and Coello Coello (2008) attribute it to Ragsdell and
    Phillips. Not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "welded_beam"


@dataclass(frozen=True)
class WeldedBeamRagsdell(Problem):
    """The welded beam in the form with five constraints, after Ragsdell and Phillips.

    The genes are x = (h, l, t, b), in inches, with :class:`WeldedBeam`'s cost, subject to the
    weld's shear stress, the bar's bending stress, h ≤ b, the buckling load
    ``64,746.022 (1 − 0.0282346 t) t b³ >= 6000`` and the deflection ``2.1952 / (t³ b) <= 0.25``.

    Bounds h in [0.125, 10], l, t, b in [0.1, 10]; best known 2.3811341 at (0.24436895, 6.2186069,
    8.2914718, 0.24436895), feasible, found with genoxide's SHADE. Reklaitis, Ravindran and
    Ragsdell (1983) report 2.38116 at (0.2444, 6.2187, 8.2915, 0.2444); printed to 4 digits, that
    solution evaluates to 2.38151. Ragsdell and Phillips's own solution, (0.2455, 6.1960, 8.2730,
    0.2455) at 2.3859 as tabulated by Coello Coello and Mezura-Montes (2002), exceeds the shear
    stress limit by 0.31 psi in this form.

    Ragsdell, K. M. and Phillips, D. T. (1976). Optimal design of a class of welded structures
    using geometric programming. Journal of Engineering for Industry 98(3): 1021-1025. Definition,
    bounds and best known solution as restated in Deb, K. (2000). An efficient constraint handling
    method for genetic algorithms. Computer Methods in Applied Mechanics and Engineering
    186(2-4): 311-338 (eq. 3); not yet checked against the originals (#168).
    """

    _type: ClassVar[str] = "welded_beam_ragsdell"


@dataclass(frozen=True)
class PressureVessel(_Mixed):
    """The pressure vessel: the cheapest cylindrical vessel with hemispherical heads that holds
    1,296,000 cubic inches, a mixed discrete-continuous problem.

    The genes are the shell's and the heads' thicknesses, the inner radius and the length, in
    inches. The thicknesses are multiples of 0.0625 in: genes 0 and 1 are rounded to the nearest
    multiple, and ``design(x)`` gives the rounded design. The cost is
    ``0.6224 T_s R L + 1.7781 T_h R² + 3.1661 T_s² L + 19.84 T_s² R``, subject to
    ``T_s >= 0.0193 R``, ``T_h >= 0.00954 R``, a volume of at least 1,296,000 and ``L <= 240``.

    Bounds T_s, T_h in [0.0625, 6.1875], R, L in [10, 200]; minimum 6059.714335048436 at
    (0.8125, 0.4375, 42.0984455958549, 176.6365958424394), proven global by Yang, Huyck,
    Karamanoglu and Khan (2013, International Journal of Bio-Inspired Computation 5(6): 329-335);
    the solution's length is a few units in the last place longer than printed, to be feasible
    despite rounding.

    Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design
    optimization. Journal of Mechanical Design 112(2): 223-229. Definition as restated in Coello
    Coello (2000, Computers in Industry 41(2): 113-127, eqs. 17-21), and the same in Coello Coello
    and Mezura-Montes (2002) and Cagnina et al. (2008); not yet checked against the original
    (#168). The originals' own designs suggest that the originals also had minimum thicknesses,
    T_s >= 1.1 and T_h >= 0.6: Sandgren's (1.125, 0.625, 47.70, 117.70), at 8129.10, has both
    walls thicker than g1 and g2 require, and Kannan and Kramer's (1.125, 0.625, 58.291, 43.690),
    at 7198.04, has a head of 0.625 where 0.5625 meets g2 and costs about 378 less. With those
    minimums the least cost is 7198.006; the minimum of 6059.714 is of the form without them.
    Some restatements print 3.1611 for 3.1661, which gives 7197.729 there.
    """

    _type: ClassVar[str] = "pressure_vessel"


@dataclass(frozen=True)
class TensionCompressionSpring(Problem):
    """The tension/compression spring: the lightest coil spring under constraints on its
    deflection, shear stress, surge frequency and outer diameter.

    The genes are the wire diameter d, the mean coil diameter D and the number of active coils N;
    the weight is ``(N + 2) D d²``.

    Bounds d in [0.05, 2], D in [0.25, 1.3], N in [2, 15]; best known 0.012665 at (0.051690,
    0.356750, 11.287126), from Cagnina, Esquivel and Coello Coello (2008), whose printed solution
    exceeds the second constraint by 2e-5.

    Belegundu, A. D. (1982). A Study of Mathematical Programming Methods for Structural
    Optimization. PhD thesis, University of Iowa; and Arora, J. S. (1989). Introduction to
    Optimum Design. McGraw-Hill. Definition and bounds as restated in Coello Coello (2000,
    Computers in Industry 41(2): 113-127, eqs. 38-42); not yet checked against the originals
    (#168).
    """

    _type: ClassVar[str] = "tension_compression_spring"


@dataclass(frozen=True)
class SpeedReducer(_Mixed):
    """Golinski's speed reducer: the lightest gearbox under eleven constraints on stresses,
    deflections and proportions, a mixed discrete-continuous problem.

    The genes are the face width, the module of the teeth, the number of teeth on the pinion (an
    integer: gene 2 is rounded, and ``design(x)`` gives the rounded design), the lengths of the
    shafts between bearings and the diameters of the shafts.

    Bounds [2.6, 3.6], [0.7, 0.8], [17, 28], [7.3, 8.3], [7.8, 8.3], [2.9, 3.9], [5.0, 5.5]; best
    known 2996.348165 at (3.5, 0.7, 17, 7.3, 7.8, 3.350214, 5.286683), from the restatement, whose
    printed solution exceeds g5 by 6.0e-7 and g6 by 1.3e-7, and evaluates to 2996.347849. Some
    formulations print 7.477 for 7.4777 and 1.5079 for 1.508, and some let x5 go down to 7.3,
    where the minimum is 2994.471 (Lin, Tsai, Hu and Chang, 2013).

    Golinski, J. (1970). Optimal synthesis problems solved by means of nonlinear programming and
    random methods. Journal of Mechanisms 5(3): 287-309, where the problem is first posed; and
    Golinski, J. (1973). An adaptive optimization system applied to machine synthesis. Mechanism
    and Machine Theory 8(4): 419-436. Definition, bounds and best known solution as restated in
    Cagnina, L. C., Esquivel, S. C. and Coello Coello, C. A. (2008). Solving engineering
    optimization problems with the simple constrained particle swarm optimizer. Informatica 32:
    319-326 (appendix); not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "speed_reducer"


@dataclass(frozen=True)
class GearTrain(Problem):
    """The gear train: the numbers of teeth of four gears whose ratio is closest to 1/6.931, on
    an :class:`genoxide.Integer` genome.

    The score is ``(1/6.931 − T_d T_b / (T_a T_f))²``, with no constraints.

    Bounds [12, 60]⁴; minimum 2.7008571488865134e-12 at T_d T_b = 16 · 19 and T_a T_f = 43 · 49,
    four genomes, checked by evaluating every genome.

    Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design
    optimization. Journal of Mechanical Design 112(2): 223-229. Definition and bounds as restated
    in Deb, K. and Goyal, M. (1996). A combined genetic adaptive search (GeneAS) for engineering
    design. Computer Science and Informatics 26(4): 30-45; not yet checked against the original
    (#168).
    """

    _type: ClassVar[str] = "gear_train"


@dataclass(frozen=True)
class ThreeBarTruss(Problem):
    """The three-bar truss: the least volume of a planar truss of three bars, subject to their
    stresses.

    The genes are the cross-sections x₁ (outer bars) and x₂ (middle bar), in cm²; the volume is
    ``(2√2 x₁ + x₂) · 100``. The stresses are undefined at x₁ = 0.

    Bounds [0, 1]²; minimum 100 (√2 + √6/2) ≈ 263.8958434 at ((3 + √3)/6, 1/√6), derived from the
    definition.

    Nowacki, H. (1974). Optimization in pre-contract ship design. In Computer Applications in the
    Automation of Shipyard Operation and Ship Design, North-Holland: 327-338. Definition and
    bounds as restated in Yang, X.-S. and Gandomi, A. H. (2012). Bat algorithm: a novel approach
    for global engineering optimization. Engineering Computations 29(5): 464-483 (eqs. 14-17);
    not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "three_bar_truss"


@dataclass(frozen=True)
class CantileverBeam(Problem):
    """The cantilever beam: the lightest beam of five square segments, subject to its
    deflection, ``61/x₁³ + 37/x₂³ + 19/x₃³ + 7/x₄³ + 1/x₅³ <= 1``.

    The weight is ``0.0624 Σ xᵢ``. Bounds [0.01, 100]⁵; minimum ≈ 1.339956361, in closed form
    from the Lagrange conditions of this convex problem, as in Yang, Huyck, Karamanoglu and Khan
    (2013).

    Fleury, C. and Braibant, V. (1986). Structural optimization: a new dual method using mixed
    variables. International Journal for Numerical Methods in Engineering 23(3): 409-428.
    Definition and bounds as restated in Yang et al. (2013, International Journal of Bio-Inspired
    Computation 5(6): 329-335, eqs. 35-37); not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "cantilever_beam"


@dataclass(frozen=True)
class CarSideImpact(Problem):
    """The car side impact: the lightest car body whose side withstands the European side-impact
    test, through response surfaces fitted to crash simulations.

    The seven genes are thicknesses of the B-pillar, the floor, the cross members, the door beam,
    the door beltline and the roof rail, subject to ten constraints on the dummy's loads,
    velocities and rib deflections and on the structure's velocities.

    Bounds x₁, x₃, x₄ in [0.5, 1.5], x₂ in [0.45, 1.35], x₅ in [0.875, 2.625], x₆, x₇ in
    [0.4, 1.2]; the restatement gives no optimum. Best known 23.585658 at (0.5, 1.225732, 0.5,
    1.207111, 0.875, 0.884329, 0.4), found by SLSQP from 300 starting points and by genoxide's
    SHADE. Not proven optimal.

    Gu, L., Yang, R. J., Tho, C. H., Makowski, M., Faruque, O. and Li, Y. (2001). Optimisation and
    robustness for crashworthiness of side impact. International Journal of Vehicle Design 26(4):
    348-360. Definition and bounds as restated in Jain, H. and Deb, K. (2014). An evolutionary
    many-objective optimization algorithm using reference-point based nondominated sorting
    approach, part II. IEEE Transactions on Evolutionary Computation 18(4): 602-622 (appendix);
    not yet checked against the original (#168). The abdomen load's 0.0092928 x₃ is as published
    there and in the implementations that follow it; the eleven-variable form's 0.484 x₃ x₉, with
    x₉ fixed at 0.192, would give 0.092928, ten times as much. Likewise the lower chest's
    0.031296 x₃ is 0.163 x₃ times x₉'s 0.192, where the eleven-variable restatements have x₈'s
    0.345. Neither constraint is active at the best known design, so neither changes it.
    """

    _type: ClassVar[str] = "car_side_impact"
