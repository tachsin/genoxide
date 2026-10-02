---
title: Rocket injector
category: multi-objective
summary: Minimize two temperatures of a single-element rocket injector and the length of its combustion, three response surfaces fitted to CFD simulations, with SMS-EMOA.
reference: "Vaidyanathan, R., Tucker, P. K., Papila, N. and Shyy, W. (2003). CFD-based design optimization for single element rocket injector. 41st AIAA Aerospace Sciences Meeting, AIAA paper 2003-296."
reference_url: https://doi.org/10.2514/6.2003-296
optimum: "not known in closed form; ideal point (0.0088934, −0.4315, 0.00488); genoxide's reference front has a hypervolume of 0.9019 in objectives scaled by the ideal point and its estimated nadir point (1.002, 1.0965, 1.0539) (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 242
---

# Rocket injector

## The problem

Vaidyanathan, Tucker, Papila and Shyy (2003) designed a single-element injector of hydrogen and
oxygen for a rocket engine: an element that injects the hydrogen at an angle towards the oxygen. Its
four design variables are the hydrogen's flow angle α, the changes in the hydrogen's and the
oxygen's flow areas ΔHA and ΔOA, and the oxidizer post tip's thickness OPTT, each scaled to [0, 1]
over its range. They ran CFD simulations of a set of designs and fitted response surfaces to four
results: the face's highest temperature TF_max, the wall temperature three inches from the face TW₄,
the post tip's highest temperature TT_max, and the combustion length X_cc, where combustion is 99%
complete. Lower temperatures mean a longer life, a shorter combustion a better performance.

```text
minimize   TF_max = 0.692 + 0.477α − 0.687ΔHA − 0.080ΔOA − 0.0650OPTT − 0.167α² − 0.0129ΔHAα
                    + 0.0796ΔHA² − 0.0634ΔOAα − 0.0257ΔOAΔHA + 0.0877ΔOA² − 0.0521OPTTα
                    + 0.00156OPTTΔHA + 0.00198OPTTΔOA + 0.0184OPTT²
           TT_max = 0.370 − 0.205α + 0.0307ΔHA + 0.108ΔOA + 1.019OPTT − 0.135α² + 0.0141ΔHAα
                    + 0.0998ΔHA² + 0.208ΔOAα − 0.0301ΔOAΔHA − 0.226ΔOA² + 0.353OPTTα
                    − 0.0497OPTTΔOA − 0.423OPTT² + 0.202ΔHAα² − 0.281ΔOAα² − 0.342ΔHA²α
                    − 0.245ΔHA²ΔOA + 0.281ΔOA²ΔHA − 0.184OPTT²α − 0.281ΔHAαΔOA
           X_cc   = 0.153 − 0.322α + 0.396ΔHA + 0.424ΔOA + 0.0226OPTT + 0.175α² + 0.0185ΔHAα
                    − 0.0701ΔHA² − 0.251ΔOAα + 0.179ΔOAΔHA + 0.0150ΔOA² + 0.0134OPTTα
                    + 0.0296OPTTΔHA + 0.0752OPTTΔOA + 0.0192OPTT²
α, ΔHA, ΔOA, OPTT in [0, 1]
```

The surfaces are the paper's eqs. A1, A3 and A4, checked in the authors' copy (NASA NTRS
20030060421). The paper's second objective, TW₄ (eq. A2), is left out, as in the three-objective
form of Goel et al. (2007, *Computer Methods in Applied Mechanics and Engineering* 196: 879-893),
which Tanabe and Ishibuchi (2020, problem RE3-4-7) restate; genoxide couldn't read Goel et al.'s
paper, and its `RocketInjector` keeps the original's order of the other three. The objectives are
the surfaces' values, scaled as in the paper; TT_max's surface falls below 0 at a corner of the box,
at (1, 1, 1, 0).

The front isn't known. Its ideal point, from genoxide's SHADE, is (0.0088934, −0.4315, 0.00488).
genoxide's reference front, the non-dominated designs of about 15,000 runs of SHADE, each minimizing
an achievement scalarizing function along one of Das and Dennis's directions, and of twenty runs of
SMS-EMOA and NSGA-III of 2,000 generations, has worst values (1.002, 1.0965, 1.0539), the estimated
nadir point, and a hypervolume of 0.9019 in the scaled objectives below: a lower bound on the whole
front's.

## What makes it hard

The surfaces are quadratic and, for TT_max, cubic, and the objectives conflict strongly: the design
that minimizes TF_max, (0, 1, 0.591, 1), is near the worst in both others. The front is a curved,
irregular surface.

## Representation

A `Real` genome of 4 genes, the scaled design variables. The problem is genoxide's
`multi::problems::engineering::RocketInjector` (`gx.problems.multi_engineering.RocketInjector` in
Python). It has no constraints besides the bounds.

## Algorithm

SMS-EMOA, which keeps the solutions that add the most hypervolume, with a population of 92 for 500
generations, simulated binary crossover (η = 20, at a rate of 0.9) and polynomial mutation (η = 20,
at a rate of 1/4 per gene). NSGA-III with 91 directions ends with less hypervolume, 93.4% to 94.4%
of the reference front's over seeds 1 to 10.

## Output

The first line gives the size of the final front and how many of its solutions are feasible (all
are), the second the range of each objective on it. The third gives its hypervolume up to the
reference point (1.1, 1.1, 1.1), in objectives scaled to [0, 1] by the ideal point and the estimated
nadir point, as a share of the reference front's. In Python, `run` evaluates the problem in Rust, so
both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/rocket-injector) plays this run
back.

## Good results

A good front spreads over the whole surface, reaching each objective's minimum. 92 points of the
reference front, chosen one by one for the most hypervolume, give 97.21% of its hypervolume.

The run's front has 92 solutions, reaching the three minima (0.0089, −0.4315, 0.0049), with 95.58%
of the reference front's hypervolume, 98.3% of what 92 chosen points reach. Its X_cc goes up to
1.0604, beyond the reference front's worst: a solution there is dominated by the reference front,
though not by the run's own. Over seeds 1 to 20, every run ends between 95.27% and 95.70%.
