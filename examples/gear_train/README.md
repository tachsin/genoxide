---
title: Gear train design
category: integer
summary: The numbers of teeth of four gears whose ratio is closest to 1/6.931, an integer problem.
reference: "Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design optimization. Journal of Mechanical Design 112(2): 223-229."
reference_url: https://doi.org/10.1115/1.2912596
optimum: "2.700857e-12 (squared error of the ratio)"
languages: [rust, python]
order: 106
---

# Gear train design

## The problem

A compound gear train of four gears passes the rotation of an input shaft to an output shaft at a
fixed ratio. Sandgren (1990) chooses the numbers of teeth of the four gears, T_a, T_b, T_d and T_f,
so that the ratio T_d T_b / (T_a T_f) is as close as possible to 1/6.931. Each gear has 12 to 60
teeth. In the form that Deb and Goyal (1996, Computer Science and Informatics 26(4): 30-45) restate,
the score is the squared error of the ratio:

```text
(1/6.931 − T_d T_b / (T_a T_f))²
```

The target, 1/6.931, is 0.14427932. With 16 and 19 teeth over 43 and 49, the ratio is 304/2107 =
0.14428097.

## What makes it hard

Every variable is an integer, and there are no constraints besides the bounds. There are 49⁴ =
5,764,801 designs, few enough to evaluate all of them, which is how the minimum is known.

The score depends only on the two products T_d T_b and T_a T_f. A change of one tooth changes a
product by 1.7% (at 60 teeth) to 8% (at 12), so neighboring designs have very different scores. The
best designs are scattered: three designs share the second best squared error, 2.3e-11, among them
13 and 20 over 34 and 53, with no gear in common with the best one.

## Representation

An `Integer` genome of 4 genes, (T_d, T_b, T_a, T_f), each from 12 to 60: the design itself. The
fitness is the squared error, to minimize. The problem is genoxide's `GearTrain`, which brings its
bounds and its minimum.

## Algorithm

A genetic algorithm with the pieces that genoxide's guide lists for integers in ranges:

- a population of 100;
- binary tournaments (size 2);
- uniform crossover, which takes each gene from either parent;
- uniform mutation, which redraws a gene with probability 0.25, one gene per child on average, since
  there are four.

Uniform crossover combines the gears of two parents, so it can join a good pair T_d T_b from one
with a good pair T_a T_f from the other. The run stops at the minimum, or after 2,000 generations.

## Output

The first line gives the squared error of the best design, the generations it took, and the known
minimum. The second gives its teeth, (T_d, T_b, T_a, T_f), its ratio and the target ratio. In
Python, `run` evaluates the problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/gear-train) plays this run back.

## Good results

The minimum is 2.700857e-12, at 16 · 19 / (43 · 49), and at the three designs that swap 16 with 19
or 43 with 49. The run reaches it after about 180 generations, at most about 18,000 evaluations: a
third of a percent of the designs.
