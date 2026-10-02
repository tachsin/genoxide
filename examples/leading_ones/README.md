---
title: LeadingOnes
category: binary
summary: Maximize the number of ones before the first zero of a string of 100 bits, with the (1+1) evolutionary algorithm, whose Θ(n²) time on it Droste, Jansen and Wegener proved.
reference: "Droste, S., Jansen, T. and Wegener, I. (2002). On the analysis of the (1+1) evolutionary algorithm. Theoretical Computer Science 276(1-2): 51-81."
reference_url: https://doi.org/10.1016/S0304-3975(01)00182-7
optimum: "100 (all ones)"
languages: [rust, python]
order: 11
---

# LeadingOnes

## The problem

LeadingOnes counts the ones of a bit string before its first zero:

```text
LeadingOnes(x) = Σᵢ₌₁ⁿ Πⱼ₌₁ⁱ xⱼ
```

The string 1101 0111 scores 2. Here the strings have 100 bits, so the best score is 100, for the
string of all ones. The definition is Droste, Jansen and Wegener's (2002, Definition 16), who took
the function from Rudolph's book (1997, *Convergence Properties of Evolutionary Algorithms*, Kovač,
Hamburg, not read). The function is genoxide's `problems::binary::LeadingOnes`.

## What makes it hard

The landscape is unimodal: appending a one to the leading ones always improves a string, so there
are no local optima. But only one bit can improve a string, the first zero, and nothing before it
may change. The bits after it don't count, and drift at random until the leading ones reach them.

Droste et al. used the function to disprove a remark, which they attribute to Mühlenbein, that
every unimodal function takes the (1+1) evolutionary algorithm O(n log n) steps, as
[OneMax](../one_max/) does. Their Theorem 17 proves that LeadingOnes takes Θ(n²) steps on average:
at most e n² (27,183 for n = 100), and at least n²/6 (1,667) except with a probability exponentially
small in n.

## Representation

A `Binary` genome of 100 bits is the string itself. The fitness is its number of leading ones, to
maximize.

## Algorithm

The (1+1) evolutionary algorithm of Droste et al.: one string, each of whose bits is flipped with
probability 1/n, the child replacing it if it's no worse. In genoxide, that's `LocalSearch` with
bit-flip mutation at a rate of 1/100 as its neighbor, one neighbor per step, and the default
acceptance of neighbors that are no worse.

`LocalSearch` draws a neighbor again when the mutation flips nothing, so each evaluation is a step
of the (1+1) EA that changes the string. A step of the (1+1) EA flips nothing with probability
(1 − 1/n)ⁿ = 0.366 for n = 100, and such steps don't change the string, so the search is the
(1+1) EA's, without its idle steps: the (1+1) EA takes on average 1 / 0.634 times as many steps
as there are evaluations here.

A run from seed 1 stops at the optimum, or after 1,000,000 evaluations. Then runs from seeds 1 to
100 count the evaluations to the optimum.

## Output

The first lines give the run from seed 1: the evaluations at which the leading ones first reach 10,
20, … 100, and the evaluations to the optimum. The last two lines give the runs from seeds 1 to 100:
how many reach the optimum, the mean of their evaluations, the fewest, the median and the most, and
n² for comparison. The function is evaluated in Rust in both languages, so both print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/leading-ones) plays back the run
from seed 1: the string, its leading ones growing from the left while the bits after them change at
random.

## Good results

The optimum is 100. The run from seed 1 reaches it after 4,452 evaluations. Every run from seeds 1
to 100 reaches it, after 5,332 evaluations on average, from 3,503 to 7,802, with a median of
5,312.5. Counted with its idle steps, the (1+1) EA takes about 5,332 / 0.634 ≈ 8,410 steps on
average, between Droste et al.'s n²/6 and e n².
