---
title: Welded beam design
category: constrained
summary: The cheapest welded beam under stress, buckling and deflection limits, in the two forms of the literature.
reference: "Ragsdell, K. M. and Phillips, D. T. (1976). Optimal design of a class of welded structures using geometric programming. Journal of Engineering for Industry 98(3): 1021-1025."
reference_url: https://doi.org/10.1115/1.3438995
optimum: "1.724852 (seven constraints) and 2.38116 (five constraints), best known"
languages: [rust, python]
order: 72
---

# Welded beam design

The welded beam problem minimizes the cost of a beam welded to a support that carries 6000 lb at
14 inches. The genes are the weld's thickness h and length l and the bar's height t and thickness
b, and the constraints limit the weld's shear stress, the bar's bending stress, its buckling load
and its deflection. Two forms circulate. `WeldedBeam` has seven constraints (Rao, 1996, as restated
by Coello Coello, 2000, Computers in Industry 41(2): 113-127), with a best known cost of 1.724852.
`WeldedBeamRagsdell` has five (Ragsdell and Phillips, 1976, as restated by Deb, 2000, Computer
Methods in Applied Mechanics and Engineering 186: 311-338), with a best known cost of 2.38116. The
example runs SHADE, a differential evolution, for 40,000 evaluations on each, with Deb's
feasibility rules, and prints the best cost, its violation and its design. On the five-constraint
form it finds 2.381134, below the best known cost. In Python, `run` evaluates the problems in Rust,
so both versions print the same.
