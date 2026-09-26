---
title: Pressure vessel design
category: constrained
summary: The cheapest cylindrical pressure vessel of a given volume, a mixed discrete-continuous engineering design problem.
reference: "Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design optimization. Journal of Mechanical Design 112(2): 223-229."
reference_url: https://doi.org/10.1115/1.2912596
optimum: "6059.714335 (cost)"
languages: [rust, python]
order: 70
---

# Pressure vessel design

The pressure vessel problem minimizes the cost of material, forming and welding of a cylindrical
vessel with hemispherical heads, under four constraints: the minimum thicknesses of the shell and
the heads for their radius, a volume of at least 1,296,000 cubic inches, and a length of at most
240 inches. The shell and head thicknesses are multiples of 0.0625 inch, and the radius and length
are continuous; the example rounds its first two real genes to whole numbers of 0.0625-inch plates.
The fitness function returns the cost and the total violation (the volume's relative to 1,296,000),
compared with Deb's feasibility rules, and SHADE, a differential evolution, runs for 50,000
evaluations. The best known cost, 6059.714335, was proven globally optimal by Yang et al. (2013,
International Journal of Bio-Inspired Computation 5(6): 329-335).
