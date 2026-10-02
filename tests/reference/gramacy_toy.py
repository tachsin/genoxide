"""The global minimum of Gramacy et al.'s (2016) toy problem, to 30 significant digits with mpmath,
independently of genoxide: the tests and the bo_constrained example embed it, rounded to the
nearest f64.

    pip install mpmath
    python tests/reference/gramacy_toy.py

Gramacy, R. B., Gray, G. A., Le Digabel, S., Lee, H. K. H., Ranjan, P., Wells, G. and Wild, S. M.
(2016). Modeling an augmented Lagrangian for blackbox constrained optimization. Technometrics
58(1): 1-11, section 1: minimize f(x) = x1 + x2 on [0, 1]^2 subject to

    c1(x) = 3/2 - x1 - 2 x2 - sin(2 pi (x1^2 - 2 x2)) / 2 <= 0
    c2(x) = x1^2 + x2^2 - 3/2 <= 0.

The paper gives the minimizer as about (0.1954, 0.4044) with f about 0.5998, c1 active there and c2
strictly satisfied, and two local minima, about (0.7197, 0.1411) with f about 0.8609, and (0, 0.75)
on the bound x1 = 0. With c1 active and c2 inactive, a minimum inside the box satisfies c1 = 0 and
the Lagrange condition grad f = -lambda grad c1, that is dc1/dx1 = dc1/dx2: two equations, solved
here by Newton's method from the paper's points. A scan of the box on a grid of 1/2000 then checks
that the first is the global minimum.
"""

from mpmath import cos, findroot, mp, mpf, nstr, pi, sin

mp.dps = 30


def c1(x1, x2):
    return mpf(3) / 2 - x1 - 2 * x2 - sin(2 * pi * (x1**2 - 2 * x2)) / 2


def c2(x1, x2):
    return x1**2 + x2**2 - mpf(3) / 2


def equations(x1, x2):
    angle = 2 * pi * (x1**2 - 2 * x2)
    dc1_dx1 = -1 - 2 * pi * x1 * cos(angle)
    dc1_dx2 = -2 + 2 * pi * cos(angle)
    return [c1(x1, x2), dc1_dx1 - dc1_dx2]


for start in [(mpf("0.1954"), mpf("0.4044")), (mpf("0.7197"), mpf("0.1411"))]:
    x1, x2 = findroot(equations, start)
    print(f"x = ({nstr(x1, 20)}, {nstr(x2, 20)}), f = {nstr(x1 + x2, 20)} ({float(x1 + x2)!r}), "
          f"c2 = {nstr(c2(x1, x2), 6)}")

# the least x1 + x2 over the feasible points of a grid of 1/2000
n = 2000
least = None
for i in range(n + 1):
    x1 = mpf(i) / n
    for j in range(n + 1):
        x2 = mpf(j) / n
        if least is not None and x1 + x2 >= least:
            break
        if c1(x1, x2) <= 0 and c2(x1, x2) <= 0:
            least = x1 + x2
            break
print(f"the grid's least feasible f: {nstr(least, 6)}")
