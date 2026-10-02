"""Reference values of the special functions in genoxide::math and of the acquisition functions,
to 50 significant digits with mpmath, independently of genoxide. The tests embed its output,
rounded to the nearest f64 (repr), with the inputs they come from.

    pip install mpmath
    python tests/reference/special_functions.py
"""

from mpmath import erfc, exp, log, mp, mpf, ncdf, npdf

mp.dps = 50

ERFCX = [-26.0, -10.0, -3.0, -1.0, -0.5, -1e-3, 0.0, 1e-10, 0.25, 0.5, 1.0, 2.0, 3.5, 5.0,
         10.0, 20.0, 25.9, 26.0, 26.1, 30.0, 50.0, 100.0, 1e3, 1e8, 1e20, 1e200]
LOG_H = [5.0, 1.0, 0.0, -0.5, -0.999, -1.0, -1.001, -2.0, -5.0, -10.0, -30.0, -40.0, -1e3,
         -1e7, -6.7e7, -6.72e7, -1e8, -1e10, -1e100]


def erfcx(x):
    if x > 1e10:
        # mpmath's erfc overflows here; the asymptotic series' first terms are exact to far below
        # an f64's precision
        t = 1 / (2 * x * x)
        return (1 - t + 3 * t * t) / (x * mp.sqrt(mp.pi))
    return exp(x * x) * erfc(x)


def h(z):
    # Ament et al. (2023): EI = sigma h(z), z = (mu - best) / sigma, maximizing
    return npdf(z) + z * ncdf(z)


print("// erfcx(x)")
for x in ERFCX:
    print(f"({x!r}, {float(erfcx(mpf(x)))!r}),")
print("// ln h(z)")
for z in LOG_H:
    print(f"({z!r}, {float(log(h(mpf(z))))!r}),")
