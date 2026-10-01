# Plan: general optimization methods

A working plan, removed when the work is done. Of batch A1, `Algorithm::is_finished` with
`StopReason::Converged`, `local::Restarts` and Nelder-Mead are implemented; the linear algebra and
the later batches aren't yet. genoxide has
evolutionary and population-based methods (GA, ES, CMA-ES, DE, PSO, local search, NSGA-II and
the other multi-objective algorithms). This plan adds the other families of a general
optimization library: local derivative-free methods, gradient-based methods, constrained
nonlinear programming and Bayesian optimization, with the same guarantees. It lists the methods
and their sources, designs their API on genoxide's ask / tell architecture, sets how they're
tested, and orders the work in batches.

**Rules.**

- Every method is implemented from its paper or book, and its docs cite them (the algorithm,
  equation or page used). Nothing is copied from another library: no code, no text, no default
  values taken on trust, no expected test values. Where the authors published a reference code
  (L-BFGS-B 3.0, Powell's Fortran codes, the MADS and ALGENCAN authors' codes), it may be read to
  compare behavior, as the CEC 2006 organizers' code was, if its license allows, and is cited as
  the authors' implementation.
- The guarantees of AGENTS.md hold for every new method: a seed gives the same results to the bit
  on every platform and thread count; errors, not panics; validated builders; ask / tell at the
  core; every example reaches its optimum.
- **Status of the citations:** every reference below was recorded from the literature while
  planning, not re-read (**U** in the problem plan's terms). Before a batch starts, the primary
  sources of its methods are read, and each entry is marked verified or corrected here, with the
  algorithm or equation numbers the implementation follows.
- **Batch A1, read on 2026-10-01.** Lagarias et al. (1998), section 2 (pp. 114-117): **verified**.
  The coefficients' conditions (eq. 2.1), the standard values (2.2), the five steps of an iteration
  with their strict and non-strict inequalities, and the two ordering rules are as implemented. The
  non-shrink rule prints the new vertex's place as `j = max{ℓ | f(v) < f(x_{ℓ+1})}`, which read
  literally is always n; the implementation follows the rule as the paper states it in words, "the
  highest possible index consistent with the ordering": after every vertex at least as good.
  Gao and Han (2012): **not re-read**, behind a paywall with no open copy found; the adaptive
  coefficients are as recorded here, and the docs cite the paper for them. They give the standard
  ones for n = 2, and for n = 1 they'd shrink to a point (σ = 0), so the standard ones are used
  there. Nelder and Mead (1965): **not re-read**; the method follows Lagarias et al.'s statement of
  it. The termination test (simplex size against a tolerance, relative to each gene's range) is
  genoxide's own: Lagarias et al. define one iteration, not when to stop.
- The methods are general. The docs and examples motivate them with generic cases (expensive
  simulations, engineering design, black-box functions, model fitting), not with an application
  domain.

**Contents.**

1. [Scope and families](#1-scope-and-families)
2. [API design](#2-api-design)
3. [Test problems and validation](#3-test-problems-and-validation)
4. [Examples](#4-examples)
5. [Benchmarks](#5-benchmarks)
6. [Order of work](#6-order-of-work)
7. [Open questions](#7-open-questions)

## 1. Scope and families

### 1.1 What genoxide has, and what's missing

| Family | genoxide today | Missing |
|---|---|---|
| Population-based, single objective | GA (4 schemes, memetic), ES, CMA-ES (IPOP, BIPOP, sep), DE (JADE, SHADE, L-SHADE), PSO, islands, asynchronous steady-state GA | |
| Single-solution search | `LocalSearch`: hill climbing, annealing, tabu, iterated local search, on any representation | direct search on reals with its own geometry (simplex, patterns, meshes) |
| Local continuous | none | Nelder-Mead, pattern search / MADS, Powell's model-based methods |
| Derivatives | none | supplied gradients, finite differences, gradient-based methods |
| Constraints | Deb's rules and penalties over an aggregate violation; per-constraint values in `problems::Problem::constraints` | methods that use each constraint's value and gradient (SQP, augmented Lagrangian, interior point) |
| Expensive functions | `AsyncEngine`, `Batch`, parallel evaluation | surrogate models (Gaussian processes) and Bayesian optimization |
| Multi-objective | NSGA-II, NSGA-III, SPEA2, MOEA/D, SMS-EMOA | ParEGO, EHVI |
| Linear algebra | a symmetric eigendecomposition inside CMA-ES (tred2 and tql2 of JAMA) | Cholesky, QR, triangular solves, a dense QP solver |

### 1.2 Local derivative-free methods

For functions without gradients, noisy in the last digits, or with a few hundred evaluations to
spare: they find a local optimum with far fewer evaluations than a population-based method, and
polish what a global method found.

| Method | Primary reference | For | Design notes | Batch |
|---|---|---|---|---|
| Nelder-Mead, with adaptive parameters | Nelder, J. A. and Mead, R. (1965). A simplex method for function minimization. *The Computer Journal* 7(4): 308-313. doi:10.1093/comjnl/7.4.308. Gao, F. and Han, L. (2012). Implementing the Nelder-Mead simplex algorithm with adaptive parameters. *Computational Optimization and Applications* 51(1): 259-277. doi:10.1007/s10589-010-9329-3. Convergence: Lagarias, J. C., Reeds, J. A., Wright, M. H. and Wright, P. E. (1998). *SIAM J. Optim.* 9(1): 112-147. doi:10.1137/S1052623496303470 | Low dimensions (up to about 10), non-smooth or noisy functions, the most-used local method | Gao and Han's parameters (reflection 1, expansion 1 + 2/n, contraction 0.75 − 1/(2n), shrink 1 − 1/n) by default, the 1965 ones as an option; Lagarias et al.'s tie-breaking rules (the definition to follow for ties); initial simplex a fraction of each gene's range, like CMA-ES's initial step; points outside the box clipped to it; restarts on a collapsed simplex; an optional speculative ask of the reflection, expansion and both contractions at once (same path, fewer rounds, more evaluations) | A1 |
| Compass search, generalized pattern search | Hooke, R. and Jeeves, T. A. (1961). "Direct search" solution of numerical and statistical problems. *JACM* 8(2): 212-229. doi:10.1145/321062.321069. Torczon, V. (1997). On the convergence of pattern search algorithms. *SIAM J. Optim.* 7(1): 1-25. doi:10.1137/S1052623493250780 | Robust local search on reals and integers; the 2n poll points of an iteration are one ask, so they're evaluated in parallel | Coordinate descent is compass search with one direction at a time: not a separate method | D2 |
| MADS (OrthoMADS), with the progressive barrier | Audet, C. and Dennis, J. E. (2006). Mesh adaptive direct search algorithms for constrained optimization. *SIAM J. Optim.* 17(1): 188-217. doi:10.1137/040603371. Abramson, M. A., Audet, C., Dennis, J. E. and Le Digabel, S. (2009). OrthoMADS. *SIAM J. Optim.* 20(2): 948-966. doi:10.1137/080716980. Audet, C. and Dennis, J. E. (2009). A progressive barrier for derivative-free nonlinear programming. *SIAM J. Optim.* 20(1): 445-472. doi:10.1137/070692662 | Non-smooth, constrained black-box functions with a convergence theory; `Real` and `Integer` genomes (the mesh maps to integers); the constraint values of section 2.5 | OrthoMADS directions from genoxide's rng, not a Halton sequence, unless the paper's deterministic sequence is needed for its properties (to check) | D2 |
| Powell's conjugate directions | Powell, M. J. D. (1964). An efficient method for finding the minimum of a function of several variables without calculating derivatives. *The Computer Journal* 7(2): 155-162. doi:10.1093/comjnl/7.2.155 | Smooth functions without gradients, a classic | Optional: BOBYQA supersedes it; kept for completeness if cheap | D2 (optional) |
| BOBYQA: trust region with quadratic interpolation models, bounds | Powell, M. J. D. (2009). *The BOBYQA algorithm for bound constrained optimization without derivatives.* Report DAMTP 2009/NA06, University of Cambridge. Powell, M. J. D. (2004). Least Frobenius norm updating of quadratic models that satisfy interpolation conditions. *Math. Programming* 100(1): 183-215. doi:10.1007/s10107-003-0490-7. Background: Conn, A. R., Scheinberg, K. and Vicente, L. N. (2009). *Introduction to Derivative-Free Optimization.* SIAM. doi:10.1137/1.9780898718768 | The strongest general choice for smooth, expensive functions without gradients, up to a few hundred variables | 2n + 1 interpolation points by default; the trust-region subproblem with bounds (Powell's truncated CG); this is the DFO-TR family, so no separate DFO-TR | D2 |
| COBYLA: linear models of objective and constraints | Powell, M. J. D. (1994). A direct search optimization method that models the objective and constraint functions by linear interpolation. In *Advances in Optimization and Numerical Analysis*, Kluwer: 51-67. doi:10.1007/978-94-015-8330-5_4 | Inequality-constrained problems without gradients | Takes the per-constraint values of section 2.5; equalities as two inequalities, documented. Zhang's PRIMA (modernized reference codes of Powell's methods, BSD-3-Clause) may be read to compare behavior | D2 |

### 1.3 Gradient-based methods

For smooth functions with gradients (supplied, or by finite differences when evaluations are
cheap enough to spend n or 2n per gradient): the fastest local convergence, and the only practical
choice from hundreds of variables up. The general reference for the whole family is Nocedal, J.
and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.
doi:10.1007/978-0-387-40065-5 (**NW**).

| Method | Primary reference | For | Design notes | Batch |
|---|---|---|---|---|
| Line search: strong Wolfe conditions, Moré-Thuente | Wolfe, P. (1969). Convergence conditions for ascent methods. *SIAM Review* 11(2): 226-235. doi:10.1137/1011036. Moré, J. J. and Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient decrease. *ACM TOMS* 20(3): 286-307. doi:10.1145/192115.192132 | The line search of L-BFGS(-B) and BFGS | c₁ = 1e-4, c₂ = 0.9 for quasi-Newton, 0.1 for CG (NW ch. 3); every trial step is an ask. With finite differences, a trial needs f and the directional derivative only: 2 evaluations, not n + 1; the full gradient only at the accepted point | A2 |
| Line search: Hager-Zhang (approximate Wolfe) | Hager, W. W. and Zhang, H. (2005). A new conjugate gradient method with guaranteed descent and an efficient line search. *SIAM J. Optim.* 16(1): 170-192. doi:10.1137/030601880. Hager, W. W. and Zhang, H. (2006). Algorithm 851: CG_DESCENT. *ACM TOMS* 32(1): 113-137. doi:10.1145/1132973.1132979 | CG, and near the optimum where the Wolfe test fails from rounding | | D1 |
| Gradient descent with backtracking (Armijo) | NW ch. 3 | Teaching, a baseline in tests | Part of the line search module, not a separate algorithm unless cheap | D1 |
| Momentum, Nesterov, Adam, AdamW | Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Comput. Math. Math. Phys.* 4(5): 1-17. doi:10.1016/0041-5553(64)90137-5. Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate O(1/k²). *Soviet Math. Dokl.* 27: 372-376. Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. ICLR 2015. arXiv:1412.6980. Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. ICLR 2019. arXiv:1711.05101 | Differentiable models with many parameters, where a line search costs too much; the step as a setting | One gradient per ask. Mini-batches: the fitness function reads its batch index from state that `Engine::control` advances, as the penalty example of `control` does; `best()` is then by mini-batch loss, documented. Full-batch runs are the tested case | D1 |
| Nonlinear conjugate gradient: PR+ and Hager-Zhang | Polak, E. and Ribière, G. (1969). Note sur la convergence de méthodes de directions conjuguées. *Revue française d'informatique et de recherche opérationnelle* 3(16): 35-43. Gilbert, J. C. and Nocedal, J. (1992). Global convergence properties of conjugate gradient methods for optimization. *SIAM J. Optim.* 2(1): 21-42. doi:10.1137/0802003. Hager and Zhang (2005) above | Very many variables, O(n) memory | Hager-Zhang's β and line search by default | D1 |
| BFGS (dense) | Broyden, C. G. (1970). *J. Inst. Math. Appl.* 6(1): 76-90. doi:10.1093/imamat/6.1.76. Fletcher, R. (1970). *The Computer Journal* 13(3): 317-322. doi:10.1093/comjnl/13.3.317. Goldfarb, D. (1970). *Math. Comp.* 24: 23-26. doi:10.1090/S0025-5718-1970-0258249-6. Shanno, D. F. (1970). *Math. Comp.* 24: 647-656. doi:10.1090/S0025-5718-1970-0274029-X | Up to a few hundred variables, unconstrained; the inverse Hessian is also an estimate of the covariance at the optimum | NW ch. 6's scaling of the first update | D1 |
| L-BFGS | Nocedal, J. (1980). Updating quasi-Newton matrices with limited storage. *Math. Comp.* 35(151): 773-782. doi:10.1090/S0025-5718-1980-0572855-7. Liu, D. C. and Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Math. Programming* 45: 503-528. doi:10.1007/BF01589116 | The default for smooth problems of any size | L-BFGS-B without active bounds: one implementation, not two (the box of `Real` is always there; see section 2.6) | A2 |
| L-BFGS-B | Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM J. Sci. Comput.* 16(5): 1190-1208. doi:10.1137/0916069. Zhu, C., Byrd, R. H., Lu, P. and Nocedal, J. (1997). Algorithm 778: L-BFGS-B. *ACM TOMS* 23(4): 550-560. doi:10.1145/279232.279236. Morales, J. L. and Nocedal, J. (2011). Remark on "Algorithm 778". *ACM TOMS* 38(1): 7. doi:10.1145/2049662.2049669 | Smooth problems with bounds: the most-used gradient method, and the inner solver of the GP hyperparameters, the acquisition functions and the augmented Lagrangian | Generalized Cauchy point, subspace minimization with Morales and Nocedal's correction, Moré-Thuente line search; memory m = 10 by default; tolerances on the projected gradient and the relative decrease (section 2.7) | A2 |
| Trust-region Newton with Steihaug-CG | Steihaug, T. (1983). The conjugate gradient method and trust regions in large scale optimization. *SIAM J. Numer. Anal.* 20(3): 626-637. doi:10.1137/0720042. Moré, J. J. and Sorensen, D. C. (1983). Computing a trust region step. *SIAM J. Sci. Stat. Comput.* 4(3): 553-572. doi:10.1137/0904038. NW ch. 4, 7 | Newton's convergence with a Hessian (supplied, a Hessian-vector product, or differences of gradients); nonconvex problems | Also covers "Newton for small problems": the exact subproblem (Moré-Sorensen) for n up to a few hundred, Steihaug-CG above. The Hessian is an optional extra (section 2.3) | D1 |
| Levenberg-Marquardt | Levenberg, K. (1944). *Quart. Appl. Math.* 2(2): 164-168. doi:10.1090/qam/10666. Marquardt, D. W. (1963). *J. SIAM* 11(2): 431-441. doi:10.1137/0111030. Moré, J. J. (1978). The Levenberg-Marquardt algorithm: implementation and theory. *Numerical Analysis*, LNM 630: 105-116. doi:10.1007/BFb0067700 | Least squares: model fitting, calibration of simulations to data | Needs the residual vector (section 2.3); Moré's scaled trust-region form with a QR of the Jacobian; bounds by projection, documented | D1 |

**Derivatives** (section 2.3 has the API):

| Source | Reference | Notes | Batch |
|---|---|---|---|
| Supplied by the user | | The gradient of the score as returned, whatever the objective; checked against central differences by `gradient::check` | A2 |
| Forward and central differences | Gill, P. E., Murray, W., Saunders, M. A. and Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. *SIAM J. Sci. Stat. Comput.* 4(2): 310-321. doi:10.1137/0904025. NW ch. 8 | h = √ε · max(\|xᵢ\|, 1) forward, ε^(1/3) · max(\|xᵢ\|, 1) central, rounded so that x + h − x = h exactly; one-sided inward at a bound so every point is in the box; fixed genes skipped; all n or 2n points in one ask, so `parallel(true)` and `Batch` evaluate them together | A2 |
| Forward-mode dual numbers (feature `dual`) | Wengert, R. E. (1964). A simple automatic derivative evaluation program. *CACM* 7(8): 463-464. doi:10.1145/355586.364791. Griewank, A. and Walther, A. (2008). *Evaluating Derivatives*, 2nd ed. SIAM. doi:10.1137/1.9780898717761 | A `Dual` number type and a `Scalar` trait; a function written once, generic over `Scalar`, gives its value (`f64`) and exact gradient (`Dual`, n passes or one vector pass). `math`'s portable functions get `Dual` versions, so gradients are the same to the bit everywhere. Reverse mode isn't planned | D1 (optional) |

### 1.4 Constrained nonlinear programming

For smooth problems with constraints g(x) ≤ 0 and h(x) = 0 whose values (and gradients, or
finite differences) are available one by one: they converge to a KKT point to many digits,
where Deb's rules only compare aggregate violations.

| Method | Primary reference | For | Design notes | Batch |
|---|---|---|---|---|
| SQP (SLSQP-style): BFGS Hessian, ℓ₁ merit function, line search | Han, S.-P. (1977). A globally convergent method for nonlinear programming. *JOTA* 22(3): 297-309. doi:10.1007/BF00932858. Powell, M. J. D. (1978). A fast algorithm for nonlinearly constrained optimization calculations. *Numerical Analysis*, LNM 630: 144-157. doi:10.1007/BFb0067703. Kraft, D. (1988). *A software package for sequential quadratic programming.* DFVLR-FB 88-28. QP: Goldfarb, D. and Idnani, A. (1983). A numerically stable dual method for solving strictly convex quadratic programs. *Math. Programming* 27: 1-33. doi:10.1007/BF02591962. NW ch. 18 | Small and medium dense problems with equalities and inequalities: the most-used constrained local method | Powell's damped BFGS keeps the Hessian positive definite, so the QP is strictly convex and Goldfarb-Idnani applies; an infeasible linearization is relaxed by an elastic variable; bounds as QP bounds | C |
| Augmented Lagrangian (PHR, safeguarded, ALGENCAN-style) | Hestenes, M. R. (1969). Multiplier and gradient methods. *JOTA* 4(5): 303-320. doi:10.1007/BF00927673. Powell, M. J. D. (1969). A method for nonlinear constraints in minimization problems. In *Optimization* (R. Fletcher, ed.), Academic Press: 283-298. Andreani, R., Birgin, E. G., Martínez, J. M. and Schuverdt, M. L. (2008). On augmented Lagrangian methods with general lower-level constraints. *SIAM J. Optim.* 18(4): 1286-1309. doi:10.1137/060654797. Birgin, E. G. and Martínez, J. M. (2014). *Practical Augmented Lagrangian Methods for Constrained Optimization.* SIAM. doi:10.1137/1.9781611973365 | Many constraints, degenerate problems, and any inner solver: L-BFGS-B by default, BOBYQA or a population method for a derivative-free variant | Bounds stay in the subproblem (L-BFGS-B), the other constraints move into the Lagrangian; multipliers safeguarded in a box, the penalty raised ×10 when the infeasibility doesn't fall by half (the book's defaults, to check). The subproblem is a changing fitness function, so an inner population method re-evaluates through `Reevaluate` | C |
| Interior point with a filter line search | Wächter, A. and Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. *Math. Programming* 106(1): 25-57. doi:10.1007/s10107-004-0559-y. Fletcher, R. and Leyffer, S. (2002). Nonlinear programming without a penalty function. *Math. Programming* 91(2): 239-269. doi:10.1007/s101070100244. Byrd, R. H., Hribar, M. E. and Nocedal, J. (1999). An interior point algorithm for large-scale nonlinear programming. *SIAM J. Optim.* 9(4): 877-900. doi:10.1137/S1052623497325107. Fiacco, A. V. and McCormick, G. P. (1968). *Nonlinear Programming: Sequential Unconstrained Minimization Techniques.* Wiley; SIAM reprint 1990. doi:10.1137/1.9781611971316 | Many inequalities; iterates strictly feasible for the inequalities, which matters when the function is undefined outside them | Dense KKT systems with an LDLᵀ (Bunch-Kaufman) factorization and inertia correction; no sparse linear algebra, so small and medium problems only. The largest item of the plan: optional, after SQP and the augmented Lagrangian | F (optional) |

**Relation to the existing constraint convention.** genoxide writes constraints as `g ≤ 0` and
`h = 0` (`problems::Constraints`), measures violations with `constraint::at_most`, `at_least`
and `equal`, and compares solutions by Deb's rules on `(score, violation)`. The NLP methods keep
all of it: they read the constraint values one by one (section 2.5), use their own merit function,
filter or Lagrangian internally, and report their best individual by Deb's rules on the same
aggregate violation, so `outcome.best()` means what it means for a GA. The test problems with
`constraints()` (CEC 2006, the engineering designs) run with SQP as they are.

### 1.5 Bayesian optimization

For expensive black-box functions (a simulation of minutes to hours, a physical experiment): a
Gaussian process models the function from every evaluation so far, and an acquisition function
picks the next points, so that tens to a few hundred evaluations suffice.

| Part | Primary reference | Design notes | Batch |
|---|---|---|---|
| Gaussian process regression | Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press (**GPML**), ch. 2, 4, 5 | Cholesky of the kernel matrix with a jitter raised ×10 from 1e-10 until it factors (deterministic); inputs scaled to the unit cube by the bounds, outputs standardized; a constant mean; the log marginal likelihood and its gradient (GPML eq. 5.8, 5.9) | B |
| Kernels: ARD Matérn 5/2 (default) and RBF | GPML ch. 4; Stein, M. L. (1999). *Interpolation of Spatial Data.* Springer. doi:10.1007/978-1-4612-1494-6. Matérn 5/2 as the default for BO: Snoek, J., Larochelle, H. and Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. NeurIPS 25. arXiv:1206.2944 | A length scale per gene (ARD); `math::exp` and `sqrt` only, so portable | B |
| Hyperparameters | GPML ch. 5 | Maximize the log marginal likelihood over log length scales, log signal variance and log noise variance (lower bound 1e-6 of the standardized variance, a setting) with L-BFGS-B from the previous optimum and a few random starts (derived streams; ties to the earlier start); an optional log-normal prior on length scales for higher dimensions: Hvarfner, C., Hellsten, E. O. and Nardi, L. (2024). Vanilla Bayesian optimization performs great in high dimensions. ICML 2024. arXiv:2402.02229 | B |
| Initial design | McKay, M. D., Beckman, R. J. and Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics* 21(2): 239-245. doi:10.1080/00401706.1979.10489755. Size: Jones et al. (1998) below; Loeppky, J. L., Sacks, J. and Welch, W. J. (2009). Choosing the sample size of a computer experiment: a practical guide. *Technometrics* 51(4): 366-376. doi:10.1198/TECH.2009.08040 | Latin hypercube, the size a setting (default to settle in batch B: 10n per the references, or 2(n + 1)), plus `initial_genomes`; a scrambled Sobol sequence (Joe, S. and Kuo, F. Y. (2008). *SIAM J. Sci. Comput.* 30(5): 2635-2654. doi:10.1137/070709359) as an option later | B |
| Expected improvement (EI) | Močkus, J. (1975). On Bayesian methods for seeking the extremum. *Optimization Techniques IFIP 1974*, LNCS 27: 400-404. doi:10.1007/3-540-07165-2_55. Jones, D. R., Schonlau, M. and Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *J. Global Optim.* 13(4): 455-492. doi:10.1023/A:1008306431147 | Closed form with the normal pdf and cdf; `math` gains portable `erf`, `erfc` (from `libm`) | B |
| log-EI (default acquisition) | Ament, S., Daulton, S., Eriksson, D., Balandat, M. and Bakshy, E. (2023). Unexpected improvements to expected improvement for Bayesian optimization. NeurIPS 2023. arXiv:2310.20708 | EI's logarithm computed stably where EI underflows (its asymptotic expansion; a portable `erfcx`), so the acquisition optimizer keeps a gradient | B |
| UCB | Srinivas, N., Krause, A., Kakade, S. and Seeger, M. (2010). Gaussian process optimization in the bandit setting: no regret and experimental design. ICML 2010. arXiv:0912.3995 | β a setting, changeable by `control` for a schedule | B |
| Probability of improvement | Kushner, H. J. (1964). A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. *J. Basic Eng.* 86(1): 97-106. doi:10.1115/1.3653121 | For completeness; ξ a setting | B |
| Knowledge gradient | Frazier, P. I., Powell, W. B. and Dayanik, S. (2009). The knowledge-gradient policy for correlated normal beliefs. *INFORMS J. Computing* 21(4): 599-613. doi:10.1287/ijoc.1080.0314 | Optional: noisy problems; costly to optimize | E (optional) |
| Acquisition optimization | | Raw samples (random in the box, a derived stream), the best k and the best observed points as starts, L-BFGS-B with the acquisition's analytic gradient from each, on rayon with a deterministic winner (value, then start index); CMA-ES instead for integer genes (rounded inside the kernel) | B |
| Batch BO (q points per ask): Kriging believer, constant liar | Ginsbourger, D., Le Riche, R. and Carraro, L. (2010). Kriging is well-suited to parallelize optimization. In *Computational Intelligence in Expensive Optimization Problems*, Springer: 131-162. doi:10.1007/978-3-642-10701-6_6 | Each chosen point is added with a fantasized value (the posterior mean, or a lie: min, mean or max) before the next is chosen; the same mechanism serves `AsyncEngine`'s pending points | B |
| Batch BO: local penalization | González, J., Dai, Z., Hennig, P. and Lawrence, N. (2016). Batch Bayesian optimization via local penalization. AISTATS 2016, PMLR 51: 648-657. arXiv:1505.08052 | Cheaper than refitting per point; an option | E |
| Constrained BO: probability of feasibility | Schonlau, M., Welch, W. J. and Jones, D. R. (1998). Global versus local search in constrained optimization of computer models. *IMS Lecture Notes* 34: 11-25. doi:10.1214/lnms/1215456182. Gardner, J. R., Kusner, M. J., Xu, Z., Weinberger, K. Q. and Cunningham, J. P. (2014). Bayesian optimization with inequality constraints. ICML 2014, PMLR 32(2): 937-945 | A GP per constraint (the values of section 2.5), acquisition EI × Π P(gᵢ ≤ 0); before a feasible point is found, maximize the probability of feasibility. Equalities: as two inequalities with the tolerance, documented | B |
| Multi-objective: ParEGO | Knowles, J. (2006). ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. *IEEE TEVC* 10(1): 50-66. doi:10.1109/TEVC.2005.851274 | A random augmented Tchebycheff scalarization per step, then single-objective EI: cheap, any number of objectives; a `MultiObjectiveAlgorithm` for `MultiEngine` | E |
| Multi-objective: EHVI | Emmerich, M. T. M., Giannakoglou, K. C. and Naujoks, B. (2006). Single- and multiobjective evolutionary optimization assisted by Gaussian random field metamodels. *IEEE TEVC* 10(4): 421-439. doi:10.1109/TEVC.2005.859463. Emmerich, M., Deutz, A. and Klinkenberg, J. W. (2011). Hypervolume-based expected improvement: monotonicity properties and exact computation. IEEE CEC 2011. doi:10.1109/CEC.2011.5949880 | Exact for 2 objectives; 3 by the box decomposition of the non-dominated region; reuses `multi::indicator::hypervolume` | E |
| Mixed and discrete variables | Garrido-Merchán, E. C. and Hernández-Lobato, D. (2020). Dealing with categorical and integer-valued variables in Bayesian optimization with Gaussian processes. *Neurocomputing* 380: 20-35. doi:10.1016/j.neucom.2019.11.004 | Integer genes rounded inside the kernel (batch B for `Integer` genomes); categorical and mixed genes need the mixed genome that ROADMAP.md plans (section 2.6) | B (integer), E (mixed) |
| Noisy observations | GPML ch. 2 | The learned noise variance; `recommendation()` gives the evaluated point with the best posterior mean, while `outcome.best()` stays the best observed value; genoxide's rule that fitness functions are deterministic is what makes runs reproducible, and a noisy function only loses that | B |
| TPE | Bergstra, J., Bardenet, R., Bengio, Y. and Kégl, B. (2011). Algorithms for hyper-parameter optimization. NeurIPS 24. Watanabe, S. (2023). Tree-structured Parzen estimator: understanding its algorithm components and their roles for better empirical performance. arXiv:2304.11127 | Cheaper than a GP per step (linear in the observations), any number of evaluations, integer and categorical genes natively; the multivariate kernel as in Falkner, S., Klein, A. and Hutter, F. (2018). BOHB. ICML 2018. arXiv:1807.01774 | E |
| Trust-region BO (TuRBO-1, TuRBO-m) | Eriksson, D., Pearce, M., Gardner, J., Turner, R. and Poloczek, M. (2019). Scalable global optimization via local Bayesian optimization. NeurIPS 2019. arXiv:1910.01739 | Tens to a few hundred variables and thousands of evaluations: local GPs on the points in a trust region that grows on success and shrinks on failure. Thompson sampling needs a joint posterior sample at thousands of candidates (a Cholesky of that size): fewer candidates, or random Fourier features, decided in batch E | E |

### 1.6 Surrogate-assisted evolution, multi-fidelity, hybrids

Brief and later: each needs the GP (batch B) and a design settled on real use.

- **Surrogate-assisted evolution.** A model pre-screens candidates so only the promising ones are
  evaluated: Jin, Y. (2011). Surrogate-assisted evolutionary computation: recent advances and
  future challenges. *Swarm and Evolutionary Computation* 1(2): 61-70.
  doi:10.1016/j.swevo.2011.05.001; Emmerich et al. (2006) above (pre-screening in an ES); Hansen,
  N. (2019). A global surrogate assisted CMA-ES. GECCO 2019. doi:10.1145/3321707.3321842
  (lq-CMA-ES). Design sketch: a wrapper algorithm `Prescreened<A, M>` that asks `A` for k × λ
  candidates, ranks them by the model and asks the engine for the best λ, telling `A` model values
  for the rest. That needs a way to tell `A` about unevaluated candidates: to design in batch F.
- **Multi-fidelity.** A fitness function with a cost-accuracy parameter; co-kriging: Kennedy, M. C.
  and O'Hagan, A. (2000). Predicting the output from a complex computer code when fast
  approximations are available. *Biometrika* 87(1): 1-13. doi:10.1093/biomet/87.1.1; Forrester,
  A. I. J., Sóbester, A. and Keane, A. J. (2007). Multi-fidelity optimization via surrogate
  modelling. *Proc. R. Soc. A* 463: 3251-3269. doi:10.1098/rspa.2007.1900. Design sketch: the
  genome carries the fidelity as a gene the algorithm sets, so the engine is unchanged.
- **Hybrids.** (1) A global method, then a local polish: CMA-ES or SHADE, then L-BFGS-B or BOBYQA
  from the best, in two engines (`initial_genome(outcome.best_genome())`); an example in batch A2
  shows it, and a combinator comes only if the pattern repeats. (2) Basin hopping: Wales, D. J.
  and Doye, J. P. K. (1997). Global optimization by basin-hopping and the lowest energy structures
  of Lennard-Jones clusters containing up to 110 atoms. *J. Phys. Chem. A* 101(28): 5111-5116.
  doi:10.1021/jp970984n: a restart mode of every local method (perturb the best, accept by the
  Metropolis rule), the same shape as `LocalSearch::restart`. (3) Memetic with gradient steps:
  `Ga::memetic` improves the best parents by mutated neighbors; a local method as the improver
  needs an "improve this genome within k evaluations" operator: later, after (1) and (2).

### 1.7 Others considered

| Method | Decision |
|---|---|
| DIRECT: Jones, D. R., Perttunen, C. D. and Stuckman, B. E. (1993). Lipschitzian optimization without the Lipschitz constant. *JOTA* 79(1): 157-181. doi:10.1007/BF00941892 | Worth having: deterministic, global, low-dimensional, bounded; batch F |
| Coordinate descent: Wright, S. J. (2015). Coordinate descent algorithms. *Math. Programming* 151(1): 3-34. doi:10.1007/s10107-015-0892-3 | Covered by compass search (derivative-free) and not needed as a separate gradient method; not planned |
| Newton's method with an exact Hessian, small problems | Inside the trust-region Newton method (the exact subproblem); not separate |
| Simulated annealing variants for reals (generalized, dual annealing: Tsallis, C. and Stariolo, D. A. (1996). *Physica A* 233: 395-406. doi:10.1016/S0378-4371(96)00271-3) | `LocalSearch` with `Acceptance::Annealing` and a Gaussian neighbor covers annealing on reals; basin hopping (section 1.6) covers the local-global mix; not planned |
| CMA-ES variants (active CMA, VD-CMA, restarts) | CMA-ES work stays in the evolutionary roadmap, not in this plan |
| Multi-start with clustering (MLSL) | Not planned: random restarts and basin hopping first |

### 1.8 Not planned

Reverse-mode automatic differentiation, neural network training frameworks and general-purpose
machine learning; linear, quadratic and mixed-integer programming solvers as products (the dense
QP inside SQP stays internal); large sparse nonlinear programming (sparse factorizations); sparse
or approximate GPs beyond what TuRBO needs. genoxide targets black-box and small-to-medium dense
problems.

## 2. API design

### 2.1 Principles

- **One engine.** The new methods are `Algorithm`s run by `Engine` (and `Incremental`s run by
  `AsyncEngine`, `MultiObjectiveAlgorithm`s run by `MultiEngine`), so stop conditions, observers,
  parallel and batch evaluation, abort flags, `control`, checkpoints, `Evaluated` extras and
  tracing work for them without new code. No second engine for gradients.
- **Nothing changes for what exists.** Every extension is a trait method with a default, so the
  existing algorithms, fitness functions and results stay the same, and the engine's hot path for
  a plain fitness function compiles to what it is today (checked by the instruction-count
  benchmarks).
- **What a fitness function gives, it declares.** A fitness function says whether it gives a
  gradient, constraint values, residuals, Jacobians; an algorithm says what it wants; the engine
  checks the two at the start of a run and returns `Error::InvalidSetting` naming the missing
  piece, with the fix in the reason.
- **Validated, typed configuration**, as for the rest: methods that need `Real` genomes take a
  `Real`; tolerances and memory sizes are builder settings, validated by `build()`.
- **Portable to the bit.** Linear algebra, kernels and special functions use only +, −, ×, ÷,
  `sqrt` and `genoxide::math` (section 2.8).

### 2.2 What exists

The traits the plan extends, as they are (`src/algorithm.rs`, `src/engine.rs`,
`src/algorithm/steady.rs`), doc comments shortened:

```rust
pub trait Algorithm {
    type Genome: Genome;
    fn objective(&self) -> Objective;
    /// The genomes to evaluate next. Asking again before telling gives the same genomes.
    fn ask(&mut self) -> Candidates<'_, Self::Genome>;
    /// The fitness of the genomes of the last ask, in the same order.
    fn tell(&mut self, fitness: &[Fitness]) -> Result<()>;
    fn population(&self) -> &Population<Self::Genome>;
    fn best(&self) -> Option<&Individual<Self::Genome>>;
    fn discarded(&self) -> &[Individual<Self::Genome>] { &[] }
    fn generation(&self) -> u64;
    fn evaluations(&self) -> u64;
    fn best_generation(&self) -> u64;
}

pub trait Reevaluate: Algorithm {
    /// Marks what the algorithm keeps for evaluation by the next ask.
    fn reevaluate(&mut self) -> Result<()>;
}

pub trait FitnessFunction<G>: Sync {
    type Output: IntoFitness; // f64, Fitness, Option<f64>, (f64, f64), Evaluated<..>
    fn evaluate(&self, genome: &G) -> Self::Output;
    fn is_batch(&self) -> bool { false }
    fn evaluate_batch(&self, genomes: &[&G]) -> Vec<Self::Output> { /* one evaluate each */ }
}

pub trait Incremental {
    type Genome: Genome;
    fn objective(&self) -> Objective;
    fn propose(&mut self) -> Self::Genome;
    fn receive(&mut self, genome: Self::Genome, fitness: Fitness)
        -> Result<Option<Individual<Self::Genome>>>;
    fn population(&self) -> &Population<Self::Genome>;
    fn population_size(&self) -> usize;
    fn best(&self) -> Option<&Individual<Self::Genome>>;
    fn evaluations(&self) -> u64;
    fn best_evaluation(&self) -> u64;
}
```

`tell` takes `Fitness` values only: a score, a violation, or invalid. That is the gap. Three
things fit as they are: finite differences (the stencil points are genomes to evaluate), line
searches (each trial step is an ask) and Bayesian optimization of an unconstrained function (one
or q points per ask, or `propose` for `AsyncEngine`).

### 2.3 Derivatives and other extras

**Decision (recommended): extras through the existing traits, declared and requested.** The
`is_batch` / `evaluate_batch` pair of `FitnessFunction` is the model: a defaulted capability
query and a defaulted method.

```rust
/// What a fitness function gives besides the fitness, per genome.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Provided {
    pub gradient: bool,          // ∂score/∂x, one per gene
    pub hessian: bool,           // optional, trust-region Newton (dense, n × n)
    pub inequalities: usize,     // values g(x) <= 0
    pub equalities: usize,       // values h(x) = 0
    pub constraint_jacobian: bool,
    pub residuals: usize,        // least squares: score = ½ Σ rᵢ²
    pub residual_jacobian: bool,
}

/// What an algorithm wants for the genomes of its current ask (a subset of what is provided).
pub struct Wanted { /* the same fields, as flags */ }

/// Buffers the engine hands to the fitness function for one genome: only the wanted ones are Some.
pub struct Extras<'a> { /* gradient, hessian, inequalities, equalities, jacobians, residuals */ }

pub trait FitnessFunction<G>: Sync {
    // ... as today, plus:
    fn provides(&self) -> Provided { Provided::default() }
    fn evaluate_with(&self, genome: &G, extras: &mut Extras<'_>) -> Self::Output {
        self.evaluate(genome)
    }
    fn evaluate_batch_with(&self, genomes: &[&G], extras: &mut BatchExtras<'_>) -> Vec<Self::Output> {
        /* one evaluate_with each */
    }
}

pub trait Algorithm {
    // ... as today, plus:
    /// Called by the engines once at the start of each run with what the fitness function
    /// provides: resolves `Gradients::Auto`, and fails with `InvalidSetting` for what's missing.
    fn prepare(&mut self, provided: Provided) -> Result<()> { Ok(()) }
    /// What the current ask wants besides the fitness. Nothing by default.
    fn wants(&self) -> Wanted { Wanted::NOTHING }
    /// The fitness and the wanted extras of the asked genomes: flat, row-major buffers the engine
    /// reuses (no allocation per generation). `tell(evaluations.fitness())` by default.
    fn tell_evaluations(&mut self, evaluations: &Evaluations<'_>) -> Result<()> {
        self.tell(evaluations.fitness())
    }
    /// Whether the algorithm has converged and has nothing more to do (section 2.7).
    fn is_finished(&self) -> bool { false }
}
```

`Incremental` gets the same `prepare`, `wants` and a `receive_evaluation` default, for constrained
BO under `AsyncEngine`; `MultiObjectiveAlgorithm` gets them for ParEGO and EHVI with constraints.

**The engine.** At the start of `run`: `algorithm.prepare(fitness.provides())?`. Each generation:
if `algorithm.wants()` is empty, the current path, unchanged; otherwise `evaluate_with` per genome
(each writes its own rows, so parallel evaluation gives the same bits) or `evaluate_batch_with`,
then `tell_evaluations`. NaN in an extra follows the `NanPolicy`: `Invalid` makes the whole
evaluation invalid, which a line search treats as a failed step (shrink and retry), the usual way
to handle points where the function isn't defined.

**Wrappers,** like `Batch`, so closures stay the normal way to write a function:

```rust
// a score and its gradient
let rosenbrock = Differentiable(|x: &Reals, gradient: &mut [f64]| { /* fill gradient */ value });

// constraint values one by one; Output is (score, violation), so a GA or SHADE takes it too
let design = Constrained::new(2, 1, |x: &Reals, g: &mut [f64], h: &mut [f64]| { /* ... */ cost })
    .tolerance(1e-6); // |h| <= tolerance counts as met, as constraint::equal

// everything, for SQP with supplied Jacobians
let design = Constrained::differentiable(2, 1, |x: &Reals, extras: &mut Extras| { /* ... */ cost });

// least squares: Output is ½ Σ r², so any algorithm takes it; Levenberg-Marquardt reads r
let fit = LeastSquares::new(40, |x: &Reals, residuals: &mut [f64]| { /* model − data */ });
```

`Differentiable` implements `FitnessFunction<Reals>`: `evaluate` computes the value with a scratch
gradient, so a GA or CMA-ES takes the same function. The gradient is of the score as returned, in
the direction of the objective's own sign (no negation for maximization).

**The test problems** gain what they have: `Problem::constraints` already gives the values of
CEC 2006 and the engineering problems, so their `provides()` declares them and SQP runs
`Engine::new(sqp, cec2006::G07)`; the smooth classic functions (Sphere, ellipsoids, Schwefel 1.2,
Rosenbrock, Zakharov, Styblinski-Tang, Levy, Rastrigin, Ackley, Griewank, Branin, Himmelblau,
Six-hump camel, Goldstein-Price, ...) gain analytic gradients, each tested against central
differences. In Python they're evaluated in Rust, gradient included.

**Gradient source of an algorithm** (a builder setting, `gradient::Gradients`):

| Setting | Meaning |
|---|---|
| `Auto` (default) | Supplied if the fitness function provides a gradient, forward differences otherwise; resolved by `prepare`. Driven by hand (ask / tell without an engine), forward differences |
| `Supplied` | `prepare` fails if the fitness function has none |
| `Forward { step }`, `Central { step }` | Finite differences even if a gradient is provided (e.g. to check it); `step` relative, default as in section 1.3 |

`gradient::check(&function, &x)` compares a supplied gradient with central differences and
returns the largest relative error per gene, for users' tests.

**Alternatives considered.**

| Alternative | Why not |
|---|---|
| Finite differences only, no supplied gradients | Simplest (no trait changes at all), but costs n + 1 evaluations per gradient where an analytic one costs about one: it gives up the reason to use gradient methods at scale. Batch A2 needs the supplied path anyway for the test problems |
| A separate `GradientEngine` and `GradientAlgorithm` trait | Duplicates stop conditions, observers, checkpoints, control, parallel and batch evaluation a fourth time (after `Engine`, `AsyncEngine`, `MultiEngine`) |
| Gradients as `Evaluated` info | The engine never lets info reach the algorithm, by design ("never uses them in the search"); using it would break that guarantee |
| A fitness `Output` type carrying the gradient (`(f64, Vec<f64>)`) | An allocation per evaluation, and `IntoFitness` would carry data the engine must route to `tell`; the buffers of `Extras` avoid both |
| An associated `Evaluation` type on `Algorithm` | Associated type defaults are unstable, so every existing algorithm would change: a breaking change for no gain over a defaulted method |

### 2.4 Methods as ask / tell state machines

- **A generation is one ask / tell round**, as for `LocalSearch`: one Nelder-Mead step (1 point,
  or n on a shrink), one line-search trial, one finite-difference stencil (n or 2n points with
  the base point), one BO batch. `Stop::generations` counts rounds, which is the wall-clock unit
  when each round is evaluated in parallel; each method also reports its own `iterations()`.
- **Stencils ride along.** With finite differences, the trial point and its stencil go in one ask
  (n + 1 points forward, 2n + 1 central), so a line-search trial with a full gradient is one round.
  For Moré-Thuente, the directional derivative is enough at trial points: 2 points per trial, the
  full stencil only at the accepted point.
- **Speculative asks** for expensive, parallel evaluation: Nelder-Mead's reflection, expansion
  and contractions in one ask; several step lengths of a line search in one ask. Each is an option
  that keeps the method's path and spends more evaluations for fewer rounds.
- **`population()`** is the current point (and simplex, for Nelder-Mead; the interpolation set,
  for BOBYQA; the observations, for BO), so `Statistics` and `HallOfFame` work.
- **`best()`** is the best evaluated point by the objective and Deb's rules; stencil points count
  (a stencil point better than the iterate is the best, as for any evaluated genome).
- **Restarts**, a builder setting of every local method: `Restarts::Never` (default), `Random {
  times }` (a new random point in the box, derived stream), `BasinHopping { step, temperature }`.
  A method is `is_finished()` once it has converged with no restart left.
- **Bayesian optimization** asks the initial design as generation 0, then q points per
  generation (`batch(q)`, default 1). It implements `Incremental` as well: `propose` chooses a
  point with the pending ones fantasized (Kriging believer), so `AsyncEngine` keeps every worker
  busy on slow evaluations; reproducible with one worker, like the steady-state GA.

Usage, as the rest of genoxide:

```rust
use genoxide::prelude::*;
use genoxide::problems::{Problem, Rosenbrock};

let problem = Rosenbrock::new(100);
let lbfgsb = Lbfgsb::builder(problem.representation())
    .initial_genome(Reals::from(vec![-1.2; 100])) // optional: a random point by default
    .minimize()
    .seed(1)
    .build()?;
let outcome = Engine::new(lbfgsb, problem) // the problem's analytic gradient, Gradients::Auto
    .stop_when(Stop::evaluations(10_000))
    .run()?;
assert_eq!(outcome.stop_reason(), StopReason::Converged);

let bo = Bo::builder(Real::new([-5.0..=10.0, 0.0..=15.0])?)
    .acquisition(bo::Acquisition::LogExpectedImprovement)
    .batch(4)
    .minimize()
    .seed(1)
    .build()?;
let outcome = Engine::new(bo, problems::Branin).parallel(true).stop_when(Stop::evaluations(60)).run()?;
```

Names follow the existing short ones (`Ga`, `De`, `Es`, `Pso`, `Cmaes`): `NelderMead`, `Lbfgsb`,
`Bfgs`, `ConjugateGradient`, `TrustRegion`, `LevenbergMarquardt`, `Adam`, `Bobyqa`, `Cobyla`,
`Mads`, `Sqp`, `AugmentedLagrangian`, `Bo`, `Tpe`, `Turbo`, `ParEgo`, `Ehvi`. Modules as today:
`algorithm::lbfgsb::{Lbfgsb, LbfgsbBuilder}`, re-exported in the prelude; line searches in
`algorithm::line_search`, the GP in a public `model::gp` (a surrogate users can fit and query on
its own), derivatives in `gradient`.

### 2.5 Constraints for nonlinear programming

- **The values, not only the violation.** SQP, the augmented Lagrangian, COBYLA, MADS and
  constrained BO need gᵢ(x) and hⱼ(x) one by one: `Provided { inequalities, equalities }`,
  written by `Constrained` closures or by a problem's `constraints()`. With only a `(score,
  violation)` tuple, `prepare` fails with a reason that points to `Constrained`.
- **`Constraints` moves to `constraint`.** `problems::Constraints` (inequalities, equalities,
  `violation(tolerance)`) becomes `constraint::Constraints`, re-exported from `problems` so
  nothing breaks.
- **Gradients of constraints**: supplied (`constraint_jacobian`) or finite differences of all
  constraints at once, from the same stencil as the objective's gradient (the stencil points'
  constraint values come back with them), so a constrained FD gradient costs no more evaluations
  than an unconstrained one.
- **One aggregate for `best()`:** Σ max(0, gᵢ) + Σ max(0, |hⱼ| − tolerance), exactly
  `Constraints::violation`, compared by Deb's rules. An NLP method converges to active
  inequalities from either side, and gᵢ = 1e-15 counts as infeasible under Deb's rules with no
  inequality tolerance: see open question 6.
- **Merit functions stay internal:** the ℓ₁ merit of SQP, the augmented Lagrangian's penalty and
  multipliers, the interior point's filter are the algorithm's state (in checkpoints), not the
  fitness.

### 2.6 Representations

- **`Real`** for every continuous method; its bounds are the box of L-BFGS-B, BOBYQA, MADS, the
  finite-difference stencils and the unit-cube scaling of BO. Fixed genes (low = high) are skipped
  by every method, as by CMA-ES.
- **Unbounded problems.** `Real` requires finite widths. A wide box (±1e10, or up to ±f64::MAX / 2)
  is unbounded in practice for a local method that starts from `initial_genome` and touches a
  bound only if the function leads it there. Recommended: no representation change now; see open
  question 3.
- **Scaling.** Initial steps (Nelder-Mead's simplex, BOBYQA's trust radius, MADS's mesh) are
  fractions of each gene's range, as CMA-ES's `initial_step`; gradient methods are scale-sensitive
  by nature, and the docs say to scale the genes.
- **`Integer`** for MADS, TPE and BO (rounding in the kernel). Mixed and categorical genes need a
  mixed genome, planned in ROADMAP.md's architecture ("mixed (per-gene types)") and not designed
  here: batch E depends on it.

### 2.7 Stop conditions and convergence

- **Tolerances are algorithm settings**, not new `Stop` variants: `gradient_tolerance` (projected
  gradient's max norm), `step_tolerance`, `function_tolerance` (relative decrease, as factr of
  L-BFGS-B), simplex size for Nelder-Mead, trust radius for BOBYQA, mesh size for MADS. They need
  the method's state, which `Progress` doesn't have, and their meaning differs by method.
- **`StopReason::Converged`** (the enum is `#[non_exhaustive]`, so adding it isn't breaking): the
  engines stop after a generation in which `algorithm.is_finished()` is true, unless a `Stop`
  condition is met in the same generation (it's reported first). The method's own `converged()`
  says which criterion, like `Cmaes::converged()`.
- **Existing stops keep working:** `Stop::evaluations` counts every evaluation, stencil points
  included (a supplied-gradient evaluation counts as one; `gradient_evaluations()` counts them
  separately); `Stop::target`, `time`, `stagnation`, `custom` as today.
- **CMA-ES and the others** don't implement `is_finished` in this plan: a CMA-ES without restarts
  keeps sampling around its point, as its docs say. Whether it should stop as converged is open
  question 5 (it changes when seeded runs end).

### 2.8 Reproducibility and linear algebra

The guarantee: a seed gives the same results to the bit on every platform and thread count.
IEEE 754 makes +, −, ×, ÷ and `sqrt` exact to the bit; Rust never fuses a × b + c into an FMA
unless `mul_add` is called; LLVM doesn't reorder floating-point sums without fast-math flags. So
plain loops in a fixed order are portable, and autovectorized element-wise loops too. What breaks
it: a reduction whose order depends on the CPU (runtime-dispatched SIMD kernels, FMA kernels) or on
the thread count (rayon's `sum` and `reduce` over splits), and platform `exp`, `ln`, `sin`.

**Decision (2026-09-29): a linear algebra dependency** (faer or nalgebra), chosen in batch A1 by
portability first, then speed, compile time and size. A `pub(crate)` module `linalg` wraps it, so
the rest of the crate depends on one small interface (row-major `Vec<f64>` as CMA-ES uses now, or
the dependency's types behind it):

| Operation | Used by |
|---|---|
| dot, axpy, gemv, symmetric rank-k updates, gemm | everything |
| Cholesky with growing jitter, triangular solves, rank-one updates | BFGS, GP, SQP, trust region |
| Householder QR (with column pivoting) | Levenberg-Marquardt, BOBYQA, least squares |
| Symmetric eigendecomposition (CMA-ES keeps its own tred2 and tql2 of JAMA unless the dependency's is portable and no slower) | CMA-ES, trust-region subproblem |
| LDLᵀ with Bunch-Kaufman pivoting and inertia | interior point, KKT systems |
| Dense strictly convex QP (Goldfarb-Idnani; in the crate if the dependency has none) | SQP |

- **Portability comes first.** As far as their docs show, nalgebra's dynamic-size products call
  the `matrixmultiply` crate above a size threshold, and faer dispatches SIMD kernels at run time,
  FMA among them: two machines could round differently, which breaks genoxide's guarantee. Batch A1
  therefore starts with a test that runs each candidate's operations on CPUs with and without AVX2
  and FMA, and on the CI's Linux, macOS (arm64) and Windows runners, comparing the bits, and looks
  for the way to pin each to a portable path (features, disabling runtime dispatch, fixed kernel
  selection). The dependency that can be pinned is the one used. If neither can be, that's reported
  with the evidence before going further, since the guarantee would then need a documented
  exception for the methods that use it, which is the user's decision.
- **Speed.** The GP's O(N³) Cholesky is the hot spot: blocked, with independent blocks on rayon
  (each output element's sum in a fixed order, so the same bits on any thread count). A criterion
  benchmark against faer, in `benches/` only, measures what portability costs.
- **Rules for the new code,** added to the review checklist: no `mul_add`; no rayon `sum`/`reduce`
  on floats (indexed `map` and a sequential sum instead); `genoxide::math` for every
  transcendental function (`exp` in kernels, `ln` in likelihoods, the normal cdf through a portable
  `erfc`); parallel multi-starts pick the winner by (value, start index).
- **Tests:** CI already runs every example on Linux, macOS and Windows and compares its output to
  `output.txt`; the new examples print enough digits to catch a one-ulp drift.

### 2.9 Reevaluate, control, observers, checkpoints

- **`Reevaluate`**: local methods rescore their current point (and simplex or interpolation set)
  and reset what compares old and new values (the BFGS memory, the trust radius's ratio test);
  the augmented Lagrangian uses it internally when its inner method is a population method. BO
  would rescore every observation, which is the point of it only for a changed function: not
  implemented in batch B, like the steady-state GA.
- **`control`**: setters for what makes sense between generations: UCB's β, the batch size, the
  L-BFGS memory, Adam's learning rate schedule, the augmented Lagrangian's penalty, as DE, PSO and
  local search have setters today.
- **Observers**: `Snapshot` as today. The traces for the site player need new plot kinds: the
  path of a local method on the contour (Nelder-Mead's simplex as a triangle), and for BO the
  posterior mean and the acquisition in 1-D or 2-D (a frame per generation).
- **Checkpoints** (`serde`): all state, including L-BFGS pairs, the GP's data and
  hyperparameters (its Cholesky factor is recomputed on load, to the same bits), the QP's working
  set. A resumed run equals an uninterrupted one, tested as for the other algorithms.
- **`Evaluated` extras** work as they are.

### 2.10 Python

The same shape as today's classes: settings in the constructor, `run(fitness, stop conditions,
...)`, and the problems of `genoxide.problems` evaluated in Rust.

```python
import genoxide as gx

lbfgsb = gx.Lbfgsb(gx.Real((-5, 10), length=100), objective="minimize", memory=10, seed=1)
result = lbfgsb.run(f, gradient=grad_f, evaluations=10_000)   # grad_f(x) -> array
result = lbfgsb.run(f_and_grad, gradient=True, evaluations=10_000)  # f(x) -> (value, gradient)
result = lbfgsb.run(gx.problems.Rosenbrock(100), evaluations=10_000)  # analytic, in Rust
result.stop_reason                                            # "converged"

nm = gx.NelderMead(genome, initial_step=0.1, adaptive=True, objective="minimize", seed=1)

sqp = gx.Sqp(genome, objective="minimize")
result = sqp.run(f, constraints=(2, 1))   # f(x) -> (score, g array, h array)

bo = gx.Bo(genome, acquisition="log-ei", batch_size=4, initial_points=20,
           objective="minimize", seed=1)
result = bo.run(expensive, evaluations=100, parallel=True)
```

- `gradient=`, `constraints=` and a later `residuals=` follow `batch=`: with `batch=True`, 2-D
  arrays in and out (a gradient row per genome).
- `finite_differences="forward" | "central"` and `gradient="auto" | "supplied"` as algorithm
  settings, like Rust's `Gradients`.
- `RunningLbfgsb` and the like for `control`, as `RunningDe` today.
- The `_describe()` / serde enum path of the existing classes; the stub file and docstrings with
  each method's citation.

### 2.11 The `genoxide` program

New `[algorithm] type`s (`nelder-mead`, `lbfgsb`, `sqp`, `bo`, ...). The fitness program
protocol extends by declaration: `gradient = true` (the program prints the value, then n gradient
components), `constraints = { inequalities = 2, equalities = 1 }` (value, then g, then h). With
`bo`, `workers` > 1 runs `AsyncEngine`. docs/cli.md documents each.

## 3. Test problems and validation

### 3.1 Existing problems per family

| Family | Problems genoxide has | Use |
|---|---|---|
| Nelder-Mead, pattern search, BOBYQA | Rosenbrock (2-D from (−1.2, 1), and 10-D), Sphere, axis-parallel ellipsoid, Himmelblau, Branin, Six-hump camel, Goldstein-Price, Zakharov | Convergence to a minimum within tolerance; the multimodal ones reach one of their minima (all are optima) or the global one with restarts |
| Gradient methods | Sphere, ellipsoids, Schwefel 1.2 (ill-conditioned), Rosenbrock (n up to 1000), Zakharov, Styblinski-Tang, Levy, Rastrigin and Ackley (with restarts or basin hopping) | Convergence rates (3.3), scaling with n, supplied vs finite-difference gradients |
| Bounds (L-BFGS-B, BOBYQA, MADS) | Problems whose optimum is on the bound: CEC 2006 box-only parts, Schwefel 2.26 (x* = 420.97 inside [−500, 500]; a shrunk box puts it on the bound), a quadratic with a known active set, built in the test | Active-set identification, projected gradient at the solution |
| SQP, augmented Lagrangian, COBYLA, MADS | CEC 2006 g01-g24 (their `constraints()`; g04, g06, g07, g09, g10 are smooth and small), welded beam, pressure vessel (its continuous relaxation), tension/compression spring, speed reducer, three-bar truss, cantilever beam | Reach the report's f* to 1e-6 relative, feasible within the tolerance; the active constraints the report lists are active |
| BO | Branin, Goldstein-Price, Six-hump camel, Hartmann 3, Hartmann 6, Shekel 5/7/10 (the classic set of Jones et al. 1998) | Evaluations to reach f* + 1e-3 (or relative), over seeds |
| Constrained BO | The engineering problems (low dimension), CEC 2006 g06, g08, g24 (2-D) | As above, feasible |
| ParEGO, EHVI | ZDT1-3 in few variables, Fonseca-Fleming, BNH, SRN, TNK | IGD+ and hypervolume against the analytic fronts after 100-200 evaluations |

### 3.2 New problems

Added under the problem plan's rules (sources, verification status, tests), as batches of
`docs/problems-plan.md`:

- **Moré, Garbow and Hillstrom's test set:** Moré, J. J., Garbow, B. S. and Hillstrom, K. E.
  (1981). Testing unconstrained optimization software. *ACM TOMS* 7(1): 17-41.
  doi:10.1145/355934.355936. The standard set for local unconstrained methods: 35 problems as
  sums of squares, with standard starting points x₀ (and 10x₀, 100x₀) and known minima; the
  residual form serves Levenberg-Marquardt directly (`LeastSquares`). Some overlap with batch 10a
  (Beale, Powell singular); the chained Rosenbrock stays genoxide's, the MGH "extended" one is
  added under its own name. Needed by batch D1; its first half by A2.
- **Hock and Schittkowski's problems:** Hock, W. and Schittkowski, K. (1981). *Test Examples for
  Nonlinear Programming Codes.* LNEMS 187. doi:10.1007/978-3-642-48320-2. A selection (small,
  with equalities, with inequalities, degenerate) for SQP and the augmented Lagrangian; CEC 2006's
  g05, g07, g09, g10 and g13 are from it already. Batch C.
- **A constrained BO toy problem:** Gramacy, R. B., Gray, G. A., Le Digabel, S., Lee, H. K. H.,
  Ranjan, P., Wells, G. and Wild, S. M. (2016). Modeling an augmented Lagrangian for blackbox
  constrained optimization. *Technometrics* 58(1): 1-11. doi:10.1080/00401706.2015.1014065
  (a linear objective with two nonlinear constraints on [0, 1]²). Batch B.
- Each keeps the `Problem` trait, `optimum()`, `reference()`, and `provides()` with analytic
  gradients (and residuals for MGH), tested against central differences.

### 3.3 Tests per method

1. **Convergence to the optimum** on the problems of 3.1, from the papers' starting points where
   they give them (Rosenbrock's (−1.2, 1), MGH's x₀), to the method's tolerance; for multimodal
   ones, to any minimum the problem lists, or with restarts to the global one.
2. **Convergence rates**, measured on the iterates: BFGS and L-BFGS superlinear on Rosenbrock
   near x* (‖xₖ₊₁ − x*‖ / ‖xₖ − x*‖ → 0); Newton quadratic (the ratio to the square bounded); CG
   with exact line search on a convex quadratic: the minimum in at most n iterations; steepest
   descent with exact line search on a quadratic: linear, at most the rate (κ − 1)/(κ + 1) of the
   condition number κ in the quadratic's norm (NW ch. 3).
3. **Finite differences:** forward error O(h), central O(h²), exact where the formula says (a
   quadratic's central difference is exact up to rounding); stencil points inside the box at and
   near the bounds; fixed genes skipped; the same gradient bits with `parallel(true)`, `Batch` and
   sequentially.
4. **Line search:** the strong Wolfe conditions hold at the returned step, on the test functions
   of Moré and Thuente's paper, with its reported steps and counts (the table cited in the test).
5. **L-BFGS-B:** the active set at the solution of a bound-constrained quadratic with a known
   answer; the projected gradient below the tolerance; never a point outside the box.
6. **SQP and the augmented Lagrangian:** KKT conditions at the solution (stationarity,
   feasibility, complementarity, multipliers of the right sign) to the tolerance; CEC 2006's f*
   with the report's active constraints; an infeasible start reaches feasibility.
7. **GP:** the posterior mean interpolates noise-free data; the variance is ~0 at the data and the
   prior's far away; the log marginal likelihood's gradient matches central differences; kernel
   matrices symmetric positive definite (property test); the Cholesky with jitter factors
   near-singular matrices of repeated points.
8. **Acquisitions:** closed-form checks derived by hand (EI at μ = f*, σ = 1 is φ(0) =
   0.3989422804014327; UCB = μ + √β σ; log-EI equals ln(EI) where EI is representable and stays
   finite and monotone where it underflows).
9. **Protocol and guarantees**, as for every algorithm: ask twice gives the same genomes; tell
   errors leave the state unchanged; checkpoint resume equals an uninterrupted run; `Reevaluate`
   draws no random number; seeded runs identical on 1 and 8 threads; the instruction-count
   benchmark of the existing algorithms unchanged by the engine's new path.

### 3.4 Where reference values come from

As rule 3.3 of the problem plan: the papers' tables and theorems (cited per test), hand
derivations written out in the test, and values computed to high precision by an independent
script in the repository (mpmath). Other implementations (SciPy, NLopt, BoTorch) aren't sources
of expected values: they appear in the benchmark suite only. The authors' reference codes may
generate extra test values under the problem plan's conditions (cited, never copied, license
permitting).

## 4. Examples

Every method gets examples in `examples/<name>/` like the rest: `main.rs`, `main.py`, `README.md`
with the front matter and the sections the others have (the problem, what makes it hard,
representation, algorithm, output, good results), `output.txt`, and `trace.json` with
`trace.rs` / `trace.py` for the site player. **Every example reaches its optimum**: the README
states the target and the output shows it reached. Examples of methods get a `category` of their
own on the site (e.g. `local`, `bayesian`), to settle with the site.

| Batch | Examples |
|---|---|
| A1 | `nelder_mead`: Rosenbrock in 2-D from (−1.2, 1), the simplex drawn on the contour; `nelder_mead_himmelblau`: restarts find all four minima |
| A2 | `lbfgsb`: Rosenbrock in 100-D, analytic gradient against forward differences (evaluations to 1e-10); `lbfgsb_bounds`: a problem whose optimum is on the bound; `polish`: SHADE on Rastrigin 10-D, then L-BFGS-B from its best to the exact minimum |
| B | `bayesian_optimization`: Branin in 2-D, posterior and acquisition drawn per step, to f* + 1e-4 with a final L-BFGS-B polish on the GP mean then one evaluation; `bo_hartmann6`: batch BO (q = 4) with `parallel(true)`; `bo_asynchronous`: `AsyncEngine` with evaluations of random duration; `bo_constrained`: Gramacy et al.'s toy problem |
| C | `sqp`: CEC 2006 g07 (or g09) from a random start to the report's f*; `sqp_welded_beam`; `augmented_lagrangian`: a problem with many constraints (g16 or g19); an ε-feasibility comparison with SHADE on the same problem |
| D1 | `conjugate_gradient` on a large quadratic or Rosenbrock 1000-D; `trust_region` on an MGH problem with a Hessian; `levenberg_marquardt`: fitting a model to data (an MGH problem such as Osborne 2, or Bard); `adam`: a small model fit with a learning-rate schedule by `control`; `dual_numbers` (if the feature lands) |
| D2 | `bobyqa` on Rosenbrock 10-D against Nelder-Mead; `cobyla` on a constrained engineering problem; `mads` on a non-smooth or integer problem (the gear train) |
| E | `parego` and `ehvi` on ZDT1 in few variables; `tpe` on an integer or mixed problem; `turbo` on a 20-D or larger problem (Ackley or Rosenbrock) |
| F | `surrogate_cmaes`; `multi_fidelity`; `direct` on the low-dimensional classics |

Per batch, AGENTS.md gains a "which method" table (smooth or not, gradients or not, cheap or
expensive, constrained or not), `docs/features.md` and README's "What's in it" the methods, and
the Python README the classes.

## 5. Benchmarks

Later, and only a note here: the benchmark suite's rules apply (rule 6: one matched method per
problem, each library's own implementation set to a written definition). Candidates, once the
methods exist and are settled:

- **Local methods:** L-BFGS-B against SciPy's L-BFGS-B (based on the authors' code) and NLopt's
  LBFGS; Nelder-Mead against SciPy's (its `adaptive` option is Gao and Han's) and NLopt's; BOBYQA
  against NLopt's and PRIMA; on MGH and BBOB-style functions from the papers' starting points.
  Measures: evaluations and time to a target, final error, gradient evaluations counted apart.
- **Constrained:** SQP against SciPy's SLSQP and NLopt's SLSQP; the interior point (if it lands)
  against SciPy's `trust-constr`; on Hock-Schittkowski and CEC 2006.
- **BO:** against BoTorch, Optuna (TPE, and its GP sampler) and scikit-optimize (if maintained);
  on Branin, Hartmann 6 and Ackley in higher dimensions; measures: the error after a budget of
  evaluations and the model's time per step. A matched definition for BO (kernel, priors,
  acquisition, its optimizer) is harder to write than for a GA; rule 6's "differences that change
  the algorithm" will exclude much, and the rules need a section on it first.

## 6. Order of work

Each batch is a PR (or two) with its methods, their tests, the Python classes, the CLI types, the
docs (module docs citing the sources, AGENTS.md, `docs/features.md`, the README) and its examples;
a batch isn't done until every example reaches its optimum on the three platforms.

| Batch | Contents | Depends on | Examples |
|---|---|---|---|
| A1 | `linalg` (Cholesky, triangular solves, QR, the eigendecomposition moved from CMA-ES) with the dependency check of 2.8; `Algorithm::is_finished` and `StopReason::Converged`; `Restarts` for local methods; Nelder-Mead (Gao-Han, 1965 option, speculative asks) | | `nelder_mead`, `nelder_mead_himmelblau` |
| A2 | The extras of 2.3 (`Provided`, `Wanted`, `Extras`, `Evaluations`, `prepare`, `tell_evaluations`) in `Engine`; `Differentiable`, `gradient::Gradients`, finite differences, `gradient::check`; analytic gradients for the smooth problems; Moré-Thuente; L-BFGS-B | A1 | `lbfgsb`, `lbfgsb_bounds`, `polish` |
| B | `model::gp` (kernels, hyperparameters by L-BFGS-B), portable `erf`/`erfc`/`erfcx` in `math`; `Bo` with EI, log-EI, UCB, PI; Latin hypercube; batch BO (Kriging believer, constant liar); `Incremental` for `AsyncEngine`; `Constrained` values (`constraint::Constraints`) and constrained BO; integer genes | A2 | `bayesian_optimization`, `bo_hartmann6`, `bo_asynchronous`, `bo_constrained` |
| C | Constraint Jacobians in `Extras` (supplied or by finite differences); the dense QP (Goldfarb-Idnani); SQP; the augmented Lagrangian (L-BFGS-B inner); `provides()` for CEC 2006 and the engineering problems; the Hock-Schittkowski selection | A2, B's `Constrained` | `sqp`, `sqp_welded_beam`, `augmented_lagrangian` |
| D1 | BFGS, nonlinear CG with Hager-Zhang, trust-region Newton (Steihaug-CG and exact), Levenberg-Marquardt with `LeastSquares`, momentum / Nesterov / Adam / AdamW; optional `dual` feature; MGH test set | A2 | `conjugate_gradient`, `trust_region`, `levenberg_marquardt`, `adam`, `dual_numbers` |
| D2 | BOBYQA, COBYLA, compass search / GPS, MADS with the progressive barrier, Powell's method (optional); basin hopping | A1, C's constraint values | `bobyqa`, `cobyla`, `mads` |
| E | ParEGO and EHVI (`MultiEngine`); TPE; TuRBO; local penalization; knowledge gradient (optional); mixed-variable BO once the mixed genome exists | B; the mixed genome | `parego`, `ehvi`, `tpe`, `turbo` |
| F | Surrogate-assisted evolution (pre-screening, lq-CMA-ES), multi-fidelity BO, DIRECT, interior point (optional) | B, C | `surrogate_cmaes`, `multi_fidelity`, `direct` |

**Recommendation and rationale.** A1 and A2 first: they build what everything else uses (the
linear algebra, convergence as a stop reason, the extras path of the engine, L-BFGS-B as the inner
solver of B and C) with the two most-used local methods, and they're useful on their own at once
(polishing what CMA-ES and SHADE find). B next: Bayesian optimization is the most-asked-for method
for expensive functions and the one genoxide's parallel and asynchronous engines suit best, and it
needs L-BFGS-B. C then, on the constraint values B introduces. D1 and D2 are independent of each
other and can run in parallel; D2 could move before B if model-based derivative-free methods
(BOBYQA) are wanted sooner for expensive smooth functions. E and F depend on usage and on the
mixed genome.

## 7. Open questions

1. **Linear algebra.** Decided (2026-09-29): a dependency, faer or nalgebra, pinned to a portable
   path; batch A1 checks it first (2.8).
2. **Gradient source default.** `Gradients::Auto` (supplied if the function provides one,
   forward differences otherwise), or an explicit setting validated against the function (no
   magic, one more line in every example)?
3. **Unbounded reals.** Keep `Real`'s finite widths (a wide box for unbounded problems, as
   recommended), or add `Real::unbounded(n)` with infinite bounds that sampling algorithms reject
   at `build()` and local methods accept with an initial genome?
4. **Names.** `Bo` or `BayesianOptimization` (and `Lbfgsb` or `LBfgsB`); `model::gp` or
   `surrogate::gp`; `Differentiable` or `WithGradient`. The crate name stays genoxide.
5. **Convergence for existing algorithms.** Should CMA-ES without restarts, and DE and PSO when
   collapsed, report `is_finished` and stop as `Converged`? It changes when seeded runs end, so
   it's a breaking change of results (`feat!:`), and the benchmark definitions exclude convergence
   criteria (rule 6.4).
6. **Feasibility of active inequalities.** Deb's rules count gᵢ = 1e-15 as infeasible. Options: a
   feasibility tolerance on inequalities in `Constrained` (breaking the CEC 2006 report's
   definition, which has a tolerance on equalities only, unless it defaults to 0 for the
   problems); or the NLP methods end with a restoration step that makes near-active inequalities
   exactly feasible (at most the tolerance's change in the score). Recommended: the second, for
   the problems' semantics, with a tolerance setting for user functions.
7. **Mixed genome.** Design it as part of batch E, or as a separate plan before it (it affects
   GA operators, DE, the CLI and Python too)?
8. **Public `linalg` and `model::gp`.** Public from their first batch (users fitting GPs), or
   crate-private until the API settles?
9. **Milestones.** Decided (2026-09-29, revised the same day): 0.10 is Rust 1.88 and the
   performance work, 0.11 genetic programming and neuroevolution, then these batches: 0.12 local
   methods (A1, A2), 0.13 Bayesian optimization (B), 0.14 constrained nonlinear programming (C),
   0.15 more local methods (D1, D2), 0.16 advanced Bayesian optimization and surrogates (E, F);
   quality-diversity after them. A1 and A2 stay first among them: L-BFGS-B optimizes the Gaussian
   processes' hyperparameters and the acquisition functions of batch B.
