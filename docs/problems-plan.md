# Plan: the test problem library

A working plan, removed when the work is done. genoxide has ZDT1-4, ZDT6 and DTLZ1-4 in
`multi::problems`, and no single-objective, constrained or engineering problems; none are in the
Python package. This plan catalogs the problems to add, designs their API in Rust and Python,
sets how they're tested, and orders the work in batches.

**Rules.**

- Every definition, constant, bound, constraint, optimum and Pareto front comes from the original
  paper, or the standard technical report (e.g. CEC 2006's), and genoxide's docs cite its authors.
- Other libraries' docs were used only to learn which problems exist. Nothing is copied from any
  library: no code, no text, no data files, no expected test values. The docs don't cite
  libraries. Section 5 lists the places that mention pymoo today.
- Each entry has a verification status:
  - **verified-original:** checked in the original paper or report during this survey;
  - **verified-secondary (source):** the original was unavailable (paywalled books, conference
    proceedings); checked in a scholarly source that reprints the definition and cites the
    original, named in the entry. Before implementing, the original is checked if at all possible;
  - **UNVERIFIED:** not checked in any source during this survey; recorded from the common form and
    must be checked in the original before it's implemented.

**Contents.**

1. [Catalog](#1-catalog)
2. [API design](#2-api-design)
3. [Test strategy](#3-test-strategy)
4. [Implementation order](#4-implementation-order)
5. [Mentions of pymoo in the code, tests and docs](#5-mentions-of-pymoo-in-the-code-tests-and-docs)

## 1. Catalog

Counts: see the summary at the end of this section.

### 1.1 Single objective, unconstrained

Most originals here are books or reports that aren't online (Rastrigin 1974, Schwefel 1981,
Ackley 1987, Himmelblau 1972, Dixon and Szegö 1978, Michalewicz 1992, De Jong's 1975 thesis).
The survey read Goldstein and Price (1971) in full, and three scholarly secondary sources: Yao,
Liu and Lin (1999, "Evolutionary programming made faster", IEEE TEVC 3(2): 82-102,
doi:10.1109/4235.771163), which set most of today's common bounds and dimensions (**Y99**); the
CEC 2005 report (Suganthan, Hansen, Liang, Deb, Chen, Auger and Tiwari, 2005, *Problem Definitions
and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization*, NTU and
KanGAL report 2005005) (**CEC05**); and the BBOB definitions (Hansen, Finck, Ros and Auger, 2009,
*Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions*, INRIA
RR-6829, <https://hal.inria.fr/inria-00362633>) (**BBOB**). Jamil and Yang's survey (2013,
arXiv:1308.4008) was consulted but has many transcription errors (listed under pitfalls): no
constant or test value may come from it alone.

Status: **VO** verified-original; **VS(x)** verified-secondary in source x; **VC** checked by hand
from the formula; **U** unverified. "Common bounds" are those of the secondary source named; the
docs of each function say where its bounds come from when the original has none.

#### Scalable, mostly unimodal

| Function | Original | n | Bounds | f*, x* | Status |
|---|---|---|---|---|---|
| Sphere (De Jong's F1) | De Jong, K. A. (1975). *An Analysis of the Behavior of a Class of Genetic Adaptive Systems.* PhD thesis, University of Michigan. hdl:2027.42/4507 | any (De Jong: 3) | De Jong [−5.12, 5.12]; Y99 [−100, 100] | 0 at 0 | VS(Y99); De Jong's n and bounds U |
| Schwefel 1.2 (double sum), Σᵢ(Σⱼ≤ᵢ xⱼ)² | Schwefel, H.-P. (1981). *Numerical Optimization of Computer Models.* Wiley. Problem 1.2 | any | Y99 [−100, 100] | 0 at 0 | VS(Y99, CEC05 F2) |
| Schwefel 2.21, maxᵢ \|xᵢ\| | Schwefel (1981), problem 2.21 | any | Y99 [−100, 100] | 0 at 0 | VS(Y99) |
| Schwefel 2.22, Σ\|xᵢ\| + Π\|xᵢ\| | Schwefel (1981), problem 2.22 | any | Y99 [−10, 10] | 0 at 0 | VS(Y99) |
| Axis-parallel ellipsoid Σ i xᵢ² and rotated hyper-ellipsoid Σᵢ Σⱼ≤ᵢ xⱼ² | U (origin not found) | any | axis-parallel [−5.12, 5.12], rotated [−65.536, 65.536] (Molga and Smutnicki 2005, sections 2.2 and 2.3) | 0 at 0 | U; not the same as Schwefel 1.2 |
| Step (Y99 f6), Σ(⌊xᵢ + 0.5⌋)² | Y99, after De Jong's F3 (Σ⌊xᵢ⌋ on [−5.12, 5.12], n = 5) | any (30) | [−100, 100] | 0 on xᵢ ∈ [−0.5, 0.5) | VS(Y99) for n, bounds, f*; formula U |
| Quartic with noise (De Jong's F4) | De Jong (1975); Y99 f7 | any (30) | [−1.28, 1.28] | 0 at 0 without noise | VS(Y99); the noise (Gaussian in De Jong, uniform [0, 1) in Y99) U |
| Rosenbrock | Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a function. *The Computer Journal* 3(3): 175-184. doi:10.1093/comjnl/3.3.175 (2-D); chained n-D form in De Jong (1975) F2 and Y99 f5 | any | none in the original (start (−1.2, 1)); Y99 [−30, 30] | 0 at (1, …, 1) | citation VO; n-D form VS(Y99, CEC05 F6); f(−1.2, 1) = 24.2 VC |
| Zakharov | U (no primary source found) | any | usually [−5, 10] | 0 at 0 | U |
| Dixon-Price | Dixon, L. C. W. and Price, R. C. (1989). Truncated Newton method for sparse unconstrained optimization using automatic differentiation. *JOTA* 60(2): 261-275. doi:10.1007/BF00940007 | any | [−10, 10] | 0 at xᵢ = 2^(−(2ⁱ − 2)/2ⁱ) | VS(Jamil-Yang, whose x* drops the minus sign); x* for i = 1..3 VC |
| Trid | U (origin not found) | any | [−n², n²] | −n(n+4)(n−1)/6 at xᵢ = i(n+1−i); n = 6: −50, n = 10: −210 | f* VC; origin U |
| Powell singular | Powell, M. J. D. (1962). An iterative method for finding stationary values of a function of several variables. *The Computer Journal* 5(2): 147-151. doi:10.1093/comjnl/5.2.147 | 4 (4k extended) | none (start (3, −1, 0, 1)) | 0 at 0 | U; f(3, −1, 0, 1) = 215 VC; the term is (x₂ − 2x₃)⁴ |
| Sum of different powers Σ\|xᵢ\|^(i+1) | U | any | [−1, 1] | 0 at 0 | U |
| High-conditioned elliptic Σ (10⁶)^((i−1)/(n−1)) zᵢ² | CEC05 F3; BBOB f2, f10 | any | CEC05 [−100, 100]; BBOB [−5, 5] | 0 (+ bias) | VO(CEC05) |
| Bent cigar, Discus, BBOB different powers | BBOB f12, f11, f14 | any | [−5, 5] | f_opt | U |

#### Scalable, multimodal

| Function | Original | Bounds | f*, x* | Status |
|---|---|---|---|---|
| Rastrigin, 10n + Σ(xᵢ² − 10 cos 2πxᵢ) | Rastrigin, L. A. (1974). *Systems of Extremal Control.* Nauka, Moscow (2-D, in Russian); generalized by Mühlenbein, H., Schomisch, M. and Born, J. (1991). The parallel genetic algorithm as function optimizer. *Parallel Computing* 17(6-7): 619-632. doi:10.1016/S0167-8191(05)80052-3 | [−5.12, 5.12] | 0 at 0 | 1991 citation VO; formula VS(CEC05 F9); bounds VS(Y99); the 2-D original U |
| Schwefel 2.26, −Σ xᵢ sin √\|xᵢ\| | Schwefel (1981), problem 2.26 | [−500, 500] | −418.9829 n at xᵢ = 420.9687 | VS(Y99); per-dimension value VC. Three variants circulate (this one; 418.9829 n + this, with f* ≈ 0; a 1/n-scaled one): genoxide implements the paper's and names it |
| Ackley, −20 exp(−0.2 √(Σxᵢ²/n)) − exp(Σ cos(2πxᵢ)/n) + 20 + e | Ackley, D. H. (1987). *A Connectionist Machine for Genetic Hillclimbing.* Kluwer. doi:10.1007/978-1-4613-1997-9 (2-D); n-D form by Bäck, T. (1996). *Evolutionary Algorithms in Theory and Practice.* Oxford University Press | Y99 [−32, 32] (checked in its table I; [−32.768, 32.768] elsewhere); CEC05 [−32, 32] | 0 at 0 | constants VS(CEC05 F8); bounds VS(Y99); the original's dimension U |
| Griewank, 1 + Σxᵢ²/4000 − Π cos(xᵢ/√i) | Griewank, A. O. (1981). Generalized descent for global optimization. *JOTA* 34(1): 11-39. doi:10.1007/BF00933356 | Y99 [−600, 600] | 0 at 0 | citation VO; form VS(Y99, CEC05 F7); the original's d = 200 for n = 2 on [−100, 100] U |
| Levy (the w = 1 + (x − 1)/4 form) | usually credited to Levy, A. V. and Montalvo, A. (1985). The tunneling algorithm for the global minimization of functions. *SIAM J. Sci. Stat. Comput.* 6(1): 15-29. doi:10.1137/0906002 | [−10, 10] | 0 at (1, …, 1) | U: at least four "Levy" functions circulate; the w-form's presence in the 1985 paper is unverified |
| Penalized 1 and 2 (Levy-type, with u(x, a, k, m)) | Y99 f12, f13 | [−50, 50] | 0 at (−1, …) and (1, …) | bounds, f* VS(Y99); formula U |
| Styblinski-Tang, ½ Σ(xᵢ⁴ − 16xᵢ² + 5xᵢ) | Styblinski, M. A. and Tang, T.-S. (1990). Experiments in nonconvex optimization: stochastic approximation with function smoothing and simulated annealing. *Neural Networks* 3(4): 467-483. doi:10.1016/0893-6080(90)90029-K | [−5, 5] | −39.16616570 n at xᵢ = −2.903534 | citation VO; values VC; domain U |
| Michalewicz, −Σ sin(xᵢ) sin²ᵐ(i xᵢ²/π), m = 10 | Michalewicz, Z. (1992). *Genetic Algorithms + Data Structures = Evolution Programs.* Springer | [0, π] | n = 2: −1.8013 at (2.20, 1.57); n = 5: −4.687658; n = 10: −9.66015 | U (values and m) |
| Weierstrass (a = 0.5, b = 3, k_max = 20) | CEC05 F11 (Weierstrass 1872 as a function) | [−0.5, 0.5] | 0 (+ bias) | VO(CEC05); BBOB f16 differs |
| Katsuura | BBOB f23; Katsuura, H. (1991). Continuous nowhere-differentiable functions. *Amer. Math. Monthly* 98(5): 411-416 | [−5, 5] | f_opt | BBOB form VO; 1991 details U |
| HappyCat, HGBat | Beyer, H.-G. and Finck, S. (2012). HappyCat: a simple function class where well-known direct search algorithms do fail. PPSN XII, LNCS 7491: 367-376. doi:10.1007/978-3-642-32937-1_37 | — | 0 at (−1, …, −1) | U (α = 1/8 vs CEC 2014's 1/4) |
| Eggholder | Whitley, D., Mathias, K., Rana, S. and Dzubera, J. (1996). Evaluating evolutionary algorithms. *Artificial Intelligence* 85(1-2): 245-276. doi:10.1016/0004-3702(95)00124-7 | [−512, 512] (original possibly [−512, 511]) | 2-D: −959.6407 at (512, 404.2319) | VS(Jamil-Yang, whose f* has the wrong sign); original bounds U |
| Schaffer F6 and F7 | Schaffer, J. D., Caruana, R. A., Eshelman, L. J. and Das, R. (1989). A study of control parameters affecting online performance of genetic algorithms for function optimization. Proc. 3rd ICGA: 51-60 | [−100, 100] | 0 at 0 | F6 VS(CEC05 F14); F7 U |
| Büche-Rastrigin; BBOB Rastrigin (f3) | BBOB f4, f3 | [−5, 5] | f_opt | VO (BBOB) |
| Shifted, shifted-rotated Rastrigin | CEC05 F9, F10 (shift vectors and matrices in the report's data files) | [−5, 5] | bias −330 | VO(CEC05) |
| Non-continuous Rastrigin | Liang, J. J., Qin, A. K., Suganthan, P. N. and Baskar, S. (2006). Comprehensive learning particle swarm optimizer. *IEEE TEVC* 10(3): 281-295. doi:10.1109/TEVC.2005.857610 | [−5.12, 5.12] | 0 | U |

#### Fixed dimension

| Function | n | Original | Bounds | f*, x* | Status |
|---|---|---|---|---|---|
| Goldstein-Price | 2 | Goldstein, A. A. and Price, J. F. (1971). On descent from local minima. *Mathematics of Computation* 25(115): 569-574. doi:10.1090/S0025-5718-1971-0312365-X | none in the original; [−2, 2] from Dixon and Szegö (1978) | 3 at (0, −1); local minima (1.2, 0.8) → 840, (1.8, 0.2) → 84, (−0.6, −0.4) → 30 (the paper's list: four test points) | **VO** |
| Branin (RCOS) | 2 | Branin, F. H. (1972). Widely convergent method for finding multiple solutions of simultaneous nonlinear equations. *IBM J. Res. Dev.* 16(5): 504-522. doi:10.1147/rd.165.0504; constants as in Dixon and Szegö (1978) | x₁ ∈ [−5, 10], x₂ ∈ [0, 15] | 5/(4π) = 0.3978874 at (−π, 12.275), (π, 2.275), (3π, 2.475) | formula VS(Y99 f17, Jamil-Yang; both misprint 2.475 as 2.425); minima VC |
| Six-hump camel | 2 | Dixon, L. C. W. and Szegö, G. P. (eds.) (1978). *Towards Global Optimisation 2.* North-Holland | Y99 f16 and Jamil-Yang [−5, 5]²; x₁ ∈ [−3, 3], x₂ ∈ [−2, 2] in other papers | −1.0316285 at ±(0.08983, −0.7126) | VS(Y99) to 8 digits; full digits VC (Newton) |
| Three-hump camel | 2 | U (Branin 1972 or Dixon and Szegö 1978) | [−5, 5] | 0 at 0 | VS(Jamil-Yang) |
| Beale | 2 | Beale, E. M. L. (1958). On an iterative method for finding a local minimum of a function of more than one variable. Tech. Rep. 25, Statistical Techniques Research Group, Princeton University | [−4.5, 4.5] | 0 at (3, 0.5) | VS(Jamil-Yang); VC |
| Booth | 2 | U | [−10, 10] | 0 at (1, 3) | VS(Jamil-Yang) |
| Matyas | 2 | U (credited to Matyas, 1965) | [−10, 10] | 0 at 0 | VS(Jamil-Yang) |
| Himmelblau | 2 | Himmelblau, D. M. (1972). *Applied Nonlinear Programming.* McGraw-Hill | [−5, 5] | 0 at (3, 2), (−2.805118, 3.131312), (−3.779310, −3.283186), (3.584428, −1.848126) | (3, 2) VC; the other three U |
| Easom | 2 | Easom, E. E. (1990). *A Survey of Global Optimization Techniques.* M.Eng. thesis, University of Louisville | [−100, 100] | −1 at (π, π) | VS(Jamil-Yang) |
| Bohachevsky 1, 2, 3 | 2 | Bohachevsky, I. O., Johnson, M. E. and Stein, M. L. (1986). Generalized simulated annealing for function optimization. *Technometrics* 28(3): 209-217. doi:10.1080/00401706.1986.10488128 | [−100, 100] | 0 at 0 | VS(Jamil-Yang); whether the paper has variants 2 and 3 U |
| Shekel's foxholes (De Jong's F5) | 2 | De Jong (1975), after Shekel (1971) | [−65.536, 65.536] | ≈ 0.998 at (−32, −32) | n VS(Y99); formula and grid U |
| Hartmann 3 and 6 | 3, 6 | Hartman, J. K. (1973). Some experiments in global optimization. *Naval Research Logistics Quarterly* 20(3): 569-576. doi:10.1002/nav.3800200316; constants tabulated in Dixon and Szegö (1978) | [0, 1]ⁿ | H3: −3.86278 at (0.114614, 0.555649, 0.852547); H6: −3.32237 at (0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301) | VS(Jamil-Yang, Y99 to 2 digits); longer digits U |
| Shekel 5, 7, 10 | 4 | Shekel, J. (1971). Test functions for multimodal search techniques. Proc. 5th Princeton Conf. on Information Sciences and Systems; constants in Dixon and Szegö (1978) | [0, 10]⁴ | −10.1532, −10.4029, −10.5364 near (4, 4, 4, 4) | VS(Y99 f21-f23); Jamil-Yang's values are at exactly (4, 4, 4, 4), not the minima |
| Langermann | 2 (orig. 10) | Bersini, H., Dorigo, M., Langerman, S., Seront, G. and Gambardella, L. (1996). Results of the first international contest on evolutionary optimisation (1st ICEO). Proc. IEEE ICEC: 611-615. doi:10.1109/ICEC.1996.542670 | [0, 10] | ≈ −4.1558 (2-D, 5 terms) | U |
| Kowalik | 4 | Y99 f15 (data fit, 11 points) | [−5, 5] | ≈ 3.075e-4 | U |

**CEC and BBOB suites as wholes** (optional, batch 10+): CEC 2005 F1-F25 (VO: shift vectors,
matrices and biases in the report's data files, which are the organizers' data and may be used
with citation); BBOB's 24 functions (instances generated per seed by the definitions; no fixed
values); CEC 2013/2014/2017 (NTU reports 201212, 201311 and Awad et al. 2016, U). genoxide's
`Shifted` and `Rotated` wrappers give the same kind of instances without shipping data; the
exact suites need their data files, which is a separate decision.

#### Binary and combinatorial (maximized)

| Problem | Original | Optimum | Status |
|---|---|---|---|
| OneMax | folklore; analyzed by Mühlenbein, H. (1992). How genetic algorithms really work: mutation and hillclimbing. PPSN II | n at 1ⁿ | U (no single origin) |
| LeadingOnes | Rudolph, G. (1997). *Convergence Properties of Evolutionary Algorithms.* Kovač, Hamburg | n at 1ⁿ | U |
| Deceptive trap (k bits per block) | Ackley (1987); Deb, K. and Goldberg, D. E. (1993). Analyzing deception in trap functions. FOGA 2: 93-108 | n at 1ⁿ (attractor 0ⁿ) | U |
| Royal road R1, R2 | Mitchell, M., Forrest, S. and Holland, J. H. (1992). The royal road for genetic algorithms. Proc. 1st ECAL: 245-254; Forrest and Mitchell (1993), FOGA 2 | 64 at 1⁶⁴ | U |
| NK landscapes | Kauffman, S. A. and Weinberger, E. D. (1989). The NK model of rugged fitness landscapes and its application to maturation of the immune response. *J. Theor. Biol.* 141(2): 211-245. doi:10.1016/S0022-5193(89)80019-0 | instance-specific; exact by dynamic programming for adjacent neighborhoods | U |
| 0/1 knapsack instance classes | Pisinger, D. (2005). Where are the hard knapsack problems? *Computers & Operations Research* 32(9): 2271-2284. doi:10.1016/j.cor.2004.03.002 | from the generated instance (dynamic programming in the test) | U |

**Pitfalls to settle per function when implementing:** Schwefel's numbering (1.2, 2.21, 2.22,
2.26, and the shifted 418.9829 n form); Ackley's bounds; Griewank's divisor and domain; the
Michalewicz m; which "Levy"; Trid's n-dependent bounds; the rotated hyper-ellipsoid is not
Schwefel 1.2; Powell's (x₂ − 2x₃)⁴; Eggholder's minimum on the bound; Shekel and Hartmann minima
from Dixon and Szegö, not from Jamil and Yang; Branin's third minimum (3π, 2.475); Goldstein-Price
has no domain in its paper; Rosenbrock's chained form (Y99, CEC05) versus the pairwise
"extended" form of Moré, Garbow and Hillstrom (1981, ACM TOMS 7(1): 17-41,
doi:10.1145/355934.355936); Schaffer F6 has √(x² + y²) inside sin².

**Hand-computed test values (VC):** Rosenbrock (−1.2, 1) = 24.2; Powell (3, −1, 0, 1) = 215;
Goldstein-Price (0, −1) = 3 and its three local minima's values from the paper; Beale (3, 0.5) =
0; Branin at its three minima = 5/(4π); Styblinski-Tang −39.1661657 per dimension at −2.903534;
Schwefel 2.26 −418.9829 per dimension at 420.968746; Trid n = 6: −50, n = 10: −210.

### 1.2 Single objective, constrained: CEC 2006 (g01-g24)

**Source:** Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N.,
Coello Coello, C. A. and Deb, K. (2006). *Problem Definitions and Evaluation Criteria for the
CEC 2006 Special Session on Constrained Real-Parameter Optimization.* Technical report, Nanyang
Technological University, Singapore, 18 September 2006. PDF: the organizers' repository,
<https://github.com/P-N-Suganthan/CEC2006> (`technical_report.pdf`). Everything below is
**verified against this report** unless marked otherwise; page numbers are the report's.

**Pin the September 2006 version.** An earlier (December 2005) version differs in the property
table (g02's ρ, g18's NI and a, and the active counts of g19, g20, g22, g23). The g20 and g22 x*
of the September version come from Takahama and Sakai's εDE (CEC 2006).

**Common definitions (p. 2, pp. 17-19).** Minimize f(x) subject to g_i(x) ≤ 0 and h_j(x) = 0;
an equality counts as satisfied when |h_j(x)| − ε ≤ 0, with **ε = 0.0001**. All variables are
continuous (g12's p, q, r are indices inside its constraint, not variables). Evaluation: 25 runs,
at most 500,000 evaluations, errors f(x) − f(x*) recorded at 5·10³, 5·10⁴ and 5·10⁵ evaluations;
a run succeeds when it finds a feasible x with f(x) − f(x*) ≤ 0.0001. The mean violation is
v̄ = (Σ max(0, g_i) + Σ |h_j| [where |h_j| > ε]) / m.

**Original sources**, as the report cites them: Floudas and Pardalos (1990, LNCS 455,
doi:10.1007/3-540-53032-0): g01, g06; Himmelblau (1972, *Applied Nonlinear Programming*,
McGraw-Hill): g04, g14-g20; Hock and Schittkowski (1981, LNEMS 187, doi:10.1007/978-3-642-48320-2):
g05, g07, g09, g10, g13; Koziel and Michalewicz (1999, Evolutionary Computation 7(1): 19-44,
doi:10.1162/evco.1999.7.1.19): g02, g08, g11, g12; Michalewicz, Nazhiyath and Michalewicz (1996,
Proc. 5th Conf. on Evolutionary Programming, pp. 305-312): g03; Epperly (test problems with
solutions): g21, g22; Xia (<http://www.mat.univie.ac.at/~neum/glopt/xia.txt>): g23; Floudas et
al. (1999, *Handbook of Test Problems in Local and Global Optimization*, Kluwer,
doi:10.1007/978-1-4757-3040-1): g24. The docs cite the report and, per problem, its source.

Table 3 (p. 16) and the problem texts; ρ is the estimated feasible fraction of the box; LI, NI,
LE, NE count linear/nonlinear inequalities/equalities; a is the number of active constraints at
x*. "Max" marks an objective negated from a maximization (inferred from the report's
`f = −…`; the report itself only states minimizations).

| Problem | n | Objective | ρ | LI | NI | LE | NE | a | Max | f(x*), all digits the text gives | Page |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g01 | 13 | quadratic | 0.0111% | 9 | 0 | 0 | 0 | 6 | | −15 | 3 |
| g02 | 20 | nonlinear | 99.9971% | 0 | 2 | 0 | 0 | 1 | yes | −0.80361910412559 | 3 |
| g03 | 10 | polynomial | 0.0000% | 0 | 0 | 0 | 1 | 1 | yes | −1.00050010001000 | 3-4 |
| g04 | 5 | quadratic | 52.1230% | 0 | 6 | 0 | 0 | 2 | | −30665.53867178332 | 4 |
| g05 | 4 | cubic | 0.0000% | 2 | 0 | 0 | 3 | 3 | | 5126.4967140071 | 4 |
| g06 | 2 | cubic | 0.0066% | 0 | 2 | 0 | 0 | 2 | | −6961.81387558015 | 4 |
| g07 | 10 | quadratic | 0.0003% | 3 | 5 | 0 | 0 | 6 | | 24.30620906818 | 5 |
| g08 | 2 | nonlinear | 0.8560% | 0 | 2 | 0 | 0 | 0 | yes | −0.0958250414180359 | 5 |
| g09 | 7 | polynomial | 0.5121% | 0 | 4 | 0 | 0 | 2 | | 680.630057374402 | 5-6 |
| g10 | 8 | linear | 0.0010% | 3 | 3 | 0 | 0 | 6 | | 7049.24802052867 | 6 |
| g11 | 2 | quadratic | 0.0000% | 0 | 0 | 0 | 1 | 1 | | 0.7499 | 6 |
| g12 | 3 | quadratic | 4.7713% | 0 | 1 | 0 | 0 | 0 | yes | −1 | 6 |
| g13 | 5 | nonlinear | 0.0000% | 0 | 0 | 0 | 3 | 3 | | 0.053941514041898 | 6-7 |
| g14 | 10 | nonlinear | 0.0000% | 0 | 0 | 3 | 0 | 3 | | −47.7648884594915 | 7 |
| g15 | 3 | quadratic | 0.0000% | 0 | 0 | 1 | 1 | 2 | | 961.715022289961 | 7 |
| g16 | 5 | nonlinear | 0.0204% | 4 | 34 | 0 | 0 | 4 | | −1.90515525853479 | 7-10 |
| g17 | 6 | nonlinear | 0.0000% | 0 | 0 | 0 | 4 | 4 | | 8853.53967480648 (see errata) | 10 |
| g18 | 9 | quadratic | 0.0000% | 0 | 13 | 0 | 0 | 6 | yes (−area) | −0.866025403784439 | 11 |
| g19 | 15 | nonlinear | 33.4761% | 0 | 5 | 0 | 0 | 0 | | 32.6555929502463 | 11-12 |
| g20 | 24 | linear | 0.0000% | 0 | 6 | 2 | 12 | 16 | | 0.2049794002 (Table 4; x* slightly infeasible) | 12-13 |
| g21 | 7 | linear | 0.0000% | 0 | 1 | 0 | 5 | 6 | | 193.724510070035 | 13 |
| g22 | 22 | linear | 0.0000% | 0 | 1 | 8 | 11 | 19 | | 236.430975504001 | 13-14 |
| g23 | 9 | linear | 0.0000% | 0 | 2 | 3 | 1 | 6 | | −400.055099999999584 | 14-15 |
| g24 | 2 | linear | 79.6556% | 0 | 2 | 0 | 0 | 2 | yes (−x₁ − x₂) | −5.50801327159536 | 15 |

"Max" for g02, g03, g08 and g12 agrees with Runarsson and Yao (2000, IEEE TEVC 4(3): 284-294,
doi:10.1109/4235.873238) and Michalewicz and Schoenauer (1996, Evolutionary Computation 4(1):
1-32, doi:10.1162/evco.1996.4.1.1) as commonly cited, **not re-checked in those papers**.

**Bounds** (Table 4, p. 17): g01 x₁₋₉ ∈ [0, 1], x₁₀₋₁₂ ∈ [0, 100], x₁₃ ∈ [0, 1]; g02 x ∈ (0, 10]
(open at 0); g03 x ∈ [0, 1]; g04 x₁ ∈ [78, 102], x₂ ∈ [33, 45], x₃₋₅ ∈ [27, 45]; g05 x₁,₂ ∈
[0, 1200], x₃,₄ ∈ [−0.55, 0.55]; g06 x₁ ∈ [13, 100], x₂ ∈ [0, 100]; g07 [−10, 10]; g08 [0, 10];
g09 [−10, 10]; g10 x₁ ∈ [100, 10000], x₂,₃ ∈ [1000, 10000], x₄₋₈ ∈ [10, 1000]; g11 [−1, 1];
g12 [0, 10]; g13 x₁,₂ ∈ [−2.3, 2.3], x₃₋₅ ∈ [−3.2, 3.2]; g14 (0, 10] (open: logarithms);
g15 [0, 10]; g16 x₁ ∈ [704.4148, 906.3855], x₂ ∈ [68.6, 288.88], x₃ ∈ [0, 134.75],
x₄ ∈ [193, 287.0966], x₅ ∈ [25, 84.1988]; g17 x₁ ∈ [0, 400], x₂ ∈ [0, 1000], x₃,₄ ∈ [340, 420],
x₅ ∈ [−1000, 1000], x₆ ∈ [0, 0.5236]; g18 x₁₋₈ ∈ [−10, 10], x₉ ∈ [0, 20]; g19 [0, 10]; g20
[0, 10]; g21 x₁ ∈ [0, 1000], x₂,₃ ∈ [0, 40], x₄ ∈ [100, 300], x₅ ∈ [6.3, 6.7], x₆ ∈ [5.9, 6.4],
x₇ ∈ [4.5, 6.25]; g22 (22 ranges on p. 14 and Table 4); g23 x₁,₂,₆ ∈ [0, 300], x₃,₅,₇ ∈ [0, 100],
x₄,₈ ∈ [0, 200], x₉ ∈ [0.01, 0.03]; g24 x₁ ∈ [0, 3], x₂ ∈ [0, 4]. `Real` needs closed bounds:
g02 and g14 use a lower bound of the smallest positive value that keeps the logarithm and the
division finite (documented), and their evaluate returns an invalid fitness at 0.

**x\*.** The report prints x* for every problem on the page given above, to 15-18 digits; the
tests use those. Printing errors to repair: g04's x* lacks an opening parenthesis; **g23's x\*
prints 8 numbers for n = 9**: a comma is missing, x₈ = 200 and x₉ = 0.0100000100000100008 (then
f = −400.0551, checked by hand); **g24's x\* is printed as "2.329520197477623.17849307411774"**:
x = (2.32952019747762, 3.17849307411774) (the sum gives f*, checked by hand); g16's x₂* is 5.7e-15
below its lower bound 68.6 (clamp it in tests).

**Data tables:** g19 needs the report's Table 1 (p. 12: e, c, d, a; b is inline on p. 11); g20
needs Table 2 (p. 13: a, b, c, d, e; k = 0.7302 × 530 × 14.7/40). g14's c₁..c₁₀ and g16's long
chain of intermediate quantities (eqs. 32-34, pp. 7-10) are inline. The report's "Table 1" and
"Table 2" are these data; the property table is Table 3 and the bounds table Table 4.

**Test tolerance.** Table 4's 10-decimal f* differs from the texts' values in the last digit for
g02, g04, g09, g16 (rounded up) and g10, g15, g17 (truncated): tests compare to the texts' values
with a tolerance of 1e-9 relative.

**Errata and caveats.**

1. g03, g05, g11 and g13 reach their f* only thanks to ε: g03's exact optimum is −1 (the listed
   −1.0005001 = −(1.0001)⁵ has Σx² = 1.0001); g11's exact optimum is 0.75. Solvers may report tiny
   negative errors (Takahama and Sakai 2006 report down to −1.2e-12).
2. **g17:** evaluating the report's own x* gives 30x₁ + 28x₂ = 8853.534016…, below the stated
   f* = 8853.53967480648; later papers use **8853.5338748065** (e.g. Xu, He and Shang,
   arXiv:1903.04886, Table VIII). **Which paper first reported it is unverified.** The objective
   jumps at x₂ = 100 and x₁ = 300 (piecewise), x₂* sits just below 100, and f₁ is defined for
   x₁ < 400 and f₂ for x₂ < 1000 while the bounds include 400 and 1000: the implementation picks the
   closed end and documents it.
3. **g20** has no known feasible solution: the printed x* is slightly infeasible (Σx ≈ 1.0001).
   `optimum()` returns the value as a best known, not proven, with that note.
4. **g22:** the earlier best known was 382.902205; 236.430975504001 is εDE's. Spettel, Ba and
   Arnold (2022, Evolutionary Computation 30(4): 531-553, doi:10.1162/evco_a_00311) state a better
   value exists: **unverified (paywalled)**.
5. g10: Table 3 says a = 6, the text names only g1-g3 as active.
6. g04: some engineering papers use Himmelblau's variant with 0.00026 x₁x₄ in g1 (optimum about
   −31025.56); the CEC form uses 0.0006262 and has −30665.539. genoxide implements the CEC form.
   **The variant's value is unverified.**
7. Typography only: "93 disjointed spheres" in g12 means 9³ = 729; "+ +2x₃" in g14's h1 and
   "+ +x₃²" in g15's h1; unbalanced parentheses in g17's h2 and h3.
8. g12's constraint is a disjunction: min over p, q, r ∈ {1..9} of (x₁−p)² + (x₂−q)² + (x₃−r)² −
   0.0625 ≤ 0, one NI constraint.

### 1.3 Multi objective, unconstrained

Read in full during the survey: Zitzler, Deb and Thiele (2000), the DTLZ technical report (2001)
and the DTLZ CEC 2002 paper. The WFG, MaF, CEC 2009, Deb and Jain (2014), Kursawe, Poloni,
Viennet and Van Veldhuizen sources were paywalled or blocked: those entries are **U** and must be
checked in the original before implementation (the WFG and MaF papers first: MaF's is open
access at Springer; WFG's through a library).

#### Classic problems

| Problem | Original | n, M | Bounds | Pareto set / front | Status |
|---|---|---|---|---|---|
| Schaffer 1 (SCH1): x², (x − 2)² | Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic algorithms. Proc. 1st ICGA: 93-100 (and his 1984 thesis, Vanderbilt University) | 1, 2 | the domain varies: [−A, A] with A from 10 to 10⁵ (Deb's book; Van Veldhuizen's MOP1 [−10⁵, 10⁵]) | PS x ∈ [0, 2]; PF f₂ = (√f₁ − 2)², f₁ ∈ [0, 4] | functions and PS VS(DTLZ report §3); domain U |
| Schaffer 2 (SCH2): piecewise f₁, f₂ = (x − 5)² | Schaffer (1985) | 1, 2 | [−5, 10] | PS [1, 2] ∪ [4, 5]: disconnected | existence VS(DTLZ report); formula and bounds U |
| Fonseca-Fleming (FON) | Fonseca, C. M. and Fleming, P. J. (1995). An overview of evolutionary algorithms in multiobjective optimization. *Evolutionary Computation* 3(1): 1-16. doi:10.1162/evco.1995.3.1.1 | any (usually 3), 2 | [−4, 4] | PS x₁ = … = xₙ = t, t ∈ [−1/√n, 1/√n]; PF f₂ = 1 − exp(−(2 − √(−ln(1 − f₁)))²) (derived) | VS(DTLZ report eq. 1); PF VC; the original's n U. The variables must be equal on the PS (the DTLZ report's wording omits it) |
| Kursawe (KUR) | Kursawe, F. (1991). A variant of evolution strategies for vector optimization. PPSN I, LNCS 496: 193-197. doi:10.1007/BFb0029752 | 3, 2 | [−5, 5] | disconnected; no closed form (sample the PS) | U: sin(xᵢ³) vs sin(xᵢ)³ differs between sources |
| Poloni (POL) | Poloni, C., Giurgevich, A., Onesti, L. and Pediroda, V. (2000). Hybridization of a multi-objective genetic algorithm, a neural network and a classical optimizer for a complex design problem in fluid dynamics. *Computer Methods in Applied Mechanics and Engineering* 186(2-4): 403-420. doi:10.1016/S0045-7825(99)00394-1 | 2, 2 | [−π, π]² | disconnected | citation VS(DTLZ report); formula U (the original maximizes) |
| Viennet 1-3 (VNT1-3) | Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic algorithm for determining a Pareto set. *International Journal of Systems Science* 27(2): 255-260. doi:10.1080/00207729608929211 | 2, 3 | VNT1 [−2, 2]², VNT2 [−4, 4]², VNT3 [−3, 3]² (Van Veldhuizen uses other bounds) | sampled | U |
| Deb's two-objective problems (multimodal, disconnected, biased) | Deb, K. (1999). Multi-objective genetic algorithms: problem difficulties and construction of test problems. *Evolutionary Computation* 7(3): 205-230. doi:10.1162/evco.1999.7.3.205 | 2, 2 | [0.1, 1] / [0, 1]² | global front at x₂ = 0.2, local at ≈ 0.6 (multimodal problem) | U (optional) |

#### ZDT5 and a check of ZDT1-4, ZDT6

Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary
algorithms: empirical results. *Evolutionary Computation* 8(2): 173-195.
doi:10.1162/106365600568202. Definition 4, eqs. 6-12. **VO** (the existing ZDT1-4, ZDT6 match:
m = 30, 30, 30, 10, 10 variables; [0, 1], with ZDT4's x₂…x₁₀ ∈ [−5, 5]).

- **ZDT5** (eq. 11): 11 binary substrings, x₁ of 30 bits and x₂…x₁₁ of 5 bits (80 bits).
  f₁ = 1 + u(x₁) (u = number of ones), g = Σᵢ₌₂¹¹ v(u(xᵢ)) with v(u) = 2 + u if u < 5 and 1 if
  u = 5, f₂ = g / f₁. **The Pareto front has g = 10** (x₂…x₁₁ all ones): f₂ = 10/f₁ at the 31
  points f₁ = 1…31. The best deceptive front has g = 11; the paper plots g = 20 (every substring
  at the deceptive attractor) for reference. All fronts convex. `Binary(80)`; `optimal_front`
  returns the 31 points.
- ZDT4: global front at g = 1, best local front at g = 1.25, 21⁹ local fronts. ZDT6's front
  starts at f₁ ≈ 0.2808 (the constant already in genoxide, derived from the formula).
- The paper ships no front files; the fronts are analytic.

#### DTLZ5-DTLZ9 and the numbering

Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). *Scalable Test Problems for
Evolutionary Multi-Objective Optimization.* TIK-Report 112, ETH Zürich.
<https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf>; and (2002). Scalable
multi-objective optimization test problems. Proc. IEEE CEC 2002: 825-830.
doi:10.1109/CEC.2002.1007032. **VO.** The 2005 book chapter (*Evolutionary Multiobjective
Optimization*, Springer, doi:10.1007/1-84628-137-7_6) is believed to follow the report's
numbering: **U**. genoxide follows the report's numbering, which is the common one, and says so.

All x ∈ [0, 1]ⁿ, n = M + k − 1.

| Problem (report) | k default | Front | Notes |
|---|---|---|---|
| DTLZ5 (eq. 25) | 10 | claimed a curve; x_M = 0.5 | **The report's eq. 25 has typos** (cos(θᵢ π/2), θ₁ undefined): read θ₁ = x₁ π/2 and θᵢ = π/(4(1 + g)) (1 + 2g xᵢ) for i = 2…M − 1, f with cos/sin(θᵢ). The front is not a curve for M ≥ 4 (Huband et al. 2006: U) |
| DTLZ6 (eq. 26) | 10 | as DTLZ5, g = Σ xᵢ^0.1 | same caveat |
| DTLZ7 (eq. 27) | **20** | 2^(M−1) disconnected regions; x_M = 0 (g = 1), f_M = 2h | the front is built by sampling f₁…f_{M−1} ∈ [0, 1] and filtering non-dominated points; the region bounds per fᵢ (≈ [0, 0.2514] ∪ [0.6316, 0.8594]) U, derived at implementation |
| DTLZ8 (eq. 28), constrained | n = 10M | a line (f₁ = … = f_{M−1} = t, f_M = 1 − 4t, t ∈ [0, 1/6]) and a hyperplane 2f_M + fᵢ + fⱼ = 1, f_M ∈ [0, 1/3] (derived) | M constraints; **index erratum:** the block sums start at ⌊(j − 1)n/M⌋ + 1 (1-based) |
| DTLZ9 (eq. 29), constrained | n = 10M | f₁ = … = f_{M−1}, f_M² + fⱼ² = 1 | M − 1 constraints |

- **Erratum (DTLZ1):** the report's text says the Pareto set is x_M = 0; it is x_M = 0.5 (the
  CEC paper's eq. 8). genoxide's DTLZ1 uses 0.5 already.
- **Numbering:** the CEC 2002 paper has seven problems: its DTLZ5 is the report's DTLZ6, its
  DTLZ6 the report's DTLZ7 and its DTLZ7 the report's DTLZ8; the report's DTLZ5 and DTLZ9 aren't
  in it. The docs of DTLZ5-7 say which numbering they follow.

#### Scaled, convex and inverted DTLZ

Deb, K. and Jain, H. (2014). An evolutionary many-objective optimization algorithm using
reference-point-based nondominated sorting approach, part I. *IEEE TEVC* 18(4): 577-601.
doi:10.1109/TEVC.2013.2281535. Jain, H. and Deb, K. (2014), part II. *IEEE TEVC* 18(4): 602-622.
doi:10.1109/TEVC.2013.2281534. **U** (both paywalled; to check in the papers):

- Scaled DTLZ1 and DTLZ2: fᵢ multiplied by s^(i−1), with s per M from a table in part I
  (recalled as 10 for M = 3 and 5, 3 for 8, 2 for 10, 1.2 for 15: U).
- Convex DTLZ2 (part I): fᵢ⁴ for i < M, f_M².
- Inverted DTLZ1 (part II): fᵢ ← 0.5 (1 + g) − fᵢ.

#### WFG1-WFG9

Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
problems and a scalable test problem toolkit. *IEEE TEVC* 10(5): 477-506.
doi:10.1109/TEVC.2005.861417. Earlier: Huband, S., Barone, L., While, L. and Hingston, P.
(2005). A scalable multi-objective test problem toolkit. EMO 2005, LNCS 3410: 280-295.
doi:10.1007/978-3-540-31880-4_20. **U: the full text couldn't be reached during the survey.**
The structure below is the commonly described one and must be checked against the TEVC paper's
tables before implementation:

- f_m = D x_M + S_m h_m(x₁…x_{M−1}), D = 1, S_m = 2m; zᵢ ∈ [0, 2i]; n = k + l, k a multiple
  of M − 1, l even for WFG2 and WFG3; typical k = 2(M − 1) (or 4), l = 20.
- Optimal distance parameters zᵢ = 0.35 × 2i (WFG1-7); WFG8 and WFG9's depend on the other
  parameters and are computed in sequence. The front is sampled from the position parameters at
  the optimal distance values.
- Shapes and features: WFG1 mixed convex, biased (polynomial) and flat regions; WFG2 disconnected
  convex; WFG3 linear, claimed degenerate but **not degenerate for M ≥ 3** (Ishibuchi, H.,
  Masuda, H. and Nojima, Y. (2016). Pareto fronts of many-objective degenerate test instances.
  *IEEE TEVC* 20(5): 807-813; DOI U); WFG4 multimodal concave; WFG5 deceptive concave; WFG6
  non-separable concave; WFG7 parameter-dependent bias (position); WFG8 parameter-dependent bias
  (distance), its Pareto set not at 0.35; WFG9 multimodal, deceptive and non-separable.
- The WFG group's C++ toolkit (University of Western Australia, formerly
  www.wfg.csse.uwa.edu.au) is the authors' reference implementation; its availability and
  license are to be checked before using it to generate test values (section 3.3).

#### Many-objective suites (optional, batch 13)

- **MaF1-MaF15:** Cheng, R., Li, M., Tian, Y., Zhang, X., Yang, S., Jin, Y. and Yao, X. (2017). A
  benchmark test suite for evolutionary many-objective optimization. *Complex & Intelligent
  Systems* 3: 67-81. doi:10.1007/s40747-017-0039-7. Built on DTLZ, WFG and LSMOP (MaF1 inverted
  linear, MaF2 concave, MaF3 convex multimodal, MaF4 inverted badly scaled, MaF5 badly scaled
  biased, MaF6 degenerate, MaF7 disconnected, MaF8-9 distance minimization in 2-D, MaF10-12 =
  WFG1, WFG2, WFG9, MaF13 degenerate, MaF14-15 large-scale). Fronts from the paper's formulas.
  **U** (open access: check first).
- **CEC 2009 UF1-UF10:** Zhang, Q., Zhou, A., Zhao, S., Suganthan, P. N., Liu, W. and Tiwari, S.
  (2008). *Multiobjective Optimization Test Instances for the CEC 2009 Special Session and
  Competition.* Technical Report CES-487, University of Essex. n = 30; UF1-7 two objectives,
  UF8-10 three; fronts analytic (UF5 has 21 points, UF6 and UF9 disconnected). **U.**

### 1.4 Multi objective, constrained

Read during the survey: the NSGA-II paper (Deb, Pratap, Agarwal and Meyarivan, 2002), Jain and
Deb's part II (as the authors' preprint, IITK report 2012010,
<https://www.egr.msu.edu/~kdeb/papers/k2012010.pdf>), Ma and Wang's MW paper and supplement (as
hosted by the authors, <https://intleo.csu.edu.cn/codes/MW.pdf>), DAS-CMOP and LIR-CMOP (arXiv
versions), the CEC 2009 report (20 April 2009 draft), the C-TAEA supplement (DC-DTLZ) and the DTLZ
report. BNH, OSY, the Van Veldhuizen constrained problems and all of CTP could not be reached:
**U**. The PDFs were read as extracted text, so load-bearing constants get a check against the
typeset paper when implemented. Constraints below are written as the papers print them (mostly
"≥ 0 is feasible"); genoxide normalizes to g ≤ 0 (section 2).

#### Classic two-objective problems

| Problem | Original | n, bounds | Objectives | Constraints | Pareto front | Status |
|---|---|---|---|---|---|---|
| CONSTR | Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE TEVC* 6(2): 182-197. doi:10.1109/4235.996017, Table V | 2; x₁ ∈ [0.1, 1], x₂ ∈ [0, 5] | f₁ = x₁, f₂ = (1 + x₂)/x₁ | x₂ + 9x₁ ≥ 6; −x₂ + 9x₁ ≥ 1 | derived: x₂ = 6 − 9x₁ for x₁ ∈ [7/18, 2/3] (f₂ = (7 − 9f₁)/f₁), then x₂ = 0 for x₁ ∈ [2/3, 1] (f₂ = 1/f₁); ideal (0.3889, 1), nadir (1, 9) | VO |
| SRN | Srinivas, N. and Deb, K. (1994). Multiobjective optimization using nondominated sorting in genetic algorithms. *Evolutionary Computation* 2(3): 221-248. doi:10.1162/evco.1994.2.3.221 (restated in NSGA-II's Table V) | 2; [−20, 20]² | f₁ = (x₁ − 2)² + (x₂ − 1)² + 2, f₂ = 9x₁ − (x₂ − 1)² | x₁² + x₂² ≤ 225; x₁ − 3x₂ ≤ −10 | derived, three pieces: the second constraint's boundary x₁ = 3x₂ − 10 for x₂ ∈ [2.5, 3.7]; the line x₁ = −2.5, x₂ ∈ [2.5, 14.79]; the first constraint's circle from there to about (−4.83, 14.2). **The often-quoted "x₁ = −2.5" is only part of the front** | formulas VO (NSGA-II); the 1994 paper not read; front derived |
| TNK | Tanaka, M., Watanabe, H., Furukawa, Y. and Tanino, T. (1995). GA-based decision support system for multicriteria optimization. Proc. IEEE SMC 2: 1556-1561. doi:10.1109/ICSMC.1995.537993 | 2; [0, π]² | f = x | −x₁² − x₂² + 1 + 0.1 cos(16 arctan(x₁/x₂)) ≤ 0; (x₁ − 0.5)² + (x₂ − 0.5)² ≤ 0.5 | disconnected, on the first constraint's boundary: sampled, filtered | VO (NSGA-II); arctan(x₁/x₂) at x₂ = 0 needs a stated convention (atan2) |
| WATER | Ray, T., Tai, K. and Seow, K. C. (2001). Multiobjective design optimization by an evolutionary algorithm. *Engineering Optimization* 33(4): 399-424. doi:10.1080/03052150108940926 (restated in NSGA-II's Table V) | 3; x₁ ∈ [0.01, 0.45], x₂, x₃ ∈ [0.01, 0.1] | 5 | 7 (≤ constants) | none given (NSGA-II's Table VI gives reached ranges) | VO (NSGA-II); its 6th constraint prints 0.417 (x₁x₂), a product where the others divide: check in Ray et al. Implemented once, as 1.5's water resource planning |
| BNH | Binh, T. T. and Korn, U. (1997). MOBES: a multiobjective evolution strategy for constrained optimization problems. Proc. 3rd Int. Conf. on Genetic Algorithms (Mendel 97), Brno: 176-182 | 2; x₁ ∈ [0, 5], x₂ ∈ [0, 3] | f₁ = 4x₁² + 4x₂², f₂ = (x₁ − 5)² + (x₂ − 5)² | (x₁ − 5)² + x₂² ≤ 25; (x₁ − 8)² + (x₂ + 3)² ≥ 7.7 | derived: x₁ = x₂ = t ∈ [0, 3] (f = (8t², 2(t − 5)²)), then x₂ = 3, x₁ = s ∈ [3, 5] (f = (4s² + 36, (s − 5)² + 4)); join at (72, 8); ideal (0, 4), nadir (136, 50) | **U** |
| OSY | Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria optimization problems using the simple genetic algorithm. *Structural Optimization* 10(2): 94-99. doi:10.1007/BF01743536 | 6; x₁, x₂, x₆ ∈ [0, 10], x₃, x₅ ∈ [1, 5], x₄ ∈ [0, 6] | f₁ = −[25(x₁ − 2)² + (x₂ − 2)² + (x₃ − 1)² + (x₄ − 4)² + (x₅ − 1)²], f₂ = Σ xᵢ² | 6 (≥ 0) | five regions with x₄ = x₆ = 0, each on a set of active constraints (as tabulated in Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms.* Wiley) | **U** |
| Viennet 4, Van Veldhuizen's MOP-C1..C3 | Van Veldhuizen, D. A. (1999). *Multiobjective Evolutionary Algorithms: Classifications, Analyses, and New Innovations.* PhD thesis, AFIT/DS/ENG/99-01 (DTIC ADA364478) | — | — | — | — | **U** (optional) |

#### CTP1-CTP7 (and CTP8)

Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective
evolutionary optimization. EMO 2001, LNCS 1993: 284-298. doi:10.1007/3-540-44719-9_20 (also
KanGAL report 200002). **U throughout: the paper couldn't be reached.** The EMO paper defines
CTP1-CTP7 (Tanabe, R. and Oyama, A. (2017). A note on constrained multi-objective optimization
benchmark problems. Proc. IEEE CEC 2017: 1127-1134. doi:10.1109/CEC.2017.7969433); CTP8 is
believed to come from the KanGAL report or Deb's 2001 book (U).

Recalled structure, to check: f₁ = x₁; CTP1 f₂ = g exp(−f₁/g) with J constraints
f₂ − a_j exp(−b_j f₁) ≥ 0 (J = 2: a = (0.858, 0.728), b = (0.541, 0.295)); CTP2-CTP8
f₂ = g (1 − f₁/g) with cos θ (f₂ − e) − sin θ f₁ ≥ a |sin(bπ(sin θ (f₂ − e) + cos θ f₁)^c)|^d and

| | θ | a | b | c | d | e |
|---|---|---|---|---|---|---|
| CTP2 | −0.2π | 0.2 | 10 | 1 | 6 | 1 |
| CTP3 | −0.2π | 0.1 | 10 | 1 | 0.5 | 1 |
| CTP4 | −0.2π | 0.75 | 10 | 1 | 0.5 | 1 |
| CTP5 | −0.2π | 0.1 | 10 | 2 | 0.5 | 1 |
| CTP6 | 0.1π | 40 | 0.5 | 1 | 2 | −2 |
| CTP7 | −0.05π | 40 | 5 | 1 | 6 | 0 |
| CTP8 (two constraints) | 0.1π / −0.05π | 40 / 40 | 0.5 / 2 | 1 / 1 | 2 / 6 | −2 / 0 |

Not established: g, n, the bounds and how CTP1's a_j, b_j are computed. Fronts: CTP2 and CTP7
disconnected, CTP3-CTP5 points (and pieces), CTP6 and CTP8 behind infeasible tunnels.

#### Constrained DTLZ (C-DTLZ)

Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using
reference-point based nondominated sorting approach, part II: handling constraints and extending
to an adaptive approach. *IEEE TEVC* 18(4): 602-622. doi:10.1109/TEVC.2013.2281534. **VO (from
the authors' preprint; the TEVC text not compared).** x ∈ [0, 1]ⁿ, M ∈ {3, 5, 8, 10, 15},
constraints ≥ 0. The paper tabulates no fronts; they're computed from the forms below.

| Problem | n | Constraints | Front |
|---|---|---|---|
| C1-DTLZ1 | M + 4 | 1 − f_M/0.6 − Σ_{i<M} fᵢ/0.5 ≥ 0 (eq. 4) | DTLZ1's (Σ f = 0.5), all feasible, behind an infeasible band |
| C1-DTLZ3 | M + 9 | (Σ fᵢ² − 16)(Σ fᵢ² − r²) ≥ 0 (eq. 5); r = 9, 12.5, 12.5, 15, 15 for M = 3, 5, 8, 10, 15 | DTLZ3's unit sphere |
| C2-DTLZ2 | M + 9 | feasible inside one of M + 1 spheres of radius r (centered at the unit vectors and at 1/√M): −min{minᵢ[(fᵢ − 1)² + Σ_{j≠i} fⱼ² − r²], Σ(fᵢ − 1/√M)² − r²} ≥ 0; r = 0.4 for M = 3, 0.5 otherwise | the parts of the unit sphere inside the spheres (disconnected) |
| Convex C2-DTLZ2 | M + 9 | Σ(fᵢ − λ)² − r² ≥ 0, λ = mean of f (eq. 6); r = 0.225, 0.225, 0.26, 0.26, 0.27 | convex DTLZ2's front outside a cylinder around the diagonal |
| C3-DTLZ1 | M + 4 | M constraints Σ_{i≠j} fᵢ + fⱼ/0.5 − 1 ≥ 0 (eq. 7) | derived: {f ≥ 0 : Σ fᵢ + minⱼ fⱼ = 1}; DTLZ1's front is infeasible |
| C3-DTLZ4 | M + 4 (as printed) | M constraints fⱼ²/4 + Σ_{i≠j} fᵢ² − 1 ≥ 0 (eq. 8) | derived: {f ≥ 0 : minⱼ[fⱼ²/4 + Σ_{i≠j} fᵢ²] = 1} |

Pitfalls: C2-DTLZ2's constraint is printed without an inequality sign (the "−min ≥ 0" reading
is the only consistent one, as in Zhang et al. 2018, arXiv:1807.10275); C3-DTLZ1 is printed with
i and j swapped; f_M/0.6 and fᵢ/0.5 are fractions (some extractions show exponents). C1-DTLZ1,
C1-DTLZ3 and C2-DTLZ2 are solvable without constraint handling (Tanabe and Oyama 2017). The
paper's settings (population per M, generations per problem) are in its tables, for examples.

#### MW1-MW14

Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite
construction and performance comparisons. *IEEE TEVC* 23(6): 972-986.
doi:10.1109/TEVC.2019.2896967. **VO** (the authors' full text and supplement). fᵢ = g(x_II) sᵢ(x_I);
n scalable, **n = 15** in the experiments; M = 2 except MW4, MW8 and MW14 (M-scalable; many-objective
tests with n = M − 1 + 13). Three distance functions g1, g2, g3 (each ≥ 1). Constraint types: I
(front unchanged), II (part of it), III (part of it plus constraint boundary), IV (on the boundary
only). **The paper gives sampled fronts (> 1000 points), not formulas**; the formulas below are
derived for this plan and are checked in tests against points generated from the Pareto set.

| Problem | Type, front | Bounds, g, constraints | Front (derived where given) |
|---|---|---|---|
| MW1 | II, disconnected | [0, 1]ⁿ, g1, 1 | f₂ = 1 − 0.85 f₁ where feasible |
| MW2 | I | [0, 1]ⁿ, g2, 1 | f₂ = 1 − f₁ |
| MW3 | III, mixed | [0, 1]ⁿ, g3, 2 | part of f₂ = 1 − f₁ plus a boundary: sampled |
| MW4 | I, M objectives | [0, 1]ⁿ, g1, 1 | the simplex Σ f = 1 |
| MW5 | II, points | [0, 1]ⁿ, g1, 3 | 16 points on the unit circle (and slivers near the axes) |
| MW6 | II, disconnected | [0, 1.1]ⁿ, g2, 1 | feasible part of f₁² + f₂² = 1.21 (the exponents of its l are ambiguous in the text: check) |
| MW7 | III | [0, 1]ⁿ, g3, 2 | part of the unit circle plus a boundary: sampled |
| MW8 | II, M objectives | [0, 1]ⁿ, g2, 1 | unit sphere in four bands of asin(f_M) |
| MW9 | IV, concave | [0, 1]ⁿ, g1, 1 | f₂ = 1 − 0.64 f₁² for f₁ ∈ [0, 0.5868], then 1.3225 − (f₁ + 0.15)² for f₁ ∈ [0.5868, 1] |
| MW10 | III, disconnected | [0, 1]ⁿ, g2, 3 (f₁ uses x₁ⁿ) | pieces of 2 − 16f₁², 1 − f₁², 2 − 4f₁² |
| MW11 | IV, disconnected | [0, √2]ⁿ, g3, 4 | pieces of four parabolas and the point (1, 1) |
| MW12 | IV, mixed | [0, 1]ⁿ, g1, 2 | boundary only: sampled |
| MW13 | III, disconnected | [0, 1.5]ⁿ, g2, 2 | sampled |
| MW14 | I, M objectives | [0, 1.5]ⁿ, g3, 1 | 2^(M−1) disconnected patches (fᵢ ∈ [0, ≈ 0.731] ∪ [≈ 1.330, 1.5]) |

Pitfalls: constraint directions are mixed (some printed ≤ 0, most ≥ 0: implement as printed);
MW12 and MW13 use |sin| in f₂ and plain sin in the constraints; MW4 and MW8 order their objectives
so f_M depends on x₁; feasible fractions are tiny (< 0.1‰ for most). The authors' code
(<https://intleo.csu.edu.cn/codes/MW.rar>) wasn't opened; differences from the paper are unchecked.

#### DAS-CMOP1-9

Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K. and Goodman, E. (2020).
Difficulty adjustable and scalable constrained multiobjective test problem toolkit.
*Evolutionary Computation* 28(3): 339-378. doi:10.1162/evco_a_00259. **VO** from arXiv:1612.07603
v3 (the journal text not compared; v1 had different problems, "DAC-MOP").

- The difficulty triplet (η, ζ, γ) ∈ [0, 1]³: η sets **diversity** hardness (type I constraint
  sin(aπx₁) − b ≥ 0, b = 2η − 1), ζ **feasibility** hardness (type II, (e − g)(g − d) ≥ 0,
  ζ = exp(d − e)), γ **convergence** hardness (type III, rotated ellipses, r = γ/2).
- 16 standard triplets (Table 3): (.25,0,0), (0,.25,0), (0,0,.25), (.25,.25,.25), (.5,0,0),
  (0,.5,0), (0,0,.5), (.5,.5,.5), (.75,0,0), (0,.75,0), (0,0,.75), (.75,.75,.75), (0,1,0),
  (.5,1,0), (0,1,.5), (.5,1,.5).
- n = 30, x ∈ [0, 1]ⁿ, a = 20, d = 0.5. DAS-CMOP1-6: 2 objectives, 11 constraints; DAS-CMOP7-9:
  3 objectives, 7 constraints.
- Fronts depend on the triplet and aren't analytic: the paper samples the unconstrained front and
  the constraint boundaries (1000 points for 2 objectives, 10000 for 3) and filters. genoxide does
  the same in `optimal_front`.
- Pitfalls: DAS-CMOP1-3's g sums from j = 1 in Table 2 but j = 2 in section 4's examples; the
  ellipse parameters a_k² = 0.3, b_k² = 1.2 (Table 2) vs 0.4 and 1.6 (eq. 5's example), and
  implementations in circulation that read 0.3 and 1.2 as the semi-axes; d at ζ = 0.

#### LIR-CMOP1-14

Fan, Z., Li, W., Cai, X., Huang, H., Fang, Y., You, Y., Mo, J., Wei, C. and Goodman, E. (2019).
An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions.
*Soft Computing* 23(23): 12491-12510. doi:10.1007/s00500-019-03794-x. **VO** from arXiv:1707.08767
(appendix Table 6; the journal text not compared). n = 30 (hard-coded in the distance functions),
x ∈ [0, 1]³⁰, constraints ≥ 0; LIR-CMOP1-12 two objectives, 13-14 three. Fronts: LIR-CMOP1-4
derived (e.g. LIR-CMOP1 f₂ = 1.5 − (f₁ − 0.5)², f₁ ∈ [0.5, 1.5]); 5 and 6 equal the unconstrained
fronts; 7-12 on constraint boundaries, sampled; 13 the sphere octant of radius 1.7057, 14 of
radius 1.75 (derived). Pitfall: LIR-CMOP10's ellipse prints p = 1.1, q = 1.2 (the others have
p = q): check the journal.

#### CEC 2009 CF1-CF10

Zhang, Q., Zhou, A., Zhao, S., Suganthan, P. N., Liu, W. and Tiwari, S. (2008, revised 2009).
*Multiobjective Optimization Test Instances for the CEC 2009 Special Session and Competition.*
Technical Report CES-487, University of Essex and NTU. **VO** (20 April 2009 draft). n = 10;
CF1-CF7 two objectives, CF8-CF10 three; constraints ≥ 0 (a point counts as feasible for IGD at
≥ −1e-6); **fronts analytic in the report**: CF1 21 points; CF2 and CF3 a point plus arcs; CF4 and
CF5 three linear pieces (1 − f₁, −0.5f₁ + 0.75, 1.125 − f₁); CF6 and CF7 three pieces; CF8 five
curves; CF9 and CF10 a curve plus surfaces. Errata: the CF9/CF10 front prints f₂ = √(1 − f₁² −
**f₂**²) for f₃; CF10's (and UF10's) f₂ and f₃ sums are printed over J1 instead of J2 and J3;
CF8's bounds are [−4, 4] where CF9 and CF10 use [−2, 2].

#### DC-DTLZ

Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for constrained
multiobjective optimization. *IEEE TEVC* 23(2): 303-315. doi:10.1109/TEVC.2018.2855411
(arXiv:1711.07907; supplement <https://colalab.ai/supplementary/ctaea-supp.pdf>). **VO** for the
forms: DC1 (one constraint cos(aπx₁) > b, a = 3, b = 0.5: feasible cones), DC2 (two constraints
on g: same front, most of the space infeasible), DC3 (M + 1 constraints: a segmented front), on
DTLZ1 and DTLZ3. **U:** n (M + 4 and M + 9 assumed), DC2's and DC3's a and b values. Constraints
are printed strict (">"). The supplement gives C2-DTLZ2's r as 0.1, which conflicts with Jain and
Deb (0.4 / 0.5): genoxide follows Jain and Deb.

### 1.5 Engineering design

None of the paywalled originals (Ragsdell and Phillips, Sandgren, Kannan and Kramer, Golinski,
Gu et al., Youn et al., Ray and Liew, Cheng and Li, Osyczka and Kundu, Amir and Hasegawa, Parsons
and Scott, Deb, Pratap and Moitra) could be read in full during the survey. Read in full: Yang,
Huyck, Karamanoglu and Khan (2013, IJBIC 5(6): 329-335, doi:10.1504/IJBIC.2013.058910,
arXiv:1403.7793) (**Yang13**), which proves the pressure vessel and cantilever optima; Chehouri
et al. (2016, J. Comput. Sci. 12(7): 350-362, doi:10.3844/jcssp.2016.350.362) (**Ch16**), which
lists the welded beam versions; Yang and Gandomi (2012, Engineering Computations 29(5): 464-483,
arXiv:1211.6663); Yang, Karamanoglu and He (2013, Procedia Computer Science 18: 861-868,
arXiv:1404.0695); and Tanabe and Ishibuchi's paper and supplement (2020, An easy-to-use
real-world multi-objective optimization problem suite, *Applied Soft Computing* 89: 106078,
doi:10.1016/j.asoc.2020.106078) (**TI20**), a secondary source that traces each multi-objective
problem to its origin. TI20's own code and reference-front files are another project's data:
genoxide doesn't use them, and the survey found errors in several of its problems (below).

Status labels as in 1.1; **HC** = checked by hand for this plan.

#### Single objective

| Problem | Original | Variables | Constraints | Best known | Status and pitfalls |
|---|---|---|---|---|---|
| Welded beam, version A | Ragsdell, K. M. and Phillips, D. T. (1976). Optimal design of a class of welded structures using geometric programming. *ASME J. Eng. Ind.* 98(3): 1021-1025. doi:10.1115/1.3438995; Deb, K. (1991). Optimal design of a welded beam via genetic algorithms. *AIAA Journal* 29(11): 2013-2015. doi:10.2514/3.10834 | 4 continuous (h, l, t, b); bounds (Deb) ≈ h ∈ [0.125, 5], l, t ∈ [0.1, 10], b ∈ [0.1, 5] (U) | 5: shear τ ≤ 13600, bending σ ≤ 30000, h ≤ b, buckling P_c = 64746.022 (1 − 0.0282346 t) t b³ ≥ 6000, deflection 2.1952/(t³b) ≤ 0.25; P = 6000, L = 14, E = 30e6, G = 12e6 | Ragsdell and Phillips 2.385937; Deb 1991 2.433116 at (0.2489, 6.1730, 8.1789, 0.2533) | values VS(Ch16 Table 7); P_c constants HC. Two versions are routinely mixed in the literature |
| Welded beam, version B | Coello Coello, C. A. (2000). Use of a self-adaptive penalty approach for engineering optimization problems. *Computers in Industry* 41(2): 113-127. doi:10.1016/S0166-3615(99)00046-9 | x₁, x₄ ∈ [0.1, 2], x₂, x₃ ∈ [0.1, 10] | 7: as A with P_c = 4.013 E √(x₃²x₄⁶/36)/L² (1 − x₃/(2L) √(E/4G)), plus x₁ ≤ x₄, 0.10471x₁² + 0.04811x₃x₄(14 + x₂) ≤ 5, x₁ ≥ 0.125 | **1.724852 at (0.205730, 3.470489, 9.036624, 0.205730)** | VS(Ch16); optimum HC (P_c active). genoxide ships both, `WeldedBeam::ragsdell()` and `WeldedBeam::coello()`, and says which each paper used |
| Pressure vessel | Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design optimization. *J. Mech. Des.* 112(2): 223-229. doi:10.1115/1.2912596; Kannan, B. K. and Kramer, S. N. (1994). *J. Mech. Des.* 116(2): 405-411. doi:10.1115/1.2919393 | mixed: shell and head thicknesses multiples of 0.0625 (k ∈ 1..99), radius and length ∈ [10, 200]; a continuous variant | 4 (volume ≥ 1,296,000 in³; length ≤ 240) | **mixed: 6059.714335048436 at (0.8125, 0.4375, 42.0984455958549, 176.6365958424394), proven global (Yang13)**; continuous: 5885.33 at (0.778169, 0.384649, 40.31962, 200) | mixed VO(Yang13); continuous HC, optimality U. The existing example uses the mixed form |
| Tension/compression spring | Belegundu, A. D. (1982). *A Study of Mathematical Programming Methods for Structural Optimization.* PhD thesis, University of Iowa; Arora, J. S. (1989). *Introduction to Optimum Design.* McGraw-Hill | 3 continuous: d ∈ [0.05, 2], D ∈ [0.25, 1.3], N ∈ [2, 15] | 4 | 0.012665233 at (0.0516891, 0.3567177, 11.288966) | constants VS(Ch16); optimum U (widely reported), f HC |
| Speed reducer | Golinski, J. (1970). Optimal synthesis problems solved by means of nonlinear programming and random methods. *J. Mechanisms* 5(3): 287-309. doi:10.1016/0022-2569(70)90064-9; Ray, T. (2003). Golinski's speed reducer problem revisited. *AIAA Journal* 41(3): 556-558. doi:10.2514/2.1984 | 7; x₃ (teeth) integer 17..28; x₅ ∈ [7.3, 8.3] or [7.8, 8.3] depending on the source | 11 | x₅ ≥ 7.3: 2994.471 at (3.5, 0.7, 17, 7.3, 7.71532, 3.35021, 5.28665); x₅ ≥ 7.8: 2996.348 | U; Ray (2003) reports formulation errors in the literature: read it before fixing the constants |
| Gear train | Sandgren (1990) | 4 integers in [12, 60] | 0 | **2.700857e-12** with {16, 19} over {43, 49} | HC; exactly checkable by enumerating 49⁴ ≈ 5.8 million points (an `#[ignore]` test) |
| Three-bar truss | Nowacki, H. (1974). Optimization in pre-contract ship design. In *Computer Applications in the Automation of Shipyard Operation and Ship Design*, North-Holland: 327-338; Ray, T. and Saini, P. (2001). *Engineering Optimization* 33(6): 735-748 | 2 continuous in [0, 1] | 3 | 263.8958434 at (0.7886751, 0.4082483) = ((3 + √3)/6, 1/√6) | formulation U; optimum HC |
| Cantilever beam | Fleury, C. and Braibant, V. (1986). Structural optimization: a new dual method using mixed variables. *IJNME* 23(3): 409-428 | 5 continuous in [0.01, 100] | 1 | **1.339956367 at (6.0160159, 5.3091739, 4.4943296, 3.5014750, 2.15266533)**, global (closed form) | VO(Yang13; its eq. 43 has x⁵ for x³) |
| Car side impact (weight) | Gu, L., Yang, R. J., Tho, C. H., Makowski, M., Faruque, O. and Li, Y. (2001). Optimisation and robustness for crashworthiness of side impact. *Int. J. Vehicle Design* 26(4): 348-360. doi:10.1504/IJVD.2001.005210; Youn, B. D., Choi, K. K., Yang, R.-J. and Gu, L. (2004). *Struct. Multidisc. Optim.* 26(3-4): 272-283. doi:10.1007/s00158-003-0345-0 | 7 continuous (+ 2 material choices, + 2 random parameters) | 10 | ≈ 22.84 (U: the x vector differs between sources) | bounds and constraints VS(Yang and Gandomi 2012); **U**: needs the original's table |
| I-beam deflection, tubular column, reinforced concrete beam, stepped cantilever | Wang, G. G. (2003). *J. Mech. Des.* 125(2): 210-220; Rao (1996, book); Amir, H. M. and Hasegawa, T. (1989). *J. Struct. Eng.* 115(3): 626-646; Thanedar, P. B. and Vanderplaats, G. N. (1995). *J. Struct. Eng.* 121(2): 301-306 | — | — | I-beam 0.0130741 at (50, 80, 0.9, 2.32179) (HC) | U: deferred until the originals are read |

#### Several objectives

| Problem | Original | Variables | M | Constraints | Front / reference | Status and pitfalls |
|---|---|---|---|---|---|---|
| Two-bar truss | Deb, K., Pratap, A. and Moitra, S. (2000). Mechanical component design for multiple objectives using elitist non-dominated sorting GA. PPSN VI, LNCS 1917: 859-868. doi:10.1007/3-540-45356-3_84 (earlier: Palli, Azarm, McCluskey and Sundararajan, 1999, U); also Chiandussi, G., Codegone, M., Ferrero, S. and Varesio, F. E. (2012). *Computers & Mathematics with Applications* 63(5): 912-942. doi:10.1016/j.camwa.2011.11.057 | x₁, x₂ ∈ [0, 0.01] m² (bar areas), y ∈ [1, 3] m | 2 (volume, max stress) | 1 (stress ≤ 10⁵ kPa) | **analytic (HC, derived for this plan):** y = 2 with both bars fully stressed gives f₁ = 400/f₂ for f₂ ∈ [8944.27, 10⁵]; below, x₂ at its bound and a 1-D problem in y, down to 8000√10/3 ≈ 8432.7 at y = 3. Deb et al. report the range (0.00407, 99755) to (0.05304, 8439) | U (only the abstract read); **TI20's version differs** (other objectives and bounds, after Coello and Pulido 2005): not the same front |
| Welded beam (cost, deflection) | Deb, Pratap and Moitra (2000); Ray, T. and Liew, K. M. (2002). A swarm metaphor for multiobjective design optimization. *Engineering Optimization* 34(2): 141-153 | 4 | 2 | 4 (deflection limit dropped) | no analytic front: reference by ε-constraint with genoxide's own single-objective solvers (section 3) | which P_c form the originals used is U |
| Disc brake | Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria optimization problems using the simple genetic algorithm. *Structural Optimization* 10(2): 94-99. doi:10.1007/BF01743536; Ray and Liew (2002) | r ∈ [55, 80], R ∈ [75, 110], F ∈ [1000, 3000], s ∈ [2, 20] integer | 2 (mass, stopping time) | 5 | ε-constraint reference | VS(Yang, Karamanoglu and He 2013); TI20's version drops a constraint and changes s's range (its footnote inverts s ≤ 11) |
| Car side impact | Gu et al. (2001); 3-objective version: Jain and Deb (2014), part II, doi:10.1109/TEVC.2013.2281534 | 7 continuous | 3 (weight, pubic force, mean of two velocities) | 10 | ε-constraint reference | VS(TI20); the survey found by hand a factor-10 difference in one constraint coefficient (0.0092928 vs 0.484 × 0.192 = 0.092928) and a swapped material constant between TI20's two versions: **check against Gu et al. (2001) and Jain and Deb (2014)** |
| Speed reducer (weight, stress) | Kurpati, A., Azarm, S. and Wu, J. (2002). *Struct. Multidisc. Optim.* 23(3): 204-213; Farhang-Mehr, A. and Azarm, S. (2002). *Struct. Multidisc. Optim.* 24(5): 351-361 | 7 (x₃ integer) | 2 | 11 (stress limits 1300 and 1100) | ε-constraint reference | VS(TI20); constants U |
| Gear train (error, max teeth) | Deb, K. and Srinivasan, A. (2006). Innovization. GECCO 2006 (and KanGAL report 2005007) | 4 integers in [12, 60] | 2 | 1 | **exact by enumeration** (5.8 million points) | VS(TI20); original U |
| Four-bar truss | Stadler, W. and Dauer, J. (1992). Multicriteria optimization in engineering: a tutorial and survey. In *Structural Optimization: Status and Promise*, Progress in Astronautics and Aeronautics 150, AIAA: 209-249; Cheng, F. Y. and Li, X. S. (1999). *Engineering Optimization* 31(5): 641-661 | x₁, x₄ ∈ [1, 3], x₂, x₃ ∈ [√2, 3] cm² | 2 (volume, displacement) | 0 | **analytic (HC, derived for this plan):** x₃ = √2; three segments, from (1400, 0.04) to (3600, 0.00714) | the volume term is √2 x₃ (VS: GAMS model "truss2" citing Stadler and Dauer); **TI20 and its copies use √x₃, dimensionally wrong** (it shifts the front by −162.16 in f₁) |
| Rocket injector | Vaidyanathan, R., Tucker, P. K., Papila, N. and Shyy, W. (2003). Computational-fluid-dynamics-based design optimization for single-element rocket injector. AIAA paper 2003-296 (J. Propulsion and Power 20(4), 2004: U); Goel, T. et al. (2007). *CMAME* 196: 879-893 | 4 in [0, 1] | 3 (the original has 4) | 0 | sampled reference | response-surface polynomials VS(TI20) |
| Water resource planning | Musselman, K. and Talavage, J. (1980). A tradeoff cut approach to multiple objective optimization. *Operations Research* 28(6): 1424-1435; Ray, T., Tai, K. and Seow, K. C. (2001). *Engineering Optimization* 33(4): 399-424 | 3 | 5 | 7 | sampled reference | VS(TI20); originals U |
| Conceptual marine design | Parsons, M. G. and Scott, R. L. (2004). Formulation of multicriterion design optimization problems for solution with scalar numerical optimization methods. *J. Ship Research* 48(1): 61-76. doi:10.5957/jsr.2004.48.1.61 | 6 | 3 | 9 | sampled reference | VS (a 2023 paper restating the model); **TI20's code computes sea days as (5000/24) Vk instead of 5000/(24 Vk)** |
| Vehicle crashworthiness | Liao, X., Li, Q., Yang, X., Zhang, W. and Li, W. (2008). Multiobjective optimization for crash safety design of vehicles using stepwise regression model. *Struct. Multidisc. Optim.* 35(6): 561-569 | 5 in [1, 3] mm | 3 | 0 | sampled reference | VS(TI20) |
| Coil compression spring, hatch cover, reinforced concrete beam, cantilever (Deb's book), multiple-disk clutch brake | Sandgren (1990); Amir and Hasegawa (1989); Deb (2001, book); Osyczka (2002, *Evolutionary Algorithms for Single and Multicriteria Design Optimization*, Physica) | — | 2 | — | — | U or VS(TI20) with discrepancies between its supplement and code (bounds, a discrete list "3,1" for 3.10): deferred until the originals are read |

**Reference fronts without a formula** are computed by genoxide itself: an ε-constraint sweep with
CMA-ES or SHADE and Deb's rules, run once by a script in the repository
(`cargo run --release --example make_fronts`, documented) and embedded as generated Rust
constants with the script's parameters in a comment. They are checked against the extreme points
the papers report. No front file from another project is used.

### 1.6 Summary

| Group | Core problems | Optional | Verified in the original (or its standard report) | Unverified or secondary only |
|---|---|---|---|---|
| Single objective, unconstrained | 59 continuous (18 scalable unimodal, 21 scalable multimodal, 20 fixed-dimension) | 6 binary and combinatorial; whole CEC/BBOB suites | Goldstein-Price; the CEC 2005 and BBOB forms; citations of Rosenbrock, Griewank, Rastrigin (1991), Styblinski-Tang | most: the originals are books and reports that aren't online; the common forms are secondary (Yao, Liu and Lin 1999; CEC 2005) |
| Single objective, constrained (CEC 2006) | 24 | | all 24 (the September 2006 report) | g17's better value, g22's claimed better value, g04's variant, the "max" origins |
| Multi objective, unconstrained | 25 (SCH1, SCH2, FON, KUR, POL, VNT1-3, ZDT5, DTLZ5-7, scaled DTLZ1/2, convex DTLZ2, inverted DTLZ1, WFG1-9) | MaF1-15, UF1-10, Deb's 1999 problems | ZDT5, DTLZ5-7 | WFG (all), scaled/convex/inverted DTLZ, Kursawe, Poloni, Viennet, SCH2's formula; FON secondary |
| Multi objective, constrained | 45 (CONSTR, SRN, TNK, BNH, OSY, CTP1-8, C-DTLZ ×6, MW1-14, DAS-CMOP1-9, DTLZ8-9) | LIR-CMOP1-14, CF1-10, DC-DTLZ ×6, Viennet 4 / MOP-C | CONSTR, SRN, TNK (as NSGA-II restates them), C-DTLZ, MW, DAS-CMOP, DTLZ8-9, and the optional LIR-CMOP, CF, DC-DTLZ | CTP (all), BNH, OSY; DC2/DC3 parameters |
| Engineering design | 8 single-objective (welded beam in two versions, pressure vessel, spring, speed reducer, gear train, three-bar truss, cantilever beam, car side impact) and 11 multi-objective (two-bar truss, welded beam, disc brake, car side impact, speed reducer, gear train, four-bar truss, rocket injector, water resource planning, marine design, crashworthiness) | I-beam, tubular column, concrete beam, stepped cantilever, coil spring, hatch cover, clutch brake, cantilever (Deb) | pressure vessel and cantilever optima (Yang et al. 2013, proofs) | the originals of all the others: secondary (Tanabe and Ishibuchi 2020, Chehouri et al. 2016) or unverified |

**Core total: 172 new problems**, on top of the 9 genoxide has (ZDT1-4, ZDT6, DTLZ1-4), and
about 75 optional ones.

**To check in the originals before implementing** (by library access, since they're paywalled or
in print only): the WFG paper (IEEE TEVC 2006), CTP (EMO 2001 / KanGAL 200002), Deb's 2001 book
(OSY's regions, CTP8), Binh and Korn (1997), Kursawe (1991), Poloni et al. (2000), Viennet et al.
(1996), Deb and Jain (2014) part I (scaled and convex DTLZ), Schwefel (1981), Dixon and Szegö
(1978, Hartmann and Shekel constants), Ragsdell and Phillips (1976), Sandgren (1990), Golinski
(1970) with Ray (2003), Gu et al. (2001), Osyczka and Kundu (1995), Deb, Pratap and Moitra (2000),
and Stadler and Dauer (1992). A batch doesn't start until the originals of its problems are read;
each problem's docs then cite the page, table or equation used.

## 2. API design

### 2.1 Principles

- **One definition, in Rust.** Every problem is written once, in Rust, from its paper. Python,
  the `genoxide` program's built-in functions and the examples use it; nothing is defined twice.
- **A problem is a fitness function.** It implements `FitnessFunction` (single objective) or
  `MultiFitnessFunction` (several), so `Engine::new(algorithm, problem)` and
  `MultiEngine::new(algorithm, problem)` take it as it is, like the ZDT and DTLZ problems today.
- **Constraints are Deb's rules.** A constrained problem returns `(score, violation)` (single
  objective) or `([f64; M], violation)` (several), the violation being the sum of the constraint
  violations measured with `constraint::at_most`, `at_least` and `equal`. The per-constraint
  values are available too, for tests and for other constraint handling.
- **Everything the paper gives, the problem gives:** its search space as a representation, its
  known optimum (value and solutions) or optimal front, and its citation.
- **Minimized, in the paper's form.** A problem is minimized unless its paper maximizes it (the
  binary problems); `objective()` says which. A problem whose original maximizes but whose
  standard form minimizes (CEC 2006's g02, g03, g08, g12) follows the standard form and says so.
- **Panics only in constructors**, for too few variables or objectives, as in `multi::problems`
  today (the one documented panic in AGENTS.md's guarantees). Fixed-size problems are unit
  structs.
- **No data copied from other libraries.** Constants come from the paper's tables; reference
  fronts are computed from the paper's analytic front or Pareto set by genoxide's own code, or
  approximated by genoxide's own runs where the paper gives neither.

### 2.2 Rust: `genoxide::problems` (single objective)

```rust
/// A single-objective test problem: a fitness function with its search space, its known
/// optimum and the paper that defines it.
pub trait Problem: FitnessFunction<<Self::Representation as Representation>::Genome> {
    /// The search space: `Real` bounds, `Integer` bounds or a `Binary` length.
    type Representation: Representation;

    /// The name, e.g. `"Rastrigin"` or `"G01"`.
    fn name(&self) -> &'static str;

    /// The representation: the number of variables and their bounds.
    fn representation(&self) -> Self::Representation;

    /// Whether the score is minimized (almost every problem) or maximized.
    fn objective(&self) -> Objective {
        Objective::Minimize
    }

    /// The global optimum, or the best known, if the paper (or a later proof) gives it for this
    /// size.
    fn optimum(&self) -> Option<Optimum<<Self::Representation as Representation>::Genome>>;

    /// The paper that defines the problem, e.g. "Rastrigin, L. A. (1974). Systems of Extremal
    /// Control. Nauka, Moscow."
    fn reference(&self) -> &'static str;

    /// Its DOI or URL, if any.
    fn reference_url(&self) -> Option<&'static str> {
        None
    }

    /// The values of the constraints at `genome`, in the paper's order, for constrained problems;
    /// empty for the others.
    fn constraints(&self, genome: &<Self::Representation as Representation>::Genome) -> Constraints {
        Constraints::none()
    }
}

/// The optimum of a problem: its value and the solutions that reach it.
pub struct Optimum<G> { /* value, solutions, proven */ }

impl<G> Optimum<G> {
    /// The optimal score.
    pub fn value(&self) -> f64;
    /// The known solutions with this score: all of them when there are few (Himmelblau's four,
    /// Branin's three), one otherwise, none when only the value is known.
    pub fn solutions(&self) -> &[G];
    /// Whether the value is the global optimum (analytic or proven), rather than the best known.
    pub fn is_proven(&self) -> bool;
}

/// The constraint values of a solution, in the paper's order and in a common form: inequalities
/// `g(x) <= 0` and equalities `h(x) = 0`.
pub struct Constraints { /* inequalities: Vec<f64>, equalities: Vec<f64> */ }

impl Constraints {
    pub fn inequalities(&self) -> &[f64];
    pub fn equalities(&self) -> &[f64];
    /// The total violation: `Σ max(0, g)` + `Σ max(0, |h| − tolerance)`, what `evaluate` returns
    /// as the violation.
    pub fn violation(&self, tolerance: f64) -> f64;
}
```

- **Outputs.** `FitnessFunction::Output` is `f64` for unconstrained problems and `(f64, f64)`
  for constrained ones, so `problem.evaluate(&x)` gives the natural value and every `Engine`
  takes the problem.
- **Equality tolerance.** CEC 2006's problems use the report's δ = 0.0001, a public constant
  (`problems::cec2006::EQUALITY_TOLERANCE`). A constructor `with_tolerance(δ)` changes it.
- **Signs.** Papers write constraints as `g ≤ 0` (CEC 2006), `g ≥ 0` (Deb) or with limits;
  `constraints()` always normalizes to `g ≤ 0`, and the docs of each problem give the paper's
  form.
- **Scalable problems** take the dimension: `Rastrigin::new(n)` (panics below the minimum), and
  `Default` is the size the paper or the standard suite uses. Fixed-size problems are unit
  structs: `Branin`, `cec2006::G01`, `engineering::WeldedBeam`.
- **Optima depending on `n`.** `optimum()` returns `None` when the value isn't known for this
  `n` (Michalewicz's is known for n = 2, 5, 10 only, from the literature).
- **Discrete variables.** A problem whose variables are integers only has an `Integer`
  representation (gear train). A mixed problem (pressure vessel, speed reducer) has a `Real`
  representation whose discrete genes are rounded to their grid inside `evaluate`, documented per
  problem; `problem.design(&genome)` returns the design variables. Grid points are genomes
  themselves, so `optimum().solutions()` are genomes. This is the pressure vessel example's
  approach, until genoxide has a mixed genome (on the roadmap).
- **Modules.** `problems` holds the classic functions (`problems::Rastrigin`); suites and groups
  get submodules: `problems::cec2006::{G01, …, G24}`, `problems::engineering::{WeldedBeam, …}`,
  `problems::binary::{OneMax, LeadingOnes, Trap, …}`. Transformations used by the CEC and BBOB
  suites, `problems::Shifted<P>` and `problems::Rotated<P>`, wrap any real problem with a shift
  vector or a rotation matrix generated from a seed (and keep its optimum: shifted solutions, same
  value).
- **Registry.** `problems::all()` lists every single-objective problem at its default size as
  `Box<dyn DynProblem>` (object-safe: name, bounds, `evaluate(&[f64]) -> Fitness`, optimum), for
  the CLI, Python and test loops over the whole catalog.

Usage:

```rust
use genoxide::prelude::*;
use genoxide::problems::{Problem, Rastrigin};

let problem = Rastrigin::new(10);
let target = problem.optimum().expect("known").value() + 1e-8;
let de = De::builder(problem.representation()).minimize().seed(1).build()?;
let outcome = Engine::new(de, problem)
    .stop_when(Stop::target(target).or(Stop::evaluations(200_000)))
    .run()?;
```

### 2.3 Rust: extending `multi::problems`

The present `TestProblem<M>` requires `Real` genomes (`real()`), unconstrained `[f64; M]`
outputs and a known front. It generalizes to:

```rust
/// A multi-objective test problem with M objectives, all minimized.
pub trait MultiProblem<const M: usize>:
    MultiFitnessFunction<<Self::Representation as Representation>::Genome, M>
{
    type Representation: Representation;
    fn name(&self) -> &'static str;
    fn representation(&self) -> Self::Representation;
    fn reference(&self) -> &'static str;
    fn reference_url(&self) -> Option<&'static str> { None }
    /// The constraint values, as for single-objective problems; empty if unconstrained.
    fn constraints(&self, genome: &<Self::Representation as Representation>::Genome) -> Constraints {
        Constraints::none()
    }
    /// At least `points` points of the optimal front (feasible part only), or `None` if no front
    /// is known (most engineering problems).
    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>>;
    /// The ideal and nadir points of the optimal front, where known: the normalization and the
    /// hypervolume reference point of the papers.
    fn ideal_point(&self) -> Option<[f64; M]> { None }
    fn nadir_point(&self) -> Option<[f64; M]> { None }
}
```

- **Breaking change (0.x):** `TestProblem` becomes `MultiProblem`, `real()` becomes
  `representation()` and `optimal_front` returns an `Option`. The doctest, `tests/multi.rs`,
  the examples and AGENTS.md follow. ZDT and DTLZ keep their behavior.
- **Outputs:** `[f64; M]` unconstrained, `([f64; M], f64)` constrained.
- **Binary ZDT5:** `Zdt5` has a `Binary` representation (80 bits by default: 30 + 10 × 5,
  the paper's sizes; the substring lengths are parameters) and `Bits` genomes.
- **Fronts that aren't a formula** are computed, not stored: from the paper's Pareto set
  (WFG: distance parameters at 0.35 × their upper bound; DTLZ5-7; constrained DTLZ variants: the
  DTLZ front filtered by the constraints; CTP: the constraint boundaries), then filtered to the
  non-dominated points. `optimal_front(points)` samples it evenly (Das and Dennis points on the
  position parameters for M ≥ 3). Engineering problems without a known front return `None`, and
  a separate `approximate_front()` gives genoxide's own ε-constraint approximation (generated by
  a script in the repository, section 1.5), named so it can't be mistaken for an exact front;
  the tests also use the extreme points the papers report (section 3).
- **Constraint count:** a const `CONSTRAINTS` or a method `constraint_count()` on each problem,
  for Python and the docs.
- **Many objectives:** DTLZ, WFG, C-DTLZ and the scalable MW and DAS-CMOP problems are generic
  over `const M` like DTLZ today. Python supports 2 to 6 objectives, like its algorithms.
- **Modules:** `multi::problems::{Zdt1, …, Dtlz7, Wfg1, …, Wfg9, Schaffer1, Schaffer2,
  FonsecaFleming, Kursawe, Poloni, Viennet1, …, Bnh, Srn, Tnk, Osy, Ctp1, …, C1Dtlz1, …,
  Mw1, …, DasCmop1, …}` and `multi::problems::engineering::{TwoBarTruss, WeldedBeam, DiscBrake,
  CarSideImpact, …}`. The file splits by suite: `src/multi/problems/{zdt, dtlz, wfg, classic,
  ctp, cdtlz, mw, das_cmop, engineering}.rs`.
- **Registry:** `multi::problems::all::<M>()`, as for single objective.

### 2.4 Python: `genoxide.problems`

The problems are Python classes that describe themselves (like `gx.Real` and the operators) and
are evaluated by the Rust code:

```python
import genoxide as gx

problem = gx.problems.Rastrigin(10)
problem.genome        # gx.Real((-5.12, 5.12), length=10)
problem.objective     # "minimize"
problem.optimum       # gx.problems.Optimum(value=0.0, solutions=array([[0., ..., 0.]]), proven=True)
problem.reference     # "Rastrigin, L. A. (1974). Systems of Extremal Control. Nauka, Moscow."
problem(np.zeros(10)) # 0.0, a fitness function like any other

de = gx.De(problem.genome, objective=problem.objective, seed=1)
result = de.run(problem, target=problem.optimum.value + 1e-8, evaluations=200_000)

g01 = gx.problems.cec2006.G01()
g01(g01.optimum.solutions[0])     # (-15.0, 0.0): score and violation
g01.constraints(x)                # numpy array of the g(x) <= 0 values, then the equalities

zdt5 = gx.problems.Zdt5()
zdt5.genome                       # gx.Binary(80)
dtlz2 = gx.problems.Dtlz2(objectives=3)
front = dtlz2.optimal_front(91)   # (91, 3) numpy array, or None if unknown
nsga3 = gx.Nsga3(dtlz2.genome, objectives=dtlz2.objectives, ...)
result = nsga3.run(dtlz2, generations=400)
gx.indicators.igd_plus(result.front_objectives, front, dtlz2.objectives)
```

- **Native evaluation.** `run` recognizes a `gx.problems` object and evaluates it in Rust, with
  no Python call per genome: no GIL, `parallel=True` uses every core, and a seed gives the same
  result as the Rust program. Calling `problem(x)` (one genome) or `problem.evaluate(X)` (a 2-D
  array, a genome per row, returning an array, or `(scores, violations)`) also runs the Rust
  code, so the problem can be wrapped in a Python function (e.g. to add noise).
- **Description.** Each class has `_describe()` (`{"type": "rastrigin", "dimensions": 10}`);
  the native module gains `problem_info(json)` (bounds, objective, optimum, reference, constraint
  count) and `evaluate(json, array)`. The Python crate maps the description to the Rust type
  with a serde enum, as it does for algorithms; `genome`, `optimum` and `reference` come from
  Rust, so nothing is defined twice.
- **Names and arguments** follow Python: `gx.problems.Rastrigin(dimensions=10)`,
  `gx.problems.Dtlz2(objectives=3, variables=12)`, `gx.problems.Wfg1(objectives=3,
  position=4, distance=20)`; suites are submodules: `gx.problems.cec2006.G01()`,
  `gx.problems.engineering.WeldedBeam()`. Every class has a docstring with the problem, its
  size, its optimum and its citation, for pdoc.
- **Indicators.** `gx.indicators.{hypervolume, igd, igd_plus, gd, spread}` expose
  `multi::indicator`, to use the fronts (the ZDT1 example computes its hypervolume by hand
  today).
- **Typing.** `_genoxide.pyi` and `py.typed` cover the new module; `problem.objectives` is a
  list for multi-objective problems, so `objectives=problem.objectives` works in every
  multi-objective algorithm.

### 2.5 The `genoxide` program

The CLI's built-in fitness programs (`builtin = "rastrigin"`, `genoxide fitness <name>`) come
from the registries instead of their own formulas: every problem becomes a built-in, and the
list prints each with its bounds and optimum.

## 3. Test strategy

The existing `values_match_pymoo` test in `src/multi/problems.rs` takes its expected values from
pymoo. It goes: under the rules of this plan, expected values come from the papers or are
derived by hand from the papers' formulas, and the tests say which.

### 3.1 Every problem

1. **The optimum.** `evaluate` at each solution of `optimum()` gives `optimum().value()`: to
   1e-12 (relative) where the optimum is analytic (Sphere, Rastrigin, …), to the digits the paper
   prints otherwise (e.g. CEC 2006's f(x*) to the report's 1e-11 or so, the engineering
   optima to their published digits). A constrained optimum is feasible within the tolerance, and
   the constraints the paper lists as active are within 1e-4 of 0.
2. **Points from the papers.** Where a paper tabulates solutions with their values (CEC 2006's
   x*, the engineering papers' tables of designs and costs, Deb's book tables, the MW paper's
   fronts), each is a test case with the paper's table cited in a comment.
3. **Points derived by hand.** Simple points whose values follow from the formula without a
   computer: the origin, all ones, a bound corner, a point with one variable changed (e.g.
   Rastrigin at `(1, 0, …, 0)` is 1; Rosenbrock at the origin is `n − 1`; Ackley at the origin
   is 0). The derivation is in a comment.
4. **Invariants**, property-tested over random genomes in the bounds: the value is finite (no
   NaN, no infinity) everywhere in the bounds, deterministic, and never below the optimum
   (a check of the optimum itself); symmetry where the function is symmetric (Rastrigin, Sphere,
   Griewank under sign changes and permutations); for `Shifted`/`Rotated`, the wrapper's optimum
   is where the wrapped one says.
5. **Metadata.** `representation()` matches the paper's bounds; `name()` is unique across the
   registry; `reference()` is non-empty and every `reference_url()` is a DOI or a stable URL.
6. **Python.** For every class: `problem(x)` equals the Rust value (the registry test
   loops over all), `genome` equals the Rust bounds, a short native run with a seed repeats
   exactly, and a native run equals a run with `lambda x: problem(x)` for the same seed.

### 3.2 Multi-objective fronts

1. **On the front.** Solutions from the Pareto set (e.g. ZDT's `x₂ … xₙ = 0`, DTLZ's distance
   variables at 0.5, WFG's at 0.35 × 2i, SRN's `x₁ = −2.5`) evaluate to points on the analytic
   front, to 1e-12.
2. **The front itself.** `optimal_front(points)` returns at least `points` points, mutually
   non-dominated (`non_dominated_sort` gives one front), feasible for constrained problems, and
   satisfying the analytic identity (DTLZ1: Σ f = 0.5; DTLZ2-4: Σ f² = 1; ZDT1: f₂ = 1 − √f₁;
   MW and CTP: the boundary equations of the paper).
3. **Extremes.** The ideal and nadir points match the papers (e.g. WFG's front spans
   `[0, 2m]` in objective m; ZDT3's pieces; DTLZ7's disconnected regions), and so do the
   extreme solutions tabulated for the classic problems (BNH, OSY, SRN, TNK in Deb's book).
4. **Constrained problems.** Points just inside and just outside each constraint boundary give
   zero and positive violations; the front points of CTP and C-DTLZ lie on the boundary their
   paper describes.
5. **Solvers**, as doctests and `tests/`: a short NSGA-II or NSGA-III run reaches an IGD+ below a
   threshold on the easy problems (as the `multi::problems` doctest does for ZDT1). The hard
   ones (WFG, MW, DAS-CMOP) are for the benchmark suite, not the unit tests.

### 3.3 Where reference values come from

- The paper's own tables and equations (cited per test).
- The original authors' reference code, if they published it with the paper (the WFG group's
  C++ toolkit, the CEC 2006 and CEC 2009 organizers' code): only to generate extra test values,
  cited as the authors' implementation, never copied into genoxide, and only if its license
  allows. No other library.
- Hand derivations, written out in the test.

## 4. Implementation order

Each batch is a PR (or two) with its problems, their tests, the Python classes, the registry
entries, the docs (the module docs list every problem with its citation), and its examples in
`examples/<name>/` with `main.rs`, `main.py` and a `README.md` with the front matter
(`title`, `category`, `summary`, `reference`, `reference_url`, `optimum`, `languages`, `order`),
plus a row in `examples/README.md`.

| Batch | Contents | Problems | Examples (`examples/<name>/`) |
|---|---|---|---|
| 1 | The `problems` module (trait, `Optimum`, `Constraints`, registry), `gx.problems` with native evaluation, `gx.indicators`; the classic continuous functions | Sphere, Axis-parallel ellipsoid, Schwefel 1.2, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel (2.26), Levy, Zakharov, Styblinski-Tang, Himmelblau, Michalewicz, Branin, Goldstein-Price, Six-hump camel (16) | `rastrigin` (switch to `problems::Rastrigin`); new `function_suite`: CMA-ES, SHADE and PSO on the batch's scalable functions in 10-D, printing the error to the optimum; new `himmelblau`: the four minima found by restarts of a local search (continuous) |
| 2 | `MultiProblem` (the breaking change), constraints in multi-objective problems, the pymoo test values replaced (section 5); the classic two- and three-objective problems | Schaffer 1, Schaffer 2, Fonseca-Fleming, Kursawe, Poloni, Viennet 1-3, BNH, SRN, TNK, OSY, CONSTR (13; WATER is batch 9's water resource planning) | `bnh` (NSGA-II on a constrained problem, IGD+ to the analytic front; multi-objective); `kursawe` (disconnected front, SPEA2 vs NSGA-II) |
| 3 | Engineering design, single objective, and the mixed-variable convention (`design()`) | Welded beam, Pressure vessel, Tension/compression spring, Speed reducer, Gear train (integer), Three-bar truss, Cantilever beam, Car side impact (single objective) (8), and CEC 2006 g01-g06 (6) | `pressure_vessel` (switch to `engineering::PressureVessel`); new `welded_beam` (constrained); new `gear_train` (integer genome; category integer) |
| 4 | Scalable many-objective problems | DTLZ5, DTLZ6, DTLZ7, ZDT5 (binary), WFG1-WFG9 (13) | `wfg_many_objective`: NSGA-III and MOEA/D on WFG4 and WFG9 with 5 objectives, IGD to the sampled front; `zdt5` (binary genome, multi-objective) |
| 5 | CEC 2006, part 2 | g07-g18 (12) | `cec2006`: SHADE with Deb's rules on all of the problems so far, printing f − f* and feasibility, as in the report's evaluation criteria |
| 6 | CEC 2006, part 3, and the low-dimensional classics with tables | g19-g24 (6), Hartmann 3-D, Hartmann 6-D, Shekel 5/7/10, Easom, Eggholder, Schaffer F6 (8) | `cec2006` covers all 24 |
| 7 | Constrained test problems with tunable difficulty (CTP needs its paper first: every CTP detail is unverified) | CTP1-CTP8 (8), C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, C2-convex-DTLZ2, C3-DTLZ1, C3-DTLZ4 (6) | `ctp` (NSGA-II on CTP2/CTP7's disconnected feasible fronts); `c2_dtlz2` (NSGA-III with constraints, 3 objectives) |
| 8 | Scaled and inverted DTLZ, and MW | Convex DTLZ2, scaled DTLZ1, scaled DTLZ2, inverted DTLZ1 (4), MW1-MW14 (14) | `mw` (constrained multi-objective, several fronts) |
| 9 | Engineering design, several objectives | Two-bar truss, welded beam (2 objectives), disc brake, car side impact (3 objectives), speed reducer (2 objectives), four-bar truss, water resource planning, rocket injector, vehicle crashworthiness, conceptual marine design (10) | `two_bar_truss`; `car_side_impact` |
| 10a | Remaining low-dimensional and classic scalable functions | Beale, Booth, Matyas, Bohachevsky 1-3, Three-hump camel, Dixon-Price, Trid, Powell, Langermann, Shekel's foxholes, Kowalik, Schwefel 2.21, Schwefel 2.22 (15) | none (covered by `function_suite`) |
| 10b | CEC and BBOB-style functions, and the shift / rotation wrappers | `Shifted<P>`, `Rotated<P>`; Sum of different powers, Step, Quartic (deterministic: without noise, or with noise seeded from the genome, since fitness functions must be deterministic), Penalized 1 and 2, High-conditioned elliptic, Bent cigar, Discus, Büche-Rastrigin, Non-continuous Rastrigin, Weierstrass, Katsuura, HappyCat, HGBat, Schaffer F7, Rotated hyper-ellipsoid, BBOB different powers (17; the shifted and rotated Rastrigin of CEC 2005 and BBOB are the wrappers around `Rastrigin`) | `rotated_functions`: CMA-ES full vs diagonal on rotated vs axis-parallel ellipsoids |
| 11 | Advanced constrained multi-objective suites | DAS-CMOP1-9 (with the 16 difficulty triplets as a parameter), DC-DTLZ (DC1-DC3 on DTLZ1/DTLZ3), DTLZ8, DTLZ9 (≈15) | `das_cmop` (difficulty triplets) |
| 12 | Binary and combinatorial problems | OneMax, LeadingOnes, deceptive trap, royal road, NK landscapes (seeded), 0/1 knapsack (generated instance classes) (6) | `one_max` and `knapsack` switch to the problems; new `nk_landscape` (binary) |
| 13 (optional) | Competition suites whose definitions are long | LIR-CMOP1-14, CEC 2009 UF1-UF10 and CF1-CF10, MaF1-MaF15, Deb's 1999 two-objective problems, Van Veldhuizen's constrained problems; the deferred engineering problems (section 1.5) once their originals are read | none; used by the benchmark suite |

Order rationale: batch 1 builds the machinery with the functions everyone starts with; batch 2
makes the multi-objective side consistent (and removes the pymoo test values) while the problems
are still small; the engineering problems (batch 3) are what users most often ask for after the
textbook functions; the rest follow by popularity in the constrained and many-objective
literature. Each batch updates AGENTS.md's list of test problems and `docs/features.md`.

## 5. Mentions of pymoo in the code, tests and docs

The library's docs should state behavior and cite the papers, not describe genoxide by another
library. Each mention below has a proposed rewording. The benchmark suite (`benchmarks/`,
`docs/benchmarks/`) compares libraries by name on purpose and stays as it is; so do the
CHANGELOG (history) and the ROADMAP's goals and benchmark lists (`ROADMAP.md:9`, `:164`, `:175`,
`:217`), which the owner may reword separately.

| Place | What it says | Proposed rewording |
|---|---|---|
| `src/multi/problems.rs:593-595` | `// values computed by pymoo 0.6.2 at random points`, test `values_match_pymoo` with 20 vectors of expected values | Replace the test (see below): `values_match_the_papers`, with expected values derived by hand from the papers' formulas at chosen points, and the Pareto-set checks that exist already |
| `src/multi/moead.rs:70-71` | children evaluated together "(like pymoo's `ParallelMOEAD`)" | "The children are evaluated together, so a generation can be evaluated in parallel." (drop the comparison) |
| `src/multi/moead.rs:72-76` | "…which matters more when a generation's children are applied together; pymoo's MOEA/D has no limit." | End at "…applied together." The limit is cited already (Li and Zhang, 2009, IEEE TEVC 13(2): 284-302, doi:10.1109/TEVC.2008.925798) |
| `src/multi/moead.rs:78-79` | "constraints are handled too (unlike pymoo's MOEA/D)" | "…so constraints are handled too." (drop the parenthesis) |
| `src/multi/moead.rs:456` | "`crossover_rate` and `mutation_rate` 1.0 (as in pymoo)" | "`crossover_rate` and `mutation_rate` 1.0 (every child is recombined and mutated, as in Zhang and Li's MOEA/D, 2007, IEEE TEVC 11(6): 712-731, doi:10.1109/TEVC.2007.892759)". Check the paper first; if it doesn't say, just "1.0" |
| `src/multi/moead.rs:529-530` | "the neighborhood size (or more) removes the limit, as in pymoo" | "the neighborhood size (or more) removes the limit, as in the original MOEA/D (Zhang and Li, 2007)" |
| `src/multi/nsga2.rs:459`, `src/multi/nsga3.rs:722`, `src/multi/sms_emoa.rs:520`, `src/multi/spea2.rs:523` | `eliminate_duplicates`: "On by default, as in pymoo." | "On by default." (a genoxide choice, explained by the sentence before it) |
| `python/genoxide/__init__.py:1447` | "With `eliminate_duplicates` (the default, as in pymoo)" | "With `eliminate_duplicates` (the default)" |
| `src/multi/nsga3.rs:29-30` | "The normalization follows pymoo's, and keeps the ideal point, the worst point and the extreme points over the whole run." | "The normalization is Deb and Jain's (2014, IEEE TEVC 18(4): 577-601, doi:10.1109/TEVC.2013.2281535), with the fallbacks of Blank, Deb and Roy (2019, 'Investigating the normalization procedure of NSGA-III', EMO 2019, LNCS 11411: 229-240, doi:10.1007/978-3-030-12598-1_19): it keeps the ideal point, the worst point and the extreme points over the whole run." Verify that the fallbacks match that paper before citing it; otherwise describe them without a citation |
| `src/multi/nsga3.rs:365-366` | "with pymoo's fallbacks to the worst point of the first front and of the population" | "with the fallbacks (Blank, Deb and Roy, 2019) to the worst point of the first front and of the population" (same check) |
| `src/multi/nsga3.rs:641-642` | "(as in Deb and Jain, and pymoo)" | "(as in Deb and Jain, 2014)" |
| `src/multi/sms_emoa.rs:21-22` | "`offspring` children (the population size by default, as in pymoo; 1 for the original steady-state algorithm)" | "`offspring` children (the population size by default, a generational variant; 1 for the original steady-state algorithm of Beume, Naujoks and Emmerich, 2007, EJOR 181(3): 1653-1669, doi:10.1016/j.ejor.2006.08.008)" |
| `src/multi/sms_emoa.rs:27-30` | reference point "at 11 in each normalized objective (pymoo's)…" and "(pymoo normalizes by the parents only, which can put a far child beyond the reference point…)" | "…and the reference point at 11 in each normalized objective, so every point is inside it and the extremes of the front are kept. Normalizing by the parents only could put a far child beyond the reference point and drop it first, even when it's the best in another objective." |
| `src/operator/crossover.rs:163-164` | "Deb and Agrawal's bounded version, as in NSGA-II and pymoo." | "Deb and Agrawal's (1995, Complex Systems 9(2): 115-148) bounded version, as in Deb et al.'s NSGA-II (2002, IEEE TEVC 6(2): 182-197, doi:10.1109/4235.996017)." |
| `src/operator/mutate.rs:328-329` | "Deb's bounded polynomial mutation, as in NSGA-II and pymoo" | "Deb's bounded polynomial mutation (Deb and Goyal, 1996, Computer Science and Informatics 26(4): 30-45; Deb, 2001, Multi-Objective Optimization Using Evolutionary Algorithms, Wiley), as in NSGA-II" |
| `tests/multi.rs:39-40` | "pymoo's NSGA-II reaches 0.8696 to 0.8699 with these settings, genoxide 0.8689 to 0.8700 (5 seeds)" | "genoxide reaches 0.8689 to 0.8700 with these settings (5 seeds)" |
| `tests/multi.rs:267-268` | "an IGD … of 0.0009 to 0.0015 for pymoo, 0.0012 to 0.0016 for genoxide" | "an IGD to the 91 optimal points of 0.0012 to 0.0016 (5 seeds)" |
| `tests/multi.rs:289` | "pymoo's SPEA2 reaches 0.8703 to 0.8706" | drop the second half |
| `tests/multi.rs:311-312` | "(pymoo's sequential MOEA/D 0.8693 to 0.8705, its ParallelMOEAD 0.78 to 0.83)" | drop the parenthesis |
| `tests/multi.rs:326` | "(pymoo 0.0005 to 0.0006)" | drop the parenthesis |
| `tests/multi.rs:347-348` | "like pymoo's SMS-EMOA (0.8715 to 0.8718)" | "0.8713 to 0.8716 over 5 seeds" |

The citations in the rewordings are from the literature as commonly cited: check each DOI and page range when applying them. How genoxide compares with other libraries belongs in the benchmark results, which name them on
purpose.

### Replacing `values_match_pymoo`

The ZDT and DTLZ values are easy to check by hand at chosen points, from the formulas in Zitzler,
Deb and Thiele (2000) and Deb, Thiele, Laumanns and Zitzler (2002/2005):

- ZDT1-3 (n = 30) at `x = (a, b, …, b)`: g = 1 + 9b, and `f₂ = g (1 − √(a/g))` (ZDT1),
  `g (1 − (a/g)²)` (ZDT2), `g (1 − √(a/g) − (a/g) sin(10πa))` (ZDT3). With a = 0.25 and b = 1/9
  (g = 2, a/g = 0.125): ZDT1 f₂ = 2 (1 − √0.125) = 2 − √2/2 = 1.292893218813452; ZDT2
  f₂ = 2 (1 − 1/64) = 1.96875; ZDT3 f₂ = 2 (1 − √0.125 − 0.125 sin(2.5π)) = 2 − √2/2 − 0.25 =
  1.042893218813452.
- ZDT4 (n = 10) at `x = (0.25, 1, 0, …, 0)`: g = 1 + 10 × 9 + (1 − 10 cos 4π) + 8 × (0 − 10 cos 0)
  = 1 + 90 − 9 − 80 = 2, so f₂ = 2 − √2/2 as for ZDT1.
- ZDT6 (n = 10) at `x₁ = 0` (f₁ = 1 − e⁰ sin⁶ 0 = 1) and `x₂ = … = x₁₀ = 1/81`: the tail mean is
  1/81, g = 1 + 9 (1/81)^0.25 = 1 + 9/3 = 4, and f₂ = 4 (1 − (1/4)²) = 3.75.
- DTLZ1 (M = 3, k = 5) at `x = (0.5, 0.5, 0.5, …)`: g = 0, f = (0.125, 0.125, 0.25); at
  `x_M = … = 1`: each term is 0.25 − cos(10π) = −0.75, g = 100 (5 − 3.75) = 125, f scaled by 126.
- DTLZ2-4 (M = 3) at the distance variables 0.5 and `x₁ = x₂ = 0` or `1`: the corners (1, 0, 0),
  (0, 0, 1) and so on; at distance variables 1 (k = 10): DTLZ2 g = 10 × 0.25 = 2.5, radius 3.5;
  DTLZ3 g = 100 (10 + 10 × (0.25 − cos 10π)) = 100 × 2.5 = 250, radius 251.
- DTLZ4 (M = 3) at `x₁ = x₂ = 0.5` and the distance variables at 0.5: g = 0, both angles are
  0.5¹⁰⁰ × π/2 ≈ 1.2391e-30, so f = (cos θ cos θ, cos θ sin θ, sin θ) = (1, 1.2391e-30, 1.2391e-30)
  to double precision: the bias towards f₁ that the paper describes.

Every expected value above is derived in a comment in the test. The test keeps the existing
Pareto-set checks (`zdt_optimal_solutions_lie_on_the_front`,
`dtlz_optimal_solutions_lie_on_the_front`).
