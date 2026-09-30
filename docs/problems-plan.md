# Plan: the test problem library

A working plan, removed when the work is done. Batches 1 to 3 are done
([#170](https://github.com/tachsin/genoxide/pull/170), [#177](https://github.com/tachsin/genoxide/pull/177),
[#191](https://github.com/tachsin/genoxide/pull/191)): `problems` has 16 classic functions, CEC 2006's
g01-g06 and 8 engineering design problems, and `multi::problems` has 13 classic problems besides
ZDT1-4, ZDT6 and DTLZ1-4, all of them in the Python package too, and each with an example of its
own ([#260](https://github.com/tachsin/genoxide/issues/260) tracks the batches). This plan catalogs the problems,
designs their API in Rust and Python, sets how they're tested, and orders the work in batches;
section 4 says which are done.

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
| Schwefel 2.21, maxᵢ \|xᵢ\| | Schwefel (1981), problem 2.21 | any | Y99 [−100, 100] | 0 at 0 | VS(Y99 table I, read in batch 10a) |
| Schwefel 2.22, Σ\|xᵢ\| + Π\|xᵢ\| | Schwefel (1981), problem 2.22 | any | Y99 [−10, 10] (Jamil-Yang [−100, 100]) | 0 at 0 | VS(Y99 table I, read in batch 10a) |
| Axis-parallel ellipsoid Σ i xᵢ² and rotated hyper-ellipsoid Σᵢ Σⱼ≤ᵢ xⱼ² | U (origin not found) | any | axis-parallel [−5.12, 5.12], rotated [−65.536, 65.536] (Molga and Smutnicki 2005, sections 2.2 and 2.3) | 0 at 0 | U; not the same as Schwefel 1.2 |
| Step (Y99 f6), Σ(⌊xᵢ + 0.5⌋)² | Y99, after De Jong's F3 (Σ⌊xᵢ⌋ on [−5.12, 5.12], n = 5) | any (30) | [−100, 100] | 0 on xᵢ ∈ [−0.5, 0.5) | VS(Y99) for n, bounds, f*; formula U |
| Quartic with noise (De Jong's F4) | De Jong (1975); Y99 f7 | any (30) | [−1.28, 1.28] | 0 at 0 without noise | VS(Y99); the noise (Gaussian in De Jong, uniform [0, 1) in Y99) U |
| Rosenbrock | Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a function. *The Computer Journal* 3(3): 175-184. doi:10.1093/comjnl/3.3.175 (2-D); chained n-D form in De Jong (1975) F2 and Y99 f5 | any | none in the original (start (−1.2, 1)); Y99 [−30, 30] | 0 at (1, …, 1) | citation VO; n-D form VS(Y99, CEC05 F6); f(−1.2, 1) = 24.2 VC |
| Zakharov | U (no primary source found) | any | usually [−5, 10] | 0 at 0 | U |
| Dixon-Price | Dixon, L. C. W. and Price, R. C. (1989). Truncated Newton method for sparse unconstrained optimization using automatic differentiation. *JOTA* 60(2): 261-275. doi:10.1007/BF00940007 | any | [−10, 10] | 0 at xᵢ = 2^(−(2ⁱ − 2)/2ⁱ), and with xₙ negated (two minima); a stationary point 2/3 at (1/3, 0, …, 0) for n ≥ 3, not a local minimum but a trap (batch 10a) | VS(Jamil-Yang, whose x* drops the minus sign; Laguna-Martí, whose sum starts at i = 1); x* VC; the paper unread |
| Trid | U (origin not found) | any | [−n², n²] | −n(n+4)(n−1)/6 at xᵢ = i(n+1−i); n = 6: −50, n = 10: −210 | f* VC (unique: tridiagonal Hessian positive definite); origin U (Hedar's collection, per Jamil-Yang, who misprint −200 for n = 10); formula VS(Laguna-Martí, who misprint xᵢxⱼ) |
| Powell singular | Powell, M. J. D. (1962). An iterative method for finding stationary values of a function of several variables. *The Computer Journal* 5(2): 147-151. doi:10.1093/comjnl/5.2.147 | 4 (4k extended) | none (start (3, −1, 0, 1)) | 0 at 0 | formula VS(Steihaug and Suleiman 2013, JOGO 56(3): 845-853, read in batch 10a, restating Powell); blocks of four and [−4, 5] VS(Laguna-Martí, n = 24); f(3, −1, 0, 1) = 215 VC; Jamil-Yang print (x₂ − x₃)⁴ and give the start as x*; the paper unread |
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
| Eggholder | Whitley, D., Mathias, K., Rana, S. and Dzubera, J. (1996). Evaluating evolutionary algorithms. *Artificial Intelligence* 85(1-2): 245-276. doi:10.1016/0004-3702(95)00124-7 | [−512, 512] (Mishra 2006); the original's is [−512, 511] | 2-D: −959.6406627 at (512, 404.2318051) on [−512, 512]; −956.9182316 at (482.35331, 432.87900) on [−512, 511] | **VO** (batch 6: the authors' copy, section 4.2, F101, [−512, 511] with 10 bits, no 2-D minimum); name, [−512, 512] and (512, 404.2319) from Mishra, S. K. (2006), MPRA paper 2718, read, whose sign is lost and whose x₂ is 404.2318 to 4 decimals |
| Schaffer F6 and F7 | Schaffer, J. D., Caruana, R. A., Eshelman, L. J. and Das, R. (1989). A study of control parameters affecting online performance of genetic algorithms for function optimization. Proc. 3rd ICGA: 51-60 | [−100, 100] | 0 at 0 | F6 VS(Whitley et al. 1996, table 1, F9, crediting Schaffer et al.; CEC05 section 2.3.2), batch 6; the original unread; F7 U |
| Büche-Rastrigin; BBOB Rastrigin (f3) | BBOB f4, f3 | [−5, 5] | f_opt | VO (BBOB) |
| Shifted, shifted-rotated Rastrigin | CEC05 F9, F10 (shift vectors and matrices in the report's data files) | [−5, 5] | bias −330 | VO(CEC05) |
| Non-continuous Rastrigin | Liang, J. J., Qin, A. K., Suganthan, P. N. and Baskar, S. (2006). Comprehensive learning particle swarm optimizer. *IEEE TEVC* 10(3): 281-295. doi:10.1109/TEVC.2005.857610 | [−5.12, 5.12] | 0 | U |

#### Fixed dimension

| Function | n | Original | Bounds | f*, x* | Status |
|---|---|---|---|---|---|
| Goldstein-Price | 2 | Goldstein, A. A. and Price, J. F. (1971). On descent from local minima. *Mathematics of Computation* 25(115): 569-574. doi:10.1090/S0025-5718-1971-0312365-X | none in the original; [−2, 2] from Dixon and Szegö (1978) | 3 at (0, −1); local minima (1.2, 0.8) → 840, (1.8, 0.2) → 84, (−0.6, −0.4) → 30 (the paper's list: four test points) | **VO** |
| Branin (RCOS) | 2 | Branin, F. H. (1972). Widely convergent method for finding multiple solutions of simultaneous nonlinear equations. *IBM J. Res. Dev.* 16(5): 504-522. doi:10.1147/rd.165.0504; constants as in Dixon and Szegö (1978) | x₁ ∈ [−5, 10], x₂ ∈ [0, 15] | 5/(4π) = 0.3978874 at (−π, 12.275), (π, 2.275), (3π, 2.475) | formula VS(Y99 f17, Jamil-Yang; both misprint 2.475 as 2.425); minima VC |
| Six-hump camel | 2 | Dixon, L. C. W. and Szegö, G. P. (eds.) (1978). *Towards Global Optimisation 2.* North-Holland | Y99 f16 and Jamil-Yang [−5, 5]²; x₁ ∈ [−3, 3], x₂ ∈ [−2, 2] in other papers | −1.0316285 at ±(0.08983, −0.7126) | VS(Y99) to 8 digits; full digits VC (Newton) |
| Three-hump camel | 2 | U (Branin 1972 or Dixon and Szegö 1978) | [−5, 5] | 0 at 0; local minima ±(1.74755, −0.87378) → 0.29864 | VS(Jamil-Yang, Adorio 2005); global minimum proven (batch 10a) |
| Beale | 2 | Beale, E. M. L. (1958). On an iterative method for finding a local minimum of a function of more than one variable. Tech. Rep. 25, Statistical Techniques Research Group, Princeton University | [−4.5, 4.5] | 0 at (3, 0.5); a minimum 0.76207 on the bound x₁ = −4.5 | VS(Jamil-Yang, Laguna-Martí); VC; the report unread |
| Booth | 2 | U | [−10, 10] | 0 at (1, 3) | VS(Jamil-Yang, Laguna-Martí) |
| Matyas | 2 | U (credited to Matyas, 1965; Jamil-Yang credit Hedar's collection) | [−10, 10] (Laguna-Martí [−5, 10]) | 0 at 0 | VS(Jamil-Yang, Laguna-Martí) |
| Himmelblau | 2 | Himmelblau, D. M. (1972). *Applied Nonlinear Programming.* McGraw-Hill | [−5, 5] | 0 at (3, 2), (−2.805118, 3.131312), (−3.779310, −3.283186), (3.584428, −1.848126) | (3, 2) VC; the other three U |
| Easom | 2 | Easom, E. E. (1990). *A Survey of Global Optimization Techniques.* M.Eng. thesis, University of Louisville; probably first in a journal in Stuckman, B. E. and Easom, E. E. (1992), *IEEE Trans. SMC* 22(5): 1024-1032, doi:10.1109/21.179841 | [−100, 100] | −1 at (π, π) (proven from the formula) | VS(Jamil-Yang) only: neither the thesis nor the 1992 paper could be read (batch 6); bounds U |
| Bohachevsky 1, 2, 3 | 2 | Bohachevsky, I. O., Johnson, M. E. and Stein, M. L. (1986). Generalized simulated annealing for function optimization. *Technometrics* 28(3): 209-217. doi:10.1080/00401706.1986.10488128 | [−100, 100] (Adorio [−50, 50]) | 0 at 0 | VS(Jamil-Yang, whose Bohachevsky 2 misprints the product of cosines; Adorio 2005 has 1 and 2); whether the paper has variants 2 and 3 U; the paper unread |
| Shekel's foxholes (De Jong's F5) | 2 | De Jong (1975), after Shekel (1971) | [−65.536, 65.536] | 0.9980038377944502 at (−31.97834, −31.97834) (batch 10a, Newton's method; best known) | **VO** (De Jong's thesis, appendix A.6, read in batch 10a: cⱼ = j, K = 500, the grid, the bounds, min ≅ 1); VS(Y99 f14) |
| Hartmann 3 and 6 | 3, 6 | Hartman, J. K. (1973). Some experiments in global optimization. *Naval Research Logistics Quarterly* 20(3): 569-576. doi:10.1002/nav.3800200316 (the 1972 report NPS-55HH72051A, read in batch 6, defines the form with random, unprinted constants and has no 3- or 6-D problem); constants tabulated in Dixon and Szegö (1978) | [0, 1]ⁿ | H3: −3.862782147820755 at (0.1146143, 0.5556488, 0.8525470); H6: −3.322368011415515 at (0.2016895, 0.1500107, 0.4768740, 0.2753324, 0.3116516, 0.6573005) (batch 6, Newton's method) | VS(Y99 tables XII-XIII, read in batch 6, whose H6 p₃₂ = 0.1415 is a misprint for 0.1451); Dixon and Szegö unread |
| Shekel 5, 7, 10 | 4 | Shekel, J. (1971). Test functions for multimodal search techniques. Proc. 5th Princeton Conf. on Information Sciences and Systems; constants in Dixon and Szegö (1978) | [0, 10]⁴ | −10.153199679, −10.402940567, −10.536409817 near (4, 4, 4, 4) (batch 6, Newton's method) | VS(Y99 table XIV, read in batch 6); Jamil-Yang put the minima at (4, 4, 4, 4) with −10.1499, −10.3999, −10.5319, which are neither the minima nor the values there (−10.153196, −10.402819, −10.536284) |
| Langermann | 2 (orig. 10) | Bersini, H., Dorigo, M., Langerman, S., Seront, G. and Gambardella, L. (1996). Results of the first international contest on evolutionary optimisation (1st ICEO). Proc. IEEE ICEC: 611-615. doi:10.1109/ICEC.1996.542670 | [0, 10] (Surjanovic-Bingham) | −4.155809291847785 at (2.793402, 1.597233) (batch 10a, Newton's method from a grid; best known) | 2-D form, sign and constants VS(Molga-Smutnicki 2005, section 2.10, read in batch 10a, no bounds); the contests' functions have 5 and 10 dimensions and a minus sign (the second ICEO's pages, Jamil-Yang function 68); with the minus sign the minimum is −5.1621262; the 1996 paper unread |
| Kowalik | 4 | Y99 f15 (data fit, 11 points); data from Kowalik and Osborne (1968) | [−5, 5] | 3.0748598780560606e-4 at (0.192833, 0.190836, 0.123117, 0.135766) (batch 10a, Newton's method; best known) | VS(Y99 f15 and table XI, read in batch 10a, 1/bᵢ exact); NIST StRD MGH09 (read) rounds bᵢ, with the certified 3.0750560385e-4, reproduced; the book unread |

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

**Checked in batch 6** (Hartmann, Shekel, Easom, Eggholder, Schaffer F6). Read: Hartman's 1972
report (NPS-55HH72051A, through the Internet Archive), Yao, Liu and Lin (1999, tables XII-XIV),
Whitley, Mathias, Rana and Dzubera (1996, the authors' copy), Mishra (2006, MPRA paper 2718) and
the CEC 2005 report. Not reachable: Hartman (1973), Dixon and Szegö (1978), Törn and Žilinskas
(1989), Shekel (1971), Easom (1990), Stuckman and Easom (1992) and Schaffer et al. (1989); a
search inside Dixon and Szegö's volume (Google Books, tokens only, not a reading) finds 0.03815,
0.0381 and 0.1451 but not 0.1415, and −3.86278, −3.32237, −10.1532, −10.4029 and −10.5364.
Findings: Hartman's report has no Hartmann 3 or 6 constants (they are Dixon and Szegö's); Y99's
H6 p₃₂ = 0.1415 is a misprint (it moves the minimum to −3.3219952 at x₂ = 0.1468, away from Y99's
own minimizer), Y99's table I gives n = 4 for f19, and its Shekel appendix drops the minus sign;
Jamil and Yang misprint H3's p₂₂ (0.4837), H6's p₁₆ (0.5586), Shekel's minima (at (4, 4, 4, 4),
with values that aren't f there) and the eggholder's sign; the eggholder's original domain is
[−512, 511], where x₁ = 512 is out of the box, and the usual x₂ = 404.2319 is 404.2318 to 4
decimals. Every minimum is recomputed to 40 digits by Newton's method (mpmath) and, but for Easom
and Schaffer F6 (proven from the formulas), stored as a best known value (`is_proven()` false).

**Checked in batch 10a** (Beale, Booth, Matyas, Bohachevsky 1-3, three-hump camel, Dixon-Price,
Trid, Powell, Langermann, Shekel's foxholes, Kowalik, Schwefel 2.21 and 2.22). Read: De Jong's
thesis (1975, appendix A.6, in the scan that George Mason University's EC lab published, through
the Internet Archive), Yao, Liu and Lin (1999, table I and the
appendix, tables XI-XIV), Jamil and Yang (2013, arXiv:1308.4008), Laguna and Martí's preprint (2002,
the appendix, whose numbering is the 2005 paper's), Molga and Smutnicki (2005), Adorio's MVF
library (2005), Steihaug and Suleiman (2013) for Powell's formula, NIST's StRD MGH09 for Kowalik and
Osborne's data, the second ICEO's function pages (through the Internet Archive), and Surjanovic and
Bingham's pages for the bounds of Langermann. Not reachable: Beale (1958), Powell (1962), Matyas
(1965), Kowalik and Osborne (1968), Bohachevsky, Johnson and Stein (1986), Dixon and Price (1989),
Bersini et al. (1996) and Moré, Garbow and Hillstrom (1981). Findings: Jamil and Yang misprint
Powell's (x₂ − 2x₃)⁴ as (x₂ − x₃)⁴ and give its start as the minimizer, misprint Bohachevsky 2's
product of cosines, give −200 for Trid 10 (it's −210) and call Trid multimodal (it's convex);
Laguna and Martí's Trid prints xᵢxⱼ and their Dixon-Price sum starts at i = 1; Adorio has no
Bohachevsky 3 and puts Bohachevsky's bounds at [−50, 50]; the usual 2-D Langermann minimum −5.1621
is the minimum of the contests' sign, not of Molga and Smutnicki's, whose is −4.15581; Dixon-Price
has, for n ≥ 3, a stationary point 2/3 at (1/3, 0, …, 0) where most runs of CMA-ES, SHADE and PSO
stall in 10 and 30 dimensions; Beale has a minimum 0.76207 on its bound. The minima of Langermann, the foxholes
and Kowalik are recomputed to 40 digits by Newton's method (mpmath) and stored as best known values;
the others are proven from their formulas.

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
solutions): g21, g22; Xia (<https://arnold-neumaier.at/glopt/xia.txt>): g23; Floudas et
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
| g19 | 15 | nonlinear | 33.4761% | 0 | 5 | 0 | 0 | 0 (5 at x*, errata 9) | | 32.6555929502463 | 11-12 |
| g20 | 24 | linear | 0.0000% | 0 | 6 | 2 | 12 | 16 | | 0.2049794002 (Table 4; x* infeasible, g₁ = 0.1438; none feasible, errata 3) | 12-13 |
| g21 | 7 | linear | 0.0000% | 0 | 1 | 0 | 5 | 6 | | 193.724510070035 | 13 |
| g22 | 22 | linear | 0.0000% | 0 | 1 | 8 | 11 | 19 | | 236.430975504001 (236.370313314566 exists, errata 9) | 13-14 |
| g23 | 9 | linear | 0.0000% | 0 | 2 | 3 | 1 | 6 | | −400.055099999999584 | 14-15 |
| g24 | 2 | linear | 79.6556% (g₁ alone; 44.206% with both, errata 9) | 0 | 2 | 0 | 0 | 2 | yes (−x₁ − x₂) | −5.50801327159536 | 15 |

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
x = (2.32952019747762, 3.17849307411774) (the sum gives f*, checked by hand); g16's x₂*,
68.5999999999999943, is 68.6 itself in double precision (checked in batch 5: no clamp needed).

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
   arXiv:1903.04886, Table VIII). **Which paper first reported it is unverified.** Checked in
   batch 5: the report's f* is 30(x₁ + h₁) + 28(x₂ + h₂) at its x*, the value of the organizers'
   code, which evaluates f₁ and f₂ at the right-hand sides of h₁ and h₂; and the report's x* with
   x₁ lowered to 201.78446249355 (h₁ at the tolerance) gives 8853.5338748065 to all its digits,
   which `G17` uses as its best known solution. The objective
   jumps at x₂ = 100 and x₁ = 300 (piecewise), x₂* sits just below 100, and f₁ is defined for
   x₁ < 400 and f₂ for x₂ < 1000 while the bounds include 400 and 1000: the implementation picks the
   closed end and documents it.
3. **g20** has no known feasible solution, and none exists (below): the report calls its x*
   "a little infeasible". `optimum()` returns the value as a best known, not proven, with that
   note. Checked in batch 6: the x* meets the 14 equalities within δ but violates g₁ by 0.1438
   (x₁₃ = 0.158 in (x₁ + x₁₃)/(Σx + e₁)); that total is 20 times the mean violation 0.00718768
   that Takahama and Sakai (2006, "Constrained optimization by the ε constrained differential
   evolution with gradient-based mutation and feasible elites", IEEE CEC 2006) give for it. It evaluates to 0.204979400285636, which table 4
   truncates to 0.2049794002 (the text gives no f(x*)); `G20` stores table 4's value. **No
   feasible solution exists** (derived in batch 6, not in the report): g₁…g₆ are nonnegative sums
   over positive denominators, so they force x₁, x₂, x₃, x₇, x₈, x₉, x₁₃, x₁₄, x₁₅, x₁₉, x₂₀, x₂₁
   to 0. With pᵢ = (xᵢ/bᵢ)/Σⱼ xⱼ/bⱼ over the six liquids left and S₂ = Σᵢ₌₁₃²⁴ xᵢ/bᵢ, h₁…h₁₂
   give Σ cᵢpᵢ = 40 (to 6δ) and Σᵢ₌₁₃²⁴ xᵢ = S₂ Σ bᵢcᵢpᵢ/40; over distributions p with Σ cᵢpᵢ =
   40 (an LP with two constraints, whose vertices mix x₅, the only cᵢ > 40, with one other), the
   least is 109.57 S₂, with x₅ and x₁₀. h₁₄ gives S₂ ≥ (1.671 − δ − Σᵢ₌₁¹² xᵢ/dᵢ)/k ≥ 0.0115, so
   Σx ≥ 1.26 where h₁₃ allows 1 + δ. The point of that bound, built in `G20`'s test, meets every
   constraint but h₁₃, which it misses by 0.287. At the origin the equalities are 0/0: the fitness
   is invalid there.
4. **g22:** the earlier best known was 382.902205; 236.430975504001 is εDE's (Takahama and Sakai
   2006 found it again, and cite it as new). Spettel, Ba and
   Arnold (2022, Evolutionary Computation 30(4): 531-553, doi:10.1162/evco_a_00311) state a better
   value exists: **unverified (paywalled)**.
5. g10: Table 3 says a = 6, the text names only g1-g3 as active; at x* all six are active (to
   1e-10, checked in batch 5). g16: Table 3 says LI = 4 and a = 4; g₂, g₅ and g₆ are the linear
   ones, and g₂, g₃, g₄, g₅ and g₃₆ are active at x* (x₂ at its lower bound). The x* of g07 and
   g09 exceed g₁ by 6e-14 and 4e-16 (rounding; the report says so for g07).
6. g04: some engineering papers use Himmelblau's variant with 0.00026 x₁x₄ in g1 (optimum about
   −31025.56); the CEC form uses 0.0006262 and has −30665.539. genoxide implements the CEC form.
   **The variant's value is unverified.**
7. Typography only: "93 disjointed spheres" in g12 means 9³ = 729; "+ +2x₃" in g14's h1 and
   "+ +x₃²" in g15's h1; unbalanced parentheses in g17's h2, h3 and h4; "c₁₀ = −22.179,." in g14; g18's constraints
   are labelled gg₁…gg₁₃; g17's eq. 35 writes f(x₁) + f(x₂) for f₁(x₁) + f₂(x₂).
8. g12's constraint is a disjunction: min over p, q, r ∈ {1..9} of (x₁−p)² + (x₂−q)² + (x₃−r)² −
   0.0625 ≤ 0, one NI constraint.
9. **Checked in batch 6 (g19-g24).** The formulas, bounds and data were read from the typeset
   report (tables 1 and 2 rendered at 300 dpi; c of g19 is symmetric, a₁₃…a₂₄ and b₁₃…b₂₄ of g20
   repeat a₁…a₁₂ and b₁…b₁₂), and table 3's LI, NI, LE and NE counted from them: all match. Each
   x* evaluates to the text's f(x*): g19 to 2.8e-14, g21 to 2.8e-14, g22 to 5.7e-14, g23 to
   1.1e-13 (with the repaired x₈, x₉) and g24 exactly; g20 as in item 3. The organizers' C code
   (fcnsuite.c, in the report's repository, with no license) was only read, to compare its
   formulas with the report's: they agree, and no value was taken from it.
   - g19: table 3 says a = 0; at x* all five constraints are active (|g| ≤ 1e-14). ρ: 33.467 %
     of 2·10⁶ random points are feasible (table 3: 33.4761 %).
   - g21: x* is on the tolerance boundary of all five equalities (|h| = δ, beyond it by 1e-12
     from rounding), with g₁ active. The equalities leave x₄ free: h₃…h₅ give x₅…x₇, and h₁, h₂
     factor as (x₄ − 300)(x₃ − 25(x₅ − x₆)) and (100 − x₄)(x₂ − 155.365 + 25x₇). With g₁ active f
     is a function of x₄: 193.788 = 35 (25 ln 2)^0.6 at x₄ = 100 (x₂ = 0, every equality exact),
     rising to 330.6 near x₄ = 293 and falling to 325.1 at x₄ = 299.53, where x₂ reaches 40. The
     second end is the local optimum (324.70 with the tolerance) where about a third of SHADE's
     runs end (32 of 100 with a population of 50). The example reaches the best known on 982 of
     1000 seeds with SHADE (population 30) at an ε level lowered to 0 over the first 150,000
     evaluations (Takahama and Sakai's ε constrained method, with `Engine::control` and re-evaluation).
   - g22: x* is feasible, every |h| < δ, and g₁ = −2.2e-7. **The report's best known is not
     the best:** the 19 equalities leave x₁, x₈ and x₉ free (h₁…h₆ and h₁₀, h₁₁ are linear in x₈
     and x₉: x₁₀ = 430 − x₈, x₁₁ = 440 − x₉ + x₈, x₁₂ = 160 + x₉, x₁₆ = 440 − x₉, x₁₇ = 160;
     h₁₂…h₁₆ give x₁₈…x₂₂ as logarithms; h₁₇…h₁₉ give x₁₃ = 30/(x₁₈ − x₁₉),
     x₁₄ = 40/(x₂₀ − x₂₁), x₁₅ = 60/(ln 160 − 4.60517); h₇…h₉ give x₂…x₄), and with g₁ active f
     is a function of x₈, x₉ on x₈ ≥ 130, x₉ ≥ x₈ + 40 (x₁₀ ≤ 300, x₁₁ ≤ 400). A 2001 × 3001
     grid and a local search find its least value at the corner x₈ = 130, x₉ = 170:
     236.370313314566, every equality met exactly, 0.0607 below the report's 236.430975504001
     (which has x₈ = 130.075, x₉ = 170.817). This agrees with Spettel, Ba and Arnold (2022, item 4
     above) that a better value exists. `G22` keeps the report's value as `optimum()` and gives
     this solution in its docs and tests. No run of genoxide's algorithms on all 22 variables (SHADE,
     L-SHADE, CMA-ES with or without restarts; L-SHADE infeasible on all of 1000 seeds, 500,000
     evaluations) found a feasible solution, as in the organizers' comparison of the session's
     entries. Deb's rules add up raw violations, and h₁…h₆ (in units up to 10⁷) drown the others:
     with each violation divided by its typical size, SHADE and L-SHADE end feasible, at 243 to
     317. The example searches x₁, x₈, x₉ with SHADE and solves the other 19 from the equalities
     in order: 236.370313314566 on all of 1000 seeds, after 23,400 to 27,500 evaluations.
   - g23: x* meets h₁…h₄ at |h| = δ, g₂ is active and g₁ = −2.5e-6. (0, 100, 0, 100, 0, 0, 100,
     200, 0.01) meets every constraint exactly, at f = −400: the best known is 0.0551 lower
     through the tolerance.
   - g24: **table 3's ρ = 79.6556 % is the share where g₁ alone holds** (79.651 %, integrated);
     with both constraints it is 44.206 %. g₂'s bound on x₂ is 4((x₁ − 1)(x₁ − 3))², so the
     report's "two disconnected sub-regions" meet at (1, 0). The optimum is proven by reducing to
     x₁ alone (the best x₂ is min(4, both bounds)): the minimum is where the bounds cross, a root
     of x₁⁴ − 12x₁³ + 40x₁² − 48x₁ + 17, which the report's x₁ is to 1.5e-14.

### 1.3 Multi objective, unconstrained

Read in full during the survey: Zitzler, Deb and Thiele (2000), the DTLZ technical report (2001)
and the DTLZ CEC 2002 paper (and in batch 8, Deb and Jain (2014), parts I and II, below). The
WFG, MaF, CEC 2009, Kursawe, Poloni,
Viennet and Van Veldhuizen sources were paywalled or blocked: those entries are **U** and must be
checked in the original before implementation (the WFG and MaF papers first: MaF's is open
access at Springer; WFG's through a library).

#### Classic problems

| Problem | Original | n, M | Bounds | Pareto set / front | Status |
|---|---|---|---|---|---|
| Schaffer 1 (SCH1): x², (x − 2)² | Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic algorithms. Proc. 1st ICGA: 93-100 (and his 1984 thesis, Vanderbilt University) | 1, 2 | the domain varies; NSGA-II's table I [−10³, 10³] (implemented) | PS x ∈ [0, 2]; PF f₂ = (√f₁ − 2)², f₁ ∈ [0, 4] | functions and PS VS(DTLZ report §3); bounds VS(NSGA-II table I); the original U |
| Schaffer 2 (SCH2): piecewise f₁, f₂ = (x − 5)² | Schaffer (1985) | 1, 2 | [−5, 10] | PS [1, 2) ∪ [4, 5]: disconnected (x = 2 is dominated by x = 4); PF (f₁ − 3)² on [−1, 0), (f₁ − 1)² on [0, 1] (VC) | formula and bounds VS(Van Veldhuizen, D. A. (1999), PhD thesis AFIT/DS/ENG/99-01, <https://scholar.afit.edu/etd/5128>, table B.1, after Srinivas and Deb 1994); the original U |
| Fonseca-Fleming (FON) | Fonseca, C. M. and Fleming, P. J. (1995). An overview of evolutionary algorithms in multiobjective optimization. *Evolutionary Computation* 3(1): 1-16. doi:10.1162/evco.1995.3.1.1 | any (NSGA-II's table I: 3), 2 | [−4, 4] | PS x₁ = … = xₙ = t, t ∈ [−1/√n, 1/√n]; PF f₂ = 1 − exp(−(2 − √(−ln(1 − f₁)))²) (derived) | VS(DTLZ report eq. 1, NSGA-II table I); PF VC; the original U. The variables must be equal on the PS (the DTLZ report's wording omits it) |
| Kursawe (KUR) | Kursawe, F. (1991). A variant of evolution strategies for vector optimization. PPSN I, LNCS 496: 193-197. doi:10.1007/BFb0029752 | 3, 2 | [−5, 5] | disconnected: the point (−20, 0) at x = 0 and three curves (Deb et al. 2002 and Van Veldhuizen 1999 say three regions, and plot the point apart); no closed form | VS(NSGA-II table I, sin(xᵢ³), implemented); Van Veldhuizen's table B.1 prints sin(xᵢ)³ and reports (from Laumanns) that the original is misprinted: U |
| Poloni (POL) | Poloni, C., Giurgevich, A., Onesti, L. and Pediroda, V. (2000). Hybridization of a multi-objective genetic algorithm, a neural network and a classical optimizer for a complex design problem in fluid dynamics. *Computer Methods in Applied Mechanics and Engineering* 186(2-4): 403-420. doi:10.1016/S0045-7825(99)00394-1 | 2, 2 | [−π, π]² | disconnected; no closed form | VS(NSGA-II table I, minimized; Van Veldhuizen's table B.1 maximizes the negatives and reports a typo in the cited paper); the original U |
| Viennet 1-3 (VNT1-3) | Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic algorithm for determining a Pareto set. *International Journal of Systems Science* 27(2): 255-260. doi:10.1080/00207729608929211 | 2, 3 | VNT1 [−2, 2]², VNT2 [−4, 4]², VNT3 [−3, 3]² (the thesis's table 5.3 uses wider ones) | VNT1: the triangle (0, 1), (0, −1), (1, 0) (each objective a squared distance plus a constant; VC); VNT2, VNT3 no closed form | VS(Van Veldhuizen, D. A. (1999), PhD thesis AFIT/DS/ENG/99-01, <https://scholar.afit.edu/etd/5128>, table B.1); the original U |
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
  starts at f₁ = 0.2807753188153697, at x₁ = atan(9π)/(6π) where tan(6πx₁) = 9π (derived); the
  constant genoxide had, 0.2807753191, was 3·10⁻¹⁰ too high.
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
| DTLZ5 (eq. 25) | 10 | claimed a curve; x_M = 0.5 | **The report's eq. 25 has typos** (cos(θᵢ π/2), θ₁ undefined): read θ₁ = x₁ π/2 and θᵢ = π/(4(1 + g)) (1 + 2g xᵢ) for i = 2…M − 1, f with cos/sin(θᵢ). The front is not a curve for M ≥ 4 (Huband et al. 2006: U; a solution with g > 0 that no point of the curve dominates is in genoxide's tests) |
| DTLZ6 (eq. 26) | 10 | as DTLZ5, g = Σ xᵢ^0.1 | same caveat |
| DTLZ7 (eq. 27) | **20** | 2^(M−1) disconnected regions; x_M = 0 (g = 1), f_M = 2h | the front is built by sampling f₁…f_{M−1} ∈ [0, 1] and filtering non-dominated points; the region bounds per fᵢ, derived at implementation: [0, 0.2514118360889171] ∪ (0.6316265307000612, 0.8594008566447239], the first two local maxima of fᵢ(1 + sin 3πfᵢ) and where it climbs back to the first |
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
doi:10.1109/TEVC.2013.2281534. **VO (batch 8)** in the authors' copies of the accepted versions
(<https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf>, k2012010.pdf; pages rendered and read; the
typeset journal versions not compared):

- Scaled DTLZ1 and DTLZ2 (part I, section V-C, tables VII and VIII): fᵢ multiplied by s^(i−1).
  The text says "a factor 10ⁱ" and its example multiplies f₁, f₂, f₃ by 10⁰, 10¹, 10², as figures
  24-29 show; the captions of tables VII and VIII say "10ⁱ, i = 1, …, M": **the paper contradicts
  itself**, and genoxide follows the example and figures. Table VIII's s: DTLZ1 10, 10, 3, 2, 1.2
  and DTLZ2 10, 10, 3, **3**, **2** for M = 3, 5, 8, 10, 15 (the survey recalled DTLZ1's values
  for both). For other M, genoxide takes the next larger M's (a documented choice), and
  `with_factor` sets any.
- Convex DTLZ2 (part I, section V-D, eq. 8): fᵢ⁴ for i < M, f_M²; front f_M + Σ √fᵢ = 1.
- Inverted DTLZ1 (part II, section VIII-A, eq. 9): fᵢ ← 0.5 (1 + g) − fᵢ with DTLZ1's g; the
  front, derived: Σ fᵢ = (M − 1)/2 with fᵢ in [0, 1/2] (for M = 2, DTLZ1's line). The paper's
  inverted DTLZ1 uses "the original formulation" of DTLZ1: M + 4 variables assumed from part I's
  DTLZ1 (k = 5), **not stated for the inverted problem**.

#### WFG1-WFG9

Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
problems and a scalable test problem toolkit. *IEEE TEVC* 10(5): 477-506.
doi:10.1109/TEVC.2005.861417. Earlier: Huband, S., Barone, L., While, L. and Hingston, P.
(2005). A scalable multi-objective test problem toolkit. EMO 2005, LNCS 3410: 280-295.
doi:10.1007/978-3-540-31880-4_20. **VO (batch 4):** checked in the TEVC paper as published (the
authors' copy, www.wfg.csse.uwa.edu.au/publications/WFG2006c.pdf, through the Internet Archive:
tables X-XIV, sections VIII-IX), in the EMO paper's corrected version of 25 May 2005 (table 6)
and its errata, and against the authors' C++ toolkit, `WFG_v2006.03.28.zip` (compiled; its
license allows use and copying for any purpose with its notice kept; used only for test values).
Two findings: the toolkit's README recommends k = 4 for M = 2 and 2(M − 1) for more (the paper's
experiments use k = 4, l = 20), which genoxide's defaults follow; and in floating point zᵢ / 2i
is never exactly 0.35 for some i (3, 6, 12, 24, …), which WFG1's b_poly(·, 0.02) turns into a
distance of about 0.48 in that parameter, so no genome reaches WFG1's front at the default sizes
(the toolkit's README warns of the bias; see `Wfg1`'s docs). Ishibuchi et al. (2016) is
doi:10.1109/TEVC.2015.2505784, still U (paywalled); genoxide gives no front for WFG3 with
M ≥ 3. The entry as surveyed:

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
  www.wfg.csse.uwa.edu.au) is the authors' reference implementation; version 2006.03.28 is kept
  by the Internet Archive, and its license allows generating test values (section 3.3).

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
report. BNH, OSY and the Van Veldhuizen constrained problems could not be reached: **U**. CTP's
report was read in batch 7 (below). The PDFs were read as extracted text, so load-bearing constants get a check against the
typeset paper when implemented. Constraints below are written as the papers print them (mostly
"≥ 0 is feasible"); genoxide normalizes to g ≤ 0 (section 2).

#### Classic two-objective problems

| Problem | Original | n, bounds | Objectives | Constraints | Pareto front | Status |
|---|---|---|---|---|---|---|
| CONSTR | Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE TEVC* 6(2): 182-197. doi:10.1109/4235.996017, Table V | 2; x₁ ∈ [0.1, 1], x₂ ∈ [0, 5] | f₁ = x₁, f₂ = (1 + x₂)/x₁ | x₂ + 9x₁ ≥ 6; −x₂ + 9x₁ ≥ 1 | derived: x₂ = 6 − 9x₁ for x₁ ∈ [7/18, 2/3] (f₂ = (7 − 9f₁)/f₁), then x₂ = 0 for x₁ ∈ [2/3, 1] (f₂ = 1/f₁); ideal (0.3889, 1), nadir (1, 9) | VO |
| SRN | Srinivas, N. and Deb, K. (1994). Multiobjective optimization using nondominated sorting in genetic algorithms. *Evolutionary Computation* 2(3): 221-248. doi:10.1162/evco.1994.2.3.221 (restated in NSGA-II's Table V) | 2; [−20, 20]² | f₁ = (x₁ − 2)² + (x₂ − 1)² + 2, f₂ = 9x₁ − (x₂ − 1)² | x₁² + x₂² ≤ 225; x₁ − 3x₂ ≤ −10 | derived, three pieces: the second constraint's boundary x₁ = 3x₂ − 10 for x₂ ∈ [2.5, 3.7]; the line x₁ = −2.5, x₂ ∈ [2.5, 14.79]; the first constraint's circle from there to (−4.840977, 14.197357), where ∂f₂ along it is 0; from (10.1, 2.61) to (222.9692, −217.7390). **The often-quoted "x₁ = −2.5" is only part of the front**; checked against random feasible points | formulas VS(NSGA-II table V; Binh and Korn 1997 §5.1; Van Veldhuizen table B.2); the 1994 paper not read; front derived |
| TNK | Tanaka, M., Watanabe, H., Furukawa, Y. and Tanino, T. (1995). GA-based decision support system for multicriteria optimization. Proc. IEEE SMC 2: 1556-1561. doi:10.1109/ICSMC.1995.537993 | 2; [0, π]² | f = x | −x₁² − x₂² + 1 + 0.1 cos(16 arctan(x₁/x₂)) ≤ 0; (x₁ − 0.5)² + (x₂ − 0.5)² ≤ 0.5 | disconnected, on the first constraint's boundary: sampled, filtered | VO (NSGA-II); arctan(x₁/x₂) at x₂ = 0 needs a stated convention (atan2) |
| WATER | Ray, T., Tai, K. and Seow, K. C. (2001). Multiobjective design optimization by an evolutionary algorithm. *Engineering Optimization* 33(4): 399-424. doi:10.1080/03052150108940926 (restated in NSGA-II's Table V) | 3; x₁ ∈ [0.01, 0.45], x₂, x₃ ∈ [0.01, 0.1] | 5 | 7 (≤ constants) | none given (NSGA-II's Table VI gives reached ranges) | VO (NSGA-II); its 6th constraint prints 0.417 (x₁x₂), a product where the others divide: check in Ray et al. Implemented once, as 1.5's water resource planning |
| BNH | Binh, T. T. and Korn, U. (1997). MOBES: a multiobjective evolution strategy for constrained optimization problems. Proc. 3rd Int. Conf. on Genetic Algorithms (Mendel 97), Brno: 176-182 | 2; **[−15, 30]² in the original** (§5.2); [0, 5] × [0, 3] is Van Veldhuizen's restatement (table B.2) | f₁ = 4x₁² + 4x₂², f₂ = (x₁ − 5)² + (x₂ − 5)² | (x₁ − 5)² + x₂² ≤ 25; (x₁ − 8)² + (x₂ + 3)² ≥ 7.7 | derived, with the original bounds: x₁ = x₂ = t ∈ [0, 5], f = (8t², 2(t − 5)²), from (0, 50) to (200, 0), both ends on the first constraint's boundary; ideal (0, 0), nadir (200, 50). With the restated bounds it's cut at x₂ = 3 (join (72, 8), end (136, 4)) | **VO** (the authors' version of the paper, from their 1997 web page) |
| OSY | Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria optimization problems using the simple genetic algorithm. *Structural Optimization* 10(2): 94-99. doi:10.1007/BF01743536 | 6; x₁, x₂, x₆ ∈ [0, 10], x₃, x₅ ∈ [1, 5], x₄ ∈ [0, 6] | f₁ = −[25(x₁ − 2)² + (x₂ − 2)² + (x₃ − 1)² + (x₄ − 4)² + (x₅ − 1)²], f₂ = Σ xᵢ² | 6 (≥ 0) | five pieces with x₄ = x₆ = 0, tabulated in Deb, Pratap and Meyarivan (2001, EMO 2001, LNCS 1993: 284-298; KanGAL report 200002), table 1, and derived again: x₁ ∈ [4.056543, 5] on the third piece and x₃ ∈ [1, 3.731685] on the fourth, where the two meet (the table's 3.732 is not 2 + √3); from (−274, 76) to (−42, 4) | VS(Deb, Pratap and Meyarivan (2001, EMO 2001, LNCS 1993: 284-298; KanGAL report 200002), eq. 3; Van Veldhuizen table B.2); the original U |
| Viennet 4, Van Veldhuizen's MOP-C1..C3 | Van Veldhuizen, D. A. (1999). *Multiobjective Evolutionary Algorithms: Classifications, Analyses, and New Innovations.* PhD thesis, AFIT/DS/ENG/99-01 (DTIC ADA364478) | — | — | — | — | **U** (optional) |

#### CTP1-CTP7 (and CTP8)

Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective
evolutionary optimization. EMO 2001, LNCS 1993: 284-298. doi:10.1007/3-540-44719-9_20.
**VO (batch 7), from the authors' KanGAL report 200002** (October 2000, 15 pages, the EMO paper's
preprint; the file `tech-rep5.ps.gz` of the KanGAL site, kept by the Internet Archive, converted
with Ghostscript and read page by page; the LNCS text not compared). Coello's list of technical
reports gives the same title as KanGAL report 200005 by Deb and Meyarivan; the file itself says
200002. The report defines CTP1 (eq. 4, p. 6) and the generator of CTP2-CTP7 (eq. 5, p. 7), with
the parameters below on pp. 7-11:

| | θ | a | b | c | d | e |
|---|---|---|---|---|---|---|
| CTP2 | −0.2π | 0.2 | 10 | 1 | 6 | 1 |
| CTP3 | −0.2π | 0.1 | 10 | 1 | 0.5 | 1 |
| CTP4 | −0.2π | 0.75 | 10 | 1 | 0.5 | 1 |
| CTP5 | −0.2π | 0.1 | 10 | 2 | 0.5 | 1 |
| CTP6 | 0.1π | 40 | 0.5 | 1 | 2 | −2 |
| CTP7 | −0.05π | 40 | 5 | 1 | 6 | 0 |
| CTP8 (two constraints) | 0.1π / −0.05π | 40 / 40 | 0.5 / 2 | 1 / 1 | 2 / 6 | −2 / 0 |

Findings:

- **g, n and the bounds aren't in the report**: its experiments use "Rastrigin's function as the g
  functional" and five variables, without the formula. The authors' NSGA-II code (version 1.1.6,
  `nsga2-gnuplot-v1.1.6.tar.gz` on Deb's MSU page, read only; its notice allows academic use) has
  all eight with two variables, g = 1 + x₂, x₁ ∈ [0, 1] and x₂ ∈ [0, 1] (CTP1-5) or [0, 10]
  (CTP6-8): genoxide follows it and says so.
- **Eq. 5 prints f₂ = g (1 − f₁/g)**; the report's figures 6-11 draw the unconstrained front as
  the curve 1 − √f₁, and the code computes g (1 − √(f₁/g)), which genoxide uses.
- CTP1's a and b come from a procedure (p. 6) that puts each boundary through the previous one at
  f₁ = j/(J + 1); the table prints them to three digits (0.858, 0.541; 0.728, 0.295), which the
  code and genoxide use. The test re-runs the procedure. With the rounded values, the front's
  corners are at f₁ = 0.33367 and 0.66789.
- The code writes each constraint as a ratio, left/right − 1 ≥ 0; genoxide keeps the report's
  difference (feasible at the same points, finite where the right side is 0).
- **CTP8 isn't in the report** (CTP1-CTP7 only, as Tanabe and Oyama 2017 say); later papers credit
  it to Deb's 2001 book, which wasn't read. Its definition is the code's (the table's row), **U**
  against the book.
- Fronts, derived here: CTP1 three analytic pieces; CTP2 13 pieces of the boundary; CTP3 and CTP4
  the 13 points (cos θ k/10, 1 + sin θ k/10); **CTP5 one continuous piece and 15 points**, not 16
  points: near v = 0, sin(bπv²)^0.5 grows linearly and the boundary leaves the line at a shallow
  slope; CTP6 one piece (v from 1.76 to 1.84, inside the report's "1 ≤ v ≤ 2"); CTP7 six pieces
  of 1 − √f₁ and the point (0, 1.0446); CTP8 three pieces of CTP6's. `optimal_front` samples the
  boundaries of the feasible region (constraint boundaries, zeros of the right side, g = 1, the
  largest g, f₁ = 0 and 1) with bisection at the ends of feasible stretches, and filters; the tests
  check it against 100,000 random genomes per problem. In floating point the points of CTP3-CTP5
  are infeasible by about 3e-8 (sin(kπ) ≈ 1e-15 under a square root).
- The report's OSY (eq. 3) and table 1 match genoxide's `Osy`.

#### Constrained DTLZ (C-DTLZ)

Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using
reference-point based nondominated sorting approach, part II: handling constraints and extending
to an adaptive approach. *IEEE TEVC* 18(4): 602-622. doi:10.1109/TEVC.2013.2281534. **VO (from
the authors' preprint; the TEVC text not compared).** x ∈ [0, 1]ⁿ, M ∈ {3, 5, 8, 10, 15},
constraints ≥ 0. The paper tabulates no fronts; they're computed from the forms below. Checked
again in batch 7 against the typeset manuscript (rendered pages, not extracted text), and part
I's convex DTLZ2 (section VII-C, eq. 8: fᵢ⁴ for i < M, f_M², front f_M + Σ √fᵢ = 1) in its
manuscript too.

| Problem | n | Constraints | Front |
|---|---|---|---|
| C1-DTLZ1 | M + 4 | 1 − f_M/0.6 − Σ_{i<M} fᵢ/0.5 ≥ 0 (eq. 4) | DTLZ1's (Σ f = 0.5), all feasible, behind an infeasible band |
| C1-DTLZ3 | M + 9 | (Σ fᵢ² − 16)(Σ fᵢ² − r²) ≥ 0 (eq. 5); r = 9, 12.5, 12.5, 15, 15 for M = 3, 5, 8, 10, 15 | DTLZ3's unit sphere |
| C2-DTLZ2 | M + 9 | feasible inside one of M + 1 spheres of radius r (centered at the unit vectors and at 1/√M): −min{minᵢ[(fᵢ − 1)² + Σ_{j≠i} fⱼ² − r²], Σ(fᵢ − 1/√M)² − r²} ≥ 0; r = 0.4 for M = 3, 0.5 otherwise | the parts of the unit sphere inside the spheres (disconnected) |
| Convex C2-DTLZ2 | M + 9 | Σ(fᵢ − λ)² − r² ≥ 0, λ = mean of f (eq. 6); r = 0.225, 0.225, 0.26, 0.26, 0.27 | convex DTLZ2's front outside a cylinder around the diagonal |
| C3-DTLZ1 | M + 4 | M constraints Σ_{i≠j} fᵢ + fⱼ/0.5 − 1 ≥ 0 (eq. 7) | derived: {f ≥ 0 : Σ fᵢ + minⱼ fⱼ = 1}; DTLZ1's front is infeasible |
| C3-DTLZ4 | M + 4 (as printed) | M constraints fⱼ²/4 + Σ_{i≠j} fᵢ² − 1 ≥ 0 (eq. 8) | derived: {f ≥ 0 : minⱼ[fⱼ²/4 + Σ_{i≠j} fᵢ²] = 1} |

Pitfalls: C2-DTLZ2's constraint is printed as c = max{maxᵢ […], […]} with no inequality sign
(batch 7, in the typeset manuscript); the text and figure 7 make the inside of the spheres
feasible, which only min{…} ≤ 0 gives. **Table V confirms that reading and the radii**: 58 of the
91 Das-Dennis directions (M = 3) and 80 of 210 (M = 5) meet the feasible part of the sphere, and 47
of 91 and 97 of 210 the convex front outside the cylinder, the table's U/H counts (a test);
C3-DTLZ1 is printed with i and j swapped (then every constraint is 2S − 1 ≥ 0 and DTLZ1's front is
feasible, against the text and figure 13); f_M/0.6 and fᵢ/0.5 are fractions (some extractions show exponents). C1-DTLZ1,
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

**Checked in batch 8.** Every formula was read from the rendered pages (eqs. 9, 12-28, table
II, the supplement's tables S-R-I, S-R-VIII and S-R-IX), and compared with the authors' C++ code
(`CMO_NSGA2/cexe/exc/AlgC.cpp` in MW.rar, which has no license: read only, nothing taken). They
agree but for g₁: the code's `i / (2 * X.size())` divides integers and is always 0, so its optimal
zᵢ are 0.5; genoxide follows the paper, 0.5 + (i − 1)/(2n). MW6's powers read by the paper's own
local adjustment A sin(B l^C)^D (eq. 9): l = cos(6 arctan(f₂/f₁)⁴)¹⁰, as the code has it. The
many-objective setting is n = m − 1 + 13 (supplement), genoxide's default M + 12. Fronts: derived
for every problem, computed as the first feasible point of each ray of x₁ (vertical for MW1-3,
radial for the others), non-dominated, and checked against solutions near the Pareto set and the
authors' sampled fronts (`TruePF/MW*.dat`, compared only: IGD of 1e-4 to 1e-3 both ways for the
two-objective problems, and the three-objective samples lie on the derived surfaces, in the
derived bands and ranges, to 1e-8). Findings: MW5's
front has, besides the sixteen points, two short curves near the axes (to l₁ = 1/72), which the
paper's figure and the authors' samples show as points; MW11's isolated point (1, 1) needs x₁ = 1
and g₃ = 1 exactly and was never reached (the example shows it); MW12's front starts at the limit
(0, 1), feasible only in exact arithmetic; MW9's front is analytic (below, confirmed); MW14's ranges
are [0, 0.7313522974897325] and (1.3296339087402259, 1.5]. The MW8 bands are [0, π/24],
[π/8, 5π/24], [7π/24, 3π/8] and [11π/24, π/2].

Pitfalls: constraint directions are mixed (some printed ≤ 0, most ≥ 0: implement as printed);
MW12 and MW13 use |sin| in f₂ and plain sin in the constraints; MW4 and MW8 order their objectives
so f_M depends on x₁; feasible fractions are tiny (< 0.1‰ for most). The authors' code
(<https://intleo.csu.edu.cn/codes/MW.rar>) was read in batch 8 (above).

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
| Four-bar truss | Stadler, W. and Dauer, J. (1992). Multicriteria optimization in engineering: a tutorial and survey. In *Structural Optimization: Status and Promise*, Progress in Astronautics and Aeronautics 150, AIAA: 209-249; Cheng, F. Y. and Li, X. S. (1999). *Engineering Optimization* 31(5): 641-661 | x₁, x₄ ∈ [1, 3], x₂, x₃ ∈ [√2, 3] cm² | 2 (volume, displacement) | 0 | **analytic (HC, derived for this plan):** x₃ = √2; three segments, from (1400, 0.04) to (3048.53, 0.0027614) (batch 9; the survey had (3600, 0.00714)) | the volume term is √2 x₃ (VS: GAMS model "truss2" citing Stadler and Dauer); **TI20 and its copies use √x₃, dimensionally wrong** (it shifts the front by −162.16 in f₁) |
| Rocket injector | Vaidyanathan, R., Tucker, P. K., Papila, N. and Shyy, W. (2003). Computational-fluid-dynamics-based design optimization for single-element rocket injector. AIAA paper 2003-296 (J. Propulsion and Power 20(4), 2004: U); Goel, T. et al. (2007). *CMAME* 196: 879-893 | 4 in [0, 1] | 3 (the original has 4) | 0 | sampled reference | response-surface polynomials VS(TI20) |
| Water resource planning | Musselman, K. and Talavage, J. (1980). A tradeoff cut approach to multiple objective optimization. *Operations Research* 28(6): 1424-1435; Ray, T., Tai, K. and Seow, K. C. (2001). *Engineering Optimization* 33(4): 399-424 | 3 | 5 | 7 | sampled reference | VS(TI20); originals U |
| Conceptual marine design | Parsons, M. G. and Scott, R. L. (2004). Formulation of multicriterion design optimization problems for solution with scalar numerical optimization methods. *J. Ship Research* 48(1): 61-76. doi:10.5957/jsr.2004.48.1.61 | 6 | 3 | 9 | sampled reference | VS (a 2023 paper restating the model); **TI20's code computes sea days as (5000/24) Vk instead of 5000/(24 Vk)** |
| Vehicle crashworthiness | Liao, X., Li, Q., Yang, X., Zhang, W. and Li, W. (2008). Multiobjective optimization for crash safety design of vehicles using stepwise regression model. *Struct. Multidisc. Optim.* 35(6): 561-569 | 5 in [1, 3] mm | 3 | 0 | sampled reference | VS(TI20) |
| Coil compression spring, hatch cover, reinforced concrete beam, cantilever (Deb's book), multiple-disk clutch brake | Sandgren (1990); Amir and Hasegawa (1989); Deb (2001, book); Osyczka (2002, *Evolutionary Algorithms for Single and Multicriteria Design Optimization*, Physica) | — | 2 | — | — | U or VS(TI20) with discrepancies between its supplement and code (bounds, a discrete list "3,1" for 3.10): deferred until the originals are read |

**Checked in batch 9.** Read: Deb, Pratap and Moitra's KanGAL report 200002 (the PPSN 2000
preprint, rendered pages): the two-bar truss (eq. 1, with 0 ≤ xᵢ ≤ 0.01 in the text) and the
welded beam (eq. 2: Ragsdell and Phillips's form, the five-constraint form's first four
constraints, bounds 0.125 ≤ h, b ≤ 5 and 0.1 ≤ l, t ≤ 10); Deb and Srinivasan's KanGAL report
2005007 (the two-bar truss's front, eqs. 1-7, and table 1); Jain and Deb's part II manuscript
(appendix: the car side impact's three objectives, and WATER); the NSGA-II paper (TEVC table V, and
the preprint's table VI: WATER, identical); Vaidyanathan et al.'s AIAA paper 2003-296 (authors'
copy, NASA NTRS 20030060421, eqs. A1-A4: the rocket injector); Costa and Fernandes (WCSMO 2009,
problem (4-truss): the four-bar truss after Stadler and Dauer); Yang, Karamanoglu and He (2013,
eqs. 10-12: the disc brake); Saad, Emam and Houssein (2025, *Sci. Rep.* 15: 5126, eqs. 22-25: the
speed reducer and the disc brake); de Carvalho and Sichman (OptMAS 2018: crashworthiness); and TI20's
supplement. Not read: Osyczka and Kundu, Ray and Liew, Kurpati et al., Stadler and Dauer, Cheng
and Li, Liao et al., Musselman and Talavage, Ray, Tai and Seow, Goel et al., Parsons and Scott.
Found: the four-bar truss's front ends at (2200 + 600√2, (2√2 − 2)/300) ≈ (3048.53, 0.0027614),
not (3600, 0.00714); WATER's front is derived (every x with x₃ = 0.01 and x₁x₂ ≥ 0.00139/1.0306,
none dominating another), and g₆'s product can't matter (g₇ implies the quotient form); the speed
reducer's restatements disagree on the second shaft's constant (1.575·10⁸ in TI20, 1.275·10⁸ in
Saad et al.; genoxide follows 1.575·10⁸, Golinski's); TI20's welded beam takes the other form's
shear stress and buckling load; the rocket injector's three objectives are the original's TF_max,
TT_max and X_cc (TW₄ left out, as TI20 restates Goel et al.). **Conceptual marine design is left
out**: Parsons and Scott couldn't be read, and the restatements disagree beyond the sea-days error
noted above (Kudela's 2023 model, *Computers* 12(11): 225, has V = 0.5114 Vk where
TI20 has 0.5144, a least deadweight of 25,000 where TI20 has 3000, and the annual cargo from the
deadweight where TI20 has the cargo deadweight). The fronts without a formula are measured in the
examples against reference fronts from ε-constraint and achievement-scalarizing runs of SHADE plus
long runs of the multi-objective algorithms; the script that made them isn't in the repository
yet, and no reference front is embedded (`approximate_front()` below isn't implemented).

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
| Multi objective, unconstrained | 25 (SCH1, SCH2, FON, KUR, POL, VNT1-3, ZDT5, DTLZ5-7, scaled DTLZ1/2, convex DTLZ2, inverted DTLZ1, WFG1-9) | MaF1-15, UF1-10, Deb's 1999 problems | ZDT5, DTLZ5-7, WFG (batch 4), scaled/convex/inverted DTLZ (batch 8) | Kursawe, Poloni, Viennet, SCH2's formula; FON secondary |
| Multi objective, constrained | 45 (CONSTR, SRN, TNK, BNH, OSY, CTP1-8, C-DTLZ ×6, MW1-14, DAS-CMOP1-9, DTLZ8-9) | LIR-CMOP1-14, CF1-10, DC-DTLZ ×6, Viennet 4 / MOP-C | CONSTR, SRN, TNK (as NSGA-II restates them), CTP1-7 (the KanGAL report), C-DTLZ, MW, DAS-CMOP, DTLZ8-9, and the optional LIR-CMOP, CF, DC-DTLZ | CTP8 (from the authors' code, the book unread), BNH, OSY; DC2/DC3 parameters |
| Engineering design | 8 single-objective (welded beam in two versions, pressure vessel, spring, speed reducer, gear train, three-bar truss, cantilever beam, car side impact) and 11 multi-objective (two-bar truss, welded beam, disc brake, car side impact, speed reducer, gear train, four-bar truss, rocket injector, water resource planning, marine design, crashworthiness) | I-beam, tubular column, concrete beam, stepped cantilever, coil spring, hatch cover, clutch brake, cantilever (Deb) | pressure vessel and cantilever optima (Yang et al. 2013, proofs) | the originals of all the others: secondary (Tanabe and Ishibuchi 2020, Chehouri et al. 2016) or unverified |

**Core total: 172 new problems**, on top of the 9 genoxide has (ZDT1-4, ZDT6, DTLZ1-4), and
about 75 optional ones.

**To check in the originals before implementing** (by library access, since they're paywalled or
in print only): Deb's 2001 book (OSY's regions, CTP8), Binh and Korn (1997), Kursawe (1991),
Poloni et al. (2000), Viennet et al. (1996), Schwefel (1981), Dixon and Szegö
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
plus a row in `examples/README.md`. Every problem of the batch has an example of its own, one
problem per example, with its `output.txt` and the `trace.json` that the project page plays
(https://tachsin.gr/projects/genoxide/examples/), with a plot that shows the solutions; a batch
isn't done until then. One problem per example: examples that only compare several problems or
algorithms aren't added (the function suite was removed once each of its functions had its own
page); a comparison belongs on the problems' own pages.

| Batch | Status | Contents | Problems | Examples (`examples/<name>/`) |
|---|---|---|---|---|
| 1 | done ([#170](https://github.com/tachsin/genoxide/pull/170)) | The `problems` module (trait, `Optimum`, `Constraints`, registry), `gx.problems` with native evaluation, `gx.indicators`; the classic continuous functions | Sphere, Axis-parallel ellipsoid, Schwefel 1.2, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel (2.26), Levy, Zakharov, Styblinski-Tang, Himmelblau, Michalewicz, Branin, Goldstein-Price, Six-hump camel (16) | `rastrigin` (switch to `problems::Rastrigin`); `function_suite` (removed later: each function has its own page); new `himmelblau`: the four minima found by restarts of a local search (continuous); new `branin`, `goldstein_price` and `six_hump_camel`, the same way; and a page per scalable function: `sphere`, `axis_parallel_ellipsoid`, `schwefel_1_2`, `zakharov`, `rosenbrock`, `levy`, `styblinski_tang`, `michalewicz`, `ackley`, `griewank`, `schwefel_2_26` |
| 2 | done ([#177](https://github.com/tachsin/genoxide/pull/177)) | `MultiProblem` (the breaking change), constraints in multi-objective problems, the pymoo test values replaced (section 5); the classic two- and three-objective problems | Schaffer 1, Schaffer 2, Fonseca-Fleming, Kursawe, Poloni, Viennet 1-3, BNH, SRN, TNK, OSY, CONSTR (13; WATER is batch 9's water resource planning) | `bnh` (NSGA-II on a constrained problem, IGD+ to the analytic front; multi-objective); `kursawe` (disconnected front, SPEA2 vs NSGA-II); new `schaffer1`, `schaffer2`, `fonseca_fleming`, `poloni`, `srn`, `tnk`, `osy` and `constr` (NSGA-II), and `viennet1`, `viennet2` and `viennet3` (NSGA-III) |
| 3 | done ([#191](https://github.com/tachsin/genoxide/pull/191)) | Engineering design, single objective, and the mixed-variable convention (`design()`) | Welded beam, Pressure vessel, Tension/compression spring, Speed reducer, Gear train (integer), Three-bar truss, Cantilever beam, Car side impact (single objective) (8), and CEC 2006 g01-g06 (6) | `pressure_vessel` (switch to `engineering::PressureVessel`); new `welded_beam` (constrained); new `gear_train` (integer genome; category integer); new `tension_compression_spring`, `speed_reducer`, `three_bar_truss`, `cantilever_beam`, `car_side_impact` and `cec2006_g01` to `cec2006_g06` |
| 4 | done (#270) | Scalable many-objective problems | DTLZ5, DTLZ6, DTLZ7, ZDT5 (binary), WFG1-WFG9 (13) | an example per problem: `zdt5`, `dtlz5_3obj`, `dtlz6_3obj`, `dtlz7_3obj`, `wfg1` to `wfg9` (2 objectives); later, `wfg_many_objective`: NSGA-III and MOEA/D on WFG4 and WFG9 with 5 objectives, IGD to the sampled front |
| 5 | done (#277) | CEC 2006, part 2 | g07-g18 (12) | `cec2006_g07` to `cec2006_g18` |
| 6 | done | CEC 2006, part 3, and the low-dimensional classics with tables | g19-g24 (6), Hartmann 3-D, Hartmann 6-D, Shekel 5/7/10, Easom, Eggholder, Schaffer F6 (8) | `cec2006_g19` to `cec2006_g24` and an example per function: `hartmann3`, `hartmann6`, `shekel5`, `shekel7`, `shekel10`, `easom`, `eggholder`, `schaffer_f6` |
| 7 | done | Constrained test problems with tunable difficulty: CTP checked in the authors' KanGAL report 200002 (g, n and bounds from their NSGA-II code, f₂'s square root from its figures and code, CTP8 from the code only), C-DTLZ in the typeset manuscript, with table V's counts as a test (section 1.4) | CTP1-CTP8 (8), C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, convex C2-DTLZ2, C3-DTLZ1, C3-DTLZ4 (6) | an example per problem: `ctp1` to `ctp8` (NSGA-II, the report's settings; CTP3 and CTP5 longer, CTP4 with MOEA/D for 70,000 generations), `c1_dtlz1_3obj`, `c1_dtlz3_3obj` (NSGA-III stops at the barrier in 19 of 20 runs; ignoring the constraint reaches the front), `c2_dtlz2_3obj`, `convex_c2_dtlz2_3obj`, `c3_dtlz1_3obj`, `c3_dtlz4_3obj` (NSGA-III, the paper's settings); the site shades CTP's feasible regions and draws infeasible solutions in 3-D |
| 8 | done | Scaled and inverted DTLZ, and MW | Convex DTLZ2, scaled DTLZ1, scaled DTLZ2, inverted DTLZ1 (4), MW1-MW14 (14) | an example per problem: `convex_dtlz2`, `scaled_dtlz1`, `scaled_dtlz2`, `inverted_dtlz1` (NSGA-III against MOEA/D or other directions) and `mw1` to `mw14` (NSGA-II, NSGA-III and SMS-EMOA, with the paper's settings and settings that reach the fronts); the `mw` comparison of all fourteen is still to come |
| 9 | done, but for the marine design | Engineering design, several objectives (`multi::problems::engineering`, section 1.5's "Checked in batch 9") | Two-bar truss, welded beam (2 objectives), disc brake, car side impact (3 objectives), speed reducer (2 objectives), four-bar truss, water resource planning, rocket injector, vehicle crashworthiness (9); conceptual marine design left out: its original is unread and its restatements disagree | an example per problem: `two_bar_truss`, `welded_beam_2obj`, `disc_brake`, `speed_reducer_2obj`, `four_bar_truss` (NSGA-II), `car_side_impact_3obj` (NSGA-III, Jain and Deb's settings), `rocket_injector`, `vehicle_crashworthiness` (SMS-EMOA), `water_resource_planning` (SPEA2) |
| 10a | done | Remaining low-dimensional and classic scalable functions (section 1.1's "Checked in batch 10a") | Beale, Booth, Matyas, Bohachevsky 1-3, Three-hump camel, Dixon-Price, Trid, Powell, Langermann, Shekel's foxholes, Kowalik, Schwefel 2.21, Schwefel 2.22 (15) | an example per function: `beale`, `booth`, `matyas`, `bohachevsky1` to `bohachevsky3`, `three_hump_camel`, `langermann`, `shekel_foxholes` and `kowalik` (30 seeds each of CMA-ES with and without IPOP, DE, PSO or a GA), `dixon_price` (CMA-ES with IPOP, DE and PSO in 5 and 10 dimensions), and `schwefel_2_21`, `schwefel_2_22`, `trid` and `powell` (CMA-ES, sep-CMA-ES, DE, PSO and a GA to errors of 1 … 1e-8) |
| 10b |  | CEC and BBOB-style functions, and the shift / rotation wrappers | `Shifted<P>`, `Rotated<P>`; Sum of different powers, Step, Quartic (deterministic: without noise, or with noise seeded from the genome, since fitness functions must be deterministic), Penalized 1 and 2, High-conditioned elliptic, Bent cigar, Discus, Büche-Rastrigin, Non-continuous Rastrigin, Weierstrass, Katsuura, HappyCat, HGBat, Schaffer F7, Rotated hyper-ellipsoid, BBOB different powers (17; the shifted and rotated Rastrigin of CEC 2005 and BBOB are the wrappers around `Rastrigin`) | an example per function (e.g. CMA-ES with full and diagonal covariance on a rotated ellipsoid's page) |
| 11 |  | Advanced constrained multi-objective suites | DAS-CMOP1-9 (with the 16 difficulty triplets as a parameter), DC-DTLZ (DC1-DC3 on DTLZ1/DTLZ3), DTLZ8, DTLZ9 (≈15) | an example per problem |
| 12 |  | Binary and combinatorial problems | OneMax, LeadingOnes, deceptive trap, royal road, NK landscapes (seeded), 0/1 knapsack (generated instance classes) (6) | an example per problem; `one_max` and `knapsack` switch to the problems |
| 13 (optional) |  | Competition suites whose definitions are long | LIR-CMOP1-14, CEC 2009 UF1-UF10 and CF1-CF10, MaF1-MaF15, Deb's 1999 two-objective problems, Van Veldhuizen's constrained problems; the deferred engineering problems (section 1.5) once their originals are read | none; used by the benchmark suite |

Order rationale: batch 1 builds the machinery with the functions everyone starts with; batch 2
makes the multi-objective side consistent (and removes the pymoo test values) while the problems
are still small; the engineering problems (batch 3) are what users most often ask for after the
textbook functions; the rest follow by popularity in the constrained and many-objective
literature. Each batch updates AGENTS.md's list of test problems and `docs/features.md`.

## 5. Mentions of pymoo in the code, tests and docs

Done in batch 2 ([#177](https://github.com/tachsin/genoxide/pull/177)): the code, tests and docs
state behavior and cite the papers, not another library, and `values_match_pymoo` became tests
whose expected values are derived from the papers' formulas. The benchmark suite (`benchmarks/`,
`docs/benchmarks/`), the CHANGELOG and the ROADMAP's benchmark lists name other libraries on
purpose.
