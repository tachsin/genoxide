//! The quadratic model of L-BFGS-B: the limited-memory BFGS matrix in compact form, the
//! generalized Cauchy point and the subspace minimization, each O(m · n) for m pairs and n genes.
//!
//! The equations are those of Byrd, Lu, Nocedal and Zhu (1995), cited by number in the comments:
//! B = θI − W M Wᵀ (3.2) with W = [Y θS] (3.3) and M = [−D Lᵀ; L θSᵀS]⁻¹ (3.4)-(3.6); the
//! generalized Cauchy point by Algorithm CP (section 4); the subspace step by the direct primal
//! method (section 5.1, eq. 5.4-5.11), then Morales and Nocedal's (2011) projection.

use crate::linalg::blas::{dot, gemv};
use crate::linalg::cholesky::cholesky;
use crate::linalg::triangular::{cholesky_solve, solve_lower, solve_lower_transposed};

/// The stored correction pairs sᵢ = xᵢ₊₁ − xᵢ and yᵢ = gᵢ₊₁ − gᵢ, oldest first, with their inner
/// products and θ: everything B needs. The pairs are rows of `n` values in a ring of `capacity`
/// rows; the inner products are `capacity × capacity` matrices in the pairs' order (row-major,
/// stride `capacity`).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct Memory {
    n: usize,
    capacity: usize,
    s: Vec<f64>,
    y: Vec<f64>,
    // the row of the oldest pair, and the number of pairs
    head: usize,
    len: usize,
    // (SᵀY)ᵢⱼ = sᵢᵀyⱼ, SᵀS and YᵀY
    sy: Vec<f64>,
    ss: Vec<f64>,
    yy: Vec<f64>,
    // θ = yᵀy / sᵀy of the newest pair, 1 without pairs
    theta: f64,
}

impl Memory {
    /// Room for `capacity` pairs of `n` genes.
    pub(super) fn new(n: usize, capacity: usize) -> Self {
        Memory {
            n,
            capacity,
            s: vec![0.0; capacity * n],
            y: vec![0.0; capacity * n],
            head: 0,
            len: 0,
            sy: vec![0.0; capacity * capacity],
            ss: vec![0.0; capacity * capacity],
            yy: vec![0.0; capacity * capacity],
            theta: 1.0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn theta(&self) -> f64 {
        self.theta
    }

    /// Drops every pair: B = I.
    pub(super) fn clear(&mut self) {
        self.len = 0;
        self.theta = 1.0;
    }

    /// The same pairs (the newest ones, if fewer fit) with room for `capacity`.
    pub(super) fn with_capacity(&self, capacity: usize) -> Memory {
        let mut memory = Memory::new(self.n, capacity);
        let kept = self.len.min(capacity);
        let first = self.len - kept;
        let n = self.n;
        for i in 0..kept {
            let row = self.row(first + i);
            memory.s[i * n..(i + 1) * n].copy_from_slice(&self.s[row * n..(row + 1) * n]);
            memory.y[i * n..(i + 1) * n].copy_from_slice(&self.y[row * n..(row + 1) * n]);
            for j in 0..kept {
                let (from, to) = ((first + i) * self.capacity + first + j, i * capacity + j);
                memory.sy[to] = self.sy[from];
                memory.ss[to] = self.ss[from];
                memory.yy[to] = self.yy[from];
            }
        }
        memory.len = kept;
        memory.theta = if kept > 0 { self.theta } else { 1.0 };
        memory
    }

    // the ring row of pair `i`, 0 the oldest
    fn row(&self, i: usize) -> usize {
        (self.head + i) % self.capacity
    }

    pub(super) fn s(&self, i: usize) -> &[f64] {
        let row = self.row(i);
        &self.s[row * self.n..(row + 1) * self.n]
    }

    pub(super) fn y(&self, i: usize) -> &[f64] {
        let row = self.row(i);
        &self.y[row * self.n..(row + 1) * self.n]
    }

    pub(super) fn sy(&self, i: usize, j: usize) -> f64 {
        self.sy[i * self.capacity + j]
    }

    fn ss(&self, i: usize, j: usize) -> f64 {
        self.ss[i * self.capacity + j]
    }

    fn yy(&self, i: usize, j: usize) -> f64 {
        self.yy[i * self.capacity + j]
    }

    /// Adds the pair s = `x_new` − `x_old`, y = `g_new` − `g_old`, dropping the oldest one if the
    /// memory is full, unless sᵀy ≤ `epsilon` ‖y‖² (Byrd et al., eq. 3.9): then the pair is
    /// skipped, the oldest one stays, and it returns false.
    pub(super) fn update(
        &mut self,
        x_new: &[f64],
        x_old: &[f64],
        g_new: &[f64],
        g_old: &[f64],
        epsilon: f64,
    ) -> bool {
        // sᵀy and yᵀy first, from the vectors: a skipped pair overwrites nothing
        let (mut sty, mut yty) = (0.0, 0.0);
        for i in 0..self.n {
            let (s, y) = (x_new[i] - x_old[i], g_new[i] - g_old[i]);
            sty += s * y;
            yty += y * y;
        }
        if sty <= epsilon * yty || !sty.is_finite() || !yty.is_finite() {
            return false;
        }
        let capacity = self.capacity;
        if self.len == capacity {
            // the oldest pair's row takes the new one: the matrices move up and left
            self.head = (self.head + 1) % capacity;
            self.len -= 1;
            for matrix in [&mut self.sy, &mut self.ss, &mut self.yy] {
                for i in 0..self.len {
                    for j in 0..self.len {
                        matrix[i * capacity + j] = matrix[(i + 1) * capacity + j + 1];
                    }
                }
            }
        }
        let new = self.len;
        let row = self.row(new);
        let n = self.n;
        for (i, (s, y)) in self.s[row * n..(row + 1) * n]
            .iter_mut()
            .zip(&mut self.y[row * n..(row + 1) * n])
            .enumerate()
        {
            *s = x_new[i] - x_old[i];
            *y = g_new[i] - g_old[i];
        }
        self.len += 1;
        // the new pair's products with every pair, itself included: rows of S and Y times the
        // new s and y, four rows at a time
        let mut stack = [[0.0; MAX_STACK]; 4];
        let mut heap;
        let products: [&mut [f64]; 4] = if self.len <= MAX_STACK {
            let [a, b, c, d] = &mut stack;
            [
                &mut a[..self.len],
                &mut b[..self.len],
                &mut c[..self.len],
                &mut d[..self.len],
            ]
        } else {
            heap = vec![0.0; 4 * self.len];
            let (a, rest) = heap.split_at_mut(self.len);
            let (b, rest) = rest.split_at_mut(self.len);
            let (c, d) = rest.split_at_mut(self.len);
            [a, b, c, d]
        };
        let [s_y, y_s, s_s, y_y] = products;
        let (new_s, new_y) = (
            &self.s[row * n..(row + 1) * n],
            &self.y[row * n..(row + 1) * n],
        );
        self.products(&self.s, new_y, s_y);
        self.products(&self.y, new_s, y_s);
        self.products(&self.s, new_s, s_s);
        self.products(&self.y, new_y, y_y);
        for i in 0..self.len {
            self.sy[i * capacity + new] = s_y[i];
            self.sy[new * capacity + i] = y_s[i];
            self.ss[i * capacity + new] = s_s[i];
            self.ss[new * capacity + i] = s_s[i];
            self.yy[i * capacity + new] = y_y[i];
            self.yy[new * capacity + i] = y_y[i];
        }
        self.theta = yty / sty;
        true
    }

    // `out[i] = vᵢ · x` for the stored rows `v` (S or Y) in the pairs' order, by `gemv` on the
    // ring's contiguous runs: each product in the order of the genes, as `dot`
    fn products(&self, rows: &[f64], x: &[f64], out: &mut [f64]) {
        let n = self.n;
        let first = (self.capacity - self.head).min(self.len);
        gemv(
            first,
            n,
            1.0,
            &rows[self.head * n..(self.head + first) * n],
            x,
            0.0,
            &mut out[..first],
        );
        let rest = self.len - first;
        if rest > 0 {
            gemv(
                rest,
                n,
                1.0,
                &rows[..rest * n],
                x,
                0.0,
                &mut out[first..self.len],
            );
        }
    }

    /// `out = Wᵀ v` = [Yᵀv; θ Sᵀv], 2 · len values.
    pub(super) fn w_transpose(&self, v: &[f64], out: &mut [f64]) {
        let len = self.len;
        self.products(&self.y, v, &mut out[..len]);
        self.products(&self.s, v, &mut out[len..2 * len]);
        for value in &mut out[len..2 * len] {
            *value *= self.theta;
        }
    }

    /// The row of W for gene `gene`: wᵢ = [y₀ᵢ … θ s₀ᵢ …].
    fn w_row(&self, gene: usize, out: &mut [f64]) {
        let len = self.len;
        for j in 0..len {
            let row = self.row(j) * self.n + gene;
            out[j] = self.y[row];
            out[len + j] = self.theta * self.s[row];
        }
    }

    /// `(W v)` at `gene`: Σⱼ yⱼ vⱼ + θ sⱼ v_{len+j}, the pairs in order.
    #[cfg(test)]
    fn w_times_at(&self, gene: usize, v: &[f64]) -> f64 {
        let len = self.len;
        let mut sum = 0.0;
        for j in 0..len {
            let row = self.row(j) * self.n + gene;
            sum += self.y[row] * v[j];
            sum += self.theta * self.s[row] * v[len + j];
        }
        sum
    }

    /// `out[k] += α (W v)` at gene `genes[k]` (or at gene k, for all genes): the terms of each
    /// element added as [`w_times_at`](Memory::w_times_at) sums them, the pairs in order, but
    /// row by row, along the stored rows.
    fn w_times_add(&self, genes: Option<&[usize]>, alpha: f64, v: &[f64], out: &mut [f64]) {
        let len = self.len;
        let n = self.n;
        for j in 0..len {
            let row = self.row(j);
            let (y, s) = (
                &self.y[row * n..(row + 1) * n],
                &self.s[row * n..(row + 1) * n],
            );
            let (a, b) = (alpha * v[j], alpha * self.theta * v[len + j]);
            match genes {
                None => {
                    for ((out, &y), &s) in out.iter_mut().zip(y).zip(s) {
                        let term = y * a;
                        *out += term;
                        *out += s * b;
                    }
                }
                Some(genes) => {
                    for (out, &i) in out.iter_mut().zip(genes) {
                        *out += y[i] * a;
                        *out += s[i] * b;
                    }
                }
            }
        }
    }
}

// the pairs whose products fit on the stack in `Memory::update`; more use the heap
const MAX_STACK: usize = 64;

/// A failed factorization: the memory's matrices are singular or indefinite from rounding, and
/// the caller drops the pairs (Zhu et al., 1997, section 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Singular;

/// The scratch buffers of a step, reused from one step to the next. Not part of a checkpoint:
/// nothing in them outlives a step.
#[derive(Clone, Debug, Default)]
pub(super) struct Workspace {
    // the Cholesky factor of T = θSᵀS + L D⁻¹ Lᵀ, len × len (stride len)
    middle: Vec<f64>,
    // 2 · len vectors
    p: Vec<f64>,
    c: Vec<f64>,
    v: Vec<f64>,
    wb: Vec<f64>,
    u: Vec<f64>,
    t1: Vec<f64>,
    // len × len matrices: YᵀZZᵀY, SᵀZZᵀY and SᵀAAᵀS (or their complements), P, C, Q
    yzy: Vec<f64>,
    szy: Vec<f64>,
    sas: Vec<f64>,
    pm: Vec<f64>,
    cm: Vec<f64>,
    qm: Vec<f64>,
    x_mat: Vec<f64>,
    // the breakpoints (t, gene) of the Cauchy point, a heap once more than one is passed
    breakpoints: Vec<(f64, usize)>,
    // the search direction of the Cauchy point, n values
    d: Vec<f64>,
    // which genes are held at a bound at the Cauchy point
    active: Vec<bool>,
    // the free genes at the Cauchy point, and the others that can move
    free: Vec<usize>,
    bound: Vec<usize>,
    // the reduced gradient and the subspace step, one per free gene
    r: Vec<f64>,
    du: Vec<f64>,
}

fn resized(buffer: &mut Vec<f64>, len: usize) -> &mut [f64] {
    buffer.clear();
    buffer.resize(len, 0.0);
    buffer
}

impl Workspace {
    /// Reserves every buffer for `n` genes and `capacity` pairs, so that no step after the first
    /// allocates, whatever the number of pairs stored.
    pub(super) fn reserve(&mut self, n: usize, capacity: usize) {
        let (pairs, square) = (2 * capacity, capacity * capacity);
        for (buffer, len) in [
            (&mut self.middle, square),
            (&mut self.p, pairs),
            (&mut self.c, pairs),
            (&mut self.v, pairs),
            (&mut self.wb, pairs),
            (&mut self.u, pairs),
            (&mut self.t1, pairs),
            (&mut self.yzy, square),
            (&mut self.szy, square),
            (&mut self.sas, square),
            (&mut self.pm, square),
            (&mut self.cm, square),
            (&mut self.qm, square),
            (&mut self.x_mat, square),
            (&mut self.d, n),
            (&mut self.r, n),
            (&mut self.du, n),
        ] {
            buffer.reserve(len.saturating_sub(buffer.len()));
        }
        self.breakpoints
            .reserve(n.saturating_sub(self.breakpoints.len()));
        self.active.reserve(n.saturating_sub(self.active.len()));
        self.free.reserve(n.saturating_sub(self.free.len()));
        self.bound.reserve(n.saturating_sub(self.bound.len()));
    }

    /// Factors T = θSᵀS + L D⁻¹ Lᵀ for products with M: [−D Lᵀ; L θSᵀS]⁻¹ eliminated by its
    /// block −D, whose Schur complement is T.
    pub(super) fn factor_middle(&mut self, memory: &Memory) -> Result<(), Singular> {
        let len = memory.len();
        let theta = memory.theta();
        let t = resized(&mut self.middle, len * len);
        for i in 0..len {
            for j in 0..=i {
                // L D⁻¹ Lᵀ: Lᵢₖ = sᵢᵀyₖ for k < i, and k < j ≤ i
                let mut sum = theta * memory.ss(i, j);
                for k in 0..j {
                    sum += memory.sy(i, k) * memory.sy(j, k) / memory.sy(k, k);
                }
                t[i * len + j] = sum;
            }
        }
        cholesky(t, len).map_err(|_| Singular)
    }

    /// `out = M v` for the 2 · len values `v` = [p; q]: b = T⁻¹(q + L D⁻¹ p), a = D⁻¹(Lᵀb − p),
    /// M v = [a; b].
    fn apply_middle(memory: &Memory, middle: &[f64], v: &[f64], out: &mut [f64]) {
        let len = memory.len();
        let (a, b) = out[..2 * len].split_at_mut(len);
        let (p, q) = v[..2 * len].split_at(len);
        for (i, (b, &q)) in b.iter_mut().zip(q).enumerate() {
            let mut sum = q;
            for (k, &p) in p[..i].iter().enumerate() {
                sum += memory.sy(i, k) * (p / memory.sy(k, k));
            }
            *b = sum;
        }
        cholesky_solve(middle, len, b);
        for (k, (a, &p)) in a.iter_mut().zip(p).enumerate() {
            let mut sum = 0.0;
            for (i, &b) in b.iter().enumerate().skip(k + 1) {
                sum += memory.sy(i, k) * b;
            }
            *a = (sum - p) / memory.sy(k, k);
        }
    }

    /// The generalized Cauchy point `xc` from `x` with gradient `g` in the box [`lower`,
    /// `upper`], by Algorithm CP of Byrd et al. (section 4), with the middle matrix factored.
    /// Afterwards `self.c` = Wᵀ(xc − x) (eq. 4.13) and `self.active` marks the genes held at a
    /// bound: those whose breakpoint the path passed or was at (tᵢ = 0), and the `fixed` ones.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cauchy_point(
        &mut self,
        memory: &Memory,
        x: &[f64],
        g: &[f64],
        lower: &[f64],
        upper: &[f64],
        fixed: &[bool],
        xc: &mut [f64],
    ) {
        let n = x.len();
        let len = memory.len();
        let theta = memory.theta();
        let d = resized(&mut self.d, n);
        self.active.clear();
        self.active.extend_from_slice(fixed);
        self.breakpoints.clear();
        // the breakpoints tᵢ (eq. 4.1) and dᵢ = −gᵢ, 0 where tᵢ = 0
        let mut moving = 0usize;
        let mut first: Option<usize> = None;
        for i in 0..n {
            if fixed[i] || g[i] == 0.0 {
                continue;
            }
            let t = if g[i] < 0.0 {
                (x[i] - upper[i]) / g[i]
            } else {
                (x[i] - lower[i]) / g[i]
            };
            if t <= 0.0 {
                self.active[i] = true;
                continue;
            }
            d[i] = -g[i];
            moving += 1;
            if t.is_finite() {
                let point = (t, i);
                if first.is_none_or(|k| earlier(point, self.breakpoints[k])) {
                    first = Some(self.breakpoints.len());
                }
                self.breakpoints.push(point);
            }
        }
        xc.copy_from_slice(x);
        let c = resized(&mut self.c, 2 * len);
        if moving == 0 {
            return;
        }
        let p = resized(&mut self.p, 2 * len);
        memory.w_transpose(d, p);
        // f′ = gᵀd = −dᵀd, f″ = θdᵀd − pᵀMp
        let dtd = dot(d, d);
        let mut f1 = -dtd;
        let v = resized(&mut self.v, 2 * len);
        let mut f2 = theta * dtd;
        if len > 0 {
            Self::apply_middle(memory, &self.middle, p, v);
            f2 -= dot(p, v);
        }
        let f2_start = f2;
        let mut dt_min = -f1 / f2;
        let mut t_old = 0.0;
        let wb = resized(&mut self.wb, 2 * len);
        let mut heap_built = false;
        loop {
            // the next breakpoint: the earliest one first, then from a heap of the others
            let next = if !heap_built {
                let Some(k) = first else { break };
                heap_built = true;
                let point = self.breakpoints.swap_remove(k);
                build_heap(&mut self.breakpoints);
                point
            } else {
                match pop_heap(&mut self.breakpoints) {
                    Some(point) => point,
                    None => break,
                }
            };
            let (t, b) = next;
            let dt = t - t_old;
            if dt_min < dt {
                break;
            }
            // the path reaches gene b's bound: it stops moving
            let bound = if d[b] > 0.0 { upper[b] } else { lower[b] };
            let z = bound - x[b];
            xc[b] = bound;
            for (c, &p) in c.iter_mut().zip(p.iter()) {
                *c += dt * p;
            }
            let gb = g[b];
            // f′ += Δt f″ + g_b² + θ g_b z_b − g_b w_bᵀMc; f″ −= θ g_b² + 2 g_b w_bᵀMp + g_b² w_bᵀMw_b
            f1 += dt * f2 + gb * gb + theta * gb * z;
            f2 -= theta * gb * gb;
            if len > 0 {
                memory.w_row(b, wb);
                Self::apply_middle(memory, &self.middle, wb, v);
                let (wmc, wmp, wmw) = (dot(v, c), dot(v, p), dot(v, wb));
                f1 -= gb * wmc;
                f2 -= 2.0 * gb * wmp + gb * gb * wmw;
                for (p, &w) in p.iter_mut().zip(wb.iter()) {
                    *p += gb * w;
                }
            }
            d[b] = 0.0;
            self.active[b] = true;
            moving -= 1;
            // the authors' implementation keeps f″ above ε of its start against rounding
            f2 = f2.max(f64::EPSILON * f2_start);
            t_old = t;
            if moving == 0 {
                dt_min = 0.0;
                break;
            }
            dt_min = -f1 / f2;
        }
        let dt_min = if dt_min > 0.0 { dt_min } else { 0.0 };
        let t = t_old + dt_min;
        for i in 0..n {
            if d[i] != 0.0 {
                xc[i] = (x[i] + t * d[i]).clamp(lower[i], upper[i]);
            }
        }
        for (c, &p) in c.iter_mut().zip(p.iter()) {
            *c += dt_min * p;
        }
    }

    /// The subspace step from the Cauchy point (after [`cauchy_point`](Self::cauchy_point)):
    /// the minimizer x̂ of the model over the free genes by the direct primal method (Byrd et al.,
    /// section 5.1), projected into the box (Morales and Nocedal, 2011), into `xbar`. If the
    /// projection isn't a descent direction from `x`, the step towards x̂ truncated at the box
    /// instead (eq. 5.8, 5.9).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn subspace_step(
        &mut self,
        memory: &Memory,
        x: &[f64],
        g: &[f64],
        lower: &[f64],
        upper: &[f64],
        fixed: &[bool],
        xc: &[f64],
        xbar: &mut [f64],
    ) -> Result<(), Singular> {
        let n = x.len();
        let len = memory.len();
        let theta = memory.theta();
        xbar.copy_from_slice(xc);
        self.free.clear();
        self.bound.clear();
        for (i, (&active, &fixed)) in self.active.iter().zip(fixed).enumerate() {
            if !active {
                self.free.push(i);
            } else if !fixed {
                self.bound.push(i);
            }
        }
        let free_count = self.free.len();
        if free_count == 0 {
            return Ok(());
        }
        // the reduced gradient r̂ᶜ = Zᵀ(g + θ(xc − x) − W M c) (eq. 5.4)
        let mc = resized(&mut self.v, 2 * len);
        if len > 0 {
            Self::apply_middle(memory, &self.middle, &self.c, mc);
        }
        let r = resized(&mut self.r, free_count);
        for (r, &i) in r.iter_mut().zip(&self.free) {
            *r = g[i] + theta * (xc[i] - x[i]);
        }
        // every gene free (no bound active): the rows whole, without the indices
        let genes = (free_count < n).then_some(&self.free[..]);
        memory.w_times_add(genes, -1.0, mc, r);
        let du = resized(&mut self.du, free_count);
        if len == 0 {
            for (du, &r) in du.iter_mut().zip(r.iter()) {
                *du = -r / theta;
            }
        } else {
            self.solve_reduced(memory)?;
        }
        // x̂ = xc + Z d̂ᵘ, projected into the box
        let du = &self.du;
        let mut clipped = false;
        for (k, &i) in self.free.iter().enumerate() {
            let value = xc[i] + du[k];
            let projected = value.clamp(lower[i], upper[i]);
            clipped |= projected != value;
            xbar[i] = projected;
        }
        if clipped {
            // the projection, if it's a descent direction (the authors' implementation reverts
            // when (x̄ − x)ᵀg > 0)
            let mut slope = 0.0;
            for i in 0..n {
                slope += (xbar[i] - x[i]) * g[i];
            }
            if slope > 0.0 {
                // α* = max {α ≤ 1 : l − xc ≤ α d̂ᵘ ≤ u − xc} (eq. 5.8)
                let mut alpha = 1.0f64;
                for (k, &i) in self.free.iter().enumerate() {
                    let step = du[k];
                    if step > 0.0 {
                        alpha = alpha.min((upper[i] - xc[i]) / step);
                    } else if step < 0.0 {
                        alpha = alpha.min((lower[i] - xc[i]) / step);
                    }
                }
                let alpha = alpha.max(0.0);
                for (k, &i) in self.free.iter().enumerate() {
                    xbar[i] = (xc[i] + alpha * du[k]).clamp(lower[i], upper[i]);
                }
            }
        }
        Ok(())
    }

    // d̂ᵘ = −B̂⁻¹ r̂ᶜ for the reduced matrix B̂ = θI − (ZᵀW) M (WᵀZ), by Sherman-Morrison-Woodbury
    // (eq. 5.10): B̂⁻¹ = I/θ + ZᵀW K⁻¹ WᵀZ / θ² with K = M⁻¹ − WᵀZZᵀW / θ, the 2 · len matrix
    // [−D − YᵀZZᵀY/θ, Lᵀ − YᵀZZᵀS; L − SᵀZZᵀY, θSᵀAAᵀS]. K is solved by eliminating its first
    // block, −P with P = D + YᵀZZᵀY/θ, whose Schur complement Q = θSᵀAAᵀS + C P⁻¹ Cᵀ (C = L −
    // SᵀZZᵀY) is positive definite: two Cholesky factorizations of len × len matrices.
    fn solve_reduced(&mut self, memory: &Memory) -> Result<(), Singular> {
        let len = memory.len();
        let theta = memory.theta();
        // the products over the free genes, or over the bound ones (fewer) subtracted from the
        // full products: O(len² · min(free, bound)) (Byrd et al., end of section 5.1)
        let yzy = resized(&mut self.yzy, len * len);
        let szy = resized(&mut self.szy, len * len);
        let sas = resized(&mut self.sas, len * len);
        let over_bound = self.bound.len() < self.free.len();
        let genes: &[usize] = if over_bound { &self.bound } else { &self.free };
        for i in 0..len {
            let (si, yi) = (memory.s(i), memory.y(i));
            for j in 0..len {
                let (sj, yj) = (memory.s(j), memory.y(j));
                let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
                for &k in genes {
                    if j <= i {
                        a += yi[k] * yj[k];
                        c += si[k] * sj[k];
                    }
                    b += si[k] * yj[k];
                }
                if j <= i {
                    yzy[i * len + j] = a;
                    yzy[j * len + i] = a;
                    sas[i * len + j] = c;
                    sas[j * len + i] = c;
                }
                szy[i * len + j] = b;
            }
        }
        if over_bound {
            // YᵀZZᵀY = YᵀY − YᵀAAᵀY, SᵀZZᵀY = SᵀY − SᵀAAᵀY; SᵀAAᵀS as computed
            for i in 0..len {
                for j in 0..len {
                    yzy[i * len + j] = memory.yy(i, j) - yzy[i * len + j];
                    szy[i * len + j] = memory.sy(i, j) - szy[i * len + j];
                }
            }
        } else {
            // SᵀAAᵀS = SᵀS − SᵀZZᵀS
            for i in 0..len {
                for j in 0..len {
                    sas[i * len + j] = memory.ss(i, j) - sas[i * len + j];
                }
            }
        }
        // P = D + YᵀZZᵀY / θ, factored
        let pm = resized(&mut self.pm, len * len);
        for i in 0..len {
            for j in 0..len {
                pm[i * len + j] = yzy[i * len + j] / theta;
            }
            pm[i * len + i] += memory.sy(i, i);
        }
        cholesky(pm, len).map_err(|_| Singular)?;
        // C = L − SᵀZZᵀY, with L the strict lower triangle of SᵀY
        let cm = resized(&mut self.cm, len * len);
        for i in 0..len {
            for j in 0..len {
                let l = if i > j { memory.sy(i, j) } else { 0.0 };
                cm[i * len + j] = l - szy[i * len + j];
            }
        }
        // X = P⁻¹ Cᵀ, column by column (column j of Cᵀ is row j of C), stored by rows of Xᵀ
        let x_mat = resized(&mut self.x_mat, len * len);
        for j in 0..len {
            let column = &mut x_mat[j * len..(j + 1) * len];
            column.copy_from_slice(&cm[j * len..(j + 1) * len]);
            cholesky_solve(pm, len, column);
        }
        // Q = θ SᵀAAᵀS + C X, factored
        let qm = resized(&mut self.qm, len * len);
        for i in 0..len {
            for j in 0..=i {
                let mut sum = theta * sas[i * len + j];
                // (C X)ᵢⱼ = Σₖ Cᵢₖ Xₖⱼ, with Xₖⱼ = x_mat[j * len + k]
                for k in 0..len {
                    sum += cm[i * len + k] * x_mat[j * len + k];
                }
                qm[i * len + j] = sum;
                qm[j * len + i] = sum;
            }
        }
        cholesky(qm, len).map_err(|_| Singular)?;
        // u = WᵀZ r̂ᶜ: with every gene free, Wᵀ r̂ᶜ by rows (the same sums)
        let u = resized(&mut self.u, 2 * len);
        if self.free.len() == memory.n {
            memory.w_transpose(&self.r, u);
        } else {
            for j in 0..len {
                let (sj, yj) = (memory.s(j), memory.y(j));
                let (mut a, mut b) = (0.0, 0.0);
                for (&i, &r) in self.free.iter().zip(self.r.iter()) {
                    a += yj[i] * r;
                    b += sj[i] * r;
                }
                u[j] = a;
                u[len + j] = theta * b;
            }
        }
        // K [a; b] = [u₁; u₂]: Q b = u₂ + C P⁻¹ u₁, a = P⁻¹(Cᵀ b − u₁)
        let t1 = resized(&mut self.t1, 2 * len);
        let (pu, rest) = t1.split_at_mut(len);
        pu.copy_from_slice(&u[..len]);
        solve_lower(pm, len, pu);
        solve_lower_transposed(pm, len, pu);
        let b = &mut rest[..len];
        for i in 0..len {
            let mut sum = u[len + i];
            for k in 0..len {
                sum += cm[i * len + k] * pu[k];
            }
            b[i] = sum;
        }
        cholesky_solve(qm, len, b);
        let solution = resized(&mut self.wb, 2 * len);
        for k in 0..len {
            let mut sum = -u[k];
            for i in 0..len {
                sum += cm[i * len + k] * b[i];
            }
            solution[k] = sum;
        }
        cholesky_solve(pm, len, &mut solution[..len]);
        solution[len..].copy_from_slice(b);
        // d̂ᵘ = −r̂ᶜ/θ − ZᵀW K⁻¹ u / θ²
        let du = resized(&mut self.du, self.free.len());
        for (du, &r) in du.iter_mut().zip(self.r.iter()) {
            *du = -r / theta;
        }
        let genes = (self.free.len() < memory.n).then_some(&self.free[..]);
        memory.w_times_add(genes, -1.0 / (theta * theta), solution, du);
        Ok(())
    }
}

// whether breakpoint `a` comes before `b`: the smaller t, then the lower gene
fn earlier(a: (f64, usize), b: (f64, usize)) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 < b.1)
}

// a binary min-heap by `earlier`, in place
fn build_heap(heap: &mut [(f64, usize)]) {
    for start in (0..heap.len() / 2).rev() {
        sift_down(heap, start);
    }
}

fn sift_down(heap: &mut [(f64, usize)], mut i: usize) {
    loop {
        let (left, right) = (2 * i + 1, 2 * i + 2);
        let mut least = i;
        if left < heap.len() && earlier(heap[left], heap[least]) {
            least = left;
        }
        if right < heap.len() && earlier(heap[right], heap[least]) {
            least = right;
        }
        if least == i {
            return;
        }
        heap.swap(i, least);
        i = least;
    }
}

fn pop_heap(heap: &mut Vec<(f64, usize)>) -> Option<(f64, usize)> {
    if heap.is_empty() {
        return None;
    }
    let last = heap.len() - 1;
    heap.swap(0, last);
    let top = heap.pop();
    sift_down(heap, 0);
    top
}

#[cfg(test)]
mod tests;
