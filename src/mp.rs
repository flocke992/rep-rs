// IMPROVEMENTS
// [x] Implement auto normalization of columns of the matrix A; A_tilde = W@A, where W is diagonal.
// [ ] Implement other greedy methods (OMP-LS, MP, Weak-MP, Thresholding)
// [ ] Improve performance by managing resources properly
// [ ] Create a trait
// [ ] Provide a config struct in the initialization

use std::collections::HashSet;

use faer::linalg::matmul::dot::inner_prod;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::SolveLstsq;
use faer::reborrow::*;
use faer::{Accum, Col, Mat, Par, unzip, zip};

pub trait Solver {
    // Reserves computational space based on expected number of nonzero coefficients.
    fn reserve_space(&mut self, expected_n_nonzero_coefs: usize);
    // Solve to approx machine precision.
    fn solve(&mut self) -> Option<Col<f64>>;
    // Solve to user-defined tolerance.
    fn solve_residual_tol(&mut self, residual_tol: f64) -> Option<Col<f64>>;
    // Solve with max number of nonzero coefficients.
    fn solve_n_nonzero_coefs(&mut self, n_nonzero_coefs: usize) -> Option<Col<f64>>;
}

fn check_normalized_columns(a_ref: &Mat<f64>, tol: f64) -> bool {
    a_ref
        .col_iter()
        .all(|col_i| (col_i.squared_norm_l2() - 1.0) < tol)
}

pub struct Omp<'a> {
    running_solution: Col<f64>,
    running_residual: Col<f64>,
    support: Vec<usize>,
    error_threshold: f64,
    a_ref: &'a Mat<f64>,
    a_sup: Mat<f64>,
    b_ref: &'a Col<f64>,
}

impl<'a> Omp<'a> {
    pub fn init(a_ref: &'a Mat<f64>, b_ref: &'a Col<f64>) -> Self {
        assert!(
            check_normalized_columns(a_ref, 1.0e-15),
            "Please normalize columns of a"
        );
        let running_solution = Col::zeros(a_ref.ncols());
        let a_sup = Mat::zeros(a_ref.nrows(), a_ref.ncols());
        Omp {
            running_solution: running_solution,
            running_residual: b_ref.cloned(),
            support: Vec::with_capacity(a_ref.nrows()),
            error_threshold: 1.0e-8,
            a_ref,
            a_sup,
            b_ref,
        }
    }

    pub fn solve(&mut self) -> Option<Col<f64>> {
        /*
        Repeat the process until all columns are visited,
        then we can surely produce b, unless b lies in the null-space of A.
        */
        for _i in 0..self.a_ref.ncols() {
            let j_0 = self.sweep();
            self.update_support(j_0);
            self.update_running_solution();
            self.update_residual();
            if self.running_residual.squared_norm_l2() < self.error_threshold {
                return Some(self.running_solution.cloned());
            }
        }
        None
    }

    fn sweep(&mut self) -> usize {
        let mut cand = (0_usize, f64::MAX);
        for i in 0..self.a_ref.ncols() {
            if !self.support.contains(&i) {
                let col_i = self.a_ref.col(i);
                let mut res = self.running_residual.squared_norm_l2();
                res = res
                    - inner_prod(
                        col_i.transpose(),
                        faer::Conj::No,
                        self.running_residual.as_ref(),
                        faer::Conj::No,
                    )
                    .powi(2);
                if res < cand.1 {
                    cand = (i, res);
                }
            }
        }
        cand.0
    }

    fn update_support(&mut self, j_0: usize) {
        self.a_sup
            .col_mut(self.support.len())
            .copy_from(self.a_ref.col(j_0));
        self.support.push(j_0);
    }

    fn update_running_solution(&mut self) {
        /*
        Solve Lstsq using the submatrix of a_sub that we support until now (using length of
        support).
        Even though we temporary allocate here this might not be to bad.
        As we want to find a sparse solution, thus this step should not be called to many times.
        The submatrix only grows by one column for each iteration.
        */
        let qr = self
            .a_sup
            .submatrix(0, 0, self.a_ref.nrows(), self.support.len())
            .col_piv_qr();
        let x = qr.solve_lstsq(self.b_ref);
        for (x_i, &j_i) in x.iter().zip(self.support.iter()) {
            self.running_solution[j_i] = *x_i;
        }
    }

    fn update_residual(&mut self) {
        // Computes `1.0 * A @ running_solution` and stores the result in running_residual
        matmul(
            self.running_residual.rb_mut(),
            Accum::Replace,
            self.a_ref,
            self.running_solution.rb(),
            1.0,
            Par::Seq,
        );
        // computes `running_residual = b - running_residual`
        zip!(&mut self.running_residual, self.b_ref).for_each(|unzip!(r, b)| *r = b - *r);
    }
}

pub fn expand_inverse(space: &mut Mat<f64>, ref_inv: &Mat<f64>) -> Option<()> {
    /*
    We divide the space matrix as follows:
    | b1 b2 |
    | b3 b4 |
    This method calculates:
    | M   b |^-1
    | b^T c |
    and expects a reference to M^-1.
    It expects this allocated space to contain at:
    | -- b |
    | -- c |
    The dimensions are ref_inf kxk, b kx1, c 1x1
    so space has dims (k+1)x(k+1).
    */
    let k = ref_inv.nrows();

    let (mut b1, b2, b3, mut b4) = space.split_at_mut(k, k);
    let (mut b2, mut b3) = (b2.col_mut(0), b3.row_mut(0));
    // First fill b3 with result to reuse
    matmul(
        b3.rb_mut(),
        Accum::Replace,
        b2.rb().transpose(),
        ref_inv,
        1.0,
        Par::Seq,
    );
    let p = 1.0 / (b4[(0, 0)] - inner_prod(b3.rb(), faer::Conj::No, b2.rb(), faer::Conj::No));
    if p.is_infinite() {
        return None;
    }

    // We now have everything we need. Lets start to fill the space.
    // Fill b4
    b4[(0, 0)] = p;

    // Fill b2 & b3
    b2.copy_from(-p * b3.rb().transpose());
    b3.copy_from(b2.rb().transpose());

    // Fill b1
    b1.copy_from(ref_inv);
    matmul(b1.rb_mut(), Accum::Add, b2.rb(), b3.rb(), 1.0 / p, Par::Seq);
    Some(())
}

fn lstsq_additional_column(
    i: usize,
    a_ref: &Mat<f64>,
    b_ref: &Col<f64>,
    stored_inverse: &Mat<f64>,
    inverse_space: &mut Mat<f64>,
    as_space: &mut Mat<f64>,
    sweep_x_opt: &mut Col<f64>,
    sweep_residual: &mut Col<f64>,
    as_b: &mut Col<f64>,
) -> Option<()> {
    /*
     * inverse_space is free space to write the new inverse to
     * as_space is divided as
     * | as ¦ |, where we can use the last column ¦ to fill it with ai and the rest is
     * filled with the matrix as, i.e. a_sup
     */
    let k = stored_inverse.ncols();
    let (a_s, mut ai) = as_space.split_at_col_mut(k);
    ai.copy_from(a_ref.col(i).as_mat());

    // adapt as_b for new ai
    as_b[k] = inner_prod(
        ai.rb().col(0).transpose(),
        faer::Conj::No,
        b_ref.rb(),
        faer::Conj::No,
    );

    // Prepare to calculate inverse into inverse_space
    let (_, mut b2) = inverse_space.split_at_col_mut(k);
    let (mut b2, mut b4) = b2.rb_mut().split_at_row_mut(k);
    matmul(
        b2.rb_mut(),
        Accum::Replace,
        a_s.transpose(),
        ai.rb(),
        1.0,
        Par::Seq,
    );
    b4[(0, 0)] = 1.0;
    // Fill inverse_space if the bigger matrix is invertible, else return None
    expand_inverse(inverse_space, stored_inverse)?;

    //Now we want to calculate the lstsq solution, the residual and, then store/return the
    //necessary variables

    // println!("A support: {as_space:.2?}");
    // println!("b: {b_ref:.2?}");
    matmul(
        sweep_x_opt.rb_mut(),
        Accum::Replace,
        inverse_space.rb(),
        as_b,
        1.0,
        Par::Seq,
    );
    // println!("calculated x opt: {sweep_x_opt:.2?}");
    sweep_residual.copy_from(b_ref);
    matmul(
        sweep_residual.rb_mut(),
        Accum::Add,
        as_space.rb(),
        sweep_x_opt.rb(),
        -1.0,
        Par::Seq,
    );
    Some(())
}

pub struct LsOmp<'a> {
    residual_error: f64,
    support: Vec<usize>,
    error_threshold: f64,
    a_ref: &'a Mat<f64>,
    b_ref: &'a Col<f64>,
    stored_inverse: Mat<f64>,
    as_space: Mat<f64>,
    inverse_space: Mat<f64>,
    sweep_x_opt: Col<f64>,
    sweep_residual: Col<f64>,
    as_b: Col<f64>,
    // Candidate is organized as:
    // (index, residual error, candidate inverse, candidate solution)
    candidate: (usize, f64, Mat<f64>, Col<f64>),
}

impl<'a> LsOmp<'a> {
    pub fn init(a_ref: &'a Mat<f64>, b_ref: &'a Col<f64>) -> Self {
        assert!(
            check_normalized_columns(a_ref, 1.0e-15),
            "Please normalize columns of a"
        );
        // arbitrary number
        let error_threshold = 1.0e-8;
        let nrows = b_ref.nrows();
        let expected_size = nrows / 5;

        let residual_error = b_ref.squared_norm_l2();
        let support = Vec::new();
        let stored_inverse = Mat::with_capacity(b_ref.nrows() / 5, b_ref.nrows() / 5);

        let mut sweep_x_opt = Col::zeros(0);
        sweep_x_opt.reserve(expected_size);
        let sweep_residual = Col::zeros(nrows);
        let mut as_b = Col::zeros(0);
        as_b.reserve(expected_size);

        let mut cand_col = Col::zeros(0);
        cand_col.reserve(expected_size);
        let candidate = (
            0,
            f64::MAX,
            Mat::with_capacity(expected_size, expected_size),
            cand_col,
        );

        LsOmp {
            residual_error: residual_error,
            support: support,
            error_threshold: error_threshold,
            a_ref: a_ref,
            b_ref: b_ref,
            stored_inverse: stored_inverse,
            as_space: Mat::with_capacity(nrows, expected_size),
            inverse_space: Mat::with_capacity(expected_size, expected_size),
            sweep_x_opt: sweep_x_opt,
            sweep_residual: sweep_residual,
            as_b: as_b,
            candidate: candidate,
        }
    }

    fn increase_size(&mut self, new_size: usize) {
        self.as_b.resize_with(new_size, |_| 0.0);
        self.sweep_x_opt.resize_with(new_size, |_| 0.0);
        self.inverse_space
            .resize_with(new_size, new_size, |_, _| 0.0);
        self.as_space
            .resize_with(self.sweep_residual.nrows(), new_size, |_, _| 0.0);
        self.candidate.2.resize_with(new_size, new_size, |_, _| 0.0);
        self.candidate.3.resize_with(new_size, |_| 0.0);
    }

    fn return_reindexed_solution(&self) -> Col<f64> {
        let mut x = Col::zeros(self.a_ref.ncols());
        for (x_i, &j_i) in self.candidate.3.iter().zip(self.support.iter()) {
            x[j_i] = *x_i;
        }
        x
    }

    pub fn solve(&mut self) -> Option<Col<f64>> {
        self.increase_size(1);
        self.scalar_sweep()?;
        if self.residual_error <= self.error_threshold {
            return Some(self.return_reindexed_solution());
        }
        for i in 1..self.a_ref.ncols() {
            self.increase_size(i + 1);
            self.sweep_and_update_running_solution()?;
            if self.residual_error <= self.error_threshold {
                return Some(self.return_reindexed_solution());
            }
        }
        None
    }

    fn scalar_sweep(&mut self) -> Option<f64> {
        self.candidate.1 = f64::MAX;
        let mut ai = self.as_space.col_mut(0);
        let ai_b = self.as_b.get_mut(0);
        let inv = self.inverse_space.get_mut(0, 0);
        let z = self.sweep_x_opt.get_mut(0);
        for i in 0..self.a_ref.ncols() {
            ai.copy_from(self.a_ref.col(i));
            *ai_b = inner_prod(
                ai.rb().transpose(),
                faer::Conj::No,
                self.b_ref.rb(),
                faer::Conj::No,
            );
            *inv = 1.0 / ai.squared_norm_l2();
            *z = *ai_b * *inv;
            let residual_error = (self.b_ref - *z * ai.rb()).squared_norm_l2();
            if residual_error < self.candidate.1 {
                self.candidate.0 = i;
                self.candidate.1 = residual_error;
                self.candidate.2[(0, 0)] = *inv;
                self.candidate.3[0] = *z;
            }
        }
        if self.candidate.1 == f64::MAX {
            return None;
        } else {
            self.support.push(self.candidate.0);
            self.residual_error = self.candidate.1;
            self.stored_inverse
                .resize_with(1, 1, |_, _| self.candidate.2[(0, 0)]);
            ai.copy_from(self.a_ref.col(self.candidate.0));
            *ai_b = inner_prod(
                ai.rb().transpose(),
                faer::Conj::No,
                self.b_ref.rb(),
                faer::Conj::No,
            );
            // println!("residual error after scalar sweep: {}", cand.1);
            return Some(self.candidate.1);
        }
    }

    fn sweep_and_update_running_solution(&mut self) -> Option<f64> {
        let k = self.support.len();

        self.candidate.1 = f64::MAX;

        for i in (0..self.a_ref.ncols()).filter(|j| !self.support.contains(j)) {
            match lstsq_additional_column(
                i,
                &self.a_ref,
                &self.b_ref,
                &self.stored_inverse,
                &mut self.inverse_space,
                &mut self.as_space,
                &mut self.sweep_x_opt,
                &mut self.sweep_residual,
                &mut self.as_b,
            ) {
                Some(()) => {
                    let residual_error = self.sweep_residual.squared_norm_l2();

                    if residual_error < self.candidate.1 {
                        self.candidate.0 = i;
                        self.candidate.1 = residual_error;
                        self.candidate.2.copy_from(self.inverse_space.rb());
                        self.candidate.3.copy_from(self.sweep_x_opt.rb());
                    }
                }
                None => continue,
            }
        }

        if self.candidate.1 == f64::MAX {
            return None;
        } else {
            self.support.push(self.candidate.0);
            self.residual_error = self.candidate.1;
            self.stored_inverse.resize_with(
                self.candidate.2.nrows(),
                self.candidate.2.ncols(),
                |_, _| 0.0,
            );
            self.stored_inverse.copy_from(self.candidate.2.rb());
            self.as_space
                .col_mut(k)
                .copy_from(self.a_ref.col(self.candidate.0));
            matmul(
                self.as_b.rb_mut(),
                Accum::Replace,
                self.as_space.transpose(),
                self.b_ref,
                1.0,
                Par::Seq,
            );
            return Some(self.candidate.1);
        }
    }
}

pub struct Mp<'a> {
    running_solution: Col<f64>,
    running_residual: Col<f64>,
    support: HashSet<usize>,
    error_threshold: f64,
    a_ref: &'a Mat<f64>,
    b_ref: &'a Col<f64>,
}

impl<'a> Mp<'a> {
    pub fn init(a_ref: &'a Mat<f64>, b_ref: &'a Col<f64>) -> Self {
        assert!(
            check_normalized_columns(a_ref, 1.0e-15),
            "Please normalize columns of a"
        );
        let running_solution = Col::zeros(a_ref.ncols());
        Mp {
            running_solution: running_solution,
            running_residual: b_ref.cloned(),
            support: HashSet::with_capacity(a_ref.nrows()),
            error_threshold: 1.0e-8,
            a_ref,
            b_ref,
        }
    }

    fn sweep(&mut self) -> (usize, f64) {
        let mut cand = (0_usize, f64::MAX, 0.0);
        for i in 0..self.a_ref.ncols() {
            let col_i = self.a_ref.col(i);
            let z_i_opt = inner_prod(
                col_i.transpose(),
                faer::Conj::No,
                self.running_residual.as_ref(),
                faer::Conj::No,
            );

            let mut res = self.running_residual.squared_norm_l2();
            res = res - z_i_opt.powi(2);
            if res < cand.1 {
                cand = (i, res, z_i_opt);
            }
        }
        (cand.0, cand.2)
    }

    fn update_support(&mut self, j_0: usize) {
        self.support.insert(j_0);
    }

    fn update_running_solution(&mut self, j_0: usize, z_j_0: f64) {
        self.running_solution[j_0] += z_j_0;
    }

    fn update_residual(&mut self, j_0: usize, z_j_0: f64) {
        zip!(&mut self.running_residual, self.a_ref.col(j_0))
            .for_each(|unzip!(r, c)| *r = *r - z_j_0 * *c);
    }

    pub fn solve(&mut self) -> Option<Col<f64>> {
        /*
        Repeat the process until all columns are visited,
        then we can surely produce b, unless b lies in the null-space of A.
        */
        for _i in 0..self.a_ref.ncols() * 2 {
            let (j_0, z_j_0) = self.sweep();
            self.update_support(j_0);
            self.update_running_solution(j_0, z_j_0);
            self.update_residual(j_0, z_j_0);
            if self.running_residual.squared_norm_l2() < self.error_threshold {
                return Some(self.running_solution.cloned());
            }
        }
        None
    }
}
