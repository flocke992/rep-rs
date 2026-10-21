// IMPROVEMENTS
// [x] Implement auto normalization of columns of the matrix A; A_tilde = W@A, where W is diagonal.
// [x] Implement other greedy methods (OMP-LS, MP, Weak-MP)
// [ ] Implement other greedy method (Thresholding)
// [x] Improve performance by managing resources properly of OMP
// [x] Create a trait
// [x] Provide a config struct in the initialization
// [x] Have unified error bounds
// [x] Comment code
// [ ] Apply Logan Smith "perfect function" video to coding structure

use std::collections::HashSet;

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::matmul::{dot::inner_prod, matmul};
use faer::linalg::qr::no_pivoting::factor;
use faer::linalg::svd::{
    ComputeSvdVectors, SvdError, pseudoinverse_from_svd, pseudoinverse_from_svd_scratch, svd,
    svd_scratch,
};
use faer::linalg::{householder, triangular_solve};
use faer::prelude::default;
use faer::{Accum, Col, Mat, Par, unzip, zip};
use faer::{MatMut, MatRef, reborrow::*};

pub trait Solver<'a> {
    // Solve to precision defined by config.
    fn solve(&mut self) -> Option<Col<f64>>;
    // Solve with max number of nonzero coefficients.
    fn solve_n_nonzero_coefs(&mut self, support_cardinality: usize) -> Option<Col<f64>>;
    fn name(&self) -> String;
}

pub trait SolverConfig<'a> {
    fn a_ref(&self) -> &'a Mat<f64>;
    fn b_ref(&self) -> &'a Col<f64>;
}

pub trait Initializable<'a, C: SolverConfig<'a>>: Solver<'a> + Sized {
    fn init(config: &'a C) -> Self;
}

pub struct BaseConfig<'a> {
    a_ref: &'a Mat<f64>,
    b_ref: &'a Col<f64>,
}

impl<'a> SolverConfig<'a> for BaseConfig<'a> {
    fn a_ref(&self) -> &'a Mat<f64> {
        self.a_ref
    }
    fn b_ref(&self) -> &'a Col<f64> {
        self.b_ref
    }
}

impl<'a> BaseConfig<'a> {
    pub fn init(a_ref: &'a Mat<f64>, b_ref: &'a Col<f64>) -> Self {
        assert!(
            check_normalized_columns(a_ref, 1.0e-15),
            "Please normalize columns of a"
        );
        BaseConfig { a_ref, b_ref }
    }
}

pub struct MpConfig<'a> {
    base: BaseConfig<'a>,
    // Threshold to declare that the algorithm is done.
    sq_error_threshold: f64,
    // Expected cardinality of solution. Used for pre-allocations.
    expected_size: usize,
}

impl<'a> SolverConfig<'a> for MpConfig<'a> {
    fn a_ref(&self) -> &'a Mat<f64> {
        self.base.a_ref
    }
    fn b_ref(&self) -> &'a Col<f64> {
        self.base.b_ref
    }
}

impl<'a> MpConfig<'a> {
    pub fn init(
        a_ref: &'a Mat<f64>,
        b_ref: &'a Col<f64>,
        error_treshold: f64,
        expected_size: usize,
    ) -> Self {
        let base = BaseConfig::init(a_ref, b_ref);
        MpConfig {
            base,
            sq_error_threshold: error_treshold.powi(2),
            expected_size,
        }
    }
}

pub struct WeakMpConfig<'a> {
    base: BaseConfig<'a>,
    // Threshold to declare that the algorithm is done.
    sq_error_threshold: f64,
    // Expected cardinality of solution. Used for pre-allocations.
    expected_size: usize,
    weaker_factor: f64,
}

impl<'a> SolverConfig<'a> for WeakMpConfig<'a> {
    fn a_ref(&self) -> &'a Mat<f64> {
        self.base.a_ref
    }
    fn b_ref(&self) -> &'a Col<f64> {
        self.base.b_ref
    }
}

impl<'a> WeakMpConfig<'a> {
    pub fn init(
        a_ref: &'a Mat<f64>,
        b_ref: &'a Col<f64>,
        error_treshold: f64,
        expected_size: usize,
        weaker_factor: f64,
    ) -> Self {
        assert!(
            0.0 < weaker_factor && 1.0 > weaker_factor,
            "weaker factor must be an element of (0,1)"
        );
        let base = BaseConfig::init(a_ref, b_ref);
        WeakMpConfig {
            base,
            sq_error_threshold: error_treshold.powi(2),
            expected_size,
            weaker_factor,
        }
    }
}

pub struct BpConfig<'a> {
    base: BaseConfig<'a>,
    stop_threshold_sq: f64,
    prune_threshold: f64,
    max_iter: usize,
}

impl<'a> BpConfig<'a> {
    pub fn init(
        a_ref: &'a Mat<f64>,
        b_ref: &'a Col<f64>,
        stop_threshold: f64,
        prune_threshold: f64,
        max_iter: usize,
    ) -> Self {
        let base = BaseConfig::init(a_ref, b_ref);
        BpConfig {
            base,
            stop_threshold_sq: stop_threshold.powi(2),
            prune_threshold,
            max_iter,
        }
    }
}

impl<'a> SolverConfig<'a> for BpConfig<'a> {
    fn a_ref(&self) -> &'a Mat<f64> {
        self.base.a_ref
    }
    fn b_ref(&self) -> &'a Col<f64> {
        self.base.b_ref
    }
}

pub fn init_solver<'a, S, C>(config: &'a C) -> S
where
    C: SolverConfig<'a>,
    S: Initializable<'a, C>,
{
    S::init(config)
}

fn check_normalized_columns(a_ref: &Mat<f64>, tol: f64) -> bool {
    a_ref
        .col_iter()
        .all(|col_i| (col_i.squared_norm_l2() - 1.0) < tol)
}

fn allocate_buffer(req: StackReq) -> MemBuffer {
    MemBuffer::new(req)
}

fn calculate_h_factor_req(nrows: usize, ncols: usize) -> (usize, usize) {
    // returns (rank, block_size)
    let rank = Ord::min(nrows, ncols);
    let block_size = factor::recommended_block_size::<f64>(nrows, ncols);
    (rank, block_size)
}

fn calculate_stack_req(nrows: usize, ncols: usize) -> StackReq {
    let block_size = factor::recommended_block_size::<f64>(nrows, ncols);
    StackReq::any_of(&[
        factor::qr_in_place_scratch::<f64>(nrows, ncols, block_size, Par::Seq, Default::default()),
        householder::apply_block_householder_transpose_on_the_left_in_place_scratch::<f64>(
            nrows, block_size, ncols,
        ),
    ])
}

fn solve_qr_inplace(
    a: &mut Mat<f64>,
    h_factor: &mut Mat<f64>,
    b: &Col<f64>,
    x: &mut Col<f64>,
    buf: &mut MemBuffer,
) {
    // writes newly solved lsq in x_cand
    let rank = Ord::min(a.nrows(), a.ncols());
    // this is not a costly operation. Just reinterpret the provided buffer as a stack.
    let mut stack = MemStack::new(buf);
    // I don't think we need to set the h_factors, they will be overwritten.
    factor::qr_in_place(
        a.as_mut(),
        h_factor.as_mut(),
        Par::Seq,
        stack.rb_mut(),
        Default::default(),
    );
    // how big does the solution mat need to be?
    x.copy_from(b);
    householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        a.as_ref(),
        h_factor.as_ref(),
        faer::Conj::Yes,
        x.as_mat_mut(),
        Par::Seq,
        stack.rb_mut(),
    );
    x.truncate(rank);
    triangular_solve::solve_upper_triangular_in_place(
        a.as_ref().split_at_row(rank).0,
        x.as_mat_mut(),
        Par::Seq,
    );
}

pub struct Omp<'a> {
    // Reserve space for running solution.
    running_solution: Col<f64>,
    // Reserve space for running residual.
    running_residual: Col<f64>,
    // Vector to store the indices that have nonzero components.
    support: Vec<usize>,
    a_sup: Mat<f64>,
    config: &'a MpConfig<'a>,
    // Buffer requirements to trigger reallocation if needed
    allocated_buf_req: StackReq,
    h_factor: Mat<f64>,
    // Space to write QR decomposition in.
    // Reuse over iterations.
    qr_mat: Mat<f64>,
    // Allocated memory for efficient qr solve
    buf: MemBuffer,
}

impl<'a> Omp<'a> {
    fn sweep(&mut self) -> usize {
        let mut cand = (0_usize, f64::MAX);
        // Precompute error, it is needed to check the reduced error with the added column a_j.
        let residual_error = self.running_residual.squared_norm_l2();
        // search for the unvisited column that reduces the residual error the most.
        for j in (0..self.config.a_ref().ncols()).filter(|i| !self.support.contains(i)) {
            let a_j = self.config.a_ref().col(j);
            let z_j_opt = inner_prod(
                a_j.transpose(),
                faer::Conj::No,
                self.running_residual.as_ref(),
                faer::Conj::No,
            );
            // update running candidate if a better column a_j is found.
            if (residual_error - z_j_opt.powi(2)) < cand.1 {
                cand = (j, residual_error - z_j_opt.powi(2));
            }
        }
        // return index of best column j
        cand.0
    }

    fn increase_allocated_memory(&mut self) {
        // increases memory according to self.a_sup
        let (nrows, ncols) = (self.a_sup.nrows(), self.a_sup.ncols());
        let (rank, block_size) = calculate_h_factor_req(nrows, ncols);
        self.h_factor.resize_with(block_size, rank, |_, _| 0.0);
        self.qr_mat.resize_with(nrows, ncols, |_, _| 0.0);
        self.running_solution
            .resize_with(self.config.b_ref().nrows(), |_| 0.0);
        let new_req = calculate_stack_req(nrows, ncols);
        if new_req.size_bytes() > self.allocated_buf_req.size_bytes() {
            self.buf = allocate_buffer(calculate_stack_req(nrows, (ncols - 1) * 2));
        }
    }

    fn update_support(&mut self, j_0: usize) {
        // Add column j to the matrix a_sup that reduces the error the most.
        self.a_sup.push_col(self.config.a_ref().col(j_0));
        // Add index j to the support vec.
        self.support.push(j_0);
    }

    fn update_running_solution(&mut self) {
        /*
        Solve Lstsq using a_sub and b.
        We want to find a sparse solution, thus this step should not be called to many times.
        */
        self.qr_mat.copy_from(self.a_sup.rb());
        solve_qr_inplace(
            &mut self.qr_mat,
            &mut self.h_factor,
            self.config.b_ref(),
            &mut self.running_solution,
            &mut self.buf,
        );
        // self.solve_qr();
    }

    fn return_reindex_solution(&self) -> Col<f64> {
        /*
         * reindex and write correct coefficients in vector that fits dimensions of the full matrix
         * A.
         */
        let mut res = Col::zeros(self.config.a_ref().ncols());
        for (x_i, &j_i) in self.running_solution.iter().zip(self.support.iter()) {
            res[j_i] = *x_i;
        }
        res
    }

    fn update_residual(&mut self) {
        // Compute running_residual = b - A @ running_solution,
        // this is the updated running_residual
        self.running_residual.copy_from(self.config.b_ref());
        matmul(
            self.running_residual.rb_mut(),
            Accum::Add,
            self.a_sup.rb(),
            self.running_solution.rb(),
            -1.0,
            Par::Seq,
        );
    }
}

impl<'a> Initializable<'a, MpConfig<'a>> for Omp<'a> {
    fn init(config: &'a MpConfig) -> Self {
        // reserve space that allows that the nullspace of a_sup can be empty.
        let mut a_sup = Mat::with_capacity(config.a_ref().nrows(), config.expected_size);
        unsafe {
            a_sup.set_dims(config.a_ref().nrows(), 0);
        }
        let req = calculate_stack_req(config.a_ref().nrows(), config.expected_size);
        let (rank, block_size) =
            calculate_h_factor_req(config.a_ref().nrows(), config.expected_size);
        let buf = allocate_buffer(req);
        let h_factor = Mat::with_capacity(block_size, rank);
        let qr_mat = Mat::with_capacity(config.a_ref().nrows(), config.expected_size);
        let running_solution = Col::zeros(config.b_ref().nrows());
        Omp {
            running_residual: config.b_ref().cloned(),
            support: Vec::with_capacity(config.a_ref().nrows()),
            a_sup,
            config,
            allocated_buf_req: req,
            h_factor: h_factor,
            qr_mat: qr_mat,
            buf: buf,
            running_solution,
        }
    }
}

impl<'a> Solver<'a> for Omp<'a> {
    fn solve(&mut self) -> Option<Col<f64>> {
        /*
        Repeat the process until at most all columns are visited,
        then we can surely produce b, unless b lies in the null-space of A then None is returned.
        */
        for _i in 0..self.config.a_ref().ncols() {
            let j_0 = self.sweep();
            self.update_support(j_0);
            self.increase_allocated_memory();
            self.update_running_solution();
            self.update_residual();
            if self.running_residual.squared_norm_l2() < self.config.sq_error_threshold {
                return Some(self.return_reindex_solution());
            }
        }
        None
    }

    fn solve_n_nonzero_coefs(&mut self, support_cardinality: usize) -> Option<Col<f64>> {
        /*
        Repeat the process until the running solution reaches the desired cardinality or visited all available columns.
        */
        for _i in 0..self.config.a_ref().ncols().min(support_cardinality) {
            let j_0 = self.sweep();
            self.update_support(j_0);
            self.increase_allocated_memory();
            self.update_running_solution();
            self.update_residual();
            if self.running_residual.squared_norm_l2() < self.config.sq_error_threshold {
                return Some(self.return_reindex_solution());
            }
        }
        Some(self.return_reindex_solution())
    }

    fn name(&self) -> String {
        "omp".to_string()
    }
}

pub fn expand_inverse(space: &mut Mat<f64>, ref_inv: &Mat<f64>) -> Option<()> {
    /*
    We divide the space matrix as follows:
    | b1 b2 |
    | b3 b4 |
    and expects a reference to M^-1.
    This method calculates:
    | M   b |^-1
    | b^T c |
    It expects space to contain:
    | -- b |
    | -- c |
    The dimensions are ref_inf kxk, b kx1, c 1x1
    so space has dims (k+1)x(k+1).
    */
    let k = ref_inv.nrows();

    // divide space and bind to mutable variables.
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
    // inversion can fail here.
    if p.is_infinite() {
        return None;
    }

    // We now have everything we need. Now fill the space.
    // Fill b4
    b4[(0, 0)] = p;

    // Fill b2 & b3
    b2.copy_from(-p * b3.rb().transpose());
    b3.copy_from(b2.rb().transpose());

    // Fill b1
    b1.copy_from(ref_inv);
    matmul(b1.rb_mut(), Accum::Add, b2.rb(), b3.rb(), 1.0 / p, Par::Seq);
    // Indicate that inversion was successful
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
     * inverse_space is free space to write the new inverse to.
     * as_space is divided as
     * | as ¦ |, where we can use the last column ¦ to fill it with ai and the rest is
     * filled with the matrix as, i.e. a_sup.
     */
    let k = stored_inverse.ncols();
    // Split as_space and bind to mutable variables.
    let (a_s, mut ai) = as_space.split_at_col_mut(k);
    // Write new new column ai space.
    ai.copy_from(a_ref.col(i).as_mat());

    // Adapt as_b for new ai.
    // We only need b^T@ai, the rest is already calculated.
    as_b[k] = inner_prod(
        ai.rb().col(0).transpose(),
        faer::Conj::No,
        b_ref.rb(),
        faer::Conj::No,
    );

    // Prepare to calculate inverse into inverse_space.
    let (_, mut b2) = inverse_space.split_at_col_mut(k);
    // Fill b2 & b4 as required by expand_inverse method.
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
    matmul(
        sweep_x_opt.rb_mut(),
        Accum::Replace,
        inverse_space.rb(),
        as_b,
        1.0,
        Par::Seq,
    );
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
    stored_inverse: Mat<f64>,
    as_space: Mat<f64>,
    inverse_space: Mat<f64>,
    sweep_x_opt: Col<f64>,
    sweep_residual: Col<f64>,
    as_b: Col<f64>,
    // Candidate is organized as:
    // (index, residual error, candidate inverse, candidate solution)
    candidate: (usize, f64, Mat<f64>, Col<f64>),
    config: &'a MpConfig<'a>,
}

impl<'a> LsOmp<'a> {
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
        // The ordering of the candidate solution doesn't match the ordering of the columns of the
        // matrix a_ref. This step takes care of the ordering.
        let mut x = Col::zeros(self.config.a_ref().ncols());
        for (x_i, &j_i) in self.candidate.3.iter().zip(self.support.iter()) {
            x[j_i] = *x_i;
        }
        x
    }

    fn scalar_sweep(&mut self) -> Option<f64> {
        // Reset residual_error of candidate.
        self.candidate.1 = f64::MAX;
        // Bind variables.
        let mut ai = self.as_space.col_mut(0);
        let ai_b = self.as_b.get_mut(0);
        let inv = self.inverse_space.get_mut(0, 0);
        let z = self.sweep_x_opt.get_mut(0);
        // Sweep through columns and update candidate, if the residual error is smaller.
        for i in 0..self.config.a_ref().ncols() {
            ai.copy_from(self.config.a_ref().col(i));
            *ai_b = inner_prod(
                ai.rb().transpose(),
                faer::Conj::No,
                self.config.b_ref().rb(),
                faer::Conj::No,
            );
            *inv = 1.0 / ai.squared_norm_l2();
            *z = *ai_b * *inv;
            let residual_error = (self.config.b_ref() - *z * ai.rb()).squared_norm_l2();
            if residual_error < self.candidate.1 {
                self.candidate.0 = i;
                self.candidate.1 = residual_error;
                self.candidate.2[(0, 0)] = *inv;
                self.candidate.3[0] = *z;
            }
        }
        if self.candidate.1 == f64::MAX {
            // No vaild column was found.
            return None;
        } else {
            // Vaild Column was found. Now update state of Solver to continue with matrix
            // calculations.
            self.support.push(self.candidate.0);
            self.residual_error = self.candidate.1;
            self.stored_inverse
                .resize_with(1, 1, |_, _| self.candidate.2[(0, 0)]);
            ai.copy_from(self.config.a_ref().col(self.candidate.0));
            *ai_b = inner_prod(
                ai.rb().transpose(),
                faer::Conj::No,
                self.config.b_ref().rb(),
                faer::Conj::No,
            );
            // println!("residual error after scalar sweep: {}", cand.1);
            return Some(self.candidate.1);
        }
    }

    fn sweep_and_update_running_solution(&mut self) -> Option<f64> {
        let k = self.support.len();

        // Reset candidate residual error.
        self.candidate.1 = f64::MAX;

        // sweep through remaining columns and keep candidate with lowest residul error.
        for i in (0..self.config.a_ref().ncols()).filter(|j| !self.support.contains(j)) {
            match lstsq_additional_column(
                i,
                &self.config.a_ref(),
                &self.config.b_ref(),
                &self.stored_inverse,
                &mut self.inverse_space,
                &mut self.as_space,
                &mut self.sweep_x_opt,
                &mut self.sweep_residual,
                &mut self.as_b,
            ) {
                Some(()) => {
                    // Inversion was successful, check if residual error is smaller and update
                    // candidate accordingly.
                    let residual_error = self.sweep_residual.squared_norm_l2();

                    if residual_error < self.candidate.1 {
                        self.candidate.0 = i;
                        self.candidate.1 = residual_error;
                        self.candidate.2.copy_from(self.inverse_space.rb());
                        self.candidate.3.copy_from(self.sweep_x_opt.rb());
                    }
                }
                // Inversion was not successful, continue with other columns.
                // Todo, do we need something else here?
                None => continue,
            }
        }

        if self.candidate.1 == f64::MAX {
            // No valid column was found.
            return None;
        } else {
            // Vaild column was found.
            // Update state of solver according to candidate.
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
                .copy_from(self.config.a_ref().col(self.candidate.0));
            matmul(
                self.as_b.rb_mut(),
                Accum::Replace,
                self.as_space.transpose(),
                self.config.b_ref(),
                1.0,
                Par::Seq,
            );
            return Some(self.candidate.1);
        }
    }
}

impl<'a> Initializable<'a, MpConfig<'a>> for LsOmp<'a> {
    fn init(config: &'a MpConfig<'a>) -> Self {
        // Initialize the problem and allocate space according to the expected_size.
        let residual_error = config.b_ref().squared_norm_l2();
        let support = Vec::new();
        let stored_inverse = Mat::with_capacity(config.expected_size, config.expected_size);

        let mut sweep_x_opt = Col::zeros(0);
        sweep_x_opt.reserve(config.expected_size);
        let sweep_residual = Col::zeros(config.b_ref().nrows());
        let mut as_b = Col::zeros(0);
        as_b.reserve(config.expected_size);

        let mut cand_col = Col::zeros(0);
        cand_col.reserve(config.expected_size);
        let candidate = (
            0,
            f64::MAX,
            Mat::with_capacity(config.expected_size, config.expected_size),
            cand_col,
        );

        LsOmp {
            residual_error: residual_error,
            support: support,
            config: config,
            stored_inverse: stored_inverse,
            as_space: Mat::with_capacity(config.b_ref().nrows(), config.expected_size),
            inverse_space: Mat::with_capacity(config.expected_size, config.expected_size),
            sweep_x_opt: sweep_x_opt,
            sweep_residual: sweep_residual,
            as_b: as_b,
            candidate: candidate,
        }
    }
}

impl<'a> Solver<'a> for LsOmp<'a> {
    fn solve(&mut self) -> Option<Col<f64>> {
        self.increase_size(1);
        // Scalar sweep, for optimiztion handle this case seperately.
        self.scalar_sweep()?;
        if self.residual_error <= self.config.sq_error_threshold {
            return Some(self.return_reindexed_solution());
        }
        // search until the value falls below the desired residual threshold.
        for i in 1..self.config.a_ref().ncols() {
            self.increase_size(i + 1);
            self.sweep_and_update_running_solution()?;
            if self.residual_error <= self.config.sq_error_threshold {
                return Some(self.return_reindexed_solution());
            }
        }
        None
    }

    fn solve_n_nonzero_coefs(&mut self, support_cardinality: usize) -> Option<Col<f64>> {
        // Same procedure as in but stop as soon as the desired cardinality is reached.
        self.increase_size(1);
        self.scalar_sweep()?;
        if self.residual_error <= self.config.sq_error_threshold || support_cardinality == 1 {
            return Some(self.return_reindexed_solution());
        }
        for i in 1..self.config.a_ref().ncols().min(support_cardinality) {
            self.increase_size(i + 1);
            self.sweep_and_update_running_solution()?;
            if self.residual_error <= self.config.sq_error_threshold {
                return Some(self.return_reindexed_solution());
            }
        }
        Some(self.return_reindexed_solution())
    }

    fn name(&self) -> String {
        "ls-omp".to_string()
    }
}

pub struct Mp<'a> {
    running_solution: Col<f64>,
    running_residual: Col<f64>,
    support: HashSet<usize>,
    max_iter: usize,
    config: &'a MpConfig<'a>,
}

impl<'a> Mp<'a> {
    fn sweep(&mut self) -> (usize, f64) {
        let mut cand = (0_usize, f64::MAX, 0.0);
        let residual_error = self.running_residual.squared_norm_l2();
        for i in 0..self.config.a_ref().ncols() {
            let col_i = self.config.a_ref().col(i);
            let z_i_opt = inner_prod(
                col_i.transpose(),
                faer::Conj::No,
                self.running_residual.as_ref(),
                faer::Conj::No,
            );

            if (residual_error - z_i_opt.powi(2)) < cand.1 {
                cand = (i, (residual_error - z_i_opt.powi(2)), z_i_opt);
            }
        }
        (cand.0, cand.2)
    }

    fn update_support(&mut self, j_0: usize) {
        if !self.support.insert(j_0) {}
    }

    fn update_running_solution(&mut self, j_0: usize, z_j_0: f64) {
        // Simplified version of OMP.
        // The coefficients of are not reevaluated with new support and only the new member is
        // added to the running solution.
        self.running_solution[j_0] += z_j_0;
    }

    fn update_residual(&mut self, j_0: usize, z_j_0: f64) {
        zip!(&mut self.running_residual, self.config.a_ref().col(j_0))
            .for_each(|unzip!(r, c)| *r = *r - z_j_0 * *c);
    }
}

impl<'a> Initializable<'a, MpConfig<'a>> for Mp<'a> {
    fn init(config: &'a MpConfig<'a>) -> Self {
        let running_solution = Col::zeros(config.a_ref().ncols());
        Mp {
            running_solution: running_solution,
            running_residual: config.b_ref().cloned(),
            support: HashSet::with_capacity(config.expected_size),
            max_iter: config.a_ref().ncols() * 200,
            config: config,
        }
    }
}

impl<'a> Solver<'a> for Mp<'a> {
    fn solve(&mut self) -> Option<Col<f64>> {
        for _i in 0..self.max_iter {
            let (j_0, z_j_0) = self.sweep();
            self.update_support(j_0);
            self.update_running_solution(j_0, z_j_0);
            self.update_residual(j_0, z_j_0);
            if self.running_residual.squared_norm_l2() < self.config.sq_error_threshold {
                return Some(self.running_solution.cloned());
            }
        }
        None
    }

    fn solve_n_nonzero_coefs(&mut self, support_cardinality: usize) -> Option<Col<f64>> {
        for _i in 0..self.max_iter {
            let (j_0, z_j_0) = self.sweep();
            self.update_support(j_0);
            self.update_running_solution(j_0, z_j_0);
            self.update_residual(j_0, z_j_0);
            if self.running_residual.squared_norm_l2() < self.config.sq_error_threshold
                || self.support.len() == support_cardinality
            {
                return Some(self.running_solution.cloned());
            }
        }
        Some(self.running_solution.cloned())
    }

    fn name(&self) -> String {
        "mp".to_string()
    }
}

pub struct WeakMp<'a> {
    running_solution: Col<f64>,
    running_residual: Col<f64>,
    support: HashSet<usize>,
    max_iter: usize,
    config: &'a WeakMpConfig<'a>,
}

impl<'a> WeakMp<'a> {
    fn sweep(&mut self) -> (usize, f64) {
        let mut cand = (0_usize, f64::MAX, 0.0);
        let residual_error = self.running_residual.squared_norm_l2();
        // Stop searching if residual_error is below residual_error * weaker_factor.
        for i in 0..self.config.a_ref().ncols() {
            let col_i = self.config.a_ref().col(i);
            let z_i_opt = inner_prod(
                col_i.transpose(),
                faer::Conj::No,
                self.running_residual.as_ref(),
                faer::Conj::No,
            );
            if z_i_opt.powi(2) >= self.config.weaker_factor.powi(2) * residual_error {
                return (i, z_i_opt);
            } else if (residual_error - z_i_opt.powi(2)) < cand.1 {
                cand.0 = i;
                cand.1 = residual_error - z_i_opt.powi(2);
                cand.2 = z_i_opt;
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
        zip!(&mut self.running_residual, self.config.a_ref().col(j_0))
            .for_each(|unzip!(r, c)| *r = *r - z_j_0 * *c);
    }
}

impl<'a> Initializable<'a, WeakMpConfig<'a>> for WeakMp<'a> {
    fn init(config: &'a WeakMpConfig<'a>) -> Self {
        assert!(
            (config.weaker_factor > 0.0) & (config.weaker_factor < 1.0),
            "Suboptimal sweep factor must be in range (0,1)"
        );
        let running_solution = Col::zeros(config.a_ref().ncols());
        WeakMp {
            running_solution: running_solution,
            running_residual: config.b_ref().cloned(),
            support: HashSet::with_capacity(config.expected_size),
            max_iter: config.a_ref().ncols() * 200,
            config,
        }
    }
}

impl<'a> Solver<'a> for WeakMp<'a> {
    fn solve(&mut self) -> Option<Col<f64>> {
        for _i in 0..self.max_iter {
            let (j_0, z_j_0) = self.sweep();
            self.update_support(j_0);
            self.update_running_solution(j_0, z_j_0);
            self.update_residual(j_0, z_j_0);
            if self.running_residual.squared_norm_l2() < self.config.sq_error_threshold {
                return Some(self.running_solution.cloned());
            }
        }
        None
    }

    fn solve_n_nonzero_coefs(&mut self, support_cardinality: usize) -> Option<Col<f64>> {
        for _i in 0..self.max_iter {
            let (j_0, z_j_0) = self.sweep();
            self.update_support(j_0);
            self.update_running_solution(j_0, z_j_0);
            self.update_residual(j_0, z_j_0);
            if self.running_residual.squared_norm_l2() < self.config.sq_error_threshold
                || self.support.len() == support_cardinality
            {
                return Some(self.running_solution.cloned());
            }
        }
        Some(self.running_solution.cloned())
    }

    fn name(&self) -> String {
        "weak-mp".to_string()
    }
}

pub struct Thresholding<'a> {
    config: &'a MpConfig<'a>,
    inner_products: Col<f64>,
    a_sup: Mat<f64>,
    // Reserve space for running solution.
    running_solution: Col<f64>,
    // Buffer requirements to trigger reallocation if needed
    allocated_buf_req: StackReq,
    h_factor: Mat<f64>,
    // Space to write QR decomposition in.
    // Reuse over iterations.
    qr_mat: Mat<f64>,
    // Allocated memory for efficient qr solve
    buf: MemBuffer,
}

impl<'a> Thresholding<'a> {
    fn compute_inner_products(&mut self) {
        matmul(
            self.inner_products.rb_mut(),
            Accum::Replace,
            self.config.a_ref().transpose(),
            self.config.b_ref(),
            1.0,
            Par::Seq,
        );
        self.inner_products.iter_mut().for_each(|e| *e = e.abs());
    }

    fn sort_indicies(&self) -> Vec<usize> {
        let mut indicies: Vec<usize> = (0..self.inner_products.nrows()).collect();
        indicies.sort_by(|&i, &j| self.inner_products[j].total_cmp(&self.inner_products[i]));
        indicies
    }

    fn increase_allocated_memory(&mut self) {
        // increases memory according to self.a_sup
        let (nrows, ncols) = (self.a_sup.nrows(), self.a_sup.ncols());
        let (rank, block_size) = calculate_h_factor_req(nrows, ncols);
        self.h_factor.resize_with(block_size, rank, |_, _| 0.0);
        self.qr_mat.resize_with(nrows, ncols, |_, _| 0.0);
        self.running_solution
            .resize_with(self.config.b_ref().nrows(), |_| 0.0);
        let new_req = calculate_stack_req(nrows, ncols);
        if new_req.size_bytes() > self.allocated_buf_req.size_bytes() {
            self.buf = allocate_buffer(calculate_stack_req(nrows, (ncols - 1) * 2));
        }
    }

    fn update_running_solution(&mut self) {
        /*
        Solve Lstsq using a_sub and b.
        We want to find a sparse solution, thus this step should not be called to many times.
        */
        self.qr_mat.copy_from(self.a_sup.rb());
        solve_qr_inplace(
            &mut self.qr_mat,
            &mut self.h_factor,
            self.config.b_ref(),
            &mut self.running_solution,
            &mut self.buf,
        );
    }

    fn reindex_solution(&self, indicies: &[usize]) -> Col<f64> {
        let mut running_solution = Col::zeros(self.config.a_ref().ncols());
        for (x_i, &j_i) in self.running_solution.iter().zip(indicies.iter()) {
            running_solution[j_i] = *x_i;
        }
        running_solution
    }
}

impl<'a> Solver<'a> for Thresholding<'a> {
    fn solve(&mut self) -> Option<Col<f64>> {
        let mut running_residual = Col::zeros(self.config.b_ref().nrows());
        self.compute_inner_products();
        let support_indicies = self.sort_indicies();
        for &idx in support_indicies.iter() {
            self.a_sup.push_col(self.config.a_ref().col(idx));
            self.increase_allocated_memory();
            self.update_running_solution();

            running_residual.copy_from(self.config.b_ref());
            //update running_residual
            matmul(
                running_residual.rb_mut(),
                Accum::Add,
                self.a_sup.rb(),
                self.running_solution.rb(),
                -1.0,
                Par::Seq,
            );

            if running_residual.squared_norm_l2() < self.config.sq_error_threshold {
                return Some(self.reindex_solution(&support_indicies));
            }
        }
        // Solution lies in the null space of A
        None
    }

    fn solve_n_nonzero_coefs(&mut self, support_cardinality: usize) -> Option<Col<f64>> {
        // Resize a_sup to n_nonzero coefs.
        self.a_sup
            .reserve(self.config.a_ref().nrows(), support_cardinality);

        self.compute_inner_products();
        let support_indicies = self.sort_indicies();
        // Fill a_sup with columns in question
        support_indicies
            .iter()
            .take(support_cardinality)
            .for_each(|&idx| self.a_sup.push_col(self.config.a_ref().col(idx)));

        self.increase_allocated_memory();
        self.update_running_solution();
        Some(self.reindex_solution(&support_indicies))
    }

    fn name(&self) -> String {
        "thresholding".to_string()
    }
}

impl<'a> Initializable<'a, MpConfig<'a>> for Thresholding<'a> {
    fn init(config: &'a MpConfig<'a>) -> Self {
        let inner_products = Col::zeros(config.a_ref().ncols());
        // reserve space that allows that the nullspace of a_sup can be empty.
        let mut a_sup = Mat::with_capacity(config.a_ref().nrows(), config.expected_size);
        unsafe {
            a_sup.set_dims(config.a_ref().nrows(), 0);
        }
        let req = calculate_stack_req(config.a_ref().nrows(), config.expected_size);
        let (rank, block_size) =
            calculate_h_factor_req(config.a_ref().nrows(), config.expected_size);
        let buf = allocate_buffer(req);
        let h_factor = Mat::with_capacity(block_size, rank);
        let qr_mat = Mat::with_capacity(config.a_ref().nrows(), config.expected_size);
        let running_solution = Col::zeros(config.b_ref().nrows());
        Thresholding {
            running_solution,
            config,
            inner_products,
            a_sup,
            allocated_buf_req: req,
            buf,
            h_factor,
            qr_mat,
        }
    }
}

pub struct Irls<'a> {
    // Reserve space for running solution.
    running_solution: Col<f64>,
    old_solution: Col<f64>,
    config: &'a BpConfig<'a>,
    weights: Diag<f64>,
    pinv: Mat<f64>,
    u: Mat<f64>,
    v: Mat<f64>,
    buf: MemBuffer,
    a_w_sq: Mat<f64>,
}

fn diag_mul(dst: MatMut<'_, f64>, d: &Diag<f64>, a: MatRef<'_, f64>) {
    assert_eq!(dst.nrows(), a.nrows());
    assert_eq!(dst.ncols(), a.ncols());
    assert_eq!(d.dim(), a.nrows());

    let d = d.column_vector();
    for (dst_j, a_j) in dst.col_iter_mut().zip(a.col_iter()) {
        zip!(dst_j, a_j, d).for_each(|unzip!(dst_ji, a_ji, d_i)| *dst_ji = *a_ji * d_i);
    }
}

impl<'a> Irls<'a> {
    fn update_weights(&mut self) {
        // we can compute W^2, because we only need this in later computations.
        self.weights
            .column_vector_mut()
            .iter_mut()
            .zip(self.running_solution.iter())
            .for_each(|(w_i, x_i)| *w_i = x_i.abs());
    }

    fn check_convergence(&self) -> bool {
        (self.running_solution.rb() - self.old_solution.rb()).squared_norm_l2()
            < self.config.stop_threshold_sq
    }

    fn prune_solution(&mut self) {
        self.running_solution.iter_mut().for_each(|x_i| {
            *x_i = if x_i.abs() < self.config.prune_threshold {
                0.0
            } else {
                *x_i
            }
        });
    }

    fn iteration_step(&mut self) -> Result<(), SvdError> {
        // compute running_solution = (weights)^2 @ A^T @ (A weights ^2 A^T)^(dagger) @ b

        // weights^2 @ A^T
        diag_mul(
            self.a_w_sq.rb_mut(),
            &self.weights,
            self.config.a_ref().transpose(),
        );

        // A @ weights^2 @ A^T
        matmul(
            self.pinv.rb_mut(),
            Accum::Replace,
            self.config.a_ref(),
            self.a_w_sq.rb(),
            1.0,
            Par::Seq,
        );

        let stack = MemStack::new(&mut self.buf);
        // Reuse allocated space from weights
        let mut s = self
            .weights
            .column_vector_mut()
            .subrows_mut(0, self.u.nrows())
            .as_diagonal_mut();
        // (A @ weights^2 @ A^T)^(dagger)
        svd(
            self.pinv.rb(),
            s.rb_mut(),
            Some(self.u.rb_mut()),
            Some(self.v.rb_mut()),
            Par::Seq,
            stack,
            default(),
        )?;
        pseudoinverse_from_svd(
            self.pinv.rb_mut(),
            s.rb(),
            self.u.rb(),
            self.v.rb(),
            Par::Seq,
            stack,
        );

        // (A @ weights^2 @ A^T)^(dagger) @ b
        // Use available matrix for intermediate result.
        let mut space = self.v.col_mut(0);
        matmul(
            space.rb_mut(),
            Accum::Replace,
            self.pinv.rb(),
            self.config.b_ref(),
            1.0,
            Par::Seq,
        );
        matmul(
            self.running_solution.rb_mut(),
            Accum::Replace,
            self.a_w_sq.rb(),
            space.rb(),
            1.0,
            Par::Seq,
        );
        Ok(())
    }
}

impl<'a> Initializable<'a, BpConfig<'a>> for Irls<'a> {
    fn init(config: &'a BpConfig) -> Self {
        let buf = allocate_buffer(StackReq::any_of(&[
            svd_scratch::<f64>(
                config.a_ref().nrows(),
                config.a_ref().nrows(),
                ComputeSvdVectors::Full,
                ComputeSvdVectors::Full,
                Par::Seq,
                default(),
            ),
            pseudoinverse_from_svd_scratch::<f64>(
                config.a_ref().nrows(),
                config.a_ref().nrows(),
                Par::Seq,
            ),
        ]));
        Self {
            running_solution: Col::ones(config.a_ref().ncols()),
            old_solution: Col::zeros(config.a_ref().ncols()),
            config,
            weights: Diag::ones(config.a_ref().ncols()),
            pinv: Mat::zeros(config.a_ref().nrows(), config.a_ref().nrows()),
            u: Mat::zeros(config.a_ref().nrows(), config.a_ref().nrows()),
            v: Mat::zeros(config.a_ref().nrows(), config.a_ref().nrows()),
            buf: buf,
            a_w_sq: Mat::zeros(config.a_ref().ncols(), config.a_ref().nrows()),
        }
    }
}

impl<'a> Solver<'a> for Irls<'a> {
    fn solve(&mut self) -> Option<Col<f64>> {
        for _i in 0..self.config.max_iter {
            self.iteration_step().ok()?;
            self.update_weights();
            if self.check_convergence() {
                self.prune_solution();
                return Some(self.running_solution.cloned());
            } else {
                self.old_solution.copy_from(&self.running_solution);
            }
        }
        None
    }

    fn solve_n_nonzero_coefs(&mut self, support_cardinality: usize) -> Option<Col<f64>> {
        // Just take the n entries with highes absolut values.
        // But this is highly non-optimal
        let mut res = self.solve()?;

        let mut indicies: Vec<usize> = (0..self.running_solution.nrows()).collect();
        indicies.sort_by(|&i, &j| self.running_solution[j].total_cmp(&self.running_solution[i]));
        res.fill(0.0);
        indicies
            .iter()
            .take(support_cardinality)
            .for_each(|&i| res[i] = self.running_solution[i]);
        Some(res)
    }

    fn name(&self) -> String {
        "irls".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 5;
    const M: usize = 3;
    fn setup() -> (Diag<f64>, Mat<f64>) {
        let a = Mat::from_fn(N, M, |i, j| (i * 10 + j) as f64 + 0.5);
        let d: Diag<f64> = Col::from_fn(N, |i| (i + 1) as f64 * 2.0).into_diagonal();
        (d, a)
    }

    #[test]
    fn left_diag() {
        let (d, a) = setup();

        let mut out = Mat::<f64>::zeros(N, M);
        diag_mul(out.rb_mut(), &d, a.as_ref());

        let dense = Mat::from_fn(N, N, |i, j| {
            if i == j {
                d.as_ref().column_vector()[i]
            } else {
                0.0
            }
        });
        let expected = &dense * &a;
        for i in 0..N {
            for j in 0..M {
                assert!(
                    (out[(i, j)] - expected[(i, j)]).abs() < 1.0e-12,
                    "missmatch at ({i}, {j})"
                );
            }
        }
    }
}
