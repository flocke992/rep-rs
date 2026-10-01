use crate::problem_analysis::{calc_solution_l2_error, calc_solution_support_dist};

mod mp {
    // IMPROVEMENTS
    // [ ] Implement auto normalization of columns of the matrix A; A_tilde = W@A, where W is diag
    // [ ] Implement other greedy methods (OMP-LS, MP, Weak-MP, Thresholding)
    // [ ] Improve performance by managing resources properly
    // [ ] Create a trait

    use faer::linalg::matmul::dot::inner_prod;
    use faer::linalg::matmul::matmul;
    use faer::linalg::solvers::SolveLstsq;
    use faer::reborrow::*;
    use faer::{Accum, Col, Mat, Par, unzip, zip};

    pub struct Omp<'a> {
        running_solution: Col<f64>,
        running_residual: Col<f64>,
        support: Vec<usize>,
        error_threshold: f64,
        a_ref: &'a Mat<f64>, // ToDo use a reference here.
        a_sup: Mat<f64>,
        b: Col<f64>,
    }

    impl<'a> Omp<'a> {
        pub fn init(a_ref: &'a Mat<f64>, b: Col<f64>) -> Self {
            let running_solution = Col::zeros(a_ref.ncols());
            let a_sup = Mat::zeros(a_ref.nrows(), a_ref.ncols());
            Omp {
                running_solution: running_solution,
                running_residual: b.cloned(),
                support: Vec::with_capacity(a_ref.nrows()),
                error_threshold: 1.0e-8,
                a_ref,
                a_sup,
                b,
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
                if self.running_residual.squared_norm_l2()
                    < self.error_threshold * self.error_threshold
                {
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
                    let col_normalizer = col_i.squared_norm_l2();
                    let mut res = self.running_residual.squared_norm_l2();
                    res = res
                        - inner_prod(
                            col_i.transpose(),
                            faer::Conj::No,
                            self.running_residual.as_ref(),
                            faer::Conj::No,
                        )
                        .powi(2)
                            / col_normalizer;
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
            self.support.insert(self.support.len(), j_0);
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
            let x = qr.solve_lstsq(&self.b);
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
            zip!(&mut self.running_residual, &self.b).for_each(|unzip!(r, b)| *r = b - *r);
        }
    }
}

mod problem_analysis {
    use std::collections::HashSet;

    use faer::{Col, Mat};
    use rand::RngExt;
    use rand_distr::{Bernoulli, Distribution, Normal, Uniform};

    fn fill_mat_with_normals(nrows: usize, ncols: usize, mean: f64, std: f64) -> Mat<f64> {
        assert!(std >= 0.0);
        let mut rng = rand::rng();
        let mut a = Mat::zeros(nrows, ncols);

        let normal = Normal::new(mean, std).unwrap();
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                a[(i, j)] = normal.sample(&mut rng)
            }
        }
        a
    }

    fn fill_x_with_uniforms(n: usize, start: f64, end: f64, cardinality_bound: usize) -> Col<f64> {
        assert!(n >= cardinality_bound);
        assert!(start <= end);

        let mut x = Col::zeros(n);

        let mut rng = rand::rng();
        let side = Uniform::new(start, end).unwrap();
        let ber = Bernoulli::new(0.5).unwrap();

        // for _ in 0..rng.random_range(1..=cardinality_bound) {
        for _ in 1..=cardinality_bound {
            let i = rng.random_range(0..n);
            let x_i = side.sample(&mut rng)
                * match ber.sample(&mut rng) {
                    true => 1.0,
                    false => -1.0,
                };
            x[i] = x_i;
        }
        x
    }

    pub fn gen_ax_b(
        nrows: usize,
        ncols: usize,
        cardinality_bound: usize,
    ) -> (Mat<f64>, Col<f64>, Col<f64>) {
        //arbitrary set values
        let mean = 0.0;
        let std = 2.5;
        let start = 1.0;
        let end = 7.0;

        let a = fill_mat_with_normals(nrows, ncols, mean, std);
        let x = fill_x_with_uniforms(ncols, start, end, cardinality_bound);
        let b = &a * &x;
        (a, x, b)
    }

    fn determine_support(x: &Col<f64>, threshold: f64) -> HashSet<usize> {
        x.iter()
            .enumerate()
            .filter(|(_i, x_i)| x_i.abs() < threshold)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn calc_solution_l2_error(x: &Col<f64>, x_cand: &Col<f64>) -> f64 {
        (x - x_cand).squared_norm_l2() / x.squared_norm_l2()
    }

    pub fn calc_solution_support_dist(x: &Col<f64>, x_cand: &Col<f64>) -> f64 {
        let x_sup = determine_support(x, 1.0e-10);
        let x_cand_sup = determine_support(x_cand, 1.0e-10);
        (x_sup.len().max(x_cand_sup.len()) - x_sup.intersection(&x_cand_sup).count()) as f64
            / x_sup.len().max(x_cand_sup.len()) as f64
    }
}

fn main() {
    let (a, x, b) = problem_analysis::gen_ax_b(100, 500, 20);
    // let (a, x, b) = problem_analysis::gen_ax_b(3, 10, 2);
    println!("{:?}", a.shape());

    let mut problem = mp::Omp::init(&a, b);
    if let Some(x_cand) = problem.solve() {
        let l2_error = calc_solution_l2_error(&x, &x_cand);
        let support_dist = calc_solution_support_dist(&x, &x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
    }
}
