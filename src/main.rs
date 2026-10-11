use crate::mp::Solver;
use crate::problem_analysis::{calc_solution_l2_error, calc_solution_support_dist};
mod mp;

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

        for _ in 0..rng.random_range(1..=cardinality_bound) {
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

        let mut a = fill_mat_with_normals(nrows, ncols, mean, std);
        let x = fill_x_with_uniforms(ncols, start, end, cardinality_bound);
        //normalize columns
        a.col_iter_mut().for_each(|col_i| {
            let scaler = col_i.norm_l2();
            col_i.iter_mut().for_each(|c_ii| *c_ii /= scaler);
        });
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
    let (a, x, b) = problem_analysis::gen_ax_b(100, 500, 33);

    println!("OMP:");
    let mut problem = mp::Omp::init(&a, &b);
    if let Some(x_cand) = problem.solve() {
        let l2_error = calc_solution_l2_error(&x, &x_cand);
        let support_dist = calc_solution_support_dist(&x, &x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
    }

    println!("LS-OMP:");
    let mut omp_ls_problem = mp::LsOmp::init(&a, &b);
    if let Some(x_cand) = omp_ls_problem.solve() {
        let l2_error = calc_solution_l2_error(&x, &x_cand);
        let support_dist = calc_solution_support_dist(&x, &x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
        // println!("x_cand: {x_cand:.2?}");
        // println!("x: {x:.2?}");
    } else {
        println!("Got None");
    }

    println!("MP");
    let mut mp_problem = mp::Mp::init(&a, &b);
    if let Some(x_cand) = mp_problem.solve() {
        let l2_error = calc_solution_l2_error(&x, &x_cand);
        let support_dist = calc_solution_support_dist(&x, &x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
        // println!("x_cand: {x_cand:.2?}");
        // println!("x: {x:.2?}");
    } else {
        println!("Got None");
    }

    println!("Weak-MP");
    let mut weak_mp_problem = mp::WeakMp::init(&a, &b, 0.5);
    if let Some(x_cand) = weak_mp_problem.solve() {
        let l2_error = calc_solution_l2_error(&x, &x_cand);
        let support_dist = calc_solution_support_dist(&x, &x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
        // println!("x_cand: {x_cand:.2?}");
        // println!("x: {x:.2?}");
    } else {
        println!("Got None");
    }
}
