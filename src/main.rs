use crate::{
    inspection::gen_ax_b,
    mp::{BaseConfig, Initializable, Solver, WeakMpConfig},
};
use faer::{linalg::qr::col_pivoting::factor, prelude::*};
// use crate::problem_analysis::{calc_solution_l2_error, calc_solution_support_dist};

mod inspection;
mod mp;

mod tester {
    use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
    use faer::linalg::qr::no_pivoting::factor;
    use faer::linalg::{householder, qr, triangular_solve};
    use faer::prelude::*;

    pub fn test_stack_req() -> bool {
        let nrows = 5;
        let ncols = 9;
        let block_size = factor::recommended_block_size::<f64>(nrows, ncols);
        let s1 = factor::qr_in_place_scratch::<f64>(
            nrows,
            ncols,
            block_size,
            Par::Seq,
            Default::default(),
        );
        let s2 = householder::apply_block_householder_transpose_on_the_left_in_place_scratch::<f64>(
            nrows, block_size, ncols,
        );
        s1.size_bytes() >= s2.size_bytes()
    }

    pub fn allocate_buffer(nrows: usize, ncols: usize) -> MemBuffer {
        let block_size = factor::recommended_block_size::<f64>(nrows, ncols);
        MemBuffer::new(StackReq::any_of(&[
            factor::qr_in_place_scratch::<f64>(
                nrows,
                ncols,
                block_size,
                Par::Seq,
                Default::default(),
            ),
            householder::apply_block_householder_transpose_on_the_left_in_place_scratch::<f64>(
                nrows, block_size, ncols,
            ),
        ]))
    }
    pub fn solve_qr(
        a_sup: &Mat<f64>,
        buf: MemBuffer,
        qr_mat: &mut Mat<f64>,
        h_factor: &mut Mat<f64>,
        b_ref: ColRef<f64>,
        x_cand: &mut Col<f64>,
    ) {
        let mut buf = buf;
        let rank = Ord::min(a_sup.nrows(), a_sup.ncols());
        let mut stack = MemStack::new(&mut buf);
        qr_mat.copy_from(a_sup.rb());
        // I don't think we need to set the h_factors, they will be overwritten.
        factor::qr_in_place(
            qr_mat.as_mut(),
            h_factor.as_mut(),
            Par::Seq,
            stack.rb_mut(),
            Default::default(),
        );
        // how big does the solution mat need to be?
        x_cand.copy_from(b_ref);
        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            qr_mat.as_ref(),
            h_factor.as_ref(),
            faer::Conj::Yes,
            x_cand.as_mat_mut(),
            Par::Seq,
            stack.rb_mut(),
        );
        x_cand.truncate(rank);
        triangular_solve::solve_upper_triangular_in_place(
            qr_mat.as_ref().split_at_row(rank).0,
            x_cand.as_mat_mut(),
            Par::Seq,
        );
    }
}

fn main() {
    // let (a, x, b) = gen_ax_b(5, 8, 4);
    // let mut a_sup = Mat::with_capacity(5, 4);
    // unsafe {
    //     a_sup.set_dims(5, 0);
    // }
    // a_sup.push_col(a.col(0));
    // a_sup.push_col(a.col(1));
    // a_sup.push_col(a.col(3));
    // a_sup.push_col(a.col(5));
    // let block_size = factor::recommended_block_size::<f64>(a_sup.nrows(), a_sup.ncols());
    // let mut x_cand = Col::zeros(b.nrows());
    // let mut h_factor = Mat::ones(block_size, 4);
    // let buf = tester::allocate_buffer(1, 1);
    // let mut qr_mat = a_sup.cloned();
    // tester::solve_qr(
    //     &a_sup,
    //     buf,
    //     &mut qr_mat,
    //     &mut h_factor,
    //     b.as_ref(),
    //     &mut x_cand,
    // );
    // println!("a_sup: {a_sup:.2?}");
    // println!("b: {b:.2?}");
    // println!("x_cand: {x_cand:.2?}");

    // inspection::experiments::repeated_random_comparisons(
    //     "30_50_exact.csv",
    //     500,
    //     30,
    //     50,
    //     &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    // )
    // .expect("got an csv error");
    // inspection::experiments::repeated_random_comparisons(
    //     "60_100_exact.csv",
    //     500,
    //     60,
    //     100,
    //     &[5, 7, 9, 11, 13, 15, 17, 19, 21],
    // )
    // .expect("got an csv error");
    // inspection::experiments::repeated_random_comparisons_fixed_cardinality(
    //     "30_50_fixed_card.csv",
    //     500,
    //     30,
    //     50,
    //     &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    // )
    // .expect("got an csv error");
    // inspection::experiments::repeated_random_comparisons_fixed_cardinality(
    //     "60_100_fixed_card.csv",
    //     500,
    //     60,
    //     100,
    //     &[5, 7, 9, 11, 13, 15, 17, 19, 21],
    // )
    // .expect("got an csv error");
    // println!("Done testing");

    let (a, x, b) = inspection::gen_ax_b(100, 300, 30);
    let conf = BaseConfig::init(&a, &b, 1.0e-6, 10);
    // let weakconf = WeakMpConfig::init(&a, &b, 1.0e-6, b.nrows() / 5, 0.5);

    println!("OMP:");
    // println!("True solution: {:.3?}", x);
    let mut problem = mp::Omp::init(&conf);
    if let Some(x_cand) = problem.solve() {
        let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
        let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
        // println!("cand solution: {:.3?}", x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
    }

    println!("Thresholding:");
    let mut problem = mp::Thresholding::init(&conf);
    if let Some(x_cand) = problem.solve() {
        let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
        let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
        // println!("cand solution: {:.3?}", x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
    }

    println!("LS-OMP:");
    let mut omp_ls_problem = mp::LsOmp::init(&conf);
    if let Some(x_cand) = omp_ls_problem.solve() {
        let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
        let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
        // println!("x_cand: {x_cand:.2?}");
        // println!("x: {x:.2?}");
    } else {
        println!("Got None");
    }
    //
    // println!("MP");
    // let mut mp_problem = mp::Mp::init(&conf);
    // if let Some(x_cand) = mp_problem.solve() {
    //     let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
    //     let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
    //     println!("l2 error: {l2_error:.4e}");
    //     println!("support dist: {support_dist:.4e}");
    //     // println!("x_cand: {x_cand:.2?}");
    //     // println!("x: {x:.2?}");
    // } else {
    //     println!("Got None");
    // }
    //
    // println!("Weak-MP");
    // let mut weak_mp_problem = mp::WeakMp::init(&weakconf);
    // if let Some(x_cand) = weak_mp_problem.solve() {
    //     let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
    //     let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
    //     println!("l2 error: {l2_error:.4e}");
    //     println!("support dist: {support_dist:.4e}");
    //     // println!("x_cand: {x_cand:.2?}");
    //     // println!("x: {x:.2?}");
    // } else {
    //     println!("Got None");
    // }
}
