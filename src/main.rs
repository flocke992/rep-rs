use crate::mp::{ProblemConfig, Solver};
// use crate::problem_analysis::{calc_solution_l2_error, calc_solution_support_dist};

mod inspection;
mod mp;

fn main() {
    // inspection::experiments::repeated_random_comparisons(
    //     "30_50_exact.csv",
    //     400,
    //     30,
    //     50,
    //     &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    // )
    // .expect("got an csv error");
    // inspection::experiments::repeated_random_comparisons(
    //     "60_100_exact.csv",
    //     400,
    //     60,
    //     100,
    //     &[5, 7, 9, 11, 13, 15, 17, 19, 21],
    // )
    // .expect("got an csv error");
    // println!("Done testing");

    let (a, x, b) = inspection::gen_ax_b(100, 500, 33);
    let conf = ProblemConfig::init(&a, &b, 1.0e-6, b.nrows() / 5);

    println!("OMP:");
    let mut problem = mp::Omp::init(&conf);
    if let Some(x_cand) = problem.solve() {
        let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
        let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
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

    println!("MP");
    let mut mp_problem = mp::Mp::init(&conf);
    if let Some(x_cand) = mp_problem.solve() {
        let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
        let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
        // println!("x_cand: {x_cand:.2?}");
        // println!("x: {x:.2?}");
    } else {
        println!("Got None");
    }

    println!("Weak-MP");
    let mut weak_mp_problem = mp::WeakMp::init(&conf);
    if let Some(x_cand) = weak_mp_problem.solve() {
        let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
        let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
        // println!("x_cand: {x_cand:.2?}");
        // println!("x: {x:.2?}");
    } else {
        println!("Got None");
    }
}
