use crate::{
    inspection::gen_ax_b,
    mp::{BaseConfig, Initializable, Solver, WeakMpConfig},
};

mod inspection;
mod mp;

fn main() {
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
    //
    // inspection::experiments::repeated_random_error_threshold_comparisons(
    //     "60_100_thresholds.csv",
    //     300,
    //     60,
    //     100,
    //     10,
    //     &[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
    // )
    // .expect("got an csv error");
    //
    // println!("Done testing");

    let (a, x, b) = inspection::gen_ax_b(50, 120, 10);
    let conf = BaseConfig::init(&a, &b, 1.0e-6, 10);
    // // let weakconf = WeakMpConfig::init(&a, &b, 1.0e-6, b.nrows() / 5, 0.5);
    //
    // // println!("True solution: {:.3?}", x);
    println!("OMP:");
    let mut problem = mp::Omp::init(&conf);
    if let Some(x_cand) = problem.solve() {
        let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
        let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
        // println!("cand solution: {:.3?}", x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
    }
    println!("irls:");
    let mut problem = mp::Irls::init(&conf);
    if let Some(x_cand) = problem.solve() {
        let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
        let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
        // println!("cand solution: {:.3?}", x_cand);
        println!("l2 error: {l2_error:.4e}");
        println!("support dist: {support_dist:.4e}");
    }
    //
    // println!("Thresholding:");
    // let mut problem = mp::Thresholding::init(&conf);
    // if let Some(x_cand) = problem.solve() {
    //     let l2_error = inspection::calc_solution_l2_error(&x, &x_cand);
    //     let support_dist = inspection::calc_solution_support_dist(&x, &x_cand);
    //     // println!("cand solution: {:.3?}", x_cand);
    //     println!("l2 error: {l2_error:.4e}");
    //     println!("support dist: {support_dist:.4e}");
    // }
    //
    // println!("LS-OMP:");
    // let mut omp_ls_problem = mp::LsOmp::init(&conf);
    // if let Some(x_cand) = omp_ls_problem.solve() {
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
