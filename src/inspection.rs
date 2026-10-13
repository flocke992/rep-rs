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

fn fill_x_with_unifroms_exact(
    n: usize,
    start: f64,
    end: f64,
    exact_caridnality: usize,
) -> Col<f64> {
    assert!(n >= exact_caridnality);
    assert!(start <= end);

    let mut x = Col::zeros(n);

    let mut rng = rand::rng();
    let side = Uniform::new(start, end).unwrap();
    let ber = Bernoulli::new(0.5).unwrap();
    let mut support = HashSet::new();

    for _ in 0..rng.random_range(1..=exact_caridnality) {
        let mut i = rng.random_range(0..n);
        loop {
            if support.insert(i) {
                break;
            } else {
                i = rng.random_range(0..n);
            }
        }
        let x_i = side.sample(&mut rng)
            * match ber.sample(&mut rng) {
                true => 1.0,
                false => -1.0,
            };
        x[i] = x_i;
    }
    x
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

pub fn gen_ax_b_exact(
    nrows: usize,
    ncols: usize,
    exact_caridnality: usize,
) -> (Mat<f64>, Col<f64>, Col<f64>) {
    //arbitrary set values
    let mean = 0.0;
    let std = 2.5;
    let start = 1.0;
    let end = 7.0;

    let mut a = fill_mat_with_normals(nrows, ncols, mean, std);
    let x = fill_x_with_unifroms_exact(ncols, start, end, exact_caridnality);
    //normalize columns
    a.col_iter_mut().for_each(|col_i| {
        let scaler = col_i.norm_l2();
        col_i.iter_mut().for_each(|c_ii| *c_ii /= scaler);
    });
    let b = &a * &x;
    (a, x, b)
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

pub mod experiments {
    use faer::Col;

    use crate::inspection::{
        calc_solution_l2_error, calc_solution_support_dist, gen_ax_b, gen_ax_b_exact,
    };
    use crate::mp::{LsOmp, Mp, Omp, ProblemConfig, Solver, WeakMp};
    use csv::Writer;
    use std::error::Error;
    use std::hint::black_box;
    use std::time::Instant;

    fn measure_time_solve<'a, S>(s: &mut S) -> Option<(f64, Col<f64>)>
    where
        S: Solver<'a>,
    {
        let start = Instant::now();
        let res = black_box(s.solve());
        let elapsed = start.elapsed();
        match res {
            Some(x_opt) => Some((elapsed.as_secs_f64(), x_opt)),
            None => None,
        }
    }
    pub fn repeated_random_comparisons(
        file_path: &str,
        nrepeats: usize,
        nrows: usize,
        ncols: usize,
        solution_caridnalities: &[usize],
    ) -> Result<(), Box<dyn Error>> {
        let mut writer = Writer::from_path(file_path)?;
        writer.write_record([
            "algorithm",
            "cardinality",
            "time",
            "l2_recovery_error",
            "solution_support_distance",
        ])?;
        for &solution_cardinality in solution_caridnalities {
            for i in 0..nrepeats {
                let (a, x, b) = gen_ax_b_exact(nrows, ncols, solution_cardinality);
                let config = ProblemConfig::init(&a, &b, 1.0e-6, b.nrows() / 5);

                let mut omp = Omp::init(black_box(&config));
                let res = measure_time_solve(&mut omp);
                match res {
                    Some((time, x_opt)) => {
                        let l2_error = calc_solution_l2_error(&x, &x_opt);
                        let support_dist = calc_solution_support_dist(&x, &x_opt);
                        writer.write_record([
                            "omp".to_string(),
                            solution_cardinality.to_string(),
                            time.to_string(),
                            format!("{:.6e}", l2_error),
                            format!("{:.6e}", support_dist),
                        ])?;
                    }
                    None => {
                        println!(
                            "could not solve problem OMP, card: {solution_cardinality}, rep: {i}"
                        );
                    }
                }

                let mut ls_omp = LsOmp::init(black_box(&config));
                let res = measure_time_solve(&mut ls_omp);
                match res {
                    Some((time, x_opt)) => {
                        let l2_error = calc_solution_l2_error(&x, &x_opt);
                        let support_dist = calc_solution_support_dist(&x, &x_opt);
                        writer.write_record([
                            "ls-omp".to_string(),
                            solution_cardinality.to_string(),
                            time.to_string(),
                            format!("{:.6e}", l2_error),
                            format!("{:.6e}", support_dist),
                        ])?;
                    }
                    None => {
                        println!(
                            "could not solve problem LS-OMP, card: {solution_cardinality}, rep: {i}"
                        );
                    }
                }

                let mut mp = Mp::init(black_box(&config));
                let res = measure_time_solve(&mut mp);
                match res {
                    Some((time, x_opt)) => {
                        let l2_error = calc_solution_l2_error(&x, &x_opt);
                        let support_dist = calc_solution_support_dist(&x, &x_opt);
                        writer.write_record([
                            "mp".to_string(),
                            solution_cardinality.to_string(),
                            time.to_string(),
                            format!("{:.6e}", l2_error),
                            format!("{:.6e}", support_dist),
                        ])?;
                    }
                    None => {
                        println!(
                            "could not solve problem MP, card: {solution_cardinality}, rep: {i}"
                        );
                    }
                }

                let mut weak_mp = WeakMp::init(black_box(&config));
                let res = measure_time_solve(&mut weak_mp);
                match res {
                    Some((time, x_opt)) => {
                        let l2_error = calc_solution_l2_error(&x, &x_opt);
                        let support_dist = calc_solution_support_dist(&x, &x_opt);
                        writer.write_record([
                            "weak-mp".to_string(),
                            solution_cardinality.to_string(),
                            time.to_string(),
                            format!("{:.6e}", l2_error),
                            format!("{:.6e}", support_dist),
                        ])?;
                    }
                    None => {
                        println!(
                            "could not solve problem Weak-MP, card: {solution_cardinality}, rep: {i}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
