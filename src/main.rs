#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use dntk_matrix::matrix::*;

fn main() {
    let ((train_x, train_t), (test_x, test_t)) = data();

    // 01
    // y(x, w) = w_0 + w_1*x_1 + ... + w_D * x_D
    // p(t|x, w, b) = N(t|y(x,w), b^-1)
    let phi = train_x.clone();
    let phi_t = phi.clone().transpose();

    let left = phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/weight01_simple.csv", content)
        .expect("fail to save weight to ./parameters/weight01_simple.csv");

    let inference = test_x.clone() * w;
    println!(
        "test: {} / 70",
        (0..70)
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count()
    );

    // 02
    // y(x, w) = w_0 + w_1*x_1 + ... + w_D * x_D
    // regularization: lambda = 0.5
    let lambda = 0.5;
    let lambda_i = {
        let mut o = [0.0; 32 * 32];
        for i in 0..32 {
            o[i * 32 + i] = lambda;
        }
        Matrix::<32, 32, _, _>::new(o)
    };

    let left = lambda_i + phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/weight02_simple_reg.csv", content)
        .expect("fail to save weight to ./parameters/weight02_simple_reg.csv");

    let inference = test_x.clone() * w;
    println!(
        "test: {} / 70",
        (0..70)
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count()
    );

    // 03
    // y(x, w) = w_0 + w_1*x_1 + ... + w_D * x_D
    // regularization: lambda = 1.0
    let lambda = 1.0;
    let lambda_i = {
        let mut o = [0.0; 32 * 32];
        for i in 0..32 {
            o[i * 32 + i] = lambda;
        }
        Matrix::<32, 32, _, _>::new(o)
    };

    let left = lambda_i + phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/weight03_simple_reg.csv", content)
        .expect("fail to save weight to ./parameters/weight03_simple_reg.csv");

    let inference = test_x.clone() * w;
    println!(
        "test: {} / 70",
        (0..70)
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count()
    );

    // 04
    // y(x, w) = w_0 + w_1*x_1 + ... + w_D * x_D
    // regularization: lambda = 10.0
    let lambda = 10.0;
    let lambda_i = {
        let mut o = [0.0; 32 * 32];
        for i in 0..32 {
            o[i * 32 + i] = lambda;
        }
        Matrix::<32, 32, _, _>::new(o)
    };

    let left = lambda_i + phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/weight04_simple_reg.csv", content)
        .expect("fail to save weight to ./parameters/weight04_simple_reg.csv");

    let inference = test_x.clone() * w;
    println!(
        "test: {} / 70",
        (0..70)
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count()
    );

    // 05
    // phi_i = exp(-(x_i - u)^2 / 2s^2)
    let gauss_0_31 = gauss_slice();
    let (phi, phi_t) = design_matrix(&train_x, &gauss_0_31);

    let left = phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/weight05_all_gauss.csv", content)
        .expect("fail to save weight to ./parameters/weight05_all_gauss.csv");

    let inference = design_matrix(&test_x, &gauss_0_31).0 * w;
    println!(
        "test: {} / 70",
        (0..70)
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count()
    );

    // 05
    // select effective parameters
    let effective = select_parameters();
    let (phi, phi_t) = design_matrix(&train_x, &effective);

    let left = phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/weight06_select.csv", content)
        .expect("fail to save weight to ./parameters/weight06_select.csv");

    let inference = design_matrix(&test_x, &effective).0 * w;
    println!(
        "test: {} / 70",
        (0..70)
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count()
    );
}

fn select_parameters() -> [Box<dyn Fn([f64; 32]) -> f64>; 32] {
    let select = move |i: usize| move |e: [f64; 32]| e[i];
    let none = move |_: [f64; 32]| 0.0;
    [
        Box::new(select.clone()(0)),
        Box::new(none.clone()),
        Box::new(select.clone()(2)),
        Box::new(select.clone()(3)),
        Box::new(select.clone()(4)),
        Box::new(select.clone()(5)),
        Box::new(select.clone()(6)),
        Box::new(none.clone()),
        Box::new(none.clone()),
        Box::new(select.clone()(9)),
        Box::new(select.clone()(10)),
        Box::new(select.clone()(11)),
        Box::new(none.clone()),
        Box::new(none.clone()),
        Box::new(none.clone()),
        Box::new(none.clone()),
        Box::new(select.clone()(16)),
        Box::new(select.clone()(17)),
        Box::new(select.clone()(18)),
        Box::new(select.clone()(19)),
        Box::new(none.clone()),
        Box::new(select.clone()(21)),
        Box::new(select.clone()(22)),
        Box::new(select.clone()(23)),
        Box::new(select.clone()(24)),
        Box::new(select.clone()(25)),
        Box::new(select.clone()(26)),
        Box::new(select.clone()(27)),
        Box::new(select.clone()(28)),
        Box::new(select.clone()(29)),
        Box::new(select.clone()(30)),
        Box::new(select.clone()(31)),
    ]
}

fn gauss_slice() -> [Box<dyn Fn([f64; 32]) -> f64>; 32] {
    let id = |e: [f64; 32]| e[31];
    let gauss = move |u: f64, s: f64, i: usize| {
        move |e: [f64; 32]| f64::exp(-(e[i] - u) * (e[i] - u) / 2.0 / (s * s))
    };
    [
        Box::new(gauss.clone()(2.0, 1.0, 0)),
        Box::new(gauss.clone()(1.5, 1.0, 1)),
        Box::new(gauss.clone()(2.0, 1.0, 2)),
        Box::new(gauss.clone()(3.0, 1.0, 3)),
        Box::new(gauss.clone()(1.5, 1.0, 4)),
        Box::new(gauss.clone()(1.5, 1.0, 5)),
        Box::new(gauss.clone()(1.5, 1.0, 6)),
        Box::new(gauss.clone()(3.0, 1.0, 7)),
        Box::new(gauss.clone()(2.5, 1.0, 8)),
        Box::new(gauss.clone()(2.5, 1.0, 9)),
        Box::new(gauss.clone()(3.5, 1.0, 10)),
        Box::new(gauss.clone()(3.5, 1.0, 11)),
        Box::new(gauss.clone()(3.0, 1.0, 12)),
        Box::new(gauss.clone()(2.0, 1.0, 13)),
        Box::new(gauss.clone()(3.5, 1.0, 14)),
        Box::new(gauss.clone()(3.0, 1.0, 15)),
        Box::new(gauss.clone()(3.0, 1.0, 16)),
        Box::new(gauss.clone()(2.0, 1.0, 17)),
        Box::new(gauss.clone()(2.0, 1.0, 18)),
        Box::new(gauss.clone()(1.5, 1.0, 19)),
        Box::new(gauss.clone()(1.5, 1.0, 20)),
        Box::new(gauss.clone()(2.0, 1.0, 21)),
        Box::new(gauss.clone()(2.0, 1.0, 22)),
        Box::new(gauss.clone()(2.0, 1.0, 23)),
        Box::new(gauss.clone()(2.0, 1.0, 24)),
        Box::new(gauss.clone()(2.0, 1.0, 25)),
        Box::new(gauss.clone()(2.0, 1.0, 26)),
        Box::new(gauss.clone()(2.0, 1.0, 27)),
        Box::new(gauss.clone()(3.0, 1.0, 28)),
        Box::new(gauss.clone()(3.0, 1.0, 29)),
        Box::new(gauss.clone()(3.0, 1.0, 30)),
        Box::new(id.clone()),
    ]
}

fn design_matrix<const N: usize, const M: usize>(
    x: &Matrix<N, M, [f64; N * M], f64>,
    phi: &[Box<dyn Fn([f64; M]) -> f64>; M],
) -> (
    Matrix<N, M, [f64; N * M], f64>,
    Matrix<M, N, [f64; M * N], f64>,
) {
    let mut matrix = [0.0; N * M];
    for i in 0..N {
        let x_i = {
            let mut slice = [0.0; M];
            for j in 0..M {
                slice[j] = x[i * M + j];
            }
            slice
        };
        for j in 0..M {
            matrix[i * M + j] = phi[j](x_i.clone())
        }
    }
    (Matrix::new(matrix.clone()), Matrix::new(matrix).transpose())
}

#[test]
fn for_design_matrix() {
    let ((train_x, _), (_, _)) = data();
    let select = move |i: usize| move |e: [f64; 32]| e[i];
    let effective: [Box<dyn Fn([f64; 32]) -> f64>; 32] = [
        Box::new(select.clone()(0)),
        Box::new(select.clone()(1)),
        Box::new(select.clone()(2)),
        Box::new(select.clone()(3)),
        Box::new(select.clone()(4)),
        Box::new(select.clone()(5)),
        Box::new(select.clone()(6)),
        Box::new(select.clone()(7)),
        Box::new(select.clone()(8)),
        Box::new(select.clone()(9)),
        Box::new(select.clone()(10)),
        Box::new(select.clone()(11)),
        Box::new(select.clone()(12)),
        Box::new(select.clone()(13)),
        Box::new(select.clone()(14)),
        Box::new(select.clone()(15)),
        Box::new(select.clone()(16)),
        Box::new(select.clone()(17)),
        Box::new(select.clone()(18)),
        Box::new(select.clone()(19)),
        Box::new(select.clone()(20)),
        Box::new(select.clone()(21)),
        Box::new(select.clone()(22)),
        Box::new(select.clone()(23)),
        Box::new(select.clone()(24)),
        Box::new(select.clone()(25)),
        Box::new(select.clone()(26)),
        Box::new(select.clone()(27)),
        Box::new(select.clone()(28)),
        Box::new(select.clone()(29)),
        Box::new(select.clone()(30)),
        Box::new(select.clone()(31)),
    ];
    let (phi, phi_t) = design_matrix(&train_x, &effective);
    assert_eq!((train_x.clone(), train_x.transpose()), (phi, phi_t));
}

fn write_to(path: &str, content: String) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(&path)?;
    write!(file, "{}", content)?;
    file.flush()?;
    Ok(())
}

fn data() -> (
    (
        Matrix<75, 32, [f64; 75 * 32], f64>,
        Matrix<75, 1, [f64; 75 * 1], f64>,
    ),
    (
        Matrix<70, 32, [f64; 70 * 32], f64>,
        Matrix<70, 1, [f64; 70 * 1], f64>,
    ),
) {
    let data: [i32; 145 * 33] = include!("../data/rand.csv");

    let train_x = {
        let mut train_x = [0.0; 75 * 32];
        for i in 0..75 {
            for j in 1..32 {
                train_x[i * 32 + j - 1] = data[i * 33 + j] as f64;
            }
            // for bias
            train_x[i * 32 + 31] = 1.0;
        }
        Matrix::<75, 32, _, _>::new(train_x)
    };
    let test_x = {
        let mut test_x = [0.0; 70 * 32];
        for i in 75..145 {
            for j in 1..32 {
                test_x[(i - 75) * 32 + j - 1] = data[i * 33 + j] as f64;
            }
            // for bias
            test_x[(i - 75) * 32 + 31] = 1.0;
        }
        Matrix::<70, 32, _, _>::new(test_x)
    };

    let train_t = {
        let mut train_t = [0.0; 1 * 75];
        for i in 0..75 {
            train_t[i] = data[i * 33 + 32] as f64;
        }
        Matrix::<75, 1, _, _>::new(train_t)
    };
    let test_t = {
        let mut test_t = [0.0; 1 * 70];

        for i in 75..145 {
            test_t[i - 75] = data[i * 33 + 32] as f64;
        }
        Matrix::<70, 1, _, _>::new(test_t)
    };

    ((train_x, train_t), (test_x, test_t))
}
