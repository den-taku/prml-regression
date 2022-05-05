#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use dntk_matrix::matrix::*;

fn main() {
    // A + B in stack
    let left = Matrix::<2, 3, _, i32>::new([
        1, 2, 3, //
        4, 5, 6,
    ]);
    let right = Matrix::<2, 3, _, i32>::new([
        1, 2, 3, //
        4, 5, 6,
    ]);
    assert_eq!(
        left + right,
        Matrix::<2, 3, _, i32>::new([
            2, 4, 6, //
            8, 10, 12
        ])
    );

    let data = include!("../data/DATA.csv");
    let mut train_x = [0.0; 100 * 32];
    let mut test_x = [0.0; 45 * 32];
    for i in 0..100 {
        for j in 1..32 {
            train_x[i * 32 + j - 1] = data[i * 33 + j] as f64;
        }
        // for bias
        train_x[i * 32 + 31] = 1.0;
    }
    for i in 100..145 {
        for j in 1..32 {
            test_x[(i - 100) * 32 + j - 1] = data[i * 33 + j] as f64;
        }
        // for bias
        test_x[(i - 100) * 32 + 31] = 1.0;
    }
    let train_x = Matrix::<100, 32, _, f64>::new(train_x);
    // println!("{train_x}");
    let test_x = Matrix::<45, 32, _, f64>::new(test_x);
    println!("{test_x}");
    let mut train_t = [0.0; 1 * 100];
    let mut test_t = [0.0; 1 * 45];
    for i in 0..100 {
        train_t[i] = data[i * 33 + 32] as f64;
    }
    for i in 100..145 {
        test_t[i - 100] = data[i * 33 + 32] as f64;
    }
    let train_t = Matrix::<100, 1, _, f64>::new(train_t);
    // println!("{train_t}");
    let test_t = Matrix::<45, 1, _, f64>::new(test_t);
    println!("{test_t}");

    // normal
    // y(x, w) = w_0 + w_1*x_1 + ... + w_D * x_D
    // p(t|x, w, b) = N(t|y(x,w), b^-1)
    let phi = train_x.clone();
    let phi_t = phi.clone().transpose();
    let right = phi_t.clone() * phi;
    let left = phi_t * train_t.clone();
    let w = solve_eqn(right, left);
    println!("{w}");

    let inference = test_x.clone() * w;
    println!("{inference}");
    println!("{test_t}");
    println!(
        "test: {}%",
        (0..45)
            .map(|i| inference[i].round() as i64 == test_t[i] as i64)
            .count() as f64
            / 45.0
    );
}
