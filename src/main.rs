#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use dntk_matrix::matrix::*;

fn main() {
    const TRAIN_CASE: usize = 115;
    const TEST_CASE: usize = 30;
    let ((train_x, train_t), (test_x, test_t)) = data::<TRAIN_CASE, TEST_CASE>();

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

    let inference = test_x.clone() * w.clone();
    println!(
        "test: {} / {TEST_CASE}\nerr: {}",
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0,
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 02
    // y(x, w) = w_0 + w_1*x_1 + ... + w_D * x_D
    // regularization: lambda = 0.5
    let lambda = 0.5;
    let lambda_i = {
        let mut o = [0.0; 32 * 32];
        for i in 0..31 {
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

    let inference = test_x.clone() * w.clone();
    println!(
        "test: {} / {TEST_CASE}\nerr: {}",
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0
            + lambda / 2.0 * w.clone().0.iter().map(|e| e * e).sum::<f64>(),
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 03
    // y(x, w) = w_0 + w_1*x_1 + ... + w_D * x_D
    // regularization: lambda = 1.0
    let lambda = 1.0;
    let lambda_i = {
        let mut o = [0.0; 32 * 32];
        for i in 0..31 {
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

    let inference = test_x.clone() * w.clone();
    println!(
        "test: {} / {TEST_CASE}\nerr: {}",
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0
            + lambda / 2.0 * w.0.iter().map(|e| e * e).sum::<f64>(),
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 04
    // y(x, w) = w_0 + w_1*x_1 + ... + w_D * x_D
    // regularization: lambda = 10.0
    let lambda = 10.0;
    let lambda_i = {
        let mut o = [0.0; 32 * 32];
        for i in 0..31 {
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

    let inference = test_x.clone() * w.clone();
    println!(
        "test: {} / {TEST_CASE}\nerr: {}",
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0
            + lambda / 2.0 * w.0.iter().map(|e| e * e).sum::<f64>(),
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

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
        "test: {} / {TEST_CASE}\nerr: {}",
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0,
    );

    // 06
    // select effective parameters
    let compress_train_x = compress_x(train_x.clone());
    let (phi, phi_t) = (compress_train_x.clone(), compress_train_x.transpose());

    let left = phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/weight06_select.csv", content)
        .expect("fail to save weight to ./parameters/weight06_select.csv");

    let inference = compress_x(test_x.clone()) * w;
    println!(
        "test: {} / {TEST_CASE}\nerr: {}",
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0,
    );

    // 07
    // select effective parameters
    // regularization: lambda = 10.0
    let lambda = 1.0;
    let lambda_i = {
        let mut o = [0.0; 20 * 20];
        for i in 0..19 {
            o[i * 19 + i] = lambda;
        }
        Matrix::<20, 20, _, _>::new(o)
    };

    let left = lambda_i + phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/weight07_select_reg.csv", content)
        .expect("fail to save weight to ./parameters/weight07_select_ref.csv");

    let inference = compress_x(test_x.clone()) * w.clone();
    println!(
        "test: {} / {TEST_CASE}\nerr: {}",
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0
            + lambda / 2.0 * w.0.iter().map(|e| e * e).sum::<f64>(),
    );

    // 08
    // select effective parameters
    // regularization: lambda = 10.0
    let lambda = 10.0;
    let lambda_i = {
        let mut o = [0.0; 20 * 20];
        for i in 0..19 {
            o[i * 19 + i] = lambda;
        }
        Matrix::<20, 20, _, _>::new(o)
    };

    let left = lambda_i + phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/weight08_select_reg.csv", content)
        .expect("fail to save weight to ./parameters/weight08_select_ref.csv");

    let inference = compress_x(test_x.clone()) * w.clone();
    println!(
        "test: {} / {TEST_CASE}\nerr: {}",
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0
            + lambda / 2.0 * w.0.iter().map(|e| e * e).sum::<f64>(),
    );

    // 09
    let count = 09;
    // select effective parameters inspired by w of 01 ~ 02
    let compress_train_x = compress_x_weight(train_x.clone());
    let (phi, phi_t) = (compress_train_x.clone(), compress_train_x.transpose());

    let left = phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    let file = "weight09_select_weight.csv";
    write_to(&format!("./parameters/{}", file), content)
        .expect(&format!("fail to save weight to ./parameters/{}", file));

    let inference = compress_x_weight(test_x.clone()) * w.clone();
    println!(
        "{}: test: {} / {TEST_CASE}\nerr: {}",
        count,
        (0..test_t.0.len())
            .map(|i| {
                // println!("{i}: t = {} but y = {}", test_t[i], inference[i]);
                i
            })
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0,
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 10
    let count = 10;
    // select effective parameters inspired by w of 01 ~ 02
    // regularization: lambda = 1.0
    let compress_train_x = compress_x_weight(train_x.clone());
    let (phi, phi_t) = (compress_train_x.clone(), compress_train_x.transpose());

    let lambda = 1.0;
    let lambda_i = {
        let mut o = [0.0; 10 * 10];
        for i in 0..10 {
            o[i * 10 + i] = lambda;
        }
        Matrix::<10, 10, _, _>::new(o)
    };

    let left = lambda_i + phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    let file = "weight10_select_weight_reg.csv";
    write_to(&format!("./parameters/{}", file), content)
        .expect(&format!("fail to save weight to ./parameters/{}", file));

    let inference = compress_x_weight(test_x.clone()) * w.clone();
    println!(
        "{}: test: {} / {TEST_CASE}\nerr: {}",
        count,
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0
            + lambda / 2.0 * w.0.iter().map(|e| e * e).sum::<f64>(),
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 11
    let count = 11;
    // select effective parameters inspired by w of 01 ~ 02
    // regularization: lambda = 5.0
    let compress_train_x = compress_x_weight(train_x.clone());
    let (phi, phi_t) = (compress_train_x.clone(), compress_train_x.transpose());

    let lambda = 5.0;
    let lambda_i = {
        let mut o = [0.0; 10 * 10];
        for i in 0..10 {
            o[i * 10 + i] = lambda;
        }
        Matrix::<10, 10, _, _>::new(o)
    };

    let left = lambda_i + phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    let file = "weight11_select_weight_reg.csv";
    write_to(&format!("./parameters/{}", file), content)
        .expect(&format!("fail to save weight to ./parameters/{}", file));

    let inference = compress_x_weight(test_x.clone()) * w.clone();
    println!(
        "{}: test: {} / {TEST_CASE}\nerr: {}",
        count,
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0
            + lambda / 2.0 * w.0.iter().map(|e| e * e).sum::<f64>(),
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 12
    let count = 12;
    // select effective parameters inspired by w of 01 ~ 02
    // regularization: lambda = 0.5
    let compress_train_x = compress_x_weight(train_x.clone());
    let (phi, phi_t) = (compress_train_x.clone(), compress_train_x.transpose());

    let lambda = 0.5;
    let lambda_i = {
        let mut o = [0.0; 10 * 10];
        for i in 0..10 {
            o[i * 10 + i] = lambda;
        }
        Matrix::<10, 10, _, _>::new(o)
    };

    let left = lambda_i + phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    let file = "weight12_select_weight_reg.csv";
    write_to(&format!("./parameters/{}", file), content)
        .expect(&format!("fail to save weight to ./parameters/{}", file));

    let inference = compress_x_weight(test_x.clone()) * w.clone();
    println!(
        "{}: test: {} / {TEST_CASE}\nerr: {}",
        count,
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0
            + lambda / 2.0 * w.0.iter().map(|e| e * e).sum::<f64>(),
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 13
    let count = 13;
    // select effective parameters inspired by w of 10 ~ 12
    let compress_train_x = compress_x_weight2(train_x.clone());
    let (phi, phi_t) = (compress_train_x.clone(), compress_train_x.transpose());

    let left = phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    let file = "weight13_select_weight2_reg.csv";
    write_to(&format!("./parameters/{}", file), content)
        .expect(&format!("fail to save weight to ./parameters/{}", file));

    let inference = compress_x_weight2(test_x.clone()) * w.clone();
    println!(
        "{}: test: {} / {TEST_CASE}\nerr: {}",
        count,
        (0..test_t.0.len())
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0,
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 14
    let count = 14;
    // select effective parameters inspired by w of 01 ~ 02
    let (phi, phi_t) = design_matrix(&compress_x_weight(train_x.clone()), &basis());
    // let (phi, phi_t) = (compress_train_x.clone(), compress_train_x.transpose());

    let left = phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    let file = "weight14_select_weight_gauss.csv";
    write_to(&format!("./parameters/{}", file), content)
        .expect(&format!("fail to save weight to ./parameters/{}", file));

    let inference = design_matrix(&compress_x_weight(test_x.clone()), &basis()).0 * w.clone();
    println!(
        "{}: test: {} / {TEST_CASE}\nerr: {}",
        count,
        (0..test_t.0.len())
            .map(|i| {
                // println!("{i}: t = {} but y = {}", test_t[i], inference[i]);
                i
            })
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0,
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 15
    let count = 15;
    // polynomial fitting
    let (phi, phi_t) = design_matrix(&train_x.clone(), &basis_all());
    // let (phi, phi_t) = (compress_train_x.clone(), compress_train_x.transpose());

    let left = phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    let file = &format!("weight{count}_poly.csv");
    write_to(&format!("./parameters/{}", file), content)
        .expect(&format!("fail to save weight to ./parameters/{}", file));

    let inference = design_matrix(&test_x.clone(), &basis_all()).0 * w.clone();
    println!(
        "{}: test: {} / {TEST_CASE}\nerr: {}",
        count,
        (0..test_t.0.len())
            .map(|i| {
                // println!("{i}: t = {} but y = {}", test_t[i], inference[i]);
                i
            })
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0,
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);

    // 16
    let count = 16;
    // polynomial fitting with regularization
    let (phi, phi_t) = design_matrix(&train_x.clone(), &basis_all());
    // let (phi, phi_t) = (compress_train_x.clone(), compress_train_x.transpose());

    let lambda = 0.5;
    let lambda_i = {
        const SIZE: usize = 63;
        let mut o = [0.0; SIZE * SIZE];
        for i in 0..SIZE {
            o[i * SIZE + i] = lambda;
        }
        Matrix::<SIZE, SIZE, _, _>::new(o)
    };

    let left = lambda_i + phi_t.clone() * phi.clone();
    let right = phi_t.clone() * train_t.clone();
    let w = solve_eqn(left, right);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    let file = &format!("weight{count}_poly_reg.csv");
    write_to(&format!("./parameters/{}", file), content)
        .expect(&format!("fail to save weight to ./parameters/{}", file));

    let inference = design_matrix(&test_x.clone(), &basis_all()).0 * w.clone();
    println!(
        "{}: test: {} / {TEST_CASE}\nerr: {}",
        count,
        (0..test_t.0.len())
            .map(|i| {
                // println!("{i}: t = {} but y = {}", test_t[i], inference[i]);
                i
            })
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count(),
        (0..test_t.0.len())
            .map(|i| (inference[i] - test_t[i]) * (inference[i] - test_t[i]))
            .sum::<f64>()
            / 2.0
            + lambda / 2.0 * w.0.iter().map(|e| e * e).sum::<f64>(),
    );
    let mut order =
        w.0.iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i + 1))
            .collect::<Vec<_>>();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // dbg!("{:#?}", order);
}

fn basis_all() -> [Box<dyn Fn([f64; 32]) -> f64>; 63] {
    let gauss = move |u: f64, s: f64, i: usize| {
        move |e: [f64; 10]| f64::exp(-(e[i] - u) * (e[i] - u) / 2.0 / (s * s))
    };
    [
        // Box::new(gauss.clone()(2.0, 1.0, 0)),
        // Box::new(gauss.clone()(1.5, 1.0, 1)),
        // Box::new(gauss.clone()(2.0, 1.0, 2)),
        // Box::new(gauss.clone()(3.0, 1.0, 3)),
        // Box::new(gauss.clone()(1.5, 1.0, 4)),
        // Box::new(gauss.clone()(1.5, 1.0, 5)),
        // Box::new(gauss.clone()(1.5, 1.0, 6)),
        // Box::new(gauss.clone()(3.0, 1.0, 7)),
        // Box::new(gauss.clone()(2.5, 1.0, 8)),
        Box::new(|e: [f64; 32]| e[0]),
        Box::new(|e: [f64; 32]| e[1]),
        Box::new(|e: [f64; 32]| e[2]),
        Box::new(|e: [f64; 32]| e[3]),
        Box::new(|e: [f64; 32]| e[4]),
        Box::new(|e: [f64; 32]| e[5]),
        Box::new(|e: [f64; 32]| e[6]),
        Box::new(|e: [f64; 32]| e[7]),
        Box::new(|e: [f64; 32]| e[8]),
        Box::new(|e: [f64; 32]| e[9]),
        Box::new(|e: [f64; 32]| e[10]),
        Box::new(|e: [f64; 32]| e[11]),
        Box::new(|e: [f64; 32]| e[12]),
        Box::new(|e: [f64; 32]| e[13]),
        Box::new(|e: [f64; 32]| e[14]),
        Box::new(|e: [f64; 32]| e[15]),
        Box::new(|e: [f64; 32]| e[16]),
        Box::new(|e: [f64; 32]| e[17]),
        Box::new(|e: [f64; 32]| e[18]),
        Box::new(|e: [f64; 32]| e[19]),
        Box::new(|e: [f64; 32]| e[20]),
        Box::new(|e: [f64; 32]| e[21]),
        Box::new(|e: [f64; 32]| e[22]),
        Box::new(|e: [f64; 32]| e[23]),
        Box::new(|e: [f64; 32]| e[24]),
        Box::new(|e: [f64; 32]| e[25]),
        Box::new(|e: [f64; 32]| e[26]),
        Box::new(|e: [f64; 32]| e[27]),
        Box::new(|e: [f64; 32]| e[28]),
        Box::new(|e: [f64; 32]| e[29]),
        Box::new(|e: [f64; 32]| e[30]),
        Box::new(|e: [f64; 32]| e[31]),
        Box::new(|e: [f64; 32]| e[0] * e[0]),
        Box::new(|e: [f64; 32]| e[1] * e[1]),
        Box::new(|e: [f64; 32]| e[2] * e[2]),
        Box::new(|e: [f64; 32]| e[3] * e[3]),
        Box::new(|e: [f64; 32]| e[4] * e[4]),
        Box::new(|e: [f64; 32]| e[5] * e[5]),
        Box::new(|e: [f64; 32]| e[6] * e[6]),
        Box::new(|e: [f64; 32]| e[7] * e[7]),
        Box::new(|e: [f64; 32]| e[8] * e[8]),
        Box::new(|e: [f64; 32]| e[9] * e[9]),
        Box::new(|e: [f64; 32]| e[10] * e[10]),
        Box::new(|e: [f64; 32]| e[11] * e[11]),
        Box::new(|e: [f64; 32]| e[12] * e[12]),
        Box::new(|e: [f64; 32]| e[13] * e[13]),
        Box::new(|e: [f64; 32]| e[14] * e[14]),
        Box::new(|e: [f64; 32]| e[15] * e[15]),
        Box::new(|e: [f64; 32]| e[16] * e[16]),
        Box::new(|e: [f64; 32]| e[17] * e[17]),
        Box::new(|e: [f64; 32]| e[18] * e[18]),
        Box::new(|e: [f64; 32]| e[19] * e[19]),
        Box::new(|e: [f64; 32]| e[20] * e[20]),
        Box::new(|e: [f64; 32]| e[21] * e[21]),
        Box::new(|e: [f64; 32]| e[22] * e[22]),
        Box::new(|e: [f64; 32]| e[23] * e[23]),
        Box::new(|e: [f64; 32]| e[24] * e[24]),
        Box::new(|e: [f64; 32]| e[25] * e[25]),
        Box::new(|e: [f64; 32]| e[26] * e[26]),
        Box::new(|e: [f64; 32]| e[27] * e[27]),
        Box::new(|e: [f64; 32]| e[28] * e[28]),
        Box::new(|e: [f64; 32]| e[29] * e[29]),
        Box::new(|e: [f64; 32]| e[30] * e[30]),
    ]
}

fn basis() -> [Box<dyn Fn([f64; 10]) -> f64>; 19] {
    let _gauss = move |u: f64, s: f64, i: usize| {
        move |e: [f64; 10]| f64::exp(-(e[i] - u) * (e[i] - u) / 2.0 / (s * s))
    };
    [
        // Box::new(gauss.clone()(2.0, 1.0, 0)),
        // Box::new(gauss.clone()(1.5, 1.0, 1)),
        // Box::new(gauss.clone()(2.0, 1.0, 2)),
        // Box::new(gauss.clone()(3.0, 1.0, 3)),
        // Box::new(gauss.clone()(1.5, 1.0, 4)),
        // Box::new(gauss.clone()(1.5, 1.0, 5)),
        // Box::new(gauss.clone()(1.5, 1.0, 6)),
        // Box::new(gauss.clone()(3.0, 1.0, 7)),
        // Box::new(gauss.clone()(2.5, 1.0, 8)),
        Box::new(|e: [f64; 10]| e[0]),
        Box::new(|e: [f64; 10]| e[1]),
        Box::new(|e: [f64; 10]| e[2]),
        Box::new(|e: [f64; 10]| e[3]),
        Box::new(|e: [f64; 10]| e[4]),
        Box::new(|e: [f64; 10]| e[5]),
        Box::new(|e: [f64; 10]| e[6]),
        Box::new(|e: [f64; 10]| e[7]),
        Box::new(|e: [f64; 10]| e[8]),
        Box::new(|e: [f64; 10]| e[9]),
        Box::new(|e: [f64; 10]| e[0] * e[0]),
        Box::new(|e: [f64; 10]| e[1] * e[1]),
        Box::new(|e: [f64; 10]| e[2] * e[2]),
        Box::new(|e: [f64; 10]| e[3] * e[3]),
        Box::new(|e: [f64; 10]| e[4] * e[4]),
        Box::new(|e: [f64; 10]| e[5] * e[5]),
        Box::new(|e: [f64; 10]| e[6] * e[6]),
        Box::new(|e: [f64; 10]| e[7] * e[7]),
        Box::new(|e: [f64; 10]| e[8] * e[8]),
    ]
}

fn compress_x_weight2<const N: usize>(
    x: Matrix<N, 32, [f64; N * 32], f64>,
) -> Matrix<N, 6, [f64; N * 6], f64> {
    let mut matrix = [0.0; N * 6];
    for i in 0..N {
        matrix[i * 6] = x[i * 32 + 1];
        matrix[i * 6 + 1] = x[i * 32 + 4];
        matrix[i * 6 + 2] = x[i * 32 + 9];
        matrix[i * 6 + 3] = x[i * 32 + 17];
        matrix[i * 6 + 4] = x[i * 32 + 19];
        matrix[i * 6 + 5] = x[i * 32 + 31];
    }
    Matrix::new(matrix)
}

fn compress_x_weight<const N: usize>(
    x: Matrix<N, 32, [f64; N * 32], f64>,
) -> Matrix<N, 10, [f64; N * 10], f64> {
    let mut matrix = [0.0; N * 10];
    for i in 0..N {
        matrix[i * 10] = x[i * 32 + 1];
        matrix[i * 10 + 1] = x[i * 32 + 2];
        matrix[i * 10 + 2] = x[i * 32 + 4];
        matrix[i * 10 + 3] = x[i * 32 + 8];
        matrix[i * 10 + 4] = x[i * 32 + 9];
        matrix[i * 10 + 5] = x[i * 32 + 13];
        matrix[i * 10 + 6] = x[i * 32 + 17];
        matrix[i * 10 + 7] = x[i * 32 + 19];
        matrix[i * 10 + 8] = x[i * 32 + 28];
        matrix[i * 10 + 9] = x[i * 32 + 31];
    }
    Matrix::new(matrix)
}

fn compress_x<const N: usize>(
    x: Matrix<N, 32, [f64; N * 32], f64>,
) -> Matrix<N, 20, [f64; N * 20], f64> {
    let mut matrix = [0.0; N * 20];
    for i in 0..N {
        matrix[i * 20] = x[i * 32 + 4];
        matrix[i * 20 + 1] = x[i * 32 + 5];
        matrix[i * 20 + 2] = x[i * 32 + 6];
        matrix[i * 20 + 3] = x[i * 32 + 10];
        matrix[i * 20 + 4] = x[i * 32 + 11];
        matrix[i * 20 + 5] = x[i * 32 + 16];
        matrix[i * 20 + 6] = x[i * 32 + 17];
        matrix[i * 20 + 7] = x[i * 32 + 18];
        matrix[i * 20 + 8] = x[i * 32 + 19];
        matrix[i * 20 + 9] = x[i * 32 + 21];
        matrix[i * 20 + 10] = x[i * 32 + 22];
        matrix[i * 20 + 11] = x[i * 32 + 23];
        matrix[i * 20 + 12] = x[i * 32 + 24];
        matrix[i * 20 + 13] = x[i * 32 + 25];
        matrix[i * 20 + 14] = x[i * 32 + 26];
        matrix[i * 20 + 15] = x[i * 32 + 27];
        matrix[i * 20 + 16] = x[i * 32 + 28];
        matrix[i * 20 + 17] = x[i * 32 + 29];
        matrix[i * 20 + 18] = x[i * 32 + 30];
        matrix[i * 20 + 19] = x[i * 32 + 31];
    }
    Matrix::new(matrix)
}

fn _select_parameters() -> [Box<dyn Fn([f64; 32]) -> f64>; 32] {
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

fn design_matrix<const N: usize, const M: usize, const L: usize>(
    x: &Matrix<N, M, [f64; N * M], f64>,
    phi: &[Box<dyn Fn([f64; M]) -> f64>; L],
) -> (
    Matrix<N, L, [f64; N * L], f64>,
    Matrix<L, N, [f64; L * N], f64>,
) {
    let mut matrix = [0.0; N * L];
    for i in 0..N {
        let x_i = {
            let mut slice = [0.0; M];
            for j in 0..M {
                slice[j] = x[i * M + j];
            }
            slice
        };
        for j in 0..L {
            matrix[i * L + j] = phi[j](x_i.clone())
        }
    }
    (Matrix::new(matrix.clone()), Matrix::new(matrix).transpose())
}

#[test]
fn for_design_matrix() {
    let ((train_x, _), (_, _)) = data::<100, 45>();
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

fn data<const TRAIN: usize, const TEST: usize>() -> (
    (
        Matrix<TRAIN, 32, [f64; TRAIN * 32], f64>,
        Matrix<TRAIN, 1, [f64; TRAIN * 1], f64>,
    ),
    (
        Matrix<TEST, 32, [f64; TEST * 32], f64>,
        Matrix<TEST, 1, [f64; TEST * 1], f64>,
    ),
) {
    let data: [i32; 145 * 33] = include!("../data/rand.csv");

    let train_x = {
        let mut train_x = [0.0; TRAIN * 32];
        for i in 0..TRAIN {
            for j in 1..32 {
                train_x[i * 32 + j - 1] = data[i * 33 + j] as f64;
            }
            // for bias
            train_x[i * 32 + 31] = 1.0;
        }
        Matrix::<TRAIN, 32, _, _>::new(train_x)
    };
    let test_x = {
        let mut test_x = [0.0; TEST * 32];
        for i in TRAIN..TRAIN + TEST {
            for j in 1..32 {
                test_x[(i - TRAIN) * 32 + j - 1] = data[i * 33 + j] as f64;
            }
            // for bias
            test_x[(i - TRAIN) * 32 + 31] = 1.0;
        }
        Matrix::<TEST, 32, _, _>::new(test_x)
    };

    let train_t = {
        let mut train_t = [0.0; TRAIN * 1];
        for i in 0..TRAIN {
            train_t[i] = data[i * 33 + 32] as f64;
        }
        Matrix::<TRAIN, 1, _, _>::new(train_t)
    };
    let test_t = {
        let mut test_t = [0.0; TEST * 1];

        for i in TRAIN..TRAIN + TEST {
            test_t[i - TRAIN] = data[i * 33 + 32] as f64;
        }
        Matrix::<TEST, 1, _, _>::new(test_t)
    };

    ((train_x, train_t), (test_x, test_t))
}
