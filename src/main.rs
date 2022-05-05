#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use dntk_matrix::matrix::*;

fn main() {
    let data: [i32; 145 * 33] = {
        use rand::Rng;
        let mut data = include!("../data/DATA.csv");
        let mut rng = rand::thread_rng();
        for i in 0..145 {
            swap(i, rng.gen::<usize>() % 145, &mut data)
        }
        data
    };

    let train_x = {
        let mut train_x = [0.0; 75 * 32];
        for i in 0..75 {
            for j in 1..32 {
                train_x[i * 32 + j - 1] = data[i * 33 + j] as f64;
            }
            // for bias
            train_x[i * 32 + 31] = 1.0;
        }
        Matrix::<75, 32, _, f64>::new(train_x)
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
        Matrix::<70, 32, _, f64>::new(test_x)
    };

    let train_t = {
        let mut train_t = [0.0; 1 * 75];
        for i in 0..75 {
            train_t[i] = data[i * 33 + 32] as f64;
        }
        Matrix::<75, 1, _, f64>::new(train_t)
    };
    let test_t = {
        let mut test_t = [0.0; 1 * 70];

        for i in 75..145 {
            test_t[i - 75] = data[i * 33 + 32] as f64;
        }
        Matrix::<70, 1, _, f64>::new(test_t)
    };

    // normal
    // y(x, w) = w_0 + w_1*x_1 + ... + w_D * x_D
    // p(t|x, w, b) = N(t|y(x,w), b^-1)
    let phi = train_x.clone();
    let phi_t = phi.clone().transpose();

    let right = phi_t.clone() * phi;
    let left = phi_t * train_t.clone();
    let w = solve_eqn(right, left);

    let mut content = String::new();
    for i in 0..w.0.len() {
        content.push_str(&format!("{},\n", w[i]))
    }
    write_to("./parameters/simple.csv", content)
        .expect("fail to save weight to ./parameters/simple.csv");

    let inference = test_x.clone() * w;
    println!(
        "test: {} / 70",
        (0..70)
            .filter_map(|i| (inference[i].round() as i64 == test_t[i] as i64).then(|| 0))
            .count()
    );
}

fn write_to(path: &str, content: String) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(&path)?;
    write!(file, "{}", content)?;
    file.flush()?;
    Ok(())
}

fn swap(i: usize, j: usize, slice: &mut [i32]) {
    let mut tmp = [0; 33];
    for k in 0..33 {
        tmp[k] = slice[i * 33 + k];
        slice[i * 33 + k] = slice[j * 33 + k];
        slice[j * 33 + k] = tmp[k];
    }
}
