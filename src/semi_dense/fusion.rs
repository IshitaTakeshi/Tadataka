use ndarray::{Array, Array2};

pub fn fusion<T, U>(mu1: T, mu2: T, var1: U, var2: U) -> (T, U)
where
    T: num::Float + From<U>,
    U: num::Float {
    let v: U = var1 + var2;
    let mu = (mu1 * From::from(var2) + mu2 * From::from(var1)) / From::from(v);
    let var = (var1 * var2) / v;
    (mu, var)
}

pub fn fusion_arrays<T, U>(
    mu1: &Array2<T>,
    mu2: &Array2<T>,
    var1: &Array2<U>,
    var2: &Array2<U>,
) -> (Array2<T>, Array2<U>)
where
    T: num::Float + From<U>,
    U: num::Float {
    let shape = mu1.shape();
    let (height, width) = (shape[0], shape[1]);
    assert_eq!(mu2.shape(), shape);
    assert_eq!(var1.shape(), shape);
    assert_eq!(var2.shape(), shape);

    let mut mu = Array::zeros((height, width));
    let mut var = Array::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let m1: T = mu1[[y, x]];
            let m2: T = mu2[[y, x]];
            let v1: U = var1[[y, x]];
            let v2: U = var2[[y, x]];
            let (m, v): (T, U) = fusion(m1, m2, v1, v2);
            mu[[y, x]] = m;
            var[[y, x]] = v;
        }
    }
    (mu, var)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_fusion() {
        let mu1 = arr2(
            &[[1.9, -2.2],
              [-3.8, 4.1],
              [-1.5, 4.5]]
        );
        let mu2 = arr2(
            &[[-4.1, -2.5],
              [1.2, 5.0],
              [6.4, 4.1]]
        );

        let var1 = arr2(
            &[[4.8, 2.2],
              [3.1, 6.8],
              [4.0, 2.1]]
        );
        let var2 = arr2(
            &[[4.2, 3.1],
              [0.01, 2.0],
              [6.0, 3.9]]
        );

        let (mu0, var0) = fusion_arrays(&mu1, &mu2, &var1, &var2);

        for y in 0..3 {
            for x in 0..2 {
                let m0 = mu0[[y, x]];
                let m1 = mu1[[y, x]];
                let m2 = mu2[[y, x]];
                let v0 = var0[[y, x]];
                let v1 = var1[[y, x]];
                let v2 = var2[[y, x]];

                assert_eq!(m0, (v2 * m1 + v1 * m2) / (v1 + v2));
                assert_eq!(v0, (v1 * v2) / (v1 + v2));
            }
        }
    }
}
