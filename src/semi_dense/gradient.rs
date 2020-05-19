use ndarray::{arr1, Array, Array1, ArrayView1};

fn gradient(intensities: ArrayView1<'_, f64>) -> Array1<f64> {
    let n = intensities.shape()[0];
    let mut grad = Array::zeros(n - 1);
    for i in 0..n - 1 {
        grad[i] = intensities[i + 1] - intensities[i];
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient() {
        let intensities = arr1(&[-1., 1., 0., 3., -2.]);
        let expected = arr1(&[1. - (-1.), 0. - 1., 3. - 0., -2. - 3.]);
        assert_eq!(gradient(intensities.view()), expected);
    }
}
