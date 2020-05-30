use ndarray::{arr1, arr2, Array, Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use crate::convolution::convolve2d;

pub fn sobel_x<S: Data<Elem = f64>>(
    map: &ArrayBase<S, Ix2>,
) -> Array2<f64> {
    let kernel = arr2(
        &[[1., 0., -1.],
          [2., 0., -2.],
          [1., 0., -1.]]
    );
    let pad_shape = (1, 1);
    convolve2d(map, &kernel, pad_shape)
}

pub fn sobel_y<S: Data<Elem = f64>>(
    map: &ArrayBase<S, Ix2>,
) -> Array2<f64> {
    let kernel = arr2(
        &[[1., 2., 1.],
          [0., 0., 0.],
          [-1., -2., -1.]]
    );
    let pad_shape = (1, 1);
    convolve2d(map, &kernel, pad_shape)
}

pub fn gradient1d<S: Data<Elem = f64>>(x: &ArrayBase<S, Ix1>) -> Array1<f64> {
    let n = x.shape()[0];
    let mut grad = Array::zeros(n - 1);
    for i in 0..n - 1 {
        grad[i] = x[i + 1] - x[i];
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobel_x() {
        let map = arr2(
            &[[1., 2., -1., 0.],
              [0., 0., -1., 1.],
              [3., -2., 0., -1.],
              [-2., 1., 1., 2.]],
        );

        let expected = arr2(
            &[[0., 0., 0., 0.],
              [0., 7., -1., 0.],
              [0., 4., -4., 0.],
              [0., 0., 0., 0.]]
        );

        assert_eq!(sobel_x(&map), expected);
    }

    #[test]
    fn test_sobel_y() {
        let map = arr2(
            &[[1., 2., -1., 0.],
              [0., 0., -1., 1.],
              [3., -2., 0., -1.],
              [-2., 1., 1., 2.]],
        );

        let expected = arr2(
            &[[0., 0., 0., 0.],
              [0., 5., 3., 0.],
              [0., -2., -6., 0.],
              [0., 0., 0., 0.]]
        );

        assert_eq!(sobel_y(&map), expected);
    }

    #[test]
    fn test_gradient1d() {
        let intensities = arr1(&[-1., 1., 0., 3., -2.]);
        let expected = arr1(&[1. - (-1.), 0. - 1., 3. - 0., -2. - 3.]);
        assert_eq!(gradient1d(&intensities), expected);
    }
}
