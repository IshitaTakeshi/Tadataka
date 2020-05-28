use ndarray::{arr2, Array2, ArrayBase, Data, Ix2};
use crate::convolution::convolve2d;

fn sobel_x<S: Data<Elem = f64>>(
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

fn sobel_y<S: Data<Elem = f64>>(
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
}
