use crate::vector::normalize;
use ndarray::{arr1, Array, Array1, Array2};
use ndarray_linalg::Norm;

static EPSILON: f64 = 1e-16;

pub fn key_coordinates(
    epipolar_direction: &Array1<f64>,
    x_key: &Array1<f64>,
    step_size: f64,
) -> Array2<f64> {
    let sampling_steps = arr1(&[-2., -1., 0., 1., 2.]);
    let direction = normalize(&epipolar_direction);
    let n = sampling_steps.shape()[0];
    let mut coordinates = Array::zeros((n, 2));
    for (i, step) in sampling_steps.iter().enumerate() {
        let c = x_key + &(step_size * step * &direction);
        coordinates.row_mut(i).assign(&c);
    }
    coordinates
}

pub fn ref_coordinates(
    x_min: &Array1<f64>,
    x_max: &Array1<f64>,
    step_size: f64,
) -> Array2<f64> {
    let direction = x_max - x_min;
    let norm = direction.norm();
    let direction = direction / (norm + EPSILON);

    let n = (norm / step_size) as usize;

    let mut xs = Array::zeros((n, 2));
    for i in 0..n {
        let x = x_min + &((i as f64) * step_size * &direction);
        xs.row_mut(i).assign(&x);
    }
    xs
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_key_coordinates() {
        let x_key = arr1(&[7., 8.]);
        let direction = arr1(&[9., 12.]);
        let step_size = 5.;

        let expected = arr2(&[
            [7. - 2. * 3., 8. - 2. * 4.],
            [7. - 1. * 3., 8. - 1. * 4.],
            [7. - 0. * 3., 8. - 0. * 4.],
            [7. + 1. * 3., 8. + 1. * 4.],
            [7. + 2. * 3., 8. + 2. * 4.],
        ]);
        assert_eq!(
            key_coordinates(&direction, &x_key, step_size),
            expected
        );
    }

    #[test]
    fn test_ref_coordinates() {
        let search_step = 5.0;
        let x_min = arr1(&[-15.0, -20.0]);
        let x_max = arr1(&[15.0, 20.0]);
        let xs = ref_coordinates(&x_min, &x_max, search_step);

        let xs_true = arr2(&[
            [-15., -20.],
            [-12., -16.],
            [-9., -12.],
            [-6., -8.],
            [-3., -4.],
            [0., 0.],
            [3., 4.],
            [6., 8.],
            [9., 12.],
            [12., 16.],
        ]);
        assert_eq!(xs, xs_true)
    }
}
