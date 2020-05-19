extern crate test;
use crate::vector::normalize;
use ndarray::{arr1, arr2, Array, Array2, ArrayView1};
use ndarray_linalg::Norm;

static EPSILON: f64 = 1e-16;

fn key_coordinates(
    epipolar_direction: ArrayView1<'_, f64>,
    x_key: ArrayView1<'_, f64>,
    step_size: f64,
) -> Array2<f64> {
    let sampling_steps = arr1(&[-2., -1., 0., 1., 2.]);
    let direction = normalize(epipolar_direction.view());
    let n = sampling_steps.shape()[0];
    let mut coordinates = Array::zeros((n, 2));
    for (i, step) in sampling_steps.iter().enumerate() {
        let c = x_key.to_owned() + step_size * step * direction.to_owned();
        coordinates.row_mut(i).assign(&c);
    }
    coordinates
}

fn ref_coordinates(
    x_min: ArrayView1<'_, f64>,
    x_max: ArrayView1<f64>,
    step_size: f64,
) -> Array2<f64> {
    let direction = x_max.to_owned() - x_min;
    let norm = direction.norm();
    let direction = direction / (norm + EPSILON);

    let n = (norm / step_size) as usize;

    let mut xs = Array::zeros((n, 2));
    for i in 0..n {
        let x = x_min.to_owned() + (i as f64) * step_size * direction.to_owned();
        xs.row_mut(i).assign(&x);
    }
    xs
}

#[cfg(test)]
mod tests {
    use super::*;

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
            key_coordinates(direction.view(), x_key.view(), step_size),
            expected
        );
    }

    #[test]
    fn test_ref_coordinates() {
        let search_step = 5.0;
        let x_min = arr1(&[-15.0, -20.0]);
        let x_max = arr1(&[15.0, 20.0]);
        let xs = ref_coordinates(x_min.view(), x_max.view(), search_step);

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
