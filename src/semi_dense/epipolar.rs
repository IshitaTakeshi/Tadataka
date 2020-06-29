use crate::vector::normalize;
use crate::projection::Projection;
use crate::transform::{get_rotation, get_translation, inv_transform};
use ndarray::{arr1, Array, Array1, Array2};
use ndarray_linalg::Norm;

static EPSILON: f64 = 1e-16;

pub fn calc_key_epipole(
    transform_wk: &Array2<f64>,
    transform_wr: &Array2<f64>,
) -> Array1<f64> {
    let t_wk = get_translation(transform_wk);
    let t_wr = get_translation(transform_wr);
    let transform_kw = inv_transform(transform_wk);
    let rot_kw = get_rotation(&transform_kw);
    let p_key = rot_kw.dot(&(&t_wr - &t_wk));
    let e_key = Projection::project(&p_key);
    e_key
}

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
    fn test_calc_key_epipole() {
        let transform_wk = arr2(
            &[[1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]]
        );

        let transform_wr = arr2(
            &[[1., 0., 0., 3.],
              [0., 1., 0., 3.],
              [0., 0., 1., 10.],
              [0., 0., 0., 1.]]
        );

        let e_key = calc_key_epipole(&transform_wk, &transform_wr);
        assert_eq!(e_key, arr1(&[0.3, 0.3]));

        // rotate pi / 2 around the y-axis
        let transform_wk = arr2(
            &[[0., 0., 1., 0.],
              [0., 1., 0., 0.],
              [-1., 0., 0., 6.],
              [0., 0., 0., 1.]]
        );

        // rotate - pi / 2 around the y-axis
        let transform_wr = arr2(
            &[[0., 0., -1., 6.],
              [0., 1., 0., 0.],
              [1., 0., 0., 3.],
              [0., 0., 0., 1.]]
        );

        let e_key = calc_key_epipole(&transform_wk, &transform_wr);
        assert_eq!(e_key, arr1(&[0.5, 0.]));

        // rotate pi around the y-axis
        let transform_wk = arr2(
            &[[-1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., -1., 4.],
              [0., 0., 0., 1.]]
        );

        // rotate - pi / 2 around the y-axis
        let transform_wr = arr2(
            &[[0., 0., -1., 4.],
              [0., 1., 0., 0.],
              [1., 0., 0., 2.],
              [0., 0., 0., 1.]]
        );

        let e_key = calc_key_epipole(&transform_wk, &transform_wr);
        assert_eq!(e_key, arr1(&[-2., 0.]));
    }

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
