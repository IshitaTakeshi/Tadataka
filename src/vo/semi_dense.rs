extern crate test;

use crate::projection::inv_project_vec;
use crate::vector::normalize;
use ndarray::{arr1, arr2, Array, Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::Norm;

static EPSILON: f64 = 1e-16;

pub fn calc_ref_depth(
    transform_rk: ArrayView2<'_, f64>,
    x_key: ArrayView1<'_, f64>,
    depth_key: f64,
) -> f64 {
    let p_key = inv_project_vec(x_key, depth_key);
    let r_rk_z: ArrayView1<'_, f64> = transform_rk.slice(s![2, 0..3]);
    let t_rk_z: f64 = transform_rk[[2, 3]];
    return r_rk_z.dot(&p_key) + t_rk_z;
}

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
    step_size: f64
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
    return xs;
}

fn gradient(intensities: ArrayView1<'_, f64>) -> Array1<f64> {
    let n = intensities.shape()[0];
    let mut grad = Array::zeros(n - 1);
    for i in 0..n - 1 {
        grad[i] = intensities[i + 1] - intensities[i];
    }
    grad
}

fn calc_error(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    let d = a.to_owned() - b;
    d.dot(&d)
}

fn search(sequence: ArrayView1<'_, f64>, kernel: ArrayView1<'_, f64>) -> usize {
    let n = sequence.shape()[0];
    let k = kernel.shape()[0];

    let mut min_error = f64::INFINITY;
    let mut argmin = 0;
    for i in 0..n - k + 1 {
        let e = calc_error(sequence.slice(s![i..i + k]), kernel);
        println!("error = {}", e);
        if e < min_error {
            min_error = e;
            argmin = i;
        }
    }

    argmin as usize
}

fn search_intensities(sequence: ArrayView1<'_, f64>, kernel: ArrayView1<'_, f64>) -> usize {
    let argmin = search(sequence, kernel);
    println!("argmin = {}", argmin);
    let k = kernel.shape()[0];
    let offset = (k / 2) as usize;
    return argmin + offset;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_inv_depth() {
        let depth_key = 4.0;
        let x_key = arr1(&[0.5, 2.0]);
        let transform_rk = arr2(&[
            [0., 0., 1., 3.],
            [0., 1., 0., 2.],
            [-1., 0., 0., 4.],
            [0., 0., 0., 1.],
        ]);

        assert_eq!(
            calc_ref_depth(transform_rk.view(), x_key.view(), depth_key),
            -2.0 + 4.0
        );
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
            [7. + 2. * 3., 8. + 2. * 4.]
        ]);
        assert_eq!(key_coordinates(direction.view(), x_key.view(), step_size),
                   expected);
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
            [ -9., -12.],
            [ -6.,  -8.],
            [ -3.,  -4.],
            [  0.,   0.],
            [  3.,   4.],
            [  6.,   8.],
            [  9.,  12.],
            [ 12.,  16.]
        ]);
        assert_eq!(xs, xs_true)
    }

    #[test]
    fn test_intensity_search() {
        // errors = [25 + 16 + 0, 4 + 9 + 4, 1 + 25 + 9, 9 + 0 + 1, 4 + 16 + 1]
        //        = [41, 17, 35, 10, 21]
        // argmin(errors) == 3
        // offset == 1 ( == len(intensities_key) // 2)
        // expected = argmin(errors) + offset == 4
        let intensities_ref = arr1(&[-4., 3., 2., 4., -1., 3., 1.]);
        let intensities_key = arr1(&[1., -1., 2.]);
        let index = search_intensities(intensities_ref.view(), intensities_key.view());
        assert_eq!(index, 4);

        // argmin(errors) == 2
        // offset == 1 ( == len(intensities_key) // 2)
        // expected = argmin(errors) + offset == 3
        let intensities_ref = arr1(&[-4., 3., 1., -1.]);
        let intensities_key = arr1(&[1., -1.]);
        let index = search_intensities(intensities_ref.view(), intensities_key.view());
        assert_eq!(index, 3);

        // argmin(errors) == 0
        // offset == 1 ( == len(intensities_key) // 2)
        // expected = argmin(errors) + offset == 1
        let intensities_ref = arr1(&[1., -1., -4., 3.]);
        let intensities_key = arr1(&[1., -1.]);
        let index = search_intensities(intensities_ref.view(), intensities_key.view());
        assert_eq!(index, 1);
    }

    #[test]
    fn test_gradient() {
        let intensities = arr1(&[-1., 1., 0., 3., -2.]);
        let expected = arr1(&[1. - (-1.), 0. - 1., 3. - 0., -2. - 3.]);
        assert_eq!(gradient(intensities.view()), expected);
    }
}
