use ndarray::{arr1, arr2, Array1, Array2, ArrayView1, ArrayView2};
use crate::camera::{CameraParameters, Normalizer};
use crate::interpolation::Interpolation;
use crate::image_range::{ImageRange, all_in_range};
use crate::transform::get_translation;
use crate::warp::Warp;
use super::depth::{calc_key_depth, calc_ref_depth, depth_search_range};
use super::epipolar::{key_coordinates, ref_coordinates};
use super::flag::Flag;
use super::gradient::ImageGradient;
use super::hypothesis::Hypothesis;
use super::intensities;
use super::numeric::{Inv, Inverse};
use super::variance::{calc_alpha, calc_variance, geo_var, photo_var};
use super::variance::VarianceCoefficients;

fn step_ratio(
    transform_rk: &Array2<f64>,
    x_key: &Array1<f64>,
    key_inv_depth: Inv,
) -> Result<f64, Flag> {
    let key_depth: f64 = key_inv_depth.inv();
    let ref_depth = calc_ref_depth(transform_rk, x_key, key_depth);
    if ref_depth <= 0. {
        return Err(Flag::NegativeRefDepth);
    }
    let ref_inv_depth = ref_depth.inv();
    let ratio = key_inv_depth / ref_inv_depth;
    Ok(f64::from(ratio))
}

fn xs_key(
    transform_rk: &Array2<f64>,
    x_key: &Array1<f64>,
    key_step_size: f64
) -> Array2<f64> {
    let t_rk = get_translation(transform_rk);
    key_coordinates(&t_rk, x_key, key_step_size)
}

fn xs_ref(
    transform_rk: &Array2<f64>,
    x_key: &Array1<f64>,
    depth_range: (f64, f64),
    ref_step_size: f64
) -> Array2<f64> {
    let (min_depth, max_depth) = depth_range;
    let x_min_ref = transform_rk.warp(x_key, min_depth);
    let x_max_ref = transform_rk.warp(x_key, max_depth);
    ref_coordinates(&x_min_ref, &x_max_ref, ref_step_size)
}

fn check_us_ref(
    us_ref: &Array2<f64>,
    us_key_size: usize,
    ref_image_shape: &[usize]
) -> Result<(), Flag> {
    if us_ref.nrows() < us_key_size {
        return Err(Flag::RefEpipolarTooShort);
    }

    if !us_ref.row(0).is_in_range(ref_image_shape) {
        return Err(Flag::RefCloseOutOfRange);
    }

    // TODO when does this condition become true?
    if !us_ref.row(us_ref.nrows()-1).is_in_range(ref_image_shape) {
        return Err(Flag::RefFarOutOfRange);
    }

    Ok(())
}

fn semi_dense(
    image_grad: ImageGradient,
    inv_depth_range: (f64, f64),
    key_camera_params: CameraParameters,
    ref_camera_params: CameraParameters,
    key_image: ArrayView2<'_, f64>,
    ref_image: ArrayView2<'_, f64>,
    transform_rk: Array2<f64>,
    u_key: ArrayView1<'_, f64>,
    prior: Hypothesis,
    ref_step_size: f64,
    min_gradient: f64,
    var_coeffs: &VarianceCoefficients
) -> Result<Hypothesis, Flag> {
    let depth_range = depth_search_range(&prior.range());
    let x_key = key_camera_params.normalize(&u_key);

    // calculate step size along the epipolar line on the keyframe
    // step size / inv depth = approximately const
    let key_step_size = match step_ratio(&transform_rk, &x_key, prior.inv_depth) {
        Err(e) => return Err(e),
        Ok(ratio) => ratio * ref_step_size,
    };

    // calculate coordinates on the keyframe image
    let xs_key = xs_key(&transform_rk, &x_key, key_step_size);
    let us_key = key_camera_params.unnormalize(&xs_key);
    if !all_in_range(&us_key, key_image.shape()) {
        return Err(Flag::KeyOutOfRange);
    }

    // extract intensities from the key coordinates
    let key_intensities = key_image.interpolate(&us_key);
    let key_gradient = intensities::gradient(&key_intensities, key_step_size);
    // most of coordinates has insufficient gradient
    // return early to reduce computation
    if key_gradient < min_gradient {
        return Err(Flag::InsufficientGradient);
    }

    // calculate coordinates on the reference frame image
    let xs_ref = xs_ref(&transform_rk, &x_key, depth_range, ref_step_size);
    let us_ref = ref_camera_params.unnormalize(&xs_ref);
    if let Err(e) = check_us_ref(&us_ref, us_key.nrows(), ref_image.shape()) {
        return Err(e);
    }

    // extract intensities from the ref coordinates
    let ref_intensities = ref_image.interpolate(&us_ref);

    // search along epipolar line and calculate depth
    let argmin = intensities::search(&ref_intensities, &key_intensities);
    let key_depth = calc_key_depth(&transform_rk, &x_key, &xs_ref.row(argmin));
    // calculate variance
    let alpha = calc_alpha(&transform_rk, &x_key, depth_range, key_depth);
    let t_rk = get_translation(&transform_rk);
    let geo_var = geo_var(&x_key, &t_rk, &image_grad.get(&u_key));
    let photo_var = photo_var(key_gradient);
    let variance = calc_variance(alpha, geo_var, photo_var, var_coeffs);

    Hypothesis::new(key_depth.inv(), variance, inv_depth_range)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_ratio() {
        let transform_rk = arr2(
            &[[0., 0., 1., -2.],
              [0., 1., 0., 2.],
              [-1., 0., 0., 7.],
              [0., 0., 0., 1.]]
        );
        let key_depth = 2.0;
        let x_key = arr1(&[2.0, 2.0]);
        // ref_depth = -(2.0 * 2.0) + 7.0 = 3.0
        let ref_depth = 3.0;

        approx::assert_abs_diff_eq!(
            step_ratio(&transform_rk, &x_key, key_depth.inv()).unwrap(),
            (1. / key_depth) / (1. / ref_depth),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_check_us_ref() {
        let ref_image_shape: [usize; 2] = [40, 30];
        let us_key_size = 2;

        let us_ref = arr2(
            &[[10., 20.],
              [0., 0.]]
        );
        let result = check_us_ref(&us_ref, us_key_size, &ref_image_shape);
        assert_eq!(result.unwrap(), ());

        let us_ref = arr2(&[[10., -2.]]);
        let result = check_us_ref(&us_ref, us_key_size, &ref_image_shape);
        assert_eq!(result.unwrap_err(), Flag::RefEpipolarTooShort);

        let us_ref = arr2(
            &[[10., -2.],
              [0., 0.]]
        );
        let result = check_us_ref(&us_ref, us_key_size, &ref_image_shape);
        assert_eq!(result.unwrap_err(), Flag::RefCloseOutOfRange);

        let us_ref = arr2(
            &[[0., 0.],
              [10., -2.]]
        );
        let result = check_us_ref(&us_ref, us_key_size, &ref_image_shape);
        assert_eq!(result.unwrap_err(), Flag::RefFarOutOfRange);
    }
}
