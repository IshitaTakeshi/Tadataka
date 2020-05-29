use ndarray::{arr1, arr2, Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::Norm;
use crate::camera::{CameraParameters, Normalizer};
use crate::interpolation::Interpolation;
use crate::image_range::{ImageRange, all_in_range};
use crate::numeric::safe_invert;
use crate::transform::get_translation;
use crate::projection::Projection;
use crate::vector;
use crate::warp::Warp;
use super::depth::{calc_key_depth, calc_ref_depth, depth_search_range, inv_depth_range};
use super::epipolar::{key_coordinates, ref_coordinates};
use super::flag::Flag;
use super::gradient::ImageGradient;
use super::hypothesis::Hypothesis;
use super::intensities;
use super::variance::{calc_alpha, geo_var, photo_var};

fn calc_ref_inv_depth(
    transform_rk: &Array2<f64>,
    x_key: &Array1<f64>,
    key_inv_depth: f64
) -> f64 {
    let key_depth = safe_invert(key_inv_depth);
    let ref_depth = calc_ref_depth(transform_rk, x_key, key_depth);
    safe_invert(ref_depth)
}

// this function is tested in the python side
fn semi_dense(
    image_grad: ImageGradient,
    valid_depth_range: (f64, f64),
    key_camera_params: CameraParameters,
    ref_camera_params: CameraParameters,
    key_image: ArrayView2<'_, f64>,
    ref_image: ArrayView2<'_, f64>,
    transform_rk: Array2<f64>,
    u_key: ArrayView1<'_, f64>,
    prior: Hypothesis,
    step_size_ref: f64,
    min_gradient: f64,
    sigma_i: f64,
    sigma_l: f64
) -> (Hypothesis, Flag) {
    if prior.inv_depth <= 0. {
        return (prior, Flag::NegativePriorDepth);
    }

    let (min_depth, max_depth) = match inv_depth_range(&prior, valid_depth_range) {
        Some(range) => depth_search_range(range),
        None => return (prior, Flag::HypothesisOutOfSerchRange),
    };

    let x_key = key_camera_params.normalize(&u_key);
    let x_min_ref = transform_rk.warp(&x_key, min_depth);
    let x_max_ref = transform_rk.warp(&x_key, max_depth);

    // step size / inv depth = approximately const
    let ref_inv_depth = calc_ref_inv_depth(&transform_rk, &x_key, prior.inv_depth);
    if ref_inv_depth <= 0. {
        return (prior, Flag::NegativeRefDepth);
    }
    let step_size_key = (prior.inv_depth / ref_inv_depth) * step_size_ref;

    let t_rk = get_translation(&transform_rk);
    let xs_key = key_coordinates(&t_rk, &x_key, step_size_key);
    let us_key = key_camera_params.unnormalize(&xs_key);
    if !all_in_range(&us_key, key_image.shape()) {
        return (prior, Flag::KeyOutOfRange);
    }

    let key_intensities = key_image.interpolate(&us_key);
    let key_gradient = intensities::gradient(&key_intensities).norm();

    if key_gradient < min_gradient {
        return (prior, Flag::InsufficientGradient);
    }

    let xs_ref = ref_coordinates(&x_min_ref, &x_max_ref, step_size_ref);
    if xs_ref.nrows() < xs_key.nrows() {
        return (prior, Flag::RefEpipolarTooShort);
    }

    let us_ref = ref_camera_params.unnormalize(&xs_ref);

    if !us_ref.row(0).is_in_range(ref_image.shape()) {
        return (prior, Flag::RefCloseOutOfRange);
    }

    // TODO when does this condition become true?
    if !us_ref.row(us_ref.nrows()-1).is_in_range(ref_image.shape()) {
        return (prior, Flag::RefFarOutOfRange);
    }

    let ref_intensities = ref_image.interpolate(&us_ref);
    let argmin = intensities::search(&key_intensities, &ref_intensities);

    let depth_key = calc_key_depth(&transform_rk, &x_key, &xs_ref.row(argmin));
    let direction = vector::normalize(&(x_max_ref - x_min_ref));
    let alpha = calc_alpha(&transform_rk, &x_key, &direction,
                           safe_invert(prior.inv_depth));

    let geo_var = geo_var(&(x_key - t_rk.project()),
                          &image_grad.get(&u_key), sigma_l);
    let photo_var = photo_var(key_gradient / step_size_key, sigma_i);
    let variance = alpha * alpha * (geo_var + photo_var);

    (Hypothesis::new(safe_invert(depth_key), variance), Flag::Success)
}
