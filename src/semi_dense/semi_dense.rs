use std::convert::TryInto;

use indicatif::ProgressBar;
use ndarray::{arr1, Array, Array1, Array2};
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::Norm;

use crate::camera::Normalizer;
use crate::gradient::gradient1d;
use crate::interpolation::Interpolation;
use crate::image_range::{ImageRange, all_in_range};
use crate::transform::get_translation;
use crate::warp::Warp;
use super::depth::{calc_key_depth, calc_ref_depth, depth_search_range};
use super::epipolar::{key_coordinates, ref_coordinates};
use super::flag::Flag;
use super::frame::Frame;
use super::gradient::ImageGradient;
use super::hypothesis;
use super::hypothesis::Hypothesis;
use super::intensities;
use super::numeric::Inv;
use super::numeric::Inverse as InverseDepth;
use super::params::Params;
use super::variance::{calc_alpha, calc_variance, geo_var, photo_var};

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
    let (x_min_ref, _) = transform_rk.warp(x_key, min_depth);
    let (x_max_ref, _) = transform_rk.warp(x_key, max_depth);
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

fn transform_rk(
    transform_wk: &Array2<f64>,
    transform_wr: &Array2<f64>,
) -> Array2<f64> {
    let transform_rw = transform_wr.inv().unwrap();
    transform_rw.dot(transform_wk)
}

pub fn estimate(
    u_key: &Array1<f64>,
    prior: &Hypothesis,
    keyframe: &Frame,
    refframe: &Frame,
    image_grad: &ImageGradient,
    params: &Params,
) -> Result<Hypothesis, Flag> {
    let transform_rk = transform_rk(&keyframe.transform, &refframe.transform);

    let depth_range = depth_search_range(&prior.range());
    let x_key = keyframe.camera_params.normalize(u_key);

    // calculate step size along the epipolar line on the keyframe
    // step size / inv depth = approximately const
    let result = step_ratio(&transform_rk, &x_key, prior.inv_depth);
    let key_step_size = match result {
        Err(e) => return Err(e),
        Ok(ratio) => ratio * params.ref_step_size,
    };

    // calculate coordinates on the keyframe image
    let xs_key = xs_key(&transform_rk, &x_key, key_step_size);
    let us_key = keyframe.camera_params.unnormalize(&xs_key);
    if !all_in_range(&us_key, keyframe.image.shape()) {
        return Err(Flag::KeyOutOfRange);
    }

    // extract intensities from the key coordinates
    let key_intensities = keyframe.image.interpolate(&us_key);
    let key_gradient = gradient1d(&key_intensities).norm();
    // most of coordinates has insufficient gradient
    // return early to reduce computation
    if key_gradient < params.min_gradient {
        return Err(Flag::InsufficientGradient);
    }

    // calculate coordinates on the reference frame image
    let xs_ref = xs_ref(&transform_rk, &x_key, depth_range, params.ref_step_size);
    let us_ref = refframe.camera_params.unnormalize(&xs_ref);
    check_us_ref(&us_ref, us_key.nrows(), refframe.image.shape())?;

    // extract intensities from the ref coordinates
    let ref_intensities = refframe.image.interpolate(&us_ref);

    // search along epipolar line and calculate depth
    let argmin = intensities::search(&ref_intensities, &key_intensities);
    let key_depth = calc_key_depth(&transform_rk, &x_key, &xs_ref.row(argmin));
    // calculate variance
    let alpha = calc_alpha(&transform_rk, &x_key, depth_range, key_depth);
    let t_rk = get_translation(&transform_rk);
    let geo_var = geo_var(&x_key, &t_rk, &image_grad.get(&u_key));
    let photo_var = photo_var(key_gradient / key_step_size);
    let variance = calc_variance(alpha, geo_var, photo_var, &params.var_coeffs);

    hypothesis::check_args(key_depth.inv(), variance, params.inv_depth_range)?;
    Ok(Hypothesis::new(key_depth.inv(), variance, params.inv_depth_range))
}

pub fn update_depth(
    keyframe: &Frame,
    refframes: &Vec<Frame>,
    age_map: &Array2<usize>,
    prior_depth: &Array2<f64>,
    prior_variance: &Array2<f64>,
    params: &Params,
) -> (Array2<i64>, Array2<f64>, Array2<f64>) {
    assert!(age_map.shape() == prior_depth.shape());
    assert!(age_map.shape() == prior_variance.shape());
    assert!(age_map.shape() == keyframe.image.shape());
    for i in 0..refframes.len() {
        assert!(age_map.shape() == refframes[i].image.shape());
    }

    let image_grad = ImageGradient::new(&keyframe.image);
    let height = keyframe.image.shape()[0];
    let width = keyframe.image.shape()[1];

    let mut flag_map = Array::zeros((height, width));
    let mut result_depth = Array::zeros((height, width));
    let mut result_variance = Array::zeros((height, width));
    let inv_depth_range = params.inv_depth_range;

    let bar = ProgressBar::new((height * width).try_into().unwrap());

    for y in 0..height {
        for x in 0..width {
            bar.inc(1);

            let age = age_map[[y, x]];
            let d = prior_depth[[y, x]];
            let v = prior_variance[[y, x]];

            if age == 0 {
                // refframe cannot be observed from this pixel
                result_depth[[y, x]] = d;
                result_variance[[y, x]] = v;
                flag_map[[y, x]] = Flag::NotProcessed as i64;
                continue;
            }

            if refframes.len() < age {
                eprintln!("Age exceeds the refframe size");
                std::process::exit(1);
            }

            let refframe = &refframes[refframes.len()-age];

            if let Err(f) = hypothesis::check_args(d.inv(), v, inv_depth_range) {
                result_depth[[y, x]] = d;
                result_variance[[y, x]] = v;
                flag_map[[y, x]] = f as i64;
                continue;
            }

            let u_key = arr1(&[x as f64, y as f64]);
            let prior = Hypothesis::new(d.inv(), v, inv_depth_range);
            let result = estimate(&u_key, &prior, &keyframe, refframe,
                                  &image_grad, &params);
            let (h, flag) = match result {
                Err(flag) => (prior, flag),
                Ok(h) => (h, Flag::Success),
            };

            result_depth[[y, x]] = h.inv_depth.inv();
            result_variance[[y, x]] = h.variance;
            flag_map[[y, x]] = flag as i64;
        }
    }

    bar.finish();

    (flag_map, result_depth, result_variance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

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

        let key_depth = 4.0;
        let x_key = arr1(&[2.0, 2.0]);
        // ref_depth = -(4.0 * 2.0) + 7.0 = -1.0
        assert_eq!(
            step_ratio(&transform_rk, &x_key, key_depth.inv()).unwrap_err(),
            Flag::NegativeRefDepth
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
