use crate::homogeneous::Homogeneous;
use crate::projection::Projection;
use crate::transform::{get_rotation, get_translation};
use crate::vector;
use crate::warp::Warp;
use ndarray::{Array1, Array2, ArrayView1};

static EPSILON: f64 = 1e-16;

pub struct VarianceCoefficients {
    pub photo: f64,
    pub geo: f64,
}

pub fn calc_variance(
    alpha: f64,
    geo_var: f64,
    photo_var: f64,
    coeffs: &VarianceCoefficients
) -> f64 {
    let a2 = alpha * alpha;
    let g2 = coeffs.geo * coeffs.geo;
    let p2 = coeffs.photo * coeffs.photo; a2 * (g2 * geo_var + p2 * photo_var)
}

pub fn photo_var(gradient: f64) -> f64 {
    2. / gradient
}

fn geo_var_(
    direction: &Array1<f64>,
    image_grad: &Array1<f64>,
) -> f64 {
    let direction = vector::normalize(direction);
    let image_grad = vector::normalize(image_grad);

    let p = direction.dot(&image_grad);

    if p == 0. {
        return 1. / EPSILON;
    }
    return 1. / (p * p);
}

pub fn geo_var(
    x_key: &Array1<f64>,
    t_rk: &ArrayView1<'_, f64>,
    image_grad: &Array1<f64>,
) -> f64 {
    let epipolar_direction = x_key - &Projection::project(t_rk);
    geo_var_(&epipolar_direction, image_grad)
}

fn alpha_(
    x_key: &Array1<f64>,
    x_ref_i: f64,
    direction_i: f64,
    ri: &ArrayView1<'_, f64>,
    rz: &ArrayView1<'_, f64>,
    ti: f64,
    tz: f64
) -> f64 {
    let y = x_key.to_homogeneous();

    let d = rz.dot(&y) * ti - ri.dot(&y) * tz;
    let n = x_ref_i * tz - ti;

    direction_i * d / (n * n)
}

fn ref_epipolar_direction(
    transform_rk: &Array2<f64>,
    x_key: &Array1<f64>,
    depth_range: (f64, f64),
) -> Array1<f64> {
    let (min_depth, max_depth) = depth_range;
    let x_min_ref = transform_rk.warp(x_key, min_depth);
    let x_max_ref = transform_rk.warp(x_key, max_depth);
    vector::normalize(&(x_max_ref - x_min_ref))
}

fn calc_alpha_(
    transform_rk: &Array2<f64>,
    x_key: &Array1<f64>,
    direction: &Array1<f64>,
    prior_depth: f64
) -> f64 {
    let rot_rk = get_rotation(&transform_rk);
    let t_rk = get_translation(&transform_rk);
    let x_ref = transform_rk.warp(x_key, prior_depth);

    let i = if f64::abs(direction[0]) > f64::abs(direction[1]) { 0 } else { 1 };
    alpha_(&x_key, x_ref[i], direction[i],
           &rot_rk.row(i), &rot_rk.row(2), t_rk[i], t_rk[2])
}

pub fn calc_alpha(
    transform_rk: &Array2<f64>,
    x_key: &Array1<f64>,
    depth_range: (f64, f64),
    prior_depth: f64
) -> f64 {
    let d = ref_epipolar_direction(&transform_rk, &x_key, depth_range);
    calc_alpha_(transform_rk, x_key, &d, prior_depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, Array};
    use ndarray_linalg::Norm;
    use crate::transform;

    #[test]
    fn test_geo_var() {
        let gradient = arr1(&[20., -30.]);
        let direction = arr1(&[6., 2.]);  // epipolar line direction

        // smoke
        let variance = geo_var_(&direction, &gradient);
        let p = vector::normalize(&direction).dot(&vector::normalize(&gradient));
        assert_abs_diff_eq!(variance, 1. / (p * p));

        // zero epipolar direction
        let variance = geo_var_(&Array::zeros(2), &gradient);
        assert_abs_diff_eq!(variance, 1. / EPSILON);

        // zero gradient
        let variance = geo_var_(&direction, &Array::zeros(2));
        assert_abs_diff_eq!(variance, 1. / EPSILON);

        // the case that the gradient is orthogonal to epipolar direction (max variance)
        let gradient = arr1(&[direction[1], -direction[0]]);
        let variance = geo_var_(&direction, &gradient);
        assert_abs_diff_eq!(variance, 1. / EPSILON);
    }

    #[test]
    fn test_photo_var() {
        let gradient = 10.0;
        let variance = photo_var(gradient);
        assert_eq!(variance, 0.2);
    }

    #[test]
    fn test_alpha_() {
        let rot = arr2(
            &[[0., -1., 0.],
              [1., 0., 0.],
              [0., 0., 1.]]
        );
        let t = arr1(&[2., 4., -3.]);

        let direction = arr1(&[0.1, 0.3]);
        let x_key = arr1(&[0.3, 0.9]);
        let x_ref = arr1(&[-0.6, 0.4]);

        let y = x_key.to_homogeneous();

        let alpha = alpha_(&x_key, x_ref[0], direction[0],
                           &rot.row(0), &rot.row(2), t[0], t[2]);

        let n = t[0] * rot.row(2).dot(&y) - t[2] * rot.row(0).dot(&y);
        let d = t[0] - x_ref[0] * t[2];
        assert_eq!(alpha, direction[0] * n / (d * d));

        let alpha = alpha_(&x_key, x_ref[1], direction[1],
                           &rot.row(1), &rot.row(2), t[1], t[2]);

        let n = t[1] * rot.row(2).dot(&y) - t[2] * rot.row(1).dot(&y);
        let d = t[1] - x_ref[1] * t[2];
        assert_eq!(alpha, direction[1] * n / (d * d));
    }

    #[test]
    fn test_calc_alpha_() {
        let rot_rk = arr2(&[
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 1.]
        ]);
        let t_rk = arr1(&[2., 4., -3.]);

        let transform_rk = transform::make_matrix(&rot_rk, &t_rk);

        let x_key = arr1(&[0.3, 0.9]);
        let prior_depth = 10.0;

        let x_ref = transform_rk.warp(&x_key, prior_depth);

        let direction = arr1(&[0.1, 0.3]);
        assert_eq!(
            calc_alpha_(&transform_rk, &x_key, &direction, prior_depth),
            alpha_(&x_key, x_ref[1], direction[1],
                        &rot_rk.row(1), &rot_rk.row(2), t_rk[1], t_rk[2]));

        let direction = arr1(&[-2., 1.]);
        assert_eq!(
            calc_alpha_(&transform_rk, &x_key, &direction, prior_depth),
            alpha_(&x_key, x_ref[0], direction[0],
                   &rot_rk.row(0), &rot_rk.row(2), t_rk[0], t_rk[2]));
    }

    #[test]
    fn test_calc_variance() {
        let coeffs = VarianceCoefficients { photo: 2., geo: 3. };
        let alpha = 0.4;
        let geo_var = 0.9;
        let photo_var = 0.8;
        let v = calc_variance(alpha, geo_var, photo_var, &coeffs);
        assert_eq!(v, 0.4 * 0.4 * (2. * 2. * 0.8 + 3. * 3. * 0.9));
     }

    #[test]
    fn test_ref_epipolar_direction() {
        let transform_rk = arr2(
            &[[0., 0., 1., -1.],
              [0., 1., 0., -2.],
              [-1., 0., 0., 4.],
              [0., 0., 0., 1.]]
        );
        let x_key = arr1(&[0.4, 1.2]);
        let (min, max) = (0.5, 1.4);
        let d1 = ref_epipolar_direction(&transform_rk, &x_key, (min, max));
        let d2 = transform_rk.warp(&x_key, max) - transform_rk.warp(&x_key, min);

        assert_eq!(d1.norm(), 1.);
        // d1 and d2 should be the same direction
        assert_eq!(d1.dot(&d2), d2.norm());
    }
}
