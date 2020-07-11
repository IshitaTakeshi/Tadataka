use crate::image_range::ImageRange;
use crate::semi_dense::fusion::fusion;
use crate::semi_dense::numeric::Inverse;
use crate::semi_dense::stat;
use crate::warp::{PerspectiveWarp, Warp};
use ndarray::{arr1, Array, Array2, Data};
use std::collections::HashMap;

fn propagate_variance(
    depth0: f64,
    depth1: f64,
    variance0: f64,
    uncertaintity: f64,
) -> f64 {
    // we assume that rotation change is small between timestamp 0 and timestamp 1
    // TODO accept the case that rotation change is significantly large
    let ratio = depth1.inv() / depth0.inv();
    f64::from(ratio).powi(4) * variance0 + uncertaintity  // variance1
}

fn handle_collision(
    depth_a: f64,
    depth_b: f64,
    variance_a: f64,
    variance_b: f64,
) -> (f64, f64) {
    if stat::are_statically_same(
        depth_a.inv(), depth_b.inv(), variance_a, variance_b
    ) {
        let (inv_depth, variance) = fusion(
            depth_a.inv(),
            depth_b.inv(),
            variance_a,
            variance_b
        );
        return (inv_depth.inv(), variance);
    }

    if depth_a < depth_b {
        // b is hidden by a
        return (depth_a, variance_a);
    } else {
        // a is hidden by b
        return (depth_b, variance_b);
    }
}

pub fn propagate<T: Data<Elem = f64>>(
    warp10: &PerspectiveWarp<T>,
    depth_map0: &Array2<f64>,
    variance_map0: &Array2<f64>,
    default_depth: f64,
    default_variance: f64,
    uncertaintity_bias: f64,
) -> (Array2<f64>, Array2<f64>) {
    let shape = depth_map0.shape();
    let (height, width) = (shape[0], shape[1]);

    let mut map1: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    for y0 in 0..height {
        for x0 in 0..width {
            let u0 = arr1(&[x0 as f64, y0 as f64]);
            let depth0 = depth_map0[[y0 as usize, x0 as usize]];
            let (u1, depth1a) = warp10.warp(&u0, depth0);
            if !u1.is_in_range(shape) {
                continue;
            }

            let variance0 = variance_map0[[y0 as usize, x0 as usize]];
            let variance1a = propagate_variance(depth0, depth1a, variance0,
                                                uncertaintity_bias);
            let u1 = (u1[0] as usize, u1[1] as usize);
            let (depth1, variance1) = match map1.get(&u1) {
                Some((depth1b, variance1b)) => {
                    handle_collision(depth1a, *depth1b, variance1a, *variance1b)
                },
                None => (depth1a, variance1a)
            };

            map1.insert(u1, (depth1, variance1));
        }
    }

    let mut depth_map1 = Array::from_elem((height, width), default_depth);
    let mut variance_map1 = Array::from_elem((height, width), default_variance);
    for ((x1, y1), (depth1, variance1)) in map1.iter() {
        depth_map1[[*y1, *x1]] = *depth1;
        variance_map1[[*y1, *x1]] = *variance1;
    }

    (depth_map1, variance_map1)
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use ndarray::arr2;
    use crate::camera::CameraParameters;

    #[test]
    fn test_propagate_variance() {
        let depth0 = 4.0;
        let depth1 = 2.0;
        let inv_depth0 = 1. / depth0;
        let inv_depth1 = 1. / depth1;
        let variance0 = 0.5;
        let r = inv_depth1 / inv_depth0;

        assert_eq!(
            propagate_variance(depth0, depth1, variance0, 1.0),
            (r * r * r * r) * variance0 + 1.0
        )
    }

    #[test]
    fn test_propagate() {
        let (width, height) = (8, 8);
        let shape = (height, width);

        let camera_params = CameraParameters::new(
            (100., 100.),
            (width as f64 / 2., height as f64 / 2.)
        );

        let default_depth = 60.;
        let default_variance = 8.;
        let uncertaintity_bias = 3.;

        let transform10 = arr2(
            &[[1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 300.],
              [0., 0., 0., 1.]]
        );

        let warp10 = PerspectiveWarp::new(&transform10, &camera_params, &camera_params);
        let depth0 = 100.;
        let variance0 = 20.;
        // -1.0 + 4, -1.0 + 4,

        let depth_map0 = Array::from_elem(shape, depth0);
        let variance_map0 = Array::from_elem(shape, variance0);

        let (depth_map1, variance_map1) = propagate(
            &warp10,
            &depth_map0,
            &variance_map0,
            default_depth,
            default_variance,
            uncertaintity_bias
        );

        let depth1 = 400.;

        let mut expected = Array::from_elem(shape, default_depth);
        expected.slice_mut(s![3..5, 3..5])
            .assign(&Array::from_elem((2, 2), depth1));

        for y in 0..height {
            for x in 0..width {
                assert_abs_diff_eq!(depth_map1[[y, x]], expected[[y, x]],
                                    epsilon = 1e-4);
            }
        }

        let variance1 = propagate_variance(depth0, depth1,
                                           variance0, uncertaintity_bias);
        // 16 pixels in variance_map0 will be
        // fused into 1 pixel in variance_map1
        // Therefore variance should be decreased to 1/16
        let mut expected = Array::from_elem(shape, default_variance);
        expected.slice_mut(s![3..5, 3..5])
            .assign(&Array::from_elem((2, 2), variance1 / 16.));

        for y in 0..height {
            for x in 0..width {
                assert_abs_diff_eq!(variance_map1[[y, x]], expected[[y, x]],
                                    epsilon = 1e-4);
            }
        }
    }
}
