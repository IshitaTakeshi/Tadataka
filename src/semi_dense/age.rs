use crate::camera::{CameraParameters, Normalizer};
use crate::warp::Warp;
use crate::image_range::ImageRange;
use ndarray::{arr1, Array, Array2, ArrayView2};

pub fn increment_age(
    age_map0: &ArrayView2<'_, usize>,
    camera_params0: &CameraParameters,
    camera_params1: &CameraParameters,
    transform10: &ArrayView2<'_, f64>,
    depth_map0: &ArrayView2<'_, f64>,
) -> Array2<usize> {
    assert_eq!(age_map0.shape(), depth_map0.shape());
    let shape = age_map0.shape();
    let (height, width) = (shape[0], shape[1]);

    let mut age_map1 = Array::zeros((height, width));
    for y0 in 0..height {
        for x0 in 0..width {
            let p0 = arr1(&[x0 as f64, y0 as f64]);
            let q0 = camera_params0.normalize(&p0);
            let (q1, _) = transform10.warp(&q0, depth_map0[[y0, x0]]);
            let p1 = camera_params1.unnormalize(&q1);
            if !p1.is_in_range(shape) {
                continue;
            }
            let (x1, y1) = (p1[0] as usize, p1[1] as usize);
            age_map1[[y1, x1]] = age_map0[[y0, x0]] + 1;
        }
    }
    return age_map1;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_increment_age() {
        let (width, height) = (12, 16);
        let shape = (height, width);

        let camera_params = CameraParameters::new(
            (10., 10.),
            ((width as f64) / 2., (height as f64) / 2.)
        );

        let transform10 = arr2(
            &[[1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 10.],
              [0., 0., 0., 1.]],
        );

        let depth_map0: Array2<f64> = 10.0 * Array::ones(shape);
        let age_map0 = Array::zeros(shape);
        let age_map1 = increment_age(&age_map0.view(), &camera_params, &camera_params,
                                     &transform10.view(), &depth_map0.view());
        let mut expected = Array::zeros(shape);
        expected.slice_mut(s![4..12, 3..9]).assign(&Array::ones((8, 6)));
        assert_eq!(age_map1, expected);
    }
}
