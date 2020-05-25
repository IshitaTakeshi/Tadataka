use ndarray::{arr2, ArrayView2};
use std::vec::Vec;

fn is_in_image_range_(x: f64, y: f64, w: f64, h: f64) -> bool {
    0. <= x && x <= w-1. && 0. <= y && y <= h-1.
}

pub fn is_in_image_range(
    keypoints: ArrayView2<'_, f64>,
    image_shape: (i64, i64)  // NOTE the order (h, w)
) -> Vec<bool> {
    let h = image_shape.0 as f64;
    let w = image_shape.1 as f64;

    let n = keypoints.shape()[0];
    let mut mask = Vec::new();
    for i in 0..n {
        mask.push(
            is_in_image_range_(keypoints[[i, 0]], keypoints[[i, 1]], w, h)
        );
    }
    mask
}

fn all_are_in_image_range(
    keypoints: ArrayView2<'_, f64>,
    image_shape: (i64, i64)  // NOTE the order (h, w)
) -> bool {
    let h = image_shape.0 as f64;
    let w = image_shape.1 as f64;

    let n = keypoints.shape()[0];
    for i in 0..n {
        if !is_in_image_range_(keypoints[[i, 0]], keypoints[[i, 1]], w, h) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_in_image_range() {
        let (width, height) = (20, 30);
        let image_shape = (height, width);

        let keypoints = arr2(
            &[[19., 29.],
              [19., 0.],
              [0., 29.],
              [-1., 29.],
              [19., -1.],
              [20., 29.],
              [19., 30.],
              [20., 30.]]
        );

        assert_eq!(
            is_in_image_range(keypoints.view(), image_shape),
            [true, true, true, false, false, false, false, false]
        );

        let keypoints = arr2(
            &[[19.00, 29.00],
              [19.01, 29.00],
              [19.00, 29.01],
              [19.01, 29.01],
              [00.00, 00.00],
              [00.00, -0.01],
              [-0.01, 00.00],
              [-0.01, -0.01]]
        );

        assert_eq!(
            is_in_image_range(keypoints.view(), image_shape),
            [true, false, false, false, true, false, false, false]
        );
    }

    #[test]
    fn test_all_are_in_image_range() {
        let (width, height) = (20, 30);
        let image_shape = (height, width);

        let keypoints = arr2(
            &[[19., 29.],
              [19., 0.],
              [0., 29.],
              [-1., 29.]] // this is out of image range
        );

        assert!(!all_are_in_image_range(keypoints.view(), image_shape));

        let keypoints = arr2(
            &[[19., 29.],
              [19., 0.],
              [0., 29.]]
        );

        assert!(all_are_in_image_range(keypoints.view(), image_shape));
    }
}
