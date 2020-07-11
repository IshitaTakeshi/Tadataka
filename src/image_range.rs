use ndarray::{ArrayBase, Data, Ix1, Ix2};
use num::NumCast;
use num_traits::float::Float;
use std::vec::Vec;

pub trait ImageRange<D, OutputType> {
    fn is_in_range(&self, image_shape: &[usize]) -> OutputType;
}

#[inline]
fn is_in_range<A: Float>(x: A, y: A, image_shape: &[usize]) -> bool {
    let h = image_shape[0] as f64;
    let w = image_shape[1] as f64;
    let x = NumCast::from(x).unwrap();
    let y = NumCast::from(y).unwrap();
    0. <= x && x <= w-1. && 0. <= y && y <= h-1.
}

impl<A, S> ImageRange<Ix1, bool> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    A: Float,
{
    fn is_in_range(&self, image_shape: &[usize]) -> bool {
        is_in_range(self[0], self[1], image_shape)
    }
}

impl<A, S> ImageRange<Ix2, Vec<bool>> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: Float,
{
    fn is_in_range(&self, image_shape: &[usize]) -> Vec<bool> {
        let n = self.shape()[0];
        let mut mask = Vec::new();
        for i in 0..n {
            let m = is_in_range(self[[i, 0]], self[[i, 1]], image_shape);
            mask.push(m);
        }
        mask
    }
}

pub fn all_in_range<S: Data<Elem = A>, A: Float>(
    keypoints: &ArrayBase<S, Ix2>,
    image_shape: &[usize],
) -> bool {
    let n = keypoints.shape()[0];
    for i in 0..n {
        if !is_in_range(keypoints[[i, 0]], keypoints[[i, 1]], image_shape) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_is_in_range() {
        let (width, height) = (20, 30);
        let image_shape: [usize; 2] = [height, width];
        assert!(arr1(&[0., 0.]).is_in_range(&image_shape));
        assert!(!arr1(&[-1., 0.]).is_in_range(&image_shape));
        assert!(!arr1(&[0., -1.]).is_in_range(&image_shape));
        assert!(!arr1(&[0., 30.]).is_in_range(&image_shape));
        assert!(!arr1(&[20., 0.]).is_in_range(&image_shape));

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
            keypoints.is_in_range(&image_shape),
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
            keypoints.is_in_range(&image_shape),
            [true, false, false, false, true, false, false, false]
        );
    }

    #[test]
    fn test_all_in_range() {
        let (width, height) = (20, 30);
        let image_shape: [usize; 2] = [height, width];

        let keypoints = arr2(
            &[[19., 29.],
              [19., 0.],
              [0., 29.],
              [-1., 29.]] // this is out of image range
        );

        assert!(!all_in_range(&keypoints, &image_shape));

        let keypoints = arr2(
            &[[19., 29.],
              [19., 0.],
              [0., 29.]]
        );

        assert!(all_in_range(&keypoints, &image_shape));
    }
}
