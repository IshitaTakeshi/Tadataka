extern crate test;

use crate::homogeneous::{to_homogeneous_vecs, from_homogeneous_vecs};
use ndarray::{arr2, Array2, ArrayView2};

pub fn transform(transform10: ArrayView2<'_, f64>, points0: ArrayView2<'_, f64>)
    -> Array2<f64> {
    let points0 = to_homogeneous_vecs(points0);
    let points0 = points0.t();
    let points1 = transform10.dot(&points0);
    let points1 = points1.t();
    from_homogeneous_vecs(&points1.view()).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn test_transform() {
        let points0 = arr2(
            &[[1.,  2., 5.],
              [4., -2., 3.]]
        );

        let transform10 = arr2(
            &[[1., 0.,  0., 1.],
              [0., 0., -1., 2.],
              [0., 1.,  0., 3.],
              [0., 0.,  0., 1.]]
        );

        let points1 = arr2(
            &[[2., -3., 5.],
              [5., -1., 1.]]
        );

        assert_eq!(transform(transform10.view(), points0.view()), points1)
    }
}
