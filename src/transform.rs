use crate::homogeneous::Homogeneous;
use ndarray::{arr1, arr2, Array, Array2, ArrayView, ArrayView2, ArrayBase,
              RawData, Data, Dim, LinalgScalar, Ix1, Ix2};

pub trait Transform<A, D, Rhs> {
    fn transform(&self, points0: &Rhs) -> Array<A, D>;
}

macro_rules! impl_transform {
    (for $($d:ty),+) => {
        $(impl<A, S1, S2> Transform<A, $d, ArrayBase<S2, $d>> for ArrayBase<S1, Ix2>
        where
            S1: Data<Elem = A>,
            S2: Data<Elem = A>,
            A: LinalgScalar,
        {
            fn transform(&self, points0: &ArrayBase<S2, $d>) -> Array<A, $d> {
                let points0 = &points0.to_homogeneous();
                let points0 = points0.t();
                let points1 = self.dot(&points0);
                let points1 = points1.t();
                points1.from_homogeneous().to_owned()
            }
        })*
    }
}

impl_transform!(for Ix1, Ix2);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_points() {
        let points0 = arr2(&[[1., 2., 5.], [4., -2., 3.]]);

        let transform10 = arr2(&[
            [1., 0., 0., 1.],
            [0., 0., -1., 2.],
            [0., 1., 0., 3.],
            [0., 0., 0., 1.],
        ]);

        let points1 = arr2(&[[2., -3., 5.], [5., -1., 1.]]);

        // assert_eq!(transform(&transform10, &points0), points1);
        assert_eq!(transform10.transform(&points0), points1);
    }

    #[test]
    fn test_transform_point() {
        let point0 = arr1(&[1., 2., 5.]);

        let transform10 = arr2(&[
            [1., 0., 0., 1.],
            [0., 0., -1., 2.],
            [0., 1., 0., 3.],
            [0., 0., 0., 1.],
        ]);

        let point1 = arr1(&[2., -3., 5.]);

        assert_eq!(transform10.transform(&point0), point1);
    }
}
