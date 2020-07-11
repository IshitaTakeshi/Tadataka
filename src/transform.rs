use crate::homogeneous::Homogeneous;
use ndarray::{Array, Array2, ArrayBase, ArrayView1, ArrayView2, Axis,
              Data, LinalgScalar, Ix1, Ix2, stack};

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

pub fn get_rotation<'a, A, S>(
    transform: &'a ArrayBase<S, Ix2>
) -> ArrayView2<'a, A>
where
    S: Data<Elem = A>,
    A: LinalgScalar,
{
    transform.slice(s![0..3, 0..3])
}

pub fn get_translation<'a, A, S>(
    transform: &'a ArrayBase<S, Ix2>
) -> ArrayView1<'a, A>
where
    S: Data<Elem = A>,
    A: LinalgScalar,
{
    transform.slice(s![0..3, 3])
}

pub fn make_matrix<A, S1, S2>(
    rotation: &ArrayBase<S1, Ix2>,
    translation: &ArrayBase<S2, Ix1>
) -> Array2<A>
where
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar
{
    let d = rotation.shape()[1];

    let translation = translation.to_owned();
    let translation = translation.into_shape((d, 1)).unwrap();

    stack![
        Axis(0),
        stack![Axis(1), rotation.view(), translation.view()],
        stack![Axis(1), Array::zeros((1, d)), Array::ones((1, 1))]
    ]
}

pub fn inv_transform<A, S>(transform: &ArrayBase<S, Ix2>) -> Array2<A>
where
    S: Data<Elem = A>,
    A: LinalgScalar + std::ops::Neg<Output = A> {
    let rotation = get_rotation(&transform);
    let t = get_translation(&transform);
    let rt = rotation.t().dot(&t);
    return make_matrix(&rotation.t(), &(rt.map(|&e| -e)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_transform_points() {
        let points0 = arr2(&[[1, 2, 5], [4, -2, 3]]);

        let transform10 = arr2(&[
            [1, 0, 0, 1],
            [0, 0, -1, 2],
            [0, 1, 0, 3],
            [0, 0, 0, 1],
        ]);

        let points1 = arr2(&[[2, -3, 5], [5, -1, 1]]);

        assert_eq!(transform10.transform(&points0), points1);
    }

    #[test]
    fn test_transform_point() {
        let point0 = arr1(&[1, 2, 5]);

        let transform10 = arr2(&[
            [1, 0, 0, 1],
            [0, 0, -1, 2],
            [0, 1, 0, 3],
            [0, 0, 0, 1],
        ]);

        let point1 = arr1(&[2, -3, 5]);

        assert_eq!(transform10.transform(&point0), point1);
    }

    #[test]
    fn test_get_rotation() {
        let transform = arr2(&[[1, 2, 3, -1],
                               [4, 5, 6, -2],
                               [7, 8, 9, -3],
                               [0, 0, 0, 1]]);
        let expected = arr2(&[[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]]);
        assert_eq!(get_rotation(&transform.view()), expected.view());
    }

    #[test]
    fn test_get_translation() {
        let transform = arr2(&[[1, 2, 3, -1],
                               [4, 5, 6, -2],
                               [7, 8, 9, -3],
                               [0, 0, 0, 1]]);
        let expected = arr1(&[-1, -2, -3]);
        assert_eq!(get_translation(&transform.view()), expected.view());
    }

    #[test]
    fn test_make_matrix() {
        let rotation = arr2(&[[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]]);
        let translation = arr1(&[-1, -2, -3]);
        let expected = arr2(&[[1, 2, 3, -1],
                              [4, 5, 6, -2],
                              [7, 8, 9, -3],
                              [0, 0, 0, 1]]);
        assert_eq!(make_matrix(&rotation, &translation), expected);
    }

    #[test]
    fn test_inv_transform() {
        let transform = arr2(&[[1., 0., 0., -1.],
                               [0., 0., 1., -2.],
                               [0., -1., 0., 3.],
                               [0., 0., 0., 1.]]);
        assert_eq!(inv_transform(&transform).dot(&transform),
                   arr2(&[[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]]));
    }
}
