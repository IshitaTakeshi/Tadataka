use ndarray::{Array, ArrayBase, ArrayView, Axis,
              Data, Ix1, Ix2, LinalgScalar, stack};

pub trait Homogeneous<A, D> {
    fn to_homogeneous(&self) -> Array<A, D>;
    fn from_homogeneous(&self) -> ArrayView<'_, A, D>;
}

impl<A, S> Homogeneous<A, Ix1> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    A: LinalgScalar,
{
    fn to_homogeneous(&self) -> Array<A, Ix1> {
        stack![Axis(0), self.view(), Array::ones(1)]
    }

    fn from_homogeneous(&self) -> ArrayView<'_, A, Ix1> {
        let n = self.shape()[0];
        self.slice(s![0..n - 1])
    }
}

impl<A, S> Homogeneous<A, Ix2> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: LinalgScalar,
{
    fn to_homogeneous(&self) -> Array<A, Ix2> {
        let n = self.shape()[0];
        stack![Axis(1), self.view(), Array::ones((n, 1))]
    }

    fn from_homogeneous(&self) -> ArrayView<'_, A, Ix2> {
        let n_cols = self.shape()[1];
        self.slice(s![.., 0..n_cols - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*; // import names from outer scope
    use ndarray::{arr1, arr2};

    #[test]
    fn test_to_homogeneous_1d() {
        let point = arr1(&[2., 3.]);
        let expected = arr1(&[2., 3., 1.]);
        assert_eq!(point.to_homogeneous(), expected);
    }

    #[test]
    fn test_to_homogeneous_2d() {
        let points = arr2(&[[2., 3.], [4., 5.]]);
        let expected = arr2(&[[2., 3., 1.], [4., 5., 1.]]);
        assert_eq!(points.to_homogeneous(), expected);
    }

    #[test]
    fn test_from_homogeneous_1d() {
        let point = arr1(&[2., 3., 1.]);
        let expected = arr1(&[2., 3.]);
        assert_eq!(point.from_homogeneous(), expected);
    }

    #[test]
    fn test_from_homogeneous_2d() {
        let points = arr2(&[[2., 3., 1.], [4., 5., 1.]]);
        let expected = arr2(&[[2., 3.], [4., 5.]]);
        assert_eq!(points.from_homogeneous(), expected);
    }
}
