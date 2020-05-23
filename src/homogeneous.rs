use ndarray::{arr1, arr2, stack, Array, Array1, Array2,
              ArrayView, ArrayView1, ArrayView2, Axis,
              LinalgScalar, Ix1, Ix2};

pub trait Homogeneous<A, D> {
    fn to_homogeneous(&self) -> Array<A, D>;
    fn from_homogeneous(&self) -> ArrayView<'_, A, D>;
}

impl<A> Homogeneous<A, Ix1> for Array<A, Ix1> where A: LinalgScalar {
    fn to_homogeneous(&self) -> Array<A, Ix1> {
        stack![Axis(0), self.view(), Array::ones(1)]
    }

    fn from_homogeneous(&self) -> ArrayView<'_, A, Ix1> {
        let n = self.shape()[0];
        self.slice(s![0..n - 1])
    }
}

impl<A> Homogeneous<A, Ix2> for Array<A, Ix2> where A: LinalgScalar {
    fn to_homogeneous(&self) -> Array<A, Ix2> {
        let n = self.shape()[0];
        stack![Axis(1), self.view(), Array::ones((n, 1))]
    }

    fn from_homogeneous(&self) -> ArrayView<'_, A, Ix2> {
        let n_cols = self.shape()[1];
        self.slice(s![.., 0..n_cols - 1])
    }
}

pub fn to_homogeneous_vec(x: ArrayView1<'_, f64>) -> Array1<f64> {
    stack![Axis(0), x, Array::ones(1)]
}

pub fn to_homogeneous_vecs(xs: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = xs.shape()[0];
    stack![Axis(1), xs, Array::ones((n, 1))]
}

pub fn from_homogeneous_vec<'a>(x: &'a ArrayView1<'a, f64>) -> ArrayView1<'a, f64> {
    let n = x.shape()[0];
    x.slice(s![0..n - 1])
}

pub fn from_homogeneous_vecs<'a>(xs: &'a ArrayView2<'a, f64>) -> ArrayView2<'a, f64> {
    let n_cols = xs.shape()[1];
    xs.slice(s![.., 0..n_cols - 1])
}

#[cfg(test)]
mod tests {
    use super::*; // import names from outer scope

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
