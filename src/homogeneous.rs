use ndarray::{arr1, arr2, stack, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};

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
    fn test_to_homogeneous_vecs() {
        let points = arr2(&[[2., 3.], [4., 5.]]);
        let expected = arr2(&[[2., 3., 1.], [4., 5., 1.]]);
        assert_eq!(to_homogeneous_vecs(points.view()), expected);
    }

    #[test]
    fn test_to_homogeneous_vec() {
        let p = arr1(&[2., 3.]);
        let expected = arr1(&[2., 3., 1.]);
        assert_eq!(to_homogeneous_vec(p.view()), expected);
    }

    #[test]
    fn test_from_homogeneous_vecs() {
        let points = arr2(&[[2., 3., 1.], [4., 5., 1.]]);
        let expected = arr2(&[[2., 3.], [4., 5.]]);
        assert_eq!(from_homogeneous_vecs(&points.view()), expected);
    }
}
