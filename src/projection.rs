use crate::homogeneous::Homogeneous;
use ndarray::{arr1, arr2, Array, Array1, Array2, ArrayBase,
              ArrayView1, ArrayView2, Data, Ix1, Ix2, LinalgScalar};

static EPSILON: f64 = 1e-16;

pub trait Projection<D, DepthType> {
    fn project(&self) -> Array<f64, D>;
    fn inv_project(&self, depth: DepthType) -> Array<f64, D>;
}

fn project_impl(x: ArrayView1<'_, f64>) -> Array1<f64> {
    let z = x[2] + EPSILON;
    &x.slice(s![0..2]) / z
}

fn inv_project_impl(x: ArrayView1<'_, f64>, depth: f64) -> Array1<f64> {
    x.to_homogeneous() * depth
}

impl<S> Projection<Ix1, f64> for ArrayBase<S, Ix1>
where
    S: Data<Elem = f64>,
{
    fn project(&self) -> Array<f64, Ix1> {
        project_impl(self.view())
    }

    fn inv_project(&self, depth: f64) -> Array1<f64> {
        inv_project_impl(self.view(), depth)
    }
}

impl<S> Projection<Ix2, ArrayView1<'_, f64>> for ArrayBase<S, Ix2>
where
    S: Data<Elem = f64>
{
    fn project(&self) -> Array<f64, Ix2> {
        let n = self.shape()[0];
        let mut us = Array::zeros((n, 2));
        for i in 0..n {
            let u = project_impl(self.slice(s![i, ..]));
            us.slice_mut(s![i, ..]).assign(&u);
        }
        us
    }

    fn inv_project(&self, depths: ArrayView1<'_, f64>) -> Array2<f64> {
        let n = self.shape()[0];
        let mut ps = Array::zeros((n, 3));
        for i in 0..n {
            let p = inv_project_impl(self.slice(s![i, ..]), depths[i]);
            ps.slice_mut(s![i, ..]).assign(&p);
        }
        ps
    }
}

#[cfg(test)]
mod tests {
    use super::*; // import names from outer scope

    #[test]
    fn test_projection() {
        let points = arr2(
            &[[0., 0., 0.],
              [1., 4., 2.],
              [-1., 3., 5.]]
        );

        assert_eq!(Projection::project(&points),
                   arr2(&[[0., 0.],
                          [0.5, 2.],
                          [-0.2, 0.6]]));

        assert_eq!(Projection::project(&arr1(&[0., 0., 0.])), arr1(&[0., 0.]));
        assert_eq!(Projection::project(&arr1(&[3., 5., 5.])), arr1(&[0.6, 1.]));
    }

    #[test]
    fn test_inv_projection() {
        let xs = arr2(
            &[[0.5, 2.],
              [-0.2, 0.6]]
        );
        let depths = arr1(&[2., 5.]);

        assert_eq!(Projection::inv_project(&xs, depths.view()),
                   arr2(&[[1., 4., 2.],
                          [-1., 3., 5.]]));

        let x = arr1(&[0.5, 2.]);
        let depth = 2.;
        assert_eq!(Projection::inv_project(&x, depth),
                   arr1(&[1., 4., 2.]));
    }
}
