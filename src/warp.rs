use crate::projection::{inv_project_vecs, project_vecs};
use crate::transform::transform;
use ndarray::{Array2, ArrayView1, ArrayView2};

pub fn warp(
    transform10: ArrayView2<'_, f64>,
    xs0: ArrayView2<'_, f64>,
    depths0: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let points0 = inv_project_vecs(xs0, depths0);
    let points1 = transform(transform10, points0.view());
    project_vecs(points1.view())
}

#[cfg(test)]
mod tests {
    use super::*;  // import names from outer scope
    use test::Bencher;
    #[test]
    fn test_warp() {
        let T10 = Array::from_shape_vec(
            (4, 4),
            vec![ 0., 0., 1., 0.,
                  0., 1., 0., 0.,
                 -1., 0., 0., 4.,
                  0., 0., 0., 1.]
        ).unwrap();
        let xs0 = Array::from_shape_vec(
            (2, 2),
            vec![0.,  0.,
                 2., -1.]
        ).unwrap();
        let depths0 = Array::from_shape_vec(2, vec![2., 4.]).unwrap();
        let xs1 = Array::from_shape_vec(
            (2, 2),
            vec![0.5, 0.0,
                 -1.0, 1.0]
        ).unwrap();
        assert_eq!(warp_(T10.view(), xs0.view(), depths0.view()), xs1);
    }

    #[bench]
    fn bench_warp(b: &mut Bencher) {
        let T10 = Array::from_shape_vec(
            (4, 4),
            vec![ 0., 0., 1., 0.,
                  0., 1., 0., 0.,
                 -1., 0., 0., 4.,
                  0., 0., 0., 1.]
        ).unwrap();
        let xs0 = Array::from_shape_vec(
            (2, 2),
            vec![0.,  0.,
                 2., -1.]
        ).unwrap();
        let depths0 = Array::from_shape_vec(2, vec![2., 4.]).unwrap();
        let xs1 = Array::from_shape_vec(
            (2, 2),
            vec![0.5, 0.0,
                 -1.0, 1.0]
        ).unwrap();
        let T10_view = T10.view();
        let xs0_view = xs0.view();
        let depths0_view = depths0.view();
        b.iter(|| warp_(T10_view, xs0_view, depths0_view))
    }
}
