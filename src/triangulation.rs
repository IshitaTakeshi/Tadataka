use ndarray::{arr1, arr2, Array1, Array2, ArrayBase,
              ArrayView1, Data, Ix1, Ix2};
use ndarray_linalg::solve::Inverse;
use crate::homogeneous::Homogeneous;
use crate::projection::Projection;
use crate::transform::{get_rotation, get_translation, make_matrix, Transform};

static EPSILON: f64 = 1e-16;

#[inline]
fn calc_depth0_(
    x0: &ArrayView1<f64>,
    x1_i: f64,
    r10_i: &ArrayView1<'_, f64>,
    r10_z: &ArrayView1<'_, f64>,
    t10_i: f64,
    t10_z: f64
) -> f64 {
    let y0 = x0.to_homogeneous();
    let n = t10_i - t10_z * x1_i;
    let d = r10_z.dot(&y0) * x1_i - r10_i.dot(&y0);
    n / (d + EPSILON)
}

pub fn calc_depth0<
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>
>(
    transform_10: &ArrayBase<S1, Ix2>,
    x0: &ArrayBase<S2, Ix1>,
    x1: &ArrayBase<S3, Ix1>
) -> f64 {
    let rot10 = get_rotation(&transform_10);
    let t10 = get_translation(&transform_10);
    let i = if f64::abs(t10[0]) > f64::abs(t10[1]) { 0 } else { 1 };
    calc_depth0_(&x0.view(), x1[i],
                 &rot10.row(i), &rot10.row(2), t10[i], t10[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_depth0() {
        fn run(
            transform_w0: Array2<f64>,
            transform_w1: Array2<f64>,
            point: Array1<f64>,
        ) {
            let transform_0w = transform_w0.inv().unwrap();
            let transform_1w = transform_w1.inv().unwrap();
            let p0 = transform_0w.transform(&point);
            let p1 = transform_1w.transform(&point);
            let x0 = Projection::project(&p0);
            let x1 = Projection::project(&p1);

            let transform_10 = transform_1w.dot(&transform_w0);
            let depth = calc_depth0(&transform_10, &x0.view(), &x1.view());

            assert_eq!(depth, p0[2]);
        }

        // rotvec = [0, np.pi/2, 0]
        let rotation0 = arr2(
            &[[0., 0., 1.],
              [0., 1., 0.],
              [-1., 0., 0.]]
        );

        // rotvec = [0, -np.pi/2, 0]
        let rotation1 = arr2(
            &[[0., 0., -1.],
              [0., 1., 0.],
              [1., 0., 0.]]
        );
        let translation0 = arr1(&[-3., 0., 1.]);
        let translation1 = arr1(&[0., 0., 2.]);
        let transform_w0 = make_matrix(&rotation0, &translation0);
        let transform_w1 = make_matrix(&rotation1, &translation1);
        let point = arr1(&[-1., 0., 1.]);

        run(transform_w0, transform_w1, point);
    }
}
