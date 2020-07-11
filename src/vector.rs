use ndarray::{Array1, ArrayBase, Data, Ix1};
use ndarray_linalg::Norm;

pub fn normalize<S: Data<Elem = f64>>(v: &ArrayBase<S, Ix1>) -> Array1<f64> {
    let norm = v.norm();
    if norm == 0. {
        return v.to_owned();
    }

    v.map(|e| e / norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_normalize() {
        let v = arr1(&[4., 3.]);
        assert_eq!(normalize(&v), v / 5.0);

        let v = arr1(&[-1., 2.]);
        assert_eq!(normalize(&v), v / (5.0_f64).sqrt());

        let v = arr1(&[0., 0.]);
        assert_eq!(normalize(&v), v);
    }
}
