use ndarray::{Array1, ArrayView1};
use ndarray_linalg::Norm;

pub fn normalize(v: ArrayView1<'_, f64>) -> Array1<f64> {
    let norm = v.norm();
    v.map(|e| e / norm)
}
