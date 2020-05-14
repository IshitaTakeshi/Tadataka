use crate::homogeneous::to_homogeneous_vec;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2};

static EPSILON: f64 = 1e-16;

pub fn project_vec(x: ArrayView1<'_, f64>) -> Array1<f64> {
    let z = x[2] + EPSILON;
    &x.slice(s![0..2]) / z
}

pub fn project_vecs(xs: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = xs.shape()[0];
    let mut us = Array::zeros((n, 2));
    for i in 0..n {
        let u = project_vec(xs.slice(s![i, ..]));
        us.slice_mut(s![i, ..]).assign(&u);
    }
    us
}

pub fn inv_project_vec(x: ArrayView1<'_, f64>, depth: f64) -> Array1<f64> {
    to_homogeneous_vec(x) * depth
}

pub fn inv_project_vecs(xs: ArrayView2<'_, f64>, depths: ArrayView1<'_, f64>)
    -> Array2<f64> {
    let n = xs.shape()[0];
    let mut ps = Array::zeros((n, 3));
    for i in 0..n {
        let p = inv_project_vec(xs.slice(s![i, ..]), depths[i]);
        ps.slice_mut(s![i, ..]).assign(&p);
    }
    ps
}
