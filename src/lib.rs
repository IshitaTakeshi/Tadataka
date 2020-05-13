#![feature(test)]
#[macro_use(s)]
extern crate ndarray;
extern crate test;

use ndarray::{stack, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};

static EPSILON: f64 = 1e-16;

fn to_homogeneous_vec(x: ArrayView1<'_, f64>) -> Array1<f64> {
    stack![Axis(0), x, Array::ones(1)]
}

fn to_homogeneous_vecs(xs: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = xs.shape()[0];
    stack![Axis(1), xs, Array::ones((n, 1))]
}

fn project_vec(x: ArrayView1<'_, f64>) -> Array1<f64> {
    let z = x[2] + EPSILON;
    &x.slice(s![0..2]) / z
}

fn project_vecs(xs: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = xs.shape()[0];
    let mut us = Array::zeros((n, 2));
    for i in 0..n {
        let u = project_vec(xs.slice(s![i, ..]));
        us.slice_mut(s![i, ..]).assign(&u);
    }
    us
}

fn inv_project_vec(x: ArrayView1<'_, f64>, depth: f64) -> Array1<f64> {
    to_homogeneous_vec(x) * depth
}

fn inv_project_vecs(xs: ArrayView2<'_, f64>, depths: ArrayView1<'_, f64>) -> Array2<f64> {
    let n = xs.shape()[0];
    let mut ps = Array::zeros((n, 3));
    for i in 0..n {
        let p = inv_project_vec(xs.slice(s![i, ..]), depths[i]);
        ps.slice_mut(s![i, ..]).assign(&p);
    }
    ps
}

fn from_homogeneous_vec<'a>(x: &'a ArrayView1<'a, f64>) -> ArrayView1<'a, f64> {
    let n = x.shape()[0];
    x.slice(s![0..n - 1])
}

fn from_homogeneous_vecs<'a>(xs: &'a ArrayView2<'a, f64>) -> ArrayView2<'a, f64> {
    let n_cols = xs.shape()[1];
    xs.slice(s![.., 0..n_cols-1])
}

fn transform(transform10: ArrayView2<'_, f64>, points0: ArrayView2<'_, f64>)
    -> Array2<f64> {
    let points0 = to_homogeneous_vecs(points0);
    let points0 = points0.t();
    let points1 = transform10.dot(&points0);
    let points1 = points1.t();
    from_homogeneous_vecs(&points1.view()).to_owned()
}

fn warp_(
    transform10: ArrayView2<'_, f64>,
    xs0: ArrayView2<'_, f64>,
    depths0: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let points0 = inv_project_vecs(xs0, depths0);
    let points1 = transform(transform10, points0.view());
    project_vecs(points1.view())
}

#[pymodule]
fn warp(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "warp")]
    fn warp_py(
        py: Python<'_>,
        transform10: &PyArray2<f64>,
        xs: &PyArray2<f64>,
        depths: &PyArray1<f64>,
    ) -> Py<PyArray2<f64>> {
        warp_(transform10.as_array(), xs.as_array(), depths.as_array())
            .into_pyarray(py)
            .to_owned()
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
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

    #[test]
    fn test_transform() {
        let P0 = Array::from_shape_vec(
            (2, 3),
            vec![1.,  2., 5.,
                 4., -2., 3.]
        ).unwrap();

        let T10 = Array::from_shape_vec(
            (4, 4),
            vec![1., 0.,  0., 1.,
                 0., 0., -1., 2.,
                 0., 1.,  0., 3.,
                 0., 0.,  0., 1.]
        ).unwrap();

        let P1 = Array::from_shape_vec(
            (2, 3),
            vec![2., -3., 5.,
                 5., -1., 1.]
        ).unwrap();

        assert_eq!(transform(T10.view(), P0.view()), P1)
    }

    #[test]
    fn test_to_homogeneous_vecs() {
        let P = Array::from_shape_vec(
            (2, 2),
            vec![2., 3.,
                 4., 5.]
        ).unwrap();
        let expected = Array::from_shape_vec(
            (2, 3),
            vec![2., 3., 1.,
                 4., 5., 1.]
        ).unwrap();
        assert_eq!(to_homogeneous_vecs(P.view()), expected);
    }

    #[test]
    fn test_to_homogeneous_vec() {
        let p = Array::from_shape_vec(2, vec![2., 3.]).unwrap();
        let expected = Array::from_shape_vec(3, vec![2., 3., 1.]).unwrap();
        assert_eq!(to_homogeneous_vec(p.view()), expected);
    }

    #[test]
    fn test_from_homogeneous_vecs() {
        let P = Array::from_shape_vec(
            (2, 3),
            vec![2., 3., 1.,
                 4., 5., 1.]
        ).unwrap();
        let expected = Array::from_shape_vec(
            (2, 2),
            vec![2., 3.,
                 4., 5.]
        ).unwrap();
        assert_eq!(from_homogeneous_vecs(&P.view()), expected);
    }
}


#[pymodule]
fn homogeneous(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "to_homogeneous_vec")]
    fn to_homogeneous_vec_py(py: Python<'_>, x: &PyArray1<f64>) -> Py<PyArray1<f64>> {
        to_homogeneous_vec(x.as_array()).into_pyarray(py).to_owned()
    }

    #[pyfn(m, "to_homogeneous_vecs")]
    fn to_homogeneous_vecs_py(py: Python<'_>, xs: &PyArray2<f64>) -> Py<PyArray2<f64>> {
        to_homogeneous_vecs(xs.as_array())
            .into_pyarray(py)
            .to_owned()
    }

    Ok(())
}

#[pymodule]
fn projection(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "project_vec")]
    fn project_vec_py(py: Python<'_>, x: &PyArray1<f64>) -> Py<PyArray1<f64>> {
        project_vec(x.as_array()).into_pyarray(py).to_owned()
    }

    #[pyfn(m, "project_vecs")]
    fn project_vecs_py(py: Python<'_>, xs: &PyArray2<f64>) -> Py<PyArray2<f64>> {
        project_vecs(xs.as_array()).into_pyarray(py).to_owned()
    }

    #[pyfn(m, "inv_project_vec")]
    fn inv_project_vec_py(py: Python<'_>, x: &PyArray1<f64>, depth: f64) -> Py<PyArray1<f64>> {
        inv_project_vec(x.as_array(), depth)
            .into_pyarray(py)
            .to_owned()
    }

    #[pyfn(m, "inv_project_vecs")]
    fn inv_project_vecs_py(
        py: Python<'_>,
        xs: &PyArray2<f64>,
        depths: &PyArray1<f64>,
    ) -> Py<PyArray2<f64>> {
        inv_project_vecs(xs.as_array(), depths.as_array())
            .into_pyarray(py)
            .to_owned()
    }

    Ok(())
}

#[pymodule]
fn interpolation(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn interpolation(image: ArrayView2<'_, f64>, cx: f64, cy: f64) -> f64 {
        let lx = cx.floor();
        let ly = cy.floor();
        let lxi = lx as usize;
        let lyi = ly as usize;

        if lx == cx && ly == cy {
            return image[[lyi, lxi]];
        }

        let ux = lx + 1.0;
        let uy = ly + 1.0;
        let uxi = ux as usize;
        let uyi = uy as usize;

        if lx == cx {
            return image[[lyi, lxi]] * (ux - cx) * (uy - cy)
                + image[[uyi, lxi]] * (ux - cx) * (cy - ly);
        }

        if ly == cy {
            return image[[lyi, lxi]] * (ux - cx) * (uy - cy)
                + image[[lyi, uxi]] * (cx - lx) * (uy - cy);
        }

        image[[lyi, lxi]] * (ux - cx) * (uy - cy)
            + image[[lyi, uxi]] * (cx - lx) * (uy - cy)
            + image[[uyi, lxi]] * (ux - cx) * (cy - ly)
            + image[[uyi, uxi]] * (cx - lx) * (cy - ly)
    }

    #[pyfn(m, "interpolation")]
    fn py_interpolation(
        py: Python<'_>,
        image: &PyArray2<f64>,
        coordinates: &PyArray2<f64>,
    ) -> Py<PyArray1<f64>> {
        let image = image.as_array();
        let coordinates = coordinates.as_array();
        let n = coordinates.shape()[0];
        let mut intensities = Array1::zeros(n);
        for i in 0..n {
            intensities[i] = interpolation(image, coordinates[[i, 0]], coordinates[[i, 1]]);
        }
        intensities.into_pyarray(py).to_owned()
    }

    Ok(())
}
