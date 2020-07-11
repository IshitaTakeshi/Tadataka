use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::types::PyTuple;
use pyo3::conversion::FromPy;
use pyo3::prelude::{pyfunction, pymodule, Py, PyModule, PyObject, PyResult, Python,
                    ToPyObject};
use pyo3::wrap_pyfunction;
use crate::warp::Warp;

enum Ret1D {
    X(Py<PyArray1<f64>>),
    DEPTH(f64),
}

impl ToPyObject for Ret1D {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        match self {
            Ret1D::X(v) => return v.to_object(py),
            Ret1D::DEPTH(v) => return v.to_object(py),
        }
    }
}

#[pyfunction]
fn warp_vec<'a>(
    py: Python<'a>,
    transform10: &PyArray2<f64>,
    x0: &PyArray1<f64>,
    depth0: f64,
) -> Py<PyTuple> {
    let (x1, depth1) = Warp::warp(&transform10.as_array(), &x0.as_array(), depth0);
    let mut ret = Vec::new();
    ret.push(Ret1D::X(x1.into_pyarray(py).to_owned()));
    ret.push(Ret1D::DEPTH(depth1));
    FromPy::from_py(PyTuple::new(py, ret), py)
}

enum Ret2D {
    X(Py<PyArray2<f64>>),
    DEPTH(Py<PyArray1<f64>>),
}

impl ToPyObject for Ret2D {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        match self {
            Ret2D::X(v) => return v.to_object(py),
            Ret2D::DEPTH(v) => return v.to_object(py),
        }
    }
}

#[pyfunction]
fn warp_vecs<'a>(
    py: Python<'a>,
    transform10: &PyArray2<f64>,
    xs: &PyArray2<f64>,
    depths: &PyArray1<f64>,
) -> Py<PyTuple> {
    let (xs1, depths1) = Warp::warp(
        &transform10.as_array(),
        &xs.as_array(),
        &depths.as_array()
    );
    let mut ret = Vec::new();
    ret.push(Ret2D::X(xs1.into_pyarray(py).to_owned()));
    ret.push(Ret2D::DEPTH(depths1.into_pyarray(py).to_owned()));
    FromPy::from_py(PyTuple::new(py, ret), py)
}

#[pymodule(warp)]
fn warp_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(warp_vecs))?;
    m.add_wrapped(wrap_pyfunction!(warp_vec))?;

    Ok(())
}
