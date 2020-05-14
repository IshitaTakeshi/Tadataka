use crate::homogeneous;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pyfunction, pymodule, Py, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;

#[pyfunction]
fn to_homogeneous_vec(py: Python<'_>, x: &PyArray1<f64>) -> Py<PyArray1<f64>> {
    homogeneous::to_homogeneous_vec(x.as_array())
        .into_pyarray(py)
        .to_owned()
}

#[pyfunction]
fn to_homogeneous_vecs(py: Python<'_>, xs: &PyArray2<f64>) -> Py<PyArray2<f64>> {
    homogeneous::to_homogeneous_vecs(xs.as_array())
        .into_pyarray(py)
        .to_owned()
}

#[pyfunction]
fn from_homogeneous_vec(py: Python<'_>, x: &PyArray1<f64>) -> Py<PyArray1<f64>> {
    homogeneous::from_homogeneous_vec(&x.as_array())
        .to_owned()
        .into_pyarray(py)
        .to_owned()
}

#[pyfunction]
fn from_homogeneous_vecs(py: Python<'_>, xs: &PyArray2<f64>) -> Py<PyArray2<f64>> {
    homogeneous::from_homogeneous_vecs(&xs.as_array())
        .to_owned()
        .into_pyarray(py)
        .to_owned()
}

#[pymodule(homogeneous)]
fn homogeneous_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(to_homogeneous_vec))?;
    m.add_wrapped(wrap_pyfunction!(to_homogeneous_vecs))?;
    m.add_wrapped(wrap_pyfunction!(from_homogeneous_vec))?;
    m.add_wrapped(wrap_pyfunction!(from_homogeneous_vecs))?;

    Ok(())
}
