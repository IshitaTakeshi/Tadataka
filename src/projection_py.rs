use crate::projection;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pyfunction, pymodule, Py, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;

#[pyfunction]
fn project_vec(py: Python<'_>, x: &PyArray1<f64>) -> Py<PyArray1<f64>> {
    projection::project_vec(x.as_array())
        .into_pyarray(py)
        .to_owned()
}

#[pyfunction]
fn project_vecs(py: Python<'_>, xs: &PyArray2<f64>) -> Py<PyArray2<f64>> {
    projection::project_vecs(xs.as_array())
        .into_pyarray(py)
        .to_owned()
}

#[pyfunction]
fn inv_project_vec(py: Python<'_>, x: &PyArray1<f64>, depth: f64) -> Py<PyArray1<f64>> {
    projection::inv_project_vec(x.as_array(), depth)
        .into_pyarray(py)
        .to_owned()
}

#[pyfunction]
fn inv_project_vecs(
    py: Python<'_>,
    xs: &PyArray2<f64>,
    depths: &PyArray1<f64>,
) -> Py<PyArray2<f64>> {
    projection::inv_project_vecs(xs.as_array(), depths.as_array())
        .into_pyarray(py)
        .to_owned()
}

#[pymodule(projection)]
fn projection_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(project_vec))?;
    m.add_wrapped(wrap_pyfunction!(project_vecs))?;
    m.add_wrapped(wrap_pyfunction!(inv_project_vec))?;
    m.add_wrapped(wrap_pyfunction!(inv_project_vecs))?;

    Ok(())
}
