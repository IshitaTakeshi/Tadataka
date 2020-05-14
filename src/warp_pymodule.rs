use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pyfunction, pymodule, Py, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;

#[pyfunction]
fn warp(py: Python<'_>, transform10: &PyArray2<f64>,
        xs: &PyArray2<f64>, depths: &PyArray1<f64>)
    -> Py<PyArray2<f64>> {
    crate::warp::warp(
        transform10.as_array(),
        xs.as_array(),
        depths.as_array())
        .into_pyarray(py)
        .to_owned()
}

#[pymodule(warp)]
fn warp_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(warp))?;
    Ok(())
}
