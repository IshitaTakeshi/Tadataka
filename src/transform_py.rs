use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::{pyfunction, pymodule, Py, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;

#[pyfunction]
fn transform(py: Python<'_>, transform10: &PyArray2<f64>,
             points0: &PyArray2<f64>) -> Py<PyArray2<f64>> {
    crate::transform::transform(transform10.as_array(), points0.as_array())
        .into_pyarray(py).to_owned()
}

#[pymodule(transform)]
fn transform_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(transform))?;

    Ok(())
}
