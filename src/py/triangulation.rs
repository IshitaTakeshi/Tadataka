use crate::triangulation;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::{pyfunction, pymodule, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;


#[pyfunction]
fn calc_depth0(
    _py: Python<'_>,
    transform_10: &PyArray2<f64>,
    x0: &PyArray1<f64>,
    x1: &PyArray1<f64>,
) -> f64 {
    triangulation::calc_depth0(
        &transform_10.as_array(),
        &x0.as_array(),
        &x1.as_array()
    )
}

#[pymodule(triangulation)]
fn triangulation_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(calc_depth0))?;

    Ok(())
}
