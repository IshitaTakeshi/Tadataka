use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pyfunction, pymodule, Py, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;
use crate::interpolation::Interpolation;

#[pyfunction]
fn interpolation(
    py: Python<'_>,
    image: &PyArray2<f64>,
    coordinates: &PyArray2<f64>,
) -> Py<PyArray1<f64>> {
    Interpolation::interpolate(&image.as_array(), &coordinates.as_array())
        .into_pyarray(py)
        .to_owned()
}

#[pymodule(interpolation)]
fn interpolation_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(interpolation))?;

    Ok(())
}
