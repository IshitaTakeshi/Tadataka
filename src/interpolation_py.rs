use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pyfunction, pymodule, Py, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;

#[pyfunction]
fn interpolation(
    py: Python<'_>,
    image: &PyArray2<f64>,
    coordinates: &PyArray2<f64>,
) -> Py<PyArray1<f64>> {
    let image = image.as_array();
    let coordinates = coordinates.as_array();
    let n = coordinates.shape()[0];
    let mut intensities = Array1::zeros(n);
    for i in 0..n {
        intensities[i] = crate::interpolation::interpolation(image, coordinates.row(i));
    }
    intensities.into_pyarray(py).to_owned()
}

#[pymodule(interpolation)]
fn interpolation_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(interpolation))?;

    Ok(())
}
