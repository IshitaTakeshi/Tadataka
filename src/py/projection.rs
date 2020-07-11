use crate::projection::Projection;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::{pyfunction, pymodule, Py, PyModule, PyResult, Python};
use pyo3::wrap_pyfunction;
use ndarray::{Ix1, Ix2, Array};

#[pyfunction]
fn project_vec(py: Python<'_>, x: &PyArray1<f64>) -> Py<PyArray1<f64>> {
    Projection::<Ix1, f64>::project(&x.as_array())
        .into_pyarray(py)
        .to_owned()
}

#[pyfunction]
fn project_vecs(py: Python<'_>, xs: &PyArray2<f64>) -> Py<PyArray2<f64>> {
    Projection::<Ix2, &Array<f64, Ix1>>::project(&xs.as_array())
        .into_pyarray(py)
        .to_owned()
}

#[pyfunction]
fn inv_project_vec(
    py: Python<'_>,
    x: &PyArray1<f64>,
    depth: f64
) -> Py<PyArray1<f64>> {
    Projection::inv_project(&x.as_array(), depth)
        .into_pyarray(py)
        .to_owned()
}

#[pyfunction]
fn inv_project_vecs(
    py: Python<'_>,
    xs: &PyArray2<f64>,
    depths: &PyArray1<f64>,
) -> Py<PyArray2<f64>> {
    Projection::inv_project(&xs.as_array(), &depths.as_array())
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
