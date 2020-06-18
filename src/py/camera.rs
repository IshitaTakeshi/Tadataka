use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::{pyclass, pymethods, pymodule, Py, Python, PyObject, PyModule, PyResult};
use pyo3::type_object::PyTypeObject;
use crate::camera::CameraParameters;

#[pyclass]
#[derive(Clone)]
pub struct PyCameraParameters {
    pub inner: CameraParameters
}

#[pymethods]
impl PyCameraParameters {
    #[new]
    fn new(
        _py: Python<'_>,
        focal_length: (f64, f64),
        offset: (f64, f64)
    ) -> Self {
        PyCameraParameters { inner: CameraParameters::new(focal_length, offset) }
    }

    #[getter]
    fn focal_length(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        let focal_length = self.inner.focal_length.to_owned();
        focal_length.into_pyarray(py).to_owned()
    }

    #[getter]
    fn offset(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        let offset = self.inner.offset.to_owned();
        offset.into_pyarray(py).to_owned()
    }
}

#[pymodule(camera)]
fn mymodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("CameraParameters", <PyCameraParameters as PyTypeObject>::type_object())?;

    Ok(())
}
