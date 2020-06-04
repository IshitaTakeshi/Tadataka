use pyo3::prelude::*;
use crate::camera::CameraParameters;

#[pymodule(camera)]
fn mymodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CameraParameters>()?;
    Ok(())
}
