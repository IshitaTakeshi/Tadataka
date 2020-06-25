use crate::camera::CameraParameters;
use ndarray::Array2;
use pyo3::prelude::{pyclass, PyObject};

#[pyclass]
#[derive(Clone)]
pub struct Frame {
    pub camera_params: CameraParameters,
    pub image: Array2<f64>,
    pub transform: Array2<f64>,  // transform from frame to world
}
