use super::numeric::Inv;
use super::variance::VarianceCoefficients;
use pyo3::prelude::{pyclass, PyObject};

#[pyclass]
pub struct Params {
    pub inv_depth_range: (Inv, Inv),
    pub var_coeffs: VarianceCoefficients,
    pub ref_step_size: f64,
    pub min_gradient: f64,
}
