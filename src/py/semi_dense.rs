use crate::camera::CameraParameters;
use crate::semi_dense::semi_dense;
use crate::semi_dense::{Frame, Params};
use crate::semi_dense::numeric::Inverse;
use crate::semi_dense::VarianceCoefficients;
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2};
use numpy::types::TypeNum;
use numpy::npyffi;
use ndarray::{Ix1, Dimension};
use pyo3::PyDowncastError;
use pyo3::types::{PyAny, PyList, PyTuple};
use pyo3::prelude::{pyfunction, pymodule, PyModule, PyResult,
                    Py, Python, pymethods, ToPyObject, PyObject, FromPyObject};
use pyo3::wrap_pyfunction;

#[pymethods]
impl Frame {
    #[new]
    pub fn new(
        _py: Python<'_>,
        camera_params: &PyAny,
        image: &PyArray2<f64>,
        transform: &PyArray2<f64>,
    ) -> PyResult<Self> {
        let focal_length = camera_params.getattr("focal_length")?;
        let offset = camera_params.getattr("offset")?;

        let focal_length: &PyArray1<f64> = FromPyObject::extract(focal_length)?;
        let offset: &PyArray1<f64> = FromPyObject::extract(offset)?;

        let focal_length = focal_length.as_array();
        let offset = offset.as_array();

        let c = CameraParameters::new(
            (focal_length[0], focal_length[1]),
            (offset[0], offset[1])
        );
        Ok(Frame {
            camera_params: c,
            image: image.as_array().to_owned(),
            transform: transform.as_array().to_owned()
        })
    }
}

#[pymethods]
impl Params {
    #[new]
    pub fn new(
        min_depth: f64,
        max_depth: f64,
        geo_coeff: f64,
        photo_coeff: f64,
        ref_step_size: f64,
        min_gradient: f64,
    ) -> Self {
        Params {
            inv_depth_range: (max_depth.inv(), min_depth.inv()),
            var_coeffs: VarianceCoefficients { geo: geo_coeff, photo: photo_coeff },
            ref_step_size: ref_step_size,
            min_gradient: min_gradient
        }
    }
}

enum Ret {
    I64(Py<PyArray2<i64>>),
    F64(Py<PyArray2<f64>>),
}

impl ToPyObject for Ret {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        match self {
            Ret::I64(v) => return v.to_object(py),
            Ret::F64(v) => return v.to_object(py),
        }
    }
}

#[pyfunction]
fn update_depth<'a>(
    py: Python<'a>,
    keyframe: &Frame,
    refframe_pylist: &PyList,
    age_map: &PyArray2<usize>,
    prior_depth: &PyArray2<f64>,
    prior_variance: &PyArray2<f64>,
    params: &Params,
) -> PyResult<&'a PyTuple> {
    let mut refframes = Vec::new();
    for f in refframe_pylist {
        let c = Frame::extract(f).unwrap();
        refframes.push(c);
    }

    let (flag_map, depth_map, variance_map) = semi_dense::update_depth(
        keyframe,
        &refframes,
        &age_map.as_array().to_owned(),
        &prior_depth.as_array().to_owned(),
        &prior_variance.as_array().to_owned(),
        &params
    );

    let mut ret = Vec::new();
    ret.push(Ret::I64(flag_map.into_pyarray(py).to_owned()));
    ret.push(Ret::F64(depth_map.into_pyarray(py).to_owned()));
    ret.push(Ret::F64(variance_map.into_pyarray(py).to_owned()));
    Ok(PyTuple::new(py, ret))
}

#[pymodule(semi_dense)]
fn semi_dense_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Frame>()?;
    m.add_class::<Params>()?;
    m.add_wrapped(wrap_pyfunction!(update_depth))?;

    Ok(())
}
