use crate::camera::CameraParameters;
use crate::semi_dense::{Flag, Frame, Hypothesis, Params, VarianceCoefficients};
use crate::semi_dense::gradient::ImageGradient;
use crate::semi_dense::hypothesis;
use crate::semi_dense::numeric::Inverse;
use crate::semi_dense::semi_dense;
use ndarray::arr1;
use numpy::{IntoPyArray, PyArray1, PyArray2};
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

// interface for debug
#[pyfunction]
fn estimate_debug_<'a>(
    _py: Python<'a>,
    u_key: &PyArray1<i64>,
    prior_depth: f64,
    prior_variance: f64,
    keyframe: &Frame,
    refframe: &Frame,
    params: &Params,
) -> (f64, f64, i64) {
    let image_grad = ImageGradient::new(&keyframe.image);

    let result = hypothesis::check_args(prior_depth.inv(),
                                        prior_variance, params.inv_depth_range);
    if let Err(flag) = result {
        return (prior_depth, prior_variance, flag as i64);
    }

    let prior = Hypothesis::new(prior_depth.inv(), prior_variance,
                                params.inv_depth_range);

    let u_key = u_key.as_array();
    let u_key = arr1(&[u_key[0] as f64, u_key[1] as f64]);
    let result = semi_dense::estimate(&u_key, &prior,
                                      keyframe, refframe, &image_grad, params);
    match result {
        Err(flag) => return (prior_depth, prior_variance, flag as i64),
        Ok(h) => return (h.inv_depth.inv(), h.variance, Flag::Success as i64),
    };
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
    m.add_wrapped(wrap_pyfunction!(estimate_debug_))?;
    m.add_wrapped(wrap_pyfunction!(update_depth))?;

    Ok(())
}
