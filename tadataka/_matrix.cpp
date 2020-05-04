#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


namespace py = pybind11;

using Matrix33d = Eigen::Matrix<double, 3, 3>;
using Matrix44d = Eigen::Matrix<double, 4, 4>;


inline Matrix33d get_rotation(Eigen::Ref<const Matrix44d> T) {
  return T.block<3, 3>(0, 0);
}


inline Eigen::Vector3d get_translation(Eigen::Ref<const Matrix44d> T) {
  return T.block<3, 1>(0, 3);
}


PYBIND11_MODULE(_matrix, m) {
  m.def("get_rotation", &get_rotation,
        py::return_value_policy::reference_internal);
  m.def("get_translation", &get_translation,
        py::return_value_policy::reference_internal);
}
