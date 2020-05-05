#include <pybind11/pybind11.h>

#include "_matrix.hpp"
#include "_transform.hpp"


namespace py = pybind11;


void transform(
    Eigen::Ref<const RowMajorMatrixXd<4, 4>> T10,
    Eigen::Ref<const Vectors3D> P0,
    Eigen::Ref<Vectors3D> P1) {

  const auto R10 = get_rotation(T10);
  const auto t10 = get_translation(T10);

  for(int i = 0; i < P0.rows(); i++) {
    auto p0 = P0.row(i).transpose();
    auto p1 = R10 * p0 + t10;
    P1.row(i) = p1.transpose();
  }
}


PYBIND11_MODULE(_transform, m) {
  m.def("transform", &transform,
        py::return_value_policy::reference_internal);
}
