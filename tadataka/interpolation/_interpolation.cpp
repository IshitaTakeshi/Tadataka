#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "tadataka/interpolation/_bilinear.hpp"
#include "tadataka/_types.hpp"

namespace py = pybind11;


Eigen::VectorXd interpolation(
    Eigen::Ref<const RowMajorMatrixXd<Eigen::Dynamic, Eigen::Dynamic>> &image,
    Eigen::Ref<const RowMajorMatrixXd<Eigen::Dynamic, 2>> &coordinates) {
  const int N = coordinates.rows();
  Eigen::VectorXd intensities(N);
  _interpolation(image.data(), image.cols(),
                 coordinates.data(), N, intensities.data());
  return intensities;
}


PYBIND11_MODULE(_interpolation, m) {
  m.def("interpolation", &interpolation,
        py::return_value_policy::reference_internal);
}
