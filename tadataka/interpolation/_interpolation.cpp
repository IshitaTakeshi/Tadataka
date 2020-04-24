#include "tadataka/interpolation/_bilinear.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

template<int rows, int cols>
using RowMajorMatrixXd = Eigen::Matrix<double, rows, cols, Eigen::RowMajor>;


Eigen::VectorXd interpolation(
    const RowMajorMatrixXd<Eigen::Dynamic, Eigen::Dynamic> &image,
    const RowMajorMatrixXd<Eigen::Dynamic, 2> &coordinates) {
  const int N = coordinates.rows();
  Eigen::VectorXd intensities(N);
  _interpolation(image.data(), image.cols(),
                 coordinates.data(), N, intensities.data());
  return intensities;
}


PYBIND11_MODULE(_interpolation, m) {
  m.def("interpolation", &interpolation);
}
