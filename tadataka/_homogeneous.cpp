#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// #include "_homogeneous.h"

namespace py = pybind11;


void to_homogeneous_(
    Eigen::Ref<const Eigen::VectorXd> x,
    Eigen::Ref<Eigen::VectorXd, Eigen::Unaligned, Eigen::InnerStride<>> y) {
  const int D = x.size();
  y(Eigen::seq(0, D-1)) = x;
  y(D) = 1.0;
}


const Eigen::VectorXd to_homogeneous_vector(
    Eigen::Ref<const Eigen::VectorXd> x) {
  Eigen::VectorXd y(x.size()+1);
  to_homogeneous_(x, y);
  return y;
}


const Eigen::MatrixXd to_homogeneous_vectors(
    Eigen::Ref<const Eigen::MatrixXd> xs) {
  const int N = xs.rows();
  const int D = xs.cols();

  Eigen::MatrixXd ys(N, D+1);
  for(int i = 0; i < N; i++) {
    to_homogeneous_(xs.row(i), ys.row(i));
  }
  return ys;
}


PYBIND11_MODULE(_homogeneous, m) {
  m.def("to_homogeneous_vectors", &to_homogeneous_vectors,
        py::return_value_policy::reference_internal);
  m.def("to_homogeneous_vector", &to_homogeneous_vector,
        py::return_value_policy::reference_internal);
}
