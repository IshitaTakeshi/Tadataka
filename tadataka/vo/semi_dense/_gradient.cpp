#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


Eigen::VectorXd gradient1d(const Eigen::VectorXd &intensities) {
  const int n = intensities.size() - 1;
  return intensities.tail(n) - intensities.head(n);
}


double calc_gradient_norm(const Eigen::VectorXd &intensities) {
  return gradient1d(intensities).norm();
}


PYBIND11_MODULE(_gradient, m) {
  m.def("gradient1d", &gradient1d);
  m.def("calc_gradient_norm", &calc_gradient_norm);
}
