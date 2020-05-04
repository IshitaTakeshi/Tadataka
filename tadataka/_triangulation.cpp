#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "tadataka/_homogeneous.h"


namespace py = pybind11;

const double EPSILON = 1e-16;


double f(Eigen::Ref<const Eigen::VectorXd> x0,
         const double x1_i,
         Eigen::Ref<const Eigen::Vector3d> r10_i,
         Eigen::Ref<const Eigen::Vector3d> r10_z,
         const double t10_i, const double t10_z) {
  const Eigen::Vector3d y0 = to_homogeneous_vector(x0);
  double n = t10_i - t10_z * x1_i;
  double d = r10_z.dot(y0) * x1_i - r10_i.dot(y0);
  return n / (d + EPSILON);
}


double calc_depth0_(
    Eigen::Ref<const Eigen::Matrix<double, 3, 3>> R10,
    Eigen::Ref<const Eigen::VectorXd> t10,
    Eigen::Ref<const Eigen::VectorXd> x0,
    Eigen::Ref<const Eigen::VectorXd> x1) {
  const int i = abs(t10[0]) > abs(t10[1]) ? 0 : 1;
  return f(x0, x1[i], R10.row(i), R10.row(2), t10[i], t10[2]);
}


PYBIND11_MODULE(_triangulation, m) {
  m.def("calc_depth0_", &calc_depth0_,
        py::return_value_policy::reference_internal);
}
