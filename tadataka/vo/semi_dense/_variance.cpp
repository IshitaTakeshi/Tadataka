#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "tadataka/_homogeneous.hpp"


namespace py = pybind11;


double calc_alpha_(Eigen::Ref<const Eigen::VectorXd> x_key,
                   const double x_ref_i, const double direction_i,
                   Eigen::Ref<const Eigen::Vector3d> ri,
                   Eigen::Ref<const Eigen::Vector3d> rz,
                   const double ti, const double tz) {
  Eigen::Ref<const Eigen::Vector3d> y = to_homogeneous_vector(x_key);

  const double d = rz.dot(y) * ti - ri.dot(y) * tz;
  const double n = x_ref_i * tz - ti;

  return direction_i * d / (n * n);
}


PYBIND11_MODULE(_variance, m) {
  m.def("calc_alpha_", &calc_alpha_,
        py::return_value_policy::reference_internal);
}
