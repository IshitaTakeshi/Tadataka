#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "tadataka/_projection.hpp"


namespace py = pybind11;


double calc_ref_depth(
    Eigen::Ref<const Eigen::Matrix<double, 4, 4>> T_rk,
    Eigen::Ref<const Eigen::Vector2d> x_key,
    const double depth_key) {
  const Eigen::Vector3d p_key = inv_pi(x_key, depth_key);
  const Eigen::Vector3d r_rk_z = T_rk.block<1, 3>(2, 0);
  const double t_rk_z = T_rk(2, 3);
  return r_rk_z.dot(p_key) + t_rk_z;
}


PYBIND11_MODULE(_depth, m) {
  m.def("calc_ref_depth", &calc_ref_depth,
        py::return_value_policy::reference_internal);
}
