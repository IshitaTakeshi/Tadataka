#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "tadataka/_types.hpp"
#include "tadataka/_homogeneous.hpp"
#include "tadataka/_matrix.hpp"
#include "tadataka/_projection.hpp"


namespace py = pybind11;


double calc_alpha_(Eigen::Ref<const Eigen::RowVectorXd> x_key,
                   const double x_ref_i, const double direction_i,
                   Eigen::Ref<const Eigen::RowVectorXd> ri,
                   Eigen::Ref<const Eigen::RowVectorXd> rz,
                   const double ti, const double tz) {
  Eigen::Ref<const Eigen::Vector3d> y = to_homogeneous_vector(x_key);

  const double d = rz.dot(y) * ti - ri.dot(y) * tz;
  const double n = x_ref_i * tz - ti;

  return direction_i * d / (n * n);
}


double calc_alpha(
    const Eigen::Ref<const RowMajorMatrixXd<4, 4>>& T_rk,
    const Eigen::Ref<const Eigen::RowVector2d>& x_key,
    const Eigen::Ref<const Eigen::RowVector2d>& direction,
    double prior_depth) {
  auto R_rk = get_rotation(T_rk);
  auto t_rk = get_translation(T_rk);
  const Eigen::Vector3d p_key = inv_pi(x_key, prior_depth);
  const Eigen::Vector3d p_ref = R_rk * p_key + t_rk;
  Eigen::Vector2d x_ref;
  project_vector(p_ref, x_ref);

  const int index = abs(direction[0]) > abs(direction[1]) ? 0 : 1;
  return calc_alpha_(x_key, x_ref[index], direction[index],
                     R_rk.row(index), R_rk.row(2), t_rk[index], t_rk[2]);
}


PYBIND11_MODULE(_variance, m) {
  m.def("calc_alpha", &calc_alpha,
        py::return_value_policy::reference_internal);
  m.def("calc_alpha_", &calc_alpha_,
        py::return_value_policy::reference_internal);
}
