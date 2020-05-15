#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "tadataka/_types.hpp"
#include "tadataka/_transform.hpp"
#include "tadataka/_projection.hpp"


namespace py = pybind11;

using RowVectors3D = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;


void warp2d(
    const Eigen::Ref<const RowMajorMatrixXd<4, 4>>& T_10,
    const Eigen::Ref<const RowMajorMatrixXd<Eigen::Dynamic, 2>>& xs0,
    const Eigen::Ref<const Eigen::VectorXd>& depths0,
    Eigen::Ref<RowMajorMatrixXd<Eigen::Dynamic, 2>>& xs1,
    Eigen::Ref<Eigen::VectorXd>& depths1) {
  RowVectors3D P = inv_pi(xs0, depths0); // P0
  transform(T_10, P, P);  // P1 = T_10 * P0
  project_vectors(P, xs1);  // P1
  depths1 = P(Eigen::all, 2);
}


PYBIND11_MODULE(_warp, m) {
  m.def("warp2d", &warp2d,
        py::return_value_policy::reference_internal);
}
