#include <assert.h>
#include <pybind11/eigen.h>

#include "tadataka/_projection.hpp"
#include "tadataka/_types.hpp"


namespace py = pybind11;

const double EPSILON = 1e-16;


void project_vector(
    const Eigen::Ref<const Eigen::Vector3d>& p,
    Eigen::Ref<Eigen::Vector2d> u) {
  const double z = p(2) + EPSILON;
  u = p(Eigen::seq(0, 1)) / z;
}


void project_vectors(
    const Eigen::Ref<const RowMajorMatrixXd<Eigen::Dynamic, 3>>& P,
    Eigen::Ref<RowMajorMatrixXd<Eigen::Dynamic, 2>> U) {
  for(int i = 0; i < P.rows(); i++) {
    const double z = P(i, 2) + EPSILON;
    U(i, Eigen::seq(0, 1)) = P(i, Eigen::seq(0, 1)) / z;
  }
}


const Eigen::Vector3d inv_pi(const Eigen::Vector2d x, double depth) {
  Eigen::Vector3d y;
  y << x(0), x(1), 1;
  return depth * y;
}


const Eigen::MatrixXd inv_pi(
    const Eigen::Matrix<double, Eigen::Dynamic, 2> xs,
    const Eigen::VectorXd depths) {
  assert(xs.rows() == depths.size());

  const int N = xs.rows();

  Eigen::MatrixXd ys(N, 3);
  for(int i = 0; i < N; i++) {
    ys.row(i) = inv_pi(xs.row(i), depths(i));
  }
  return ys;
}


PYBIND11_MODULE(_projection, m) {
  m.def("inv_pi",
        py::overload_cast<const Eigen::Vector2d, double>(&inv_pi));

  m.def("inv_pi",
        py::overload_cast<
          const Eigen::Matrix<double, Eigen::Dynamic, 2>,
                              const Eigen::VectorXd>(&inv_pi));

  m.def("project_vector", &project_vector,
        py::return_value_policy::reference_internal);
  m.def("project_vectors", &project_vectors,
        py::return_value_policy::reference_internal);
}
