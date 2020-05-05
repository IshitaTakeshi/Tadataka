#include <assert.h>
#include <pybind11/eigen.h>

#include "tadataka/_projection.hpp"
#include "tadataka/_types.hpp"


namespace py = pybind11;

const double EPSILON = 1e-16;


const RowMajorMatrixXd<Eigen::Dynamic, 2> pi(
    const Eigen::Ref<const RowMajorMatrixXd<Eigen::Dynamic, 3>>& P) {
  auto zs = P(Eigen::all, 2).array() + EPSILON;
  return P(Eigen::all, Eigen::seq(0, 1)).array().colwise() / zs;
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
  m.def("pi",
        py::overload_cast<
          const Eigen::Ref<const RowMajorMatrixXd<Eigen::Dynamic, 3>>&>(&pi));
}
