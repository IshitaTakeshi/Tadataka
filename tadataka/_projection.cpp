#include <pybind11/eigen.h>

namespace py = pybind11;


Eigen::VectorXd inv_pi(Eigen::Ref<const Eigen::Vector2d> x, double depth) {
  Eigen::Vector3d y;
  y << x(0), x(1), 1;
  return depth * y;
}


Eigen::MatrixXd inv_pi(
    Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2>> xs,
    Eigen::Ref<const Eigen::VectorXd> depths) {
  const int N = xs.rows();
  Eigen::MatrixXd ys(N, 3);
  ys(Eigen::seq(0, N), Eigen::seq(0, 1)) = depths.asDiagonal() * xs;
  ys(Eigen::seq(0, N), 2) = depths;
  return ys;
}


PYBIND11_MODULE(_projection, m) {
  m.def("inv_pi",
        py::overload_cast<Eigen::Ref<const Eigen::Vector2d>,
                          double>(&inv_pi));
  m.def("inv_pi",
        py::overload_cast<
          Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2>>,
          Eigen::Ref<const Eigen::VectorXd>>(&inv_pi));
}
