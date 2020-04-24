#include <assert.h>
#include <pybind11/eigen.h>

namespace py = pybind11;


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
}
