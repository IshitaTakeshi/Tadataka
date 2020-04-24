#include <pybind11/eigen.h>


const Eigen::MatrixXd key_coordinates_(
    const Eigen::Vector2d epipolar_direction,
    const Eigen::RowVector2d x_key,
    const double step_size) {
  Eigen::VectorXd sampling_steps(5);
  sampling_steps << -2, -1, 0, 1, 2;

  const double n = epipolar_direction.norm();
  const Eigen::Vector2d direction = epipolar_direction / n;
  Eigen::MatrixXd steps = step_size * sampling_steps * direction.transpose();

  steps.rowwise() += x_key;
  return steps;
}


PYBIND11_MODULE(_epipolar, m) {
  m.def("key_coordinates_", &key_coordinates_);
}
