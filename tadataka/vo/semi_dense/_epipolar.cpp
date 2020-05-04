#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

const double EPSILON = 1e-16;


const Eigen::MatrixXd key_coordinates_(
    Eigen::Ref<const Eigen::RowVector2d> epipolar_direction,
    Eigen::Ref<const Eigen::RowVector2d> x_key,
    const double step_size) {
  Eigen::VectorXd sampling_steps(5);
  sampling_steps << -2, -1, 0, 1, 2;

  const double n = epipolar_direction.norm();
  const Eigen::Vector2d direction = epipolar_direction / n;
  Eigen::MatrixXd steps = step_size * sampling_steps * direction.transpose();

  steps.rowwise() += x_key;
  return steps;
}

const Eigen::MatrixXd calc_coordinates(
    Eigen::Ref<const Eigen::Vector2d> x_min,
    Eigen::Ref<const Eigen::Vector2d> x_max,
    const double step_size) {

    Eigen::Vector2d direction = x_max - x_min;
    const double norm = direction.norm();
    direction = direction / (norm + EPSILON);
    int N = (int)(norm / step_size);

    Eigen::MatrixXd xs(N, 2);
    for(int i = 0; i < N; i++) {
      xs.row(i) = x_min + i * step_size * direction;
    }
    return xs;
}


PYBIND11_MODULE(_epipolar, m) {
  m.def("key_coordinates_", &key_coordinates_,
        py::return_value_policy::reference_internal);
  m.def("calc_coordinates", &calc_coordinates,
        py::return_value_policy::reference_internal);
}
