#include <pybind11/eigen.h>


using Matrix33d = Eigen::Matrix<double, 3, 3>;
using Matrix44d = Eigen::Matrix<double, 4, 4>;


inline Matrix33d get_rotation(Eigen::Ref<const Matrix44d> T) {
  return T.block<3, 3>(0, 0);
}


inline Eigen::Vector3d get_translation(Eigen::Ref<const Matrix44d> T) {
  return T.block<3, 1>(0, 3);
}
