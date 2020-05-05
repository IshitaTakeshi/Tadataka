#include <pybind11/eigen.h>


using Vectors3D = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;


void transform(
    Eigen::Ref<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T10,
    Eigen::Ref<const Vectors3D> P0,
    Eigen::Ref<Vectors3D> P1);
