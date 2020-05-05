#include <pybind11/eigen.h>
#include "tadataka/_types.hpp"


using Vectors3D = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;


void transform(
    Eigen::Ref<const RowMajorMatrixXd<4, 4>> T10,
    Eigen::Ref<const Vectors3D> P0,
    Eigen::Ref<Vectors3D> P1);
