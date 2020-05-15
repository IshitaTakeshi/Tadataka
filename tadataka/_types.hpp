#include <pybind11/eigen.h>
template<int rows, int cols>
using RowMajorMatrixXd = Eigen::Matrix<double, rows, cols, Eigen::RowMajor>;
