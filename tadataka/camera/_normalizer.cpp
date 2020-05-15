#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "tadataka/_types.hpp"


namespace py = pybind11;

using RowMajorMatrix2d = RowMajorMatrixXd<Eigen::Dynamic, 2>;


const RowMajorMatrix2d normalize(
    Eigen::Ref<const RowMajorMatrix2d> keypoints,
    Eigen::Ref<const Eigen::RowVector2d> focal_length,
    Eigen::Ref<const Eigen::RowVector2d> offset) {
  auto v = keypoints.array().rowwise() - offset.array();
  return v.array().rowwise() / focal_length.array();
}


const RowMajorMatrix2d unnormalize(
    Eigen::Ref<const RowMajorMatrix2d> keypoints,
    Eigen::Ref<const Eigen::RowVector2d> focal_length,
    Eigen::Ref<const Eigen::RowVector2d> offset) {
  auto v = keypoints.array().rowwise() * focal_length.array();
  return v.array().rowwise() + offset.array();
}


PYBIND11_MODULE(_normalizer, m) {
  m.def("normalize", &normalize,
        py::return_value_policy::reference_internal);
  m.def("unnormalize", &unnormalize,
        py::return_value_policy::reference_internal);
}
