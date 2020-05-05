#include <assert.h>
#include <pybind11/pybind11.h>

#include "_matrix.hpp"
#include "_transform.hpp"


namespace py = pybind11;


void transform(const Eigen::Ref<const RowMajorMatrixXd<3, 3>> R10,
               const Eigen::Ref<const Eigen::VectorXd> t10,
               const Eigen::Ref<const Eigen::RowVectorXd> p0,
               Eigen::Ref<Eigen::RowVectorXd> p1) {
  p1 = (R10 * p0.transpose() + t10).transpose();
}


void transform(const Eigen::Ref<const RowMajorMatrixXd<4, 4>>& T10,
               const Eigen::Ref<const RowVectors3D>& P0,
               Eigen::Ref<RowVectors3D>& P1) {
  assert(P0.rows() == P1.rows());
  assert(P0.cols() == P1.cols());

  const auto R10 = get_rotation(T10);
  const auto t10 = get_translation(T10);

  for(int i = 0; i < P0.rows(); i++) {
    transform(R10, t10, P0.row(i), P1.row(i));
  }
}


PYBIND11_MODULE(_transform, m) {
  m.def("transform",
        py::overload_cast<
        const Eigen::Ref<const RowMajorMatrixXd<4, 4>>&,
        const Eigen::Ref<const RowVectors3D>&,
        Eigen::Ref<RowVectors3D>&>(&transform),
        py::return_value_policy::reference_internal);
}
