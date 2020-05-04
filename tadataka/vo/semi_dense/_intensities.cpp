#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


double inf = std::numeric_limits<double>::infinity();

namespace py = pybind11;


int search_(Eigen::Ref<const Eigen::VectorXd> sequence,
            Eigen::Ref<const Eigen::VectorXd> kernel) {

  const int N = kernel.size();
  double min_error = inf;
  int argmin = -1;

  for(int i = 0; i < sequence.size() - N + 1; i++) {
    const Eigen::VectorXd d = sequence(Eigen::seq(i, i + N - 1)) - kernel;
    double error = d.dot(d);
    if(error < min_error) {
      min_error = error;
      argmin = i;
    }
  }
  return argmin;
}


int search_intensities(Eigen::Ref<const Eigen::VectorXd> intensities_key,
                       Eigen::Ref<const Eigen::VectorXd> intensities_ref) {
  const int argmin = search_(intensities_ref, intensities_key);
  const int offset = (int)(intensities_key.size() / 2);
  return argmin + offset;
}


PYBIND11_MODULE(_intensities, m) {
  m.def("search_intensities", &search_intensities,
        py::return_value_policy::reference_internal);
}
