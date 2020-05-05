#include "tadataka/_matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
  m.def("get_rotation", &get_rotation,
        py::return_value_policy::reference_internal);
  m.def("get_translation", &get_translation,
        py::return_value_policy::reference_internal);
}
