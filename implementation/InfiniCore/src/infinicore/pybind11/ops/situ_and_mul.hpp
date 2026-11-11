#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/situ_and_mul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_situ_and_mul(py::module &m) {
    m.def("situ_and_mul",
          &op::situ_and_mul,
          py::arg("gate"),
          py::arg("up"),
          py::arg("beta") = 4.0f,
          py::arg("linear_beta") = 25.0f,
          R"doc(Fused SiTU activation and multiplication.)doc");

    m.def("situ_and_mul_",
          &op::situ_and_mul_,
          py::arg("output"),
          py::arg("gate"),
          py::arg("up"),
          py::arg("beta") = 4.0f,
          py::arg("linear_beta") = 25.0f,
          R"doc(Fused SiTU activation and multiplication into output.)doc");
}

} // namespace infinicore::ops
