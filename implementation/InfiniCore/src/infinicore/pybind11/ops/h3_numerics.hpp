#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "infinicore/ops/h3_numerics.hpp"
namespace infinicore::ops {
inline void bind_h3_numerics(pybind11::module &m) {
    m.def("h3_rms_norm", &op::h3_rms_norm);
    m.def("h3_swiglu", &op::h3_swiglu);
    m.def("h3_rms_modulate", &op::h3_rms_modulate);
    m.def("h3_rope", &op::h3_rope);
    m.def("h3_rope_cache", &op::h3_rope_cache);
    m.def("h3_gated_residual", &op::h3_gated_residual);
    m.def("h3_gated_residual_rms_modulate", &op::h3_gated_residual_rms_modulate);
}
}
