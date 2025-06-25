#include <pybind11/pybind11.h>
#include <jarvis/core.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "JARVIS Core C++ Module";
    
    py::class_<jarvis::Core>(m, "Core")
        .def(py::init<>())
        .def("process", &jarvis::Core::process, "Process input string");
}
