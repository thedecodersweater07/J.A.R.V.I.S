#include <pybind11/pybind11.h>
#include <jarvis/my_new_module.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_my_new_module, m) {
    m.doc() = "JARVIS MyNewModule C++ Module";
    
    py::class_<jarvis::MyNewModule>(m, "MyNewModule")
        .def(py::init<>())
        .def("process", &jarvis::MyNewModule::process, "Process input string");
}
