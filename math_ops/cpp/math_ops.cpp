#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
double add(double a, double b);
double multiply(double a, double b);

PYBIND11_MODULE(_math_ops, m) {
    m.doc() = "JARVIS Math Operations C++ Module";
    
    m.def("add", &add, "Add two numbers");
    m.def("multiply", &multiply, "Multiply two numbers");
    
    // Add a flag to indicate this is the C++ implementation
    m.attr("IS_CPP_IMPLEMENTATION") = true;
}

// Implementation
double add(double a, double b) {
    return a + b;
}

double multiply(double a, double b) {
    return a * b;
}
