#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "jarvis.h"

// For Python module initialization
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace jarvis::models;

// Wrapper class to handle Python GIL
class PyModel : public Model {
    using Model::Model;
    
    bool load(const std::string& model_path) override {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(
            bool,        /* Return type */
            Model,       /* Parent class */
            load,        /* Name of function in C++ */
            model_path   /* Argument(s) */
        );
    }
    
    std::vector<float> predict(const std::vector<float>& input) override {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(
            std::vector<float>,
            Model,
            predict,
            input
        );
    }
    
    std::string get_model_info() const override {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(
            std::string,
            Model,
            get_model_info,
        );
    }
    
    std::string get_framework() const override {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(
            std::string,
            Model,
            get_framework,
        );
    }
};

// Create Python bindings
PYBIND11_MODULE(_models, m) {
    m.doc() = "JARVIS Models C++ Module";
    
    // Bind the Model base class
    py::class_<Model, PyModel, std::shared_ptr<Model>>(m, "Model")
        .def(py::init<>())
        .def("load", &Model::load, "Load a pre-trained model from disk")
        .def("predict", &Model::predict, "Make a prediction using the model")
        .def("get_model_info", &Model::get_model_info, "Get model information")
        .def("get_framework", &Model::get_framework, "Get the framework used by the model");
    
    // Factory function
    m.def("create_model", &create_model, "Create a model instance");
}
