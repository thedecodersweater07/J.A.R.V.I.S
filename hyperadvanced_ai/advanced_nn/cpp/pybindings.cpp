#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "advanced_neural_net.hpp"

namespace py = pybind11;
using namespace jarvis::hyperai::advanced_nn;

PYBIND11_MODULE(_advanced_nn, m) {
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<>())
        .def("initialize", &NeuralNetwork::initialize,
             py::arg("model_config") = "",
             "Initialize the neural network with a configuration file.")
        .def("predict", &NeuralNetwork::predict,
             "Perform inference on input data and return predictions.")
        .def("train", &NeuralNetwork::train,
             py::arg("inputs"),
             py::arg("targets"),
             py::arg("epochs") = 1,
             "Train the network with input and target data.")
        .def("save_weights", &NeuralNetwork::save_weights,
             py::arg("path"),
             "Save the model weights to a file.")
        .def("load_weights", &NeuralNetwork::load_weights,
             py::arg("path"),
             "Load model weights from a file.")
        .def("is_initialized", &NeuralNetwork::is_initialized,
             "Check if the network is initialized.");
}
