#pragma once

#include <string>
#include <memory>
#include <vector>

namespace jarvis {
namespace hyperai {
namespace advanced_nn {

class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

    // Initialize the neural network with a specific architecture
    bool initialize(const std::string& model_config = "");
    
    // Perform inference on input data
    std::vector<float> predict(const std::vector<float>& input);
    
    // Train the network with input and target data
    void train(const std::vector<std::vector<float>>& inputs,
              const std::vector<std::vector<float>>& targets,
              int epochs = 1);
    
    // Save/Load model weights
    bool save_weights(const std::string& path) const;
    bool load_weights(const std::string& path);
    
    // Check if the network is initialized
    bool is_initialized() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace advanced_nn
} // namespace hyperai
} // namespace jarvis
