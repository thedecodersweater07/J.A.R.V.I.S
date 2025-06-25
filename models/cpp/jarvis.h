#ifndef JARVIS_MODEL_H
#define JARVIS_MODEL_H

#include <string>
#include <vector>
#include <memory>
#include <cmath>      // For std::exp, std::sin, std::cos
#include <numeric>    // For std::accumulate
#include <algorithm>  // For std::transform, std::max_element
#include <stdexcept>  // For std::runtime_error, std::invalid_argument

namespace jarvis {
namespace models {

/**
 * @brief Base class for machine learning models in JARVIS
 */
class Model {
public:
    virtual ~Model() = default;

    /**
     * @brief Load a pre-trained model from disk
     * @param model_path Path to the model file
     * @return True if loading was successful, false otherwise
     */
    virtual bool load(const std::string& model_path) = 0;

    /**
     * @brief Make a prediction using the model
     * @param input Input data for prediction
     * @return Vector of prediction results
     */
    virtual std::vector<float> predict(const std::vector<float>& input) = 0;

    /**
     * @brief Get model information
     * @return String containing model metadata
     */
    virtual std::string get_model_info() const = 0;

    /**
     * @brief Get the framework used by the model
     * @return Name of the framework (e.g., "ONNX", "TensorFlow", "PyTorch")
     */
    virtual std::string get_framework() const = 0;
};

/**
 * @brief Factory function to create a model instance
 * @param model_type Type of model to create (e.g., "onnx", "tensorflow")
 * @return Shared pointer to the created model, or nullptr if type is not supported
 */
std::shared_ptr<Model> create_model(const std::string& model_type);

} // namespace models
} // namespace jarvis

#endif // JARVIS_MODEL_H