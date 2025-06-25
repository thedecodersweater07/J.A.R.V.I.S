#include <jarvis/core.hpp>

namespace jarvis {

class Core::Impl {
public:
    std::string process(const std::string& input) {
        // Core processing logic will go here
        return "Processed: " + input;
    }
};

Core::Core() : pImpl(std::make_unique<Impl>()) {}
Core::~Core() = default;

std::string Core::process(const std::string& input) const {
    return pImpl->process(input);
}

} // namespace jarvis
