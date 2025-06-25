#include <jarvis/my_new_module.hpp>

namespace jarvis {

class MyNewModule::Impl {
public:
    std::string process(const std::string& input) {
        // TODO: Implement my_new_module processing
        return "Processed by my_new_module: " + input;
    }
};

MyNewModule::MyNewModule() : pImpl(std::make_unique<Impl>()) {}
MyNewModule::~MyNewModule() = default;

std::string MyNewModule::process(const std::string& input) const {
    return pImpl->process(input);
}

} // namespace jarvis
