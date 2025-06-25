#pragma once

#include <string>
#include <memory>

namespace jarvis {

class Core {
public:
    Core();
    ~Core();
    
    std::string process(const std::string& input) const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace jarvis
