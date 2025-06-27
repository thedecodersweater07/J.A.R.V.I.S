#ifndef JARVIS_MODEL_H
#define JARVIS_MODEL_H

#include <string>
#include <memory>

#ifdef _WIN32
#ifdef JARVIS_MODEL_EXPORT
#define JARVIS_MODEL_API __declspec(dllexport)
#else
#define JARVIS_MODEL_API __declspec(dllimport)
#endif
#else
#define JARVIS_MODEL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

JARVIS_MODEL_API void* CreateModel(const char* config_path);
JARVIS_MODEL_API void DeleteModel(void* model);
JARVIS_MODEL_API const char* ProcessInput(void* model, const char* input);

#ifdef __cplusplus
}

namespace Jarvis {
namespace Models {

class JarvisModel {
public:
    static std::shared_ptr<JarvisModel> Create(const std::string& config_path = "");
    virtual ~JarvisModel() = default;
    
    virtual std::string Process(const std::string& input) = 0;
};

} // namespace Models
} // namespace Jarvis

#endif // __cplusplus

#endif // JARVIS_MODEL_H
