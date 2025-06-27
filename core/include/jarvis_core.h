#ifndef JARVIS_CORE_H
#define JARVIS_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
#define JARVIS_WINDOWS
#elif defined(__linux__)
#define JARVIS_LINUX
#elif defined(__APPLE__)
#define JARVIS_MACOS
#endif

// DLL export/import
#ifdef JARVIS_WINDOWS
    #ifdef JARVIS_CORE_EXPORT
        #define JARVIS_API __declspec(dllexport)
    #else
        #define JARVIS_API __declspec(dllimport)
    #endif
#else
    #define JARVIS_API __attribute__((visibility("default")))
#endif

// Basic types
typedef enum {
    JARVIS_SUCCESS = 0,
    JARVIS_ERROR = -1,
    JARVIS_INVALID_INPUT = -2,
    JARVIS_NOT_IMPLEMENTED = -3
} JarvisStatus;

// Core functions
JARVIS_API JarvisStatus jarvis_initialize(const char* config_path);
JARVIS_API JarvisStatus jarvis_process(const char* input, char** output);
JARVIS_API void jarvis_cleanup();

// Memory management
JARVIS_API void jarvis_free_string(char* str);

// Language specific bindings
#ifdef __cplusplus
}

// C++ specific API
#include <string>
#include <memory>

namespace Jarvis {
    class Core {
    public:
        static std::shared_ptr<Core> create(const std::string& config_path = "");
        virtual ~Core() = default;
        virtual std::string process(const std::string& input) = 0;
    };
}

#endif // __cplusplus

#endif // JARVIS_CORE_H
