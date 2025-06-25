#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "nlp_engine.hpp"

namespace py = pybind11;
using namespace jarvis::nlp;

// Forward declarations
void init_language(py::module &m);
void init_processing_mode(py::module &m);
void init_processing_result(py::module &m);
void init_nlp_config(py::module &m);
void init_nl_engine(py::module &m);

PYBIND11_MODULE(_nlp_engine, m) {
    m.doc() = "JARVIS NLP Engine Python Bindings";
    
    // Initialize submodules
    init_language(m);
    init_processing_mode(m);
    init_processing_result(m);
    init_nlp_config(m);
    init_nl_engine(m);
    
    // Constants
    m.attr("DEFAULT_LANGUAGE") = static_cast<int>(DEFAULT_LANGUAGE);
    
    // Supported languages
    py::list supported_langs;
    for (const auto& lang : SUPPORTED_LANGUAGES) {
        supported_langs.append(lang);
    }
    m.attr("SUPPORTED_LANGUAGES") = supported_langs;
    
    // Utility functions
    m.def("string_to_language", &string_to_language, 
          "Convert language string to Language enum");
    m.def("language_to_string", &language_to_string,
          "Convert Language enum to string");
}

void init_language(py::module &m) {
    py::enum_<Language>(m, "Language")
        .value("ENGLISH", Language::ENGLISH)
        .value("SPANISH", Language::SPANISH)
        .value("FRENCH", Language::FRENCH)
        .value("GERMAN", Language::GERMAN)
        .export_values();
}

void init_processing_mode(py::module &m) {
    py::enum_<ProcessingMode>(m, "ProcessingMode")
        .value("DEFAULT", ProcessingMode::DEFAULT)
        .value("FAST", ProcessingMode::FAST)
        .value("ACCURATE", ProcessingMode::ACCURATE)
        .export_values();
}

void init_processing_result(py::module &m) {
    py::class_<ProcessingResult>(m, "ProcessingResult")
        .def_readwrite("processed_text", &ProcessingResult::processed_text)
        .def_readwrite("tokens", &ProcessingResult::tokens)
        .def_readwrite("entities", &ProcessingResult::entities)
        .def_readwrite("sentiment_scores", &ProcessingResult::sentiment_scores)
        .def_readwrite("confidence", &ProcessingResult::confidence);
}

void init_nlp_config(py::module &m) {
    py::class_<NLPConfig>(m, "NLPConfig")
        .def(py::init<>())
        .def_readwrite("language", &NLPConfig::language)
        .def_readwrite("mode", &NLPConfig::mode)
        .def_readwrite("enable_stemming", &NLPConfig::enable_stemming)
        .def_readwrite("enable_entity_extraction", &NLPConfig::enable_entity_extraction)
        .def_readwrite("enable_sentiment_analysis", &NLPConfig::enable_sentiment_analysis)
        .def_readwrite("custom_stopwords", &NLPConfig::custom_stopwords);
}

void init_nl_engine(py::module &m) {
    py::class_<NLEngine>(m, "NLEngine")
        .def(py::init<>())
        .def("initialize", py::overload_cast<const std::string&>(&NLEngine::initialize),
             "Initialize with language code")
        .def("initialize", py::overload_cast<const NLPConfig&>(&NLEngine::initialize),
             "Initialize with config")
        .def("process_text", &NLEngine::process_text, "Process text and get results")
        .def("tokenize", &NLEngine::tokenize, "Tokenize text")
        .def("normalize_text", &NLEngine::normalize_text, "Normalize text")
        .def("extract_entities", &NLEngine::extract_entities, "Extract entities from text")
        .def("detect_language", &NLEngine::detect_language, "Detect language of text")
        .def("set_language", &NLEngine::set_language, "Set processing language")
        .def("get_language_name", &NLEngine::get_language_name, "Get current language name")
        .def("calculate_sentiment", &NLEngine::calculate_sentiment, "Calculate sentiment scores")
        .def("get_keywords", &NLEngine::get_keywords, "Extract keywords")
        .def("stem_word", &NLEngine::stem_word, "Stem a word")
        .def("is_initialized", &NLEngine::is_initialized, "Check if engine is initialized")
        .def("get_config", &NLEngine::get_config, "Get current configuration")
        .def("update_config", &NLEngine::update_config, "Update configuration")
        .def("get_processed_count", &NLEngine::get_processed_count, "Get number of processed texts")
        .def("get_average_processing_time", &NLEngine::get_average_processing_time,
             "Get average processing time in milliseconds")
        .def("reset_statistics", &NLEngine::reset_statistics, "Reset statistics")
        .def("shutdown", &NLEngine::shutdown, "Shutdown and clean up resources");
}
