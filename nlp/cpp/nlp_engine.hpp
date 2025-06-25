#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>


namespace jarvis {
namespace nlp {

enum class Language {
    DUTCH,
    ENGLISH,
    GERMAN,
    FRENCH
};

enum class ProcessingMode {
    BASIC,
    ADVANCED,
    SEMANTIC
};

struct ProcessingResult {
    std::string processed_text;
    std::vector<std::string> tokens;
    std::vector<std::string> entities;
    std::unordered_map<std::string, double> sentiment_scores;
    double confidence = 0.0;
    
    ProcessingResult() = default;
    ProcessingResult(const std::string& text) : processed_text(text) {}
};

struct NLPConfig {
    Language language = Language::DUTCH;
    ProcessingMode mode = ProcessingMode::BASIC;
    bool enable_stemming = true;
    bool enable_entity_extraction = false;
    bool enable_sentiment_analysis = false;
    std::vector<std::string> custom_stopwords;
};

class NLEngine {
public:
    NLEngine();
    explicit NLEngine(const NLPConfig& config);
    ~NLEngine();

    // Core initialization
    bool initialize(const NLPConfig& config = NLPConfig{});
    bool initialize(const std::string& language);
    void shutdown();

    // Text processing methods
    ProcessingResult process_text(const std::string& text);
    std::vector<std::string> tokenize(const std::string& text);
    std::string normalize_text(const std::string& text);
    std::vector<std::string> extract_entities(const std::string& text);
    
    // Language detection and handling
    Language detect_language(const std::string& text);
    bool set_language(Language lang);
    std::string get_language_name() const;
    
    // Advanced features
    double calculate_sentiment(const std::string& text);
    std::vector<std::string> get_keywords(const std::string& text, size_t max_keywords = 10);
    std::string stem_word(const std::string& word);
    
    // Configuration and status
    bool is_initialized() const { return initialized_; }
    const NLPConfig& get_config() const;
    void update_config(const NLPConfig& config);
    
    // Statistics and metrics
    size_t get_processed_count() const;
    double get_average_processing_time() const;
    void reset_statistics();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    bool initialized_ = false;
    NLPConfig config_;
    
    // Statistics
    mutable size_t processed_count_ = 0;
    mutable double total_processing_time_ = 0.0;
};

// Utility functions
std::string language_to_string(Language lang);
Language string_to_language(const std::string& lang_str);
std::vector<std::string> split_sentences(const std::string& text);
bool is_valid_language_code(const std::string& code);

} // namespace nlp
} // namespace jarvis