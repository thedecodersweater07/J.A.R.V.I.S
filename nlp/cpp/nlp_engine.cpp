#include "C:\nova_industrie\jarvis\nlp\cpp\nlp_engine.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <regex>
#include <sstream>
#include <unordered_set>
#include <chrono>

namespace jarvis {
namespace nlp {

// Language-specific resources
class LanguageResources {
public:
    std::unordered_set<std::string> stopwords;
    std::unordered_map<std::string, std::string> stemming_rules;
    std::regex entity_patterns;
    std::unordered_map<std::string, double> sentiment_lexicon;
    
    void load_resources(Language lang) {
        switch (lang) {
            case Language::DUTCH:
                load_dutch_resources();
                break;
            case Language::ENGLISH:
                load_english_resources();
                break;
            case Language::GERMAN:
                load_german_resources();
                break;
            case Language::FRENCH:
                load_french_resources();
                break;
        }
    }
    
private:
    void load_dutch_resources() {
        stopwords = {"de", "het", "een", "en", "van", "te", "dat", "die", "in", "voor", 
                    "is", "op", "met", "als", "zijn", "er", "aan", "ook", "door", "maar"};
        
        // Basic Dutch entity patterns
        entity_patterns = std::regex(R"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)");
        
        // Simple sentiment words for Dutch
        sentiment_lexicon = {
            {"goed", 0.8}, {"slecht", -0.8}, {"geweldig", 0.9}, {"vreselijk", -0.9},
            {"mooi", 0.7}, {"lelijk", -0.7}, {"leuk", 0.6}, {"saai", -0.5}
        };
    }
    
    void load_english_resources() {
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
                    "for", "of", "with", "by", "is", "are", "was", "were", "be"};
        
        entity_patterns = std::regex(R"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)");
        
        sentiment_lexicon = {
            {"good", 0.8}, {"bad", -0.8}, {"great", 0.9}, {"terrible", -0.9},
            {"beautiful", 0.7}, {"ugly", -0.7}, {"nice", 0.6}, {"boring", -0.5}
        };
    }
    
    void load_german_resources() {
        stopwords = {"der", "die", "das", "und", "oder", "aber", "in", "auf", "mit", 
                    "zu", "für", "von", "ist", "sind", "war", "waren", "sein"};
        
        entity_patterns = std::regex(R"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)");
        
        sentiment_lexicon = {
            {"gut", 0.8}, {"schlecht", -0.8}, {"großartig", 0.9}, {"schrecklich", -0.9}
        };
    }
    
    void load_french_resources() {
        stopwords = {"le", "la", "les", "de", "du", "des", "et", "ou", "mais", "dans", 
                    "sur", "avec", "pour", "par", "est", "sont", "était", "étaient"};
        
        entity_patterns = std::regex(R"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)");
        
        sentiment_lexicon = {
            {"bon", 0.8}, {"mauvais", -0.8}, {"excellent", 0.9}, {"terrible", -0.9}
        };
    }
};

class NLEngine::Impl {
private:
    LanguageResources resources_;
    NLPConfig config_;
    
public:
    Impl() = default;
    ~Impl() = default;
    
    bool initialize(const NLPConfig& config) {
        config_ = config;
        resources_.load_resources(config.language);
        return true;
    }
    
    ProcessingResult process_text(const std::string& text) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ProcessingResult result;
        
        // Basic normalization
        result.processed_text = normalize_text_impl(text);
        
        // Tokenization
        result.tokens = tokenize_impl(result.processed_text);
        
        // Advanced processing based on mode
        if (config_.mode == ProcessingMode::ADVANCED || config_.mode == ProcessingMode::SEMANTIC) {
            if (config_.enable_entity_extraction) {
                result.entities = extract_entities_impl(result.processed_text);
            }
            
            if (config_.enable_sentiment_analysis) {
                double sentiment = calculate_sentiment_impl(result.processed_text);
                result.sentiment_scores["overall"] = sentiment;
                result.sentiment_scores["positive"] = std::max(0.0, sentiment);
                result.sentiment_scores["negative"] = std::max(0.0, -sentiment);
            }
        }
        
        // Calculate confidence based on text length and processing success
        result.confidence = calculate_confidence(text, result);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        return result;
    }
    
    std::string normalize_text_impl(const std::string& text) {
        std::string result = text;
        
        // Convert to lowercase
        std::transform(result.begin(), result.end(), result.begin(), 
                      [](unsigned char c){ return std::tolower(c); });
        
        // Remove extra whitespace
        result = std::regex_replace(result, std::regex(R"(\s+)"), " ");
        
        // Trim
        result.erase(0, result.find_first_not_of(" \t\n\r\f\v"));
        result.erase(result.find_last_not_of(" \t\n\r\f\v") + 1);
        
        // Remove punctuation for basic processing
        if (config_.mode == ProcessingMode::BASIC) {
            result = std::regex_replace(result, std::regex(R"([^\w\s])"), "");
        }
        
        return result;
    }
    
    std::vector<std::string> tokenize_impl(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        
        while (iss >> token) {
            // Skip stopwords if configured
            if (resources_.stopwords.find(token) == resources_.stopwords.end() || 
                !config_.enable_stemming) {
                
                if (config_.enable_stemming) {
                    token = stem_word_impl(token);
                }
                
                tokens.push_back(token);
            }
        }
        
        return tokens;
    }
    
    std::vector<std::string> extract_entities_impl(const std::string& text) {
        std::vector<std::string> entities;
        std::sregex_iterator iter(text.begin(), text.end(), resources_.entity_patterns);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            entities.push_back(iter->str());
        }
        
        return entities;
    }
    
    double calculate_sentiment_impl(const std::string& text) {
        std::istringstream iss(text);
        std::string word;
        double total_sentiment = 0.0;
        int word_count = 0;
        
        while (iss >> word) {
            auto it = resources_.sentiment_lexicon.find(word);
            if (it != resources_.sentiment_lexicon.end()) {
                total_sentiment += it->second;
                word_count++;
            }
        }
        
        return word_count > 0 ? total_sentiment / word_count : 0.0;
    }
    
    std::string stem_word_impl(const std::string& word) {
        // Simple suffix removal for demonstration
        std::string result = word;
        
        // Dutch/English common suffixes
        std::vector<std::string> suffixes = {"ing", "ed", "er", "est", "en", "de", "te"};
        
        for (const auto& suffix : suffixes) {
            if (result.length() > suffix.length() + 2 && 
                result.substr(result.length() - suffix.length()) == suffix) {
                result = result.substr(0, result.length() - suffix.length());
                break;
            }
        }
        
        return result;
    }
    
    Language detect_language_impl(const std::string& text) {
        // Simple language detection based on common words
        std::unordered_map<Language, int> scores;
        
        // Count language-specific indicators
        if (text.find(" de ") != std::string::npos || text.find(" het ") != std::string::npos) {
            scores[Language::DUTCH]++;
        }
        if (text.find(" the ") != std::string::npos || text.find(" and ") != std::string::npos) {
            scores[Language::ENGLISH]++;
        }
        if (text.find(" der ") != std::string::npos || text.find(" und ") != std::string::npos) {
            scores[Language::GERMAN]++;
        }
        if (text.find(" le ") != std::string::npos || text.find(" et ") != std::string::npos) {
            scores[Language::FRENCH]++;
        }
        
        // Return language with highest score, default to current config
        auto max_elem = std::max_element(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
            
        return max_elem != scores.end() ? max_elem->first : config_.language;
    }
    
    std::vector<std::string> get_keywords_impl(const std::string& text, size_t max_keywords) {
        auto tokens = tokenize_impl(text);
        std::unordered_map<std::string, int> frequency;
        
        for (const auto& token : tokens) {
            if (token.length() > 3) { // Only consider longer words
                frequency[token]++;
            }
        }
        
        std::vector<std::pair<std::string, int>> sorted_words(frequency.begin(), frequency.end());
        std::sort(sorted_words.begin(), sorted_words.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::vector<std::string> keywords;
        for (size_t i = 0; i < std::min(max_keywords, sorted_words.size()); ++i) {
            keywords.push_back(sorted_words[i].first);
        }
        
        return keywords;
    }
    
private:
    double calculate_confidence(const std::string& original_text, const ProcessingResult& result) {
        double confidence = 0.7; // Base confidence
        
        // Increase confidence based on text length
        if (original_text.length() > 50) confidence += 0.1;
        if (original_text.length() > 200) confidence += 0.1;
        
        // Increase confidence if entities were found
        if (!result.entities.empty()) confidence += 0.05;
        
        // Increase confidence if sentiment analysis was successful
        if (!result.sentiment_scores.empty()) confidence += 0.05;
        
        return std::min(1.0, confidence);
    }
};

// Implementation of NLEngine public methods
NLEngine::NLEngine() : impl_(std::make_unique<Impl>()) {}

NLEngine::NLEngine(const NLPConfig& config) : impl_(std::make_unique<Impl>()), config_(config) {}

NLEngine::~NLEngine() = default;

bool NLEngine::initialize(const NLPConfig& config) {
    try {
        config_ = config;
        initialized_ = impl_->initialize(config);
        return initialized_;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize NLP engine: " << e.what() << std::endl;
        return false;
    }
}

bool NLEngine::initialize(const std::string& language) {
    NLPConfig config;
    config.language = string_to_language(language);
    return initialize(config);
}

void NLEngine::reset_statistics() {
    processed_count_ = 0;
    total_processing_time_ = 0.0;
}

void NLEngine::shutdown() {
    // Cleanup resources
    reset_statistics();
    initialized_ = false;
}

ProcessingResult NLEngine::process_text(const std::string& text) {
    if (!initialized_) {
        throw std::runtime_error("NLP Engine not initialized");
    }
    
    processed_count_++;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto result = impl_->process_text(text);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    total_processing_time_ += static_cast<double>(duration.count()) / 1000.0; // Convert to milliseconds
    
    return result;
}

std::vector<std::string> NLEngine::tokenize(const std::string& text) {
    if (!initialized_) {
        throw std::runtime_error("NLP Engine not initialized");
    }
    return impl_->tokenize_impl(text);
}

std::string NLEngine::normalize_text(const std::string& text) {
    if (!initialized_) {
        throw std::runtime_error("NLP Engine not initialized");
    }
    return impl_->normalize_text_impl(text);
}

// Utility functions
std::string language_to_string(Language lang) {
    switch (lang) {
        case Language::DUTCH: return "nl";
        case Language::ENGLISH: return "en";
        case Language::GERMAN: return "de";
        case Language::FRENCH: return "fr";
        default: return "nl";
    }
}

Language string_to_language(const std::string& lang_str) {
    if (lang_str == "en" || lang_str == "english") return Language::ENGLISH;
    if (lang_str == "de" || lang_str == "german") return Language::GERMAN;
    if (lang_str == "fr" || lang_str == "french") return Language::FRENCH;
    return Language::DUTCH; // Default
}

std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::regex sentence_regex(R"([.!?]+\s+)");
    std::sregex_token_iterator iter(text.begin(), text.end(), sentence_regex, -1);
    std::sregex_token_iterator end;
    
    for (; iter != end; ++iter) {
        std::string sentence = iter->str();
        if (!sentence.empty() && sentence != " ") {
            sentences.push_back(sentence);
        }
    }
    
    return sentences;
}

bool is_valid_language_code(const std::string& code) {
    return code == "nl" || code == "en" || code == "de" || code == "fr";
}

} // namespace nlp
} // namespace jarvis