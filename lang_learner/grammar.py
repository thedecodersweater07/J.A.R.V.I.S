"""
Grammar Checking Module - Optimized Version

Provides functionality to check and correct grammar in the target language.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Optional spaCy import with graceful fallback
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    nlp = None

@dataclass
class GrammarRule:
    """Represents a grammar rule with pattern and correction."""
    name: str
    pattern: str
    correction: str
    description: str
    examples: List[Tuple[str, str]]

class GrammarChecker:
    """Checks and corrects grammar in the target language."""
    
    def __init__(self, language: str = 'dutch', level: str = 'beginner'):
        self.language = language.lower()
        self.level = level
        self.rules = self._load_grammar_rules()
        self.nlp = self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model for the language if available."""
        if not SPACY_AVAILABLE:
            return None
            
        model_map = {
            'dutch': 'nl_core_news_sm',
            'english': 'en_core_web_sm',
            'german': 'de_core_news_sm',
            'french': 'fr_core_news_sm',
            'spanish': 'es_core_news_sm'
        }
        
        try:
            model_name = model_map.get(self.language)
            return spacy.load(model_name) if model_name else None
        except OSError:
            return None
    
    def _load_grammar_rules(self) -> Dict[str, GrammarRule]:
        """Load grammar rules for the current language."""
        rules = {
            'dutch': {
                'double_subject': GrammarRule(
                    name="Double Subject",
                    pattern=r"\b([A-Z][a-z]*)\s+([A-Z][a-z]*)\s+(is|heeft|kan|wil|gaat|doet)\b",
                    correction=r"\1 \3",
                    description="Avoid using both a name and pronoun as subject",
                    examples=[("Jan hij is mijn vriend.", "Jan is mijn vriend.")]
                ),
                'verb_placement': GrammarRule(
                    name="Verb Placement",
                    pattern=r"\b(Ik|Jij|Hij|Zij|Wij|Jullie)\s+(\w+)\s+(niet|altijd|vaak|soms)\s+(\w+)",
                    correction=r"\1 \2 \3 \4",
                    description="Adverb placement in Dutch main clauses",
                    examples=[("Ik altijd eet ontbijt.", "Ik eet altijd ontbijt.")]
                )
            },
            'english': {
                'a_an': GrammarRule(
                    name="A/An Usage",
                    pattern=r"\ba\s+[aeiouAEIOU]\w*\b",
                    correction="an",
                    description="Use 'an' before vowel sounds",
                    examples=[("a apple", "an apple")]
                )
            }
        }
        return rules.get(self.language, {})
    
    def check(self, text: str) -> Dict:
        """Check grammar and return results with corrections."""
        if not text.strip():
            return {
                'original_text': text,
                'is_correct': False,
                'errors': [{'type': 'empty_input', 'message': 'No text provided'}],
                'corrected_text': text,
                'suggestions': []
            }
        
        results = {
            'original_text': text,
            'is_correct': True,
            'errors': [],
            'corrected_text': text,
            'suggestions': []
        }
        
        corrected_text = text
        
        # Apply grammar rules
        for rule_name, rule in self.rules.items():
            matches = list(re.finditer(rule.pattern, text, re.IGNORECASE))
            for match in matches:
                corrected = match.expand(rule.correction)
                if match.group(0) != corrected:
                    results['is_correct'] = False
                    results['errors'].append({
                        'type': rule_name,
                        'message': rule.description,
                        'context': match.group(0),
                        'replacement': corrected,
                        'examples': rule.examples
                    })
                    corrected_text = corrected_text.replace(match.group(0), corrected, 1)
        
        # Advanced spaCy analysis for Dutch
        if self.nlp and self.language == 'dutch' and len(text) < 1000:
            doc = self.nlp(text)
            for token in doc:
                if token.text.lower() == 'hun' and token.dep_ == 'obj':
                    results['is_correct'] = False
                    results['errors'].append({
                        'type': 'hen_hun',
                        'message': "Use 'hen' as direct object, not 'hun'",
                        'context': token.text,
                        'replacement': 'hen'
                    })
                    corrected_text = corrected_text.replace(" hun ", " hen ", 1)
        
        results['corrected_text'] = corrected_text
        results['suggestions'] = self._generate_suggestions(text)
        return results
    
    def _generate_suggestions(self, text: str) -> List[str]:
        """Generate writing suggestions based on text analysis."""
        suggestions = []
        word_count = len(text.split())
        
        if word_count < 5:
            suggestions.append("Consider adding more details to your sentence.")
        elif word_count > 30:
            suggestions.append("Consider breaking this into multiple sentences.")
        
        if 'wordt' in text.lower() and 'door' in text.lower():
            suggestions.append("Consider using active voice for clearer communication.")
        
        return suggestions
    
    def explain_rule(self, rule_name: str) -> Optional[Dict]:
        """Get detailed explanation of a grammar rule."""
        rule = self.rules.get(rule_name)
        return {
            'name': rule.name,
            'description': rule.description,
            'examples': rule.examples
        } if rule else None
    
    def add_custom_rule(self, rule: GrammarRule) -> bool:
        """Add a custom grammar rule."""
        if rule.name and rule.pattern and rule.correction:
            self.rules[rule.name] = rule
            return True
        return False
    
    def get_available_rules(self) -> List[Dict]:
        """Get list of available grammar rules."""
        return [
            {
                'name': rule.name,
                'description': rule.description,
                'example_count': len(rule.examples)
            }
            for rule in self.rules.values()
        ]