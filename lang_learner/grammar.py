"""
Grammar Checking Module

Provides functionality to check and correct grammar in the target language.
"""

import re
from typing import Dict, List, Tuple, Optional
try:
    import spacy
    SPACY_AVAILABLE = True
    
    # Try to load the English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If model not found, download it
        import spacy.cli
        print("Downloading spaCy model (this may take a while)...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
except ImportError:
    import warnings
    warnings.warn("spaCy not installed. Grammar checking will be limited.")
    SPACY_AVAILABLE = False
    
    # Create a dummy nlp object for type checking
    class DummyNLP:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, text):
            return DummyDoc()
    
    class DummyDoc:
        def __init__(self, *args, **kwargs):
            self.sents = []
            self.tokens = []
    
    nlp = DummyNLP()
from dataclasses import dataclass

@dataclass
class GrammarRule:
    """Represents a grammar rule with its pattern and correction."""
    name: str
    pattern: str
    correction: str
    description: str
    examples: List[Tuple[str, str]]  # (incorrect, correct) pairs

class GrammarChecker:
    """Checks and corrects grammar in the target language."""
    
    def __init__(self, language: str = 'dutch', level: str = 'beginner'):
        """
        Initialize the grammar checker.
        
        Args:
            language: Target language (default: 'dutch')
            level: Proficiency level (beginner, intermediate, advanced)
        """
        self.language = language
        self.level = level
        self.rules = self._load_grammar_rules()
        
        # Try to load spaCy model for the language
        self.nlp = None
        try:
            model_name = {
                'dutch': 'nl_core_news_sm',
                'english': 'en_core_web_sm',
                'german': 'de_core_news_sm',
                'french': 'fr_core_news_sm',
                'spanish': 'es_core_news_sm'
            }.get(language.lower())
            
            if model_name:
                self.nlp = spacy.load(model_name)
        except Exception as e:
            print(f"Warning: Could not load spaCy model for {language}: {e}")
    
    def _load_grammar_rules(self) -> Dict[str, GrammarRule]:
        """Load grammar rules for the current language and level."""
        # This is a simplified example. In a real application, these rules would be more comprehensive
        # and possibly loaded from external files or a database.
        
        common_rules = {
            'dutch': {
                'double_subject': GrammarRule(
                    name="Double Subject",
                    pattern=r"\b([A-Z][a-z]*)\s+([A-Z][a-z]*)\s+(is|heeft|kan|wil|gaat|doet)\b",
                    correction="\1 \3",
                    description="Avoid using both a name and a pronoun as subject",
                    examples=[
                        ("Jan hij is mijn vriend.", "Jan is mijn vriend."),
                        ("Mijn moeder zij heeft een hond.", "Mijn moeder heeft een hond.")
                    ]
                ),
                'verb_placement': GrammarRule(
                    name="Verb Placement",
                    pattern=r"\b(Ik|Jij|Hij|Zij|Wij|Jullie|Zij)\s+\w+\s+(niet|altijd|vaak|soms|meestal|meestal|vaak|soms|zelden|nooit)\s+\w+\s*[.!?]$",
                    correction="\1 \2 niet \3",
                    description="In Dutch, the adverb comes after the verb in main clauses",
                    examples=[
                        ("Ik eet altijd ontbijt.", "Ik eet altijd ontbijt."),
                        ("Ik altijd eet ontbijt.", "Ik eet altijd ontbijt.")
                    ]
                )
            },
            'english': {
                'a_an': GrammarRule(
                    name="A/An Usage",
                    pattern=r"\b(a)\s+[aeiouAEIOU]\w*\b",
                    correction="an",
                    description="Use 'an' before words that start with a vowel sound",
                    examples=[
                        ("a apple", "an apple"),
                        ("a hour", "an hour")
                    ]
                )
            }
        }
        
        # Return rules for the current language or empty dict if not found
        return common_rules.get(self.language.lower(), {})
    
    def check(self, text: str) -> Dict:
        """
        Check the grammar of the given text.
        
        Args:
            text: The text to check
            
        Returns:
            Dictionary with results including errors and corrections
        """
        results = {
            'original_text': text,
            'is_correct': True,
            'errors': [],
            'corrected_text': text,
            'suggestions': []
        }
        
        if not text.strip():
            results['is_correct'] = False
            results['errors'].append({
                'type': 'empty_input',
                'message': 'No text provided',
                'context': '',
                'replacement': ''
            })
            return results
        
        # Apply each grammar rule
        corrected_text = text
        
        for rule_name, rule in self.rules.items():
            try:
                # Find all matches of the pattern
                for match in re.finditer(rule.pattern, text, re.IGNORECASE):
                    # Get the matched text
                    matched_text = match.group(0)
                    
                    # Apply the correction
                    if '\1' in rule.correction:
                        # If the correction contains groups, use them
                        corrected = match.expand(rule.correction)
                    else:
                        # Otherwise, replace the entire match
                        corrected = re.sub(rule.pattern, rule.correction, matched_text, flags=re.IGNORECASE)
                    
                    # Only add if there's an actual change
                    if matched_text != corrected:
                        results['is_correct'] = False
                        results['errors'].append({
                            'type': rule_name,
                            'message': rule.description,
                            'context': matched_text,
                            'replacement': corrected,
                            'examples': rule.examples
                        })
                        
                        # Update the corrected text
                        corrected_text = corrected_text.replace(matched_text, corrected, 1)
            except Exception as e:
                print(f"Error applying rule {rule_name}: {e}")
        
        # If we have spaCy, do more advanced analysis
        if self.nlp and len(text) < 1000:  # Limit text length for performance
            doc = self.nlp(text)
            
            # Check for common errors using spaCy's linguistic features
            # This is language-specific and would need to be expanded
            if self.language.lower() == 'dutch':
                # Example: Check for common Dutch errors
                for token in doc:
                    # Check for confusion between 'hen' and 'hun'
                    if token.text.lower() in ['hen', 'hun'] and token.dep_ == 'obj':
                        # 'Hun' as object is incorrect, should be 'hen'
                        if token.text.lower() == 'hun' and 'Case=Acc' in token.morph:
                            results['is_correct'] = False
                            results['errors'].append({
                                'type': 'hen_hun',
                                'message': "Use 'hen' as direct object, not 'hun'",
                                'context': token.text,
                                'replacement': 'hen',
                                'examples': [
                                    ("Ik geef het boek aan hun.", "Ik geef het boek aan hen.")
                                ]
                            })
                            corrected_text = corrected_text.replace(" hun ", " hen ", 1)
        
        results['corrected_text'] = corrected_text
        
        # Generate general suggestions based on the text
        self._generate_suggestions(results, text)
        
        return results
    
    def _generate_suggestions(self, results: Dict, text: str) -> None:
        """Generate general writing suggestions."""
        suggestions = []
        
        # Check text length
        word_count = len(text.split())
        if word_count < 5:
            suggestions.append("Your sentence is quite short. Consider adding more details.")
        elif word_count > 30:
            suggestions.append("Your sentence is quite long. Consider breaking it into multiple sentences for clarity.")
        
        # Check for passive voice (simplified)
        if 'wordt' in text.lower() and 'door' in text.lower():
            suggestions.append("Consider using active voice instead of passive for more direct communication.")
        
        results['suggestions'] = suggestions
    
    def explain_rule(self, rule_name: str) -> Optional[Dict]:
        """Get detailed explanation of a specific grammar rule."""
        rule = self.rules.get(rule_name)
        if not rule:
            return None
            
        return {
            'name': rule.name,
            'description': rule.description,
            'examples': rule.examples,
            'pattern': rule.pattern,
            'correction': rule.correction
        }
    
    def add_custom_rule(self, rule: GrammarRule) -> bool:
        """Add a custom grammar rule."""
        if not rule.name or not rule.pattern or not rule.correction:
            return False
            
        self.rules[rule.name] = rule
        return True
    
    def get_available_rules(self) -> List[Dict]:
        """Get a list of all available grammar rules."""
        return [
            {
                'name': rule.name,
                'description': rule.description,
                'example_count': len(rule.examples)
            }
            for rule in self.rules.values()
        ]
