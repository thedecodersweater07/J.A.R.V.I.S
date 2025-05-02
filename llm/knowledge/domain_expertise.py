import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class DomainKnowledge:
    domain: str
    concepts: Set[str]
    relationships: Dict[str, List[str]]
    rules: List[str]
    confidence: float

class DomainExpert:
    """Manages domain-specific knowledge and expertise."""
    
    def __init__(self, domains_config: Optional[str] = None):
        self.domains = {}
        self.active_domain = None
        if domains_config:
            self.load_domains(domains_config)

    def load_domains(self, config_path: str) -> bool:
        """Load domain configurations from file."""
        try:
            with open(config_path, 'r') as f:
                configs = json.load(f)
            
            for domain_config in configs:
                domain = domain_config['name']
                self.domains[domain] = DomainKnowledge(
                    domain=domain,
                    concepts=set(domain_config.get('concepts', [])),
                    relationships=domain_config.get('relationships', {}),
                    rules=domain_config.get('rules', []),
                    confidence=domain_config.get('confidence', 0.8)
                )
            return True
        except Exception as e:
            logger.error(f"Failed to load domain configurations: {e}")
            return False

    def switch_domain(self, domain: str) -> bool:
        """Switch active domain context."""
        if domain in self.domains:
            self.active_domain = domain
            return True
        return False

    def get_domain_knowledge(self, concept: str) -> Optional[Dict]:
        """Retrieve domain-specific knowledge about a concept."""
        if not self.active_domain:
            return None
            
        domain = self.domains[self.active_domain]
        if concept not in domain.concepts:
            return None
            
        return {
            'concept': concept,
            'domain': self.active_domain,
            'relationships': domain.relationships.get(concept, []),
            'rules': [rule for rule in domain.rules if concept in rule],
            'confidence': domain.confidence
        }

    def validate_domain_consistency(self, statements: List[str]) -> Dict[str, bool]:
        """Validate statements against domain rules and knowledge."""
        if not self.active_domain:
            return {}
            
        domain = self.domains[self.active_domain]
        results = {}
        
        for statement in statements:
            # Check statement against domain rules
            consistency = self._check_consistency(statement, domain.rules)
            results[statement] = consistency
            
        return results

    def _check_consistency(self, statement: str, rules: List[str]) -> bool:
        """Check if statement is consistent with domain rules."""
        # Simplified implementation - could be enhanced with formal logic
        for rule in rules:
            if rule in statement:
                return True
        return False

    def suggest_related_concepts(self, concept: str, limit: int = 5) -> List[str]:
        """Suggest related concepts from the active domain."""
        if not self.active_domain:
            return []
            
        domain = self.domains[self.active_domain]
        if concept not in domain.concepts:
            return []
            
        related = domain.relationships.get(concept, [])
        return related[:limit]
