import networkx as nx
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_cache = {}
        
    def add_entity(self, entity_id: str, properties: Dict[str, Any]):
        """Add an entity to the knowledge graph"""
        self.graph.add_node(entity_id, **properties)
        self.entity_cache[entity_id] = properties

    def add_relationship(self, from_entity: str, to_entity: str, relationship_type: str, properties: Dict[str, Any] = None):
        """Add a relationship between entities"""
        self.graph.add_edge(from_entity, to_entity, 
                           relationship=relationship_type, 
                           **properties if properties else {})

    def query_relationships(self, entity_id: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query relationships for an entity"""
        relationships = []
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            if not relationship_type or data.get('relationship') == relationship_type:
                relationships.append({
                    'target': target,
                    'type': data.get('relationship'),
                    'properties': data
                })
        return relationships

    def find_path(self, start_entity: str, end_entity: str, max_depth: int = 5) -> List[str]:
        """Find the shortest path between entities"""
        try:
            path = nx.shortest_path(self.graph, start_entity, end_entity)
            return path if len(path) <= max_depth else []
        except nx.NetworkXNoPath:
            return []
