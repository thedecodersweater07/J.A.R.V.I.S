from typing import Dict, Any, List, Optional
import networkx as nx
import json
from pathlib import Path

class KnowledgeGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_node(self, node_id: str, **attrs):
        """Add a node to the knowledge graph"""
        self.graph.add_node(node_id, **attrs)
        
    def add_edge(self, source: str, target: str, **attrs):
        """Add an edge between nodes"""
        self.graph.add_edge(source, target, **attrs)
        
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node attributes"""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None
        
    def get_connected_nodes(self, node_id: str) -> List[str]:
        """Get nodes connected to the given node"""
        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        return []

    def save(self, path: str):
        """Save the knowledge graph"""
        data = nx.node_link_data(self.graph)
        with open(path, 'w') as f:
            json.dump(data, f)
            
    def load(self, path: str):
        """Load a knowledge graph"""
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
