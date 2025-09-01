"""
Base class for chat log integrations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from integrations.base import DataSource, Node, Edge

class ChatLogSource(DataSource):
    """Base class for all chat log data sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.conversations = []
        self.link_strategy = config.get('link_strategy', 'temporal') if config else 'temporal'
        self.temporal_window = config.get('temporal_window', 3600) if config else 3600
    
    def get_node_content_for_embedding(self, node: Node) -> str:
        """Extract text content from a conversation node for embedding."""
        messages = node.content.get('messages', [])
        
        role_weights = self.config.get('role_weights', {
            'user': 1.5,
            'assistant': 1.0,
            'system': 0.5
        })
        
        weighted_texts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            weight = role_weights.get(role, 1.0)
            
            if weight > 1:
                weighted_texts.extend([content] * int(weight))
            else:
                weighted_texts.append(content)
        
        return ' '.join(weighted_texts)
    
    def extract_edges(self, nodes: Optional[List[Node]] = None) -> List[Edge]:
        """
        Extract edges based on the configured linking strategy.
        """
        edges = []
        
        if not nodes:
            nodes = list(self.extract_nodes())
        
        if self.link_strategy == 'temporal':
            edges.extend(self._temporal_edges(nodes))
        elif self.link_strategy == 'sequential':
            edges.extend(self._sequential_edges(nodes))
        elif self.link_strategy == 'topic':
            edges.extend(self._topic_edges(nodes))
        elif self.link_strategy == 'none':
            pass
        
        return edges
    
    def _temporal_edges(self, nodes: List[Node]) -> List[Edge]:
        """Create edges between temporally close conversations."""
        edges = []
        
        sorted_nodes = sorted(nodes, key=lambda n: n.timestamp or datetime.min)
        
        for i, node1 in enumerate(sorted_nodes):
            if not node1.timestamp:
                continue
                
            for node2 in sorted_nodes[i+1:]:
                if not node2.timestamp:
                    continue
                    
                time_diff = abs((node2.timestamp - node1.timestamp).total_seconds())
                
                if time_diff <= self.temporal_window:
                    weight = 1.0 - (time_diff / self.temporal_window)
                    edges.append(Edge(
                        source_id=node1.id,
                        target_id=node2.id,
                        type='temporal',
                        weight=weight,
                        metadata={'time_diff_seconds': time_diff}
                    ))
                else:
                    break
        
        return edges
    
    def _sequential_edges(self, nodes: List[Node]) -> List[Edge]:
        """Create edges between sequential conversations."""
        edges = []
        sorted_nodes = sorted(nodes, key=lambda n: n.timestamp or datetime.min)
        
        for i in range(len(sorted_nodes) - 1):
            edges.append(Edge(
                source_id=sorted_nodes[i].id,
                target_id=sorted_nodes[i+1].id,
                type='sequential',
                weight=1.0
            ))
        
        return edges
    
    def _topic_edges(self, nodes: List[Node]) -> List[Edge]:
        """Create edges based on shared topics/keywords."""
        edges = []
        
        for i, node1 in enumerate(nodes):
            topics1 = set(node1.metadata.get('topics', []))
            if not topics1:
                continue
                
            for node2 in nodes[i+1:]:
                topics2 = set(node2.metadata.get('topics', []))
                if not topics2:
                    continue
                    
                shared = topics1.intersection(topics2)
                if shared:
                    weight = len(shared) / max(len(topics1), len(topics2))
                    edges.append(Edge(
                        source_id=node1.id,
                        target_id=node2.id,
                        type='topic',
                        weight=weight,
                        metadata={'shared_topics': list(shared)}
                    ))
        
        return edges