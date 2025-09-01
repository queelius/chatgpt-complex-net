"""
Integrations framework for semantic network generation from various data sources.
"""

from .base import DataSource, Node, Edge, IntegrationRegistry
from .loader import load_integration, discover_integrations

__all__ = [
    'DataSource',
    'Node', 
    'Edge',
    'IntegrationRegistry',
    'load_integration',
    'discover_integrations'
]