"""
LLM Semantic Network - A toolkit for generating semantic networks from various data sources.
"""

__version__ = "0.1.0"
__author__ = "Alex Towell"

# Only import when used as a package
try:
    from integrations.base import DataSource, Node, Edge, IntegrationRegistry
    
    __all__ = [
        'DataSource',
        'Node',
        'Edge',
        'IntegrationRegistry'
    ]
except ImportError:
    # When running as scripts or tests, imports might not be available
    pass