"""
Dynamic loader for integration plugins.
"""

import os
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .base import DataSource, IntegrationRegistry

logger = logging.getLogger(__name__)

def discover_integrations(integrations_dir: Optional[Path] = None) -> List[str]:
    """
    Discover all available integrations in the integrations directory.
    
    Args:
        integrations_dir: Path to integrations directory (default: ./integrations)
        
    Returns:
        List of discovered integration names
    """
    if integrations_dir is None:
        integrations_dir = Path(__file__).parent
    
    discovered = []
    
    for item in integrations_dir.iterdir():
        if item.is_dir() and not item.name.startswith('_'):
            # Look for Python files in the integration directory
            for py_file in item.glob('*.py'):
                if py_file.name.startswith('_'):
                    continue
                    
                try:
                    module_name = f"integrations.{item.name}.{py_file.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                issubclass(attr, DataSource) and 
                                attr != DataSource and
                                not getattr(attr, '__abstractmethods__', None)):
                                # Register with dot notation for specific classes
                                full_name = f"{item.name}.{py_file.stem}"
                                IntegrationRegistry.register(full_name, attr)
                                discovered.append(full_name)
                                logger.info(f"Discovered integration: {full_name}")
                except Exception as e:
                    logger.warning(f"Failed to load integration {item.name}.{py_file.stem}: {e}")
    
    return discovered

def load_integration(
    name: str, 
    config: Optional[Dict[str, Any]] = None,
    auto_discover: bool = True
) -> DataSource:
    """
    Load a specific integration by name.
    
    Args:
        name: Name of the integration to load
        config: Configuration for the integration
        auto_discover: Whether to auto-discover integrations if not found
        
    Returns:
        Initialized DataSource instance
    """
    integration = IntegrationRegistry.get(name)
    
    if not integration and auto_discover:
        discovered = discover_integrations()
        logger.info(f"Auto-discovered {len(discovered)} integrations")
        integration = IntegrationRegistry.get(name)
    
    if not integration:
        available = IntegrationRegistry.list_available()
        raise ValueError(
            f"Integration '{name}' not found. "
            f"Available: {', '.join(available) if available else 'none'}"
        )
    
    return IntegrationRegistry.create_instance(name, config)