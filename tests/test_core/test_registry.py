"""
Tests for IntegrationRegistry.
"""

import pytest
from unittest.mock import Mock, patch

from integrations.base import IntegrationRegistry, DataSource, Node, Edge
from typing import Iterator, Any, List, Optional


class TestIntegrationRegistry:
    """Test the integration registry system."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clear registry before and after each test."""
        # Clear registry before test
        IntegrationRegistry._integrations.clear()
        yield
        # Clear registry after test
        IntegrationRegistry._integrations.clear()
    
    @pytest.mark.unit
    def test_register_integration(self):
        """Test registering a new integration."""
        # Create a mock integration class
        class MockIntegration(DataSource):
            def load(self, source: Any) -> None:
                pass
            
            def extract_nodes(self) -> Iterator[Node]:
                yield Node("test", "test", {})
            
            def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
                yield Edge("a", "b", "test")
            
            def get_node_content_for_embedding(self, node: Node) -> str:
                return "test content"
        
        # Register the integration
        IntegrationRegistry.register("mock_integration", MockIntegration)
        
        # Check it was registered
        assert "mock_integration" in IntegrationRegistry._integrations
        assert IntegrationRegistry._integrations["mock_integration"] == MockIntegration
    
    @pytest.mark.unit
    def test_get_integration(self):
        """Test retrieving a registered integration."""
        class TestIntegration(DataSource):
            def load(self, source: Any) -> None:
                pass
            
            def extract_nodes(self) -> Iterator[Node]:
                return iter([])
            
            def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
                return iter([])
            
            def get_node_content_for_embedding(self, node: Node) -> str:
                return ""
        
        IntegrationRegistry.register("test", TestIntegration)
        
        retrieved = IntegrationRegistry.get("test")
        assert retrieved == TestIntegration
    
    @pytest.mark.unit
    def test_get_nonexistent_integration(self):
        """Test retrieving a non-existent integration returns None."""
        result = IntegrationRegistry.get("nonexistent")
        assert result is None
    
    @pytest.mark.unit
    def test_list_integrations(self):
        """Test listing all registered integrations."""
        class Integration1(DataSource):
            def load(self, source: Any) -> None:
                pass
            def extract_nodes(self) -> Iterator[Node]:
                return iter([])
            def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
                return iter([])
            def get_node_content_for_embedding(self, node: Node) -> str:
                return ""
        
        class Integration2(DataSource):
            def load(self, source: Any) -> None:
                pass
            def extract_nodes(self) -> Iterator[Node]:
                return iter([])
            def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
                return iter([])
            def get_node_content_for_embedding(self, node: Node) -> str:
                return ""
        
        IntegrationRegistry.register("integration1", Integration1)
        IntegrationRegistry.register("integration2", Integration2)
        
        integrations = IntegrationRegistry.list_available()
        assert "integration1" in integrations
        assert "integration2" in integrations
        assert len(integrations) == 2
    
    @pytest.mark.unit
    def test_register_duplicate_integration(self):
        """Test that registering with same name overwrites."""
        class Integration1(DataSource):
            version = 1
            def load(self, source: Any) -> None:
                pass
            def extract_nodes(self) -> Iterator[Node]:
                return iter([])
            def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
                return iter([])
            def get_node_content_for_embedding(self, node: Node) -> str:
                return ""
        
        class Integration2(DataSource):
            version = 2
            def load(self, source: Any) -> None:
                pass
            def extract_nodes(self) -> Iterator[Node]:
                return iter([])
            def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
                return iter([])
            def get_node_content_for_embedding(self, node: Node) -> str:
                return ""
        
        IntegrationRegistry.register("test", Integration1)
        IntegrationRegistry.register("test", Integration2)
        
        retrieved = IntegrationRegistry.get("test")
        assert retrieved == Integration2
        assert retrieved.version == 2
    
    @pytest.mark.unit
    def test_clear_registry(self):
        """Test clearing the registry."""
        class TestIntegration(DataSource):
            def load(self, source: Any) -> None:
                pass
            def extract_nodes(self) -> Iterator[Node]:
                return iter([])
            def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
                return iter([])
            def get_node_content_for_embedding(self, node: Node) -> str:
                return ""
        
        IntegrationRegistry.register("test", TestIntegration)
        assert len(IntegrationRegistry.list_available()) > 0
        
        IntegrationRegistry._integrations.clear()
        assert len(IntegrationRegistry.list_available()) == 0