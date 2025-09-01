"""
Tests for graph export utilities.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import networkx as nx

from graph.export_utils import export_to_network_format


class TestExportToNetworkFormat:
    """Test graph export functionality."""
    
    @pytest.fixture
    def sample_nodes_dir(self, tmp_path):
        """Create sample node directory."""
        nodes_dir = tmp_path / "nodes"
        nodes_dir.mkdir()
        
        # Create sample nodes
        for i in range(3):
            node_data = {
                "id": f"node_{i}",
                "title": f"Node {i}",
                "conversation_id": f"conv_{i}",
                "model": "test-model",
                "created": "2024-01-01",
                "updated": "2024-01-02",
                "safe_urls": ["http://example.com"],
                "openai_url": "http://openai.example.com",
                "extra_field": "should not be included"
            }
            
            with open(nodes_dir / f"node_{i}.json", 'w') as f:
                json.dump(node_data, f)
        
        return nodes_dir
    
    @pytest.fixture
    def sample_edges_file(self, tmp_path):
        """Create sample edges file."""
        edges = [
            ["node_0", "node_1", 0.8],
            ["node_0", "node_2", 0.6],
            ["node_1", "node_2", 0.7]
        ]
        
        edges_file = tmp_path / "edges.json"
        with open(edges_file, 'w') as f:
            json.dump(edges, f)
        
        return edges_file
    
    @pytest.mark.unit
    @patch('graph.export_utils.nx.write_gexf')
    def test_export_gexf_format(self, mock_write_gexf, sample_nodes_dir, sample_edges_file, tmp_path):
        """Test exporting to GEXF format."""
        output_file = tmp_path / "graph.gexf"
        
        export_to_network_format(
            str(sample_nodes_dir),
            str(sample_edges_file),
            "gexf",
            str(output_file)
        )
        
        # Check that write_gexf was called
        mock_write_gexf.assert_called_once()
        
        # Verify the graph structure
        graph_arg = mock_write_gexf.call_args[0][0]
        assert isinstance(graph_arg, nx.Graph)
        assert len(graph_arg.nodes) == 3
        assert len(graph_arg.edges) == 3
    
    @pytest.mark.unit
    @patch('graph.export_utils.nx.write_graphml')
    def test_export_graphml_format(self, mock_write_graphml, sample_nodes_dir, sample_edges_file, tmp_path):
        """Test exporting to GraphML format."""
        output_file = tmp_path / "graph.graphml"
        
        export_to_network_format(
            str(sample_nodes_dir),
            str(sample_edges_file),
            "graphml",
            str(output_file)
        )
        
        mock_write_graphml.assert_called_once()
    
    @pytest.mark.unit
    @patch('graph.export_utils.nx.write_gml')
    def test_export_gml_format(self, mock_write_gml, sample_nodes_dir, sample_edges_file, tmp_path):
        """Test exporting to GML format."""
        output_file = tmp_path / "graph.gml"
        
        export_to_network_format(
            str(sample_nodes_dir),
            str(sample_edges_file),
            "gml",
            str(output_file)
        )
        
        mock_write_gml.assert_called_once()
    
    @pytest.mark.unit
    def test_export_unsupported_format(self, sample_nodes_dir, sample_edges_file, tmp_path, capsys):
        """Test handling of unsupported format."""
        output_file = tmp_path / "graph.xyz"
        
        export_to_network_format(
            str(sample_nodes_dir),
            str(sample_edges_file),
            "xyz",
            str(output_file)
        )
        
        captured = capsys.readouterr()
        assert "Unsupported format: xyz" in captured.out
    
    @pytest.mark.unit
    @patch('graph.export_utils.nx.write_gexf')
    def test_export_with_custom_keys(self, mock_write_gexf, sample_nodes_dir, sample_edges_file, tmp_path):
        """Test exporting with custom keys to include."""
        output_file = tmp_path / "graph.gexf"
        
        export_to_network_format(
            str(sample_nodes_dir),
            str(sample_edges_file),
            "gexf",
            str(output_file),
            keys_to_include=["title", "model"]
        )
        
        # Verify only specified keys are included
        graph_arg = mock_write_gexf.call_args[0][0]
        for node_id in graph_arg.nodes:
            node_data = graph_arg.nodes[node_id]
            assert "title" in node_data
            assert "model" in node_data
            assert "extra_field" not in node_data
            assert "conversation_id" not in node_data
    
    @pytest.mark.unit
    def test_export_nonexistent_nodes_dir(self, sample_edges_file, tmp_path, capsys):
        """Test handling of non-existent nodes directory."""
        output_file = tmp_path / "graph.gexf"
        
        export_to_network_format(
            "/nonexistent/path",
            str(sample_edges_file),
            "gexf",
            str(output_file)
        )
        
        captured = capsys.readouterr()
        assert "does not exist" in captured.out
    
    @pytest.mark.unit
    @patch('graph.export_utils.nx.write_gexf')
    def test_export_with_none_nodes_dir(self, mock_write_gexf, sample_edges_file, tmp_path):
        """Test exporting with None nodes directory (edges only)."""
        output_file = tmp_path / "graph.gexf"
        
        export_to_network_format(
            None,
            str(sample_edges_file),
            "gexf",
            str(output_file)
        )
        
        # Should still create graph with edges
        mock_write_gexf.assert_called_once()
        graph_arg = mock_write_gexf.call_args[0][0]
        # Nodes are created implicitly from edges even without node data
        assert len(graph_arg.nodes) == 3  # 3 nodes created from edge endpoints
        assert len(graph_arg.edges) == 3
    
    @pytest.mark.unit
    @patch('graph.export_utils.nx.write_gexf')
    def test_export_with_none_weight_edges(self, mock_write_gexf, sample_nodes_dir, tmp_path):
        """Test handling edges with None weights."""
        # Create edges with None weights
        edges = [
            ["node_0", "node_1", 0.8],
            ["node_0", "node_2", None],
            ["node_1", "node_2", 0.7]
        ]
        
        edges_file = tmp_path / "edges_with_none.json"
        with open(edges_file, 'w') as f:
            json.dump(edges, f)
        
        output_file = tmp_path / "graph.gexf"
        
        export_to_network_format(
            str(sample_nodes_dir),
            str(edges_file),
            "gexf",
            str(output_file)
        )
        
        # Should skip edges with None weight
        graph_arg = mock_write_gexf.call_args[0][0]
        assert len(graph_arg.edges) == 2  # Only 2 valid edges
    
    @pytest.mark.unit
    @patch('graph.export_utils.nx.write_gexf')
    def test_export_replaces_none_values(self, mock_write_gexf, tmp_path):
        """Test that None values in node data are replaced with empty strings."""
        nodes_dir = tmp_path / "nodes"
        nodes_dir.mkdir()
        
        # Create node with None values
        node_data = {
            "title": "Test Node",
            "model": None,
            "created": None,
            "updated": "2024-01-01"
        }
        
        with open(nodes_dir / "node.json", 'w') as f:
            json.dump(node_data, f)
        
        edges = [["node", "node", 1.0]]
        edges_file = tmp_path / "edges.json"
        with open(edges_file, 'w') as f:
            json.dump(edges, f)
        
        output_file = tmp_path / "graph.gexf"
        
        export_to_network_format(
            str(nodes_dir),
            str(edges_file),
            "gexf",
            str(output_file)
        )
        
        graph_arg = mock_write_gexf.call_args[0][0]
        node_data = graph_arg.nodes["node"]
        assert node_data["model"] == ""  # None replaced with ""
        assert node_data["created"] == ""
        assert node_data["updated"] == "2024-01-01"