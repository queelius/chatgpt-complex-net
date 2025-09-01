"""
Tests for edge generation utilities.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call
import numpy as np

from graph.edge_utils import (
    filter_edges,
    generate_edges_from_node_embeddings
)


class TestFilterEdges:
    """Test edge filtering function."""
    
    @pytest.fixture
    def sample_edges_file(self, tmp_path):
        """Create a sample edges file."""
        edges = [
            ["node1", "node2", 0.9],
            ["node1", "node3", 0.7],
            ["node2", "node3", 0.5],
            ["node3", "node4", 0.3],
            ["node4", "node5", 0.1]
        ]
        
        input_file = tmp_path / "edges.json"
        with open(input_file, 'w') as f:
            json.dump(edges, f)
        
        return input_file
    
    @pytest.mark.unit
    def test_filter_edges_basic(self, sample_edges_file, tmp_path, capsys):
        """Test basic edge filtering."""
        output_file = tmp_path / "filtered.json"
        
        filter_edges(str(sample_edges_file), str(output_file), cutoff=0.5)
        
        # Check output file exists
        assert output_file.exists()
        
        # Load filtered edges
        with open(output_file, 'r') as f:
            filtered = json.load(f)
        
        # Should have 3 edges with weight >= 0.5
        assert len(filtered) == 3
        assert all(edge[2] >= 0.5 for edge in filtered)
        
        # Check printed statistics
        captured = capsys.readouterr()
        assert "Before filtering:" in captured.out
        assert "After filtering" in captured.out
        assert "Nodes:" in captured.out
        assert "Edges:" in captured.out
        assert "Density:" in captured.out
    
    @pytest.mark.unit
    def test_filter_edges_high_cutoff(self, sample_edges_file, tmp_path):
        """Test filtering with high cutoff."""
        output_file = tmp_path / "filtered_high.json"
        
        filter_edges(str(sample_edges_file), str(output_file), cutoff=0.8)
        
        with open(output_file, 'r') as f:
            filtered = json.load(f)
        
        # Should only have 1 edge
        assert len(filtered) == 1
        assert filtered[0][2] == 0.9
    
    @pytest.mark.unit
    def test_filter_edges_no_edges_remain(self, sample_edges_file, tmp_path):
        """Test filtering when no edges remain."""
        output_file = tmp_path / "filtered_none.json"
        
        filter_edges(str(sample_edges_file), str(output_file), cutoff=1.0)
        
        with open(output_file, 'r') as f:
            filtered = json.load(f)
        
        assert filtered == []
    
    @pytest.mark.unit
    def test_filter_edges_statistics(self, sample_edges_file, tmp_path, capsys):
        """Test that statistics are calculated correctly."""
        output_file = tmp_path / "filtered.json"
        
        filter_edges(str(sample_edges_file), str(output_file), cutoff=0.5)
        
        captured = capsys.readouterr()
        
        # Before: 5 nodes, 5 edges
        assert "Nodes: 5" in captured.out
        assert "Edges: 5" in captured.out
        
        # After: fewer nodes and edges
        assert "Nodes: 3" in captured.out or "Nodes: 4" in captured.out
        assert "Edges: 3" in captured.out
    
    @pytest.mark.unit
    def test_filter_edges_empty_input(self, tmp_path):
        """Test filtering empty edge list."""
        input_file = tmp_path / "empty.json"
        output_file = tmp_path / "empty_filtered.json"
        
        with open(input_file, 'w') as f:
            json.dump([], f)
        
        filter_edges(str(input_file), str(output_file), cutoff=0.5)
        
        with open(output_file, 'r') as f:
            filtered = json.load(f)
        
        assert filtered == []


class TestGenerateEdges:
    """Test edge generation from node embeddings."""
    
    @pytest.fixture
    def sample_nodes_dir(self, tmp_path):
        """Create sample node files with embeddings."""
        nodes_dir = tmp_path / "nodes"
        nodes_dir.mkdir()
        
        # Create sample node files
        for i in range(3):
            node_data = {
                "id": f"node_{i}",
                "content": {"text": f"Content {i}"},
                "embeddings": {
                    "role_aggregate": {
                        "vector": np.random.randn(10).tolist()
                    },
                    "user": {
                        "vector": np.random.randn(10).tolist()
                    },
                    "assistant": {
                        "vector": np.random.randn(10).tolist()
                    }
                }
            }
            
            with open(nodes_dir / f"node_{i}.json", 'w') as f:
                json.dump(node_data, f)
        
        return nodes_dir
    
    @pytest.mark.unit
    @patch('graph.edge_utils.utils.cosine_similarity')
    def test_generate_edges_basic(self, mock_cosine_sim, sample_nodes_dir, tmp_path):
        """Test basic edge generation."""
        output_file = tmp_path / "edges.json"
        
        # Mock cosine similarity to return predictable values
        mock_cosine_sim.side_effect = lambda a, b: 0.5
        
        generate_edges_from_node_embeddings(
            str(sample_nodes_dir),
            str(output_file)
        )
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            edges = json.load(f)
        
        # With 3 nodes, should have 3 edges (3 choose 2)
        assert len(edges) == 3
        
        # Check edge structure
        for edge in edges:
            assert len(edge) == 3  # [source, target, weight]
            assert isinstance(edge[0], str)
            assert isinstance(edge[1], str)
            assert isinstance(edge[2], (int, float))
    
    @pytest.mark.unit
    def test_generate_edges_custom_embedding_type(self, sample_nodes_dir, tmp_path):
        """Test edge generation with custom embedding type."""
        output_file = tmp_path / "edges.json"
        
        # Mock similarity function
        mock_sim = Mock(return_value=0.7)
        
        generate_edges_from_node_embeddings(
            str(sample_nodes_dir),
            str(output_file),
            sim=mock_sim,
            embedding_type="user"
        )
        
        with open(output_file, 'r') as f:
            edges = json.load(f)
        
        assert len(edges) == 3
        assert all(edge[2] == 0.7 for edge in edges)
        
        # Verify the mock was called 3 times (for 3 pairs)
        assert mock_sim.call_count == 3
    
    @pytest.mark.unit
    def test_generate_edges_with_actual_similarity(self, sample_nodes_dir, tmp_path):
        """Test edge generation with actual cosine similarity."""
        output_file = tmp_path / "edges.json"
        
        # This will use the actual cosine similarity function
        generate_edges_from_node_embeddings(
            str(sample_nodes_dir),
            str(output_file)
        )
        
        with open(output_file, 'r') as f:
            edges = json.load(f)
        
        assert len(edges) == 3
        
        # Check that weights are in valid range
        for edge in edges:
            assert -1 <= edge[2] <= 1
    
    @pytest.mark.unit
    def test_generate_edges_single_node(self, tmp_path):
        """Test edge generation with single node."""
        nodes_dir = tmp_path / "single_node"
        nodes_dir.mkdir()
        
        # Create single node
        node_data = {
            "id": "single",
            "embeddings": {
                "role_aggregate": {
                    "combined": [1, 2, 3]
                }
            }
        }
        
        with open(nodes_dir / "single.json", 'w') as f:
            json.dump(node_data, f)
        
        output_file = tmp_path / "edges.json"
        
        generate_edges_from_node_embeddings(
            str(nodes_dir),
            str(output_file)
        )
        
        with open(output_file, 'r') as f:
            edges = json.load(f)
        
        # No edges with single node
        assert edges == []
    
    @pytest.mark.unit
    def test_generate_edges_missing_embeddings(self, tmp_path):
        """Test handling nodes without embeddings."""
        nodes_dir = tmp_path / "nodes"
        nodes_dir.mkdir()
        
        # Create node without embeddings
        node_data = {
            "id": "no_embedding",
            "content": {"text": "Content"}
        }
        
        with open(nodes_dir / "node.json", 'w') as f:
            json.dump(node_data, f)
        
        output_file = tmp_path / "edges.json"
        
        # Should handle gracefully
        generate_edges_from_node_embeddings(
            str(nodes_dir),
            str(output_file)
        )
        
        with open(output_file, 'r') as f:
            edges = json.load(f)
        
        assert edges == []
    
    @pytest.mark.unit
    @patch('graph.edge_utils.glob.glob')
    def test_generate_edges_no_json_files(self, mock_glob, tmp_path):
        """Test when no JSON files are found."""
        mock_glob.return_value = []
        
        output_file = tmp_path / "edges.json"
        
        generate_edges_from_node_embeddings(
            str(tmp_path),
            str(output_file)
        )
        
        with open(output_file, 'r') as f:
            edges = json.load(f)
        
        assert edges == []
    
    @pytest.mark.unit
    def test_generate_edges_large_dataset(self, tmp_path):
        """Test edge generation with larger dataset."""
        nodes_dir = tmp_path / "many_nodes"
        nodes_dir.mkdir()
        
        # Create 10 nodes
        for i in range(10):
            node_data = {
                "id": f"node_{i}",
                "embeddings": {
                    "role_aggregate": {
                        "vector": np.random.randn(50).tolist()
                    }
                }
            }
            with open(nodes_dir / f"node_{i}.json", 'w') as f:
                json.dump(node_data, f)
        
        output_file = tmp_path / "edges.json"
        
        generate_edges_from_node_embeddings(
            str(nodes_dir),
            str(output_file)
        )
        
        with open(output_file, 'r') as f:
            edges = json.load(f)
        
        # Should have (10 * 9) / 2 = 45 edges
        assert len(edges) == 45
        
        # All edges should have valid structure
        for edge in edges:
            assert len(edge) == 3
            assert isinstance(edge[2], float)
            assert -1 <= edge[2] <= 1  # Cosine similarity range
    
    @pytest.mark.unit
    def test_generate_edges_mixed_embedding_presence(self, tmp_path):
        """Test when some nodes have embeddings and others don't."""
        nodes_dir = tmp_path / "mixed_nodes"
        nodes_dir.mkdir()
        
        # Create nodes with embeddings
        for i in range(2):
            node_data = {
                "id": f"node_with_{i}",
                "embeddings": {
                    "role_aggregate": {
                        "vector": [1.0] * 10
                    }
                }
            }
            with open(nodes_dir / f"with_{i}.json", 'w') as f:
                json.dump(node_data, f)
        
        # Create nodes without embeddings
        for i in range(2):
            node_data = {
                "id": f"node_without_{i}",
                "content": {"text": "No embedding"}
            }
            with open(nodes_dir / f"without_{i}.json", 'w') as f:
                json.dump(node_data, f)
        
        output_file = tmp_path / "edges.json"
        
        generate_edges_from_node_embeddings(
            str(nodes_dir),
            str(output_file)
        )
        
        with open(output_file, 'r') as f:
            edges = json.load(f)
        
        # Should only have edges between nodes with embeddings (1 edge)
        assert len(edges) == 1
        # Node IDs are based on filename, not the "id" field
        assert edges[0][0].startswith("with_")
        assert edges[0][1].startswith("with_")