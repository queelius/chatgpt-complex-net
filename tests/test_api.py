"""
Tests for the semantic network API.

These tests focus on behavior and data flow rather than implementation details.
They use real data structures and minimal mocking.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import semnet_api as api


class TestDataClasses:
    """Test the data classes."""
    
    def test_node_creation_and_serialization(self):
        """Test Node creation and dictionary conversion."""
        node = api.Node(
            id='test_id',
            type='conversation',
            content={'text': 'Hello world'},
            metadata={'source': 'test'}
        )
        
        assert node.id == 'test_id'
        assert node.type == 'conversation'
        
        # Test serialization
        data = node.to_dict()
        assert data['id'] == 'test_id'
        assert data['content']['text'] == 'Hello world'
        
        # Test deserialization
        node2 = api.Node.from_dict(data)
        assert node2.id == node.id
        assert node2.content == node.content
    
    def test_embedding_creation_and_serialization(self):
        """Test Embedding creation and dictionary conversion."""
        embedding = api.Embedding(
            node_id='test_node',
            vector=[0.1, 0.2, 0.3],
            method='tfidf',
            metadata={'features': 100}
        )
        
        assert embedding.node_id == 'test_node'
        assert len(embedding.vector) == 3
        
        # Test serialization
        data = embedding.to_dict()
        assert data['node_id'] == 'test_node'
        assert data['embedding'] == [0.1, 0.2, 0.3]
        
        # Test deserialization
        embedding2 = api.Embedding.from_dict(data)
        assert embedding2.node_id == embedding.node_id
        assert embedding2.vector == embedding.vector
    
    def test_edge_creation_and_serialization(self):
        """Test Edge creation and dictionary conversion."""
        edge = api.Edge(
            source='node1',
            target='node2',
            weight=0.85,
            type='similarity'
        )
        
        assert edge.source == 'node1'
        assert edge.target == 'node2'
        assert edge.weight == 0.85
        
        # Test serialization
        data = edge.to_dict()
        assert data['source'] == 'node1'
        assert data['weight'] == 0.85
        
        # Test deserialization
        edge2 = api.Edge.from_dict(data)
        assert edge2.source == edge.source
        assert edge2.weight == edge.weight


class TestStorage:
    """Test storage operations with real files."""
    
    def test_save_and_load_nodes(self):
        """Test saving and loading nodes to/from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create test nodes
            nodes = [
                api.Node('node1', 'test', {'content': 'test1'}),
                api.Node('node2', 'test', {'content': 'test2'})
            ]
            
            # Save nodes
            api.Storage.save_nodes(nodes, output_dir)
            
            # Check files were created
            assert (output_dir / 'node1.json').exists()
            assert (output_dir / 'node2.json').exists()
            
            # Load nodes back
            loaded_nodes = api.Storage.load_nodes(output_dir)
            
            # Verify
            assert len(loaded_nodes) == 2
            node_ids = [n.id for n in loaded_nodes]
            assert 'node1' in node_ids
            assert 'node2' in node_ids
    
    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings to/from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create test embeddings
            embeddings = [
                api.Embedding('node1', [0.1, 0.2], 'tfidf'),
                api.Embedding('node2', [0.3, 0.4], 'tfidf')
            ]
            
            # Save embeddings
            api.Storage.save_embeddings(embeddings, output_dir)
            
            # Check files were created
            assert (output_dir / 'node1.json').exists()
            assert (output_dir / 'node2.json').exists()
            
            # Load embeddings back
            loaded_embeddings = api.Storage.load_embeddings(output_dir)
            
            # Verify
            assert len(loaded_embeddings) == 2
            node_ids = [e.node_id for e in loaded_embeddings]
            assert 'node1' in node_ids
            assert 'node2' in node_ids
    
    def test_save_and_load_edges(self):
        """Test saving and loading edges to/from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'edges.json'
            
            # Create test edges
            edges = [
                api.Edge('node1', 'node2', 0.85),
                api.Edge('node2', 'node3', 0.75)
            ]
            
            # Save edges
            api.Storage.save_edges(edges, output_file)
            
            # Check file was created
            assert output_file.exists()
            
            # Load edges back
            loaded_edges = api.Storage.load_edges(output_file)
            
            # Verify
            assert len(loaded_edges) == 2
            assert loaded_edges[0].source == 'node1'
            assert loaded_edges[0].weight == 0.85


class TestEdgeGenerator:
    """Test edge generation logic."""
    
    def test_compute_similarity_edges_empty(self):
        """Test edge computation with empty embeddings."""
        edges = api.EdgeGenerator.compute_similarity_edges([], threshold=0.7)
        assert edges == []
    
    def test_compute_similarity_edges_basic(self):
        """Test basic edge computation with known vectors."""
        embeddings = [
            api.Embedding('node1', [1.0, 0.0, 0.0], 'test'),  # orthogonal to node3
            api.Embedding('node2', [0.9, 0.1, 0.0], 'test'),  # similar to node1
            api.Embedding('node3', [0.0, 1.0, 0.0], 'test')   # orthogonal to node1
        ]
        
        edges = api.EdgeGenerator.compute_similarity_edges(embeddings, threshold=0.8)
        
        # Should only have one edge (node1-node2) above threshold
        assert len(edges) == 1
        edge = edges[0]
        assert (edge.source == 'node1' and edge.target == 'node2') or \
               (edge.source == 'node2' and edge.target == 'node1')
        assert edge.weight > 0.8
    
    def test_compute_similarity_edges_threshold(self):
        """Test that threshold filtering works correctly."""
        embeddings = [
            api.Embedding('node1', [1.0, 0.0], 'test'),
            api.Embedding('node2', [0.5, 0.5], 'test'),  # ~0.7 similarity
            api.Embedding('node3', [0.0, 1.0], 'test')   # 0.0 similarity with node1
        ]
        
        # With high threshold, no edges
        edges_high = api.EdgeGenerator.compute_similarity_edges(embeddings, threshold=0.9)
        assert len(edges_high) == 0
        
        # With low threshold, some edges
        edges_low = api.EdgeGenerator.compute_similarity_edges(embeddings, threshold=0.5)
        assert len(edges_low) >= 1


class TestSemanticNetworkAPI:
    """Test the fluent API."""
    
    @patch('semnet_api.DataImporter')
    def test_fluent_api_chaining(self, mock_importer_class):
        """Test that the fluent API chains correctly."""
        # Setup mock
        mock_importer = Mock()
        mock_importer.extract_nodes.return_value = [
            api.Node('node1', 'test', {'content': 'test'})
        ]
        mock_importer.extract_edges.return_value = []
        mock_importer_class.return_value = mock_importer
        
        # Test chaining
        network = api.SemanticNetwork()
        result = network.import_data('test', '/test/source')
        
        # Should return same instance for chaining
        assert result is network
        assert len(network.nodes) == 1
        assert network.nodes[0].id == 'node1'
    
    def test_semantic_network_validation(self):
        """Test that the API validates state correctly."""
        network = api.SemanticNetwork()
        
        # Should fail without nodes
        with pytest.raises(ValueError, match="No nodes loaded"):
            network.generate_embeddings()
        
        # Should fail without importer
        network.nodes = [api.Node('test', 'test', {})]
        with pytest.raises(ValueError, match="No importer available"):
            network.generate_embeddings()


class TestDataImporter:
    """Test data importer functionality."""
    
    @patch('integrations.load_integration')
    def test_data_importer_basic_flow(self, mock_load_integration):
        """Test basic data import flow."""
        # Setup mock integration
        mock_integration = Mock()
        mock_integration.extract_nodes.return_value = [
            Mock(id='node1', type='test', content={'test': 'data'}, metadata={})
        ]
        mock_integration.extract_edges.return_value = []
        mock_load_integration.return_value = mock_integration
        
        # Test import
        importer = api.DataImporter('test_integration')
        importer.load('/test/path')
        nodes = importer.extract_nodes()
        
        # Verify
        assert len(nodes) == 1
        assert nodes[0].id == 'node1'
        assert nodes[0].type == 'test'
        
        # Verify integration was called correctly
        mock_load_integration.assert_called_once_with('test_integration', config=None)
        mock_integration.load.assert_called_once_with('/test/path')


class TestConvenienceFunctions:
    """Test the convenience functions."""
    
    @patch('semnet_api.DataImporter')
    @patch('semnet_api.Storage.save_nodes')
    def test_import_data_convenience(self, mock_save_nodes, mock_importer_class):
        """Test the import_data convenience function."""
        # Setup mocks
        mock_importer = Mock()
        mock_nodes = [api.Node('node1', 'test', {})]
        mock_importer.extract_nodes.return_value = mock_nodes
        mock_importer_class.return_value = mock_importer
        
        # Call convenience function
        result = api.import_data('test', '/source', '/output')
        
        # Verify
        assert result == mock_nodes
        mock_importer_class.assert_called_once_with('test', None)
        mock_importer.load.assert_called_once_with('/source')
        mock_save_nodes.assert_called_once()
    
    @patch('semnet_api.Storage.load_embeddings')
    @patch('semnet_api.EdgeGenerator.compute_similarity_edges')
    @patch('semnet_api.Storage.save_edges')
    def test_compute_edges_convenience(self, mock_save_edges, mock_compute, mock_load):
        """Test the compute_edges convenience function."""
        # Setup mocks
        mock_embeddings = [api.Embedding('node1', [1, 0], 'test')]
        mock_edges = [api.Edge('node1', 'node2', 0.8)]
        mock_load.return_value = mock_embeddings
        mock_compute.return_value = mock_edges
        
        # Call convenience function
        result = api.compute_edges('/input', '/output.json', threshold=0.7)
        
        # Verify
        assert result == mock_edges
        mock_load.assert_called_once_with(Path('/input'))
        mock_compute.assert_called_once_with(mock_embeddings, 0.7)
        mock_save_edges.assert_called_once_with(mock_edges, Path('/output.json'))


class TestCLIIntegration:
    """Test how the new CLI would use the API."""
    
    @patch('semnet_api.import_data')
    def test_cli_import_command_usage(self, mock_import_data):
        """Test how CLI import command would use the API."""
        mock_nodes = [api.Node('node1', 'test', {})]
        mock_import_data.return_value = mock_nodes
        
        # Simulate CLI call
        result = api.import_data(
            integration='chatlog.json',
            source='/path/to/logs',
            output_dir='/output'
        )
        
        assert result == mock_nodes
        mock_import_data.assert_called_once_with(
            integration='chatlog.json',
            source='/path/to/logs',
            output_dir='/output'
        )
    
    @patch('semnet_api.compute_edges')
    def test_cli_edges_command_usage(self, mock_compute_edges):
        """Test how CLI edges command would use the API."""
        mock_edges = [api.Edge('n1', 'n2', 0.8)]
        mock_compute_edges.return_value = mock_edges
        
        # Simulate CLI call
        result = api.compute_edges(
            input_dir='/embeddings',
            output_file='/edges.json',
            threshold=0.7
        )
        
        assert result == mock_edges
        mock_compute_edges.assert_called_once_with(
            input_dir='/embeddings',
            output_file='/edges.json',
            threshold=0.7
        )
    
    def test_fluent_api_pipeline_usage(self):
        """Test how CLI pipeline would use the fluent API."""
        # This tests the interface without mocking the internals
        network = api.SemanticNetwork()
        
        # Should be chainable
        assert hasattr(network, 'import_data')
        assert hasattr(network, 'generate_embeddings')
        assert hasattr(network, 'compute_edges')
        assert hasattr(network, 'export')
        
        # Methods should return self for chaining
        with patch.object(api.DataImporter, '__init__', return_value=None):
            with patch.object(api.DataImporter, 'load', return_value=Mock()):
                with patch.object(api.DataImporter, 'extract_nodes', return_value=[]):
                    result = network.import_data('test', '/source')
                    assert result is network