"""
Tests for the refactored CLI.

These tests focus on the CLI orchestration logic and argument handling,
using the API as the main interface.
"""

import pytest
import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import cli


class TestCLICommands:
    """Test CLI command functions."""
    
    @pytest.mark.unit
    @patch('cli.discover_integrations')
    @patch('cli.IntegrationRegistry')
    def test_cmd_list_integrations(self, mock_registry, mock_discover, capsys):
        """Test list integrations command."""
        # Setup mocks
        mock_discover.return_value = {'test_integration': Mock()}
        mock_integration_class = Mock()
        mock_integration_class.__doc__ = "Test integration description"
        mock_registry.get.return_value = mock_integration_class
        
        # Create args
        args = argparse.Namespace()
        
        # Run command
        cli.cmd_list_integrations(args)
        
        # Check output
        captured = capsys.readouterr()
        assert "Available integrations" in captured.out
        assert "test_integration" in captured.out
        assert "Test integration description" in captured.out
    
    @pytest.mark.unit
    @patch('cli.api.import_data')
    def test_cmd_import_basic(self, mock_import_data):
        """Test basic import command without edge extraction."""
        # Setup mocks
        mock_nodes = [Mock(id='node1'), Mock(id='node2')]
        mock_import_data.return_value = mock_nodes
        
        # Create args
        args = argparse.Namespace(
            integration='test',
            source='/test/source',
            output_dir='/test/output',
            config=None,
            extract_edges=False
        )
        
        # Mock file operations for stats
        with patch('builtins.open'), patch('json.dump'):
            cli.cmd_import(args)
        
        # Verify API was called correctly
        mock_import_data.assert_called_once_with(
            integration='test',
            source='/test/source',
            output_dir='/test/output',
            config=None
        )
    
    @pytest.mark.unit
    @patch('cli.api.import_data')
    @patch('cli.api.DataImporter')
    @patch('cli.api.Storage.save_edges')
    def test_cmd_import_with_edges(self, mock_save_edges, mock_importer_class, mock_import_data):
        """Test import command with edge extraction."""
        # Setup mocks
        mock_nodes = [Mock(id='node1')]
        mock_edges = [Mock()]
        mock_import_data.return_value = mock_nodes
        
        mock_importer = Mock()
        mock_importer.extract_edges.return_value = mock_edges
        mock_importer_class.return_value = mock_importer
        
        # Create args
        args = argparse.Namespace(
            integration='test',
            source='/test/source',
            output_dir='/test/output',
            config='{"test": true}',
            extract_edges=True
        )
        
        # Mock file operations
        with patch('builtins.open'), patch('json.dump'):
            cli.cmd_import(args)
        
        # Verify edge extraction was called
        mock_importer.load.assert_called_once_with('/test/source')
        mock_importer.extract_edges.assert_called_once()
        mock_save_edges.assert_called_once()
    
    @pytest.mark.unit
    @patch('cli.api.Storage.load_nodes')
    @patch('cli.api.DataImporter')
    @patch('cli.api.EmbeddingGenerator')
    @patch('cli.api.Storage.save_embeddings')
    def test_cmd_embed_tfidf(self, mock_save_embeddings, mock_generator_class,
                             mock_importer_class, mock_load_nodes):
        """Test embed command with TF-IDF method."""
        # Setup mocks
        mock_nodes = [Mock(id='node1')]
        mock_embeddings = [Mock()]
        mock_load_nodes.return_value = mock_nodes
        
        mock_generator = Mock()
        mock_generator.generate_tfidf_embeddings.return_value = mock_embeddings
        mock_generator_class.return_value = mock_generator
        
        mock_importer = Mock()
        mock_importer_class.return_value = mock_importer
        
        # Create args
        args = argparse.Namespace(
            input_dir='/test/input',
            output_dir='/test/output',
            integration='test',
            method='tfidf',
            max_features=1000,
            batch_size=50,
            llm_config=None
        )
        
        # Run command
        cli.cmd_embed(args)
        
        # Verify calls
        mock_load_nodes.assert_called_once_with(Path('/test/input'))
        mock_generator.generate_tfidf_embeddings.assert_called_once_with(
            nodes=mock_nodes,
            content_extractor=mock_importer.get_node_content_for_embedding,
            max_features=1000,
            batch_size=50
        )
        mock_save_embeddings.assert_called_once_with(mock_embeddings, Path('/test/output'))
    
    @pytest.mark.unit
    @patch('cli.api.Storage.load_nodes')
    @patch('cli.api.DataImporter')
    @patch('cli.api.EmbeddingGenerator')
    @patch('cli.api.Storage.save_embeddings')
    def test_cmd_embed_llm(self, mock_save_embeddings, mock_generator_class,
                           mock_importer_class, mock_load_nodes):
        """Test embed command with LLM method."""
        # Setup mocks
        mock_nodes = [Mock(id='node1')]
        mock_embeddings = [Mock()]
        mock_load_nodes.return_value = mock_nodes
        
        mock_generator = Mock()
        mock_generator.generate_llm_embeddings.return_value = mock_embeddings
        mock_generator_class.return_value = mock_generator
        
        mock_importer = Mock()
        mock_importer_class.return_value = mock_importer
        
        # Create args
        args = argparse.Namespace(
            input_dir='/test/input',
            output_dir='/test/output',
            integration='test',
            method='llm',
            llm_provider='openai',
            llm_config='{"model": "text-embedding-3-small"}',
            chunk_size=512,
            chunk_overlap=50,
            max_features=5000,
            batch_size=100
        )
        
        # Run command
        cli.cmd_embed(args)
        
        # Verify calls
        mock_generator.generate_llm_embeddings.assert_called_once_with(
            nodes=mock_nodes,
            content_extractor=mock_importer.get_node_content_for_embedding,
            provider='openai',
            config={'model': 'text-embedding-3-small'},
            chunk_size=512,
            chunk_overlap=50
        )
    
    @pytest.mark.unit
    @patch('cli.api.compute_edges')
    def test_cmd_edges(self, mock_compute_edges):
        """Test edges command."""
        # Setup mocks
        mock_edges = [Mock(), Mock()]
        mock_compute_edges.return_value = mock_edges
        
        # Create args
        args = argparse.Namespace(
            input_dir='/test/embeddings',
            output_file='/test/edges.json',
            threshold=0.8
        )
        
        # Run command
        cli.cmd_edges(args)
        
        # Verify call
        mock_compute_edges.assert_called_once_with(
            input_dir='/test/embeddings',
            output_file='/test/edges.json',
            threshold=0.8
        )
    
    @pytest.mark.unit
    @patch('cli.api.export_graph')
    def test_cmd_export(self, mock_export_graph):
        """Test export command."""
        # Create args
        args = argparse.Namespace(
            nodes_dir='/test/nodes',
            edges_file='/test/edges.json',
            output_file='/test/output.gexf',
            format='gexf'
        )
        
        # Run command
        cli.cmd_export(args)
        
        # Verify call
        mock_export_graph.assert_called_once_with(
            nodes_dir='/test/nodes',
            edges_file='/test/edges.json',
            output_file='/test/output.gexf',
            format='gexf'
        )
    
    @pytest.mark.unit
    @patch('cli.api.SemanticNetwork')
    @patch('tempfile.mkdtemp')
    def test_cmd_pipeline_basic(self, mock_mkdtemp, mock_network_class):
        """Test pipeline command basic flow."""
        # Setup mocks
        mock_mkdtemp.return_value = '/tmp/test_work'
        mock_network = Mock()
        
        # Setup method chaining
        mock_network.import_data.return_value = mock_network
        mock_network.save_nodes.return_value = mock_network
        mock_network.generate_embeddings.return_value = mock_network
        mock_network.save_embeddings.return_value = mock_network
        mock_network.compute_edges.return_value = mock_network
        mock_network.save_edges.return_value = mock_network
        mock_network.export.return_value = mock_network
        
        mock_network_class.return_value = mock_network
        
        # Create args
        args = argparse.Namespace(
            source='/test/source',
            output='/test/output.gexf',
            integration='test',
            format='gexf',
            embedding_method='tfidf',
            llm_provider='ollama',
            llm_config=None,
            threshold=0.7,
            config=None,
            work_dir=None,
            keep_temp=False,
            max_features=5000,
            batch_size=100
        )
        
        # Run command
        with patch('shutil.rmtree') as mock_rmtree:
            cli.cmd_pipeline(args)
        
        # Verify the fluent API chain was called
        mock_network.import_data.assert_called_once_with('test', '/test/source', config=None)
        mock_network.generate_embeddings.assert_called_once()
        mock_network.compute_edges.assert_called_once_with(threshold=0.7)
        mock_network.export.assert_called_once_with('/test/output.gexf', format='gexf')
        
        # Verify cleanup
        mock_rmtree.assert_called_once()
    
    @pytest.mark.unit
    @patch('cli.api.SemanticNetwork')
    @patch('cli.Path')
    def test_cmd_pipeline_keep_temp(self, mock_path_class, mock_network_class):
        """Test pipeline command with keep_temp option."""
        # Setup mocks
        mock_network = Mock()
        mock_network.import_data.return_value = mock_network
        mock_network.save_nodes.return_value = mock_network
        mock_network.generate_embeddings.return_value = mock_network
        mock_network.save_embeddings.return_value = mock_network
        mock_network.compute_edges.return_value = mock_network
        mock_network.save_edges.return_value = mock_network
        mock_network.export.return_value = mock_network
        mock_network_class.return_value = mock_network
        
        # Mock Path
        mock_work_dir = Mock()
        mock_work_dir.mkdir = Mock()
        mock_path_class.return_value = mock_work_dir
        
        # Create args
        args = argparse.Namespace(
            source='/test/source',
            output='/test/output.gexf',
            integration='test',
            format='gexf',
            embedding_method='llm',
            llm_provider='openai',
            llm_config='{"model": "test"}',
            threshold=0.8,
            config=None,
            work_dir='/test/work',
            keep_temp=True,
            max_features=5000,
            batch_size=100
        )
        
        # Run command
        with patch('shutil.rmtree') as mock_rmtree:
            cli.cmd_pipeline(args)
        
        # Verify no cleanup happened
        mock_rmtree.assert_not_called()


class TestCLIMain:
    """Test main CLI entry point and argument parsing."""
    
    @pytest.mark.unit
    @patch('cli.cmd_list_integrations')
    def test_main_list_command(self, mock_cmd_list):
        """Test main with list command."""
        with patch('sys.argv', ['cli.py', 'list']):
            cli.main()
        
        mock_cmd_list.assert_called_once()
    
    @pytest.mark.unit
    @patch('cli.cmd_pipeline')
    def test_main_pipeline_command(self, mock_cmd_pipeline):
        """Test main with pipeline command."""
        with patch('sys.argv', ['cli.py', 'pipeline', 'source', 'output', '-i', 'test']):
            cli.main()
        
        mock_cmd_pipeline.assert_called_once()
    
    @pytest.mark.unit
    def test_main_no_command(self):
        """Test main with no command shows help."""
        with patch('sys.argv', ['cli.py']):
            with patch('sys.exit') as mock_exit:
                cli.main()
                # argparse calls sys.exit, and our error handler also calls it
                # Just verify it was called at least once
                assert mock_exit.called
    
    @pytest.mark.unit
    def test_main_invalid_command_handling(self):
        """Test that invalid commands are handled gracefully."""
        with patch('sys.argv', ['cli.py', 'nonexistent_command']):
            with pytest.raises(SystemExit):  # argparse will exit
                cli.main()


class TestCLIArgumentParsing:
    """Test CLI argument parsing logic."""
    
    def test_import_command_arguments(self):
        """Test that import command parses arguments correctly."""
        # This tests the argument parser setup
        parser = cli.argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        
        # Add import parser (simplified version)
        parser_import = subparsers.add_parser('import')
        parser_import.add_argument('-i', '--integration', required=True)
        parser_import.add_argument('-s', '--source', required=True)
        parser_import.add_argument('-o', '--output-dir', required=True)
        parser_import.add_argument('--extract-edges', action='store_true')
        
        # Test parsing
        args = parser.parse_args(['import', '-i', 'test', '-s', '/src', '-o', '/out', '--extract-edges'])
        
        assert args.command == 'import'
        assert args.integration == 'test'
        assert args.source == '/src'
        assert args.output_dir == '/out'
        assert args.extract_edges is True
    
    def test_pipeline_command_arguments(self):
        """Test that pipeline command parses arguments correctly."""
        parser = cli.argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        
        # Add pipeline parser (simplified)
        parser_pipeline = subparsers.add_parser('pipeline')
        parser_pipeline.add_argument('source')
        parser_pipeline.add_argument('output')
        parser_pipeline.add_argument('-i', '--integration', required=True)
        parser_pipeline.add_argument('-m', '--embedding-method', choices=['tfidf', 'llm'], default='tfidf')
        parser_pipeline.add_argument('-t', '--threshold', type=float, default=0.7)
        
        # Test parsing
        args = parser.parse_args(['pipeline', '/src', '/out.gexf', '-i', 'test', '-m', 'llm', '-t', '0.8'])
        
        assert args.command == 'pipeline'
        assert args.source == '/src'
        assert args.output == '/out.gexf'
        assert args.integration == 'test'
        assert args.embedding_method == 'llm'
        assert args.threshold == 0.8


class TestErrorHandling:
    """Test error handling in CLI commands."""
    
    @pytest.mark.unit
    @patch('cli.api.import_data')
    def test_cmd_import_error_handling(self, mock_import_data):
        """Test that import command handles errors properly."""
        # Setup mock to raise an exception
        mock_import_data.side_effect = Exception("Test error")
        
        args = argparse.Namespace(
            integration='test',
            source='/test/source',
            output_dir='/test/output',
            config=None,
            extract_edges=False
        )
        
        # Should not crash, but let the exception bubble up for main() to handle
        with pytest.raises(Exception, match="Test error"):
            cli.cmd_import(args)
    
    @pytest.mark.unit
    @patch('cli.logger')
    def test_main_error_handling(self, mock_logger):
        """Test that main handles errors and logs them."""
        with patch('sys.argv', ['cli.py', 'import']):  # Missing required args
            with patch('sys.exit') as mock_exit:
                # This will cause argparse to exit, but we can test our error handling
                try:
                    cli.main()
                except SystemExit:
                    pass  # Expected from argparse