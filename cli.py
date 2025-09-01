#!/usr/bin/env python3
"""
Refactored CLI using the semantic network API.
This is a thin wrapper around the semnet_api module.
"""

import os
import sys
import argparse
import logging
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import semnet_api as api
from integrations import discover_integrations, IntegrationRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_list_integrations(args):
    """List all available integrations."""
    discovered = discover_integrations()
    print(f"\nAvailable integrations ({len(discovered)}):")
    for name in sorted(discovered):
        integration_class = IntegrationRegistry.get(name)
        if integration_class:
            doc = integration_class.__doc__.strip() if integration_class.__doc__ else 'No description'
            print(f"  - {name}: {doc}")


def cmd_import(args):
    """Import data using a specific integration."""
    config = json.loads(args.config) if args.config else None
    
    # Use the API to import data
    nodes = api.import_data(
        integration=args.integration,
        source=args.source,
        output_dir=args.output_dir,
        config=config
    )
    
    # Handle edge extraction if requested
    if args.extract_edges:
        importer = api.DataImporter(args.integration, config)
        importer.load(args.source)
        edges = importer.extract_edges()
        
        if edges:
            edges_file = Path(args.output_dir) / "edges.json"
            api.Storage.save_edges(edges, edges_file)
            logger.info(f"Saved {len(edges)} edges to {edges_file}")
    
    # Save statistics
    stats = {
        'node_count': len(nodes),
        'integration': args.integration,
        'source': args.source
    }
    if args.extract_edges:
        stats['edge_count'] = len(edges) if 'edges' in locals() else 0
    
    stats_file = Path(args.output_dir) / "import_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"Import complete. Saved {len(nodes)} nodes to {args.output_dir}")


def cmd_embed(args):
    """Generate embeddings for imported nodes."""
    # Parse configuration
    llm_config = json.loads(args.llm_config) if args.llm_config else {}
    
    # Load nodes
    nodes = api.Storage.load_nodes(Path(args.input_dir))
    
    # Create importer for content extraction
    importer = api.DataImporter(args.integration)
    
    # Generate embeddings using the API
    generator = api.EmbeddingGenerator(api.EmbeddingMethod(args.method))
    
    if args.method == 'tfidf':
        embeddings = generator.generate_tfidf_embeddings(
            nodes=nodes,
            content_extractor=importer.get_node_content_for_embedding,
            max_features=args.max_features,
            batch_size=args.batch_size
        )
    elif args.method == 'llm':
        embeddings = generator.generate_llm_embeddings(
            nodes=nodes,
            content_extractor=importer.get_node_content_for_embedding,
            provider=args.llm_provider,
            config=llm_config,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    else:
        raise ValueError(f"Unknown embedding method: {args.method}")
    
    # Save embeddings
    api.Storage.save_embeddings(embeddings, Path(args.output_dir))
    logger.info(f"Generated {len(embeddings)} embeddings")


def cmd_edges(args):
    """Generate edges based on embeddings."""
    # Use the API to compute edges
    edges = api.compute_edges(
        input_dir=args.input_dir,
        output_file=args.output_file,
        threshold=args.threshold
    )
    logger.info(f"Generated {len(edges)} edges")


def cmd_export(args):
    """Export the semantic network to various formats."""
    # Use the API to export
    api.export_graph(
        nodes_dir=args.nodes_dir,
        edges_file=args.edges_file,
        output_file=args.output_file,
        format=args.format
    )
    logger.info(f"Exported graph to {args.output_file}")


def cmd_pipeline(args):
    """Run complete pipeline using the fluent API."""
    # Parse configurations
    config = json.loads(args.config) if args.config else None
    llm_config = json.loads(args.llm_config) if args.llm_config else {}
    
    # Determine working directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="semnet_"))
        logger.info(f"Using temporary directory: {work_dir}")
    
    try:
        # Use the fluent API for the complete pipeline
        network = api.SemanticNetwork()
        
        # Build pipeline based on arguments
        pipeline_kwargs = {
            'max_features': args.max_features if args.embedding_method == 'tfidf' else None,
            'batch_size': args.batch_size if args.embedding_method == 'tfidf' else None,
            'provider': args.llm_provider if args.embedding_method == 'llm' else None,
            'config': llm_config if args.embedding_method == 'llm' else None,
        }
        # Remove None values
        pipeline_kwargs = {k: v for k, v in pipeline_kwargs.items() if v is not None}
        
        # Run pipeline
        network.import_data(args.integration, args.source, config=config) \
               .save_nodes(work_dir / 'nodes') \
               .generate_embeddings(method=args.embedding_method, **pipeline_kwargs) \
               .save_embeddings(work_dir / 'embeddings') \
               .compute_edges(threshold=args.threshold) \
               .save_edges(work_dir / 'edges.json') \
               .export(args.output, format=args.format)
        
        logger.info(f"Pipeline complete! Output: {args.output}")
        
    finally:
        # Cleanup if not keeping temp files
        if not args.keep_temp and not args.work_dir:
            logger.info(f"Cleaning up temporary directory: {work_dir}")
            shutil.rmtree(work_dir)
        elif args.keep_temp or args.work_dir:
            logger.info(f"Working directory preserved at: {work_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='semnet',
        description='Semantic Network Generator - Build knowledge graphs from various data sources'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    parser_list = subparsers.add_parser('list', help='List available integrations')
    
    # Import command
    parser_import = subparsers.add_parser('import', help='Import data from a source')
    parser_import.add_argument('-i', '--integration', required=True, help='Integration to use')
    parser_import.add_argument('-s', '--source', required=True, help='Source path/URL')
    parser_import.add_argument('-o', '--output-dir', required=True, help='Output directory for nodes')
    parser_import.add_argument('-c', '--config', help='JSON configuration for the integration')
    parser_import.add_argument('--extract-edges', action='store_true', 
                              help='Extract edges during import')
    
    # Embed command
    parser_embed = subparsers.add_parser('embed', help='Generate embeddings for nodes')
    parser_embed.add_argument('-i', '--input-dir', required=True, help='Input directory with nodes')
    parser_embed.add_argument('-o', '--output-dir', required=True, help='Output directory for embeddings')
    parser_embed.add_argument('--integration', required=True, help='Integration for content extraction')
    parser_embed.add_argument('-m', '--method', choices=['tfidf', 'llm'], default='tfidf',
                             help='Embedding method')
    
    # TF-IDF options
    parser_embed.add_argument('--max-features', type=int, default=5000,
                             help='Max features for TF-IDF')
    parser_embed.add_argument('--batch-size', type=int, default=100,
                             help='Batch size for TF-IDF')
    
    # LLM options
    parser_embed.add_argument('--llm-provider', default='ollama',
                             help='LLM provider (ollama, openai, anthropic, etc.)')
    parser_embed.add_argument('--llm-config', help='JSON configuration for LLM provider')
    parser_embed.add_argument('--chunk-size', type=int, default=0,
                             help='Chunk size for text splitting (0 = no chunking)')
    parser_embed.add_argument('--chunk-overlap', type=int, default=50,
                             help='Overlap between chunks')
    
    # Edges command
    parser_edges = subparsers.add_parser('edges', help='Generate edges from embeddings')
    parser_edges.add_argument('-i', '--input-dir', required=True, help='Input directory with embeddings')
    parser_edges.add_argument('-o', '--output-file', required=True, help='Output file for edges')
    parser_edges.add_argument('-t', '--threshold', type=float, default=0.7,
                              help='Similarity threshold for edges')
    
    # Export command
    parser_export = subparsers.add_parser('export', help='Export graph to various formats')
    parser_export.add_argument('-n', '--nodes-dir', required=True, help='Directory with nodes')
    parser_export.add_argument('-e', '--edges-file', required=True, help='File with edges')
    parser_export.add_argument('-o', '--output-file', required=True, help='Output file')
    parser_export.add_argument('-f', '--format', choices=['gexf', 'graphml', 'gml', 'json'],
                               default='gexf', help='Export format')
    
    # Pipeline command
    parser_pipeline = subparsers.add_parser('pipeline', 
                                           help='Run complete pipeline: import → embed → edges → export')
    parser_pipeline.add_argument('source', help='Source path/URL')
    parser_pipeline.add_argument('output', help='Output file')
    parser_pipeline.add_argument('-i', '--integration', required=True, help='Integration to use')
    parser_pipeline.add_argument('-f', '--format', choices=['gexf', 'graphml', 'gml', 'json'],
                                 default='gexf', help='Export format')
    parser_pipeline.add_argument('-m', '--embedding-method', choices=['tfidf', 'llm'],
                                 default='tfidf', help='Embedding method')
    parser_pipeline.add_argument('--llm-provider', default='ollama',
                                 help='LLM provider for embeddings')
    parser_pipeline.add_argument('--llm-config', help='JSON configuration for LLM provider')
    parser_pipeline.add_argument('-t', '--threshold', type=float, default=0.7,
                                 help='Similarity threshold for edges')
    parser_pipeline.add_argument('-c', '--config', help='JSON configuration for the integration')
    parser_pipeline.add_argument('--work-dir', help='Working directory (default: temp directory)')
    parser_pipeline.add_argument('--keep-temp', action='store_true',
                                 help='Keep temporary files')
    parser_pipeline.add_argument('--max-features', type=int, default=5000,
                                 help='Max features for TF-IDF')
    parser_pipeline.add_argument('--batch-size', type=int, default=100,
                                 help='Batch size for TF-IDF')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command
    commands = {
        'list': cmd_list_integrations,
        'import': cmd_import,
        'embed': cmd_embed,
        'edges': cmd_edges,
        'export': cmd_export,
        'pipeline': cmd_pipeline
    }
    
    try:
        commands[args.command](args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()