#!/usr/bin/env python3
import os
import sys
import json
import argparse
import networks
import ast
import numpy as np
import glob
from tqdm import tqdm
import logging
from graph.normalization import check_embeddings_normalized, normalize_node_embeddings
from graph.edge_utils import generate_edges_from_node_embeddings, filter_edges
from embedding.utils import chunk_text
from graph.export_utils import export_to_network_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def clean_embeddings(input_dir, embedding_root="embeddings", embedding_type=None):
    """
    Remove embeddings from JSON files.
    If embedding_type is None, remove the entire embedding_root tree.
    Otherwise, remove only the specified type under the root.
    """
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory does not exist or is not a directory: {input_dir}")
        sys.exit(1)
    file_paths = glob.glob(os.path.join(input_dir, "*.json"))
    for file_path in tqdm(file_paths, desc="Cleaning embeddings"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if embedding_root in data:
                if embedding_type:
                    if embedding_type in data[embedding_root]:
                        del data[embedding_root][embedding_type]
                        logging.info(f"Removed {embedding_type} from {file_path}")
                else:
                    del data[embedding_root]
                    logging.info(f"Removed all embeddings under '{embedding_root}' from {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to clean {file_path}: {e}")

def validate_json_files(input_dir):
    """
    Check all .json files in input_dir for JSON validity.
    Prints any files that are invalid.
    """
    import glob
    import json
    import os

    if not os.path.isdir(input_dir):
        print(f"Directory does not exist: {input_dir}")
        return

    file_paths = glob.glob(os.path.join(input_dir, "*.json"))
    invalid_files = []
    for file_path in tqdm(file_paths, desc="Validating JSON files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            print(f"Invalid JSON: {file_path} ({e})")
            invalid_files.append(file_path)
    if not invalid_files:
        print("All JSON files are valid.")
    else:
        print(f"\n{len(invalid_files)} invalid JSON file(s) found.")

def main():
    parser = argparse.ArgumentParser(
        description="Graph tool: generate complex networks based off en embedding model to generate edges based on some notion of semantic similarity. We also provide a means to export to commnon graph formats."
    )
    
    subparsers = parser.add_subparsers(
        dest="command", help="Subcommand to run")

    # Subcommand: Generate node embeddings.
    parser_emb = subparsers.add_parser("node-embeddings", help="Generate node embeddings from a directory of JSON documents")
    parser_emb.add_argument("--input-dir", "-i", required=True, help="Directory containing JSON documents.")
    parser_emb.add_argument("--method", "-m", choices=["chunked", "role-aggregate"], default="chunked", 
        help="Embedding method: chunked (transcript, with chunking) or role-aggregate (per-role, with optional chunking).")
    parser_emb.add_argument("--user-weight", "-u", type=float, default=1.5, 
        help="Weight for user messages when using role-aggregate method")
    parser_emb.add_argument("--assistant-weight", "-a", type=float, default=1.0, 
        help="Weight for assistant messages when using role-aggregate method")
    parser_emb.add_argument("--role-config", type=str, default=None,
        help="JSON string or path to JSON file specifying roles and weights, e.g. '{\"user\": {\"weight\": 1.5}, \"assistant\": {\"weight\": 1.0}}'")
    parser_emb.add_argument("--embedding-root", type=str, default="embeddings",
        help="Root key for embeddings in JSON (default: 'embeddings').")
    parser_emb.add_argument("--embedding-method", "-e", choices=["llm", "tfidf"], default="llm",
        help="Embedding method: 'llm' (default, uses LLM API) or 'tfidf' (classical TF-IDF).")
    parser_emb.add_argument("--chunk-size", type=int, default=2000,
        help="Chunk size (in words) for chunked methods. Use a large value for no chunking.")
    parser_emb.add_argument("--chunk-overlap", type=int, default=0,
        help="Number of words to overlap between chunks (default: 0).")
    parser_emb.add_argument("--chunk-aggregation", type=str, choices=["mean", "sum"], default="mean",
        help="How to aggregate chunk embeddings: mean or sum (default: mean).")
    parser_emb.add_argument("--valid-roles", type=str, default=None,
        help="Comma-separated list of roles to include (e.g. 'user,assistant'). Applies to chunked method.")

    # Subcommand: Normalize node embeddings.
    parser_norm = subparsers.add_parser("normalize", help="Normalize node embeddings to unit length.")
    parser_norm.add_argument("--input-dir", "-i", required=True, help="Directory containing JSON documents.")
    parser_norm.add_argument("--output-dir", "-o", required=False, default=None,
                          help="Directory to store the normalized embeddings in. If not specified, files are overwritten in place.")
    parser_norm.add_argument("--embedding-root", type=str, default="embeddings",
        help="Root key for embeddings in JSON (default: 'embeddings').")

    # Subcommand: Generate edges.
    parser_edges = subparsers.add_parser("edges", help="Generate graph edges from node embeddings.")
    parser_edges.add_argument("--input-dir", "-i", required=True, help="Directory containing the JSON documents.")
    parser_edges.add_argument("--output-file", "-o", required=True, help="JSON file to store the edges.")
    parser_edges.add_argument("--embedding-root", type=str, default="embeddings",
        help="Root key for embeddings in JSON (default: 'embeddings').")
    parser_edges.add_argument("--embedding-type", type=str, default="role_aggregate",
        help="Embedding type under the root key (default: 'role_aggregate').")

    # Subcommand: Generate edges using GPU.
    parser_edges_gpu = subparsers.add_parser("edges-gpu", help="Generate graph edges from node embeddings using GPU acceleration (PyTorch).")
    parser_edges_gpu.add_argument("--input-dir", "-i", required=True, help="Directory containing the JSON documents.")
    parser_edges_gpu.add_argument("--output-file", "-o", required=True, help="JSON file to store the edges.")
    parser_edges_gpu.add_argument("--embedding-root", type=str, default="embeddings",
        help="Root key for embeddings in JSON (default: 'embeddings').")
    parser_edges_gpu.add_argument("--embedding-type", type=str, default="role_aggregate",
        help="Embedding type under the root key (default: 'role_aggregate').")

    # Subcommand: Filter edges.
    parser_filter = subparsers.add_parser("cut-off", help="Remove edges based below the cut-off.")
    parser_filter.add_argument("--input-file", "-i", required=True, help="JSON file containing edges.")
    parser_filter.add_argument("--output-file", "-o", required=True, help="Output JSON file.")
    parser_filter.add_argument("--cutoff", "-c", type=float, required=True,
                               help="Edges with a similarity value below the cut-off will be removed.")
    
    # Subcommand: Rescale
    edge_rescale = subparsers.add_parser("edge-rescale", help="Rescale edge similarity scores to range [min, max].")
    edge_rescale.add_argument("--input", "-i", required=True, help="JSON file containing the edges.")
    edge_rescale.add_argument("--output", "-o", required=True, help="JSON file where the the filtered edges are stored.")
    edge_rescale.add_argument("--max", type=float, required=False, default=1.0,
                               help="Maximum similarity score.")
    edge_rescale.add_argument("--min", type=float, required=False, default=0.25,
                               help="Minimum similarity score.")

    # Subcommand: Export graph.
    parser_export = subparsers.add_parser("export", help="Export nodes and edges to a standard graph format for network analysis.")
    parser_export.add_argument("--nodes-dir", "-n", required=True, help="Directory containing JSON documents with node embeddings.")
    parser_export.add_argument("--edges-file", "-e", required=True, help="Input edges JSON file.")
    parser_export.add_argument("--format", "-f", required=True,
                               help="Output format (gexf for Gephi, graphml, or gml).", choices=["gexf", "graphml", "gml"])
    parser_export.add_argument("--output-file", "-o", required=True, help="Output file name.")

    # Subcommand: Check if embeddings are normalized.
    parser_isnorm = subparsers.add_parser("is-normalized", help="Check if all embedding vectors are unit length.")
    parser_isnorm.add_argument("--input-dir", "-i", required=True, help="Directory containing JSON documents.")
    parser_isnorm.add_argument("--tol", type=float, default=1e-4, help="Tolerance for norm check (default: 1e-4)")
    parser_isnorm.add_argument("--embedding-root", type=str, default="embeddings",
        help="Root key for embeddings in JSON (default: 'embeddings').")

    # Subcommand: Clean embeddings.
    parser_clean = subparsers.add_parser("clean-embeddings", help="Remove embeddings from JSON files.")
    parser_clean.add_argument("--input-dir", "-i", required=True, help="Directory containing JSON documents.")
    parser_clean.add_argument("--embedding-root", type=str, default="embeddings",
        help="Root key for embeddings in JSON (default: 'embeddings').")
    parser_clean.add_argument("--embedding-type", type=str, default=None,
        help="Embedding type under the root key to remove (default: remove all under root).")

    # Subcommand: Test chunking.
    parser_chunk = subparsers.add_parser("test-chunking", help="Test chunking of text.")
    parser_chunk.add_argument("--text", type=str, help="Text to chunk.")
    parser_chunk.add_argument("--file", type=str, help="File containing text to chunk.")
    parser_chunk.add_argument("--chunk-size", type=int, default=2000, help="Chunk size (in words).")
    parser_chunk.add_argument("--overlap", type=int, default=0, help="Number of words to overlap between chunks.")
    
    # Subcommand: Validate JSON files.
    parser_validate = subparsers.add_parser("validate-json", help="Check all JSON files in a directory for validity.")
    parser_validate.add_argument("--input-dir", "-i", required=True, help="Directory containing JSON files to validate.")

    args = parser.parse_args()
    
    if args.command == "node-embeddings":
        if not os.path.isdir(args.input_dir):
            logging.error(f"Input directory does not exist or is not a directory: {args.input_dir}")
            sys.exit(1)

        # Select embedding function
        if args.embedding_method == "llm":
            from embedding.llm_embedding_model import get_llm_embedding as embedding_fn
        elif args.embedding_method == "tfidf":
            from embedding.tdidf_embedding_model import TfidfVectorizer
            # Gather all docs for fitting
            file_paths = glob.glob(os.path.join(args.input_dir, "*.json"))
            docs = []
            for file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_doc = json.load(f)
                except Exception as e:
                    print(f"Error loading JSON from {file_path}: {e}")
                    continue
                msgs = json_doc['messages']
                # Use the same transcript logic as in chunked
                from networks import messages_to_transcript
                valid_roles = args.valid_roles.split(",") if args.valid_roles else None
                content = messages_to_transcript(msgs, valid_roles=valid_roles)
                docs.append(content)
            vectorizer = TfidfVectorizer()
            vectorizer.fit(docs)
            embedding_fn = vectorizer.get_tfidf_embedding
        else:
            logging.error("Unknown embedding method.")
            sys.exit(1)

        if args.method == "chunked":
            valid_roles = args.valid_roles.split(",") if args.valid_roles else None
            networks.generate_node_embeddings_chunking(
                path=args.input_dir,
                embedding_fn=embedding_fn,
                embedding_root=args.embedding_root,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
                aggregation=args.chunk_aggregation,
                valid_roles=valid_roles
            )
        elif args.method == "role-aggregate":
            # Determine role config
            if args.role_config:
                try:
                    if os.path.isfile(args.role_config):
                        with open(args.role_config, "r") as f:
                            role_config = json.load(f)
                    else:
                        role_config = json.loads(args.role_config)
                except Exception:
                    role_config = ast.literal_eval(args.role_config)
            else:
                role_config = {
                    "user": {"weight": args.user_weight, "aggregate": "mean"},
                    "assistant": {"weight": args.assistant_weight, "aggregate": "mean"},
                }
            networks.generate_node_embeddings_role_aggregate(
                input_dir=args.input_dir,
                embedding_fn=embedding_fn,
                role_config=role_config,
                embedding_key=args.embedding_root,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                chunk_aggregation=args.chunk_aggregation
            )

    elif args.command == "normalize":
        if not os.path.isdir(args.input_dir):
            logging.error(f"Input directory does not exist or is not a directory: {args.input_dir}")
            sys.exit(1)
        if args.output_dir is None or os.path.abspath(args.output_dir) == os.path.abspath(args.input_dir):
            print("WARNING: You are about to overwrite files in-place in", args.input_dir)
            confirm = input("Proceed with in-place normalization? [y/N]: ")
            if confirm.lower() != "y":
                print("Aborting normalization.")
                sys.exit(1)
        normalize_node_embeddings(args.input_dir, args.output_dir, embedding_root=args.embedding_root)

    elif args.command == "edges":
        if not os.path.isdir(args.input_dir):
            logging.error(f"Input directory does not exist or is not a directory: {args.input_dir}")
            sys.exit(1)
        generate_edges_from_node_embeddings(args.input_dir, args.output_file,
                                            embedding_root=args.embedding_root,
                                            embedding_type=args.embedding_type)

    elif args.command == "cut-off":
        filter_edges(args.input_file, args.output_file, args.cutoff)

    elif args.command == "export":
        export_to_network_format(args.nodes_dir, args.edges_file, args.format, args.output_file)

    elif args.command == "edge-rescale":
        with open(args.input, "r", encoding="utf-8") as f:
            edges = json.load(f)

        # Normalize edge scores to the specified range
        min_score = min(edge[2] for edge in edges)
        max_score = max(edge[2] for edge in edges)
        range_score = max_score - min_score
        rescaled_edges = [
            (edge[0], edge[1], ((edge[2] - min_score) / range_score) * (args.max - args.min) + args.min)
            for edge in edges
        ]

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rescaled_edges, f, ensure_ascii=False, indent=2)
            
    elif args.command == "is-normalized":
        if not os.path.isdir(args.input_dir):
            logging.error(f"Input directory does not exist or is not a directory: {args.input_dir}")
            sys.exit(1)
        check_embeddings_normalized(args.input_dir, args.tol, embedding_root=args.embedding_root)

    elif args.command == "clean-embeddings":
        if not os.path.isdir(args.input_dir):
            logging.error(f"Input directory does not exist or is not a directory: {args.input_dir}")
            sys.exit(1)
        clean_embeddings(args.input_dir, args.embedding_root, args.embedding_type)

    elif args.command == "test-chunking":
        if args.text:
            text = args.text
        elif args.file:
            # Check if it's a JSON file
            if args.file.endswith(".json"):
                with open(args.file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Import the transcript utility
                from networks import messages_to_transcript
                msgs = data.get("messages", [])
                text = messages_to_transcript(msgs)
            else:
                with open(args.file, "r", encoding="utf-8") as f:
                    text = f.read()
        else:
            print("Please provide --text or --file.")
            sys.exit(1)
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---\n`{chunk}`\n")
        
    elif args.command == "validate-json":
        validate_json_files(args.input_dir)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
