import os
import glob
import json
import torch
from tqdm import tqdm

def generate_edges_from_node_embeddings_gpu(
    input_dir,
    output_file,
    embedding_root="embeddings",
    embedding_type="role_aggregate"
):
    """
    Generate undirected graph edges from node embeddings using GPU acceleration (PyTorch).
    Computes all pairwise cosine similarities and saves as edge list.
    """
    file_paths = sorted([f for f in glob.glob(os.path.join(input_dir, "*.json")) if f.endswith('.json')])
    node_ids = []
    vectors = []

    # Load all embeddings
    for file_path in tqdm(file_paths, desc="Loading embeddings"):
        with open(file_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        vec = doc.get(embedding_root, {}).get(embedding_type, {}).get('vector')
        if vec is not None:
            node_ids.append(os.path.basename(file_path).split('.')[0])
            vectors.append(vec)

    if not vectors:
        print("No valid embeddings found.")
        return

    # Convert to torch tensor on GPU
    X = torch.tensor(vectors, dtype=torch.float32, device="cuda")
    X = X / X.norm(dim=1, keepdim=True)  # Normalize

    # Compute cosine similarity matrix
    sim_matrix = torch.mm(X, X.t()).cpu().numpy()

    # Extract upper triangle (excluding diagonal)
    n = len(node_ids)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((node_ids[i], node_ids[j], float(sim_matrix[i, j])))

    # Save edges
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU-accelerated edge construction from node embeddings.")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory containing JSON documents with embeddings.")
    parser.add_argument("--output-file", "-o", required=True, help="Output JSON file for edges.")
    parser.add_argument("--embedding-root", type=str, default="embeddings", help="Root key for embeddings in JSON.")
    parser.add_argument("--embedding-type", type=str, default="role_aggregate", help="Embedding type under the root key.")
    args = parser.parse_args()

    generate_edges_from_node_embeddings_gpu(
        input_dir=args.input_dir,
        output_file=args.output_file,
        embedding_root=args.embedding_root,
        embedding_type=args.embedding_type
    )