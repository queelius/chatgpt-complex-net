from itertools import combinations
import os
import json
import glob
from tqdm import tqdm
import utils

def filter_edges(input_file, output_file, cutoff):
    """
    Filter edges below a similarity cutoff and print graph statistics before and after.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        edges = json.load(f)

    # Gather stats before filtering
    nodes_before = set()
    for src, dst, _ in edges:
        nodes_before.add(src)
        nodes_before.add(dst)
    num_edges_before = len(edges)
    num_nodes_before = len(nodes_before)
    max_possible_edges = num_nodes_before * (num_nodes_before - 1) // 2
    density_before = num_edges_before / max_possible_edges if max_possible_edges else 0

    print(f"Before filtering:")
    print(f"  Nodes: {num_nodes_before}")
    print(f"  Edges: {num_edges_before}")
    print(f"  Density: {density_before:.6f}")

    # Filter edges
    filtered_edges = [(src, dst, weight) for src, dst, weight in edges if weight >= cutoff]

    # Gather stats after filtering
    nodes_after = set()
    for src, dst, _ in filtered_edges:
        nodes_after.add(src)
        nodes_after.add(dst)
    num_edges_after = len(filtered_edges)
    num_nodes_after = len(nodes_after)
    max_possible_edges_after = num_nodes_after * (num_nodes_after - 1) // 2
    density_after = num_edges_after / max_possible_edges_after if max_possible_edges_after else 0

    print(f"After filtering (cutoff={cutoff}):")
    print(f"  Nodes: {num_nodes_after}")
    print(f"  Edges: {num_edges_after}")
    print(f"  Density: {density_after:.6f}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_edges, f, ensure_ascii=False, indent=2)


def generate_edges_from_node_embeddings(
        input_dir,
        output_file,
        sim=utils.cosine_similarity,
        embedding_root="embeddings",
        embedding_type="role_aggregate"):
    """
    Generate undirected graph edges from embeddings of nodes (documents).

    The function computes a similarity score for each pair of nodes (documents) and returns a list of edges.
    The output_file will be a JSON array of arrays, where each array encodes an edge as
    [node1_id, node2_id, similarity_score].

    Args:
        input_dir (str): Directory containing JSON files with node embeddings.
        output_file (str): Path to the output JSON file for edges.
        sim (callable): Similarity function to compute similarity score between two vectors.
        embedding_root (str): Root key in the JSON file where embeddings are stored.
        embedding_type (str): Type of embedding to use for similarity computation.

    Returns:
        None: The function saves the edges to a JSON file called `output_file`.
    """

    file_paths = glob.glob(os.path.join(input_dir, "*"))
    n = len(file_paths)
    
    iterator = tqdm(file_paths,
                    total=n,
                    unit="nodes",
                    desc="Generating node map")
    nodes = {}
    for i, file_path in enumerate(iterator, start=1):
        if not file_path.endswith('.json'):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            json_doc = json.load(f)
            node_id = os.path.basename(file_path).split('.')[0]
            nodes[node_id] = json_doc

    edges = []
    edge_iterator = tqdm(combinations(nodes.items(), 2),
                         total=(n * (n - 1)) // 2,
                         unit="edges",
                         desc="Generating edges")
    for (node1_id, node1), (node2_id, node2) in edge_iterator:

        n1_embedding = node1.get(embedding_root, {}).get(embedding_type, {}).get('vector')
        n2_embedding = node2.get(embedding_root, {}).get(embedding_type, {}).get('vector')
        # check if either is None and if so, print the
        # names of the nodes and skip
        if n1_embedding is None or n2_embedding is None:
            print(f"Skipping edge ({node1_id}, {node2_id}) due to missing embedding.")
            continue

        sim_score = sim(n1_embedding, n2_embedding)
        edges.append((node1_id, node2_id, sim_score))
        edge_iterator.set_postfix({"similarity": sim_score})
        edge_iterator.update(1)
        # Update progress bar with the current similarity score
        edge_iterator.refresh()

    # Save edges to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)
