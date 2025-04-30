import os
import json
import glob
import networkx as nx

def export_to_network_format(
    nodes_dir,
    edges_file,
    output_format,
    output_file,
    keys_to_include=None
):
    """
    Export nodes and edges into a standard graph format for network analysis.
    Only includes specified keys from each node's JSON.

    Args:
        nodes_dir (str): Directory containing JSON files with node data.
        edges_file (str): Path to the JSON file containing edges.
        output_format (str): Format for the output graph file (e.g., "gexf", "graphml", "gml").
        output_file (str): Path to save the output graph file.
        keys_to_include (list or None): List of keys to include from each node's JSON. If None, use default.
    """
    if keys_to_include is None:
        keys_to_include = [
            "title", "conversation_id", "model", "created", "updated", "safe_urls", "openai_url"
        ]

    if nodes_dir is not None:
        if not os.path.exists(nodes_dir):
            print(f"Nodes directory {nodes_dir} does not exist.")
            return
        node_files = glob.glob(os.path.join(nodes_dir, "*.json"))
        nodes = {}
        for node_file in node_files:
            with open(node_file, "r", encoding="utf-8") as f:
                json_doc = json.load(f)
                node_id = os.path.basename(node_file).split('.')[0]
                # Only include specified keys, and replace None with ""
                filtered_doc = {k: (v if v is not None else "") for k, v in json_doc.items() if k in keys_to_include}
                nodes[node_id] = filtered_doc
    else:
        nodes = {}

    with open(edges_file, "r", encoding="utf-8") as f:
        edges = json.load(f)
    
    G = nx.Graph()
    for node_id, node_data in nodes.items():
        G.add_node(node_id, **node_data)
    skipped_edges = []
    for src, dst, weight in edges:
        if weight is not None:
            G.add_edge(src, dst, weight=weight)
        else:
            print(f"Skipped {len(skipped_edges)} edges with None weight. Example(s): {skipped_edges[:5]}")

    if output_format == "gexf":
        nx.write_gexf(G, output_file)
    elif output_format == "graphml":
        nx.write_graphml(G, output_file)
    elif output_format == "gml":
        nx.write_gml(G, output_file)
    else:
        print(f"Unsupported format: {output_format}")
        return
