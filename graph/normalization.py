import os
import json
import glob
import numpy as np
from tqdm import tqdm

def is_normalized_vectors_in_dict(d, tol=1e-4):
    """
    Recursively check if all 'vector' keys in a nested dictionary are unit length.
    Returns True if all are normalized, False otherwise.
    """
    if isinstance(d, dict):
        for key, value in d.items():
            if key == "vector" and isinstance(value, list):
                arr = np.array(value)
                norm = np.linalg.norm(arr)
                if not np.isclose(norm, 1.0, atol=tol):
                    return False
            else:
                if not is_normalized_vectors_in_dict(value, tol):
                    return False
    elif isinstance(d, list):
        for item in d:
            if not is_normalized_vectors_in_dict(item, tol):
                return False
    return True

def check_embeddings_normalized(input_dir, tol=1e-4, embedding_root="embeddings"):
    """
    Check if all vectors in the 'embeddings' dictionary of each JSON file are normalized.
    """
    file_paths = glob.glob(os.path.join(input_dir, "*"))
    iterator = tqdm(file_paths, total=len(file_paths), desc="Checking normalization")
    all_normalized = True
    for file_path in iterator:
        if not file_path.endswith('.json'):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            json_doc = json.load(f)
            embeddings = json_doc.get(embedding_root)
            if embeddings is None:
                print(f"Skipping file {file_path} due to missing '{embedding_root}' key.")
                continue
            if not is_normalized_vectors_in_dict(embeddings, tol):
                print(f"Not normalized: {file_path}")
                all_normalized = False
    if all_normalized:
        print("All embeddings are normalized.")
    else:
        print("Some embeddings are not normalized.")

def normalize_vectors_in_dict(d):
    """
    Recursively normalize all 'vector' keys in a nested dictionary.
    """
    if isinstance(d, dict):
        for key, value in d.items():
            if key == "vector" and isinstance(value, list):
                arr = np.array(value)
                norm = np.linalg.norm(arr)
                if norm != 0:
                    d[key] = (arr / norm).tolist()
            else:
                normalize_vectors_in_dict(value)
    elif isinstance(d, list):
        for item in d:
            normalize_vectors_in_dict(item)

def normalize_node_embeddings(input_dir, output_dir=None, embedding_root="embeddings"):
    """
    Normalize all vectors in the 'embeddings' dictionary of each JSON file,
    regardless of the structure or roles present.
    """
    file_paths = glob.glob(os.path.join(input_dir, "*"))
    iterator = tqdm(file_paths, total=len(file_paths), desc="Normalizing embeddings")
    for file_path in iterator:
        if not file_path.endswith('.json'):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            json_doc = json.load(f)
            embeddings = json_doc.get(embedding_root)
            if embeddings is None:
                print(f"File {file_path} is missing '{embedding_root}'.")
            else:
                normalize_vectors_in_dict(embeddings)

        # Determine output path
        if output_dir is None or os.path.abspath(output_dir) == os.path.abspath(input_dir):
            output_path = file_path
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(file_path))

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_doc, f, ensure_ascii=False, indent=2)

