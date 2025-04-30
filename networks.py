import networkx as nx
import json
import glob
import numpy as np
import os
from time import time
from tqdm import tqdm
from data.message_utils import messages_to_transcript
from embedding.utils import chunk_text
import datetime

def generate_node_embeddings_role_aggregate(
        input_dir,
        embedding_fn,
        role_config,
        embedding_key='embeddings',
        aggregation_fn=None,
        chunk_size=None,  # New: None means no chunking, otherwise chunk messages
        chunk_overlap=0,
        chunk_aggregation="mean"
    ):
    """
    Generalized role-based node embedding generator with optional chunking for long messages.
    """
    files = glob.glob(os.path.join(input_dir, "*"))
    iterator = tqdm(files, total=len(files), desc="General role-based node embeddings")

    for file in iterator:
        if not file.endswith('.json'):
            continue

        with open(file, 'r', encoding='utf-8') as f:
            json_doc = json.load(f)
            msgs = json_doc['messages']

            role_embs = {}
            per_role = {}
            for role, cfg in role_config.items():
                role_msgs = [m['content'].strip() for m in msgs if m.get('role') == role and m.get('content', '').strip()]
                msg_embs = []
                for msg in role_msgs:
                    if chunk_size is not None and len(msg) > chunk_size:
                        chunks = chunk_text(msg, chunk_size=chunk_size, overlap=chunk_overlap)
                        chunk_embs = [embedding_fn(chunk) for chunk in chunks if chunk.strip()]
                        if chunk_embs:
                            if chunk_aggregation == "mean":
                                msg_emb = np.mean(chunk_embs, axis=0)
                            elif chunk_aggregation == "sum":
                                msg_emb = np.sum(chunk_embs, axis=0)
                            else:
                                raise ValueError(f"Unknown chunk aggregation: {chunk_aggregation}")
                            msg_embs.append(msg_emb)
                    else:
                        msg_embs.append(embedding_fn(msg))
                agg = cfg.get("aggregate", "mean")
                if msg_embs:
                    if agg == "mean":
                        role_emb = np.mean(msg_embs, axis=0)
                    elif agg == "max":
                        role_emb = np.max(msg_embs, axis=0)
                    elif agg == "sum":
                        role_emb = np.sum(msg_embs, axis=0)
                    elif agg == "first":
                        role_emb = msg_embs[0]
                    elif agg == "last":
                        role_emb = msg_embs[-1]
                    elif callable(agg):
                        role_emb = agg(msg_embs)
                    else:
                        raise ValueError(f"Unknown aggregation: {agg}")
                    role_embs[role] = role_emb
                    per_role[role] = {
                        "vector": role_emb.tolist(),
                        "metadata": {
                            "role": role,
                            "aggregate": agg,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "chunk_aggregation": chunk_aggregation if chunk_size else None
                        }
                    }
                else:
                    role_embs[role] = None
                    per_role[role] = {
                        "vector": None,
                        "metadata": {"role": role, "aggregate": agg}
                    }

            # Combine role embeddings (unchanged)
            if aggregation_fn is None:
                weights = np.array([cfg["weight"] for role, cfg in role_config.items() if role_embs[role] is not None])
                embs = np.array([role_embs[role] for role, cfg in role_config.items() if role_embs[role] is not None])
                if len(embs) > 0:
                    emb = np.average(embs, axis=0, weights=weights)
                else:
                    emb = None
            else:
                emb = aggregation_fn(role_embs, role_config)

            # Store in nested structure with metadata
            if embedding_key not in json_doc:
                json_doc[embedding_key] = {}
            json_doc[embedding_key]["role_aggregate"] = {
                "vector": emb.tolist() if emb is not None else None,
                "metadata": {
                    "algorithm": embedding_fn.__name__,
                    "roles": role_config,
                    "aggregation": "weighted_mean" if aggregation_fn is None else aggregation_fn.__name__,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z"
                },
                "per_role": per_role
            }

        with open(file, 'w', encoding='utf-8') as f:
            json.dump(json_doc, f, ensure_ascii=False, indent=2)

def generate_node_embeddings_chunking(
        path,
        embedding_fn,
        embedding_root="embeddings",
        chunk_size=512,
        overlap=0,
        aggregation="mean",
        valid_roles=None):
    """
    Generate node embeddings by chunking the transcript and aggregating chunk embeddings.
    Only messages with roles in valid_roles are included.
    """
    file_paths = glob.glob(os.path.join(path, "*"))
    iterator = tqdm(file_paths, total=len(file_paths), desc="Generating chunked embeddings")
    for file_path in iterator:
        if not file_path.endswith('.json'):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_doc = json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {file_path}: {e}")
            continue

        msgs = json_doc['messages']
        transcript = messages_to_transcript(msgs, valid_roles=valid_roles)
        chunks = chunk_text(transcript, chunk_size=chunk_size, overlap=overlap)
        
        # see if
        chunk_embs = [embedding_fn(chunk) for chunk in chunks if chunk.strip()]
        if chunk_embs:
            if aggregation == "mean":
                emb = np.mean(chunk_embs, axis=0)
            elif aggregation == "sum":
                emb = np.sum(chunk_embs, axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        else:
            emb = None
        if embedding_root not in json_doc:
            json_doc[embedding_root] = {}
        json_doc[embedding_root]['chunked'] = {
            "vector": emb.tolist() if emb is not None else None,
            "metadata": {
                "algorithm": embedding_fn.__name__,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "aggregation": aggregation,
                "valid_roles": valid_roles,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z"
            }
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_doc, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing JSON to {file_path}: {e}")