#!/usr/bin/env python3
"""
Fix missing embeddings in imported data.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from embedding.tfidf_memory_efficient import MemoryEfficientTfidfVectorizer
from integrations.loader import load_integration
from integrations.base import Node

def fix_embeddings(input_dir):
    input_dir = Path(input_dir)
    
    # Load integration
    integration = load_integration('chatlog.json')
    
    # Collect all texts
    print("Loading all texts...")
    all_texts = []
    node_files = []
    
    for file_path in tqdm(list(input_dir.glob("*.json"))):
        if file_path.name in ['edges.json', 'import_stats.json']:
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            node_data = json.load(f)
        
        node = Node(
            id=node_data['id'],
            type=node_data['type'],
            content=node_data['content'],
            metadata=node_data.get('metadata', {})
        )
        
        text = integration.get_node_content_for_embedding(node)
        if not text:
            text = " "  # Use single space for empty text
        
        all_texts.append(text)
        node_files.append(file_path)
    
    print(f"Collected {len(all_texts)} texts")
    
    # Fit TF-IDF model
    print("Fitting TF-IDF model...")
    model = MemoryEfficientTfidfVectorizer(max_features=5000)
    model.fit(all_texts)
    
    # Generate and save embeddings
    print("Generating embeddings...")
    fixed_count = 0
    
    for file_path, text in tqdm(zip(node_files, all_texts), total=len(node_files)):
        with open(file_path, 'r', encoding='utf-8') as f:
            node_data = json.load(f)
        
        # Check if embedding needs fixing
        if node_data.get('embedding') is None:
            try:
                # Generate embedding
                embedding = model.transform([text])[0].tolist()
                node_data['embedding'] = embedding
                
                # Save updated file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(node_data, f, ensure_ascii=False, indent=2, default=str)
                
                fixed_count += 1
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
    
    print(f"Fixed {fixed_count} files with missing embeddings")

if __name__ == "__main__":
    fix_embeddings("dev/data/imported")