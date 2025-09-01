"""
JSON chat log integration (compatible with existing format).
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Iterator, List, Optional, Dict, Any

from integrations.base import Node
from .base import ChatLogSource

class JSONChatSource(ChatLogSource):
    """
    Parse JSON-formatted chat logs.
    Compatible with the existing conversation format.
    """
    
    def load(self, source: Any) -> None:
        """Load JSON files from a path or list of paths."""
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_file():
                self.file_paths = [source]
            elif source.is_dir():
                self.file_paths = list(source.glob('**/*.json'))
            else:
                raise ValueError(f"Path does not exist: {source}")
        elif isinstance(source, list):
            self.file_paths = [Path(p) for p in source]
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
        
        self._load_all_files()
    
    def _load_all_files(self):
        """Load all JSON files."""
        for file_path in self.file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if 'messages' in item:
                            self.conversations.append(item)
                elif 'messages' in data:
                    self.conversations.append(data)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse {file_path}: {e}")
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
    
    def extract_nodes(self) -> Iterator[Node]:
        """Convert loaded conversations to nodes."""
        for conv in self.conversations:
            node_id = conv.get('id')
            if not node_id:
                content = json.dumps(conv['messages'], sort_keys=True)
                node_id = hashlib.md5(content.encode()).hexdigest()[:12]
            
            timestamp = None
            if 'timestamp' in conv:
                try:
                    timestamp = datetime.fromisoformat(conv['timestamp'])
                except:
                    pass
            
            if not timestamp:
                for msg in conv.get('messages', []):
                    if 'timestamp' in msg:
                        try:
                            timestamp = datetime.fromisoformat(msg['timestamp'])
                            break
                        except:
                            pass
            
            metadata = conv.get('metadata', {})
            metadata.update({
                'message_count': len(conv.get('messages', [])),
                'roles': list(set(m.get('role', 'user') for m in conv.get('messages', [])))
            })
            
            if 'embeddings' in conv:
                embedding = None
                emb_data = conv['embeddings']
                
                if isinstance(emb_data, dict):
                    for emb_type in ['role_aggregate', 'chunked', 'default']:
                        if emb_type in emb_data:
                            emb_info = emb_data[emb_type]
                            if isinstance(emb_info, dict) and 'vector' in emb_info:
                                embedding = emb_info['vector']
                                break
                            elif isinstance(emb_info, list):
                                embedding = emb_info
                                break
                elif isinstance(emb_data, list):
                    embedding = emb_data
            else:
                embedding = None
            
            yield Node(
                id=node_id,
                type='conversation',
                content={'messages': conv.get('messages', [])},
                metadata=metadata,
                timestamp=timestamp,
                embedding=embedding
            )