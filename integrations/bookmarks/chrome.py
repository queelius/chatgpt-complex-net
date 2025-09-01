"""
Chrome/Chromium bookmarks integration.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Iterator, List, Optional, Dict, Any
from urllib.parse import urlparse

from integrations.base import DataSource, Node, Edge

class ChromeBookmarkSource(DataSource):
    """
    Parse Chrome/Chromium bookmarks JSON file.
    
    Chrome bookmarks are typically located at:
    - macOS: ~/Library/Application Support/Google/Chrome/Default/Bookmarks
    - Linux: ~/.config/google-chrome/Default/Bookmarks
    - Windows: %LOCALAPPDATA%\\Google\\Chrome\\User Data\\Default\\Bookmarks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.bookmarks = []
        self.folders = []
        self.link_by_domain = config.get('link_by_domain', True) if config else True
        self.link_by_folder = config.get('link_by_folder', True) if config else True
        self.extract_tags = config.get('extract_tags', True) if config else True
    
    def load(self, source: Any) -> None:
        """Load Chrome bookmarks from JSON file."""
        if isinstance(source, (str, Path)):
            source = Path(source)
            if not source.exists():
                raise FileNotFoundError(f"Bookmarks file not found: {source}")
            
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._parse_bookmarks(data['roots'])
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
    
    def _parse_bookmarks(self, roots: Dict[str, Any], parent_folder: str = None):
        """Recursively parse bookmark tree."""
        for key, node in roots.items():
            if isinstance(node, dict):
                self._parse_node(node, parent_folder)
    
    def _parse_node(self, node: Dict[str, Any], parent_folder: str = None):
        """Parse a single bookmark node."""
        node_type = node.get('type')
        
        if node_type == 'folder':
            folder_name = node.get('name', 'Unknown')
            folder_path = f"{parent_folder}/{folder_name}" if parent_folder else folder_name
            
            self.folders.append({
                'id': node.get('id'),
                'name': folder_name,
                'path': folder_path,
                'date_added': self._chrome_timestamp_to_datetime(node.get('date_added')),
                'date_modified': self._chrome_timestamp_to_datetime(node.get('date_modified'))
            })
            
            if 'children' in node:
                for child in node['children']:
                    self._parse_node(child, folder_path)
        
        elif node_type == 'url':
            bookmark = {
                'id': node.get('id'),
                'name': node.get('name', ''),
                'url': node.get('url', ''),
                'folder': parent_folder,
                'date_added': self._chrome_timestamp_to_datetime(node.get('date_added')),
                'guid': node.get('guid'),
                'meta_info': node.get('meta_info', {})
            }
            
            if bookmark['url']:
                try:
                    parsed = urlparse(bookmark['url'])
                    bookmark['domain'] = parsed.netloc
                    bookmark['scheme'] = parsed.scheme
                except:
                    bookmark['domain'] = None
                    bookmark['scheme'] = None
            
            self.bookmarks.append(bookmark)
    
    def _chrome_timestamp_to_datetime(self, timestamp: Optional[str]) -> Optional[datetime]:
        """Convert Chrome timestamp (microseconds since 1601) to datetime."""
        if not timestamp:
            return None
        
        try:
            timestamp = int(timestamp)
            epoch_delta = 11644473600
            return datetime.fromtimestamp((timestamp / 1000000) - epoch_delta)
        except:
            return None
    
    def extract_nodes(self) -> Iterator[Node]:
        """Convert bookmarks to nodes."""
        for bookmark in self.bookmarks:
            tags = []
            
            if self.extract_tags:
                if bookmark['folder']:
                    tags.extend(bookmark['folder'].split('/'))
                
                if bookmark['domain']:
                    domain_parts = bookmark['domain'].split('.')
                    tags.extend([p for p in domain_parts if len(p) > 2])
            
            yield Node(
                id=f"bookmark_{bookmark['id']}",
                type='bookmark',
                content={
                    'title': bookmark['name'],
                    'url': bookmark['url'],
                    'description': bookmark.get('meta_info', {}).get('description', '')
                },
                metadata={
                    'folder': bookmark['folder'],
                    'domain': bookmark['domain'],
                    'tags': tags,
                    'guid': bookmark['guid']
                },
                timestamp=bookmark['date_added']
            )
        
        for folder in self.folders:
            yield Node(
                id=f"folder_{folder['id']}",
                type='bookmark_folder',
                content={
                    'name': folder['name'],
                    'path': folder['path']
                },
                metadata={},
                timestamp=folder['date_added']
            )
    
    def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
        """Create edges between bookmarks."""
        if not nodes:
            nodes = list(self.extract_nodes())
        
        bookmark_nodes = [n for n in nodes if n.type == 'bookmark']
        folder_nodes = [n for n in nodes if n.type == 'bookmark_folder']
        
        if self.link_by_domain:
            domain_groups = {}
            for node in bookmark_nodes:
                domain = node.metadata.get('domain')
                if domain:
                    if domain not in domain_groups:
                        domain_groups[domain] = []
                    domain_groups[domain].append(node)
            
            for domain, group_nodes in domain_groups.items():
                for i, node1 in enumerate(group_nodes):
                    for node2 in group_nodes[i+1:]:
                        yield Edge(
                            source_id=node1.id,
                            target_id=node2.id,
                            type='same_domain',
                            weight=0.8,
                            metadata={'domain': domain}
                        )
        
        if self.link_by_folder:
            folder_groups = {}
            for node in bookmark_nodes:
                folder = node.metadata.get('folder')
                if folder:
                    if folder not in folder_groups:
                        folder_groups[folder] = []
                    folder_groups[folder].append(node)
            
            for folder, group_nodes in folder_groups.items():
                folder_node = next((n for n in folder_nodes if n.content['path'] == folder), None)
                
                if folder_node:
                    for bookmark_node in group_nodes:
                        yield Edge(
                            source_id=folder_node.id,
                            target_id=bookmark_node.id,
                            type='contains',
                            weight=1.0,
                            metadata={'relationship': 'folder_contains_bookmark'}
                        )
                
                for i, node1 in enumerate(group_nodes):
                    for node2 in group_nodes[i+1:]:
                        yield Edge(
                            source_id=node1.id,
                            target_id=node2.id,
                            type='same_folder',
                            weight=0.6,
                            metadata={'folder': folder}
                        )
    
    def get_node_content_for_embedding(self, node: Node) -> str:
        """Get text content for embedding generation."""
        if node.type == 'bookmark':
            parts = []
            
            if node.content.get('title'):
                parts.append(node.content['title'])
            
            if node.content.get('description'):
                parts.append(node.content['description'])
            
            if node.metadata.get('tags'):
                parts.append(' '.join(node.metadata['tags']))
            
            if node.content.get('url'):
                parsed = urlparse(node.content['url'])
                path_parts = [p for p in parsed.path.split('/') if p]
                if path_parts:
                    parts.append(' '.join(path_parts))
            
            return ' '.join(parts)
        
        elif node.type == 'bookmark_folder':
            return node.content.get('path', node.content.get('name', ''))
        
        return ''