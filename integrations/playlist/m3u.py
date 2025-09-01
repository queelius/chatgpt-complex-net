"""
M3U playlist integration.
"""

import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Iterator, List, Optional, Dict, Any
from urllib.parse import urlparse, unquote

from integrations.base import DataSource, Node, Edge

class M3UPlaylistSource(DataSource):
    """
    Parse M3U/M3U8 playlist files.
    
    Supports extended M3U format with #EXTINF tags.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.playlists = []
        self.link_sequential = config.get('link_sequential', True) if config else True
        self.link_by_artist = config.get('link_by_artist', True) if config else True
        self.link_by_album = config.get('link_by_album', False) if config else False
    
    def load(self, source: Any) -> None:
        """Load M3U playlist files."""
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_file():
                file_paths = [source]
            elif source.is_dir():
                file_paths = list(source.glob('**/*.m3u')) + list(source.glob('**/*.m3u8'))
            else:
                raise ValueError(f"Path does not exist: {source}")
        elif isinstance(source, list):
            file_paths = [Path(p) for p in source]
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
        
        for file_path in file_paths:
            playlist = self._parse_m3u_file(file_path)
            if playlist:
                self.playlists.append(playlist)
    
    def _parse_m3u_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a single M3U file."""
        playlist = {
            'name': file_path.stem,
            'path': str(file_path),
            'tracks': [],
            'metadata': {}
        }
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        current_track = None
        track_index = 0
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('#EXTM3U'):
                continue
            
            elif line.startswith('#EXTINF:'):
                match = re.match(r'#EXTINF:(-?\d+),(.+?)(?:\s*-\s*(.+))?$', line)
                if match:
                    duration = int(match.group(1))
                    artist_or_full = match.group(2).strip()
                    title = match.group(3)
                    
                    if title:
                        artist = artist_or_full
                        title = title.strip()
                    else:
                        parts = artist_or_full.split(' - ', 1)
                        if len(parts) == 2:
                            artist, title = parts
                        else:
                            artist = None
                            title = artist_or_full
                    
                    current_track = {
                        'index': track_index,
                        'duration': duration,
                        'artist': artist,
                        'title': title,
                        'metadata': {}
                    }
            
            elif line.startswith('#EXT-X-'):
                if current_track:
                    key_value = line.split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0][7:]
                        current_track['metadata'][key] = key_value[1]
            
            elif line and not line.startswith('#'):
                if current_track:
                    current_track['path'] = line
                    
                    if line.startswith(('http://', 'https://')):
                        current_track['type'] = 'stream'
                        current_track['url'] = line
                    else:
                        current_track['type'] = 'file'
                        current_track['filename'] = Path(line).name
                        
                        filename_parts = Path(line).stem.split(' - ')
                        if len(filename_parts) >= 2 and not current_track['artist']:
                            current_track['artist'] = filename_parts[0].strip()
                            if not current_track['title']:
                                current_track['title'] = ' - '.join(filename_parts[1:]).strip()
                    
                    playlist['tracks'].append(current_track)
                    track_index += 1
                    current_track = None
                else:
                    simple_track = {
                        'index': track_index,
                        'path': line,
                        'title': Path(line).stem if not line.startswith('http') else line,
                        'type': 'stream' if line.startswith(('http://', 'https://')) else 'file'
                    }
                    playlist['tracks'].append(simple_track)
                    track_index += 1
        
        return playlist if playlist['tracks'] else None
    
    def extract_nodes(self) -> Iterator[Node]:
        """Convert playlists and tracks to nodes."""
        for playlist in self.playlists:
            playlist_id = hashlib.md5(playlist['path'].encode()).hexdigest()[:12]
            
            yield Node(
                id=f"playlist_{playlist_id}",
                type='playlist',
                content={
                    'name': playlist['name'],
                    'track_count': len(playlist['tracks']),
                    'total_duration': sum(t.get('duration', 0) for t in playlist['tracks'] if t.get('duration', 0) > 0)
                },
                metadata={
                    'path': playlist['path'],
                    'artists': list(set(t.get('artist') for t in playlist['tracks'] if t.get('artist')))
                },
                timestamp=datetime.fromtimestamp(Path(playlist['path']).stat().st_mtime)
            )
            
            for track in playlist['tracks']:
                track_text = f"{track.get('artist', '')} {track.get('title', track.get('path', ''))}"
                track_id = hashlib.md5(track_text.encode()).hexdigest()[:12]
                
                yield Node(
                    id=f"track_{track_id}",
                    type='track',
                    content={
                        'title': track.get('title', 'Unknown'),
                        'artist': track.get('artist'),
                        'duration': track.get('duration'),
                        'path': track.get('path')
                    },
                    metadata={
                        'playlist_id': playlist_id,
                        'playlist_name': playlist['name'],
                        'index': track['index'],
                        'type': track.get('type', 'file'),
                        **track.get('metadata', {})
                    }
                )
    
    def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
        """Create edges between tracks and playlists."""
        if not nodes:
            nodes = list(self.extract_nodes())
        
        playlist_nodes = {n.id: n for n in nodes if n.type == 'playlist'}
        track_nodes = [n for n in nodes if n.type == 'track']
        
        for track_node in track_nodes:
            playlist_id = f"playlist_{track_node.metadata.get('playlist_id')}"
            if playlist_id in playlist_nodes:
                yield Edge(
                    source_id=playlist_id,
                    target_id=track_node.id,
                    type='contains',
                    weight=1.0,
                    metadata={'index': track_node.metadata.get('index')}
                )
        
        if self.link_sequential:
            playlist_tracks = {}
            for track in track_nodes:
                pid = track.metadata.get('playlist_id')
                if pid not in playlist_tracks:
                    playlist_tracks[pid] = []
                playlist_tracks[pid].append(track)
            
            for pid, tracks in playlist_tracks.items():
                sorted_tracks = sorted(tracks, key=lambda t: t.metadata.get('index', 0))
                for i in range(len(sorted_tracks) - 1):
                    yield Edge(
                        source_id=sorted_tracks[i].id,
                        target_id=sorted_tracks[i+1].id,
                        type='next_track',
                        weight=0.9,
                        metadata={'playlist': pid}
                    )
        
        if self.link_by_artist:
            artist_groups = {}
            for track in track_nodes:
                artist = track.content.get('artist')
                if artist:
                    if artist not in artist_groups:
                        artist_groups[artist] = []
                    artist_groups[artist].append(track)
            
            for artist, artist_tracks in artist_groups.items():
                if len(artist_tracks) > 1:
                    for i, track1 in enumerate(artist_tracks):
                        for track2 in artist_tracks[i+1:]:
                            yield Edge(
                                source_id=track1.id,
                                target_id=track2.id,
                                type='same_artist',
                                weight=0.7,
                                metadata={'artist': artist}
                            )
    
    def get_node_content_for_embedding(self, node: Node) -> str:
        """Get text content for embedding generation."""
        if node.type == 'playlist':
            parts = [node.content.get('name', '')]
            if node.metadata.get('artists'):
                parts.extend(node.metadata['artists'][:10])
            return ' '.join(parts)
        
        elif node.type == 'track':
            parts = []
            if node.content.get('artist'):
                parts.append(node.content['artist'])
            if node.content.get('title'):
                parts.append(node.content['title'])
            if node.metadata.get('playlist_name'):
                parts.append(f"from {node.metadata['playlist_name']}")
            return ' '.join(parts)
        
        return ''