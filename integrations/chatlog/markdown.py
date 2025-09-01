"""
Markdown chat log integration.
"""

import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Iterator, List, Optional, Dict, Any

from integrations.base import Node
from .base import ChatLogSource

class MarkdownChatSource(ChatLogSource):
    """
    Parse markdown-formatted chat logs.
    
    Expected format:
    ## User
    User message content...
    
    ## Assistant
    Assistant response...
    
    Or with timestamps:
    ## User [2024-01-15 10:30:00]
    Message content...
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.file_paths = []
        self.role_patterns = config.get('role_patterns', {
            'user': r'^#{1,3}\s*(User|Human|You)',
            'assistant': r'^#{1,3}\s*(Assistant|AI|Claude|GPT|Bot)',
            'system': r'^#{1,3}\s*(System|Note)'
        }) if config else {
            'user': r'^#{1,3}\s*(User|Human|You)',
            'assistant': r'^#{1,3}\s*(Assistant|AI|Claude|GPT|Bot)',
            'system': r'^#{1,3}\s*(System|Note)'
        }
        self.timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\]'
    
    def load(self, source: Any) -> None:
        """Load markdown files from a path or list of paths."""
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_file():
                self.file_paths = [source]
            elif source.is_dir():
                self.file_paths = list(source.glob('**/*.md'))
            else:
                raise ValueError(f"Path does not exist: {source}")
        elif isinstance(source, list):
            self.file_paths = [Path(p) for p in source]
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
        
        self._parse_all_files()
    
    def _parse_all_files(self):
        """Parse all loaded markdown files."""
        for file_path in self.file_paths:
            conversations = self._parse_markdown_file(file_path)
            self.conversations.extend(conversations)
    
    def _parse_markdown_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a single markdown file into conversations."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        conversations = []
        current_conv = {'messages': [], 'metadata': {'source': str(file_path)}}
        current_message = None
        conversation_boundary = self.config.get('conversation_boundary', '---')
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            if line.strip() == conversation_boundary and current_conv['messages']:
                conversations.append(current_conv)
                current_conv = {'messages': [], 'metadata': {'source': str(file_path)}}
                current_message = None
                i += 1
                continue
            
            role = self._detect_role(line)
            if role:
                if current_message:
                    current_conv['messages'].append(current_message)
                
                timestamp = self._extract_timestamp(line)
                current_message = {
                    'role': role,
                    'content': '',
                    'timestamp': timestamp
                }
                i += 1
                continue
            
            if current_message and line.strip():
                if current_message['content']:
                    current_message['content'] += '\n'
                current_message['content'] += line
            
            i += 1
        
        if current_message:
            current_conv['messages'].append(current_message)
        
        if current_conv['messages']:
            conversations.append(current_conv)
        
        return conversations
    
    def _detect_role(self, line: str) -> Optional[str]:
        """Detect the role from a markdown header line."""
        for role, pattern in self.role_patterns.items():
            if re.match(pattern, line, re.IGNORECASE):
                return role
        return None
    
    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from a line."""
        match = re.search(self.timestamp_pattern, line)
        if match:
            timestamp_str = match.group(1)
            try:
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                except ValueError:
                    pass
        return None
    
    def extract_nodes(self) -> Iterator[Node]:
        """Convert parsed conversations to nodes."""
        for conv in self.conversations:
            conv_id = self._generate_conversation_id(conv)
            
            first_msg_timestamp = None
            for msg in conv['messages']:
                if msg.get('timestamp'):
                    first_msg_timestamp = msg['timestamp']
                    break
            
            topics = self._extract_topics(conv)
            
            yield Node(
                id=conv_id,
                type='conversation',
                content=conv,
                metadata={
                    'source': conv['metadata'].get('source'),
                    'message_count': len(conv['messages']),
                    'topics': topics,
                    'roles': list(set(m['role'] for m in conv['messages']))
                },
                timestamp=first_msg_timestamp
            )
    
    def _generate_conversation_id(self, conv: Dict[str, Any]) -> str:
        """Generate a unique ID for a conversation."""
        content = json.dumps(conv['messages'], sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _extract_topics(self, conv: Dict[str, Any]) -> List[str]:
        """Extract topics/keywords from a conversation."""
        from collections import Counter
        import re
        
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                        'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 
                        'was', 'are', 'were', 'been', 'be', 'have', 'has', 
                        'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                        'should', 'may', 'might', 'must', 'can', 'could',
                        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
                        'which', 'who', 'when', 'where', 'why', 'how'])
        
        all_text = ' '.join(m['content'] for m in conv['messages'])
        words = re.findall(r'\b[a-z]+\b', all_text.lower())
        
        word_counts = Counter(w for w in words if w not in stopwords and len(w) > 3)
        
        return [word for word, _ in word_counts.most_common(10)]