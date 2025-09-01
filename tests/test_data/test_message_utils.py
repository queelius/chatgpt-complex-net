"""
Tests for message utility functions.
"""

import pytest
from data.message_utils import messages_to_transcript


class TestMessagesToTranscript:
    """Test messages_to_transcript function."""
    
    @pytest.mark.unit
    def test_basic_conversation(self):
        """Test converting basic conversation to transcript."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        transcript = messages_to_transcript(messages)
        expected = "User: Hello\nAssistant: Hi there!"
        
        assert transcript == expected
    
    @pytest.mark.unit
    def test_with_valid_roles_filter(self):
        """Test filtering messages by valid roles."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # Only include user and assistant messages
        transcript = messages_to_transcript(messages, valid_roles=["user", "assistant"])
        expected = "User: Hello\nAssistant: Hi there!"
        
        assert transcript == expected
    
    @pytest.mark.unit
    def test_empty_messages(self):
        """Test with empty message list."""
        messages = []
        transcript = messages_to_transcript(messages)
        assert transcript == ""
    
    @pytest.mark.unit
    def test_messages_with_empty_content(self):
        """Test handling messages with empty content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},  # Empty content
            {"role": "user", "content": "Are you there?"}
        ]
        
        transcript = messages_to_transcript(messages)
        expected = "User: Hello\nUser: Are you there?"
        
        assert transcript == expected
    
    @pytest.mark.unit
    def test_messages_with_whitespace_content(self):
        """Test handling messages with only whitespace."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "   \n\t  "},  # Only whitespace
            {"role": "user", "content": "Hello?"}
        ]
        
        transcript = messages_to_transcript(messages)
        expected = "User: Hello\nUser: Hello?"
        
        assert transcript == expected
    
    @pytest.mark.unit
    def test_missing_role_field(self):
        """Test handling messages without role field."""
        messages = [
            {"content": "No role"},  # Missing role
            {"role": "user", "content": "Hello"}
        ]
        
        transcript = messages_to_transcript(messages)
        expected = "Unknown: No role\nUser: Hello"
        
        assert transcript == expected
    
    @pytest.mark.unit
    def test_missing_content_field(self):
        """Test handling messages without content field."""
        messages = [
            {"role": "user"},  # Missing content
            {"role": "assistant", "content": "Hi!"}
        ]
        
        transcript = messages_to_transcript(messages)
        expected = "Assistant: Hi!"
        
        assert transcript == expected
    
    @pytest.mark.unit
    def test_role_capitalization(self):
        """Test that roles are properly capitalized."""
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "ASSISTANT", "content": "Response"},
            {"role": "System", "content": "Info"}
        ]
        
        transcript = messages_to_transcript(messages)
        
        # Check that roles are capitalized
        assert "User:" in transcript
        assert "Assistant:" in transcript
        assert "System:" in transcript
    
    @pytest.mark.unit
    def test_multiline_content(self):
        """Test handling multiline content."""
        messages = [
            {"role": "user", "content": "Line 1\nLine 2\nLine 3"},
            {"role": "assistant", "content": "Response"}
        ]
        
        transcript = messages_to_transcript(messages)
        expected = "User: Line 1\nLine 2\nLine 3\nAssistant: Response"
        
        assert transcript == expected
    
    @pytest.mark.unit
    def test_valid_roles_none(self):
        """Test that valid_roles=None includes all messages."""
        messages = [
            {"role": "user", "content": "User message"},
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "custom", "content": "Custom role"}
        ]
        
        transcript = messages_to_transcript(messages, valid_roles=None)
        
        # All messages should be included
        assert "User: User message" in transcript
        assert "System: System message" in transcript
        assert "Assistant: Assistant message" in transcript
        assert "Custom: Custom role" in transcript
    
    @pytest.mark.unit
    def test_valid_roles_empty_list(self):
        """Test that empty valid_roles list excludes all messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        transcript = messages_to_transcript(messages, valid_roles=[])
        assert transcript == ""