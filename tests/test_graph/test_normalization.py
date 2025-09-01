"""
Tests for graph normalization utilities.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open

from graph.normalization import (
    is_normalized_vectors_in_dict,
    normalize_vectors_in_dict,
    check_embeddings_normalized,
    normalize_node_embeddings
)


class TestIsNormalizedVectorsInDict:
    """Test is_normalized_vectors_in_dict function."""
    
    @pytest.mark.unit
    def test_normalized_vector(self):
        """Test detection of normalized vectors."""
        data = {
            "vector": [0.6, 0.8]  # Normalized (3-4-5 triangle)
        }
        assert is_normalized_vectors_in_dict(data) == True
    
    @pytest.mark.unit
    def test_unnormalized_vector(self):
        """Test detection of unnormalized vectors."""
        data = {
            "vector": [1.0, 1.0]  # Not normalized (norm = sqrt(2))
        }
        assert is_normalized_vectors_in_dict(data) == False
    
    @pytest.mark.unit
    def test_nested_structure_all_normalized(self):
        """Test nested dictionary with all normalized vectors."""
        data = {
            "embeddings": {
                "role1": {"vector": [1.0, 0.0, 0.0]},
                "role2": {"vector": [0.0, 1.0, 0.0]},
                "combined": {"vector": [0.6, 0.8, 0.0]}
            }
        }
        assert is_normalized_vectors_in_dict(data) == True
    
    @pytest.mark.unit
    def test_nested_structure_some_unnormalized(self):
        """Test nested dictionary with some unnormalized vectors."""
        data = {
            "embeddings": {
                "role1": {"vector": [1.0, 0.0]},  # Normalized
                "role2": {"vector": [1.0, 1.0]}   # Not normalized
            }
        }
        assert is_normalized_vectors_in_dict(data) == False
    
    @pytest.mark.unit
    def test_list_of_vectors(self):
        """Test list containing vector dictionaries."""
        data = [
            {"vector": [1.0, 0.0]},
            {"vector": [0.0, 1.0]},
            {"vector": [0.6, 0.8]}
        ]
        assert is_normalized_vectors_in_dict(data) == True
    
    @pytest.mark.unit
    def test_empty_structure(self):
        """Test empty dictionary."""
        assert is_normalized_vectors_in_dict({}) == True
        assert is_normalized_vectors_in_dict([]) == True
    
    @pytest.mark.unit
    def test_no_vectors(self):
        """Test structure without vector keys."""
        data = {
            "other_key": "value",
            "nested": {"still_no": "vectors"}
        }
        assert is_normalized_vectors_in_dict(data) == True
    
    @pytest.mark.unit
    def test_tolerance(self):
        """Test tolerance parameter for normalization check."""
        # Slightly off from normalized
        data = {"vector": [0.6001, 0.8]}  # Very close to normalized
        
        assert is_normalized_vectors_in_dict(data, tol=0.001) == True
        assert is_normalized_vectors_in_dict(data, tol=0.00001) == False


class TestNormalizeVectorsInDict:
    """Test normalize_vectors_in_dict function."""
    
    @pytest.mark.unit
    def test_normalize_single_vector(self):
        """Test normalizing a single vector."""
        data = {"vector": [3.0, 4.0]}
        normalize_vectors_in_dict(data)
        
        # Should be normalized to [0.6, 0.8]
        np.testing.assert_array_almost_equal(data["vector"], [0.6, 0.8])
    
    @pytest.mark.unit
    def test_normalize_nested_vectors(self):
        """Test normalizing nested vectors."""
        data = {
            "embeddings": {
                "role1": {"vector": [1.0, 1.0]},
                "role2": {"vector": [2.0, 0.0]}
            }
        }
        normalize_vectors_in_dict(data)
        
        # Check all vectors are normalized
        vec1 = np.array(data["embeddings"]["role1"]["vector"])
        vec2 = np.array(data["embeddings"]["role2"]["vector"])
        
        np.testing.assert_almost_equal(np.linalg.norm(vec1), 1.0)
        np.testing.assert_almost_equal(np.linalg.norm(vec2), 1.0)
    
    @pytest.mark.unit
    def test_normalize_zero_vector(self):
        """Test handling of zero vectors."""
        data = {"vector": [0.0, 0.0, 0.0]}
        normalize_vectors_in_dict(data)
        
        # Zero vector should remain zero
        assert data["vector"] == [0.0, 0.0, 0.0]
    
    @pytest.mark.unit
    def test_normalize_list_structure(self):
        """Test normalizing vectors in list structure."""
        data = [
            {"vector": [3.0, 4.0]},
            {"vector": [5.0, 12.0]}
        ]
        normalize_vectors_in_dict(data)
        
        np.testing.assert_array_almost_equal(data[0]["vector"], [0.6, 0.8])
        np.testing.assert_array_almost_equal(data[1]["vector"], [5/13, 12/13])
    
    @pytest.mark.unit
    def test_preserve_non_vector_data(self):
        """Test that non-vector data is preserved."""
        data = {
            "vector": [1.0, 1.0],
            "metadata": {"key": "value"},
            "score": 0.95
        }
        normalize_vectors_in_dict(data)
        
        # Check vector is normalized
        assert is_normalized_vectors_in_dict(data)
        # Check other data is preserved
        assert data["metadata"] == {"key": "value"}
        assert data["score"] == 0.95


class TestCheckEmbeddingsNormalized:
    """Test check_embeddings_normalized function."""
    
    @pytest.mark.unit
    def test_all_normalized(self, tmp_path, capsys):
        """Test checking when all embeddings are normalized."""
        # Create test files
        for i in range(2):
            data = {
                "embeddings": {
                    "vector": [1.0, 0.0, 0.0]
                }
            }
            with open(tmp_path / f"file_{i}.json", 'w') as f:
                json.dump(data, f)
        
        check_embeddings_normalized(str(tmp_path))
        
        captured = capsys.readouterr()
        assert "All embeddings are normalized" in captured.out
    
    @pytest.mark.unit
    def test_some_unnormalized(self, tmp_path, capsys):
        """Test checking when some embeddings are not normalized."""
        # Create normalized file
        data1 = {"embeddings": {"vector": [1.0, 0.0]}}
        with open(tmp_path / "normalized.json", 'w') as f:
            json.dump(data1, f)
        
        # Create unnormalized file
        data2 = {"embeddings": {"vector": [1.0, 1.0]}}
        with open(tmp_path / "unnormalized.json", 'w') as f:
            json.dump(data2, f)
        
        check_embeddings_normalized(str(tmp_path))
        
        captured = capsys.readouterr()
        assert "Not normalized: " in captured.out
        assert "unnormalized.json" in captured.out
        assert "Some embeddings are not normalized" in captured.out
    
    @pytest.mark.unit
    def test_missing_embeddings_key(self, tmp_path, capsys):
        """Test handling files without embeddings key."""
        data = {"other_data": "value"}
        with open(tmp_path / "no_embeddings.json", 'w') as f:
            json.dump(data, f)
        
        check_embeddings_normalized(str(tmp_path))
        
        captured = capsys.readouterr()
        assert "Skipping file" in captured.out
        assert "missing 'embeddings' key" in captured.out
    
    @pytest.mark.unit
    def test_custom_embedding_root(self, tmp_path, capsys):
        """Test with custom embedding root key."""
        data = {
            "custom_embeddings": {
                "vector": [0.6, 0.8]
            }
        }
        with open(tmp_path / "custom.json", 'w') as f:
            json.dump(data, f)
        
        check_embeddings_normalized(str(tmp_path), embedding_root="custom_embeddings")
        
        captured = capsys.readouterr()
        assert "All embeddings are normalized" in captured.out


class TestNormalizeNodeEmbeddings:
    """Test normalize_node_embeddings function."""
    
    @pytest.mark.unit
    def test_normalize_in_place(self, tmp_path):
        """Test normalizing embeddings in place."""
        # Create test file with unnormalized vectors
        data = {
            "embeddings": {
                "role1": {"vector": [3.0, 4.0]},
                "role2": {"vector": [5.0, 12.0]}
            }
        }
        input_file = tmp_path / "test.json"
        with open(input_file, 'w') as f:
            json.dump(data, f)
        
        # Normalize in place
        normalize_node_embeddings(str(tmp_path))
        
        # Read back and check
        with open(input_file, 'r') as f:
            result = json.load(f)
        
        vec1 = result["embeddings"]["role1"]["vector"]
        vec2 = result["embeddings"]["role2"]["vector"]
        
        np.testing.assert_array_almost_equal(vec1, [0.6, 0.8])
        np.testing.assert_array_almost_equal(vec2, [5/13, 12/13])
    
    @pytest.mark.unit
    def test_normalize_to_different_directory(self, tmp_path):
        """Test normalizing to a different output directory."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        # Create test file
        data = {
            "embeddings": {"vector": [1.0, 1.0, 1.0]}
        }
        with open(input_dir / "test.json", 'w') as f:
            json.dump(data, f)
        
        # Normalize to different directory
        normalize_node_embeddings(str(input_dir), str(output_dir))
        
        # Check output file exists and is normalized
        output_file = output_dir / "test.json"
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            result = json.load(f)
        
        vec = np.array(result["embeddings"]["vector"])
        np.testing.assert_almost_equal(np.linalg.norm(vec), 1.0)
    
    @pytest.mark.unit
    def test_skip_non_json_files(self, tmp_path):
        """Test that non-JSON files are skipped."""
        # Create various files
        (tmp_path / "test.json").write_text('{"embeddings": {"vector": [1,1]}}')
        (tmp_path / "readme.txt").write_text("Not JSON")
        (tmp_path / "data.csv").write_text("csv,data")
        
        # Should only process JSON files
        normalize_node_embeddings(str(tmp_path))
        
        # Only JSON file should be modified
        with open(tmp_path / "test.json", 'r') as f:
            data = json.load(f)
        
        assert is_normalized_vectors_in_dict(data["embeddings"])
        
        # Other files should be unchanged
        assert (tmp_path / "readme.txt").read_text() == "Not JSON"
        assert (tmp_path / "data.csv").read_text() == "csv,data"
    
    @pytest.mark.unit
    def test_preserve_other_data(self, tmp_path):
        """Test that non-embedding data is preserved."""
        data = {
            "id": "node_1",
            "metadata": {"key": "value"},
            "embeddings": {"vector": [2.0, 2.0]},
            "other": "data"
        }
        
        test_file = tmp_path / "test.json"
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        normalize_node_embeddings(str(tmp_path))
        
        with open(test_file, 'r') as f:
            result = json.load(f)
        
        # Check normalization
        assert is_normalized_vectors_in_dict(result["embeddings"])
        
        # Check other data preserved
        assert result["id"] == "node_1"
        assert result["metadata"] == {"key": "value"}
        assert result["other"] == "data"