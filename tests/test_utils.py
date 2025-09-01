"""
Tests for utility functions.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock
import json
from pathlib import Path
import os

import utils


class TestMakeFilenameUnique:
    """Test make_filename_unique function."""
    
    @pytest.mark.unit
    def test_make_filename_unique_no_conflict(self, tmp_path):
        """Test when filename doesn't exist."""
        os.chdir(tmp_path)
        filename = "test.txt"
        unique = utils.make_filename_unique(filename)
        assert unique == filename
    
    @pytest.mark.unit
    def test_make_filename_unique_with_conflict(self, tmp_path):
        """Test when filename already exists."""
        os.chdir(tmp_path)
        
        # Create existing file
        Path("test.txt").touch()
        
        unique = utils.make_filename_unique("test.txt")
        assert unique == "test_1.txt"
    
    @pytest.mark.unit
    def test_make_filename_unique_multiple_conflicts(self, tmp_path):
        """Test with multiple existing files."""
        os.chdir(tmp_path)
        
        # Create existing files
        Path("test.txt").touch()
        Path("test_1.txt").touch()
        Path("test_2.txt").touch()
        
        unique = utils.make_filename_unique("test.txt")
        assert unique == "test_3.txt"


class TestCosineSimilarity:
    """Test cosine similarity function."""
    
    @pytest.mark.unit
    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vec = [1, 2, 3, 4, 5]
        similarity = utils.cosine_similarity(vec, vec)
        assert pytest.approx(similarity, 0.0001) == 1.0
    
    @pytest.mark.unit
    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        similarity = utils.cosine_similarity(vec1, vec2)
        assert pytest.approx(similarity, 0.0001) == 0.0
    
    @pytest.mark.unit
    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = [1, 2, 3]
        vec2 = [-1, -2, -3]
        similarity = utils.cosine_similarity(vec1, vec2)
        assert pytest.approx(similarity, 0.0001) == -1.0
    
    @pytest.mark.unit
    def test_cosine_similarity_normalized(self):
        """Test cosine similarity with different magnitudes."""
        vec1 = [1, 1, 1]
        vec2 = [2, 2, 2]  # Same direction, different magnitude
        similarity = utils.cosine_similarity(vec1, vec2)
        assert pytest.approx(similarity, 0.0001) == 1.0
    
    @pytest.mark.unit
    def test_cosine_similarity_zero_vector(self):
        """Test handling of zero vectors."""
        vec1 = [0, 0, 0]
        vec2 = [1, 2, 3]
        
        # Should handle gracefully, typically returns 0
        similarity = utils.cosine_similarity(vec1, vec2)
        assert similarity == 0.0
    
    @pytest.mark.unit
    def test_cosine_similarity_numpy_arrays(self):
        """Test cosine similarity with numpy arrays."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        
        similarity = utils.cosine_similarity(vec1, vec2)
        assert isinstance(similarity, (float, np.floating))
        assert -1 <= similarity <= 1


class TestJaccardSimilarity:
    """Test Jaccard similarity function."""
    
    @pytest.mark.unit
    def test_jaccard_similarity_identical_sets(self):
        """Test Jaccard similarity of identical sets."""
        set_a = {1, 2, 3}
        similarity = utils.jaccard_similarity(set_a, set_a)
        assert similarity == 1.0
    
    @pytest.mark.unit
    def test_jaccard_similarity_disjoint_sets(self):
        """Test Jaccard similarity of disjoint sets."""
        set_a = {1, 2, 3}
        set_b = {4, 5, 6}
        similarity = utils.jaccard_similarity(set_a, set_b)
        assert similarity == 0.0
    
    @pytest.mark.unit
    def test_jaccard_similarity_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        set_a = {1, 2, 3}
        set_b = {2, 3, 4}
        similarity = utils.jaccard_similarity(set_a, set_b)
        # Intersection: {2, 3} = 2
        # Union: {1, 2, 3, 4} = 4
        # Similarity: 2/4 = 0.5
        assert similarity == 0.5
    
    @pytest.mark.unit
    def test_jaccard_similarity_empty_sets(self):
        """Test Jaccard similarity with empty sets."""
        set_a = set()
        set_b = set()
        similarity = utils.jaccard_similarity(set_a, set_b)
        assert similarity == 0.0
    
    @pytest.mark.unit
    def test_jaccard_similarity_one_empty_set(self):
        """Test Jaccard similarity when one set is empty."""
        set_a = {1, 2, 3}
        set_b = set()
        similarity = utils.jaccard_similarity(set_a, set_b)
        assert similarity == 0.0


class TestWordCountPenalty:
    """Test word count penalty function."""
    
    @pytest.mark.unit
    def test_word_count_penalty_below_threshold(self):
        """Test penalty when count is below threshold."""
        penalty = utils.word_count_penalty(50, 100, 100)
        assert penalty == 0.5  # 50/100
    
    @pytest.mark.unit
    def test_word_count_penalty_at_threshold(self):
        """Test penalty when count equals threshold."""
        penalty = utils.word_count_penalty(100, 100, 100)
        assert penalty == 1.0
    
    @pytest.mark.unit
    def test_word_count_penalty_above_threshold(self):
        """Test penalty when count is above threshold."""
        penalty = utils.word_count_penalty(150, 100, 100)
        assert penalty == 1.0  # Capped at 1.0
    
    @pytest.mark.unit
    def test_word_count_penalty_zero_count(self):
        """Test penalty with zero count."""
        penalty = utils.word_count_penalty(0, 100, 100)
        assert penalty == 0.0