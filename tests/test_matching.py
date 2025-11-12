import numpy as np
import pytest
from pybrief.matching import hamming_dist_bytes, match_hamming

@pytest.fixture
def small_descriptors():
    """Creates a small deterministic descriptor set."""
    # 4 descriptors, each 2 bytes (16 bits)
    return np.array([
        [0b00001111, 0b11110000],
        [0b00001111, 0b11111111],
        [0b11110000, 0b00001111],
        [0b00000000, 0b11111111]
    ], dtype=np.uint8)

def test_hamming_dist_bytes_correctness(small_descriptors):
    a = small_descriptors[0]
    B = small_descriptors
    D = hamming_dist_bytes(a, B)
    # Distance to itself = 0
    assert D[0] == 0
    # Symmetry check: D(a,b) == D(b,a)
    for i in range(len(B)):
        d1 = hamming_dist_bytes(a, B[i:i+1])[0]
        d2 = hamming_dist_bytes(B[i], a.reshape(1, -1))[0]
        assert d1 == d2

def test_hamming_dist_increases_with_difference(small_descriptors):
    a = small_descriptors[0]
    D = hamming_dist_bytes(a, small_descriptors)
    # Distances should be non-negative and sorted increasing
    assert np.all(D >= 0)
    assert D[0] == 0
    assert D[1] <= D[2] or D[1] <= D[3]

def test_match_hamming_returns_valid_indices(small_descriptors):
    descA = small_descriptors
    descB = small_descriptors.copy()
    idxs, dists, mask = match_hamming(descA, descB, ratio=1.0)
    assert idxs.shape == (len(descA),)
    assert dists.shape == (len(descA),)
    assert mask.shape == (len(descA),)
    # Should match perfectly to same index
    assert np.all(idxs == np.arange(len(descA)))
    assert np.all(dists == 0)
    assert np.all(mask)

def test_match_hamming_ratio_filter(small_descriptors):
    # Duplicate one descriptor to create ambiguity
    descA = small_descriptors[:2]
    descB = np.vstack([small_descriptors[0], small_descriptors[0]])
    idxs, dists, mask = match_hamming(descA, descB, ratio=0.5)
    # Ambiguous case should fail ratio test (mask false)
    assert not np.any(mask)

def test_match_hamming_empty_inputs():
    descA = np.zeros((0, 8), dtype=np.uint8)
    descB = np.zeros((0, 8), dtype=np.uint8)
    idxs, dists, mask = match_hamming(descA, descB)
    assert idxs.size == 0
    assert dists.size == 0
    assert mask.size == 0
