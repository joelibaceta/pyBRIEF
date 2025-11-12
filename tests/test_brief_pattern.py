import numpy as np
import pytest
from pybrief.brief_pattern import generate_brief_tests

def test_generate_brief_tests_shape():
    p, q = generate_brief_tests(n_bits=256, patch_size=31, rng_seed=42)
    assert p.shape == (256, 2)
    assert q.shape == (256, 2)

def test_generate_brief_tests_within_bounds():
    patch_size = 31
    half = (patch_size - 1) / 2
    p, q = generate_brief_tests(n_bits=128, patch_size=patch_size, rng_seed=0)
    assert np.all(np.abs(p) <= half)
    assert np.all(np.abs(q) <= half)

def test_generate_brief_tests_reproducibility():
    p1, q1 = generate_brief_tests(n_bits=64, patch_size=31, rng_seed=123)
    p2, q2 = generate_brief_tests(n_bits=64, patch_size=31, rng_seed=123)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(q1, q2)

def test_generate_brief_tests_distribution_centered():
    p, q = generate_brief_tests(n_bits=512, patch_size=31, rng_seed=7)
    mean_p = np.mean(p)
    mean_q = np.mean(q)
    # Expect approximately centered around 0
    assert abs(mean_p) < 1.0
    assert abs(mean_q) < 1.0

def test_generate_brief_tests_different_seed_changes_output():
    p1, q1 = generate_brief_tests(n_bits=32, patch_size=31, rng_seed=1)
    p2, q2 = generate_brief_tests(n_bits=32, patch_size=31, rng_seed=2)
    # Different seeds should yield different results
    assert not np.array_equal(p1, p2)
    assert not np.array_equal(q1, q2)
