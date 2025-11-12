import numpy as np
import pytest
from pybrief.brief_descriptor import extract_brief
from pybrief.brief_pattern import generate_brief_tests

@pytest.fixture
def small_image():
    """Simple 8x8 grayscale image with a bright pixel in the center."""
    img = np.zeros((8, 8), dtype=np.float64)
    img[4, 4] = 1.0
    return img

@pytest.fixture
def keypoints():
    """A few fixed keypoints to test descriptor extraction."""
    return np.array([[4, 4], [2, 2], [6, 6]], dtype=np.int32)

@pytest.fixture
def tests():
    """Pre-generated BRIEF test pairs."""
    return generate_brief_tests(n_bits=128, patch_size=31, rng_seed=42)

def test_extract_brief_shape(small_image, keypoints, tests):
    desc = extract_brief(small_image, keypoints, tests, patch_size=31)
    # 128 bits = 16 bytes
    assert desc.shape == (len(keypoints), 16)
    assert desc.dtype == np.uint8

def test_extract_brief_reproducibility(small_image, keypoints, tests):
    desc1 = extract_brief(small_image, keypoints, tests, patch_size=31)
    desc2 = extract_brief(small_image, keypoints, tests, patch_size=31)
    np.testing.assert_array_equal(desc1, desc2)

def test_extract_brief_image_change_affects_descriptor(small_image, keypoints, tests):
    desc1 = extract_brief(small_image, keypoints, tests)
    # Change image brightness pattern
    img2 = small_image.copy()
    img2[1:3, 1:3] = 1.0
    desc2 = extract_brief(img2, keypoints, tests)
    # Expect at least one descriptor to diffe
