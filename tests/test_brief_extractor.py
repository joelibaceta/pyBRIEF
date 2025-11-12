import numpy as np
import pytest
from pybrief import BriefDescriptorExtractor, create_brief_extractor


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    return img


@pytest.fixture
def sample_keypoints():
    """Create sample keypoints within valid bounds."""
    return np.array([
        [20, 20],
        [30, 40], 
        [50, 60],
        [70, 80]
    ])


def test_brief_extractor_initialization():
    """Test BriefDescriptorExtractor initialization."""
    extractor = BriefDescriptorExtractor()
    assert extractor.get_n_bits() == 256
    assert extractor.get_patch_size() == 31
    assert extractor.descriptor_size() == 32  # 256 bits = 32 bytes
    assert extractor.descriptor_type() == np.uint8
    assert extractor.descriptor_shape() == (32,)


def test_brief_extractor_custom_params():
    """Test BriefDescriptorExtractor with custom parameters."""
    extractor = BriefDescriptorExtractor(n_bits=128, patch_size=21)
    assert extractor.get_n_bits() == 128
    assert extractor.get_patch_size() == 21
    assert extractor.descriptor_size() == 16  # 128 bits = 16 bytes


def test_create_brief_extractor():
    """Test the convenience function."""
    extractor = create_brief_extractor(n_bits=64)
    assert isinstance(extractor, BriefDescriptorExtractor)
    assert extractor.get_n_bits() == 64
    assert extractor.descriptor_size() == 8  # 64 bits = 8 bytes


def test_compute_basic(sample_image, sample_keypoints):
    """Test basic compute functionality."""
    extractor = BriefDescriptorExtractor()
    
    kp_out, descriptors = extractor.compute(sample_image, sample_keypoints)
    
    assert kp_out.shape == sample_keypoints.shape
    assert descriptors.shape[0] == len(sample_keypoints)
    assert descriptors.shape[1] == extractor.descriptor_size()
    assert descriptors.dtype == np.uint8


def test_compute_empty_keypoints(sample_image):
    """Test compute with empty keypoints."""
    extractor = BriefDescriptorExtractor()
    
    kp_out, descriptors = extractor.compute(sample_image, np.array([]))
    
    assert len(kp_out) == 0
    assert descriptors.shape == (0, extractor.descriptor_size())


def test_compute_none_inputs():
    """Test compute with None inputs."""
    extractor = BriefDescriptorExtractor()
    
    kp_out, descriptors = extractor.compute(None, None)
    
    assert len(descriptors) == 0
    assert descriptors.shape == (0, extractor.descriptor_size())


def test_compute_invalid_keypoints(sample_image):
    """Test compute with keypoints outside image bounds."""
    extractor = BriefDescriptorExtractor(patch_size=31)
    
    # Keypoints too close to borders (within patch_size//2 = 15 pixels)
    invalid_kp = np.array([
        [5, 5],    # Too close to top-left
        [95, 95],  # Too close to bottom-right  
        [10, 50],  # Too close to top
        [50, 5]    # Too close to left
    ])
    
    kp_out, descriptors = extractor.compute(sample_image, invalid_kp)
    
    # Should filter out invalid keypoints
    assert len(kp_out) == 0
    assert descriptors.shape == (0, extractor.descriptor_size())


def test_compute_mixed_keypoints(sample_image):
    """Test compute with mix of valid and invalid keypoints."""
    extractor = BriefDescriptorExtractor(patch_size=31)
    
    mixed_kp = np.array([
        [5, 5],     # Invalid - too close to border
        [30, 30],   # Valid
        [50, 50],   # Valid
        [95, 95]    # Invalid - too close to border
    ])
    
    kp_out, descriptors = extractor.compute(sample_image, mixed_kp)
    
    # Should only return valid keypoints
    assert len(kp_out) == 2
    assert descriptors.shape == (2, extractor.descriptor_size())


def test_descriptor_properties():
    """Test descriptor property methods."""
    extractor = BriefDescriptorExtractor(n_bits=512)
    
    assert extractor.descriptor_size() == 64  # 512 bits = 64 bytes
    assert extractor.descriptor_type() == np.uint8
    assert extractor.descriptor_shape() == (64,)


def test_smoothing_params():
    """Test smoothing parameter getter."""
    extractor = BriefDescriptorExtractor(smoothing_sigma=1.5, smoothing_size=7)
    
    sigma, size = extractor.get_smoothing_params()
    assert sigma == 1.5
    assert size == 7


def test_extractor_repr():
    """Test string representation."""
    extractor = BriefDescriptorExtractor(n_bits=128, patch_size=21)
    repr_str = repr(extractor)
    
    assert "BriefDescriptorExtractor" in repr_str
    assert "n_bits=128" in repr_str
    assert "patch_size=21" in repr_str


def test_compute_reproducibility(sample_image, sample_keypoints):
    """Test that compute produces consistent results."""
    extractor1 = BriefDescriptorExtractor(rng_seed=42)
    extractor2 = BriefDescriptorExtractor(rng_seed=42)
    
    _, desc1 = extractor1.compute(sample_image, sample_keypoints)
    _, desc2 = extractor2.compute(sample_image, sample_keypoints)
    
    np.testing.assert_array_equal(desc1, desc2)


def test_compute_different_seeds(sample_image, sample_keypoints):
    """Test that different seeds produce different results."""
    extractor1 = BriefDescriptorExtractor(rng_seed=0)
    extractor2 = BriefDescriptorExtractor(rng_seed=1)
    
    _, desc1 = extractor1.compute(sample_image, sample_keypoints)
    _, desc2 = extractor2.compute(sample_image, sample_keypoints)
    
    # Should be different (with high probability)
    assert not np.array_equal(desc1, desc2)