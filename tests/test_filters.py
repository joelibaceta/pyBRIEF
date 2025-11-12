import numpy as np
import pytest
from pybrief.filters import smooth, sobel_gradients, normalize
from pybrief.utils import gaussian_kernel

# --- Fixtures (pytest) ---

@pytest.fixture
def sample_image():
    """Creates a simple 5x5 test image with a single bright pixel."""
    img = np.zeros((5, 5))
    img[2, 2] = 1.0
    return img

@pytest.fixture
def gradient_image():
    """Simple image with increasing horizontal intensity."""
    return np.tile(np.linspace(0, 1, 5), (5, 1))

# --- Tests ---

def test_smooth_with_default_kernel(sample_image):
    result = smooth(sample_image)
    assert result.shape == sample_image.shape
    assert result[2, 2] < 1.0  # Center pixel should be reduced
    assert result.sum() > 1.0  # Due to reflection padding, sum increases
    assert np.max(result) == result[2, 2]  # Center should still be maximum

def test_smooth_with_custom_kernel(sample_image):
    kernel = gaussian_kernel(size=3, sigma=1.0)
    result = smooth(sample_image, kernel)
    assert result.shape == sample_image.shape
    assert result[2, 2] < 1.0  # Center pixel should be reduced
    assert result.sum() > 0.9  # Reasonable lower bound

def test_sobel_gradients_shape(gradient_image):
    Ix, Iy = sobel_gradients(gradient_image)
    assert Ix.shape == gradient_image.shape
    assert Iy.shape == gradient_image.shape

def test_sobel_gradients_behavior(gradient_image):
    Ix, Iy = sobel_gradients(gradient_image)
    # Should have strong horizontal gradient, near zero vertical
    assert np.abs(Ix).mean() > np.abs(Iy).mean()

def test_normalize_range():
    img = np.array([[0, 50], [100, 150]], dtype=np.float64)
    norm = normalize(img)
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)

def test_normalize_constant_image():
    img = np.full((3, 3), 42.0)
    norm = normalize(img)
    assert np.all(norm == 0.0)
