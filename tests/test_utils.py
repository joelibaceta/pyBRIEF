import numpy as np
import pytest
from pybrief.utils import pad_reflect, gaussian_kernel, convolve2d_same

def test_pad_reflect_shape_and_values():
    img = np.array([[1, 2],
                    [3, 4]], dtype=np.float64)
    padded = pad_reflect(img, 1, 1)
    assert padded.shape == (4, 4)
    # Check reflected corners
    assert padded[0, 0] == 4
    assert padded[-1, -1] == 1

def test_gaussian_kernel_properties():
    k = gaussian_kernel(size=5, sigma=1.0)
    assert k.shape == (5, 5)
    np.testing.assert_almost_equal(k.sum(), 1.0, decimal=6)
    # Center pixel should be maximum
    assert k[2, 2] == pytest.approx(k.max())

def test_gaussian_kernel_symmetry():
    k = gaussian_kernel(size=7, sigma=2.0)
    assert np.allclose(k, np.flipud(k))
    assert np.allclose(k, np.fliplr(k))

def test_convolve2d_same_identity_kernel():
    img = np.arange(9, dtype=np.float64).reshape(3, 3)
    identity = np.zeros((3, 3))
    identity[1, 1] = 1.0
    result = convolve2d_same(img, identity)
    np.testing.assert_array_equal(result, img)

def test_convolve2d_same_gaussian_smoothing():
    img = np.zeros((5, 5))
    img[2, 2] = 1.0
    k = gaussian_kernel(size=3, sigma=1.0)
    result = convolve2d_same(img, k)
    assert result[2, 2] < 1.0
    np.testing.assert_almost_equal(result.sum(), 1.0, decimal=4)
