
import numpy as np
import pytest
from pybrief.corners import harris_corners

@pytest.fixture
def simple_corner_image():
    """
    Creates a 5x5 image with a bright quadrant — a synthetic corner.
    Example:
        [0 0 1 1 1]
        [0 0 1 1 1]
        [1 1 1 1 1]
        [1 1 1 1 1]
        [1 1 1 1 1]
    """
    img = np.ones((5, 5))
    img[:2, :2] = 0.0
    return img

def test_harris_corners_output_shape(simple_corner_image):
    R = harris_corners(simple_corner_image)
    assert R.shape == simple_corner_image.shape

def test_harris_response_positive(simple_corner_image):
    R = harris_corners(simple_corner_image, k=0.04, thresh_rel=0.0)
    # Expect at least one strong corner response > 0
    assert np.any(R > 0)

def test_harris_threshold_removes_weak(simple_corner_image):
    R_low = harris_corners(simple_corner_image, thresh_rel=0.0)
    R_high = harris_corners(simple_corner_image, thresh_rel=0.5)
    assert np.count_nonzero(R_high) < np.count_nonzero(R_low)

def test_harris_response_symmetry():
    img = np.zeros((7, 7))
    img[3:, 3:] = 1.0  # bright bottom-right quadrant
    R1 = harris_corners(img)
    R2 = harris_corners(np.flipud(np.fliplr(img)))  # flipped image
    # Harris should be invariant under reflection
    assert np.isclose(R1.sum(), R2.sum(), atol=1e-6)
