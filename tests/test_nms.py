import numpy as np
import pytest
from pybrief.nms import nms_points

@pytest.fixture
def sample_response_map():
    """Simple 10x10 map with several peaks."""
    R = np.zeros((10, 10))
    R[2, 2] = 10
    R[5, 5] = 8
    R[8, 8] = 9
    R[3, 3] = 4
    return R

def test_nms_output_shape(sample_response_map):
    kp = nms_points(sample_response_map, radius=2, max_points=5)
    assert kp.ndim == 2
    assert kp.shape[1] == 2  # (y, x) pairs
    assert issubclass(kp.dtype.type, np.integer)

def test_nms_respects_max_points(sample_response_map):
    kp = nms_points(sample_response_map, radius=1, max_points=2)
    assert len(kp) <= 2

def test_nms_suppresses_nearby_points():
    R = np.zeros((8, 8))
    R[3, 3] = 10
    R[4, 4] = 9
    R[7, 7] = 8
    kp = nms_points(R, radius=2)
    # With radius=2, the strong point (3,3) should suppress nearby points
    assert len(kp) == 1
    assert tuple(kp[0]) == (3, 3)  # The strongest point should be selected

def test_nms_no_positive_values():
    R = np.zeros((5, 5))
    kp = nms_points(R)
    assert kp.shape == (0, 2)

def test_nms_picks_highest_response(sample_response_map):
    kp = nms_points(sample_response_map, radius=1, max_points=3)
    yx = [tuple(k) for k in kp]
    # Ensure the strongest (2,2) point is included
    assert (2, 2) in yx
