"""
pyBRIEF - Python implementation of BRIEF descriptor
"""

__version__ = "0.1.0"

from .brief_descriptor import extract_brief
from .brief_pattern import generate_brief_tests
from .brief_extractor import BriefDescriptorExtractor, create_brief_extractor
from .corners import harris_corners
from .filters import smooth, sobel_gradients, normalize
from .matching import hamming_dist_bytes, match_hamming
from .nms import nms_points
from .utils import pad_reflect, gaussian_kernel, convolve2d_same

__all__ = [
    # Version
    '__version__',
    
    # Core BRIEF functionality
    'BriefDescriptorExtractor',
    'create_brief_extractor', 
    'extract_brief',
    'generate_brief_tests',
    
    # Corner detection and NMS
    'harris_corners',
    'nms_points',
    
    # Image processing
    'smooth',
    'sobel_gradients', 
    'normalize',
    'pad_reflect',
    'gaussian_kernel',
    'convolve2d_same',
    
    # Feature matching
    'hamming_dist_bytes',
    'match_hamming',
]