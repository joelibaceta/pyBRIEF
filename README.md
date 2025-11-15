# pyBRIEF

Python pure implementation of BRIEF (Binary Robust Independent Elementary Features) descriptor for computer vision applications.

## Features

- **BRIEF Descriptor Extraction**: Fast and efficient binary feature descriptor
- **Corner Detection**: Harris corner detection with Non-Maximum Suppression (NMS)
- **Feature Matching**: Hamming distance-based descriptor matching
- **Image Processing Utilities**: Gaussian smoothing, Sobel gradients, and more

## Installation

### From PyPI (recommended)

```bash
pip install pybrief
```

### From source

```bash
git clone https://github.com/joelibaceta/pyBRIEF.git
cd pyBRIEF
pip install -e .
```

## Quick Start

```python
import numpy as np
from pybrief import (
    create_brief_extractor,
    harris_corners,
    nms_points,
    match_hamming
)

# Load your image (grayscale numpy array)
image = np.array(...)  # Your image here

# Detect corners
corners = harris_corners(image, sigma=1.0, k=0.04, threshold=0.01)

# Apply Non-Maximum Suppression
keypoints = nms_points(corners, window_size=5)

# Create BRIEF extractor and compute descriptors
extractor = create_brief_extractor(patch_size=31, n_tests=256)
descriptors = extractor.compute(image, keypoints)

# Match features between two images
matches = match_hamming(descriptors1, descriptors2, threshold=50)
```

## API Reference

### Core Functions

- `create_brief_extractor(patch_size, n_tests)`: Create a BRIEF descriptor extractor
- `extract_brief(image, keypoints, tests, patch_size)`: Extract BRIEF descriptors
- `harris_corners(image, sigma, k, threshold)`: Detect Harris corners
- `nms_points(corners, window_size)`: Non-Maximum Suppression on corner points
- `match_hamming(desc1, desc2, threshold)`: Match descriptors using Hamming distance

### Image Processing

- `smooth(image, sigma)`: Gaussian smoothing
- `sobel_gradients(image)`: Compute Sobel gradients
- `normalize(image)`: Normalize image to [0, 1]
- `gaussian_kernel(size, sigma)`: Generate Gaussian kernel
- `convolve2d_same(image, kernel)`: 2D convolution with 'same' padding

## References

BRIEF descriptor was introduced in:

- Calonder, M., Lepetit, V., Strecha, C., & Fua, P. (2010). BRIEF: Binary robust independent elementary features. In European conference on computer vision (pp. 778-792). Springer, Berlin, Heidelberg.
