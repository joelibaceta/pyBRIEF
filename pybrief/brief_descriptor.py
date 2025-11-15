import numpy as np
from typing import Optional, Tuple
from .filters import smooth

def extract_brief(
    img: np.ndarray,
    keypoints: np.ndarray,
    tests: Tuple[np.ndarray, np.ndarray],
    patch_size: int = 31,
    kernel: Optional[np.ndarray] = None,
    n_bits: Optional[int] = None
) -> np.ndarray:
    """
    Extracts BRIEF descriptors for a set of keypoints.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (2D, uint8 or float64).
    keypoints : np.ndarray
        Array of (y, x) keypoint coordinates.
    tests : tuple of np.ndarray
        (p, q) coordinates from generate_brief_tests().
    patch_size : int
        Size of the local patch around each keypoint.
    kernel : np.ndarray, optional
        Gaussian kernel for smoothing (default: sigma=2, size=9).
    n_bits : int, optional
        Number of binary tests (default: use all).

    Returns
    -------
    np.ndarray
        Descriptor array of shape (N, n_bytes) with dtype=uint8.
    """
    if img.dtype != np.float64:
        img = img.astype(np.float64)

    # 1. Smooth the image
    img_s = smooth(img, kernel)

    p, q = tests
    total_bits = p.shape[0]
    if n_bits is None:
        n_bits = total_bits
    n_bits = min(n_bits, total_bits)
    p = p[:n_bits]
    q = q[:n_bits]

    H, W = img_s.shape
    half = (patch_size - 1) // 2

    # 2. Pad reflect to handle borders
    pad = np.pad(img_s, ((half, half), (half, half)), mode='reflect')

    N = keypoints.shape[0]
    n_bytes = (n_bits + 7) // 8
    desc = np.zeros((N, n_bytes), dtype=np.uint8)

    # Precompute relative coordinates
    py, px = p[:, 0], p[:, 1]
    qy, qx = q[:, 0], q[:, 1]

    # 3. Iterate over keypoints
    for i, (y, x) in enumerate(keypoints):
        Y, X = y + half, x + half  # offset inside padded image
        vals_p = pad[Y + py, X + px]
        vals_q = pad[Y + qy, X + qx]
        bits = (vals_p < vals_q).astype(np.uint8)

        # 4. Pack bits into bytes
        for b in range(n_bits):
            if bits[b]:
                desc[i, b // 8] |= (1 << (b % 8))

    return desc
