import numpy as np
from .filters import sobel_gradients, smooth

def harris_corners(
    img: np.ndarray,
    k: float = 0.04,
    window_size: int = 3,
    thresh_rel: float = 0.01
) -> np.ndarray:
    """
    Computes the Harris corner response map.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (2D, uint8 or float64).
    k : float
        Harris detector free parameter, usually in [0.04, 0.06].
    window_size : int
        Size of the window used to average gradient products.
    thresh_rel : float
        Relative threshold w.r.t max response (values below are set to 0).

    Returns
    -------
    np.ndarray
        Harris response map (same shape as input).
    """
    # Ensure float64 for numerical precision
    img = img.astype(np.float64)

    # 1. Compute gradients
    Ix, Iy = sobel_gradients(img)

    # 2. Products of derivatives
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 3. Smooth gradient products
    w = np.ones((window_size, window_size), dtype=np.float64)
    w /= w.sum()  # simple box filter
    Sxx = smooth(Ixx, w)
    Syy = smooth(Iyy, w)
    Sxy = smooth(Ixy, w)

    # 4. Harris response
    detM = (Sxx * Syy) - (Sxy ** 2)
    traceM = Sxx + Syy
    R = detM - k * (traceM ** 2)

    # 5. Threshold relative to maximum
    Rmax = R.max()
    R[R < thresh_rel * Rmax] = 0

    return R
