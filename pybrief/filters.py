import numpy as np
from .utils import convolve2d_same, gaussian_kernel

def smooth(img: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Applies Gaussian smoothing to the image.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (2D).
    kernel : np.ndarray, optional
        Custom Gaussian kernel. If None, uses default 9x9 sigma=2.

    Returns
    -------
    np.ndarray
        Smoothed image.
    """
    if kernel is None:
        kernel = gaussian_kernel(size=9, sigma=2.0)
    return convolve2d_same(img, kernel)


def sobel_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Sobel gradients in X and Y directions.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (2D).

    Returns
    -------
    (Ix, Iy) : tuple of np.ndarray
        Gradients along X and Y.
    """
    # Sobel kernels
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float64)
    Ky = Kx.T

    Ix = convolve2d_same(img, Kx)
    Iy = convolve2d_same(img, Ky)
    return Ix, Iy


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalizes an image to the [0, 1] range.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    img = img.astype(np.float64)
    min_val, max_val = img.min(), img.max()
    if max_val - min_val < 1e-10:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)
