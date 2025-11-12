import numpy as np

def pad_reflect(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """
    Pads an image using reflection mode.

    Parameters
    ----------
    img : np.ndarray
        Input 2D grayscale image.
    pad_h : int
        Vertical padding size.
    pad_w : int
        Horizontal padding size.

    Returns
    -------
    np.ndarray
        Padded image.
    """
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')


def gaussian_kernel(size: int = 9, sigma: float = 2.0) -> np.ndarray:
    """
    Creates a normalized 2D Gaussian kernel.

    Parameters
    ----------
    size : int
        Size of the kernel (odd number).
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    np.ndarray
        2D Gaussian kernel normalized to sum = 1.
    """
    assert size % 2 == 1, "Kernel size must be odd."
    ax = np.arange(size) - (size - 1) / 2
    g1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g1d /= g1d.sum()
    kernel = np.outer(g1d, g1d)
    return kernel


def convolve2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies 2D convolution with 'same' output size using reflection padding.

    Parameters
    ----------
    img : np.ndarray
        Input 2D grayscale image.
    kernel : np.ndarray
        Convolution kernel.

    Returns
    -------
    np.ndarray
        Convolved image (same size as input).
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = pad_reflect(img, ph, pw)
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float64)

    # Flip the kernel for convolution (not correlation)
    kernel_flipped = np.flipud(np.fliplr(kernel))

    for i in range(H):
        for j in range(W):
            patch = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(patch * kernel_flipped)
    return out
