import numpy as np

def generate_brief_tests(
    n_bits: int = 256,
    patch_size: int = 31,
    rng_seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates BRIEF binary test pairs (p, q) using Gaussian-II sampling.

    Each test compares the intensity of two pixels within a local patch.

    Parameters
    ----------
    n_bits : int
        Number of binary tests (descriptor length = n_bits).
    patch_size : int
        Size of the local square patch centered on the keypoint.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    (p, q) : tuple of np.ndarray
        Two arrays of shape (n_bits, 2) containing relative (y, x) coordinates.
        Values are in pixel offsets relative to the patch center.
    """
    rng = np.random.default_rng(rng_seed)

    # Standard deviation per paper: sigma = S / 5
    sigma = patch_size / 5.0

    # Sample 2 * n_bits coordinates from N(0, sigma^2)
    pts = rng.normal(loc=0.0, scale=sigma, size=(2 * n_bits, 2))

    # Clamp coordinates within patch bounds
    half = (patch_size - 1) / 2
    pts = np.clip(pts, -half + 1, half - 1)

    # Split into p and q pairs
    p = np.round(pts[:n_bits]).astype(np.int16)
    q = np.round(pts[n_bits:]).astype(np.int16)

    return p, q
