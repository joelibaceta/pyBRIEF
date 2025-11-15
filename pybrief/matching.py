import numpy as np
from typing import Tuple

# Precompute popcount lookup table for 0..255
_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def hamming_dist_bytes(a_row: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes Hamming distance between a single descriptor and all others.

    Parameters
    ----------
    a_row : np.ndarray
        1D array of bytes (one descriptor).
    B : np.ndarray
        2D array of bytes (M descriptors).

    Returns
    -------
    np.ndarray
        1D array of Hamming distances (length = M).
    """
    xor = np.bitwise_xor(B, a_row)
    # Sum the number of 1 bits across all bytes
    return _POPCNT8[xor].sum(axis=1)


def match_hamming(
    descA: np.ndarray,
    descB: np.ndarray,
    ratio: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Matches BRIEF descriptors using Hamming distance and ratio test.

    Parameters
    ----------
    descA : np.ndarray
        Descriptor set A, shape (N, n_bytes).
    descB : np.ndarray
        Descriptor set B, shape (M, n_bytes).
    ratio : float
        Ratio test threshold (0 < ratio <= 1). Lower = stricter.

    Returns
    -------
    (idxs, dists, mask) : tuple
        idxs : np.ndarray
            Indices of matched descriptors in B for each descriptor in A.
        dists : np.ndarray
            Corresponding Hamming distances.
        mask : np.ndarray
            Boolean mask of valid matches passing the ratio test.
    """
    N = descA.shape[0]
    M = descB.shape[0]

    idxs = np.full(N, -1, dtype=np.int32)
    dists = np.full(N, np.iinfo(np.int32).max, dtype=np.int32)
    mask = np.zeros(N, dtype=bool)

    for i, a in enumerate(descA):
        D = hamming_dist_bytes(a, descB)
        if M < 2:
            j = np.argmin(D)
            idxs[i] = j
            dists[i] = int(D[j])
            mask[i] = True
        else:
            j1, j2 = np.argsort(D)[:2]
            d1, d2 = int(D[j1]), int(D[j2])
            if d1 < ratio * d2:
                idxs[i] = j1
                dists[i] = d1
                mask[i] = True

    return idxs, dists, mask
