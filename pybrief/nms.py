import numpy as np

def nms_points(
    R: np.ndarray,
    radius: int = 8,
    max_points: int = 1000
) -> np.ndarray:
    """
    Applies Non-Maximum Suppression (NMS) to a response map.

    Parameters
    ----------
    R : np.ndarray
        2D Harris (or other) response map.
    radius : int
        Minimum distance (in pixels) between keypoints.
    max_points : int
        Maximum number of keypoints to keep.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with keypoint coordinates (y, x).
    """
    # 1. Get all coordinates with positive response
    coords = np.argwhere(R > 0)
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    # 2. Sort by descending response strength
    scores = R[R > 0]
    order = np.argsort(scores)[::-1]

    H, W = R.shape
    grid = np.zeros((H, W), dtype=np.uint8)
    keypoints = []

    # 3. Iterate through sorted points
    for idx in order:
        y, x = coords[idx]
        if grid[max(0, y - radius):y + radius + 1,
                max(0, x - radius):x + radius + 1].any():
            continue  # skip if close to an already chosen point
        keypoints.append((y, x))
        grid[max(0, y - radius):y + radius + 1,
             max(0, x - radius):x + radius + 1] = 1
        if len(keypoints) >= max_points:
            break

    return np.array(keypoints, dtype=np.int32)
