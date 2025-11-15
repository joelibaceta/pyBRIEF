import numpy as np
from .brief_descriptor import extract_brief
from .brief_pattern import generate_brief_tests
from .utils import gaussian_kernel


class PyBrief:
    """
    BRIEF (Binary Robust Independent Elementary Features) descriptor extractor.
    """
    
    def __init__(self, 
                 n_bits: int = 256,
                 patch_size: int = 31,
                 smoothing_sigma: float = 2.0,
                 smoothing_size: int = 9,
                 rng_seed: int = 0):
        """
        Initialize the BRIEF descriptor extractor.
        
        Parameters
        ----------
        n_bits : int
            Number of binary tests (descriptor length in bits).
        patch_size : int
            Size of the local square patch around each keypoint.
        smoothing_sigma : float
            Standard deviation for Gaussian smoothing kernel.
        smoothing_size : int
            Size of the Gaussian smoothing kernel.
        rng_seed : int
            Random seed for reproducible test pattern generation.
        """
        self.n_bits = n_bits
        self.patch_size = patch_size
        self.smoothing_sigma = smoothing_sigma
        self.smoothing_size = smoothing_size
        self.rng_seed = rng_seed
        
        # Pre-generate the test pattern and smoothing kernel
        self._tests = generate_brief_tests(n_bits, patch_size, rng_seed)
        self._kernel = gaussian_kernel(smoothing_size, smoothing_sigma)
        
    def compute(self, image: np.ndarray, keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute BRIEF descriptors for the given keypoints.
        
        Parameters
        ----------
        image : np.ndarray
            Input grayscale image (2D, uint8 or float64).
        keypoints : np.ndarray
            Array of keypoint coordinates, shape (N, 2) as (y, x) or (x, y).
            
        Returns
        -------
        keypoints : np.ndarray
            Input keypoints (unchanged).
        descriptors : np.ndarray
            BRIEF descriptors, shape (N, n_bytes) with dtype=uint8.
        """
        if image is None or keypoints is None or len(keypoints) == 0:
            empty_kp = np.array([], dtype=np.int32).reshape(0, 2) if keypoints is None else keypoints
            return empty_kp, np.array([], dtype=np.uint8).reshape(0, self.descriptor_size())
            
        # Ensure keypoints are in (y, x) format
        kp = self._ensure_yx_format(keypoints)
        
        # Filter keypoints to be within valid image bounds
        valid_kp = self._filter_valid_keypoints(kp, image.shape)
        
        if len(valid_kp) == 0:
            return valid_kp, np.array([], dtype=np.uint8).reshape(0, self.descriptor_size())
        
        # Extract descriptors
        descriptors = extract_brief(
            image, 
            valid_kp, 
            self._tests,
            patch_size=self.patch_size,
            kernel=self._kernel,
            n_bits=self.n_bits
        )
        
        return valid_kp, descriptors
    
    def descriptor_size(self) -> int:
        """
        Get the descriptor size in bytes.
        
        Returns
        -------
        int
            Number of bytes per descriptor.
        """
        return (self.n_bits + 7) // 8
    
    def descriptor_type(self) -> type:
        """
        Get the descriptor data type.
        
        Returns
        -------
        type
            Descriptor data type (np.uint8).
        """
        return np.uint8
    
    def descriptor_shape(self) -> tuple[int]:
        """
        Get the shape of a single descriptor.
        
        Returns
        -------
        tuple
            Shape of descriptor (n_bytes,).
        """
        return (self.descriptor_size(),)
    
    def get_n_bits(self) -> int:
        """
        Get the number of bits in the descriptor.
        
        Returns
        -------
        int
            Number of bits per descriptor.
        """
        return self.n_bits
    
    def get_patch_size(self) -> int:
        """
        Get the patch size used for descriptor extraction.
        
        Returns
        -------
        int
            Patch size in pixels.
        """
        return self.patch_size
    
    def get_smoothing_params(self) -> tuple[float, int]:
        """
        Get the smoothing parameters.
        
        Returns
        -------
        tuple
            (sigma, kernel_size) for Gaussian smoothing.
        """
        return self.smoothing_sigma, self.smoothing_size
    
    def _ensure_yx_format(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Ensure keypoints are in (y, x) format.
        
        Parameters
        ----------
        keypoints : np.ndarray
            Keypoints in either (y, x) or (x, y) format.
            
        Returns
        -------
        np.ndarray
            Keypoints in (y, x) format.
        """
        # Assume input is already in (y, x) format by default
        # You can add logic here to detect/convert if needed
        return np.asarray(keypoints)
    
    def _filter_valid_keypoints(self, keypoints: np.ndarray, image_shape: tuple) -> np.ndarray:
        """
        Filter keypoints to ensure they're within valid image bounds.
        
        Parameters
        ----------
        keypoints : np.ndarray
            Input keypoints as (y, x).
        image_shape : tuple
            Shape of the image (H, W).
            
        Returns
        -------
        np.ndarray
            Valid keypoints within image bounds.
        """
        if len(keypoints) == 0:
            return keypoints
            
        H, W = image_shape[:2]
        half = (self.patch_size - 1) // 2
        
        # Filter keypoints that are too close to borders
        valid_mask = (
            (keypoints[:, 0] >= half) & 
            (keypoints[:, 0] < H - half) &
            (keypoints[:, 1] >= half) & 
            (keypoints[:, 1] < W - half)
        )
        
        return keypoints[valid_mask]
    
    def __repr__(self) -> str:
        """String representation of the extractor."""
        return (f"PyBrief(n_bits={self.n_bits}, "
                f"patch_size={self.patch_size}, "
                f"smoothing_sigma={self.smoothing_sigma}, "
                f"smoothing_size={self.smoothing_size})")


# Convenience function for easy instantiation
def create_brief_extractor(n_bits: int = 256, 
                          patch_size: int = 31,
                          smoothing_sigma: float = 2.0,
                          smoothing_size: int = 9,
                          rng_seed: int = 0) -> PyBrief:
    """
    Create a BRIEF descriptor extractor.
    
    Similar to cv.xfeatures2d.BriefDescriptorExtractor_create()
    
    Parameters
    ----------
    n_bits : int
        Number of binary tests (descriptor length in bits).
    patch_size : int
        Size of the local square patch around each keypoint.
    smoothing_sigma : float
        Standard deviation for Gaussian smoothing kernel.
    smoothing_size : int
        Size of the Gaussian smoothing kernel.
    rng_seed : int
        Random seed for reproducible test pattern generation.
        
    Returns
    -------
    PyBrief
        Initialized BRIEF descriptor extractor.
    """
    return PyBrief(n_bits, patch_size, smoothing_sigma, 
                                   smoothing_size, rng_seed)