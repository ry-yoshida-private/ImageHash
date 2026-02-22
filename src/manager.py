import cv2
import numpy as np
from typing import Union
from scipy.spatial.distance import pdist, cdist, squareform
from dataclasses import dataclass, field
from typing import Callable

from .method import HashMethod
from .wavelet import WaveletHash, WaveletMode

@dataclass
class ImageHashManager:
    """
    Manages hash distance computation. Uses Hamming (bit) or Euclidean depending on the hash method.
    """
    method: HashMethod
    hash_obj: Union[cv2.img_hash.ImgHashBase, WaveletHash] = field(init=False)
    cross_fn: Callable[[np.ndarray, np.ndarray], float] = field(init=False)
    pairwise_fn: Callable[[np.ndarray], np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.hash_obj = self.method.obj
        if self.method.is_bit_convertible:
            self.cross_fn = self._measure_cross_by_hamming
            self.pairwise_fn = self._measure_pairwise_by_hamming
        else:
            self.cross_fn = self._measure_cross_by_euclidean
            self.pairwise_fn = self._measure_pairwise_by_euclidean

    def compute_hash(self, image: np.ndarray) -> np.ndarray:
        """
        Compute hash value from an image using the underlying hash object (_obj).

        NOTE
        - Standard cv2.img_hash.ImgHashBase.compare() does not work because flatten() is conducted.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale image (OpenCV-style).

        Returns
        -------
        np.ndarray
            Hash array (e.g. shape (8,) for 64-bit methods). dtype and shape depend on the method.
        """
        return self.hash_obj.compute(image).flatten()

    @staticmethod
    def to_binary(arr: np.ndarray) -> np.ndarray:
        """
        Convert hash array(s) (e.g. uint8 bytes) to binary array(s) (0/1).
        Use this before measure_* when the hash is raw bytes.
        Supports both a single hash and a stack of hashes in one call.

        Parameters
        ----------
        arr : np.ndarray
            - 1D: single hash, e.g. shape (8,) uint8 for 8-byte hash.
            - 2D: stacked hashes, shape (N, num_bytes) e.g. (N, 8).

        Returns
        -------
        np.ndarray
            - 1D input -> 1D binary, shape (num_bits,) dtype uint8, values 0 or 1.
            - 2D input -> 2D binary, shape (N, num_bits) for use with measure_pairwise_distance.
        """
        arr = np.asarray(arr).astype(np.uint8)
        if arr.ndim == 1:
            return np.unpackbits(arr)
        if arr.ndim == 2:
            return np.unpackbits(arr, axis=1)
        raise ValueError("arr must be 1D (single hash) or 2D (stack of hashes).")

    def measure_cross_distance(
        self, 
        hash_one: np.ndarray, 
        hash_two: np.ndarray
        ) -> float:
        """
        Measure distance between two hashes (one cross comparison).
        For bit methods, inputs must be already binarized (use to_binary() if needed).

        Parameters
        ----------
        hash_one: np.ndarray
            First hash.
        hash_two: np.ndarray
            Second hash.

        Returns
        -------
        float
            Distance between the two hashes.
        """
        return self.cross_fn(hash_one, hash_two)

    def measure_pairwise_distance(
        self, 
        hash_values: np.ndarray
        ) -> np.ndarray:
        """
        Measure pairwise distance matrix between multiple hashes (all pairs).

        Parameters
        ----------
        hash_values: np.ndarray
            Array of hashes (already binarized; use to_binary() beforehand if needed).

        Returns
        -------
        np.ndarray
            Pairwise distance matrix (N, N).
        """
        return self.pairwise_fn(hash_values)

    def _measure_cross_by_hamming(
        self, 
        hash_one: np.ndarray, 
        hash_two: np.ndarray
        ) -> np.ndarray:
        """
        Distance between two hashes (matrix-based via cdist).

        Parameters
        ----------
        hash_one: np.ndarray
            First hash, shape (N, num_bits).
        hash_two: np.ndarray
            Second hash, shape (N, num_bits).

        Returns
        -------
        np.ndarray
            Distance matrix (N, N).
        """
        return cdist(hash_one, hash_two, metric="cityblock")      

    def _measure_pairwise_by_hamming(
        self, 
        hash_values: np.ndarray
        ) -> np.ndarray:
        """
        Pairwise distance matrix between multiple hashes (all pairs).

        Parameters
        ----------
        hash_values: np.ndarray
            Array of hashes with shape (N, num_bits) already binarized.
            -> use to_binary() beforehand if needed.

        Returns
        -------
        np.ndarray
            Distance matrix (N, N).
        """
        dist_v = pdist(hash_values, metric="cityblock")
        return squareform(dist_v)

    def _measure_cross_by_euclidean(
        self, 
        hash_one: np.ndarray, 
        hash_two: np.ndarray
        ) -> float:
        """
        Distance between two hashes (matrix-based via cdist).

        Parameters
        ----------
        hash_one: np.ndarray
            First hash, shape (N, num_bits).
        hash_two: np.ndarray
            Second hash, shape (N, num_bits).
        Returns
        -------
        float
            Distance between the two hashes.
        """
        return cdist(hash_one, hash_two, metric="euclidean") * 10000

    def _measure_pairwise_by_euclidean(
        self, 
        hash_values: np.ndarray
        ) -> np.ndarray:
        """
        Pairwise distance matrix between multiple hashes (N x N).

        Parameters
        ----------
        hash_values: np.ndarray
            Array of hashes with shape (N, num_bits). Caller must ensure shape.

        Returns
        -------
        np.ndarray
            Distance matrix of shape (N, N).
        """
        return cdist(hash_values, hash_values, metric="euclidean") * 10000

    def update_wavelet_obj(
        self,
        hash_size: int,
        image_scale: int | None,
        mode: WaveletMode = WaveletMode.Haar,
        remove_max_haar_ll: bool = True
        ) -> None:
        """
        Update the wavelet object.

        Parameters
        ----------
        hash_size: int
            The size of the hash.
        image_scale: int | None
            The scale of the image.
        mode: WaveletMode
            The mode of the wavelet.
        remove_max_haar_ll: bool
            Whether to remove the maximum Haar LL band.
        """
        if self.method != HashMethod.WaveletHash:
            raise ValueError("This method does not support wavelet hash.")
        wavelet_obj = WaveletHash(
            hash_size=hash_size,
            image_scale=image_scale,
            mode=mode,
            remove_max_haar_ll=remove_max_haar_ll
        )
        self.hash_obj = wavelet_obj
