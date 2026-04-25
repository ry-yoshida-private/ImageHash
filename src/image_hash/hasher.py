from __future__ import annotations
import cv2
import warnings
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from .method import HashMethod
from .wavelet import WaveletHash


@dataclass
class ImageHasher(ABC):
    """
    Base class for hash distance computation.
    Use build(method) to get BitwiseHasher or VectorHasher depending on the method.
    """
    method: HashMethod
    hash_obj: Union[cv2.img_hash.ImgHashBase, WaveletHash]

    def compute(
        self, 
        image: np.ndarray
        ) -> np.ndarray:
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

    @abstractmethod
    def measure_cross_distance(
        self,
        hash_one: np.ndarray,
        hash_two: np.ndarray,
        ) -> np.ndarray:
        """
        Measure cross distance(s) between hash_one and hash_two.
        For bit methods, inputs must be already binarized (use to_binary() if needed).

        Parameters
        ----------
        hash_one : np.ndarray
            First hash or stack of hashes (1D or 2D).
        hash_two : np.ndarray
            Second hash or stack of hashes (1D or 2D).

        Returns
        -------
        np.ndarray
            - Two 1D hashes -> shape (1, 1).
            - 1D vs 2D or 2D vs 2D -> shape (n1, n2) distance matrix.
        """

    @abstractmethod
    def measure_pairwise_distance(
        self, 
        hash_values: np.ndarray
        ) -> np.ndarray:
        """
        Measure pairwise distance matrix between multiple hashes (all pairs).

        Parameters
        ----------
        hash_values : np.ndarray
            Array of hashes (already binarized; use to_binary() beforehand if needed).

        Returns
        -------
        np.ndarray
            Pairwise distance matrix (N, N).
        """

    @classmethod
    def build(
        cls, 
        method: HashMethod,
        wavelet_obj: WaveletHash | None = None
        ) -> ImageHasher:
        """
        Create BitwiseHasher or VectorHasher according to the method.

        Parameters
        ----------
        method : HashMethod
            The hash method to use.
        wavelet_obj: WaveletHash | None
            The wavelet object to use.
        Returns
        -------
        ImageHasher
            BitwiseHasher for bit-convertible methods (Hamming distance),
            VectorHasher otherwise (Euclidean distance).
        """
        from .hashers.bitwise import BitwiseHasher
        if method == HashMethod.WAVELET:
            if wavelet_obj is None:
                wavelet_obj = WaveletHash()
                warnings.warn("Wavelet object is not provided, using default values.")
            return BitwiseHasher(method=method, hash_obj=wavelet_obj)
        else:
            if wavelet_obj is not None:
                warnings.warn("Wavelet object is not used for non-WAVELET methods.")

        if method.is_bit_convertible:
            return BitwiseHasher(method=method, hash_obj=method.object)
        else:
            from .hashers.vector import VectorHasher
            return VectorHasher(method=method, hash_obj=method.object)
