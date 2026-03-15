from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pywt

from .mode import WaveletMode

@dataclass
class WaveletHash:
    """
    Wavelet Hash: hash_size * hash_size bits from DWT low-frequency band.

    - compute(image) -> hash (np.ndarray, shape (hash_size**2 / 8,), dtype uint8)
    - compare(hash_one, hash_two) -> Hamming distance (float)

    Attributes:
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
    hash_size: int = 8
    image_scale: int | None = None
    mode: WaveletMode = WaveletMode.Haar
    remove_max_haar_ll: bool = True

    def __post_init__(self) -> None:
        if self.hash_size <= 0 or (self.hash_size & (self.hash_size - 1)) != 0:
            raise ValueError("hash_size must be a power of 2")
        if self.image_scale is not None and (self.image_scale & (self.image_scale - 1)) != 0:
            raise ValueError("image_scale must be a power of 2")

    def compute(
        self, 
        image: np.ndarray
        ) -> np.ndarray:
        """
        Compute Wavelet Hash from an image.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale image (OpenCV-style).

        Returns
        -------
        np.ndarray
            Hash of shape (hash_size**2 / 8,) dtype uint8.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape

        image_scale = self._compute_image_scale(h=h, w=w)
        float_img = self._resize_and_normalize(gray, image_scale)
        
        if self.remove_max_haar_ll:
            ll_max_level = int(np.log2(image_scale))
            float_img = self._remove_max_haar_ll(float_img, ll_max_level)

        dwt_level = self._compute_safe_dwt_level(image_scale)
        coeffs = pywt.wavedec2(float_img, self.mode.value, level=dwt_level)
        dwt_low = coeffs[0]

        if dwt_low.shape != (self.hash_size, self.hash_size):
            dwt_low = cv2.resize(
                dwt_low, (self.hash_size, self.hash_size), interpolation=cv2.INTER_AREA
            )

        threshold = np.median(dwt_low)
        hash_bit_mask = (dwt_low > threshold).astype(np.uint8).flatten()
        return np.packbits(hash_bit_mask)

    def compare(
        self, 
        hash_one: np.ndarray, 
        hash_two: np.ndarray
        ) -> float:
        """
        Return Hamming distance between two hashes.

        Parameters
        ----------
        hash_one, hash_two : np.ndarray
            Hashes from compute().

        Returns
        -------
        float
            Number of differing bits (0 = identical).
        """
        a = np.unpackbits(np.asarray(hash_one, dtype=np.uint8))
        b = np.unpackbits(np.asarray(hash_two, dtype=np.uint8))
        return float(np.count_nonzero(a != b))

    def _resize_and_normalize(
        self, gray: np.ndarray, image_scale: int) -> np.ndarray:
        """
        Resize image to image_scale (width, height) and normalize to [0, 1] float64.
        
        Parameters
        ----------
        gray: np.ndarray
            Grayscale image.
        image_scale: int
            The scale of the image (power of 2).

        Returns
        -------
        np.ndarray
            Resized and normalized image.
        """
        resized = cv2.resize(
            gray, (image_scale, image_scale), interpolation=cv2.INTER_AREA
            )
        return np.asarray(resized, dtype=np.float64) / 255.0

    def _remove_max_haar_ll(
        self, 
        img_float: np.ndarray, 
        ll_max_level: int
        ) -> np.ndarray:
        """
        Remove the maximum-level Haar LL band (zero it out and reconstruct).
        
        Parameters
        ----------
        img_float: np.ndarray
            Resized and normalized image.
        ll_max_level: int
            The maximum level of the Haar LL band.

        Returns
        -------
        np.ndarray
            Image with the maximum-level Haar LL band removed.
        """
        coeffs = pywt.wavedec2(img_float, "haar", level=ll_max_level)
        coeffs = list(coeffs)
        coeffs[0] *= 0
        return pywt.waverec2(coeffs, "haar")

    def _compute_safe_dwt_level(
        self, 
        image_scale: int
        ) -> int:
        """
        Compute the safe DWT level based on the selected mode and hash_size.

        Parameters
        ----------
        image_scale : int
            The scale of the image (power of 2).

        Returns
        -------
        int
            The level of the DWT.

        Raises
        ------
        ValueError
            If hash_size is incompatible with the image_scale or mode.
        """
        wavelet = pywt.Wavelet(self.mode.value)
        required_level = int(np.log2(image_scale // self.hash_size))
        max_level = pywt.dwt_max_level(image_scale, wavelet.dec_len)
        
        if required_level > max_level:
            raise ValueError(
                f"Mode {self.mode.value} requires larger image_scale for decomposition."
            )
        if required_level < 0:
            raise ValueError("hash_size cannot be larger than image_scale.")
            
        return required_level

    def _compute_image_scale(
        self, 
        h: int, 
        w: int
        ) -> int:
        """
        Compute the scale of the image.

        Parameters
        ----------
        h: int
            The height of the image.
        w: int
            The width of the image.

        Returns
        -------
        int
            The scale of the image.
        """
        if self.image_scale is not None:
            return self.image_scale
        
        image_natural_scale = 2 ** int(np.log2(min(h, w)))
        return max(image_natural_scale, self.hash_size)