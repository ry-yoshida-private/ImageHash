"""
Shared helpers for image_hash tests (no pytest dependency).
Used by test.py (script mode) and conftest.py (fixtures).
"""
from pathlib import Path

import cv2
import numpy as np

from image_hash.method import HashMethod
from image_hash.hasher import ImageHasher
from image_hash.wavelet import WaveletHash


def build_hasher(method: HashMethod) -> ImageHasher:
    """Build ImageHasher for the given method (single place for WaveletHash handling)."""
    if method == HashMethod.WAVELET:
        wavelet_obj = method.object
        if not isinstance(wavelet_obj, WaveletHash):
            raise TypeError("HashMethod.WAVELET must provide WaveletHash object.")
        return ImageHasher.build(method=method, wavelet_obj=wavelet_obj)
    return ImageHasher.build(method=method)


def load_images_from_paths(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    """Load images from paths. Raises if any path is invalid."""
    images: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        path = str(path)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read {path}")
        images[name] = img
    return images
