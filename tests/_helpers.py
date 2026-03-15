"""
Shared helpers for image_hash tests (no pytest dependency).
Used by test.py (script mode) and conftest.py (fixtures).
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import cv2
import numpy as np

from src.image_hash.method import HashMethod
from src.image_hash.hasher import ImageHasher


def build_hasher(method: HashMethod) -> ImageHasher:
    """Build ImageHasher for the given method (single place for WaveletHash handling)."""
    if method == HashMethod.WaveletHash:
        return ImageHasher.build(method=method, wavelet_obj=method.obj)
    return ImageHasher.build(method=method)


def load_images_from_paths(paths: dict[str, str | Path]) -> dict[str, np.ndarray]:
    """Load images from paths. Raises if any path is invalid."""
    images = {}
    for name, path in paths.items():
        path = str(path)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read {path}")
        images[name] = img
    return images
