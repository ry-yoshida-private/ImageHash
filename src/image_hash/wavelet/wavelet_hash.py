from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import cv2
import numpy as np
import pywt  # type: ignore[import-untyped]

from .mode import WaveletMode

type DetailCoeff2D = tuple[np.ndarray, np.ndarray, np.ndarray]
type Coeffs2D = list[np.ndarray | DetailCoeff2D]


class WaveletSpec(Protocol):
    dec_len: int


class PywtModule(Protocol):
    def wavedec2(
        self,
        data: np.ndarray,
        wavelet: str,
        mode: str = "symmetric",
        level: int | None = None,
        axes: tuple[int, int] = (-2, -1),
    ) -> Coeffs2D: ...

    def waverec2(
        self,
        coeffs: Coeffs2D,
        wavelet: str,
        mode: str = "symmetric",
        axes: tuple[int, int] = (-2, -1),
    ) -> np.ndarray: ...

    def Wavelet(self, name: str) -> WaveletSpec: ...

    def dwt_max_level(self, data_len: int, filter_len: int) -> int: ...


PYWT = cast(PywtModule, pywt)


@dataclass
class WaveletHash:
    """
    Wavelet Hash: hash_size * hash_size bits from DWT low-frequency band.

    - compute(image) -> hash (np.ndarray, shape (hash_size**2 / 8,), dtype uint8)
    - compare(hash_one, hash_two) -> Hamming distance (float)
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
        coeffs = PYWT.wavedec2(float_img, self.mode.value, level=dwt_level)
        dwt_low = cast(np.ndarray, coeffs[0])

        if dwt_low.shape != (self.hash_size, self.hash_size):
            dwt_low = cv2.resize(
                dwt_low, (self.hash_size, self.hash_size), interpolation=cv2.INTER_AREA
            )

        threshold = float(np.median(dwt_low))
        hash_bit_mask = (dwt_low > threshold).astype(np.uint8).flatten()
        return np.packbits(hash_bit_mask)

    def compare(
        self,
        hash_one: np.ndarray,
        hash_two: np.ndarray
    ) -> float:
        a = np.unpackbits(np.asarray(hash_one, dtype=np.uint8))
        b = np.unpackbits(np.asarray(hash_two, dtype=np.uint8))
        return float(np.count_nonzero(a != b))

    def _resize_and_normalize(
        self, gray: np.ndarray, image_scale: int
    ) -> np.ndarray:
        resized = cv2.resize(
            gray, (image_scale, image_scale), interpolation=cv2.INTER_AREA
        )
        return np.asarray(resized, dtype=np.float64) / 255.0

    def _remove_max_haar_ll(
        self,
        img_float: np.ndarray,
        ll_max_level: int
    ) -> np.ndarray:
        coeffs = PYWT.wavedec2(img_float, "haar", level=ll_max_level)
        dwt_low = cast(np.ndarray, coeffs[0])
        dwt_low *= 0
        coeffs[0] = dwt_low
        return PYWT.waverec2(coeffs, "haar")

    def _compute_safe_dwt_level(
        self,
        image_scale: int
    ) -> int:
        wavelet = PYWT.Wavelet(self.mode.value)
        required_level = int(np.log2(image_scale // self.hash_size))
        max_level = PYWT.dwt_max_level(image_scale, wavelet.dec_len)

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
        if self.image_scale is not None:
            return self.image_scale

        image_natural_scale = 2 ** int(np.log2(min(h, w)))
        return max(image_natural_scale, self.hash_size)
