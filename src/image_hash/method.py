
from __future__ import annotations
import cv2
from enum import Enum
from typing import Union

from .wavelet import WaveletHash

class HashMethod(Enum):
    """
    Hash methods.

    Attributes:
    ----------
    P: PHash
    AVERAGE: AverageHash
    BLOCK_MEAN: BlockMeanHash
    COLOR_MOMENT: ColorMomentHash
    MARR_HILDRETH: MarrHildrethHash
    RADIAL_VARIANCE: RadialVarianceHash
    WAVELET: WaveletHash (original implementation)
    """
    P = "PHash"
    AVERAGE = "AverageHash"
    BLOCK_MEAN = "BlockMeanHash"
    COLOR_MOMENT = "ColorMomentHash"
    MARR_HILDRETH = "MarrHildrethHash"
    RADIAL_VARIANCE = "RadialVarianceHash"
    WAVELET = "WaveletHash"
    # DifferenceHash = "DifferenceHash"

    @property
    def obj(self) -> Union[cv2.img_hash.ImgHashBase, WaveletHash]:
        match self:
            case self.WAVELET:
                return WaveletHash()
            case _:
                return getattr(cv2.img_hash, f"{self.value}_create")()

    @property
    def hash_size(self) -> int:
        """
        Hash size.

        Returns:
        ----------
        int: The size of the hash.
        """
        match self:
            case self.P | self.AVERAGE | self.WAVELET:
                return 8
            case self.BLOCK_MEAN:
                return 32
            case self.COLOR_MOMENT:
                return 42
            case self.MARR_HILDRETH:
                return 72
            case self.RADIAL_VARIANCE:
                return 40
            case _:
                raise ValueError(f"Invalid hash method: {self}")

    @property
    def is_bit_convertible(self) -> bool:
        """
        Whether the hash is binary.

        Returns:
        ----------
        bool: Whether the hash is binary.
        """
        match self:
            case self.COLOR_MOMENT:
                return False
            case _:
                return True

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


