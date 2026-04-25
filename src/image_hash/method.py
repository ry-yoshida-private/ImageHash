
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
    P: P
    AVERAGE: AVERAGE
    BLOCK_MEAN: BLOCK_MEAN
    COLOR_MOMENT: COLOR_MOMENT
    MARR_HILDRETH: MARR_HILDRETH
    RADIAL_VARIANCE: RADIAL_VARIANCE
    WAVELET: WAVELET
    """
    P = "P"
    AVERAGE = "AVERAGE"
    BLOCK_MEAN = "BLOCK_MEAN"
    COLOR_MOMENT = "COLOR_MOMENT"
    MARR_HILDRETH = "MARR_HILDRETH"
    RADIAL_VARIANCE = "RADIAL_VARIANCE"
    WAVELET = "WAVELET"
    # DifferenceHash = "DifferenceHash"

    @property
    def object(self) -> Union[cv2.img_hash.ImgHashBase, WaveletHash]:
        """
        Get the object of the hash method.

        Returns:
        ----------
        Union[cv2.img_hash.ImgHashBase, WaveletHash]: The object of the hash method.
        """
        match self:
            case self.WAVELET:
                return WaveletHash()
            case _:
                return getattr(cv2.img_hash, f"{self.opencv_name}_create")()

    @property
    def opencv_name(self) -> str:
        match self:
            case self.P:
                return "PHash"
            case self.AVERAGE:
                return "AverageHash"
            case self.BLOCK_MEAN:
                return "BlockMeanHash"
            case self.COLOR_MOMENT:
                return "ColorMomentHash"
            case self.MARR_HILDRETH:
                return "MarrHildrethHash"
            case self.RADIAL_VARIANCE:
                return "RadialVarianceHash"
            case self.WAVELET:
                return "WaveletHash"

    @property
    def hash_size(self) -> int:
        """
        Hash size.

        Returns:
        ----------
        int: The size of the hash.
        """
        match self:
            case self.P | self.AVERAGE:
                return 8
            case self.BLOCK_MEAN:
                return 32
            case self.COLOR_MOMENT:
                return 42
            case self.MARR_HILDRETH:
                return 72
            case self.RADIAL_VARIANCE:
                return 40
            case self.WAVELET:
                raise ValueError(
                    "HashMethod.WAVELET does not have a fixed hash_size. "
                    "Use WaveletHash.hash_size from your WaveletHash instance instead."
                )

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


