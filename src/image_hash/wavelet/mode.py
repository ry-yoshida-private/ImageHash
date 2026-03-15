from enum import Enum

class WaveletMode(Enum):
    """
    Wavelet modes.

    Attributes:
    ----------
    Haar: Haar wavelet
    Daubechies: Daubechies wavelet
    Symlet: Symlet wavelet
    Coiflet: Coiflet wavelet
    Biorthogonal: Biorthogonal wavelet
    ReverseBiorthogonal: Reverse Biorthogonal wavelet
    """
    Haar = "haar"
    Daubechies = "db4"
    Symlet = "sym4"
    Coiflet = "coif1"
    Biorthogonal = "bior1.1"
    ReverseBiorthogonal = "rbio1.1"