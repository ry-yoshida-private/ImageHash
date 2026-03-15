# wavelet

Wavelet-based perceptual hash (DWT, low-frequency band, median threshold, 64-bit hash).

## Modes

| Mode | Description |
|------|-------------|
| Haar | Haar wavelet |
| Daubechies | Daubechies wavelet (db4) |
| Symlet | Symlet wavelet (sym4) |
| Coiflet | Coiflet wavelet (coif1) |
| Biorthogonal | Biorthogonal wavelet (bior1.1) |
| ReverseBiorthogonal | Reverse Biorthogonal wavelet (rbio1.1) |

## Components

| File | Role |
|------|------|
| [wavelet_hash.py](wavelet_hash.py) | WaveletHash implementation (PyWavelets) |
| [mode.py](mode.py) | WaveletMode enum |
