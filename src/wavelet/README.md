# wavelet

Wavelet-based perceptual hash (DWT, low-frequency band, median threshold, 64-bit hash).

| File | Role |
|------|------|
| [wavelet_hash.py](wavelet_hash.py) | `WaveletHash`: `compute(image)`, `compare(hash_a, hash_b)`; uses PyWavelets |
| [mode.py](mode.py) | `WaveletMode` enum: Haar, Daubechies, Symlet, Coiflet, Biorthogonal, ReverseBiorthogonal, MRA |
