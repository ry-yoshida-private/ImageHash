# src

CLI and core logic for image perceptual hashing.

| File | Role |
|------|------|
| [__main__.py](__main__.py) | Entry point: argument parsing, image loading, reporting of pairwise distances and timing |
| [method.py](method.py) | `HashMethod` enum and mapping to OpenCV / custom hash implementations (hash size, bit vs float) |
| [manager.py](manager.py) | `ImageHashManager`: computes hashes and pairwise/cross distances (Hamming or Euclidean by method) |
| [wavelet/](wavelet/) | Wavelet-based perceptual hash (DWT, 64-bit); `WaveletHash`, `WaveletMode` |
