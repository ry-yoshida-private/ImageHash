# hashers

## Overview

Hasher implementations used by `ImageHasher.build()`.

## Components

| File / Dir | Role |
|------------|------|
| [bitwise.py](bitwise.py) | `BitwiseHasher`: base hasher for bit-convertible methods using Hamming distance (`cityblock`) |
| [vector.py](vector.py) | `VectorHasher`: hasher for non-bit-convertible methods (`COLOR_MOMENT`) using Euclidean distance |

## Notes

- Input shape handling: both hashers accept 1D (single hash) and 2D (stacked hashes) for cross-distance.
- Pairwise distance: `BitwiseHasher` uses `pdist + squareform`; `VectorHasher` uses `cdist`.
- `WAVELET` is handled by `BitwiseHasher` with `WaveletHash` object passed as `hash_obj`.
