# hashers

## Overview

Hasher implementations used by `ImageHasher.build()`.

## Components

| File | Role |
|------|------|
| [bitwise.py](bitwise.py) | `BitwiseHasher`: for bit-convertible methods (`P`, `AVERAGE`, `BLOCK_MEAN`, `MARR_HILDRETH`, `RADIAL_VARIANCE`, `WAVELET`) using Hamming distance (`cityblock`) |
| [vector.py](vector.py) | `VectorHasher`: for non-bit-convertible methods (`COLOR_MOMENT`) using Euclidean distance |

## Notes

- Input shape handling: both hashers accept 1D (single hash) and 2D (stacked hashes) for cross-distance.
- Pairwise distance: `BitwiseHasher` uses `pdist + squareform`; `VectorHasher` uses `cdist`.
