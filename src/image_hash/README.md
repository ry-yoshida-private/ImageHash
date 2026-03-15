# image_hash

## Overview

CLI and core logic for image perceptual hashing.

## Available methods

| Method | Pros | Cons |
|--------|------|------|
| AverageHash | Fast, simple, high performance for coarse similarity | Sensitive to scaling, rotation, and lighting changes |
| PHash | Industry standard; robust against minor edits and noise | Slightly higher CPU cost than AverageHash |
| WaveletHash | Excellent for texture/structure; multi-scale analysis | Implementation complexity; slower than DCT-based methods |
| ColorMomentHash | Invariant to structural changes; focuses on color distribution | Metric mismatch: uses Euclidean distance, not Hamming distance |

## Components

| File / Dir | Role |
|------------|------|
| [method.py](method.py) | Hash method definitions|
| [hasher.py](hasher.py) | Base class and factory for image hashing |
| [hashers/](hashers/) | Bit-based (Hamming) and vector-based (Euclidean) hasher implementations |
| [wavelet/](wavelet/) | Wavelet-based perceptual hash |
