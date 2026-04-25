# ImageHash

## Overview

Compare images using perceptual hash methods. Computes hashes for a source image and positive/negative references, then reports pairwise distances and timing.
See the [image_hash README](src/image_hash/README.md#available-methods) for method details (Pros/Cons).

## Requirements

- Python 3.x
- numpy, opencv-contrib-python, scipy

## Setup

```bash
pip install -r requirements.txt
```

## Example

Load an image with OpenCV and compute its hash:

```python
import cv2
from image_hash import HashMethod, ImageHasher

# Load image (BGR or grayscale)
image = cv2.imread("path/to/image.jpg")

# Build hasher with desired method (e.g. PHash)
hasher = ImageHasher.build(HashMethod.P)

# Compute hash (e.g. 8-byte ndarray)
hash_value = hasher.compute_hash(image)
print(hash_value)  # e.g. [123 45 67 89 ...]

# As bit array (for Hamming distance)
hash_bits = ImageHasher.to_binary(hash_value)
print(hash_bits.shape)  # (64,) for 64-bit methods
```

