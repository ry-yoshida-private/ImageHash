# ImageHash

Compare images using perceptual hash methods. Computes hashes for a source image and positive/negative references, then reports pairwise distances and timing.

## Requirements

- Python 3.x
- numpy, opencv-contrib-python, scipy

## Setup

```bash
pip install -r requirements.txt
```


## Available methods

| Method | Pros | Cons |
|--------|------|------|
| AverageHash | Fast, simple, high performance for coarse similarity | Sensitive to scaling, rotation, and lighting changes |
| PHash | Industry standard; robust against minor edits and noise | Slightly higher CPU cost than AverageHash |
| WaveletHash | Excellent for texture/structure; multi-scale analysis | Implementation complexity; slower than DCT-based methods |
| ColorMomentHash | Invariant to structural changes; focuses on color distribution | Metric mismatch: uses Euclidean distance, not Hamming distance |

## Example

Run from the project root:

```bash
python -m src \
    --source_image_path data/source.jpg \
    --positive_image_path data/positive.jpg \
    --negative_image_path data/negative.jpg \
    --methods AverageHash WaveletHash BlockMeanHash MarrHildrethHash RadialVarianceHash PHash ColorMomentHash
```

- **source_image_path** – main image to compare (default: `data/source.jpg`)
- **positive_image_path** – reference image expected to be similar (default: `data/positive.jpg`)
- **negative_image_path** – reference image expected to be different (default: `data/negative.jpg`)
- **methods** – one or more hash methods to run (default: all)


