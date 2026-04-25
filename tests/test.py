"""
Tests for image_hash: hash computation and cross distance matrix per method.
- Pytest: uses fixtures from conftest (dummy_images, real_images) and _helpers (build_hasher).
- Script: run with optional CLI for ad-hoc checks (e.g. --data-dir, --methods, --list-methods).
"""
from pathlib import Path

import argparse
import time
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None  # script-only run without pytest

from image_hash.method import HashMethod
from image_hash.hasher import ImageHasher

from _helpers import build_hasher, load_images_from_paths


def run_cross_distance(
    hasher: ImageHasher,
    images: dict[str, np.ndarray],
    method: HashMethod,
) -> np.ndarray:
    """
    Compute hashes for all images and return the cross distance matrix.
    Handles bit-convertible vs vector methods internally.
    """
    hashes = {name: hasher.compute(img) for name, img in images.items()}
    raw = np.array([h for h in hashes.values()])
    if method.is_bit_convertible:
        binary = hasher.to_binary(raw)
        return hasher.measure_cross_distance(binary, binary)
    return hasher.measure_cross_distance(raw, raw)


# --- Pytest: parametrized with dummy images / real paths ---

if pytest is not None:

    @pytest.mark.parametrize("method", list(HashMethod))
    def test_cross_distance_dummy(method: HashMethod, dummy_images: dict[str, np.ndarray]):
        """Each method: build hasher, compute cross distance on dummy images; diagonal is 0."""
        hasher = build_hasher(method)
        cross_dist = run_cross_distance(hasher, dummy_images, method)
        n = len(dummy_images)
        assert cross_dist.shape == (n, n)
        np.testing.assert_array_almost_equal(np.diag(cross_dist), 0)

    def test_cross_distance_real_paths(real_images: dict[str, np.ndarray]):
        """
        Run all methods on real images from data/; print timing and matrix.
        Skipped when data/source.jpg, data/positive.jpg, data/negative.jpg are not present.
        """
        for method in HashMethod:
            hasher = build_hasher(method)
            t0 = time.perf_counter()
            cross_dist = run_cross_distance(hasher, real_images, method)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert cross_dist.shape == (3, 3)
            print(f"--- method: {method.value} ---")
            print(f"elapsed time: {elapsed_ms:.2f} ms")
            print(f"cross_dist_matrix:\n{cross_dist}")


# --- Script mode: flexible CLI ---

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run image_hash cross-distance on images (dummy or from paths)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing source.jpg, positive.jpg, negative.jpg (default: repo data/)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated method names or 'all' (default: all)",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print available methods and exit",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only print errors",
    )
    return parser.parse_args()


def _resolve_methods(names: str) -> list[HashMethod]:
    if names.strip().lower() == "all":
        return list(HashMethod)
    chosen: list[HashMethod] = []
    for s in names.split(","):
        s = s.strip()
        try:
            chosen.append(HashMethod[s])
        except KeyError:
            raise SystemExit(f"Unknown method: {s}. Use --list-methods to see options.")
    return chosen


def main() -> None:
    args = _parse_args()

    if args.list_methods:
        print("Available methods:", ", ".join(m.value for m in HashMethod))
        return

    # Image source
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        paths = {
            "source": data_dir / "source.jpg",
            "positive": data_dir / "positive.jpg",
            "negative": data_dir / "negative.jpg",
        }
        if not all(p.exists() for p in paths.values()):
            raise SystemExit(f"Missing images in {data_dir}: need source.jpg, positive.jpg, negative.jpg")
        images = load_images_from_paths(paths)
        source_label = str(data_dir)
    else:
        h, w = 64, 64
        rng = np.random.default_rng(114514)
        images = {
            "source": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
            "positive": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
            "negative": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        }
        source_label = "dummy"

    methods = _resolve_methods(args.methods)

    for method in methods:
        hasher = build_hasher(method)
        t0 = time.perf_counter()
        cross_dist = run_cross_distance(hasher, images, method)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if not args.quiet:
            print(f"--- method: {method.value} (images: {source_label}) ---")
            print(f"elapsed time: {elapsed_ms:.2f} ms")
            print(f"cross_dist_matrix:\n{cross_dist}")


if __name__ == "__main__":
    main()
