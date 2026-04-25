"""
Pytest fixtures for image_hash tests.
Provides hasher factory, image sources (dummy / from paths), and config.
"""
from pathlib import Path

import numpy as np
import pytest

from _helpers import build_hasher, load_images_from_paths


# --- Paths ---

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def default_data_dir(project_root: Path) -> Path:
    return project_root / "data"


@pytest.fixture(scope="session")
def default_image_paths(default_data_dir: Path) -> dict[str, Path]:
    """Default real image paths (source, positive, negative)."""
    return {
        "source": default_data_dir / "source.jpg",
        "positive": default_data_dir / "positive.jpg",
        "negative": default_data_dir / "negative.jpg",
    }


@pytest.fixture(scope="session")
def real_images_available(default_image_paths: dict[str, Path]) -> bool:
    return all(p.exists() for p in default_image_paths.values())


# --- Image data ---

@pytest.fixture
def dummy_images() -> dict[str, np.ndarray]:
    """In-memory random BGR images (no file I/O)."""
    h, w = 64, 64
    rng = np.random.default_rng(114514)
    return {
        "source": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "positive": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "negative": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
    }


@pytest.fixture(scope="session")
def real_images(default_image_paths: dict[str, Path], real_images_available: bool):
    """Real images from default data dir. Skips if files are missing."""
    if not real_images_available:
        pytest.skip(
            "Data images not found (data/source.jpg, data/positive.jpg, data/negative.jpg)"
        )
    return load_images_from_paths(default_image_paths)


# --- Hasher ---

@pytest.fixture
def get_hasher():
    """Factory: given a HashMethod, returns a built ImageHasher."""
    return build_hasher
