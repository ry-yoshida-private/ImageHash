import time
import cv2
import numpy as np
from argparse import ArgumentParser

from .method import HashMethod
from .manager import ImageHashManager


def main() -> None:
    all_method_names = [m.value for m in HashMethod]
    parser = ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="*",
        type=str,
        default=all_method_names,
        metavar="NAME",
        help=f"Hash method name(s). Default: all. Choices: {', '.join(all_method_names)}",
    )
    parser.add_argument("--source_image_path", type=str, default="data/source.jpg")
    parser.add_argument("--positive_image_path", type=str, default="data/positive.jpg")
    parser.add_argument("--negative_image_path", type=str, default="data/negative.jpg")
    args = parser.parse_args()

    image_paths = {
        "source": args.source_image_path,
        "positive": args.positive_image_path,
        "negative": args.negative_image_path,
    }
    images = {}
    for name, path in image_paths.items():
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read {path}")
        images[name] = img

    methods = [HashMethod(name) for name in args.methods]

    for method in methods:
        t0 = time.perf_counter()
        manager = ImageHashManager(method=method)
        hashes = {name: manager.compute_hash(img) for name, img in images.items()}
        raw_hashes = np.array([h for h in hashes.values()])
        if method.is_bit_convertible:
            bin_hashes = manager.to_binary(raw_hashes)
            #pairwise_dist_matrix = manager.measure_pairwise_distance(bin_hashes)
            cross_dist_matrix = manager.measure_cross_distance(
                bin_hashes, bin_hashes
            )
        else:
            #pairwise_dist_matrix = manager.measure_pairwise_distance(raw_hashes)
            cross_dist_matrix = manager.measure_cross_distance(
                raw_hashes, raw_hashes
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"--- method: {method.value} ---")
        print(f"elapsed time: {elapsed_ms:.2f} ms")
        print(f"cross_dist_matrix: \n{cross_dist_matrix}")

if __name__ == "__main__":
    main()
