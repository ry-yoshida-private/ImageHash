import numpy as np
from scipy.spatial.distance import cdist

from ..hasher import ImageHasher


class VectorHasher(ImageHasher):
    """
    Hasher for non-bit-convertible (vector) methods. Uses Euclidean distance.

    Attributes:
    ----------
    method: HashMethod
        The hash method to use.
    hash_obj: Union[cv2.img_hash.ImgHashBase, WaveletHash]
        The hash object to use.
    """

    def __post_init__(self) -> None:       
        if self.method.is_bit_convertible:
            raise ValueError(
                f"VectorHasher requires a non-bit-convertible method, got {self.method}"
            )

    def measure_cross_distance(
        self, 
        hash_one: np.ndarray, 
        hash_two: np.ndarray
        ) -> np.ndarray:
        """
        Measure cross distance(s) between hash_one and hash_two.

        Parameters
        ----------
        hash_one : np.ndarray
            First hash or stack of hashes (1D or 2D).
        hash_two : np.ndarray
            Second hash or stack of hashes (1D or 2D).

        Returns
        -------
        np.ndarray
            Distance matrix (n1, n2).

            NOTE
            - The distance is 1/10000 of the distance evaluated by OpenCV's built-in function.
        """
        one = np.atleast_2d(hash_one)
        two = np.atleast_2d(hash_two)
        return cdist(one, two, metric="euclidean")

    def measure_pairwise_distance(
        self, 
        hash_values: np.ndarray
        ) -> np.ndarray:
        """
        Measure pairwise distance matrix between multiple hashes (all pairs).

        Parameters
        ----------
        hash_values : np.ndarray
            Array of hashes (already binarized; use to_binary() beforehand if needed).

        Returns
        -------
        np.ndarray
            Pairwise distance matrix (N, N).

            NOTE
            - The distance is 1/10000 of the distance evaluated by OpenCV's built-in function.
        """
        return cdist(hash_values, hash_values, metric="euclidean")
