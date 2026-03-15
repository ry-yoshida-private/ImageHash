import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

from ..hasher import ImageHasher


class BitwiseHasher(ImageHasher):
    """
    Hasher for bit-convertible methods. Uses Hamming (cityblock) distance.

    Attributes:
    ----------
    method: HashMethod
        The hash method to use.
    hash_obj: Union[cv2.img_hash.ImgHashBase, WaveletHash]
        The hash object to use.
    """
    def __post_init__(self) -> None:
        if not self.method.is_bit_convertible:
            raise ValueError(
                f"BitwiseHasher requires a bit-convertible method, got {self.method}"
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
        """
        one = np.atleast_2d(hash_one)
        two = np.atleast_2d(hash_two)
        return cdist(one, two, metric="cityblock")

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
        """
        dist_v = pdist(hash_values, metric="cityblock")
        return squareform(dist_v)
