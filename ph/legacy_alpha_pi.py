import cv2
import numpy as np
from gudhi import AlphaComplex

def remove_noisy_pts(pd: np.ndarray) -> np.ndarray:
    """
    Simplify persistence diagrams from pixel-level holes.
    """
    if pd is None or len(pd) == 0:
        return pd
    value_to_remove = np.array([0.25, 0.5])  # remove redundant noise
    mask = np.any(pd != value_to_remove, axis=1)
    denoised_pd = pd[mask]
    return denoised_pd


def pointcloud2D(dilated_img: np.ndarray) -> np.ndarray:
    """
    Only needed for alpha complex.
    Convert nonzero pixels to 2D point cloud (x,y).
    """
    y_coords, x_coords = np.where(dilated_img > 0)
    pointcloud = np.column_stack((x_coords, y_coords))
    return pointcloud


def alphacomplex(img: np.ndarray, alpha_value: float = None) -> list[np.ndarray]:
    """
    Alpha complex persistence (dim=1).
    """
    points = pointcloud2D(img)

    # Empty Point Cloud Protection: Preventing Errors in AlphaComplex when Input is Empty
    if points is None or len(points) == 0:
        return [np.empty((0, 2), dtype=np.float32)]

    alpha_complex = AlphaComplex(points=points)

    if alpha_value is None:
        st = alpha_complex.create_simplex_tree()
    else:
        st = alpha_complex.create_simplex_tree(max_alpha_square=alpha_value**2)

    st.persistence()
    dgm = np.array(st.persistence_intervals_in_dimension(1), dtype=np.float32)
    dgm = remove_noisy_pts(dgm)
    return [dgm]


class preprocess:
    """
    Image conditioning for topological extraction.
    """

    def __init__(self, thresh: int = None, kernel_size: int = 2, iterate: int = 1):
        self.thresh = thresh
        self.kernel = kernel_size
        self.iterate = iterate

    def threshold(self, img: np.ndarray) -> np.ndarray:
        """
        Keep pixels <= thresh, set others to 0.
        """
        if self.thresh is None:
            raise ValueError("preprocess(thresh=...) must be set.")
        thresh_img = np.where(img <= self.thresh, img, 0)
        return thresh_img

    def dilate(self, thresh_img: np.ndarray) -> np.ndarray:
        """
        Dilate to emphasize boundaries.
        """
        kernel = np.ones((self.kernel, self.kernel), np.uint8)
        dilated_img = cv2.dilate(cv2.convertScaleAbs(thresh_img), kernel, iterations=self.iterate)
        return np.array(dilated_img)
