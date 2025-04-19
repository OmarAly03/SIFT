import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def load_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """Load an image in grayscale or color."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(image_path), flag)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    return img

def load_video(video_path: str) -> cv2.VideoCapture:
    """Load a video and verify it's accessible."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")
    return cap

def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Compute homography matrix using RANSAC."""
    if len(src_pts) < 4 or len(dst_pts) < 4:
        return None, np.array([])
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

def get_query_corners(height: int, width: int) -> np.ndarray:
    """Define corners of the query image for homography transformation."""
    return np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)