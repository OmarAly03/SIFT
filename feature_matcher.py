import cv2
import numpy as np
from typing import Tuple, List

class FeatureMatcher:
    """Class for SIFT-based feature detection and matching."""
    
    def __init__(self, distance_ratio: float = 0.8):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.distance_ratio = distance_ratio
    
    def detect_and_compute(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect keypoints and compute descriptors."""
        kp, des = self.sift.detectAndCompute(img, None)
        if des is None:
            return kp, np.array([])
        return kp, des
    
    def match_descriptors(self, des1: np.ndarray, des2: np.ndarray) -> List:
        """Match descriptors using KNN and apply distance ratio test."""
        if des1.size == 0 or des2.size == 0:
            return []
        matches = self.bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2 and m_n[0].distance < self.distance_ratio * m_n[1].distance:
                good_matches.append(m_n[0])
        return good_matches
    
    def get_matched_points(self, kp1: List, kp2: List, matches: List) -> Tuple[np.ndarray, np.ndarray]:
        """Extract matched keypoint coordinates."""
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return src_pts, dst_pts