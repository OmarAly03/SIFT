import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

class Visualizer:
    """Class for visualizing and saving results."""
    
    @staticmethod
    def draw_outline(img: np.ndarray, corners: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
        """Draw a polygon outline on the image."""
        img_copy = img.copy()
        cv2.polylines(img_copy, [np.int32(corners)], True, color, 3)
        return img_copy
    
    @staticmethod
    def draw_matches(img1: np.ndarray, kp1: List, img2: np.ndarray, kp2: List, matches: List) -> np.ndarray:
        """Draw matched keypoints between two images."""
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    @staticmethod
    def save_image(img: np.ndarray, output_path: str):
        """Save an image to disk."""
        cv2.imwrite(output_path, img)
    
    @staticmethod
    def create_video_writer(video_path: str, fps: int, width: int, height: int) -> cv2.VideoWriter:
        """Create a video writer object."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    @staticmethod
    def display_image_results(query_img: np.ndarray, target_img: np.ndarray, matches_img: np.ndarray):
        """Display image processing results using matplotlib."""
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Query Image")
        plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Target Image with Outline")
        plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("SIFT Matches")
        plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()