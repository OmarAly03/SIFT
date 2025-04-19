from pathlib import Path
from feature_matcher import FeatureMatcher
from visualizer import Visualizer
from utils import load_image, compute_homography, get_query_corners
import cv2

def process_images(query_path: str, target_path: str, output_dir: str):
    """Process two images to detect and outline an object using SIFT."""
    matcher = FeatureMatcher()
    visualizer = Visualizer()
    
    query_img_gray = load_image(query_path, grayscale=True)
    query_img_color = load_image(query_path)
    target_img_gray = load_image(target_path, grayscale=True)
    target_img_color = load_image(target_path)
    
    # Detecting and computing keypoints/descriptors
    kp1, des1 = matcher.detect_and_compute(query_img_gray)
    kp2, des2 = matcher.detect_and_compute(target_img_gray)
    
    # Matching descriptors
    matches = matcher.match_descriptors(des1, des2)
    
    # Computing homography for outlining
    target_img_with_outline = target_img_color
    inlier_matches = matches[:20]  # top 20 matches
    
    if len(matches) >= 4:
        src_pts, dst_pts = matcher.get_matched_points(kp1, kp2, matches)
        H, mask = compute_homography(src_pts, dst_pts)
        
        if H is not None:
            # Transforming query image corners
            h, w = query_img_gray.shape
            corners = get_query_corners(h, w)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            # Drawing outline
            target_img_with_outline = visualizer.draw_outline(target_img_color, transformed_corners)
            inlier_matches = [m for i, m in enumerate(matches) if mask[i][0] == 1]
    
    matches_img = visualizer.draw_matches(query_img_color, kp1, target_img_color, kp2, inlier_matches[:20])
    visualizer.display_image_results(query_img_color, target_img_with_outline, matches_img)
    
    query_name_prefix = Path(query_path).stem.split('_')[0]

    output_subdir = Path(output_dir) / query_name_prefix
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    visualizer.save_image(matches_img, str(output_subdir / "img_matches.jpg"))
    visualizer.save_image(target_img_with_outline, str(output_subdir / "target_with_outline.jpg"))

if __name__ == "__main__":
    query_path = "images/aquaphor2_query.jpg"
    target_path = "images/aquaphor2_target.png"
    output_dir = "outputs"
    process_images(query_path, target_path, output_dir)