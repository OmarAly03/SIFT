from pathlib import Path
import cv2
from feature_matcher import FeatureMatcher
from visualizer import Visualizer
from utils import load_image, load_video, compute_homography, get_query_corners

def process_video(query_path: str, video_path: str, output_path: str):
    """Process a video to detect and outline an object in each frame using SIFT."""
    matcher = FeatureMatcher()
    visualizer = Visualizer()
    
    query_img_gray = load_image(query_path, grayscale=True)
    query_img_color = load_image(query_path)
    
    # Detecting and computing keypoints/descriptors for query image
    kp1, des1 = matcher.detect_and_compute(query_img_gray)
    
    # Loading video and getting its properties
    cap = load_video(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = visualizer.create_video_writer(output_path, fps, frame_width, frame_height)
    
    h, w = query_img_gray.shape
    corners = get_query_corners(h, w)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = matcher.detect_and_compute(frame_gray)
        matches = matcher.match_descriptors(des1, des2)
        
        # Computing homography and outline
        if len(matches) >= 4:
            src_pts, dst_pts = matcher.get_matched_points(kp1, kp2, matches)
            H, _ = compute_homography(src_pts, dst_pts)
            
            if H is not None:
                transformed_corners = cv2.perspectiveTransform(corners, H)
                frame = visualizer.draw_outline(frame, transformed_corners)

        out.write(frame)
        
        # Displaying frame
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    query_path = "images/bic_query.png"
    video_path = "videos/bic_target.mp4"
    output_name = Path(query_path).stem.split('_')[0] + "_video_output.mp4"
    output_path = f"outputs/{output_name}"
    process_video(query_path, video_path, output_path)