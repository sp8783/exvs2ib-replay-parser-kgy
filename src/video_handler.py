import cv2
import os
import yaml
from src.regions_matching import ocr_on_matching_regions

def extract_frames(video_path, frame_interval_sec, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画ファイルが開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_interval_sec)
    frame_count = 0
    saved_paths = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{idx:05d}.png")
            cv2.imwrite(frame_path, frame)
            saved_paths.append(frame_path)
            idx += 1
        frame_count += 1

    cap.release()
    return saved_paths
