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

def process_video_frames(video_path, frame_interval_sec=1.0, output_dir="output/frames"):
    frame_paths = extract_frames(video_path, frame_interval_sec, output_dir)

    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"画像読み込み失敗: {path}")
            continue

        result = ocr_on_matching_regions(img)

        print(f"Frame: {os.path.basename(path)}")
        for key, val in result.items():
            print(f"  {key}: {val}")
        print("-" * 30)


if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    video_file = config["video_path"]
    frame_interval = config["frame_interval"]
    output_dir = config.get("output_dir", "output/frames")

    process_video_frames(video_file, frame_interval, output_dir)
