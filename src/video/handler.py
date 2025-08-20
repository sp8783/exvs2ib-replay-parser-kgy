import cv2
import os
from tqdm import tqdm
from src.util.io import ensure_dir

def open_video(video_path):
    """
    動画ファイルを開いてcv2.VideoCaptureオブジェクトを返す。
    開けない場合はFileNotFoundErrorを投げる。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画ファイルが開けません: {video_path}")
    return cap

def get_fps(cap):
    """
    動画のFPS（フレームレート）を取得する。
    """
    return cap.get(cv2.CAP_PROP_FPS)

def save_frame(frame, output_dir, idx):
    """
    フレーム画像を指定ディレクトリに保存し、保存パスを返す。
    """
    frame_path = os.path.join(output_dir, f"frame_{idx:05d}.png")
    cv2.imwrite(frame_path, frame)
    return frame_path

def extract_frames(video_path, frame_interval_sec, output_dir):
    """
    動画ファイルから指定間隔ごとにフレーム画像を抽出し、保存パスのリストを返す。
    """
    ensure_dir(output_dir)
    cap = open_video(video_path)
    fps = get_fps(cap)
    frame_interval = int(fps * frame_interval_sec)
    frame_count = 0
    saved_paths = []
    idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="フレーム抽出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = save_frame(frame, output_dir, idx)
            saved_paths.append(frame_path)
            idx += 1
        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    return saved_paths
