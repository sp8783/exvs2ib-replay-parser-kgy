import os
import cv2
import pandas as pd
from tqdm import tqdm
from src.screen.classifier import ScreenClassifier
from src.util.io import ensure_dir


def save_screen_log(log_rows, results_dir):
    """
    画面判定結果をCSVファイルに保存する。
    統合フロー・従来フローどちらからも呼び出せる独立関数。
    """
    ensure_dir(results_dir)
    log_path = os.path.join(results_dir, "screen_log.csv")
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"画面判定結果を {log_path} に保存しました。")


class ScreenDetector:
    """
    フレーム画像リストから画面種別を検出し、マッチングカウントを算出するクラス。
    """
    
    def __init__(self, config_path):
        self.classifier = ScreenClassifier(config_path)
    
    def detect_screens(self, frame_paths, results_dir):
        """
        フレーム画像のリストから、マッチング画面・リザルト画面のみを抽出し、
        画面種別とパスのリストを返す。
        """
        screens = []
        match_count = 0
        prev_type = None
        log_rows = []

        for frame_path in tqdm(frame_paths, desc="画面判定"):
            img = cv2.imread(frame_path)
            screen_type = self.classifier.classify(img)
            log_rows.append({"frame_path": frame_path, "screen_type": screen_type})
            
            if screen_type in ("matching", "result_win", "result_lose"):
                screens.append({"type": screen_type, "path": frame_path})
                if prev_type == "matching" and screen_type in ("result_win", "result_lose"):
                    match_count += 1
                prev_type = screen_type

        # 画面判定ログをCSVに保存
        save_screen_log(log_rows, results_dir)
        return screens, match_count
