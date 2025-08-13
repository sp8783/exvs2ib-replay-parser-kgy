import argparse
import os
import yaml
import cv2
import pandas as pd
from src.video_handler import extract_frames
from src.ocr_handler import run_ocr
from src.regions_matching import ocr_on_matching_regions
from src.screen_classifier import classify_screen

def load_config(config_path):
    """
    設定ファイル（YAML）を読み込んで辞書として返す
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def detect_screens(frame_paths):
    """
    フレーム画像のリストから、マッチング画面・リザルト画面のみを抽出し、
    画面種別とパスのリストを返す
    """
    screens = []
    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        screen_type = classify_screen(img)
        if screen_type in ("matching", "result_win", "result_lose"):
            screens.append({"type": screen_type, "path": frame_path})
    return screens

def extract_match_results(screens):
    """
    マッチング画面→リザルト画面のペアごとにOCRを実行し、
    1試合分の情報（プレイヤー名・機体名・勝敗など）を辞書としてまとめてリストで返す
    """
    results = []
    i = 0
    while i < len(screens) - 1:
        if screens[i]["type"] == "matching" and screens[i+1]["type"] in ("result_win", "result_lose"):
            match_img = cv2.imread(screens[i]["path"])
            match_info = ocr_on_matching_regions(match_img)
            result_type = screens[i+1]["type"]
            if result_type == "result_win":
                result_info = {
                    "player1_result": "WIN",
                    "player2_result": "WIN",
                    "player3_result": "LOSE",
                    "player4_result": "LOSE"
                }
            else:
                result_info = {
                    "player1_result": "LOSE",
                    "player2_result": "LOSE",
                    "player3_result": "WIN",
                    "player4_result": "WIN"
                }
            row = {**match_info, **result_info}
            results.append(row)
            i += 2
        else:
            i += 1
    return results

def save_to_csv(results, output_path):
    """
    試合ごとの情報リストをCSVファイルとして保存する
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"結果を {output_path} に保存しました。")

def main():
    """
    コマンドライン引数を受け取り、動画から戦績情報を抽出してCSVに保存するメイン処理
    """
    parser = argparse.ArgumentParser(description="EXVS2IB 戦績トラッカー")
    parser.add_argument("--input", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output", default="output/results/results.csv", help="出力CSVファイルパス")
    parser.add_argument("--config", default="config/config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()

    config = load_config(args.config)
    frame_paths = extract_frames(args.input, config["frame_interval"], "output/frames")
    screens = detect_screens(frame_paths)
    results = extract_match_results(screens)
    save_to_csv(results, args.output)

if __name__ == "__main__":
    main()
