import argparse
import os
import yaml
import cv2
import pandas as pd
from tqdm import tqdm
from src.video_handler import extract_frames
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
    match_count = 0
    prev_type = None
    for frame_path in tqdm(frame_paths, desc="画面判定"):
        img = cv2.imread(frame_path)
        screen_type = classify_screen(img)
        if screen_type in ("matching", "result_win", "result_lose"):
            screens.append({"type": screen_type, "path": frame_path})
            if prev_type == "matching" and screen_type in ("result_win", "result_lose"):
                match_count += 1
            prev_type = screen_type
    return screens, match_count

def get_frame_timestamp(frame_path, frame_interval):
    """
    フレーム画像ファイル名から連番を取得し、frame_interval（秒）からタイムスタンプをhh:mm:ss形式で返す。
    """
    basename = os.path.basename(frame_path)
    frame_num = int(basename.split("_")[1].split(".")[0])
    total_seconds = int(frame_num * frame_interval)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def extract_match_results(screens, match_count, frame_interval):
    """
    マッチング画面→リザルト画面のペアごとにOCRを実行し、
    1試合分の情報（プレイヤー名・機体名・勝敗・開始タイムスタンプなど）を辞書としてまとめてリストで返す
    """
    results = []
    i = 0
    pbar = tqdm(total=match_count, desc="試合情報抽出")
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
            match_timestamp = get_frame_timestamp(screens[i]["path"], frame_interval)
            row = {**match_info, **result_info, "start_time": match_timestamp}
            results.append(row)
            i += 2
            pbar.update(1)
        else:
            i += 1
    pbar.close()
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
    parser.add_argument("--frames", default="output/frames", help="フレーム画像の保存先ディレクトリ")
    args = parser.parse_args()

    config = load_config(args.config)
    frame_interval = config["frame_interval"]

    frame_dir = args.frames
    frame_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(".png")
    ])
    if frame_paths:
        print(f"{len(frame_paths)}枚のフレーム画像が既に存在するため、抽出をスキップします。")
    else:
        frame_paths = extract_frames(args.input, frame_interval, frame_dir)

    screens, match_count = detect_screens(frame_paths)
    results = extract_match_results(screens, match_count, frame_interval)
    save_to_csv(results, args.output)

if __name__ == "__main__":
    main()
