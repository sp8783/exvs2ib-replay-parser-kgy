import argparse
import os
import yaml
import cv2
import pandas as pd
import pickle
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

def extract_frames_with_cache(video_path, frame_interval, frame_dir, cache_path):
    """
    フレーム抽出結果（画像パスリスト）をキャッシュし、再実行時はキャッシュを利用して重複作業をスキップする
    キャッシュがなければ、フレームを抽出し結果をキャッシュファイルに保存する
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            frame_paths = pickle.load(f)
        print(f"フレーム抽出キャッシュを {cache_path} から読み込みました。")
        return frame_paths

    frame_paths = extract_frames(video_path, frame_interval, frame_dir)
    with open(cache_path, "wb") as f:
        pickle.dump(frame_paths, f)
    print(f"フレーム抽出結果を {cache_path} に保存しました。")
    return frame_paths

def detect_screens_with_cache(frame_paths, cache_path):
    """
    フレーム画像のリストから、マッチング画面・リザルト画面のみを抽出し、
    画面種別とパスのリストを返す。
    画面判定結果を output/results/screen_log.csv に保存する。
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            screens, match_count = pickle.load(f)
        print(f"画面判定キャッシュを {cache_path} から読み込みました。")
        return screens, match_count

    screens = []
    match_count = 0
    prev_type = None
    log_rows = []

    for frame_path in tqdm(frame_paths, desc="画面判定"):
        img = cv2.imread(frame_path)
        screen_type = classify_screen(img)
        log_rows.append({"frame_path": frame_path, "screen_type": screen_type})
        if screen_type in ("matching", "result_win", "result_lose"):
            screens.append({"type": screen_type, "path": frame_path})
            if prev_type == "matching" and screen_type in ("result_win", "result_lose"):
                match_count += 1
            prev_type = screen_type

    log_path = os.path.join("output", "results", "screen_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"画面判定結果を {log_path} に保存しました。")

    with open(cache_path, "wb") as f:
        pickle.dump((screens, match_count), f)
    print(f"画面判定結果を {cache_path} に保存しました。")
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
    OCRに使用したmatching画面のファイル名も記録する
    """
    results = []
    i = 0
    pbar = tqdm(total=match_count, desc="試合情報抽出")
    while i < len(screens) - 1:
        # マッチンググループの収集
        matching_frames = []
        while i < len(screens) and screens[i]["type"] == "matching":
            matching_frames.append(screens[i])
            i += 1
        # 次がリザルト画面か判定
        if i < len(screens) and screens[i]["type"] in ("result_win", "result_lose") and matching_frames:
            # 欠損がなくなるまで順次OCR
            final_info = None
            used_frame_name = None
            for frame in matching_frames:
                match_img = cv2.imread(frame["path"])
                match_info = ocr_on_matching_regions(match_img)
                # 欠損がなければ採用して終了
                if all(v is not None for v in match_info.values()):
                    final_info = match_info
                    used_frame_name = os.path.basename(frame["path"])
                    break
                # 欠損がある場合も、より多く埋まったものを優先
                if final_info is None or sum(v is not None for v in match_info.values()) > sum(v is not None for v in final_info.values()):
                    final_info = match_info
                    used_frame_name = os.path.basename(frame["path"])
            result_type = screens[i]["type"]
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
            match_timestamp = get_frame_timestamp(used_frame_name, frame_interval)
            row = {**final_info, **result_info, "start_time": match_timestamp, "ocr_frame_name": used_frame_name}
            results.append(row)
            i += 1
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
    フレーム抽出・画面判定の両方でキャッシュ専用フォルダ(output/cache)を利用し、再実行時は重複作業をスキップする。
    """
    parser = argparse.ArgumentParser(description="EXVS2IB 戦績トラッカー")
    parser.add_argument("--input", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--config", default="config/config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()

    config = load_config(args.config)
    frame_interval = config["frame_interval"]

    input_basename = os.path.basename(args.input)
    input_name, _ = os.path.splitext(input_basename)

    frame_dir = os.path.join("output", "frames", input_name)
    os.makedirs(frame_dir, exist_ok=True)

    cache_dir = os.path.join("output", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    frame_cache_path = os.path.join(cache_dir, f"frame_cache_{input_name}.pkl")
    screen_cache_path = os.path.join(cache_dir, f"screen_cache_{input_name}.pkl")

    frame_paths = extract_frames_with_cache(args.input, frame_interval, frame_dir, frame_cache_path)
    screens, match_count = detect_screens_with_cache(frame_paths, screen_cache_path)

    results = extract_match_results(screens, match_count, frame_interval)

    output_dir = "output/results"
    output_path = os.path.join(output_dir, f"result_{input_name}.csv")
    save_to_csv(results, output_path)

if __name__ == "__main__":
    main()
